
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import sys
2: import types
3: import inspect
4: import os
5: 
6: from ...python_lib_copy.python_types_copy.type_inference_copy import type_inference_proxy_copy
7: from ...errors_copy.type_error_copy import TypeError
8: import python_library_modules_copy
9: from ... import stypy_main_copy, stypy_parameters_copy
10: 
11: '''
12: Helper functions to deal with imports on type inference generated code. These were moved here for improving the
13: readability of the code. These functions are called by the equivalent functions in python_interface.py
14: '''
15: 
16: # Types for special names passed to import
17: __known_types = {
18:     'False': type_inference_proxy_copy.TypeInferenceProxy.instance(bool, value=False),
19:     'True': type_inference_proxy_copy.TypeInferenceProxy.instance(bool, value=True),
20:     'None': type_inference_proxy_copy.TypeInferenceProxy.instance(types.NoneType, value=None),
21: }
22: 
23: 
24: ############################################ IMPORT PYTHON LIBRARY ELEMENTS ##########################################
25: 
26: def __load_python_module_dynamically(module_name, put_in_cache=True):
27:     '''
28:     Loads a Python library module dynamically if it has not been previously loaded
29:     :param module_name:
30:     :return: Proxy holding the module
31:     '''
32:     if module_name in sys.modules:
33:         module_obj = sys.modules[module_name]
34:     else:
35:         exec ("import {0}".format(module_name))
36:         module_obj = eval(module_name)
37: 
38:     module_obj = type_inference_proxy_copy.TypeInferenceProxy(module_obj).clone()
39:     if put_in_cache:
40:         __put_module_in_sys_cache(module_name, module_obj)
41:     return module_obj
42: 
43: 
44: def __preload_sys_module_cache():
45:     '''
46:     The "sys" Python module holds a cache of stypy-generated module files in order to save time. A Python library
47:     module was chosen to hold these data so it can be available through executions and module imports from external
48:     files. This function preloads
49:     :return:
50:     '''
51:     # Preload sys module
52:     sys.stypy_module_cache = {
53:         'sys': __load_python_module_dynamically('sys', False)}  # By default, add original sys module clone
54: 
55:     # Preload builtins module
56:     sys.stypy_module_cache['__builtin__'] = __load_python_module_dynamically('__builtin__', False)
57:     sys.stypy_module_cache['ctypes'] = __load_python_module_dynamically('ctypes', False)
58: 
59: 
60: def __exist_module_in_sys_cache(module_name):
61:     '''
62:     Determines if a module called "module_name" (or whose .py file is equal to the argument) has been previously loaded
63:     :param module_name: Module name (Python library modules) or file path (other modules) to check
64:     :return: bool
65:     '''
66:     try:
67:         if hasattr(sys, 'stypy_module_cache'):
68:             return module_name in sys.stypy_module_cache
69:         else:
70:             __preload_sys_module_cache()
71:             return False
72:     except:
73:         return False
74: 
75: 
76: def get_module_from_sys_cache(module_name):
77:     '''
78:     Gets a previously loaded module from the sys module cache
79:     :param module_name: Module name
80:     :return: A Type object or None if there is no such module
81:     '''
82:     try:
83:         if hasattr(sys, 'stypy_module_cache'):
84:             return sys.stypy_module_cache[module_name]
85:         else:
86:             __preload_sys_module_cache()
87:             return sys.stypy_module_cache[module_name]
88:     except:
89:         return None
90: 
91: 
92: def __put_module_in_sys_cache(module_name, module_obj):
93:     '''
94:     Puts a module in the sys stypy module cache
95:     :param module_name: Name of the module
96:     :param module_obj: Object representing the module
97:     :return: None
98:     '''
99:     #try:
100:         #if hasattr(sys, 'stypy_module_cache'):
101:     sys.stypy_module_cache[module_name] = module_obj
102:         # else:
103:         #     __preload_sys_module_cache()
104:         #     sys.stypy_module_cache[module_name] = module_obj
105:     # except:
106:     #     pass
107:     # finally:
108:     #     return None
109: 
110: 
111: def __import_python_library_module(localization, module_name="__builtin__"):
112:     '''
113:     Import a full Python library module (models the "import <module>" statement for Python library modules
114:     :param localization: Caller information
115:     :param module_name: Module to import
116:     :return: A TypeInferenceProxy for the specified module or a TypeError if the requested module do not exist
117:     '''
118:     try:
119:         module_obj = get_module_from_sys_cache(module_name)
120:         if module_obj is None:
121:             module_obj = __load_python_module_dynamically(module_name)
122:             module = module_obj.get_python_entity()
123: 
124:             module_members = module.__dict__
125:             for member in module_members:
126:                 if inspect.ismodule(module_members[member]):
127:                     member_module_name = module_members[member].__name__
128:                     # Is not our own member
129:                     if member_module_name is not module_name:
130:                         if not __exist_module_in_sys_cache(member_module_name):
131:                             module_ti = __load_python_module_dynamically(member_module_name)
132:                             module_obj.set_type_of_member(localization, member, module_ti)
133:         return module_obj
134:     except Exception as exc:
135:         return TypeError(localization, "Could not load Python library module '{0}': {1}".format(module_name, str(exc)))
136: 
137: 
138: def __get_non_python_library_module_file(module_name, environment=sys.path):
139:     '''
140:     Obtains the source file in which a module source code resides.
141:     :module_name Name of the module whose source file we intend to find
142:     :environment (Optional) List of paths to use to search the module (defaults to sys.path)
143:     :return: str or None
144:     '''
145:     found = None
146: 
147:     # Use the longer paths first
148:     paths = reversed(sorted(environment))
149:     for path in paths:
150:         base_path = path.replace("\\", "/")
151:         if stypy_parameters_copy.type_inference_file_directory_name in path:
152:             base_path = base_path.replace("/" + stypy_parameters_copy.type_inference_file_directory_name, "")
153: 
154:         temp = base_path + "/" + module_name.replace('.', '/') + ".py"
155:         if os.path.isfile(temp):
156:             found = temp
157:         # Module (__init__) names have precedence over file names
158:         temp = base_path + "/" + module_name.replace('.', '/') + "/__init__.py"
159:         if os.path.isfile(temp):
160:             found = temp
161:             break
162:     if found is None:
163:         pass
164: 
165:     return found
166: 
167: 
168: def __get_module_file(module_name):
169:     module_file = None
170:     loaded_module = None
171:     module_type_store = None
172:     if module_name in sys.modules:
173:         loaded_module = sys.modules[module_name]
174:         if hasattr(loaded_module, '__file__'):
175:             module_file = loaded_module.__file__
176:     else:
177:         loaded_module = __import__(module_name)
178:         if hasattr(loaded_module, '__file__'):
179:             module_file = loaded_module.__file__
180:     if module_file is None:
181:         raise Exception(module_name)
182:     return module_file
183: 
184: 
185: def __import_external_non_python_library_module(localization, module_name, environment=sys.path):
186:     '''
187:     Returns the TypeStore object that represent a non Python library module object
188:     :localization Caller information
189:     :module_name Name of the module to load
190:     :environment (Optional) List of paths to use to search the module (defaults to sys.path)
191:     :return: A TypeStore object or a TypeError if the module cannot be loaded
192:     '''
193:     try:
194:         module_file = __get_module_file(module_name)
195:         # print "Importing " + module_name + " (" + module_file + ")"
196:         module_obj = get_module_from_sys_cache(module_file)
197:         if module_obj is None:
198:             # print "Cache miss: " + module_name
199:             # sys.path.append(os.path.dirname(module_file))
200:             source_path = __get_non_python_library_module_file(module_name, environment)
201:             module_obj = stypy_main_copy.Stypy(source_path, generate_type_annotated_program=False)
202:             # This way the same module will not be analyzed again
203:             __put_module_in_sys_cache(module_file, module_obj)
204:             module_obj.analyze()
205:             # sys.path.remove(os.path.dirname(module_file))
206: 
207:             module_type_store = module_obj.get_analyzed_program_type_store()
208:             if module_type_store is None:
209:                 return TypeError(localization, "Could not import external module '{0}'".format(module_name))
210:         # else:
211:         #     print "Cache hit"
212: 
213:         return module_obj
214: 
215:     except Exception as exc:
216:         # import traceback
217:         # traceback.print_exc()
218:         # sys.exit(-1)
219:         return TypeError(localization, "Could not import external module '{0}': {1}".format(module_name, str(exc)))
220: 
221: 
222: ######################################### MODULE EXTERNAL INTERFACE #########################################
223: 
224: 
225: def import_python_module(localization, imported_module_name, environment=sys.path):
226:     '''
227:     This function imports all the declared public members of a user-defined or Python library module into the specified
228:     type store
229:     It modules the from <module> import <element1>, <element2>, ... or * sentences and also the import <module> sentence
230: 
231:     GENERAL ALGORITHM PSEUDO-CODE:
232: 
233:     This will be divided in two functions. One for importing a module object. The other to process the elements
234:     of the module that will be imported once returned. We have three options:
235:     - elements = []: (import module) -> The module object will be added to the destination type store
236:     - elements = ['*']: All the public members of the module will be added to the destionation type store. NOT the module
237:     itself
238:     - elements = ['member1', 'member2',...]: A particular case of the previous one. We have to check the behavior of Python
239:     if someone explicitly try to import __xxx or _xxx members.
240: 
241:     - Python library modules are represented by a TypeInferenceProxy object (subtype of Type)
242:     - "User" modules (those whose source code has been processed by stypy) are represented by a TypeStore (subtype of Type)
243: 
244:     Import module function:
245:         - Input is a string: Get module object from sys.modules
246:         - Once we have the module object:
247:         - If it is a Python library module:
248:             Check if there is a cache for stypy modules in sys
249:             If not, create it
250:             Else check if the module it is already cached (using module name)
251:                 If it is, return the cached module
252:             TypeInferenceProxy a module clone (to not to modify the original one)
253:             for each member of the module:
254:                 if the member is another module that is not already cached:
255:                     Recursively apply the function, obtaining a module
256:                 Assign the resulting module to the member value (in the module clone)
257: 
258:         - If it is not a Python library module:
259:             Check if there is a cache for stypy modules in sys
260:             If not, create it
261:             Else check if the module it is already cached (using the module path)
262:                 If it is, return the cached module
263:             Create an Stypy object using the module source path
264:             Analyze the module with stypy
265:                 This will trigger secondary imports when executing the type inference program, as they can contain other
266:                 imports. So there is no need to recursively call this function or to analyze the module members, as this
267:                 will be done automatically by calling secondary stypy instances from this one
268:             return the Stypy object
269: 
270:     Other considerations:
271: 
272:     Type inference programs will use this line to import external modules:
273: 
274:     import_elements_from_external_module(<localization object>,
275:          <module name as it appears in the original code>, type_store)
276: 
277:     This function will:
278:         - Obtain the imported module following the previous algorithm
279:         - If a TypeInference proxy is obtained, proceed to assign members
280:         - If an stypy object is obtained, obtain its type store and proceed to assign members.
281: 
282: 
283: 
284:     :param localization: Caller information
285:     :param main_module_path: Path of the module to import, i. e. path of the .py file of the module
286:     :param imported_module_name: Name of the module
287:     :param dest_type_store: Type store to add the module elements
288:     :param elements: A variable list of arguments with the elements to import. The value '*' means all elements. No
289:     value models the "import <module>" sentence
290:     :return: None or a TypeError if the requested type do not exist
291:     '''
292:     sys.setrecursionlimit(8000)
293: 
294:     if not python_library_modules_copy.is_python_library_module(imported_module_name):
295:         stypy_obj = __import_external_non_python_library_module(localization, imported_module_name, environment)
296:         if isinstance(stypy_obj, TypeError):
297:             return stypy_obj
298:         return stypy_obj.get_analyzed_program_type_store()
299:     else:
300:         return __import_python_library_module(localization, imported_module_name)
301: 
302: 
303: def __get_public_names_and_types_of_module(module_obj):
304:     '''
305:     Get the public (importable) elements of a module
306:     :param module_obj: Module object (either a TypeInferenceProxy or a TypeStore)
307:     :return: list of str
308:     '''
309:     if isinstance(module_obj, type_inference_proxy_copy.TypeInferenceProxy):
310:         return filter(lambda name: not name.startswith("__"), dir(module_obj.get_python_entity()))
311:     else:
312:         return module_obj.get_public_names_and_types()
313: 
314: 
315: def __import_module_element(localization, imported_module_name, module_obj, element, dest_type_store, environment):
316:     # Import each specified member
317:     member_type = module_obj.get_type_of_member(localization, element)
318:     if isinstance(member_type, TypeError):
319:         module_file = __get_non_python_library_module_file(element)
320:         if module_file is None:
321:             return member_type  # TypeError
322: 
323:         module_dir = os.path.dirname(module_file)
324: 
325:         # possible_module_member_file = module_dir + "/" + element + ".py"
326:         # Element imported is a module not previously loaded
327:         if os.path.isfile(module_file):
328:             restricted_environment = [module_dir] + environment
329:             import_elements_from_external_module(localization, element, dest_type_store,
330:                                                  restricted_environment,
331:                                                  *[])
332:             TypeError.remove_error_msg(member_type)
333:         else:
334:             dest_type_store.set_type_of(localization, element, member_type)
335:     else:
336:         # The imported elements may be other not loaded modules. We check this and load them
337:         dest_type_store.set_type_of(localization, element, member_type)
338: 
339: 
340: def import_elements_from_external_module(localization, imported_module_name, dest_type_store, environment,
341:                                          *elements):
342:     '''
343:     Imports the listed elements from the provided module name in the dest_type_store TypeStore, using the provided
344:     environment as a module search path
345: 
346:     :param localization: Caller information
347:     :param imported_module_name: Name of the module to import
348:     :param dest_type_store: Type store to store the imported elements in
349:     :param environment: List of paths for module seach
350:     :param elements: Elements of the module to import ([] for import the whole module, ['*'] for 'from module import *'
351:     statements and a list of names for importing concrete module members.
352:     :return: None
353:     '''
354:     sys.setrecursionlimit(8000)  # Necessary for large files
355: 
356:     if not python_library_modules_copy.is_python_library_module(imported_module_name):
357:         # Path of the module that is going to import elements
358:         destination_module_path = os.path.dirname(dest_type_store.program_name)
359:         destination_path_added = False
360:         if not destination_module_path in environment:
361:             destination_path_added = True
362:             environment.append(destination_module_path)
363: 
364:     module_obj = import_python_module(localization, imported_module_name, environment)
365: 
366:     if not python_library_modules_copy.is_python_library_module(imported_module_name):
367:         # File of the imported module
368:         imported_module_file = __get_module_file(imported_module_name)
369:         imported_module_path = os.path.dirname(imported_module_file)
370:         imported_path_added = False
371:         if not imported_module_path in environment:
372:             imported_path_added = True
373:             environment.append(imported_module_path)
374: 
375:     if len(elements) == 0:
376:         # Covers 'import <module>'
377:         dest_type_store.set_type_of(localization, imported_module_name, module_obj)
378:         return None
379: 
380:     # Covers 'from <module> import <elements>', with <elements> being '*' or a list of members
381:     for element in elements:
382:         # Import all elements from module
383:         if element == '*':
384:             public_elements = __get_public_names_and_types_of_module(module_obj)
385:             for public_element in public_elements:
386:                 __import_module_element(localization, imported_module_name, module_obj, public_element, dest_type_store,
387:                                         environment)
388:             break
389: 
390:         # Import each specified member
391:         __import_module_element(localization, imported_module_name, module_obj, element, dest_type_store, environment)
392: 
393:     if not python_library_modules_copy.is_python_library_module(imported_module_name):
394:         if destination_path_added:
395:             environment.remove(destination_module_path)
396:         if imported_path_added:
397:             environment.remove(imported_module_path)
398: 
399: 
400: ######################################### IMPORT FROM #########################################
401: 
402: 
403: def __import_from(localization, member_name, module_name="__builtin__"):
404:     '''
405:     Imports a single member from a module. If no module is specified, the builtin module is used instead. Models the
406:     "from <module> import <member>" sentence, being a sort version of the import_elements_from_external_module function
407:     but only for Python library modules. This is a helper function of the following one.
408:     :param localization: Caller information
409:     :param member_name: Member to import
410:     :param module_name: Python library module that contains the member or nothing to use the __builtins__ module
411:     :return: A TypeInferenceProxy for the specified member or a TypeError if the requested element do not exist
412:     '''
413:     module = import_python_module(localization, module_name)
414:     if isinstance(module, TypeError):
415:         return module, None
416: 
417:     try:
418:         return module, module.get_type_of_member(localization, member_name)
419:     except Exception as exc:
420:         return module, TypeError(localization,
421:                                  "Could not load member '{0}' from module '{1}': {2}".format(member_name, module_name,
422:                                                                                              str(exc)))
423: 
424: 
425: def import_from(localization, member_name, module_name="__builtin__"):
426:     '''
427:     Imports a single member from a module. If no module is specified, the builtin module is used instead. Models the
428:     "from <module> import <member>" sentence, being a sort version of the import_elements_from_external_module function
429:     but only for Python library modules
430:     :param localization: Caller information
431:     :param member_name: Member to import
432:     :param module_name: Python library module that contains the member or nothing to use the __builtins__ module
433:     :return: A TypeInferenceProxy for the specified member or a TypeError if the requested element do not exist
434:     '''
435:     # Known types are always returned first.
436:     if member_name in __known_types:
437:         return __known_types[member_name]
438: 
439:     module, member = __import_from(localization, member_name, module_name)
440:     if not isinstance(member, TypeError):
441:         m = type_inference_proxy_copy.TypeInferenceProxy.instance(module.python_entity)
442:         return type_inference_proxy_copy.TypeInferenceProxy.instance(member, parent=m)
443: 
444:     return member
445: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import sys' statement (line 1)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import types' statement (line 2)
import types

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'types', types, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import inspect' statement (line 3)
import inspect

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'inspect', inspect, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import type_inference_proxy_copy' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/module_imports_copy/')
import_7914 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_7914) is not StypyTypeError):

    if (import_7914 != 'pyd_module'):
        __import__(import_7914)
        sys_modules_7915 = sys.modules[import_7914]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', sys_modules_7915.module_type_store, module_type_store, ['type_inference_proxy_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_7915, sys_modules_7915.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import type_inference_proxy_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['type_inference_proxy_copy'], [type_inference_proxy_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', import_7914)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/module_imports_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/module_imports_copy/')
import_7916 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy')

if (type(import_7916) is not StypyTypeError):

    if (import_7916 != 'pyd_module'):
        __import__(import_7916)
        sys_modules_7917 = sys.modules[import_7916]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', sys_modules_7917.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_7917, sys_modules_7917.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', import_7916)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/module_imports_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import python_library_modules_copy' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/module_imports_copy/')
import_7918 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'python_library_modules_copy')

if (type(import_7918) is not StypyTypeError):

    if (import_7918 != 'pyd_module'):
        __import__(import_7918)
        sys_modules_7919 = sys.modules[import_7918]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'python_library_modules_copy', sys_modules_7919.module_type_store, module_type_store)
    else:
        import python_library_modules_copy

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'python_library_modules_copy', python_library_modules_copy, module_type_store)

else:
    # Assigning a type to the variable 'python_library_modules_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'python_library_modules_copy', import_7918)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/module_imports_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy import stypy_main_copy, stypy_parameters_copy' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/module_imports_copy/')
import_7920 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy')

if (type(import_7920) is not StypyTypeError):

    if (import_7920 != 'pyd_module'):
        __import__(import_7920)
        sys_modules_7921 = sys.modules[import_7920]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', sys_modules_7921.module_type_store, module_type_store, ['stypy_main_copy', 'stypy_parameters_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_7921, sys_modules_7921.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy import stypy_main_copy, stypy_parameters_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', None, module_type_store, ['stypy_main_copy', 'stypy_parameters_copy'], [stypy_main_copy, stypy_parameters_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', import_7920)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/module_imports_copy/')

str_7922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, (-1)), 'str', '\nHelper functions to deal with imports on type inference generated code. These were moved here for improving the\nreadability of the code. These functions are called by the equivalent functions in python_interface.py\n')

# Assigning a Dict to a Name (line 17):

# Assigning a Dict to a Name (line 17):

# Obtaining an instance of the builtin type 'dict' (line 17)
dict_7923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 17)
# Adding element type (key, value) (line 17)
str_7924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 4), 'str', 'False')

# Call to instance(...): (line 18)
# Processing the call arguments (line 18)
# Getting the type of 'bool' (line 18)
bool_7928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 67), 'bool', False)
# Processing the call keyword arguments (line 18)
# Getting the type of 'False' (line 18)
False_7929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 79), 'False', False)
keyword_7930 = False_7929
kwargs_7931 = {'value': keyword_7930}
# Getting the type of 'type_inference_proxy_copy' (line 18)
type_inference_proxy_copy_7925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 13), 'type_inference_proxy_copy', False)
# Obtaining the member 'TypeInferenceProxy' of a type (line 18)
TypeInferenceProxy_7926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 13), type_inference_proxy_copy_7925, 'TypeInferenceProxy')
# Obtaining the member 'instance' of a type (line 18)
instance_7927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 13), TypeInferenceProxy_7926, 'instance')
# Calling instance(args, kwargs) (line 18)
instance_call_result_7932 = invoke(stypy.reporting.localization.Localization(__file__, 18, 13), instance_7927, *[bool_7928], **kwargs_7931)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 16), dict_7923, (str_7924, instance_call_result_7932))
# Adding element type (key, value) (line 17)
str_7933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 4), 'str', 'True')

# Call to instance(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of 'bool' (line 19)
bool_7937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 66), 'bool', False)
# Processing the call keyword arguments (line 19)
# Getting the type of 'True' (line 19)
True_7938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 78), 'True', False)
keyword_7939 = True_7938
kwargs_7940 = {'value': keyword_7939}
# Getting the type of 'type_inference_proxy_copy' (line 19)
type_inference_proxy_copy_7934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'type_inference_proxy_copy', False)
# Obtaining the member 'TypeInferenceProxy' of a type (line 19)
TypeInferenceProxy_7935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 12), type_inference_proxy_copy_7934, 'TypeInferenceProxy')
# Obtaining the member 'instance' of a type (line 19)
instance_7936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 12), TypeInferenceProxy_7935, 'instance')
# Calling instance(args, kwargs) (line 19)
instance_call_result_7941 = invoke(stypy.reporting.localization.Localization(__file__, 19, 12), instance_7936, *[bool_7937], **kwargs_7940)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 16), dict_7923, (str_7933, instance_call_result_7941))
# Adding element type (key, value) (line 17)
str_7942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 4), 'str', 'None')

# Call to instance(...): (line 20)
# Processing the call arguments (line 20)
# Getting the type of 'types' (line 20)
types_7946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 66), 'types', False)
# Obtaining the member 'NoneType' of a type (line 20)
NoneType_7947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 66), types_7946, 'NoneType')
# Processing the call keyword arguments (line 20)
# Getting the type of 'None' (line 20)
None_7948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 88), 'None', False)
keyword_7949 = None_7948
kwargs_7950 = {'value': keyword_7949}
# Getting the type of 'type_inference_proxy_copy' (line 20)
type_inference_proxy_copy_7943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'type_inference_proxy_copy', False)
# Obtaining the member 'TypeInferenceProxy' of a type (line 20)
TypeInferenceProxy_7944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 12), type_inference_proxy_copy_7943, 'TypeInferenceProxy')
# Obtaining the member 'instance' of a type (line 20)
instance_7945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 12), TypeInferenceProxy_7944, 'instance')
# Calling instance(args, kwargs) (line 20)
instance_call_result_7951 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), instance_7945, *[NoneType_7947], **kwargs_7950)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 16), dict_7923, (str_7942, instance_call_result_7951))

# Assigning a type to the variable '__known_types' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), '__known_types', dict_7923)

@norecursion
def __load_python_module_dynamically(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 26)
    True_7952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 63), 'True')
    defaults = [True_7952]
    # Create a new context for function '__load_python_module_dynamically'
    module_type_store = module_type_store.open_function_context('__load_python_module_dynamically', 26, 0, False)
    
    # Passed parameters checking function
    __load_python_module_dynamically.stypy_localization = localization
    __load_python_module_dynamically.stypy_type_of_self = None
    __load_python_module_dynamically.stypy_type_store = module_type_store
    __load_python_module_dynamically.stypy_function_name = '__load_python_module_dynamically'
    __load_python_module_dynamically.stypy_param_names_list = ['module_name', 'put_in_cache']
    __load_python_module_dynamically.stypy_varargs_param_name = None
    __load_python_module_dynamically.stypy_kwargs_param_name = None
    __load_python_module_dynamically.stypy_call_defaults = defaults
    __load_python_module_dynamically.stypy_call_varargs = varargs
    __load_python_module_dynamically.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__load_python_module_dynamically', ['module_name', 'put_in_cache'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__load_python_module_dynamically', localization, ['module_name', 'put_in_cache'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__load_python_module_dynamically(...)' code ##################

    str_7953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, (-1)), 'str', '\n    Loads a Python library module dynamically if it has not been previously loaded\n    :param module_name:\n    :return: Proxy holding the module\n    ')
    
    # Getting the type of 'module_name' (line 32)
    module_name_7954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 7), 'module_name')
    # Getting the type of 'sys' (line 32)
    sys_7955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 22), 'sys')
    # Obtaining the member 'modules' of a type (line 32)
    modules_7956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 22), sys_7955, 'modules')
    # Applying the binary operator 'in' (line 32)
    result_contains_7957 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 7), 'in', module_name_7954, modules_7956)
    
    # Testing if the type of an if condition is none (line 32)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 32, 4), result_contains_7957):
        # Dynamic code evaluation using an exec statement
        
        # Call to format(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'module_name' (line 35)
        module_name_7966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 34), 'module_name', False)
        # Processing the call keyword arguments (line 35)
        kwargs_7967 = {}
        str_7964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 14), 'str', 'import {0}')
        # Obtaining the member 'format' of a type (line 35)
        format_7965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 14), str_7964, 'format')
        # Calling format(args, kwargs) (line 35)
        format_call_result_7968 = invoke(stypy.reporting.localization.Localization(__file__, 35, 14), format_7965, *[module_name_7966], **kwargs_7967)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 35, 8), format_call_result_7968, 'exec parameter', 'StringType', 'FileType', 'CodeType')
        enable_usage_of_dynamic_types_warning(stypy.reporting.localization.Localization(__file__, 35, 8))
        
        # Assigning a Call to a Name (line 36):
        
        # Assigning a Call to a Name (line 36):
        
        # Call to eval(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'module_name' (line 36)
        module_name_7970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 26), 'module_name', False)
        # Processing the call keyword arguments (line 36)
        kwargs_7971 = {}
        # Getting the type of 'eval' (line 36)
        eval_7969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 21), 'eval', False)
        # Calling eval(args, kwargs) (line 36)
        eval_call_result_7972 = invoke(stypy.reporting.localization.Localization(__file__, 36, 21), eval_7969, *[module_name_7970], **kwargs_7971)
        
        # Assigning a type to the variable 'module_obj' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'module_obj', eval_call_result_7972)
    else:
        
        # Testing the type of an if condition (line 32)
        if_condition_7958 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 4), result_contains_7957)
        # Assigning a type to the variable 'if_condition_7958' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'if_condition_7958', if_condition_7958)
        # SSA begins for if statement (line 32)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 33):
        
        # Assigning a Subscript to a Name (line 33):
        
        # Obtaining the type of the subscript
        # Getting the type of 'module_name' (line 33)
        module_name_7959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 33), 'module_name')
        # Getting the type of 'sys' (line 33)
        sys_7960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 21), 'sys')
        # Obtaining the member 'modules' of a type (line 33)
        modules_7961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 21), sys_7960, 'modules')
        # Obtaining the member '__getitem__' of a type (line 33)
        getitem___7962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 21), modules_7961, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 33)
        subscript_call_result_7963 = invoke(stypy.reporting.localization.Localization(__file__, 33, 21), getitem___7962, module_name_7959)
        
        # Assigning a type to the variable 'module_obj' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'module_obj', subscript_call_result_7963)
        # SSA branch for the else part of an if statement (line 32)
        module_type_store.open_ssa_branch('else')
        # Dynamic code evaluation using an exec statement
        
        # Call to format(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'module_name' (line 35)
        module_name_7966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 34), 'module_name', False)
        # Processing the call keyword arguments (line 35)
        kwargs_7967 = {}
        str_7964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 14), 'str', 'import {0}')
        # Obtaining the member 'format' of a type (line 35)
        format_7965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 14), str_7964, 'format')
        # Calling format(args, kwargs) (line 35)
        format_call_result_7968 = invoke(stypy.reporting.localization.Localization(__file__, 35, 14), format_7965, *[module_name_7966], **kwargs_7967)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 35, 8), format_call_result_7968, 'exec parameter', 'StringType', 'FileType', 'CodeType')
        enable_usage_of_dynamic_types_warning(stypy.reporting.localization.Localization(__file__, 35, 8))
        
        # Assigning a Call to a Name (line 36):
        
        # Assigning a Call to a Name (line 36):
        
        # Call to eval(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'module_name' (line 36)
        module_name_7970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 26), 'module_name', False)
        # Processing the call keyword arguments (line 36)
        kwargs_7971 = {}
        # Getting the type of 'eval' (line 36)
        eval_7969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 21), 'eval', False)
        # Calling eval(args, kwargs) (line 36)
        eval_call_result_7972 = invoke(stypy.reporting.localization.Localization(__file__, 36, 21), eval_7969, *[module_name_7970], **kwargs_7971)
        
        # Assigning a type to the variable 'module_obj' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'module_obj', eval_call_result_7972)
        # SSA join for if statement (line 32)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 38):
    
    # Assigning a Call to a Name (line 38):
    
    # Call to clone(...): (line 38)
    # Processing the call keyword arguments (line 38)
    kwargs_7979 = {}
    
    # Call to TypeInferenceProxy(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'module_obj' (line 38)
    module_obj_7975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 62), 'module_obj', False)
    # Processing the call keyword arguments (line 38)
    kwargs_7976 = {}
    # Getting the type of 'type_inference_proxy_copy' (line 38)
    type_inference_proxy_copy_7973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 17), 'type_inference_proxy_copy', False)
    # Obtaining the member 'TypeInferenceProxy' of a type (line 38)
    TypeInferenceProxy_7974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 17), type_inference_proxy_copy_7973, 'TypeInferenceProxy')
    # Calling TypeInferenceProxy(args, kwargs) (line 38)
    TypeInferenceProxy_call_result_7977 = invoke(stypy.reporting.localization.Localization(__file__, 38, 17), TypeInferenceProxy_7974, *[module_obj_7975], **kwargs_7976)
    
    # Obtaining the member 'clone' of a type (line 38)
    clone_7978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 17), TypeInferenceProxy_call_result_7977, 'clone')
    # Calling clone(args, kwargs) (line 38)
    clone_call_result_7980 = invoke(stypy.reporting.localization.Localization(__file__, 38, 17), clone_7978, *[], **kwargs_7979)
    
    # Assigning a type to the variable 'module_obj' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'module_obj', clone_call_result_7980)
    # Getting the type of 'put_in_cache' (line 39)
    put_in_cache_7981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 7), 'put_in_cache')
    # Testing if the type of an if condition is none (line 39)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 39, 4), put_in_cache_7981):
        pass
    else:
        
        # Testing the type of an if condition (line 39)
        if_condition_7982 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 4), put_in_cache_7981)
        # Assigning a type to the variable 'if_condition_7982' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'if_condition_7982', if_condition_7982)
        # SSA begins for if statement (line 39)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __put_module_in_sys_cache(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'module_name' (line 40)
        module_name_7984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 34), 'module_name', False)
        # Getting the type of 'module_obj' (line 40)
        module_obj_7985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 47), 'module_obj', False)
        # Processing the call keyword arguments (line 40)
        kwargs_7986 = {}
        # Getting the type of '__put_module_in_sys_cache' (line 40)
        put_module_in_sys_cache_7983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), '__put_module_in_sys_cache', False)
        # Calling __put_module_in_sys_cache(args, kwargs) (line 40)
        put_module_in_sys_cache_call_result_7987 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), put_module_in_sys_cache_7983, *[module_name_7984, module_obj_7985], **kwargs_7986)
        
        # SSA join for if statement (line 39)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'module_obj' (line 41)
    module_obj_7988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'module_obj')
    # Assigning a type to the variable 'stypy_return_type' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type', module_obj_7988)
    
    # ################# End of '__load_python_module_dynamically(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__load_python_module_dynamically' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_7989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7989)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__load_python_module_dynamically'
    return stypy_return_type_7989

# Assigning a type to the variable '__load_python_module_dynamically' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '__load_python_module_dynamically', __load_python_module_dynamically)

@norecursion
def __preload_sys_module_cache(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__preload_sys_module_cache'
    module_type_store = module_type_store.open_function_context('__preload_sys_module_cache', 44, 0, False)
    
    # Passed parameters checking function
    __preload_sys_module_cache.stypy_localization = localization
    __preload_sys_module_cache.stypy_type_of_self = None
    __preload_sys_module_cache.stypy_type_store = module_type_store
    __preload_sys_module_cache.stypy_function_name = '__preload_sys_module_cache'
    __preload_sys_module_cache.stypy_param_names_list = []
    __preload_sys_module_cache.stypy_varargs_param_name = None
    __preload_sys_module_cache.stypy_kwargs_param_name = None
    __preload_sys_module_cache.stypy_call_defaults = defaults
    __preload_sys_module_cache.stypy_call_varargs = varargs
    __preload_sys_module_cache.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__preload_sys_module_cache', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__preload_sys_module_cache', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__preload_sys_module_cache(...)' code ##################

    str_7990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, (-1)), 'str', '\n    The "sys" Python module holds a cache of stypy-generated module files in order to save time. A Python library\n    module was chosen to hold these data so it can be available through executions and module imports from external\n    files. This function preloads\n    :return:\n    ')
    
    # Assigning a Dict to a Attribute (line 52):
    
    # Assigning a Dict to a Attribute (line 52):
    
    # Obtaining an instance of the builtin type 'dict' (line 52)
    dict_7991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 29), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 52)
    # Adding element type (key, value) (line 52)
    str_7992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 8), 'str', 'sys')
    
    # Call to __load_python_module_dynamically(...): (line 53)
    # Processing the call arguments (line 53)
    str_7994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 48), 'str', 'sys')
    # Getting the type of 'False' (line 53)
    False_7995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 55), 'False', False)
    # Processing the call keyword arguments (line 53)
    kwargs_7996 = {}
    # Getting the type of '__load_python_module_dynamically' (line 53)
    load_python_module_dynamically_7993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 15), '__load_python_module_dynamically', False)
    # Calling __load_python_module_dynamically(args, kwargs) (line 53)
    load_python_module_dynamically_call_result_7997 = invoke(stypy.reporting.localization.Localization(__file__, 53, 15), load_python_module_dynamically_7993, *[str_7994, False_7995], **kwargs_7996)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 29), dict_7991, (str_7992, load_python_module_dynamically_call_result_7997))
    
    # Getting the type of 'sys' (line 52)
    sys_7998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'sys')
    # Setting the type of the member 'stypy_module_cache' of a type (line 52)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 4), sys_7998, 'stypy_module_cache', dict_7991)
    
    # Assigning a Call to a Subscript (line 56):
    
    # Assigning a Call to a Subscript (line 56):
    
    # Call to __load_python_module_dynamically(...): (line 56)
    # Processing the call arguments (line 56)
    str_8000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 77), 'str', '__builtin__')
    # Getting the type of 'False' (line 56)
    False_8001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 92), 'False', False)
    # Processing the call keyword arguments (line 56)
    kwargs_8002 = {}
    # Getting the type of '__load_python_module_dynamically' (line 56)
    load_python_module_dynamically_7999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 44), '__load_python_module_dynamically', False)
    # Calling __load_python_module_dynamically(args, kwargs) (line 56)
    load_python_module_dynamically_call_result_8003 = invoke(stypy.reporting.localization.Localization(__file__, 56, 44), load_python_module_dynamically_7999, *[str_8000, False_8001], **kwargs_8002)
    
    # Getting the type of 'sys' (line 56)
    sys_8004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'sys')
    # Obtaining the member 'stypy_module_cache' of a type (line 56)
    stypy_module_cache_8005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 4), sys_8004, 'stypy_module_cache')
    str_8006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 27), 'str', '__builtin__')
    # Storing an element on a container (line 56)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 4), stypy_module_cache_8005, (str_8006, load_python_module_dynamically_call_result_8003))
    
    # Assigning a Call to a Subscript (line 57):
    
    # Assigning a Call to a Subscript (line 57):
    
    # Call to __load_python_module_dynamically(...): (line 57)
    # Processing the call arguments (line 57)
    str_8008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 72), 'str', 'ctypes')
    # Getting the type of 'False' (line 57)
    False_8009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 82), 'False', False)
    # Processing the call keyword arguments (line 57)
    kwargs_8010 = {}
    # Getting the type of '__load_python_module_dynamically' (line 57)
    load_python_module_dynamically_8007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 39), '__load_python_module_dynamically', False)
    # Calling __load_python_module_dynamically(args, kwargs) (line 57)
    load_python_module_dynamically_call_result_8011 = invoke(stypy.reporting.localization.Localization(__file__, 57, 39), load_python_module_dynamically_8007, *[str_8008, False_8009], **kwargs_8010)
    
    # Getting the type of 'sys' (line 57)
    sys_8012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'sys')
    # Obtaining the member 'stypy_module_cache' of a type (line 57)
    stypy_module_cache_8013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 4), sys_8012, 'stypy_module_cache')
    str_8014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 27), 'str', 'ctypes')
    # Storing an element on a container (line 57)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 4), stypy_module_cache_8013, (str_8014, load_python_module_dynamically_call_result_8011))
    
    # ################# End of '__preload_sys_module_cache(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__preload_sys_module_cache' in the type store
    # Getting the type of 'stypy_return_type' (line 44)
    stypy_return_type_8015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8015)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__preload_sys_module_cache'
    return stypy_return_type_8015

# Assigning a type to the variable '__preload_sys_module_cache' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), '__preload_sys_module_cache', __preload_sys_module_cache)

@norecursion
def __exist_module_in_sys_cache(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__exist_module_in_sys_cache'
    module_type_store = module_type_store.open_function_context('__exist_module_in_sys_cache', 60, 0, False)
    
    # Passed parameters checking function
    __exist_module_in_sys_cache.stypy_localization = localization
    __exist_module_in_sys_cache.stypy_type_of_self = None
    __exist_module_in_sys_cache.stypy_type_store = module_type_store
    __exist_module_in_sys_cache.stypy_function_name = '__exist_module_in_sys_cache'
    __exist_module_in_sys_cache.stypy_param_names_list = ['module_name']
    __exist_module_in_sys_cache.stypy_varargs_param_name = None
    __exist_module_in_sys_cache.stypy_kwargs_param_name = None
    __exist_module_in_sys_cache.stypy_call_defaults = defaults
    __exist_module_in_sys_cache.stypy_call_varargs = varargs
    __exist_module_in_sys_cache.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__exist_module_in_sys_cache', ['module_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__exist_module_in_sys_cache', localization, ['module_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__exist_module_in_sys_cache(...)' code ##################

    str_8016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, (-1)), 'str', '\n    Determines if a module called "module_name" (or whose .py file is equal to the argument) has been previously loaded\n    :param module_name: Module name (Python library modules) or file path (other modules) to check\n    :return: bool\n    ')
    
    
    # SSA begins for try-except statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Type idiom detected: calculating its left and rigth part (line 67)
    str_8017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 24), 'str', 'stypy_module_cache')
    # Getting the type of 'sys' (line 67)
    sys_8018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 19), 'sys')
    
    (may_be_8019, more_types_in_union_8020) = may_provide_member(str_8017, sys_8018)

    if may_be_8019:

        if more_types_in_union_8020:
            # Runtime conditional SSA (line 67)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'sys' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'sys', remove_not_member_provider_from_union(sys_8018, 'stypy_module_cache'))
        
        # Getting the type of 'module_name' (line 68)
        module_name_8021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'module_name')
        # Getting the type of 'sys' (line 68)
        sys_8022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 34), 'sys')
        # Obtaining the member 'stypy_module_cache' of a type (line 68)
        stypy_module_cache_8023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 34), sys_8022, 'stypy_module_cache')
        # Applying the binary operator 'in' (line 68)
        result_contains_8024 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 19), 'in', module_name_8021, stypy_module_cache_8023)
        
        # Assigning a type to the variable 'stypy_return_type' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'stypy_return_type', result_contains_8024)

        if more_types_in_union_8020:
            # Runtime conditional SSA for else branch (line 67)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_8019) or more_types_in_union_8020):
        # Assigning a type to the variable 'sys' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'sys', remove_member_provider_from_union(sys_8018, 'stypy_module_cache'))
        
        # Call to __preload_sys_module_cache(...): (line 70)
        # Processing the call keyword arguments (line 70)
        kwargs_8026 = {}
        # Getting the type of '__preload_sys_module_cache' (line 70)
        preload_sys_module_cache_8025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), '__preload_sys_module_cache', False)
        # Calling __preload_sys_module_cache(args, kwargs) (line 70)
        preload_sys_module_cache_call_result_8027 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), preload_sys_module_cache_8025, *[], **kwargs_8026)
        
        # Getting the type of 'False' (line 71)
        False_8028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'stypy_return_type', False_8028)

        if (may_be_8019 and more_types_in_union_8020):
            # SSA join for if statement (line 67)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the except part of a try statement (line 66)
    # SSA branch for the except '<any exception>' branch of a try statement (line 66)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'False' (line 73)
    False_8029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'stypy_return_type', False_8029)
    # SSA join for try-except statement (line 66)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '__exist_module_in_sys_cache(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__exist_module_in_sys_cache' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_8030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8030)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__exist_module_in_sys_cache'
    return stypy_return_type_8030

# Assigning a type to the variable '__exist_module_in_sys_cache' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), '__exist_module_in_sys_cache', __exist_module_in_sys_cache)

@norecursion
def get_module_from_sys_cache(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_module_from_sys_cache'
    module_type_store = module_type_store.open_function_context('get_module_from_sys_cache', 76, 0, False)
    
    # Passed parameters checking function
    get_module_from_sys_cache.stypy_localization = localization
    get_module_from_sys_cache.stypy_type_of_self = None
    get_module_from_sys_cache.stypy_type_store = module_type_store
    get_module_from_sys_cache.stypy_function_name = 'get_module_from_sys_cache'
    get_module_from_sys_cache.stypy_param_names_list = ['module_name']
    get_module_from_sys_cache.stypy_varargs_param_name = None
    get_module_from_sys_cache.stypy_kwargs_param_name = None
    get_module_from_sys_cache.stypy_call_defaults = defaults
    get_module_from_sys_cache.stypy_call_varargs = varargs
    get_module_from_sys_cache.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_module_from_sys_cache', ['module_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_module_from_sys_cache', localization, ['module_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_module_from_sys_cache(...)' code ##################

    str_8031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, (-1)), 'str', '\n    Gets a previously loaded module from the sys module cache\n    :param module_name: Module name\n    :return: A Type object or None if there is no such module\n    ')
    
    
    # SSA begins for try-except statement (line 82)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Type idiom detected: calculating its left and rigth part (line 83)
    str_8032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 24), 'str', 'stypy_module_cache')
    # Getting the type of 'sys' (line 83)
    sys_8033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'sys')
    
    (may_be_8034, more_types_in_union_8035) = may_provide_member(str_8032, sys_8033)

    if may_be_8034:

        if more_types_in_union_8035:
            # Runtime conditional SSA (line 83)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'sys' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'sys', remove_not_member_provider_from_union(sys_8033, 'stypy_module_cache'))
        
        # Obtaining the type of the subscript
        # Getting the type of 'module_name' (line 84)
        module_name_8036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 42), 'module_name')
        # Getting the type of 'sys' (line 84)
        sys_8037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 19), 'sys')
        # Obtaining the member 'stypy_module_cache' of a type (line 84)
        stypy_module_cache_8038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 19), sys_8037, 'stypy_module_cache')
        # Obtaining the member '__getitem__' of a type (line 84)
        getitem___8039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 19), stypy_module_cache_8038, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 84)
        subscript_call_result_8040 = invoke(stypy.reporting.localization.Localization(__file__, 84, 19), getitem___8039, module_name_8036)
        
        # Assigning a type to the variable 'stypy_return_type' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'stypy_return_type', subscript_call_result_8040)

        if more_types_in_union_8035:
            # Runtime conditional SSA for else branch (line 83)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_8034) or more_types_in_union_8035):
        # Assigning a type to the variable 'sys' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'sys', remove_member_provider_from_union(sys_8033, 'stypy_module_cache'))
        
        # Call to __preload_sys_module_cache(...): (line 86)
        # Processing the call keyword arguments (line 86)
        kwargs_8042 = {}
        # Getting the type of '__preload_sys_module_cache' (line 86)
        preload_sys_module_cache_8041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), '__preload_sys_module_cache', False)
        # Calling __preload_sys_module_cache(args, kwargs) (line 86)
        preload_sys_module_cache_call_result_8043 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), preload_sys_module_cache_8041, *[], **kwargs_8042)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'module_name' (line 87)
        module_name_8044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 42), 'module_name')
        # Getting the type of 'sys' (line 87)
        sys_8045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), 'sys')
        # Obtaining the member 'stypy_module_cache' of a type (line 87)
        stypy_module_cache_8046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 19), sys_8045, 'stypy_module_cache')
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___8047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 19), stypy_module_cache_8046, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_8048 = invoke(stypy.reporting.localization.Localization(__file__, 87, 19), getitem___8047, module_name_8044)
        
        # Assigning a type to the variable 'stypy_return_type' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'stypy_return_type', subscript_call_result_8048)

        if (may_be_8034 and more_types_in_union_8035):
            # SSA join for if statement (line 83)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the except part of a try statement (line 82)
    # SSA branch for the except '<any exception>' branch of a try statement (line 82)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'None' (line 89)
    None_8049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'stypy_return_type', None_8049)
    # SSA join for try-except statement (line 82)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'get_module_from_sys_cache(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_module_from_sys_cache' in the type store
    # Getting the type of 'stypy_return_type' (line 76)
    stypy_return_type_8050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8050)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_module_from_sys_cache'
    return stypy_return_type_8050

# Assigning a type to the variable 'get_module_from_sys_cache' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'get_module_from_sys_cache', get_module_from_sys_cache)

@norecursion
def __put_module_in_sys_cache(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__put_module_in_sys_cache'
    module_type_store = module_type_store.open_function_context('__put_module_in_sys_cache', 92, 0, False)
    
    # Passed parameters checking function
    __put_module_in_sys_cache.stypy_localization = localization
    __put_module_in_sys_cache.stypy_type_of_self = None
    __put_module_in_sys_cache.stypy_type_store = module_type_store
    __put_module_in_sys_cache.stypy_function_name = '__put_module_in_sys_cache'
    __put_module_in_sys_cache.stypy_param_names_list = ['module_name', 'module_obj']
    __put_module_in_sys_cache.stypy_varargs_param_name = None
    __put_module_in_sys_cache.stypy_kwargs_param_name = None
    __put_module_in_sys_cache.stypy_call_defaults = defaults
    __put_module_in_sys_cache.stypy_call_varargs = varargs
    __put_module_in_sys_cache.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__put_module_in_sys_cache', ['module_name', 'module_obj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__put_module_in_sys_cache', localization, ['module_name', 'module_obj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__put_module_in_sys_cache(...)' code ##################

    str_8051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, (-1)), 'str', '\n    Puts a module in the sys stypy module cache\n    :param module_name: Name of the module\n    :param module_obj: Object representing the module\n    :return: None\n    ')
    
    # Assigning a Name to a Subscript (line 101):
    
    # Assigning a Name to a Subscript (line 101):
    # Getting the type of 'module_obj' (line 101)
    module_obj_8052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 42), 'module_obj')
    # Getting the type of 'sys' (line 101)
    sys_8053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'sys')
    # Obtaining the member 'stypy_module_cache' of a type (line 101)
    stypy_module_cache_8054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 4), sys_8053, 'stypy_module_cache')
    # Getting the type of 'module_name' (line 101)
    module_name_8055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'module_name')
    # Storing an element on a container (line 101)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 4), stypy_module_cache_8054, (module_name_8055, module_obj_8052))
    
    # ################# End of '__put_module_in_sys_cache(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__put_module_in_sys_cache' in the type store
    # Getting the type of 'stypy_return_type' (line 92)
    stypy_return_type_8056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8056)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__put_module_in_sys_cache'
    return stypy_return_type_8056

# Assigning a type to the variable '__put_module_in_sys_cache' (line 92)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), '__put_module_in_sys_cache', __put_module_in_sys_cache)

@norecursion
def __import_python_library_module(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_8057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 61), 'str', '__builtin__')
    defaults = [str_8057]
    # Create a new context for function '__import_python_library_module'
    module_type_store = module_type_store.open_function_context('__import_python_library_module', 111, 0, False)
    
    # Passed parameters checking function
    __import_python_library_module.stypy_localization = localization
    __import_python_library_module.stypy_type_of_self = None
    __import_python_library_module.stypy_type_store = module_type_store
    __import_python_library_module.stypy_function_name = '__import_python_library_module'
    __import_python_library_module.stypy_param_names_list = ['localization', 'module_name']
    __import_python_library_module.stypy_varargs_param_name = None
    __import_python_library_module.stypy_kwargs_param_name = None
    __import_python_library_module.stypy_call_defaults = defaults
    __import_python_library_module.stypy_call_varargs = varargs
    __import_python_library_module.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__import_python_library_module', ['localization', 'module_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__import_python_library_module', localization, ['localization', 'module_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__import_python_library_module(...)' code ##################

    str_8058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, (-1)), 'str', '\n    Import a full Python library module (models the "import <module>" statement for Python library modules\n    :param localization: Caller information\n    :param module_name: Module to import\n    :return: A TypeInferenceProxy for the specified module or a TypeError if the requested module do not exist\n    ')
    
    
    # SSA begins for try-except statement (line 118)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 119):
    
    # Assigning a Call to a Name (line 119):
    
    # Call to get_module_from_sys_cache(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'module_name' (line 119)
    module_name_8060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 47), 'module_name', False)
    # Processing the call keyword arguments (line 119)
    kwargs_8061 = {}
    # Getting the type of 'get_module_from_sys_cache' (line 119)
    get_module_from_sys_cache_8059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 21), 'get_module_from_sys_cache', False)
    # Calling get_module_from_sys_cache(args, kwargs) (line 119)
    get_module_from_sys_cache_call_result_8062 = invoke(stypy.reporting.localization.Localization(__file__, 119, 21), get_module_from_sys_cache_8059, *[module_name_8060], **kwargs_8061)
    
    # Assigning a type to the variable 'module_obj' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'module_obj', get_module_from_sys_cache_call_result_8062)
    
    # Type idiom detected: calculating its left and rigth part (line 120)
    # Getting the type of 'module_obj' (line 120)
    module_obj_8063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 11), 'module_obj')
    # Getting the type of 'None' (line 120)
    None_8064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 25), 'None')
    
    (may_be_8065, more_types_in_union_8066) = may_be_none(module_obj_8063, None_8064)

    if may_be_8065:

        if more_types_in_union_8066:
            # Runtime conditional SSA (line 120)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 121):
        
        # Assigning a Call to a Name (line 121):
        
        # Call to __load_python_module_dynamically(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'module_name' (line 121)
        module_name_8068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 58), 'module_name', False)
        # Processing the call keyword arguments (line 121)
        kwargs_8069 = {}
        # Getting the type of '__load_python_module_dynamically' (line 121)
        load_python_module_dynamically_8067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 25), '__load_python_module_dynamically', False)
        # Calling __load_python_module_dynamically(args, kwargs) (line 121)
        load_python_module_dynamically_call_result_8070 = invoke(stypy.reporting.localization.Localization(__file__, 121, 25), load_python_module_dynamically_8067, *[module_name_8068], **kwargs_8069)
        
        # Assigning a type to the variable 'module_obj' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'module_obj', load_python_module_dynamically_call_result_8070)
        
        # Assigning a Call to a Name (line 122):
        
        # Assigning a Call to a Name (line 122):
        
        # Call to get_python_entity(...): (line 122)
        # Processing the call keyword arguments (line 122)
        kwargs_8073 = {}
        # Getting the type of 'module_obj' (line 122)
        module_obj_8071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 21), 'module_obj', False)
        # Obtaining the member 'get_python_entity' of a type (line 122)
        get_python_entity_8072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 21), module_obj_8071, 'get_python_entity')
        # Calling get_python_entity(args, kwargs) (line 122)
        get_python_entity_call_result_8074 = invoke(stypy.reporting.localization.Localization(__file__, 122, 21), get_python_entity_8072, *[], **kwargs_8073)
        
        # Assigning a type to the variable 'module' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'module', get_python_entity_call_result_8074)
        
        # Assigning a Attribute to a Name (line 124):
        
        # Assigning a Attribute to a Name (line 124):
        # Getting the type of 'module' (line 124)
        module_8075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 29), 'module')
        # Obtaining the member '__dict__' of a type (line 124)
        dict___8076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 29), module_8075, '__dict__')
        # Assigning a type to the variable 'module_members' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'module_members', dict___8076)
        
        # Getting the type of 'module_members' (line 125)
        module_members_8077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 26), 'module_members')
        # Assigning a type to the variable 'module_members_8077' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'module_members_8077', module_members_8077)
        # Testing if the for loop is going to be iterated (line 125)
        # Testing the type of a for loop iterable (line 125)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 125, 12), module_members_8077)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 125, 12), module_members_8077):
            # Getting the type of the for loop variable (line 125)
            for_loop_var_8078 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 125, 12), module_members_8077)
            # Assigning a type to the variable 'member' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'member', for_loop_var_8078)
            # SSA begins for a for statement (line 125)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to ismodule(...): (line 126)
            # Processing the call arguments (line 126)
            
            # Obtaining the type of the subscript
            # Getting the type of 'member' (line 126)
            member_8081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 51), 'member', False)
            # Getting the type of 'module_members' (line 126)
            module_members_8082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 36), 'module_members', False)
            # Obtaining the member '__getitem__' of a type (line 126)
            getitem___8083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 36), module_members_8082, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 126)
            subscript_call_result_8084 = invoke(stypy.reporting.localization.Localization(__file__, 126, 36), getitem___8083, member_8081)
            
            # Processing the call keyword arguments (line 126)
            kwargs_8085 = {}
            # Getting the type of 'inspect' (line 126)
            inspect_8079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 19), 'inspect', False)
            # Obtaining the member 'ismodule' of a type (line 126)
            ismodule_8080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 19), inspect_8079, 'ismodule')
            # Calling ismodule(args, kwargs) (line 126)
            ismodule_call_result_8086 = invoke(stypy.reporting.localization.Localization(__file__, 126, 19), ismodule_8080, *[subscript_call_result_8084], **kwargs_8085)
            
            # Testing if the type of an if condition is none (line 126)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 126, 16), ismodule_call_result_8086):
                pass
            else:
                
                # Testing the type of an if condition (line 126)
                if_condition_8087 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 16), ismodule_call_result_8086)
                # Assigning a type to the variable 'if_condition_8087' (line 126)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'if_condition_8087', if_condition_8087)
                # SSA begins for if statement (line 126)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Attribute to a Name (line 127):
                
                # Assigning a Attribute to a Name (line 127):
                
                # Obtaining the type of the subscript
                # Getting the type of 'member' (line 127)
                member_8088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 56), 'member')
                # Getting the type of 'module_members' (line 127)
                module_members_8089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 41), 'module_members')
                # Obtaining the member '__getitem__' of a type (line 127)
                getitem___8090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 41), module_members_8089, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                subscript_call_result_8091 = invoke(stypy.reporting.localization.Localization(__file__, 127, 41), getitem___8090, member_8088)
                
                # Obtaining the member '__name__' of a type (line 127)
                name___8092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 41), subscript_call_result_8091, '__name__')
                # Assigning a type to the variable 'member_module_name' (line 127)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'member_module_name', name___8092)
                
                # Getting the type of 'member_module_name' (line 129)
                member_module_name_8093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), 'member_module_name')
                # Getting the type of 'module_name' (line 129)
                module_name_8094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 49), 'module_name')
                # Applying the binary operator 'isnot' (line 129)
                result_is_not_8095 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 23), 'isnot', member_module_name_8093, module_name_8094)
                
                # Testing if the type of an if condition is none (line 129)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 129, 20), result_is_not_8095):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 129)
                    if_condition_8096 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 20), result_is_not_8095)
                    # Assigning a type to the variable 'if_condition_8096' (line 129)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 20), 'if_condition_8096', if_condition_8096)
                    # SSA begins for if statement (line 129)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    
                    # Call to __exist_module_in_sys_cache(...): (line 130)
                    # Processing the call arguments (line 130)
                    # Getting the type of 'member_module_name' (line 130)
                    member_module_name_8098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 59), 'member_module_name', False)
                    # Processing the call keyword arguments (line 130)
                    kwargs_8099 = {}
                    # Getting the type of '__exist_module_in_sys_cache' (line 130)
                    exist_module_in_sys_cache_8097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 31), '__exist_module_in_sys_cache', False)
                    # Calling __exist_module_in_sys_cache(args, kwargs) (line 130)
                    exist_module_in_sys_cache_call_result_8100 = invoke(stypy.reporting.localization.Localization(__file__, 130, 31), exist_module_in_sys_cache_8097, *[member_module_name_8098], **kwargs_8099)
                    
                    # Applying the 'not' unary operator (line 130)
                    result_not__8101 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 27), 'not', exist_module_in_sys_cache_call_result_8100)
                    
                    # Testing if the type of an if condition is none (line 130)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 130, 24), result_not__8101):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 130)
                        if_condition_8102 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 24), result_not__8101)
                        # Assigning a type to the variable 'if_condition_8102' (line 130)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'if_condition_8102', if_condition_8102)
                        # SSA begins for if statement (line 130)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Call to a Name (line 131):
                        
                        # Assigning a Call to a Name (line 131):
                        
                        # Call to __load_python_module_dynamically(...): (line 131)
                        # Processing the call arguments (line 131)
                        # Getting the type of 'member_module_name' (line 131)
                        member_module_name_8104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 73), 'member_module_name', False)
                        # Processing the call keyword arguments (line 131)
                        kwargs_8105 = {}
                        # Getting the type of '__load_python_module_dynamically' (line 131)
                        load_python_module_dynamically_8103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 40), '__load_python_module_dynamically', False)
                        # Calling __load_python_module_dynamically(args, kwargs) (line 131)
                        load_python_module_dynamically_call_result_8106 = invoke(stypy.reporting.localization.Localization(__file__, 131, 40), load_python_module_dynamically_8103, *[member_module_name_8104], **kwargs_8105)
                        
                        # Assigning a type to the variable 'module_ti' (line 131)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 28), 'module_ti', load_python_module_dynamically_call_result_8106)
                        
                        # Call to set_type_of_member(...): (line 132)
                        # Processing the call arguments (line 132)
                        # Getting the type of 'localization' (line 132)
                        localization_8109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 58), 'localization', False)
                        # Getting the type of 'member' (line 132)
                        member_8110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 72), 'member', False)
                        # Getting the type of 'module_ti' (line 132)
                        module_ti_8111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 80), 'module_ti', False)
                        # Processing the call keyword arguments (line 132)
                        kwargs_8112 = {}
                        # Getting the type of 'module_obj' (line 132)
                        module_obj_8107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 28), 'module_obj', False)
                        # Obtaining the member 'set_type_of_member' of a type (line 132)
                        set_type_of_member_8108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 28), module_obj_8107, 'set_type_of_member')
                        # Calling set_type_of_member(args, kwargs) (line 132)
                        set_type_of_member_call_result_8113 = invoke(stypy.reporting.localization.Localization(__file__, 132, 28), set_type_of_member_8108, *[localization_8109, member_8110, module_ti_8111], **kwargs_8112)
                        
                        # SSA join for if statement (line 130)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 129)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 126)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        

        if more_types_in_union_8066:
            # SSA join for if statement (line 120)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'module_obj' (line 133)
    module_obj_8114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'module_obj')
    # Assigning a type to the variable 'stypy_return_type' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'stypy_return_type', module_obj_8114)
    # SSA branch for the except part of a try statement (line 118)
    # SSA branch for the except 'Exception' branch of a try statement (line 118)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 134)
    Exception_8115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 11), 'Exception')
    # Assigning a type to the variable 'exc' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'exc', Exception_8115)
    
    # Call to TypeError(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'localization' (line 135)
    localization_8117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 25), 'localization', False)
    
    # Call to format(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'module_name' (line 135)
    module_name_8120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 96), 'module_name', False)
    
    # Call to str(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'exc' (line 135)
    exc_8122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 113), 'exc', False)
    # Processing the call keyword arguments (line 135)
    kwargs_8123 = {}
    # Getting the type of 'str' (line 135)
    str_8121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 109), 'str', False)
    # Calling str(args, kwargs) (line 135)
    str_call_result_8124 = invoke(stypy.reporting.localization.Localization(__file__, 135, 109), str_8121, *[exc_8122], **kwargs_8123)
    
    # Processing the call keyword arguments (line 135)
    kwargs_8125 = {}
    str_8118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 39), 'str', "Could not load Python library module '{0}': {1}")
    # Obtaining the member 'format' of a type (line 135)
    format_8119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 39), str_8118, 'format')
    # Calling format(args, kwargs) (line 135)
    format_call_result_8126 = invoke(stypy.reporting.localization.Localization(__file__, 135, 39), format_8119, *[module_name_8120, str_call_result_8124], **kwargs_8125)
    
    # Processing the call keyword arguments (line 135)
    kwargs_8127 = {}
    # Getting the type of 'TypeError' (line 135)
    TypeError_8116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 135)
    TypeError_call_result_8128 = invoke(stypy.reporting.localization.Localization(__file__, 135, 15), TypeError_8116, *[localization_8117, format_call_result_8126], **kwargs_8127)
    
    # Assigning a type to the variable 'stypy_return_type' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'stypy_return_type', TypeError_call_result_8128)
    # SSA join for try-except statement (line 118)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '__import_python_library_module(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__import_python_library_module' in the type store
    # Getting the type of 'stypy_return_type' (line 111)
    stypy_return_type_8129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8129)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__import_python_library_module'
    return stypy_return_type_8129

# Assigning a type to the variable '__import_python_library_module' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), '__import_python_library_module', __import_python_library_module)

@norecursion
def __get_non_python_library_module_file(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'sys' (line 138)
    sys_8130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 66), 'sys')
    # Obtaining the member 'path' of a type (line 138)
    path_8131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 66), sys_8130, 'path')
    defaults = [path_8131]
    # Create a new context for function '__get_non_python_library_module_file'
    module_type_store = module_type_store.open_function_context('__get_non_python_library_module_file', 138, 0, False)
    
    # Passed parameters checking function
    __get_non_python_library_module_file.stypy_localization = localization
    __get_non_python_library_module_file.stypy_type_of_self = None
    __get_non_python_library_module_file.stypy_type_store = module_type_store
    __get_non_python_library_module_file.stypy_function_name = '__get_non_python_library_module_file'
    __get_non_python_library_module_file.stypy_param_names_list = ['module_name', 'environment']
    __get_non_python_library_module_file.stypy_varargs_param_name = None
    __get_non_python_library_module_file.stypy_kwargs_param_name = None
    __get_non_python_library_module_file.stypy_call_defaults = defaults
    __get_non_python_library_module_file.stypy_call_varargs = varargs
    __get_non_python_library_module_file.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__get_non_python_library_module_file', ['module_name', 'environment'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__get_non_python_library_module_file', localization, ['module_name', 'environment'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__get_non_python_library_module_file(...)' code ##################

    str_8132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, (-1)), 'str', '\n    Obtains the source file in which a module source code resides.\n    :module_name Name of the module whose source file we intend to find\n    :environment (Optional) List of paths to use to search the module (defaults to sys.path)\n    :return: str or None\n    ')
    
    # Assigning a Name to a Name (line 145):
    
    # Assigning a Name to a Name (line 145):
    # Getting the type of 'None' (line 145)
    None_8133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'None')
    # Assigning a type to the variable 'found' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'found', None_8133)
    
    # Assigning a Call to a Name (line 148):
    
    # Assigning a Call to a Name (line 148):
    
    # Call to reversed(...): (line 148)
    # Processing the call arguments (line 148)
    
    # Call to sorted(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'environment' (line 148)
    environment_8136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 28), 'environment', False)
    # Processing the call keyword arguments (line 148)
    kwargs_8137 = {}
    # Getting the type of 'sorted' (line 148)
    sorted_8135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 21), 'sorted', False)
    # Calling sorted(args, kwargs) (line 148)
    sorted_call_result_8138 = invoke(stypy.reporting.localization.Localization(__file__, 148, 21), sorted_8135, *[environment_8136], **kwargs_8137)
    
    # Processing the call keyword arguments (line 148)
    kwargs_8139 = {}
    # Getting the type of 'reversed' (line 148)
    reversed_8134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'reversed', False)
    # Calling reversed(args, kwargs) (line 148)
    reversed_call_result_8140 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), reversed_8134, *[sorted_call_result_8138], **kwargs_8139)
    
    # Assigning a type to the variable 'paths' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'paths', reversed_call_result_8140)
    
    # Getting the type of 'paths' (line 149)
    paths_8141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'paths')
    # Assigning a type to the variable 'paths_8141' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'paths_8141', paths_8141)
    # Testing if the for loop is going to be iterated (line 149)
    # Testing the type of a for loop iterable (line 149)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 149, 4), paths_8141)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 149, 4), paths_8141):
        # Getting the type of the for loop variable (line 149)
        for_loop_var_8142 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 149, 4), paths_8141)
        # Assigning a type to the variable 'path' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'path', for_loop_var_8142)
        # SSA begins for a for statement (line 149)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 150):
        
        # Assigning a Call to a Name (line 150):
        
        # Call to replace(...): (line 150)
        # Processing the call arguments (line 150)
        str_8145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 33), 'str', '\\')
        str_8146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 39), 'str', '/')
        # Processing the call keyword arguments (line 150)
        kwargs_8147 = {}
        # Getting the type of 'path' (line 150)
        path_8143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 20), 'path', False)
        # Obtaining the member 'replace' of a type (line 150)
        replace_8144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 20), path_8143, 'replace')
        # Calling replace(args, kwargs) (line 150)
        replace_call_result_8148 = invoke(stypy.reporting.localization.Localization(__file__, 150, 20), replace_8144, *[str_8145, str_8146], **kwargs_8147)
        
        # Assigning a type to the variable 'base_path' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'base_path', replace_call_result_8148)
        
        # Getting the type of 'stypy_parameters_copy' (line 151)
        stypy_parameters_copy_8149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'stypy_parameters_copy')
        # Obtaining the member 'type_inference_file_directory_name' of a type (line 151)
        type_inference_file_directory_name_8150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 11), stypy_parameters_copy_8149, 'type_inference_file_directory_name')
        # Getting the type of 'path' (line 151)
        path_8151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 71), 'path')
        # Applying the binary operator 'in' (line 151)
        result_contains_8152 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 11), 'in', type_inference_file_directory_name_8150, path_8151)
        
        # Testing if the type of an if condition is none (line 151)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 151, 8), result_contains_8152):
            pass
        else:
            
            # Testing the type of an if condition (line 151)
            if_condition_8153 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 8), result_contains_8152)
            # Assigning a type to the variable 'if_condition_8153' (line 151)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'if_condition_8153', if_condition_8153)
            # SSA begins for if statement (line 151)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 152):
            
            # Assigning a Call to a Name (line 152):
            
            # Call to replace(...): (line 152)
            # Processing the call arguments (line 152)
            str_8156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 42), 'str', '/')
            # Getting the type of 'stypy_parameters_copy' (line 152)
            stypy_parameters_copy_8157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 48), 'stypy_parameters_copy', False)
            # Obtaining the member 'type_inference_file_directory_name' of a type (line 152)
            type_inference_file_directory_name_8158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 48), stypy_parameters_copy_8157, 'type_inference_file_directory_name')
            # Applying the binary operator '+' (line 152)
            result_add_8159 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 42), '+', str_8156, type_inference_file_directory_name_8158)
            
            str_8160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 106), 'str', '')
            # Processing the call keyword arguments (line 152)
            kwargs_8161 = {}
            # Getting the type of 'base_path' (line 152)
            base_path_8154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 24), 'base_path', False)
            # Obtaining the member 'replace' of a type (line 152)
            replace_8155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 24), base_path_8154, 'replace')
            # Calling replace(args, kwargs) (line 152)
            replace_call_result_8162 = invoke(stypy.reporting.localization.Localization(__file__, 152, 24), replace_8155, *[result_add_8159, str_8160], **kwargs_8161)
            
            # Assigning a type to the variable 'base_path' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'base_path', replace_call_result_8162)
            # SSA join for if statement (line 151)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 154):
        
        # Assigning a BinOp to a Name (line 154):
        # Getting the type of 'base_path' (line 154)
        base_path_8163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'base_path')
        str_8164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 27), 'str', '/')
        # Applying the binary operator '+' (line 154)
        result_add_8165 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 15), '+', base_path_8163, str_8164)
        
        
        # Call to replace(...): (line 154)
        # Processing the call arguments (line 154)
        str_8168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 53), 'str', '.')
        str_8169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 58), 'str', '/')
        # Processing the call keyword arguments (line 154)
        kwargs_8170 = {}
        # Getting the type of 'module_name' (line 154)
        module_name_8166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 33), 'module_name', False)
        # Obtaining the member 'replace' of a type (line 154)
        replace_8167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 33), module_name_8166, 'replace')
        # Calling replace(args, kwargs) (line 154)
        replace_call_result_8171 = invoke(stypy.reporting.localization.Localization(__file__, 154, 33), replace_8167, *[str_8168, str_8169], **kwargs_8170)
        
        # Applying the binary operator '+' (line 154)
        result_add_8172 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 31), '+', result_add_8165, replace_call_result_8171)
        
        str_8173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 65), 'str', '.py')
        # Applying the binary operator '+' (line 154)
        result_add_8174 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 63), '+', result_add_8172, str_8173)
        
        # Assigning a type to the variable 'temp' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'temp', result_add_8174)
        
        # Call to isfile(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'temp' (line 155)
        temp_8178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 26), 'temp', False)
        # Processing the call keyword arguments (line 155)
        kwargs_8179 = {}
        # Getting the type of 'os' (line 155)
        os_8175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'os', False)
        # Obtaining the member 'path' of a type (line 155)
        path_8176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 11), os_8175, 'path')
        # Obtaining the member 'isfile' of a type (line 155)
        isfile_8177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 11), path_8176, 'isfile')
        # Calling isfile(args, kwargs) (line 155)
        isfile_call_result_8180 = invoke(stypy.reporting.localization.Localization(__file__, 155, 11), isfile_8177, *[temp_8178], **kwargs_8179)
        
        # Testing if the type of an if condition is none (line 155)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 155, 8), isfile_call_result_8180):
            pass
        else:
            
            # Testing the type of an if condition (line 155)
            if_condition_8181 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 8), isfile_call_result_8180)
            # Assigning a type to the variable 'if_condition_8181' (line 155)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'if_condition_8181', if_condition_8181)
            # SSA begins for if statement (line 155)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 156):
            
            # Assigning a Name to a Name (line 156):
            # Getting the type of 'temp' (line 156)
            temp_8182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 20), 'temp')
            # Assigning a type to the variable 'found' (line 156)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'found', temp_8182)
            # SSA join for if statement (line 155)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 158):
        
        # Assigning a BinOp to a Name (line 158):
        # Getting the type of 'base_path' (line 158)
        base_path_8183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 15), 'base_path')
        str_8184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 27), 'str', '/')
        # Applying the binary operator '+' (line 158)
        result_add_8185 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 15), '+', base_path_8183, str_8184)
        
        
        # Call to replace(...): (line 158)
        # Processing the call arguments (line 158)
        str_8188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 53), 'str', '.')
        str_8189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 58), 'str', '/')
        # Processing the call keyword arguments (line 158)
        kwargs_8190 = {}
        # Getting the type of 'module_name' (line 158)
        module_name_8186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 33), 'module_name', False)
        # Obtaining the member 'replace' of a type (line 158)
        replace_8187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 33), module_name_8186, 'replace')
        # Calling replace(args, kwargs) (line 158)
        replace_call_result_8191 = invoke(stypy.reporting.localization.Localization(__file__, 158, 33), replace_8187, *[str_8188, str_8189], **kwargs_8190)
        
        # Applying the binary operator '+' (line 158)
        result_add_8192 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 31), '+', result_add_8185, replace_call_result_8191)
        
        str_8193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 65), 'str', '/__init__.py')
        # Applying the binary operator '+' (line 158)
        result_add_8194 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 63), '+', result_add_8192, str_8193)
        
        # Assigning a type to the variable 'temp' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'temp', result_add_8194)
        
        # Call to isfile(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'temp' (line 159)
        temp_8198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 26), 'temp', False)
        # Processing the call keyword arguments (line 159)
        kwargs_8199 = {}
        # Getting the type of 'os' (line 159)
        os_8195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 11), 'os', False)
        # Obtaining the member 'path' of a type (line 159)
        path_8196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 11), os_8195, 'path')
        # Obtaining the member 'isfile' of a type (line 159)
        isfile_8197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 11), path_8196, 'isfile')
        # Calling isfile(args, kwargs) (line 159)
        isfile_call_result_8200 = invoke(stypy.reporting.localization.Localization(__file__, 159, 11), isfile_8197, *[temp_8198], **kwargs_8199)
        
        # Testing if the type of an if condition is none (line 159)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 159, 8), isfile_call_result_8200):
            pass
        else:
            
            # Testing the type of an if condition (line 159)
            if_condition_8201 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 8), isfile_call_result_8200)
            # Assigning a type to the variable 'if_condition_8201' (line 159)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'if_condition_8201', if_condition_8201)
            # SSA begins for if statement (line 159)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 160):
            
            # Assigning a Name to a Name (line 160):
            # Getting the type of 'temp' (line 160)
            temp_8202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'temp')
            # Assigning a type to the variable 'found' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'found', temp_8202)
            # SSA join for if statement (line 159)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Type idiom detected: calculating its left and rigth part (line 162)
    # Getting the type of 'found' (line 162)
    found_8203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 7), 'found')
    # Getting the type of 'None' (line 162)
    None_8204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'None')
    
    (may_be_8205, more_types_in_union_8206) = may_be_none(found_8203, None_8204)

    if may_be_8205:

        if more_types_in_union_8206:
            # Runtime conditional SSA (line 162)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        pass

        if more_types_in_union_8206:
            # SSA join for if statement (line 162)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'found' (line 165)
    found_8207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 11), 'found')
    # Assigning a type to the variable 'stypy_return_type' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'stypy_return_type', found_8207)
    
    # ################# End of '__get_non_python_library_module_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__get_non_python_library_module_file' in the type store
    # Getting the type of 'stypy_return_type' (line 138)
    stypy_return_type_8208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8208)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__get_non_python_library_module_file'
    return stypy_return_type_8208

# Assigning a type to the variable '__get_non_python_library_module_file' (line 138)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), '__get_non_python_library_module_file', __get_non_python_library_module_file)

@norecursion
def __get_module_file(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__get_module_file'
    module_type_store = module_type_store.open_function_context('__get_module_file', 168, 0, False)
    
    # Passed parameters checking function
    __get_module_file.stypy_localization = localization
    __get_module_file.stypy_type_of_self = None
    __get_module_file.stypy_type_store = module_type_store
    __get_module_file.stypy_function_name = '__get_module_file'
    __get_module_file.stypy_param_names_list = ['module_name']
    __get_module_file.stypy_varargs_param_name = None
    __get_module_file.stypy_kwargs_param_name = None
    __get_module_file.stypy_call_defaults = defaults
    __get_module_file.stypy_call_varargs = varargs
    __get_module_file.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__get_module_file', ['module_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__get_module_file', localization, ['module_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__get_module_file(...)' code ##################

    
    # Assigning a Name to a Name (line 169):
    
    # Assigning a Name to a Name (line 169):
    # Getting the type of 'None' (line 169)
    None_8209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 18), 'None')
    # Assigning a type to the variable 'module_file' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'module_file', None_8209)
    
    # Assigning a Name to a Name (line 170):
    
    # Assigning a Name to a Name (line 170):
    # Getting the type of 'None' (line 170)
    None_8210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 20), 'None')
    # Assigning a type to the variable 'loaded_module' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'loaded_module', None_8210)
    
    # Assigning a Name to a Name (line 171):
    
    # Assigning a Name to a Name (line 171):
    # Getting the type of 'None' (line 171)
    None_8211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 24), 'None')
    # Assigning a type to the variable 'module_type_store' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'module_type_store', None_8211)
    
    # Getting the type of 'module_name' (line 172)
    module_name_8212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 7), 'module_name')
    # Getting the type of 'sys' (line 172)
    sys_8213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 22), 'sys')
    # Obtaining the member 'modules' of a type (line 172)
    modules_8214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 22), sys_8213, 'modules')
    # Applying the binary operator 'in' (line 172)
    result_contains_8215 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 7), 'in', module_name_8212, modules_8214)
    
    # Testing if the type of an if condition is none (line 172)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 172, 4), result_contains_8215):
        
        # Assigning a Call to a Name (line 177):
        
        # Assigning a Call to a Name (line 177):
        
        # Call to __import__(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'module_name' (line 177)
        module_name_8229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 35), 'module_name', False)
        # Processing the call keyword arguments (line 177)
        kwargs_8230 = {}
        # Getting the type of '__import__' (line 177)
        import___8228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 24), '__import__', False)
        # Calling __import__(args, kwargs) (line 177)
        import___call_result_8231 = invoke(stypy.reporting.localization.Localization(__file__, 177, 24), import___8228, *[module_name_8229], **kwargs_8230)
        
        # Assigning a type to the variable 'loaded_module' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'loaded_module', import___call_result_8231)
        
        # Type idiom detected: calculating its left and rigth part (line 178)
        str_8232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 34), 'str', '__file__')
        # Getting the type of 'loaded_module' (line 178)
        loaded_module_8233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 19), 'loaded_module')
        
        (may_be_8234, more_types_in_union_8235) = may_provide_member(str_8232, loaded_module_8233)

        if may_be_8234:

            if more_types_in_union_8235:
                # Runtime conditional SSA (line 178)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'loaded_module' (line 178)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'loaded_module', remove_not_member_provider_from_union(loaded_module_8233, '__file__'))
            
            # Assigning a Attribute to a Name (line 179):
            
            # Assigning a Attribute to a Name (line 179):
            # Getting the type of 'loaded_module' (line 179)
            loaded_module_8236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 26), 'loaded_module')
            # Obtaining the member '__file__' of a type (line 179)
            file___8237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 26), loaded_module_8236, '__file__')
            # Assigning a type to the variable 'module_file' (line 179)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'module_file', file___8237)

            if more_types_in_union_8235:
                # SSA join for if statement (line 178)
                module_type_store = module_type_store.join_ssa_context()


        
    else:
        
        # Testing the type of an if condition (line 172)
        if_condition_8216 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 4), result_contains_8215)
        # Assigning a type to the variable 'if_condition_8216' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'if_condition_8216', if_condition_8216)
        # SSA begins for if statement (line 172)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 173):
        
        # Assigning a Subscript to a Name (line 173):
        
        # Obtaining the type of the subscript
        # Getting the type of 'module_name' (line 173)
        module_name_8217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 36), 'module_name')
        # Getting the type of 'sys' (line 173)
        sys_8218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 24), 'sys')
        # Obtaining the member 'modules' of a type (line 173)
        modules_8219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 24), sys_8218, 'modules')
        # Obtaining the member '__getitem__' of a type (line 173)
        getitem___8220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 24), modules_8219, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 173)
        subscript_call_result_8221 = invoke(stypy.reporting.localization.Localization(__file__, 173, 24), getitem___8220, module_name_8217)
        
        # Assigning a type to the variable 'loaded_module' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'loaded_module', subscript_call_result_8221)
        
        # Type idiom detected: calculating its left and rigth part (line 174)
        str_8222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 34), 'str', '__file__')
        # Getting the type of 'loaded_module' (line 174)
        loaded_module_8223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 19), 'loaded_module')
        
        (may_be_8224, more_types_in_union_8225) = may_provide_member(str_8222, loaded_module_8223)

        if may_be_8224:

            if more_types_in_union_8225:
                # Runtime conditional SSA (line 174)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'loaded_module' (line 174)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'loaded_module', remove_not_member_provider_from_union(loaded_module_8223, '__file__'))
            
            # Assigning a Attribute to a Name (line 175):
            
            # Assigning a Attribute to a Name (line 175):
            # Getting the type of 'loaded_module' (line 175)
            loaded_module_8226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 26), 'loaded_module')
            # Obtaining the member '__file__' of a type (line 175)
            file___8227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 26), loaded_module_8226, '__file__')
            # Assigning a type to the variable 'module_file' (line 175)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'module_file', file___8227)

            if more_types_in_union_8225:
                # SSA join for if statement (line 174)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA branch for the else part of an if statement (line 172)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 177):
        
        # Assigning a Call to a Name (line 177):
        
        # Call to __import__(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'module_name' (line 177)
        module_name_8229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 35), 'module_name', False)
        # Processing the call keyword arguments (line 177)
        kwargs_8230 = {}
        # Getting the type of '__import__' (line 177)
        import___8228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 24), '__import__', False)
        # Calling __import__(args, kwargs) (line 177)
        import___call_result_8231 = invoke(stypy.reporting.localization.Localization(__file__, 177, 24), import___8228, *[module_name_8229], **kwargs_8230)
        
        # Assigning a type to the variable 'loaded_module' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'loaded_module', import___call_result_8231)
        
        # Type idiom detected: calculating its left and rigth part (line 178)
        str_8232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 34), 'str', '__file__')
        # Getting the type of 'loaded_module' (line 178)
        loaded_module_8233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 19), 'loaded_module')
        
        (may_be_8234, more_types_in_union_8235) = may_provide_member(str_8232, loaded_module_8233)

        if may_be_8234:

            if more_types_in_union_8235:
                # Runtime conditional SSA (line 178)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'loaded_module' (line 178)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'loaded_module', remove_not_member_provider_from_union(loaded_module_8233, '__file__'))
            
            # Assigning a Attribute to a Name (line 179):
            
            # Assigning a Attribute to a Name (line 179):
            # Getting the type of 'loaded_module' (line 179)
            loaded_module_8236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 26), 'loaded_module')
            # Obtaining the member '__file__' of a type (line 179)
            file___8237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 26), loaded_module_8236, '__file__')
            # Assigning a type to the variable 'module_file' (line 179)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'module_file', file___8237)

            if more_types_in_union_8235:
                # SSA join for if statement (line 178)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 172)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Type idiom detected: calculating its left and rigth part (line 180)
    # Getting the type of 'module_file' (line 180)
    module_file_8238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 7), 'module_file')
    # Getting the type of 'None' (line 180)
    None_8239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 22), 'None')
    
    (may_be_8240, more_types_in_union_8241) = may_be_none(module_file_8238, None_8239)

    if may_be_8240:

        if more_types_in_union_8241:
            # Runtime conditional SSA (line 180)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to Exception(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'module_name' (line 181)
        module_name_8243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 24), 'module_name', False)
        # Processing the call keyword arguments (line 181)
        kwargs_8244 = {}
        # Getting the type of 'Exception' (line 181)
        Exception_8242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 14), 'Exception', False)
        # Calling Exception(args, kwargs) (line 181)
        Exception_call_result_8245 = invoke(stypy.reporting.localization.Localization(__file__, 181, 14), Exception_8242, *[module_name_8243], **kwargs_8244)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 181, 8), Exception_call_result_8245, 'raise parameter', BaseException)

        if more_types_in_union_8241:
            # SSA join for if statement (line 180)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'module_file' (line 180)
    module_file_8246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'module_file')
    # Assigning a type to the variable 'module_file' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'module_file', remove_type_from_union(module_file_8246, types.NoneType))
    # Getting the type of 'module_file' (line 182)
    module_file_8247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 11), 'module_file')
    # Assigning a type to the variable 'stypy_return_type' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'stypy_return_type', module_file_8247)
    
    # ################# End of '__get_module_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__get_module_file' in the type store
    # Getting the type of 'stypy_return_type' (line 168)
    stypy_return_type_8248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8248)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__get_module_file'
    return stypy_return_type_8248

# Assigning a type to the variable '__get_module_file' (line 168)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), '__get_module_file', __get_module_file)

@norecursion
def __import_external_non_python_library_module(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'sys' (line 185)
    sys_8249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 87), 'sys')
    # Obtaining the member 'path' of a type (line 185)
    path_8250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 87), sys_8249, 'path')
    defaults = [path_8250]
    # Create a new context for function '__import_external_non_python_library_module'
    module_type_store = module_type_store.open_function_context('__import_external_non_python_library_module', 185, 0, False)
    
    # Passed parameters checking function
    __import_external_non_python_library_module.stypy_localization = localization
    __import_external_non_python_library_module.stypy_type_of_self = None
    __import_external_non_python_library_module.stypy_type_store = module_type_store
    __import_external_non_python_library_module.stypy_function_name = '__import_external_non_python_library_module'
    __import_external_non_python_library_module.stypy_param_names_list = ['localization', 'module_name', 'environment']
    __import_external_non_python_library_module.stypy_varargs_param_name = None
    __import_external_non_python_library_module.stypy_kwargs_param_name = None
    __import_external_non_python_library_module.stypy_call_defaults = defaults
    __import_external_non_python_library_module.stypy_call_varargs = varargs
    __import_external_non_python_library_module.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__import_external_non_python_library_module', ['localization', 'module_name', 'environment'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__import_external_non_python_library_module', localization, ['localization', 'module_name', 'environment'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__import_external_non_python_library_module(...)' code ##################

    str_8251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, (-1)), 'str', '\n    Returns the TypeStore object that represent a non Python library module object\n    :localization Caller information\n    :module_name Name of the module to load\n    :environment (Optional) List of paths to use to search the module (defaults to sys.path)\n    :return: A TypeStore object or a TypeError if the module cannot be loaded\n    ')
    
    
    # SSA begins for try-except statement (line 193)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 194):
    
    # Assigning a Call to a Name (line 194):
    
    # Call to __get_module_file(...): (line 194)
    # Processing the call arguments (line 194)
    # Getting the type of 'module_name' (line 194)
    module_name_8253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 40), 'module_name', False)
    # Processing the call keyword arguments (line 194)
    kwargs_8254 = {}
    # Getting the type of '__get_module_file' (line 194)
    get_module_file_8252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 22), '__get_module_file', False)
    # Calling __get_module_file(args, kwargs) (line 194)
    get_module_file_call_result_8255 = invoke(stypy.reporting.localization.Localization(__file__, 194, 22), get_module_file_8252, *[module_name_8253], **kwargs_8254)
    
    # Assigning a type to the variable 'module_file' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'module_file', get_module_file_call_result_8255)
    
    # Assigning a Call to a Name (line 196):
    
    # Assigning a Call to a Name (line 196):
    
    # Call to get_module_from_sys_cache(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of 'module_file' (line 196)
    module_file_8257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 47), 'module_file', False)
    # Processing the call keyword arguments (line 196)
    kwargs_8258 = {}
    # Getting the type of 'get_module_from_sys_cache' (line 196)
    get_module_from_sys_cache_8256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 21), 'get_module_from_sys_cache', False)
    # Calling get_module_from_sys_cache(args, kwargs) (line 196)
    get_module_from_sys_cache_call_result_8259 = invoke(stypy.reporting.localization.Localization(__file__, 196, 21), get_module_from_sys_cache_8256, *[module_file_8257], **kwargs_8258)
    
    # Assigning a type to the variable 'module_obj' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'module_obj', get_module_from_sys_cache_call_result_8259)
    
    # Type idiom detected: calculating its left and rigth part (line 197)
    # Getting the type of 'module_obj' (line 197)
    module_obj_8260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 11), 'module_obj')
    # Getting the type of 'None' (line 197)
    None_8261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 25), 'None')
    
    (may_be_8262, more_types_in_union_8263) = may_be_none(module_obj_8260, None_8261)

    if may_be_8262:

        if more_types_in_union_8263:
            # Runtime conditional SSA (line 197)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 200):
        
        # Assigning a Call to a Name (line 200):
        
        # Call to __get_non_python_library_module_file(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'module_name' (line 200)
        module_name_8265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 63), 'module_name', False)
        # Getting the type of 'environment' (line 200)
        environment_8266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 76), 'environment', False)
        # Processing the call keyword arguments (line 200)
        kwargs_8267 = {}
        # Getting the type of '__get_non_python_library_module_file' (line 200)
        get_non_python_library_module_file_8264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 26), '__get_non_python_library_module_file', False)
        # Calling __get_non_python_library_module_file(args, kwargs) (line 200)
        get_non_python_library_module_file_call_result_8268 = invoke(stypy.reporting.localization.Localization(__file__, 200, 26), get_non_python_library_module_file_8264, *[module_name_8265, environment_8266], **kwargs_8267)
        
        # Assigning a type to the variable 'source_path' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'source_path', get_non_python_library_module_file_call_result_8268)
        
        # Assigning a Call to a Name (line 201):
        
        # Assigning a Call to a Name (line 201):
        
        # Call to Stypy(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'source_path' (line 201)
        source_path_8271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 47), 'source_path', False)
        # Processing the call keyword arguments (line 201)
        # Getting the type of 'False' (line 201)
        False_8272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 92), 'False', False)
        keyword_8273 = False_8272
        kwargs_8274 = {'generate_type_annotated_program': keyword_8273}
        # Getting the type of 'stypy_main_copy' (line 201)
        stypy_main_copy_8269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 25), 'stypy_main_copy', False)
        # Obtaining the member 'Stypy' of a type (line 201)
        Stypy_8270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 25), stypy_main_copy_8269, 'Stypy')
        # Calling Stypy(args, kwargs) (line 201)
        Stypy_call_result_8275 = invoke(stypy.reporting.localization.Localization(__file__, 201, 25), Stypy_8270, *[source_path_8271], **kwargs_8274)
        
        # Assigning a type to the variable 'module_obj' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'module_obj', Stypy_call_result_8275)
        
        # Call to __put_module_in_sys_cache(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'module_file' (line 203)
        module_file_8277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 38), 'module_file', False)
        # Getting the type of 'module_obj' (line 203)
        module_obj_8278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 51), 'module_obj', False)
        # Processing the call keyword arguments (line 203)
        kwargs_8279 = {}
        # Getting the type of '__put_module_in_sys_cache' (line 203)
        put_module_in_sys_cache_8276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), '__put_module_in_sys_cache', False)
        # Calling __put_module_in_sys_cache(args, kwargs) (line 203)
        put_module_in_sys_cache_call_result_8280 = invoke(stypy.reporting.localization.Localization(__file__, 203, 12), put_module_in_sys_cache_8276, *[module_file_8277, module_obj_8278], **kwargs_8279)
        
        
        # Call to analyze(...): (line 204)
        # Processing the call keyword arguments (line 204)
        kwargs_8283 = {}
        # Getting the type of 'module_obj' (line 204)
        module_obj_8281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'module_obj', False)
        # Obtaining the member 'analyze' of a type (line 204)
        analyze_8282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), module_obj_8281, 'analyze')
        # Calling analyze(args, kwargs) (line 204)
        analyze_call_result_8284 = invoke(stypy.reporting.localization.Localization(__file__, 204, 12), analyze_8282, *[], **kwargs_8283)
        
        
        # Assigning a Call to a Name (line 207):
        
        # Assigning a Call to a Name (line 207):
        
        # Call to get_analyzed_program_type_store(...): (line 207)
        # Processing the call keyword arguments (line 207)
        kwargs_8287 = {}
        # Getting the type of 'module_obj' (line 207)
        module_obj_8285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 32), 'module_obj', False)
        # Obtaining the member 'get_analyzed_program_type_store' of a type (line 207)
        get_analyzed_program_type_store_8286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 32), module_obj_8285, 'get_analyzed_program_type_store')
        # Calling get_analyzed_program_type_store(args, kwargs) (line 207)
        get_analyzed_program_type_store_call_result_8288 = invoke(stypy.reporting.localization.Localization(__file__, 207, 32), get_analyzed_program_type_store_8286, *[], **kwargs_8287)
        
        # Assigning a type to the variable 'module_type_store' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'module_type_store', get_analyzed_program_type_store_call_result_8288)
        
        # Type idiom detected: calculating its left and rigth part (line 208)
        # Getting the type of 'module_type_store' (line 208)
        module_type_store_8289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'module_type_store')
        # Getting the type of 'None' (line 208)
        None_8290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 36), 'None')
        
        (may_be_8291, more_types_in_union_8292) = may_be_none(module_type_store_8289, None_8290)

        if may_be_8291:

            if more_types_in_union_8292:
                # Runtime conditional SSA (line 208)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to TypeError(...): (line 209)
            # Processing the call arguments (line 209)
            # Getting the type of 'localization' (line 209)
            localization_8294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 33), 'localization', False)
            
            # Call to format(...): (line 209)
            # Processing the call arguments (line 209)
            # Getting the type of 'module_name' (line 209)
            module_name_8297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 95), 'module_name', False)
            # Processing the call keyword arguments (line 209)
            kwargs_8298 = {}
            str_8295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 47), 'str', "Could not import external module '{0}'")
            # Obtaining the member 'format' of a type (line 209)
            format_8296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 47), str_8295, 'format')
            # Calling format(args, kwargs) (line 209)
            format_call_result_8299 = invoke(stypy.reporting.localization.Localization(__file__, 209, 47), format_8296, *[module_name_8297], **kwargs_8298)
            
            # Processing the call keyword arguments (line 209)
            kwargs_8300 = {}
            # Getting the type of 'TypeError' (line 209)
            TypeError_8293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 23), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 209)
            TypeError_call_result_8301 = invoke(stypy.reporting.localization.Localization(__file__, 209, 23), TypeError_8293, *[localization_8294, format_call_result_8299], **kwargs_8300)
            
            # Assigning a type to the variable 'stypy_return_type' (line 209)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 16), 'stypy_return_type', TypeError_call_result_8301)

            if more_types_in_union_8292:
                # SSA join for if statement (line 208)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'module_type_store' (line 208)
        module_type_store_8302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'module_type_store')
        # Assigning a type to the variable 'module_type_store' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'module_type_store', remove_type_from_union(module_type_store_8302, types.NoneType))

        if more_types_in_union_8263:
            # SSA join for if statement (line 197)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'module_obj' (line 213)
    module_obj_8303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 15), 'module_obj')
    # Assigning a type to the variable 'stypy_return_type' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'stypy_return_type', module_obj_8303)
    # SSA branch for the except part of a try statement (line 193)
    # SSA branch for the except 'Exception' branch of a try statement (line 193)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 215)
    Exception_8304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 11), 'Exception')
    # Assigning a type to the variable 'exc' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'exc', Exception_8304)
    
    # Call to TypeError(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'localization' (line 219)
    localization_8306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 25), 'localization', False)
    
    # Call to format(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'module_name' (line 219)
    module_name_8309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 92), 'module_name', False)
    
    # Call to str(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'exc' (line 219)
    exc_8311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 109), 'exc', False)
    # Processing the call keyword arguments (line 219)
    kwargs_8312 = {}
    # Getting the type of 'str' (line 219)
    str_8310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 105), 'str', False)
    # Calling str(args, kwargs) (line 219)
    str_call_result_8313 = invoke(stypy.reporting.localization.Localization(__file__, 219, 105), str_8310, *[exc_8311], **kwargs_8312)
    
    # Processing the call keyword arguments (line 219)
    kwargs_8314 = {}
    str_8307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 39), 'str', "Could not import external module '{0}': {1}")
    # Obtaining the member 'format' of a type (line 219)
    format_8308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 39), str_8307, 'format')
    # Calling format(args, kwargs) (line 219)
    format_call_result_8315 = invoke(stypy.reporting.localization.Localization(__file__, 219, 39), format_8308, *[module_name_8309, str_call_result_8313], **kwargs_8314)
    
    # Processing the call keyword arguments (line 219)
    kwargs_8316 = {}
    # Getting the type of 'TypeError' (line 219)
    TypeError_8305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 15), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 219)
    TypeError_call_result_8317 = invoke(stypy.reporting.localization.Localization(__file__, 219, 15), TypeError_8305, *[localization_8306, format_call_result_8315], **kwargs_8316)
    
    # Assigning a type to the variable 'stypy_return_type' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'stypy_return_type', TypeError_call_result_8317)
    # SSA join for try-except statement (line 193)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '__import_external_non_python_library_module(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__import_external_non_python_library_module' in the type store
    # Getting the type of 'stypy_return_type' (line 185)
    stypy_return_type_8318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8318)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__import_external_non_python_library_module'
    return stypy_return_type_8318

# Assigning a type to the variable '__import_external_non_python_library_module' (line 185)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 0), '__import_external_non_python_library_module', __import_external_non_python_library_module)

@norecursion
def import_python_module(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'sys' (line 225)
    sys_8319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 73), 'sys')
    # Obtaining the member 'path' of a type (line 225)
    path_8320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 73), sys_8319, 'path')
    defaults = [path_8320]
    # Create a new context for function 'import_python_module'
    module_type_store = module_type_store.open_function_context('import_python_module', 225, 0, False)
    
    # Passed parameters checking function
    import_python_module.stypy_localization = localization
    import_python_module.stypy_type_of_self = None
    import_python_module.stypy_type_store = module_type_store
    import_python_module.stypy_function_name = 'import_python_module'
    import_python_module.stypy_param_names_list = ['localization', 'imported_module_name', 'environment']
    import_python_module.stypy_varargs_param_name = None
    import_python_module.stypy_kwargs_param_name = None
    import_python_module.stypy_call_defaults = defaults
    import_python_module.stypy_call_varargs = varargs
    import_python_module.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'import_python_module', ['localization', 'imported_module_name', 'environment'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'import_python_module', localization, ['localization', 'imported_module_name', 'environment'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'import_python_module(...)' code ##################

    str_8321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, (-1)), 'str', '\n    This function imports all the declared public members of a user-defined or Python library module into the specified\n    type store\n    It modules the from <module> import <element1>, <element2>, ... or * sentences and also the import <module> sentence\n\n    GENERAL ALGORITHM PSEUDO-CODE:\n\n    This will be divided in two functions. One for importing a module object. The other to process the elements\n    of the module that will be imported once returned. We have three options:\n    - elements = []: (import module) -> The module object will be added to the destination type store\n    - elements = [\'*\']: All the public members of the module will be added to the destionation type store. NOT the module\n    itself\n    - elements = [\'member1\', \'member2\',...]: A particular case of the previous one. We have to check the behavior of Python\n    if someone explicitly try to import __xxx or _xxx members.\n\n    - Python library modules are represented by a TypeInferenceProxy object (subtype of Type)\n    - "User" modules (those whose source code has been processed by stypy) are represented by a TypeStore (subtype of Type)\n\n    Import module function:\n        - Input is a string: Get module object from sys.modules\n        - Once we have the module object:\n        - If it is a Python library module:\n            Check if there is a cache for stypy modules in sys\n            If not, create it\n            Else check if the module it is already cached (using module name)\n                If it is, return the cached module\n            TypeInferenceProxy a module clone (to not to modify the original one)\n            for each member of the module:\n                if the member is another module that is not already cached:\n                    Recursively apply the function, obtaining a module\n                Assign the resulting module to the member value (in the module clone)\n\n        - If it is not a Python library module:\n            Check if there is a cache for stypy modules in sys\n            If not, create it\n            Else check if the module it is already cached (using the module path)\n                If it is, return the cached module\n            Create an Stypy object using the module source path\n            Analyze the module with stypy\n                This will trigger secondary imports when executing the type inference program, as they can contain other\n                imports. So there is no need to recursively call this function or to analyze the module members, as this\n                will be done automatically by calling secondary stypy instances from this one\n            return the Stypy object\n\n    Other considerations:\n\n    Type inference programs will use this line to import external modules:\n\n    import_elements_from_external_module(<localization object>,\n         <module name as it appears in the original code>, type_store)\n\n    This function will:\n        - Obtain the imported module following the previous algorithm\n        - If a TypeInference proxy is obtained, proceed to assign members\n        - If an stypy object is obtained, obtain its type store and proceed to assign members.\n\n\n\n    :param localization: Caller information\n    :param main_module_path: Path of the module to import, i. e. path of the .py file of the module\n    :param imported_module_name: Name of the module\n    :param dest_type_store: Type store to add the module elements\n    :param elements: A variable list of arguments with the elements to import. The value \'*\' means all elements. No\n    value models the "import <module>" sentence\n    :return: None or a TypeError if the requested type do not exist\n    ')
    
    # Call to setrecursionlimit(...): (line 292)
    # Processing the call arguments (line 292)
    int_8324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 26), 'int')
    # Processing the call keyword arguments (line 292)
    kwargs_8325 = {}
    # Getting the type of 'sys' (line 292)
    sys_8322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'sys', False)
    # Obtaining the member 'setrecursionlimit' of a type (line 292)
    setrecursionlimit_8323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 4), sys_8322, 'setrecursionlimit')
    # Calling setrecursionlimit(args, kwargs) (line 292)
    setrecursionlimit_call_result_8326 = invoke(stypy.reporting.localization.Localization(__file__, 292, 4), setrecursionlimit_8323, *[int_8324], **kwargs_8325)
    
    
    
    # Call to is_python_library_module(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'imported_module_name' (line 294)
    imported_module_name_8329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 64), 'imported_module_name', False)
    # Processing the call keyword arguments (line 294)
    kwargs_8330 = {}
    # Getting the type of 'python_library_modules_copy' (line 294)
    python_library_modules_copy_8327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 11), 'python_library_modules_copy', False)
    # Obtaining the member 'is_python_library_module' of a type (line 294)
    is_python_library_module_8328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 11), python_library_modules_copy_8327, 'is_python_library_module')
    # Calling is_python_library_module(args, kwargs) (line 294)
    is_python_library_module_call_result_8331 = invoke(stypy.reporting.localization.Localization(__file__, 294, 11), is_python_library_module_8328, *[imported_module_name_8329], **kwargs_8330)
    
    # Applying the 'not' unary operator (line 294)
    result_not__8332 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 7), 'not', is_python_library_module_call_result_8331)
    
    # Testing if the type of an if condition is none (line 294)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 294, 4), result_not__8332):
        
        # Call to __import_python_library_module(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'localization' (line 300)
        localization_8350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 46), 'localization', False)
        # Getting the type of 'imported_module_name' (line 300)
        imported_module_name_8351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 60), 'imported_module_name', False)
        # Processing the call keyword arguments (line 300)
        kwargs_8352 = {}
        # Getting the type of '__import_python_library_module' (line 300)
        import_python_library_module_8349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 15), '__import_python_library_module', False)
        # Calling __import_python_library_module(args, kwargs) (line 300)
        import_python_library_module_call_result_8353 = invoke(stypy.reporting.localization.Localization(__file__, 300, 15), import_python_library_module_8349, *[localization_8350, imported_module_name_8351], **kwargs_8352)
        
        # Assigning a type to the variable 'stypy_return_type' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'stypy_return_type', import_python_library_module_call_result_8353)
    else:
        
        # Testing the type of an if condition (line 294)
        if_condition_8333 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 4), result_not__8332)
        # Assigning a type to the variable 'if_condition_8333' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'if_condition_8333', if_condition_8333)
        # SSA begins for if statement (line 294)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Call to __import_external_non_python_library_module(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'localization' (line 295)
        localization_8335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 64), 'localization', False)
        # Getting the type of 'imported_module_name' (line 295)
        imported_module_name_8336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 78), 'imported_module_name', False)
        # Getting the type of 'environment' (line 295)
        environment_8337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 100), 'environment', False)
        # Processing the call keyword arguments (line 295)
        kwargs_8338 = {}
        # Getting the type of '__import_external_non_python_library_module' (line 295)
        import_external_non_python_library_module_8334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 20), '__import_external_non_python_library_module', False)
        # Calling __import_external_non_python_library_module(args, kwargs) (line 295)
        import_external_non_python_library_module_call_result_8339 = invoke(stypy.reporting.localization.Localization(__file__, 295, 20), import_external_non_python_library_module_8334, *[localization_8335, imported_module_name_8336, environment_8337], **kwargs_8338)
        
        # Assigning a type to the variable 'stypy_obj' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'stypy_obj', import_external_non_python_library_module_call_result_8339)
        
        # Type idiom detected: calculating its left and rigth part (line 296)
        # Getting the type of 'TypeError' (line 296)
        TypeError_8340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 33), 'TypeError')
        # Getting the type of 'stypy_obj' (line 296)
        stypy_obj_8341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 22), 'stypy_obj')
        
        (may_be_8342, more_types_in_union_8343) = may_be_subtype(TypeError_8340, stypy_obj_8341)

        if may_be_8342:

            if more_types_in_union_8343:
                # Runtime conditional SSA (line 296)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_obj' (line 296)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'stypy_obj', remove_not_subtype_from_union(stypy_obj_8341, TypeError))
            # Getting the type of 'stypy_obj' (line 297)
            stypy_obj_8344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 19), 'stypy_obj')
            # Assigning a type to the variable 'stypy_return_type' (line 297)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'stypy_return_type', stypy_obj_8344)

            if more_types_in_union_8343:
                # SSA join for if statement (line 296)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to get_analyzed_program_type_store(...): (line 298)
        # Processing the call keyword arguments (line 298)
        kwargs_8347 = {}
        # Getting the type of 'stypy_obj' (line 298)
        stypy_obj_8345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 15), 'stypy_obj', False)
        # Obtaining the member 'get_analyzed_program_type_store' of a type (line 298)
        get_analyzed_program_type_store_8346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 15), stypy_obj_8345, 'get_analyzed_program_type_store')
        # Calling get_analyzed_program_type_store(args, kwargs) (line 298)
        get_analyzed_program_type_store_call_result_8348 = invoke(stypy.reporting.localization.Localization(__file__, 298, 15), get_analyzed_program_type_store_8346, *[], **kwargs_8347)
        
        # Assigning a type to the variable 'stypy_return_type' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'stypy_return_type', get_analyzed_program_type_store_call_result_8348)
        # SSA branch for the else part of an if statement (line 294)
        module_type_store.open_ssa_branch('else')
        
        # Call to __import_python_library_module(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'localization' (line 300)
        localization_8350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 46), 'localization', False)
        # Getting the type of 'imported_module_name' (line 300)
        imported_module_name_8351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 60), 'imported_module_name', False)
        # Processing the call keyword arguments (line 300)
        kwargs_8352 = {}
        # Getting the type of '__import_python_library_module' (line 300)
        import_python_library_module_8349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 15), '__import_python_library_module', False)
        # Calling __import_python_library_module(args, kwargs) (line 300)
        import_python_library_module_call_result_8353 = invoke(stypy.reporting.localization.Localization(__file__, 300, 15), import_python_library_module_8349, *[localization_8350, imported_module_name_8351], **kwargs_8352)
        
        # Assigning a type to the variable 'stypy_return_type' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'stypy_return_type', import_python_library_module_call_result_8353)
        # SSA join for if statement (line 294)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'import_python_module(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'import_python_module' in the type store
    # Getting the type of 'stypy_return_type' (line 225)
    stypy_return_type_8354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8354)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'import_python_module'
    return stypy_return_type_8354

# Assigning a type to the variable 'import_python_module' (line 225)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'import_python_module', import_python_module)

@norecursion
def __get_public_names_and_types_of_module(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__get_public_names_and_types_of_module'
    module_type_store = module_type_store.open_function_context('__get_public_names_and_types_of_module', 303, 0, False)
    
    # Passed parameters checking function
    __get_public_names_and_types_of_module.stypy_localization = localization
    __get_public_names_and_types_of_module.stypy_type_of_self = None
    __get_public_names_and_types_of_module.stypy_type_store = module_type_store
    __get_public_names_and_types_of_module.stypy_function_name = '__get_public_names_and_types_of_module'
    __get_public_names_and_types_of_module.stypy_param_names_list = ['module_obj']
    __get_public_names_and_types_of_module.stypy_varargs_param_name = None
    __get_public_names_and_types_of_module.stypy_kwargs_param_name = None
    __get_public_names_and_types_of_module.stypy_call_defaults = defaults
    __get_public_names_and_types_of_module.stypy_call_varargs = varargs
    __get_public_names_and_types_of_module.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__get_public_names_and_types_of_module', ['module_obj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__get_public_names_and_types_of_module', localization, ['module_obj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__get_public_names_and_types_of_module(...)' code ##################

    str_8355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, (-1)), 'str', '\n    Get the public (importable) elements of a module\n    :param module_obj: Module object (either a TypeInferenceProxy or a TypeStore)\n    :return: list of str\n    ')
    
    # Call to isinstance(...): (line 309)
    # Processing the call arguments (line 309)
    # Getting the type of 'module_obj' (line 309)
    module_obj_8357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 18), 'module_obj', False)
    # Getting the type of 'type_inference_proxy_copy' (line 309)
    type_inference_proxy_copy_8358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 30), 'type_inference_proxy_copy', False)
    # Obtaining the member 'TypeInferenceProxy' of a type (line 309)
    TypeInferenceProxy_8359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 30), type_inference_proxy_copy_8358, 'TypeInferenceProxy')
    # Processing the call keyword arguments (line 309)
    kwargs_8360 = {}
    # Getting the type of 'isinstance' (line 309)
    isinstance_8356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 309)
    isinstance_call_result_8361 = invoke(stypy.reporting.localization.Localization(__file__, 309, 7), isinstance_8356, *[module_obj_8357, TypeInferenceProxy_8359], **kwargs_8360)
    
    # Testing if the type of an if condition is none (line 309)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 309, 4), isinstance_call_result_8361):
        
        # Call to get_public_names_and_types(...): (line 312)
        # Processing the call keyword arguments (line 312)
        kwargs_8383 = {}
        # Getting the type of 'module_obj' (line 312)
        module_obj_8381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 15), 'module_obj', False)
        # Obtaining the member 'get_public_names_and_types' of a type (line 312)
        get_public_names_and_types_8382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 15), module_obj_8381, 'get_public_names_and_types')
        # Calling get_public_names_and_types(args, kwargs) (line 312)
        get_public_names_and_types_call_result_8384 = invoke(stypy.reporting.localization.Localization(__file__, 312, 15), get_public_names_and_types_8382, *[], **kwargs_8383)
        
        # Assigning a type to the variable 'stypy_return_type' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'stypy_return_type', get_public_names_and_types_call_result_8384)
    else:
        
        # Testing the type of an if condition (line 309)
        if_condition_8362 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 309, 4), isinstance_call_result_8361)
        # Assigning a type to the variable 'if_condition_8362' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'if_condition_8362', if_condition_8362)
        # SSA begins for if statement (line 309)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to filter(...): (line 310)
        # Processing the call arguments (line 310)

        @norecursion
        def _stypy_temp_lambda_16(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_16'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_16', 310, 22, True)
            # Passed parameters checking function
            _stypy_temp_lambda_16.stypy_localization = localization
            _stypy_temp_lambda_16.stypy_type_of_self = None
            _stypy_temp_lambda_16.stypy_type_store = module_type_store
            _stypy_temp_lambda_16.stypy_function_name = '_stypy_temp_lambda_16'
            _stypy_temp_lambda_16.stypy_param_names_list = ['name']
            _stypy_temp_lambda_16.stypy_varargs_param_name = None
            _stypy_temp_lambda_16.stypy_kwargs_param_name = None
            _stypy_temp_lambda_16.stypy_call_defaults = defaults
            _stypy_temp_lambda_16.stypy_call_varargs = varargs
            _stypy_temp_lambda_16.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_16', ['name'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_16', ['name'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            
            # Call to startswith(...): (line 310)
            # Processing the call arguments (line 310)
            str_8366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 55), 'str', '__')
            # Processing the call keyword arguments (line 310)
            kwargs_8367 = {}
            # Getting the type of 'name' (line 310)
            name_8364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 39), 'name', False)
            # Obtaining the member 'startswith' of a type (line 310)
            startswith_8365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 39), name_8364, 'startswith')
            # Calling startswith(args, kwargs) (line 310)
            startswith_call_result_8368 = invoke(stypy.reporting.localization.Localization(__file__, 310, 39), startswith_8365, *[str_8366], **kwargs_8367)
            
            # Applying the 'not' unary operator (line 310)
            result_not__8369 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 35), 'not', startswith_call_result_8368)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 310)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 22), 'stypy_return_type', result_not__8369)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_16' in the type store
            # Getting the type of 'stypy_return_type' (line 310)
            stypy_return_type_8370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 22), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_8370)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_16'
            return stypy_return_type_8370

        # Assigning a type to the variable '_stypy_temp_lambda_16' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 22), '_stypy_temp_lambda_16', _stypy_temp_lambda_16)
        # Getting the type of '_stypy_temp_lambda_16' (line 310)
        _stypy_temp_lambda_16_8371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 22), '_stypy_temp_lambda_16')
        
        # Call to dir(...): (line 310)
        # Processing the call arguments (line 310)
        
        # Call to get_python_entity(...): (line 310)
        # Processing the call keyword arguments (line 310)
        kwargs_8375 = {}
        # Getting the type of 'module_obj' (line 310)
        module_obj_8373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 66), 'module_obj', False)
        # Obtaining the member 'get_python_entity' of a type (line 310)
        get_python_entity_8374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 66), module_obj_8373, 'get_python_entity')
        # Calling get_python_entity(args, kwargs) (line 310)
        get_python_entity_call_result_8376 = invoke(stypy.reporting.localization.Localization(__file__, 310, 66), get_python_entity_8374, *[], **kwargs_8375)
        
        # Processing the call keyword arguments (line 310)
        kwargs_8377 = {}
        # Getting the type of 'dir' (line 310)
        dir_8372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 62), 'dir', False)
        # Calling dir(args, kwargs) (line 310)
        dir_call_result_8378 = invoke(stypy.reporting.localization.Localization(__file__, 310, 62), dir_8372, *[get_python_entity_call_result_8376], **kwargs_8377)
        
        # Processing the call keyword arguments (line 310)
        kwargs_8379 = {}
        # Getting the type of 'filter' (line 310)
        filter_8363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 15), 'filter', False)
        # Calling filter(args, kwargs) (line 310)
        filter_call_result_8380 = invoke(stypy.reporting.localization.Localization(__file__, 310, 15), filter_8363, *[_stypy_temp_lambda_16_8371, dir_call_result_8378], **kwargs_8379)
        
        # Assigning a type to the variable 'stypy_return_type' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'stypy_return_type', filter_call_result_8380)
        # SSA branch for the else part of an if statement (line 309)
        module_type_store.open_ssa_branch('else')
        
        # Call to get_public_names_and_types(...): (line 312)
        # Processing the call keyword arguments (line 312)
        kwargs_8383 = {}
        # Getting the type of 'module_obj' (line 312)
        module_obj_8381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 15), 'module_obj', False)
        # Obtaining the member 'get_public_names_and_types' of a type (line 312)
        get_public_names_and_types_8382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 15), module_obj_8381, 'get_public_names_and_types')
        # Calling get_public_names_and_types(args, kwargs) (line 312)
        get_public_names_and_types_call_result_8384 = invoke(stypy.reporting.localization.Localization(__file__, 312, 15), get_public_names_and_types_8382, *[], **kwargs_8383)
        
        # Assigning a type to the variable 'stypy_return_type' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'stypy_return_type', get_public_names_and_types_call_result_8384)
        # SSA join for if statement (line 309)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of '__get_public_names_and_types_of_module(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__get_public_names_and_types_of_module' in the type store
    # Getting the type of 'stypy_return_type' (line 303)
    stypy_return_type_8385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8385)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__get_public_names_and_types_of_module'
    return stypy_return_type_8385

# Assigning a type to the variable '__get_public_names_and_types_of_module' (line 303)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 0), '__get_public_names_and_types_of_module', __get_public_names_and_types_of_module)

@norecursion
def __import_module_element(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__import_module_element'
    module_type_store = module_type_store.open_function_context('__import_module_element', 315, 0, False)
    
    # Passed parameters checking function
    __import_module_element.stypy_localization = localization
    __import_module_element.stypy_type_of_self = None
    __import_module_element.stypy_type_store = module_type_store
    __import_module_element.stypy_function_name = '__import_module_element'
    __import_module_element.stypy_param_names_list = ['localization', 'imported_module_name', 'module_obj', 'element', 'dest_type_store', 'environment']
    __import_module_element.stypy_varargs_param_name = None
    __import_module_element.stypy_kwargs_param_name = None
    __import_module_element.stypy_call_defaults = defaults
    __import_module_element.stypy_call_varargs = varargs
    __import_module_element.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__import_module_element', ['localization', 'imported_module_name', 'module_obj', 'element', 'dest_type_store', 'environment'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__import_module_element', localization, ['localization', 'imported_module_name', 'module_obj', 'element', 'dest_type_store', 'environment'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__import_module_element(...)' code ##################

    
    # Assigning a Call to a Name (line 317):
    
    # Assigning a Call to a Name (line 317):
    
    # Call to get_type_of_member(...): (line 317)
    # Processing the call arguments (line 317)
    # Getting the type of 'localization' (line 317)
    localization_8388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 48), 'localization', False)
    # Getting the type of 'element' (line 317)
    element_8389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 62), 'element', False)
    # Processing the call keyword arguments (line 317)
    kwargs_8390 = {}
    # Getting the type of 'module_obj' (line 317)
    module_obj_8386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 18), 'module_obj', False)
    # Obtaining the member 'get_type_of_member' of a type (line 317)
    get_type_of_member_8387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 18), module_obj_8386, 'get_type_of_member')
    # Calling get_type_of_member(args, kwargs) (line 317)
    get_type_of_member_call_result_8391 = invoke(stypy.reporting.localization.Localization(__file__, 317, 18), get_type_of_member_8387, *[localization_8388, element_8389], **kwargs_8390)
    
    # Assigning a type to the variable 'member_type' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'member_type', get_type_of_member_call_result_8391)
    
    # Type idiom detected: calculating its left and rigth part (line 318)
    # Getting the type of 'TypeError' (line 318)
    TypeError_8392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 31), 'TypeError')
    # Getting the type of 'member_type' (line 318)
    member_type_8393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 18), 'member_type')
    
    (may_be_8394, more_types_in_union_8395) = may_be_subtype(TypeError_8392, member_type_8393)

    if may_be_8394:

        if more_types_in_union_8395:
            # Runtime conditional SSA (line 318)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'member_type' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'member_type', remove_not_subtype_from_union(member_type_8393, TypeError))
        
        # Assigning a Call to a Name (line 319):
        
        # Assigning a Call to a Name (line 319):
        
        # Call to __get_non_python_library_module_file(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'element' (line 319)
        element_8397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 59), 'element', False)
        # Processing the call keyword arguments (line 319)
        kwargs_8398 = {}
        # Getting the type of '__get_non_python_library_module_file' (line 319)
        get_non_python_library_module_file_8396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 22), '__get_non_python_library_module_file', False)
        # Calling __get_non_python_library_module_file(args, kwargs) (line 319)
        get_non_python_library_module_file_call_result_8399 = invoke(stypy.reporting.localization.Localization(__file__, 319, 22), get_non_python_library_module_file_8396, *[element_8397], **kwargs_8398)
        
        # Assigning a type to the variable 'module_file' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'module_file', get_non_python_library_module_file_call_result_8399)
        
        # Type idiom detected: calculating its left and rigth part (line 320)
        # Getting the type of 'module_file' (line 320)
        module_file_8400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 11), 'module_file')
        # Getting the type of 'None' (line 320)
        None_8401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 26), 'None')
        
        (may_be_8402, more_types_in_union_8403) = may_be_none(module_file_8400, None_8401)

        if may_be_8402:

            if more_types_in_union_8403:
                # Runtime conditional SSA (line 320)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'member_type' (line 321)
            member_type_8404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 19), 'member_type')
            # Assigning a type to the variable 'stypy_return_type' (line 321)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'stypy_return_type', member_type_8404)

            if more_types_in_union_8403:
                # SSA join for if statement (line 320)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'module_file' (line 320)
        module_file_8405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'module_file')
        # Assigning a type to the variable 'module_file' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'module_file', remove_type_from_union(module_file_8405, types.NoneType))
        
        # Assigning a Call to a Name (line 323):
        
        # Assigning a Call to a Name (line 323):
        
        # Call to dirname(...): (line 323)
        # Processing the call arguments (line 323)
        # Getting the type of 'module_file' (line 323)
        module_file_8409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 37), 'module_file', False)
        # Processing the call keyword arguments (line 323)
        kwargs_8410 = {}
        # Getting the type of 'os' (line 323)
        os_8406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 21), 'os', False)
        # Obtaining the member 'path' of a type (line 323)
        path_8407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 21), os_8406, 'path')
        # Obtaining the member 'dirname' of a type (line 323)
        dirname_8408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 21), path_8407, 'dirname')
        # Calling dirname(args, kwargs) (line 323)
        dirname_call_result_8411 = invoke(stypy.reporting.localization.Localization(__file__, 323, 21), dirname_8408, *[module_file_8409], **kwargs_8410)
        
        # Assigning a type to the variable 'module_dir' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'module_dir', dirname_call_result_8411)
        
        # Call to isfile(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'module_file' (line 327)
        module_file_8415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 26), 'module_file', False)
        # Processing the call keyword arguments (line 327)
        kwargs_8416 = {}
        # Getting the type of 'os' (line 327)
        os_8412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 11), 'os', False)
        # Obtaining the member 'path' of a type (line 327)
        path_8413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 11), os_8412, 'path')
        # Obtaining the member 'isfile' of a type (line 327)
        isfile_8414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 11), path_8413, 'isfile')
        # Calling isfile(args, kwargs) (line 327)
        isfile_call_result_8417 = invoke(stypy.reporting.localization.Localization(__file__, 327, 11), isfile_8414, *[module_file_8415], **kwargs_8416)
        
        # Testing if the type of an if condition is none (line 327)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 327, 8), isfile_call_result_8417):
            
            # Call to set_type_of(...): (line 334)
            # Processing the call arguments (line 334)
            # Getting the type of 'localization' (line 334)
            localization_8438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 40), 'localization', False)
            # Getting the type of 'element' (line 334)
            element_8439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 54), 'element', False)
            # Getting the type of 'member_type' (line 334)
            member_type_8440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 63), 'member_type', False)
            # Processing the call keyword arguments (line 334)
            kwargs_8441 = {}
            # Getting the type of 'dest_type_store' (line 334)
            dest_type_store_8436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'dest_type_store', False)
            # Obtaining the member 'set_type_of' of a type (line 334)
            set_type_of_8437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 12), dest_type_store_8436, 'set_type_of')
            # Calling set_type_of(args, kwargs) (line 334)
            set_type_of_call_result_8442 = invoke(stypy.reporting.localization.Localization(__file__, 334, 12), set_type_of_8437, *[localization_8438, element_8439, member_type_8440], **kwargs_8441)
            
        else:
            
            # Testing the type of an if condition (line 327)
            if_condition_8418 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 327, 8), isfile_call_result_8417)
            # Assigning a type to the variable 'if_condition_8418' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'if_condition_8418', if_condition_8418)
            # SSA begins for if statement (line 327)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 328):
            
            # Assigning a BinOp to a Name (line 328):
            
            # Obtaining an instance of the builtin type 'list' (line 328)
            list_8419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 37), 'list')
            # Adding type elements to the builtin type 'list' instance (line 328)
            # Adding element type (line 328)
            # Getting the type of 'module_dir' (line 328)
            module_dir_8420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 38), 'module_dir')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 37), list_8419, module_dir_8420)
            
            # Getting the type of 'environment' (line 328)
            environment_8421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 52), 'environment')
            # Applying the binary operator '+' (line 328)
            result_add_8422 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 37), '+', list_8419, environment_8421)
            
            # Assigning a type to the variable 'restricted_environment' (line 328)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'restricted_environment', result_add_8422)
            
            # Call to import_elements_from_external_module(...): (line 329)
            # Processing the call arguments (line 329)
            # Getting the type of 'localization' (line 329)
            localization_8424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 49), 'localization', False)
            # Getting the type of 'element' (line 329)
            element_8425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 63), 'element', False)
            # Getting the type of 'dest_type_store' (line 329)
            dest_type_store_8426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 72), 'dest_type_store', False)
            # Getting the type of 'restricted_environment' (line 330)
            restricted_environment_8427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 49), 'restricted_environment', False)
            
            # Obtaining an instance of the builtin type 'list' (line 331)
            list_8428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 50), 'list')
            # Adding type elements to the builtin type 'list' instance (line 331)
            
            # Processing the call keyword arguments (line 329)
            kwargs_8429 = {}
            # Getting the type of 'import_elements_from_external_module' (line 329)
            import_elements_from_external_module_8423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'import_elements_from_external_module', False)
            # Calling import_elements_from_external_module(args, kwargs) (line 329)
            import_elements_from_external_module_call_result_8430 = invoke(stypy.reporting.localization.Localization(__file__, 329, 12), import_elements_from_external_module_8423, *[localization_8424, element_8425, dest_type_store_8426, restricted_environment_8427, list_8428], **kwargs_8429)
            
            
            # Call to remove_error_msg(...): (line 332)
            # Processing the call arguments (line 332)
            # Getting the type of 'member_type' (line 332)
            member_type_8433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 39), 'member_type', False)
            # Processing the call keyword arguments (line 332)
            kwargs_8434 = {}
            # Getting the type of 'TypeError' (line 332)
            TypeError_8431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'TypeError', False)
            # Obtaining the member 'remove_error_msg' of a type (line 332)
            remove_error_msg_8432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 12), TypeError_8431, 'remove_error_msg')
            # Calling remove_error_msg(args, kwargs) (line 332)
            remove_error_msg_call_result_8435 = invoke(stypy.reporting.localization.Localization(__file__, 332, 12), remove_error_msg_8432, *[member_type_8433], **kwargs_8434)
            
            # SSA branch for the else part of an if statement (line 327)
            module_type_store.open_ssa_branch('else')
            
            # Call to set_type_of(...): (line 334)
            # Processing the call arguments (line 334)
            # Getting the type of 'localization' (line 334)
            localization_8438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 40), 'localization', False)
            # Getting the type of 'element' (line 334)
            element_8439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 54), 'element', False)
            # Getting the type of 'member_type' (line 334)
            member_type_8440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 63), 'member_type', False)
            # Processing the call keyword arguments (line 334)
            kwargs_8441 = {}
            # Getting the type of 'dest_type_store' (line 334)
            dest_type_store_8436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'dest_type_store', False)
            # Obtaining the member 'set_type_of' of a type (line 334)
            set_type_of_8437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 12), dest_type_store_8436, 'set_type_of')
            # Calling set_type_of(args, kwargs) (line 334)
            set_type_of_call_result_8442 = invoke(stypy.reporting.localization.Localization(__file__, 334, 12), set_type_of_8437, *[localization_8438, element_8439, member_type_8440], **kwargs_8441)
            
            # SSA join for if statement (line 327)
            module_type_store = module_type_store.join_ssa_context()
            


        if more_types_in_union_8395:
            # Runtime conditional SSA for else branch (line 318)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_8394) or more_types_in_union_8395):
        # Assigning a type to the variable 'member_type' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'member_type', remove_subtype_from_union(member_type_8393, TypeError))
        
        # Call to set_type_of(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'localization' (line 337)
        localization_8445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 36), 'localization', False)
        # Getting the type of 'element' (line 337)
        element_8446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 50), 'element', False)
        # Getting the type of 'member_type' (line 337)
        member_type_8447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 59), 'member_type', False)
        # Processing the call keyword arguments (line 337)
        kwargs_8448 = {}
        # Getting the type of 'dest_type_store' (line 337)
        dest_type_store_8443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'dest_type_store', False)
        # Obtaining the member 'set_type_of' of a type (line 337)
        set_type_of_8444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 8), dest_type_store_8443, 'set_type_of')
        # Calling set_type_of(args, kwargs) (line 337)
        set_type_of_call_result_8449 = invoke(stypy.reporting.localization.Localization(__file__, 337, 8), set_type_of_8444, *[localization_8445, element_8446, member_type_8447], **kwargs_8448)
        

        if (may_be_8394 and more_types_in_union_8395):
            # SSA join for if statement (line 318)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of '__import_module_element(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__import_module_element' in the type store
    # Getting the type of 'stypy_return_type' (line 315)
    stypy_return_type_8450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8450)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__import_module_element'
    return stypy_return_type_8450

# Assigning a type to the variable '__import_module_element' (line 315)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 0), '__import_module_element', __import_module_element)

@norecursion
def import_elements_from_external_module(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'import_elements_from_external_module'
    module_type_store = module_type_store.open_function_context('import_elements_from_external_module', 340, 0, False)
    
    # Passed parameters checking function
    import_elements_from_external_module.stypy_localization = localization
    import_elements_from_external_module.stypy_type_of_self = None
    import_elements_from_external_module.stypy_type_store = module_type_store
    import_elements_from_external_module.stypy_function_name = 'import_elements_from_external_module'
    import_elements_from_external_module.stypy_param_names_list = ['localization', 'imported_module_name', 'dest_type_store', 'environment']
    import_elements_from_external_module.stypy_varargs_param_name = 'elements'
    import_elements_from_external_module.stypy_kwargs_param_name = None
    import_elements_from_external_module.stypy_call_defaults = defaults
    import_elements_from_external_module.stypy_call_varargs = varargs
    import_elements_from_external_module.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'import_elements_from_external_module', ['localization', 'imported_module_name', 'dest_type_store', 'environment'], 'elements', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'import_elements_from_external_module', localization, ['localization', 'imported_module_name', 'dest_type_store', 'environment'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'import_elements_from_external_module(...)' code ##################

    str_8451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, (-1)), 'str', "\n    Imports the listed elements from the provided module name in the dest_type_store TypeStore, using the provided\n    environment as a module search path\n\n    :param localization: Caller information\n    :param imported_module_name: Name of the module to import\n    :param dest_type_store: Type store to store the imported elements in\n    :param environment: List of paths for module seach\n    :param elements: Elements of the module to import ([] for import the whole module, ['*'] for 'from module import *'\n    statements and a list of names for importing concrete module members.\n    :return: None\n    ")
    
    # Call to setrecursionlimit(...): (line 354)
    # Processing the call arguments (line 354)
    int_8454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 26), 'int')
    # Processing the call keyword arguments (line 354)
    kwargs_8455 = {}
    # Getting the type of 'sys' (line 354)
    sys_8452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'sys', False)
    # Obtaining the member 'setrecursionlimit' of a type (line 354)
    setrecursionlimit_8453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 4), sys_8452, 'setrecursionlimit')
    # Calling setrecursionlimit(args, kwargs) (line 354)
    setrecursionlimit_call_result_8456 = invoke(stypy.reporting.localization.Localization(__file__, 354, 4), setrecursionlimit_8453, *[int_8454], **kwargs_8455)
    
    
    
    # Call to is_python_library_module(...): (line 356)
    # Processing the call arguments (line 356)
    # Getting the type of 'imported_module_name' (line 356)
    imported_module_name_8459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 64), 'imported_module_name', False)
    # Processing the call keyword arguments (line 356)
    kwargs_8460 = {}
    # Getting the type of 'python_library_modules_copy' (line 356)
    python_library_modules_copy_8457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 11), 'python_library_modules_copy', False)
    # Obtaining the member 'is_python_library_module' of a type (line 356)
    is_python_library_module_8458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 11), python_library_modules_copy_8457, 'is_python_library_module')
    # Calling is_python_library_module(args, kwargs) (line 356)
    is_python_library_module_call_result_8461 = invoke(stypy.reporting.localization.Localization(__file__, 356, 11), is_python_library_module_8458, *[imported_module_name_8459], **kwargs_8460)
    
    # Applying the 'not' unary operator (line 356)
    result_not__8462 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 7), 'not', is_python_library_module_call_result_8461)
    
    # Testing if the type of an if condition is none (line 356)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 356, 4), result_not__8462):
        pass
    else:
        
        # Testing the type of an if condition (line 356)
        if_condition_8463 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 356, 4), result_not__8462)
        # Assigning a type to the variable 'if_condition_8463' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'if_condition_8463', if_condition_8463)
        # SSA begins for if statement (line 356)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 358):
        
        # Assigning a Call to a Name (line 358):
        
        # Call to dirname(...): (line 358)
        # Processing the call arguments (line 358)
        # Getting the type of 'dest_type_store' (line 358)
        dest_type_store_8467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 50), 'dest_type_store', False)
        # Obtaining the member 'program_name' of a type (line 358)
        program_name_8468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 50), dest_type_store_8467, 'program_name')
        # Processing the call keyword arguments (line 358)
        kwargs_8469 = {}
        # Getting the type of 'os' (line 358)
        os_8464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 358)
        path_8465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 34), os_8464, 'path')
        # Obtaining the member 'dirname' of a type (line 358)
        dirname_8466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 34), path_8465, 'dirname')
        # Calling dirname(args, kwargs) (line 358)
        dirname_call_result_8470 = invoke(stypy.reporting.localization.Localization(__file__, 358, 34), dirname_8466, *[program_name_8468], **kwargs_8469)
        
        # Assigning a type to the variable 'destination_module_path' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'destination_module_path', dirname_call_result_8470)
        
        # Assigning a Name to a Name (line 359):
        
        # Assigning a Name to a Name (line 359):
        # Getting the type of 'False' (line 359)
        False_8471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 33), 'False')
        # Assigning a type to the variable 'destination_path_added' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'destination_path_added', False_8471)
        
        
        # Getting the type of 'destination_module_path' (line 360)
        destination_module_path_8472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 15), 'destination_module_path')
        # Getting the type of 'environment' (line 360)
        environment_8473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 42), 'environment')
        # Applying the binary operator 'in' (line 360)
        result_contains_8474 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 15), 'in', destination_module_path_8472, environment_8473)
        
        # Applying the 'not' unary operator (line 360)
        result_not__8475 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 11), 'not', result_contains_8474)
        
        # Testing if the type of an if condition is none (line 360)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 360, 8), result_not__8475):
            pass
        else:
            
            # Testing the type of an if condition (line 360)
            if_condition_8476 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 360, 8), result_not__8475)
            # Assigning a type to the variable 'if_condition_8476' (line 360)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'if_condition_8476', if_condition_8476)
            # SSA begins for if statement (line 360)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 361):
            
            # Assigning a Name to a Name (line 361):
            # Getting the type of 'True' (line 361)
            True_8477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 37), 'True')
            # Assigning a type to the variable 'destination_path_added' (line 361)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'destination_path_added', True_8477)
            
            # Call to append(...): (line 362)
            # Processing the call arguments (line 362)
            # Getting the type of 'destination_module_path' (line 362)
            destination_module_path_8480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 31), 'destination_module_path', False)
            # Processing the call keyword arguments (line 362)
            kwargs_8481 = {}
            # Getting the type of 'environment' (line 362)
            environment_8478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'environment', False)
            # Obtaining the member 'append' of a type (line 362)
            append_8479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 12), environment_8478, 'append')
            # Calling append(args, kwargs) (line 362)
            append_call_result_8482 = invoke(stypy.reporting.localization.Localization(__file__, 362, 12), append_8479, *[destination_module_path_8480], **kwargs_8481)
            
            # SSA join for if statement (line 360)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 356)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 364):
    
    # Assigning a Call to a Name (line 364):
    
    # Call to import_python_module(...): (line 364)
    # Processing the call arguments (line 364)
    # Getting the type of 'localization' (line 364)
    localization_8484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 38), 'localization', False)
    # Getting the type of 'imported_module_name' (line 364)
    imported_module_name_8485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 52), 'imported_module_name', False)
    # Getting the type of 'environment' (line 364)
    environment_8486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 74), 'environment', False)
    # Processing the call keyword arguments (line 364)
    kwargs_8487 = {}
    # Getting the type of 'import_python_module' (line 364)
    import_python_module_8483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 17), 'import_python_module', False)
    # Calling import_python_module(args, kwargs) (line 364)
    import_python_module_call_result_8488 = invoke(stypy.reporting.localization.Localization(__file__, 364, 17), import_python_module_8483, *[localization_8484, imported_module_name_8485, environment_8486], **kwargs_8487)
    
    # Assigning a type to the variable 'module_obj' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'module_obj', import_python_module_call_result_8488)
    
    
    # Call to is_python_library_module(...): (line 366)
    # Processing the call arguments (line 366)
    # Getting the type of 'imported_module_name' (line 366)
    imported_module_name_8491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 64), 'imported_module_name', False)
    # Processing the call keyword arguments (line 366)
    kwargs_8492 = {}
    # Getting the type of 'python_library_modules_copy' (line 366)
    python_library_modules_copy_8489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 11), 'python_library_modules_copy', False)
    # Obtaining the member 'is_python_library_module' of a type (line 366)
    is_python_library_module_8490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 11), python_library_modules_copy_8489, 'is_python_library_module')
    # Calling is_python_library_module(args, kwargs) (line 366)
    is_python_library_module_call_result_8493 = invoke(stypy.reporting.localization.Localization(__file__, 366, 11), is_python_library_module_8490, *[imported_module_name_8491], **kwargs_8492)
    
    # Applying the 'not' unary operator (line 366)
    result_not__8494 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 7), 'not', is_python_library_module_call_result_8493)
    
    # Testing if the type of an if condition is none (line 366)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 366, 4), result_not__8494):
        pass
    else:
        
        # Testing the type of an if condition (line 366)
        if_condition_8495 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 366, 4), result_not__8494)
        # Assigning a type to the variable 'if_condition_8495' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'if_condition_8495', if_condition_8495)
        # SSA begins for if statement (line 366)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 368):
        
        # Assigning a Call to a Name (line 368):
        
        # Call to __get_module_file(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of 'imported_module_name' (line 368)
        imported_module_name_8497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 49), 'imported_module_name', False)
        # Processing the call keyword arguments (line 368)
        kwargs_8498 = {}
        # Getting the type of '__get_module_file' (line 368)
        get_module_file_8496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 31), '__get_module_file', False)
        # Calling __get_module_file(args, kwargs) (line 368)
        get_module_file_call_result_8499 = invoke(stypy.reporting.localization.Localization(__file__, 368, 31), get_module_file_8496, *[imported_module_name_8497], **kwargs_8498)
        
        # Assigning a type to the variable 'imported_module_file' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'imported_module_file', get_module_file_call_result_8499)
        
        # Assigning a Call to a Name (line 369):
        
        # Assigning a Call to a Name (line 369):
        
        # Call to dirname(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 'imported_module_file' (line 369)
        imported_module_file_8503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 47), 'imported_module_file', False)
        # Processing the call keyword arguments (line 369)
        kwargs_8504 = {}
        # Getting the type of 'os' (line 369)
        os_8500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 31), 'os', False)
        # Obtaining the member 'path' of a type (line 369)
        path_8501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 31), os_8500, 'path')
        # Obtaining the member 'dirname' of a type (line 369)
        dirname_8502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 31), path_8501, 'dirname')
        # Calling dirname(args, kwargs) (line 369)
        dirname_call_result_8505 = invoke(stypy.reporting.localization.Localization(__file__, 369, 31), dirname_8502, *[imported_module_file_8503], **kwargs_8504)
        
        # Assigning a type to the variable 'imported_module_path' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'imported_module_path', dirname_call_result_8505)
        
        # Assigning a Name to a Name (line 370):
        
        # Assigning a Name to a Name (line 370):
        # Getting the type of 'False' (line 370)
        False_8506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 30), 'False')
        # Assigning a type to the variable 'imported_path_added' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'imported_path_added', False_8506)
        
        
        # Getting the type of 'imported_module_path' (line 371)
        imported_module_path_8507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 15), 'imported_module_path')
        # Getting the type of 'environment' (line 371)
        environment_8508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 39), 'environment')
        # Applying the binary operator 'in' (line 371)
        result_contains_8509 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 15), 'in', imported_module_path_8507, environment_8508)
        
        # Applying the 'not' unary operator (line 371)
        result_not__8510 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 11), 'not', result_contains_8509)
        
        # Testing if the type of an if condition is none (line 371)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 371, 8), result_not__8510):
            pass
        else:
            
            # Testing the type of an if condition (line 371)
            if_condition_8511 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 371, 8), result_not__8510)
            # Assigning a type to the variable 'if_condition_8511' (line 371)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'if_condition_8511', if_condition_8511)
            # SSA begins for if statement (line 371)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 372):
            
            # Assigning a Name to a Name (line 372):
            # Getting the type of 'True' (line 372)
            True_8512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 34), 'True')
            # Assigning a type to the variable 'imported_path_added' (line 372)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'imported_path_added', True_8512)
            
            # Call to append(...): (line 373)
            # Processing the call arguments (line 373)
            # Getting the type of 'imported_module_path' (line 373)
            imported_module_path_8515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 31), 'imported_module_path', False)
            # Processing the call keyword arguments (line 373)
            kwargs_8516 = {}
            # Getting the type of 'environment' (line 373)
            environment_8513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'environment', False)
            # Obtaining the member 'append' of a type (line 373)
            append_8514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 12), environment_8513, 'append')
            # Calling append(args, kwargs) (line 373)
            append_call_result_8517 = invoke(stypy.reporting.localization.Localization(__file__, 373, 12), append_8514, *[imported_module_path_8515], **kwargs_8516)
            
            # SSA join for if statement (line 371)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 366)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Call to len(...): (line 375)
    # Processing the call arguments (line 375)
    # Getting the type of 'elements' (line 375)
    elements_8519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 11), 'elements', False)
    # Processing the call keyword arguments (line 375)
    kwargs_8520 = {}
    # Getting the type of 'len' (line 375)
    len_8518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 7), 'len', False)
    # Calling len(args, kwargs) (line 375)
    len_call_result_8521 = invoke(stypy.reporting.localization.Localization(__file__, 375, 7), len_8518, *[elements_8519], **kwargs_8520)
    
    int_8522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 24), 'int')
    # Applying the binary operator '==' (line 375)
    result_eq_8523 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 7), '==', len_call_result_8521, int_8522)
    
    # Testing if the type of an if condition is none (line 375)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 375, 4), result_eq_8523):
        pass
    else:
        
        # Testing the type of an if condition (line 375)
        if_condition_8524 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 375, 4), result_eq_8523)
        # Assigning a type to the variable 'if_condition_8524' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'if_condition_8524', if_condition_8524)
        # SSA begins for if statement (line 375)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_type_of(...): (line 377)
        # Processing the call arguments (line 377)
        # Getting the type of 'localization' (line 377)
        localization_8527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 36), 'localization', False)
        # Getting the type of 'imported_module_name' (line 377)
        imported_module_name_8528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 50), 'imported_module_name', False)
        # Getting the type of 'module_obj' (line 377)
        module_obj_8529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 72), 'module_obj', False)
        # Processing the call keyword arguments (line 377)
        kwargs_8530 = {}
        # Getting the type of 'dest_type_store' (line 377)
        dest_type_store_8525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'dest_type_store', False)
        # Obtaining the member 'set_type_of' of a type (line 377)
        set_type_of_8526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 8), dest_type_store_8525, 'set_type_of')
        # Calling set_type_of(args, kwargs) (line 377)
        set_type_of_call_result_8531 = invoke(stypy.reporting.localization.Localization(__file__, 377, 8), set_type_of_8526, *[localization_8527, imported_module_name_8528, module_obj_8529], **kwargs_8530)
        
        # Getting the type of 'None' (line 378)
        None_8532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'stypy_return_type', None_8532)
        # SSA join for if statement (line 375)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'elements' (line 381)
    elements_8533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 19), 'elements')
    # Assigning a type to the variable 'elements_8533' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'elements_8533', elements_8533)
    # Testing if the for loop is going to be iterated (line 381)
    # Testing the type of a for loop iterable (line 381)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 381, 4), elements_8533)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 381, 4), elements_8533):
        # Getting the type of the for loop variable (line 381)
        for_loop_var_8534 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 381, 4), elements_8533)
        # Assigning a type to the variable 'element' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'element', for_loop_var_8534)
        # SSA begins for a for statement (line 381)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'element' (line 383)
        element_8535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 11), 'element')
        str_8536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 22), 'str', '*')
        # Applying the binary operator '==' (line 383)
        result_eq_8537 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 11), '==', element_8535, str_8536)
        
        # Testing if the type of an if condition is none (line 383)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 383, 8), result_eq_8537):
            pass
        else:
            
            # Testing the type of an if condition (line 383)
            if_condition_8538 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 383, 8), result_eq_8537)
            # Assigning a type to the variable 'if_condition_8538' (line 383)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'if_condition_8538', if_condition_8538)
            # SSA begins for if statement (line 383)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 384):
            
            # Assigning a Call to a Name (line 384):
            
            # Call to __get_public_names_and_types_of_module(...): (line 384)
            # Processing the call arguments (line 384)
            # Getting the type of 'module_obj' (line 384)
            module_obj_8540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 69), 'module_obj', False)
            # Processing the call keyword arguments (line 384)
            kwargs_8541 = {}
            # Getting the type of '__get_public_names_and_types_of_module' (line 384)
            get_public_names_and_types_of_module_8539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 30), '__get_public_names_and_types_of_module', False)
            # Calling __get_public_names_and_types_of_module(args, kwargs) (line 384)
            get_public_names_and_types_of_module_call_result_8542 = invoke(stypy.reporting.localization.Localization(__file__, 384, 30), get_public_names_and_types_of_module_8539, *[module_obj_8540], **kwargs_8541)
            
            # Assigning a type to the variable 'public_elements' (line 384)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'public_elements', get_public_names_and_types_of_module_call_result_8542)
            
            # Getting the type of 'public_elements' (line 385)
            public_elements_8543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 34), 'public_elements')
            # Assigning a type to the variable 'public_elements_8543' (line 385)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'public_elements_8543', public_elements_8543)
            # Testing if the for loop is going to be iterated (line 385)
            # Testing the type of a for loop iterable (line 385)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 385, 12), public_elements_8543)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 385, 12), public_elements_8543):
                # Getting the type of the for loop variable (line 385)
                for_loop_var_8544 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 385, 12), public_elements_8543)
                # Assigning a type to the variable 'public_element' (line 385)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'public_element', for_loop_var_8544)
                # SSA begins for a for statement (line 385)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to __import_module_element(...): (line 386)
                # Processing the call arguments (line 386)
                # Getting the type of 'localization' (line 386)
                localization_8546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 40), 'localization', False)
                # Getting the type of 'imported_module_name' (line 386)
                imported_module_name_8547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 54), 'imported_module_name', False)
                # Getting the type of 'module_obj' (line 386)
                module_obj_8548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 76), 'module_obj', False)
                # Getting the type of 'public_element' (line 386)
                public_element_8549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 88), 'public_element', False)
                # Getting the type of 'dest_type_store' (line 386)
                dest_type_store_8550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 104), 'dest_type_store', False)
                # Getting the type of 'environment' (line 387)
                environment_8551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 40), 'environment', False)
                # Processing the call keyword arguments (line 386)
                kwargs_8552 = {}
                # Getting the type of '__import_module_element' (line 386)
                import_module_element_8545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 16), '__import_module_element', False)
                # Calling __import_module_element(args, kwargs) (line 386)
                import_module_element_call_result_8553 = invoke(stypy.reporting.localization.Localization(__file__, 386, 16), import_module_element_8545, *[localization_8546, imported_module_name_8547, module_obj_8548, public_element_8549, dest_type_store_8550, environment_8551], **kwargs_8552)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 383)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to __import_module_element(...): (line 391)
        # Processing the call arguments (line 391)
        # Getting the type of 'localization' (line 391)
        localization_8555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 32), 'localization', False)
        # Getting the type of 'imported_module_name' (line 391)
        imported_module_name_8556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 46), 'imported_module_name', False)
        # Getting the type of 'module_obj' (line 391)
        module_obj_8557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 68), 'module_obj', False)
        # Getting the type of 'element' (line 391)
        element_8558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 80), 'element', False)
        # Getting the type of 'dest_type_store' (line 391)
        dest_type_store_8559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 89), 'dest_type_store', False)
        # Getting the type of 'environment' (line 391)
        environment_8560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 106), 'environment', False)
        # Processing the call keyword arguments (line 391)
        kwargs_8561 = {}
        # Getting the type of '__import_module_element' (line 391)
        import_module_element_8554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), '__import_module_element', False)
        # Calling __import_module_element(args, kwargs) (line 391)
        import_module_element_call_result_8562 = invoke(stypy.reporting.localization.Localization(__file__, 391, 8), import_module_element_8554, *[localization_8555, imported_module_name_8556, module_obj_8557, element_8558, dest_type_store_8559, environment_8560], **kwargs_8561)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    
    # Call to is_python_library_module(...): (line 393)
    # Processing the call arguments (line 393)
    # Getting the type of 'imported_module_name' (line 393)
    imported_module_name_8565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 64), 'imported_module_name', False)
    # Processing the call keyword arguments (line 393)
    kwargs_8566 = {}
    # Getting the type of 'python_library_modules_copy' (line 393)
    python_library_modules_copy_8563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 11), 'python_library_modules_copy', False)
    # Obtaining the member 'is_python_library_module' of a type (line 393)
    is_python_library_module_8564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 11), python_library_modules_copy_8563, 'is_python_library_module')
    # Calling is_python_library_module(args, kwargs) (line 393)
    is_python_library_module_call_result_8567 = invoke(stypy.reporting.localization.Localization(__file__, 393, 11), is_python_library_module_8564, *[imported_module_name_8565], **kwargs_8566)
    
    # Applying the 'not' unary operator (line 393)
    result_not__8568 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 7), 'not', is_python_library_module_call_result_8567)
    
    # Testing if the type of an if condition is none (line 393)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 393, 4), result_not__8568):
        pass
    else:
        
        # Testing the type of an if condition (line 393)
        if_condition_8569 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 393, 4), result_not__8568)
        # Assigning a type to the variable 'if_condition_8569' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'if_condition_8569', if_condition_8569)
        # SSA begins for if statement (line 393)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'destination_path_added' (line 394)
        destination_path_added_8570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 11), 'destination_path_added')
        # Testing if the type of an if condition is none (line 394)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 394, 8), destination_path_added_8570):
            pass
        else:
            
            # Testing the type of an if condition (line 394)
            if_condition_8571 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 394, 8), destination_path_added_8570)
            # Assigning a type to the variable 'if_condition_8571' (line 394)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'if_condition_8571', if_condition_8571)
            # SSA begins for if statement (line 394)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to remove(...): (line 395)
            # Processing the call arguments (line 395)
            # Getting the type of 'destination_module_path' (line 395)
            destination_module_path_8574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 31), 'destination_module_path', False)
            # Processing the call keyword arguments (line 395)
            kwargs_8575 = {}
            # Getting the type of 'environment' (line 395)
            environment_8572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'environment', False)
            # Obtaining the member 'remove' of a type (line 395)
            remove_8573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 12), environment_8572, 'remove')
            # Calling remove(args, kwargs) (line 395)
            remove_call_result_8576 = invoke(stypy.reporting.localization.Localization(__file__, 395, 12), remove_8573, *[destination_module_path_8574], **kwargs_8575)
            
            # SSA join for if statement (line 394)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'imported_path_added' (line 396)
        imported_path_added_8577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 11), 'imported_path_added')
        # Testing if the type of an if condition is none (line 396)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 396, 8), imported_path_added_8577):
            pass
        else:
            
            # Testing the type of an if condition (line 396)
            if_condition_8578 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 396, 8), imported_path_added_8577)
            # Assigning a type to the variable 'if_condition_8578' (line 396)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'if_condition_8578', if_condition_8578)
            # SSA begins for if statement (line 396)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to remove(...): (line 397)
            # Processing the call arguments (line 397)
            # Getting the type of 'imported_module_path' (line 397)
            imported_module_path_8581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 31), 'imported_module_path', False)
            # Processing the call keyword arguments (line 397)
            kwargs_8582 = {}
            # Getting the type of 'environment' (line 397)
            environment_8579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'environment', False)
            # Obtaining the member 'remove' of a type (line 397)
            remove_8580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 12), environment_8579, 'remove')
            # Calling remove(args, kwargs) (line 397)
            remove_call_result_8583 = invoke(stypy.reporting.localization.Localization(__file__, 397, 12), remove_8580, *[imported_module_path_8581], **kwargs_8582)
            
            # SSA join for if statement (line 396)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 393)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'import_elements_from_external_module(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'import_elements_from_external_module' in the type store
    # Getting the type of 'stypy_return_type' (line 340)
    stypy_return_type_8584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8584)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'import_elements_from_external_module'
    return stypy_return_type_8584

# Assigning a type to the variable 'import_elements_from_external_module' (line 340)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 0), 'import_elements_from_external_module', import_elements_from_external_module)

@norecursion
def __import_from(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_8585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 57), 'str', '__builtin__')
    defaults = [str_8585]
    # Create a new context for function '__import_from'
    module_type_store = module_type_store.open_function_context('__import_from', 403, 0, False)
    
    # Passed parameters checking function
    __import_from.stypy_localization = localization
    __import_from.stypy_type_of_self = None
    __import_from.stypy_type_store = module_type_store
    __import_from.stypy_function_name = '__import_from'
    __import_from.stypy_param_names_list = ['localization', 'member_name', 'module_name']
    __import_from.stypy_varargs_param_name = None
    __import_from.stypy_kwargs_param_name = None
    __import_from.stypy_call_defaults = defaults
    __import_from.stypy_call_varargs = varargs
    __import_from.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__import_from', ['localization', 'member_name', 'module_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__import_from', localization, ['localization', 'member_name', 'module_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__import_from(...)' code ##################

    str_8586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, (-1)), 'str', '\n    Imports a single member from a module. If no module is specified, the builtin module is used instead. Models the\n    "from <module> import <member>" sentence, being a sort version of the import_elements_from_external_module function\n    but only for Python library modules. This is a helper function of the following one.\n    :param localization: Caller information\n    :param member_name: Member to import\n    :param module_name: Python library module that contains the member or nothing to use the __builtins__ module\n    :return: A TypeInferenceProxy for the specified member or a TypeError if the requested element do not exist\n    ')
    
    # Assigning a Call to a Name (line 413):
    
    # Assigning a Call to a Name (line 413):
    
    # Call to import_python_module(...): (line 413)
    # Processing the call arguments (line 413)
    # Getting the type of 'localization' (line 413)
    localization_8588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 34), 'localization', False)
    # Getting the type of 'module_name' (line 413)
    module_name_8589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 48), 'module_name', False)
    # Processing the call keyword arguments (line 413)
    kwargs_8590 = {}
    # Getting the type of 'import_python_module' (line 413)
    import_python_module_8587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 13), 'import_python_module', False)
    # Calling import_python_module(args, kwargs) (line 413)
    import_python_module_call_result_8591 = invoke(stypy.reporting.localization.Localization(__file__, 413, 13), import_python_module_8587, *[localization_8588, module_name_8589], **kwargs_8590)
    
    # Assigning a type to the variable 'module' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'module', import_python_module_call_result_8591)
    
    # Type idiom detected: calculating its left and rigth part (line 414)
    # Getting the type of 'TypeError' (line 414)
    TypeError_8592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 26), 'TypeError')
    # Getting the type of 'module' (line 414)
    module_8593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 18), 'module')
    
    (may_be_8594, more_types_in_union_8595) = may_be_subtype(TypeError_8592, module_8593)

    if may_be_8594:

        if more_types_in_union_8595:
            # Runtime conditional SSA (line 414)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'module' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'module', remove_not_subtype_from_union(module_8593, TypeError))
        
        # Obtaining an instance of the builtin type 'tuple' (line 415)
        tuple_8596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 415)
        # Adding element type (line 415)
        # Getting the type of 'module' (line 415)
        module_8597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 15), 'module')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 15), tuple_8596, module_8597)
        # Adding element type (line 415)
        # Getting the type of 'None' (line 415)
        None_8598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 23), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 15), tuple_8596, None_8598)
        
        # Assigning a type to the variable 'stypy_return_type' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'stypy_return_type', tuple_8596)

        if more_types_in_union_8595:
            # SSA join for if statement (line 414)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # SSA begins for try-except statement (line 417)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining an instance of the builtin type 'tuple' (line 418)
    tuple_8599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 418)
    # Adding element type (line 418)
    # Getting the type of 'module' (line 418)
    module_8600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 15), 'module')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 15), tuple_8599, module_8600)
    # Adding element type (line 418)
    
    # Call to get_type_of_member(...): (line 418)
    # Processing the call arguments (line 418)
    # Getting the type of 'localization' (line 418)
    localization_8603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 49), 'localization', False)
    # Getting the type of 'member_name' (line 418)
    member_name_8604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 63), 'member_name', False)
    # Processing the call keyword arguments (line 418)
    kwargs_8605 = {}
    # Getting the type of 'module' (line 418)
    module_8601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 23), 'module', False)
    # Obtaining the member 'get_type_of_member' of a type (line 418)
    get_type_of_member_8602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 23), module_8601, 'get_type_of_member')
    # Calling get_type_of_member(args, kwargs) (line 418)
    get_type_of_member_call_result_8606 = invoke(stypy.reporting.localization.Localization(__file__, 418, 23), get_type_of_member_8602, *[localization_8603, member_name_8604], **kwargs_8605)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 15), tuple_8599, get_type_of_member_call_result_8606)
    
    # Assigning a type to the variable 'stypy_return_type' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'stypy_return_type', tuple_8599)
    # SSA branch for the except part of a try statement (line 417)
    # SSA branch for the except 'Exception' branch of a try statement (line 417)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 419)
    Exception_8607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 11), 'Exception')
    # Assigning a type to the variable 'exc' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'exc', Exception_8607)
    
    # Obtaining an instance of the builtin type 'tuple' (line 420)
    tuple_8608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 420)
    # Adding element type (line 420)
    # Getting the type of 'module' (line 420)
    module_8609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 15), 'module')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 15), tuple_8608, module_8609)
    # Adding element type (line 420)
    
    # Call to TypeError(...): (line 420)
    # Processing the call arguments (line 420)
    # Getting the type of 'localization' (line 420)
    localization_8611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 33), 'localization', False)
    
    # Call to format(...): (line 421)
    # Processing the call arguments (line 421)
    # Getting the type of 'member_name' (line 421)
    member_name_8614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 93), 'member_name', False)
    # Getting the type of 'module_name' (line 421)
    module_name_8615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 106), 'module_name', False)
    
    # Call to str(...): (line 422)
    # Processing the call arguments (line 422)
    # Getting the type of 'exc' (line 422)
    exc_8617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 97), 'exc', False)
    # Processing the call keyword arguments (line 422)
    kwargs_8618 = {}
    # Getting the type of 'str' (line 422)
    str_8616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 93), 'str', False)
    # Calling str(args, kwargs) (line 422)
    str_call_result_8619 = invoke(stypy.reporting.localization.Localization(__file__, 422, 93), str_8616, *[exc_8617], **kwargs_8618)
    
    # Processing the call keyword arguments (line 421)
    kwargs_8620 = {}
    str_8612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 33), 'str', "Could not load member '{0}' from module '{1}': {2}")
    # Obtaining the member 'format' of a type (line 421)
    format_8613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 33), str_8612, 'format')
    # Calling format(args, kwargs) (line 421)
    format_call_result_8621 = invoke(stypy.reporting.localization.Localization(__file__, 421, 33), format_8613, *[member_name_8614, module_name_8615, str_call_result_8619], **kwargs_8620)
    
    # Processing the call keyword arguments (line 420)
    kwargs_8622 = {}
    # Getting the type of 'TypeError' (line 420)
    TypeError_8610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 23), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 420)
    TypeError_call_result_8623 = invoke(stypy.reporting.localization.Localization(__file__, 420, 23), TypeError_8610, *[localization_8611, format_call_result_8621], **kwargs_8622)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 15), tuple_8608, TypeError_call_result_8623)
    
    # Assigning a type to the variable 'stypy_return_type' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'stypy_return_type', tuple_8608)
    # SSA join for try-except statement (line 417)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '__import_from(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__import_from' in the type store
    # Getting the type of 'stypy_return_type' (line 403)
    stypy_return_type_8624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8624)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__import_from'
    return stypy_return_type_8624

# Assigning a type to the variable '__import_from' (line 403)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 0), '__import_from', __import_from)

@norecursion
def import_from(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_8625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 55), 'str', '__builtin__')
    defaults = [str_8625]
    # Create a new context for function 'import_from'
    module_type_store = module_type_store.open_function_context('import_from', 425, 0, False)
    
    # Passed parameters checking function
    import_from.stypy_localization = localization
    import_from.stypy_type_of_self = None
    import_from.stypy_type_store = module_type_store
    import_from.stypy_function_name = 'import_from'
    import_from.stypy_param_names_list = ['localization', 'member_name', 'module_name']
    import_from.stypy_varargs_param_name = None
    import_from.stypy_kwargs_param_name = None
    import_from.stypy_call_defaults = defaults
    import_from.stypy_call_varargs = varargs
    import_from.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'import_from', ['localization', 'member_name', 'module_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'import_from', localization, ['localization', 'member_name', 'module_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'import_from(...)' code ##################

    str_8626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, (-1)), 'str', '\n    Imports a single member from a module. If no module is specified, the builtin module is used instead. Models the\n    "from <module> import <member>" sentence, being a sort version of the import_elements_from_external_module function\n    but only for Python library modules\n    :param localization: Caller information\n    :param member_name: Member to import\n    :param module_name: Python library module that contains the member or nothing to use the __builtins__ module\n    :return: A TypeInferenceProxy for the specified member or a TypeError if the requested element do not exist\n    ')
    
    # Getting the type of 'member_name' (line 436)
    member_name_8627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 7), 'member_name')
    # Getting the type of '__known_types' (line 436)
    known_types_8628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 22), '__known_types')
    # Applying the binary operator 'in' (line 436)
    result_contains_8629 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 7), 'in', member_name_8627, known_types_8628)
    
    # Testing if the type of an if condition is none (line 436)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 436, 4), result_contains_8629):
        pass
    else:
        
        # Testing the type of an if condition (line 436)
        if_condition_8630 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 436, 4), result_contains_8629)
        # Assigning a type to the variable 'if_condition_8630' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'if_condition_8630', if_condition_8630)
        # SSA begins for if statement (line 436)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        # Getting the type of 'member_name' (line 437)
        member_name_8631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 29), 'member_name')
        # Getting the type of '__known_types' (line 437)
        known_types_8632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 15), '__known_types')
        # Obtaining the member '__getitem__' of a type (line 437)
        getitem___8633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 15), known_types_8632, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 437)
        subscript_call_result_8634 = invoke(stypy.reporting.localization.Localization(__file__, 437, 15), getitem___8633, member_name_8631)
        
        # Assigning a type to the variable 'stypy_return_type' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'stypy_return_type', subscript_call_result_8634)
        # SSA join for if statement (line 436)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Tuple (line 439):
    
    # Assigning a Call to a Name:
    
    # Call to __import_from(...): (line 439)
    # Processing the call arguments (line 439)
    # Getting the type of 'localization' (line 439)
    localization_8636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 35), 'localization', False)
    # Getting the type of 'member_name' (line 439)
    member_name_8637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 49), 'member_name', False)
    # Getting the type of 'module_name' (line 439)
    module_name_8638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 62), 'module_name', False)
    # Processing the call keyword arguments (line 439)
    kwargs_8639 = {}
    # Getting the type of '__import_from' (line 439)
    import_from_8635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 21), '__import_from', False)
    # Calling __import_from(args, kwargs) (line 439)
    import_from_call_result_8640 = invoke(stypy.reporting.localization.Localization(__file__, 439, 21), import_from_8635, *[localization_8636, member_name_8637, module_name_8638], **kwargs_8639)
    
    # Assigning a type to the variable 'call_assignment_7911' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'call_assignment_7911', import_from_call_result_8640)
    
    # Assigning a Call to a Name (line 439):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_7911' (line 439)
    call_assignment_7911_8641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'call_assignment_7911', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_8642 = stypy_get_value_from_tuple(call_assignment_7911_8641, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_7912' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'call_assignment_7912', stypy_get_value_from_tuple_call_result_8642)
    
    # Assigning a Name to a Name (line 439):
    # Getting the type of 'call_assignment_7912' (line 439)
    call_assignment_7912_8643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'call_assignment_7912')
    # Assigning a type to the variable 'module' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'module', call_assignment_7912_8643)
    
    # Assigning a Call to a Name (line 439):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_7911' (line 439)
    call_assignment_7911_8644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'call_assignment_7911', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_8645 = stypy_get_value_from_tuple(call_assignment_7911_8644, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_7913' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'call_assignment_7913', stypy_get_value_from_tuple_call_result_8645)
    
    # Assigning a Name to a Name (line 439):
    # Getting the type of 'call_assignment_7913' (line 439)
    call_assignment_7913_8646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'call_assignment_7913')
    # Assigning a type to the variable 'member' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 12), 'member', call_assignment_7913_8646)
    
    # Type idiom detected: calculating its left and rigth part (line 440)
    # Getting the type of 'TypeError' (line 440)
    TypeError_8647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 30), 'TypeError')
    # Getting the type of 'member' (line 440)
    member_8648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 22), 'member')
    
    (may_be_8649, more_types_in_union_8650) = may_not_be_subtype(TypeError_8647, member_8648)

    if may_be_8649:

        if more_types_in_union_8650:
            # Runtime conditional SSA (line 440)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'member' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'member', remove_subtype_from_union(member_8648, TypeError))
        
        # Assigning a Call to a Name (line 441):
        
        # Assigning a Call to a Name (line 441):
        
        # Call to instance(...): (line 441)
        # Processing the call arguments (line 441)
        # Getting the type of 'module' (line 441)
        module_8654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 66), 'module', False)
        # Obtaining the member 'python_entity' of a type (line 441)
        python_entity_8655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 66), module_8654, 'python_entity')
        # Processing the call keyword arguments (line 441)
        kwargs_8656 = {}
        # Getting the type of 'type_inference_proxy_copy' (line 441)
        type_inference_proxy_copy_8651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'type_inference_proxy_copy', False)
        # Obtaining the member 'TypeInferenceProxy' of a type (line 441)
        TypeInferenceProxy_8652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 12), type_inference_proxy_copy_8651, 'TypeInferenceProxy')
        # Obtaining the member 'instance' of a type (line 441)
        instance_8653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 12), TypeInferenceProxy_8652, 'instance')
        # Calling instance(args, kwargs) (line 441)
        instance_call_result_8657 = invoke(stypy.reporting.localization.Localization(__file__, 441, 12), instance_8653, *[python_entity_8655], **kwargs_8656)
        
        # Assigning a type to the variable 'm' (line 441)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'm', instance_call_result_8657)
        
        # Call to instance(...): (line 442)
        # Processing the call arguments (line 442)
        # Getting the type of 'member' (line 442)
        member_8661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 69), 'member', False)
        # Processing the call keyword arguments (line 442)
        # Getting the type of 'm' (line 442)
        m_8662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 84), 'm', False)
        keyword_8663 = m_8662
        kwargs_8664 = {'parent': keyword_8663}
        # Getting the type of 'type_inference_proxy_copy' (line 442)
        type_inference_proxy_copy_8658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 15), 'type_inference_proxy_copy', False)
        # Obtaining the member 'TypeInferenceProxy' of a type (line 442)
        TypeInferenceProxy_8659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 15), type_inference_proxy_copy_8658, 'TypeInferenceProxy')
        # Obtaining the member 'instance' of a type (line 442)
        instance_8660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 15), TypeInferenceProxy_8659, 'instance')
        # Calling instance(args, kwargs) (line 442)
        instance_call_result_8665 = invoke(stypy.reporting.localization.Localization(__file__, 442, 15), instance_8660, *[member_8661], **kwargs_8664)
        
        # Assigning a type to the variable 'stypy_return_type' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'stypy_return_type', instance_call_result_8665)

        if more_types_in_union_8650:
            # SSA join for if statement (line 440)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'member' (line 444)
    member_8666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 11), 'member')
    # Assigning a type to the variable 'stypy_return_type' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'stypy_return_type', member_8666)
    
    # ################# End of 'import_from(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'import_from' in the type store
    # Getting the type of 'stypy_return_type' (line 425)
    stypy_return_type_8667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8667)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'import_from'
    return stypy_return_type_8667

# Assigning a type to the variable 'import_from' (line 425)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 0), 'import_from', import_from)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
