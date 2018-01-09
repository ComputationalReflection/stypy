
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function
2: 
3: import os
4: import genapi
5: 
6: from genapi import \
7:         TypeApi, GlobalVarApi, FunctionApi, BoolValuesApi
8: 
9: import numpy_api
10: 
11: # use annotated api when running under cpychecker
12: h_template = r'''
13: #if defined(_MULTIARRAYMODULE) || defined(WITH_CPYCHECKER_STEALS_REFERENCE_TO_ARG_ATTRIBUTE)
14: 
15: typedef struct {
16:         PyObject_HEAD
17:         npy_bool obval;
18: } PyBoolScalarObject;
19: 
20: extern NPY_NO_EXPORT PyTypeObject PyArrayMapIter_Type;
21: extern NPY_NO_EXPORT PyTypeObject PyArrayNeighborhoodIter_Type;
22: extern NPY_NO_EXPORT PyBoolScalarObject _PyArrayScalar_BoolValues[2];
23: 
24: %s
25: 
26: #else
27: 
28: #if defined(PY_ARRAY_UNIQUE_SYMBOL)
29: #define PyArray_API PY_ARRAY_UNIQUE_SYMBOL
30: #endif
31: 
32: #if defined(NO_IMPORT) || defined(NO_IMPORT_ARRAY)
33: extern void **PyArray_API;
34: #else
35: #if defined(PY_ARRAY_UNIQUE_SYMBOL)
36: void **PyArray_API;
37: #else
38: static void **PyArray_API=NULL;
39: #endif
40: #endif
41: 
42: %s
43: 
44: #if !defined(NO_IMPORT_ARRAY) && !defined(NO_IMPORT)
45: static int
46: _import_array(void)
47: {
48:   int st;
49:   PyObject *numpy = PyImport_ImportModule("numpy.core.multiarray");
50:   PyObject *c_api = NULL;
51: 
52:   if (numpy == NULL) {
53:       PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
54:       return -1;
55:   }
56:   c_api = PyObject_GetAttrString(numpy, "_ARRAY_API");
57:   Py_DECREF(numpy);
58:   if (c_api == NULL) {
59:       PyErr_SetString(PyExc_AttributeError, "_ARRAY_API not found");
60:       return -1;
61:   }
62: 
63: #if PY_VERSION_HEX >= 0x03000000
64:   if (!PyCapsule_CheckExact(c_api)) {
65:       PyErr_SetString(PyExc_RuntimeError, "_ARRAY_API is not PyCapsule object");
66:       Py_DECREF(c_api);
67:       return -1;
68:   }
69:   PyArray_API = (void **)PyCapsule_GetPointer(c_api, NULL);
70: #else
71:   if (!PyCObject_Check(c_api)) {
72:       PyErr_SetString(PyExc_RuntimeError, "_ARRAY_API is not PyCObject object");
73:       Py_DECREF(c_api);
74:       return -1;
75:   }
76:   PyArray_API = (void **)PyCObject_AsVoidPtr(c_api);
77: #endif
78:   Py_DECREF(c_api);
79:   if (PyArray_API == NULL) {
80:       PyErr_SetString(PyExc_RuntimeError, "_ARRAY_API is NULL pointer");
81:       return -1;
82:   }
83: 
84:   /* Perform runtime check of C API version */
85:   if (NPY_VERSION != PyArray_GetNDArrayCVersion()) {
86:       PyErr_Format(PyExc_RuntimeError, "module compiled against "\
87:              "ABI version 0x%%x but this version of numpy is 0x%%x", \
88:              (int) NPY_VERSION, (int) PyArray_GetNDArrayCVersion());
89:       return -1;
90:   }
91:   if (NPY_FEATURE_VERSION > PyArray_GetNDArrayCFeatureVersion()) {
92:       PyErr_Format(PyExc_RuntimeError, "module compiled against "\
93:              "API version 0x%%x but this version of numpy is 0x%%x", \
94:              (int) NPY_FEATURE_VERSION, (int) PyArray_GetNDArrayCFeatureVersion());
95:       return -1;
96:   }
97: 
98:   /*
99:    * Perform runtime check of endianness and check it matches the one set by
100:    * the headers (npy_endian.h) as a safeguard
101:    */
102:   st = PyArray_GetEndianness();
103:   if (st == NPY_CPU_UNKNOWN_ENDIAN) {
104:       PyErr_Format(PyExc_RuntimeError, "FATAL: module compiled as unknown endian");
105:       return -1;
106:   }
107: #if NPY_BYTE_ORDER == NPY_BIG_ENDIAN
108:   if (st != NPY_CPU_BIG) {
109:       PyErr_Format(PyExc_RuntimeError, "FATAL: module compiled as "\
110:              "big endian, but detected different endianness at runtime");
111:       return -1;
112:   }
113: #elif NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN
114:   if (st != NPY_CPU_LITTLE) {
115:       PyErr_Format(PyExc_RuntimeError, "FATAL: module compiled as "\
116:              "little endian, but detected different endianness at runtime");
117:       return -1;
118:   }
119: #endif
120: 
121:   return 0;
122: }
123: 
124: #if PY_VERSION_HEX >= 0x03000000
125: #define NUMPY_IMPORT_ARRAY_RETVAL NULL
126: #else
127: #define NUMPY_IMPORT_ARRAY_RETVAL
128: #endif
129: 
130: #define import_array() {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); return NUMPY_IMPORT_ARRAY_RETVAL; } }
131: 
132: #define import_array1(ret) {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); return ret; } }
133: 
134: #define import_array2(msg, ret) {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, msg); return ret; } }
135: 
136: #endif
137: 
138: #endif
139: '''
140: 
141: 
142: c_template = r'''
143: /* These pointers will be stored in the C-object for use in other
144:     extension modules
145: */
146: 
147: void *PyArray_API[] = {
148: %s
149: };
150: '''
151: 
152: c_api_header = '''
153: ===========
154: Numpy C-API
155: ===========
156: '''
157: 
158: def generate_api(output_dir, force=False):
159:     basename = 'multiarray_api'
160: 
161:     h_file = os.path.join(output_dir, '__%s.h' % basename)
162:     c_file = os.path.join(output_dir, '__%s.c' % basename)
163:     d_file = os.path.join(output_dir, '%s.txt' % basename)
164:     targets = (h_file, c_file, d_file)
165: 
166:     sources = numpy_api.multiarray_api
167: 
168:     if (not force and not genapi.should_rebuild(targets, [numpy_api.__file__, __file__])):
169:         return targets
170:     else:
171:         do_generate_api(targets, sources)
172: 
173:     return targets
174: 
175: def do_generate_api(targets, sources):
176:     header_file = targets[0]
177:     c_file = targets[1]
178:     doc_file = targets[2]
179: 
180:     global_vars = sources[0]
181:     scalar_bool_values = sources[1]
182:     types_api = sources[2]
183:     multiarray_funcs = sources[3]
184: 
185:     multiarray_api = sources[:]
186: 
187:     module_list = []
188:     extension_list = []
189:     init_list = []
190: 
191:     # Check multiarray api indexes
192:     multiarray_api_index = genapi.merge_api_dicts(multiarray_api)
193:     genapi.check_api_dict(multiarray_api_index)
194: 
195:     numpyapi_list = genapi.get_api_functions('NUMPY_API',
196:                                               multiarray_funcs)
197:     ordered_funcs_api = genapi.order_dict(multiarray_funcs)
198: 
199:     # Create dict name -> *Api instance
200:     api_name = 'PyArray_API'
201:     multiarray_api_dict = {}
202:     for f in numpyapi_list:
203:         name = f.name
204:         index = multiarray_funcs[name][0]
205:         annotations = multiarray_funcs[name][1:]
206:         multiarray_api_dict[f.name] = FunctionApi(f.name, index, annotations,
207:                                                   f.return_type,
208:                                                   f.args, api_name)
209: 
210:     for name, val in global_vars.items():
211:         index, type = val
212:         multiarray_api_dict[name] = GlobalVarApi(name, index, type, api_name)
213: 
214:     for name, val in scalar_bool_values.items():
215:         index = val[0]
216:         multiarray_api_dict[name] = BoolValuesApi(name, index, api_name)
217: 
218:     for name, val in types_api.items():
219:         index = val[0]
220:         multiarray_api_dict[name] = TypeApi(name, index, 'PyTypeObject', api_name)
221: 
222:     if len(multiarray_api_dict) != len(multiarray_api_index):
223:         raise AssertionError("Multiarray API size mismatch %d %d" %
224:                         (len(multiarray_api_dict), len(multiarray_api_index)))
225: 
226:     extension_list = []
227:     for name, index in genapi.order_dict(multiarray_api_index):
228:         api_item = multiarray_api_dict[name]
229:         extension_list.append(api_item.define_from_array_api_string())
230:         init_list.append(api_item.array_api_define())
231:         module_list.append(api_item.internal_define())
232: 
233:     # Write to header
234:     fid = open(header_file, 'w')
235:     s = h_template % ('\n'.join(module_list), '\n'.join(extension_list))
236:     fid.write(s)
237:     fid.close()
238: 
239:     # Write to c-code
240:     fid = open(c_file, 'w')
241:     s = c_template % ',\n'.join(init_list)
242:     fid.write(s)
243:     fid.close()
244: 
245:     # write to documentation
246:     fid = open(doc_file, 'w')
247:     fid.write(c_api_header)
248:     for func in numpyapi_list:
249:         fid.write(func.to_ReST())
250:         fid.write('\n\n')
251:     fid.close()
252: 
253:     return targets
254: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import genapi' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_5408 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'genapi')

if (type(import_5408) is not StypyTypeError):

    if (import_5408 != 'pyd_module'):
        __import__(import_5408)
        sys_modules_5409 = sys.modules[import_5408]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'genapi', sys_modules_5409.module_type_store, module_type_store)
    else:
        import genapi

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'genapi', genapi, module_type_store)

else:
    # Assigning a type to the variable 'genapi' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'genapi', import_5408)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from genapi import TypeApi, GlobalVarApi, FunctionApi, BoolValuesApi' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_5410 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'genapi')

if (type(import_5410) is not StypyTypeError):

    if (import_5410 != 'pyd_module'):
        __import__(import_5410)
        sys_modules_5411 = sys.modules[import_5410]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'genapi', sys_modules_5411.module_type_store, module_type_store, ['TypeApi', 'GlobalVarApi', 'FunctionApi', 'BoolValuesApi'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_5411, sys_modules_5411.module_type_store, module_type_store)
    else:
        from genapi import TypeApi, GlobalVarApi, FunctionApi, BoolValuesApi

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'genapi', None, module_type_store, ['TypeApi', 'GlobalVarApi', 'FunctionApi', 'BoolValuesApi'], [TypeApi, GlobalVarApi, FunctionApi, BoolValuesApi])

else:
    # Assigning a type to the variable 'genapi' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'genapi', import_5410)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy_api' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_5412 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy_api')

if (type(import_5412) is not StypyTypeError):

    if (import_5412 != 'pyd_module'):
        __import__(import_5412)
        sys_modules_5413 = sys.modules[import_5412]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy_api', sys_modules_5413.module_type_store, module_type_store)
    else:
        import numpy_api

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy_api', numpy_api, module_type_store)

else:
    # Assigning a type to the variable 'numpy_api' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy_api', import_5412)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')


# Assigning a Str to a Name (line 12):

# Assigning a Str to a Name (line 12):
str_5414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, (-1)), 'str', '\n#if defined(_MULTIARRAYMODULE) || defined(WITH_CPYCHECKER_STEALS_REFERENCE_TO_ARG_ATTRIBUTE)\n\ntypedef struct {\n        PyObject_HEAD\n        npy_bool obval;\n} PyBoolScalarObject;\n\nextern NPY_NO_EXPORT PyTypeObject PyArrayMapIter_Type;\nextern NPY_NO_EXPORT PyTypeObject PyArrayNeighborhoodIter_Type;\nextern NPY_NO_EXPORT PyBoolScalarObject _PyArrayScalar_BoolValues[2];\n\n%s\n\n#else\n\n#if defined(PY_ARRAY_UNIQUE_SYMBOL)\n#define PyArray_API PY_ARRAY_UNIQUE_SYMBOL\n#endif\n\n#if defined(NO_IMPORT) || defined(NO_IMPORT_ARRAY)\nextern void **PyArray_API;\n#else\n#if defined(PY_ARRAY_UNIQUE_SYMBOL)\nvoid **PyArray_API;\n#else\nstatic void **PyArray_API=NULL;\n#endif\n#endif\n\n%s\n\n#if !defined(NO_IMPORT_ARRAY) && !defined(NO_IMPORT)\nstatic int\n_import_array(void)\n{\n  int st;\n  PyObject *numpy = PyImport_ImportModule("numpy.core.multiarray");\n  PyObject *c_api = NULL;\n\n  if (numpy == NULL) {\n      PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");\n      return -1;\n  }\n  c_api = PyObject_GetAttrString(numpy, "_ARRAY_API");\n  Py_DECREF(numpy);\n  if (c_api == NULL) {\n      PyErr_SetString(PyExc_AttributeError, "_ARRAY_API not found");\n      return -1;\n  }\n\n#if PY_VERSION_HEX >= 0x03000000\n  if (!PyCapsule_CheckExact(c_api)) {\n      PyErr_SetString(PyExc_RuntimeError, "_ARRAY_API is not PyCapsule object");\n      Py_DECREF(c_api);\n      return -1;\n  }\n  PyArray_API = (void **)PyCapsule_GetPointer(c_api, NULL);\n#else\n  if (!PyCObject_Check(c_api)) {\n      PyErr_SetString(PyExc_RuntimeError, "_ARRAY_API is not PyCObject object");\n      Py_DECREF(c_api);\n      return -1;\n  }\n  PyArray_API = (void **)PyCObject_AsVoidPtr(c_api);\n#endif\n  Py_DECREF(c_api);\n  if (PyArray_API == NULL) {\n      PyErr_SetString(PyExc_RuntimeError, "_ARRAY_API is NULL pointer");\n      return -1;\n  }\n\n  /* Perform runtime check of C API version */\n  if (NPY_VERSION != PyArray_GetNDArrayCVersion()) {\n      PyErr_Format(PyExc_RuntimeError, "module compiled against "\\\n             "ABI version 0x%%x but this version of numpy is 0x%%x", \\\n             (int) NPY_VERSION, (int) PyArray_GetNDArrayCVersion());\n      return -1;\n  }\n  if (NPY_FEATURE_VERSION > PyArray_GetNDArrayCFeatureVersion()) {\n      PyErr_Format(PyExc_RuntimeError, "module compiled against "\\\n             "API version 0x%%x but this version of numpy is 0x%%x", \\\n             (int) NPY_FEATURE_VERSION, (int) PyArray_GetNDArrayCFeatureVersion());\n      return -1;\n  }\n\n  /*\n   * Perform runtime check of endianness and check it matches the one set by\n   * the headers (npy_endian.h) as a safeguard\n   */\n  st = PyArray_GetEndianness();\n  if (st == NPY_CPU_UNKNOWN_ENDIAN) {\n      PyErr_Format(PyExc_RuntimeError, "FATAL: module compiled as unknown endian");\n      return -1;\n  }\n#if NPY_BYTE_ORDER == NPY_BIG_ENDIAN\n  if (st != NPY_CPU_BIG) {\n      PyErr_Format(PyExc_RuntimeError, "FATAL: module compiled as "\\\n             "big endian, but detected different endianness at runtime");\n      return -1;\n  }\n#elif NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN\n  if (st != NPY_CPU_LITTLE) {\n      PyErr_Format(PyExc_RuntimeError, "FATAL: module compiled as "\\\n             "little endian, but detected different endianness at runtime");\n      return -1;\n  }\n#endif\n\n  return 0;\n}\n\n#if PY_VERSION_HEX >= 0x03000000\n#define NUMPY_IMPORT_ARRAY_RETVAL NULL\n#else\n#define NUMPY_IMPORT_ARRAY_RETVAL\n#endif\n\n#define import_array() {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); return NUMPY_IMPORT_ARRAY_RETVAL; } }\n\n#define import_array1(ret) {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); return ret; } }\n\n#define import_array2(msg, ret) {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, msg); return ret; } }\n\n#endif\n\n#endif\n')
# Assigning a type to the variable 'h_template' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'h_template', str_5414)

# Assigning a Str to a Name (line 142):

# Assigning a Str to a Name (line 142):
str_5415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, (-1)), 'str', '\n/* These pointers will be stored in the C-object for use in other\n    extension modules\n*/\n\nvoid *PyArray_API[] = {\n%s\n};\n')
# Assigning a type to the variable 'c_template' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'c_template', str_5415)

# Assigning a Str to a Name (line 152):

# Assigning a Str to a Name (line 152):
str_5416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, (-1)), 'str', '\n===========\nNumpy C-API\n===========\n')
# Assigning a type to the variable 'c_api_header' (line 152)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'c_api_header', str_5416)

@norecursion
def generate_api(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 158)
    False_5417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 35), 'False')
    defaults = [False_5417]
    # Create a new context for function 'generate_api'
    module_type_store = module_type_store.open_function_context('generate_api', 158, 0, False)
    
    # Passed parameters checking function
    generate_api.stypy_localization = localization
    generate_api.stypy_type_of_self = None
    generate_api.stypy_type_store = module_type_store
    generate_api.stypy_function_name = 'generate_api'
    generate_api.stypy_param_names_list = ['output_dir', 'force']
    generate_api.stypy_varargs_param_name = None
    generate_api.stypy_kwargs_param_name = None
    generate_api.stypy_call_defaults = defaults
    generate_api.stypy_call_varargs = varargs
    generate_api.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generate_api', ['output_dir', 'force'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generate_api', localization, ['output_dir', 'force'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generate_api(...)' code ##################

    
    # Assigning a Str to a Name (line 159):
    
    # Assigning a Str to a Name (line 159):
    str_5418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 15), 'str', 'multiarray_api')
    # Assigning a type to the variable 'basename' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'basename', str_5418)
    
    # Assigning a Call to a Name (line 161):
    
    # Assigning a Call to a Name (line 161):
    
    # Call to join(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'output_dir' (line 161)
    output_dir_5422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 26), 'output_dir', False)
    str_5423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 38), 'str', '__%s.h')
    # Getting the type of 'basename' (line 161)
    basename_5424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 49), 'basename', False)
    # Applying the binary operator '%' (line 161)
    result_mod_5425 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 38), '%', str_5423, basename_5424)
    
    # Processing the call keyword arguments (line 161)
    kwargs_5426 = {}
    # Getting the type of 'os' (line 161)
    os_5419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 13), 'os', False)
    # Obtaining the member 'path' of a type (line 161)
    path_5420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 13), os_5419, 'path')
    # Obtaining the member 'join' of a type (line 161)
    join_5421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 13), path_5420, 'join')
    # Calling join(args, kwargs) (line 161)
    join_call_result_5427 = invoke(stypy.reporting.localization.Localization(__file__, 161, 13), join_5421, *[output_dir_5422, result_mod_5425], **kwargs_5426)
    
    # Assigning a type to the variable 'h_file' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'h_file', join_call_result_5427)
    
    # Assigning a Call to a Name (line 162):
    
    # Assigning a Call to a Name (line 162):
    
    # Call to join(...): (line 162)
    # Processing the call arguments (line 162)
    # Getting the type of 'output_dir' (line 162)
    output_dir_5431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 26), 'output_dir', False)
    str_5432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 38), 'str', '__%s.c')
    # Getting the type of 'basename' (line 162)
    basename_5433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 49), 'basename', False)
    # Applying the binary operator '%' (line 162)
    result_mod_5434 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 38), '%', str_5432, basename_5433)
    
    # Processing the call keyword arguments (line 162)
    kwargs_5435 = {}
    # Getting the type of 'os' (line 162)
    os_5428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 13), 'os', False)
    # Obtaining the member 'path' of a type (line 162)
    path_5429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 13), os_5428, 'path')
    # Obtaining the member 'join' of a type (line 162)
    join_5430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 13), path_5429, 'join')
    # Calling join(args, kwargs) (line 162)
    join_call_result_5436 = invoke(stypy.reporting.localization.Localization(__file__, 162, 13), join_5430, *[output_dir_5431, result_mod_5434], **kwargs_5435)
    
    # Assigning a type to the variable 'c_file' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'c_file', join_call_result_5436)
    
    # Assigning a Call to a Name (line 163):
    
    # Assigning a Call to a Name (line 163):
    
    # Call to join(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'output_dir' (line 163)
    output_dir_5440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 26), 'output_dir', False)
    str_5441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 38), 'str', '%s.txt')
    # Getting the type of 'basename' (line 163)
    basename_5442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 49), 'basename', False)
    # Applying the binary operator '%' (line 163)
    result_mod_5443 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 38), '%', str_5441, basename_5442)
    
    # Processing the call keyword arguments (line 163)
    kwargs_5444 = {}
    # Getting the type of 'os' (line 163)
    os_5437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 13), 'os', False)
    # Obtaining the member 'path' of a type (line 163)
    path_5438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 13), os_5437, 'path')
    # Obtaining the member 'join' of a type (line 163)
    join_5439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 13), path_5438, 'join')
    # Calling join(args, kwargs) (line 163)
    join_call_result_5445 = invoke(stypy.reporting.localization.Localization(__file__, 163, 13), join_5439, *[output_dir_5440, result_mod_5443], **kwargs_5444)
    
    # Assigning a type to the variable 'd_file' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'd_file', join_call_result_5445)
    
    # Assigning a Tuple to a Name (line 164):
    
    # Assigning a Tuple to a Name (line 164):
    
    # Obtaining an instance of the builtin type 'tuple' (line 164)
    tuple_5446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 164)
    # Adding element type (line 164)
    # Getting the type of 'h_file' (line 164)
    h_file_5447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 15), 'h_file')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 15), tuple_5446, h_file_5447)
    # Adding element type (line 164)
    # Getting the type of 'c_file' (line 164)
    c_file_5448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 23), 'c_file')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 15), tuple_5446, c_file_5448)
    # Adding element type (line 164)
    # Getting the type of 'd_file' (line 164)
    d_file_5449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 31), 'd_file')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 15), tuple_5446, d_file_5449)
    
    # Assigning a type to the variable 'targets' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'targets', tuple_5446)
    
    # Assigning a Attribute to a Name (line 166):
    
    # Assigning a Attribute to a Name (line 166):
    # Getting the type of 'numpy_api' (line 166)
    numpy_api_5450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 14), 'numpy_api')
    # Obtaining the member 'multiarray_api' of a type (line 166)
    multiarray_api_5451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 14), numpy_api_5450, 'multiarray_api')
    # Assigning a type to the variable 'sources' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'sources', multiarray_api_5451)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'force' (line 168)
    force_5452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'force')
    # Applying the 'not' unary operator (line 168)
    result_not__5453 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 8), 'not', force_5452)
    
    
    
    # Call to should_rebuild(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'targets' (line 168)
    targets_5456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 48), 'targets', False)
    
    # Obtaining an instance of the builtin type 'list' (line 168)
    list_5457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 57), 'list')
    # Adding type elements to the builtin type 'list' instance (line 168)
    # Adding element type (line 168)
    # Getting the type of 'numpy_api' (line 168)
    numpy_api_5458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 58), 'numpy_api', False)
    # Obtaining the member '__file__' of a type (line 168)
    file___5459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 58), numpy_api_5458, '__file__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 57), list_5457, file___5459)
    # Adding element type (line 168)
    # Getting the type of '__file__' (line 168)
    file___5460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 78), '__file__', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 57), list_5457, file___5460)
    
    # Processing the call keyword arguments (line 168)
    kwargs_5461 = {}
    # Getting the type of 'genapi' (line 168)
    genapi_5454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 26), 'genapi', False)
    # Obtaining the member 'should_rebuild' of a type (line 168)
    should_rebuild_5455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 26), genapi_5454, 'should_rebuild')
    # Calling should_rebuild(args, kwargs) (line 168)
    should_rebuild_call_result_5462 = invoke(stypy.reporting.localization.Localization(__file__, 168, 26), should_rebuild_5455, *[targets_5456, list_5457], **kwargs_5461)
    
    # Applying the 'not' unary operator (line 168)
    result_not__5463 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 22), 'not', should_rebuild_call_result_5462)
    
    # Applying the binary operator 'and' (line 168)
    result_and_keyword_5464 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 8), 'and', result_not__5453, result_not__5463)
    
    # Testing the type of an if condition (line 168)
    if_condition_5465 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 4), result_and_keyword_5464)
    # Assigning a type to the variable 'if_condition_5465' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'if_condition_5465', if_condition_5465)
    # SSA begins for if statement (line 168)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'targets' (line 169)
    targets_5466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'targets')
    # Assigning a type to the variable 'stypy_return_type' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'stypy_return_type', targets_5466)
    # SSA branch for the else part of an if statement (line 168)
    module_type_store.open_ssa_branch('else')
    
    # Call to do_generate_api(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'targets' (line 171)
    targets_5468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 24), 'targets', False)
    # Getting the type of 'sources' (line 171)
    sources_5469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 33), 'sources', False)
    # Processing the call keyword arguments (line 171)
    kwargs_5470 = {}
    # Getting the type of 'do_generate_api' (line 171)
    do_generate_api_5467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'do_generate_api', False)
    # Calling do_generate_api(args, kwargs) (line 171)
    do_generate_api_call_result_5471 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), do_generate_api_5467, *[targets_5468, sources_5469], **kwargs_5470)
    
    # SSA join for if statement (line 168)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'targets' (line 173)
    targets_5472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 11), 'targets')
    # Assigning a type to the variable 'stypy_return_type' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'stypy_return_type', targets_5472)
    
    # ################# End of 'generate_api(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generate_api' in the type store
    # Getting the type of 'stypy_return_type' (line 158)
    stypy_return_type_5473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5473)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generate_api'
    return stypy_return_type_5473

# Assigning a type to the variable 'generate_api' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'generate_api', generate_api)

@norecursion
def do_generate_api(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'do_generate_api'
    module_type_store = module_type_store.open_function_context('do_generate_api', 175, 0, False)
    
    # Passed parameters checking function
    do_generate_api.stypy_localization = localization
    do_generate_api.stypy_type_of_self = None
    do_generate_api.stypy_type_store = module_type_store
    do_generate_api.stypy_function_name = 'do_generate_api'
    do_generate_api.stypy_param_names_list = ['targets', 'sources']
    do_generate_api.stypy_varargs_param_name = None
    do_generate_api.stypy_kwargs_param_name = None
    do_generate_api.stypy_call_defaults = defaults
    do_generate_api.stypy_call_varargs = varargs
    do_generate_api.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'do_generate_api', ['targets', 'sources'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'do_generate_api', localization, ['targets', 'sources'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'do_generate_api(...)' code ##################

    
    # Assigning a Subscript to a Name (line 176):
    
    # Assigning a Subscript to a Name (line 176):
    
    # Obtaining the type of the subscript
    int_5474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 26), 'int')
    # Getting the type of 'targets' (line 176)
    targets_5475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 18), 'targets')
    # Obtaining the member '__getitem__' of a type (line 176)
    getitem___5476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 18), targets_5475, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 176)
    subscript_call_result_5477 = invoke(stypy.reporting.localization.Localization(__file__, 176, 18), getitem___5476, int_5474)
    
    # Assigning a type to the variable 'header_file' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'header_file', subscript_call_result_5477)
    
    # Assigning a Subscript to a Name (line 177):
    
    # Assigning a Subscript to a Name (line 177):
    
    # Obtaining the type of the subscript
    int_5478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 21), 'int')
    # Getting the type of 'targets' (line 177)
    targets_5479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 13), 'targets')
    # Obtaining the member '__getitem__' of a type (line 177)
    getitem___5480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 13), targets_5479, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 177)
    subscript_call_result_5481 = invoke(stypy.reporting.localization.Localization(__file__, 177, 13), getitem___5480, int_5478)
    
    # Assigning a type to the variable 'c_file' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'c_file', subscript_call_result_5481)
    
    # Assigning a Subscript to a Name (line 178):
    
    # Assigning a Subscript to a Name (line 178):
    
    # Obtaining the type of the subscript
    int_5482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 23), 'int')
    # Getting the type of 'targets' (line 178)
    targets_5483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 15), 'targets')
    # Obtaining the member '__getitem__' of a type (line 178)
    getitem___5484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 15), targets_5483, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 178)
    subscript_call_result_5485 = invoke(stypy.reporting.localization.Localization(__file__, 178, 15), getitem___5484, int_5482)
    
    # Assigning a type to the variable 'doc_file' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'doc_file', subscript_call_result_5485)
    
    # Assigning a Subscript to a Name (line 180):
    
    # Assigning a Subscript to a Name (line 180):
    
    # Obtaining the type of the subscript
    int_5486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 26), 'int')
    # Getting the type of 'sources' (line 180)
    sources_5487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 18), 'sources')
    # Obtaining the member '__getitem__' of a type (line 180)
    getitem___5488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 18), sources_5487, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 180)
    subscript_call_result_5489 = invoke(stypy.reporting.localization.Localization(__file__, 180, 18), getitem___5488, int_5486)
    
    # Assigning a type to the variable 'global_vars' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'global_vars', subscript_call_result_5489)
    
    # Assigning a Subscript to a Name (line 181):
    
    # Assigning a Subscript to a Name (line 181):
    
    # Obtaining the type of the subscript
    int_5490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 33), 'int')
    # Getting the type of 'sources' (line 181)
    sources_5491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 25), 'sources')
    # Obtaining the member '__getitem__' of a type (line 181)
    getitem___5492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 25), sources_5491, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 181)
    subscript_call_result_5493 = invoke(stypy.reporting.localization.Localization(__file__, 181, 25), getitem___5492, int_5490)
    
    # Assigning a type to the variable 'scalar_bool_values' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'scalar_bool_values', subscript_call_result_5493)
    
    # Assigning a Subscript to a Name (line 182):
    
    # Assigning a Subscript to a Name (line 182):
    
    # Obtaining the type of the subscript
    int_5494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 24), 'int')
    # Getting the type of 'sources' (line 182)
    sources_5495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'sources')
    # Obtaining the member '__getitem__' of a type (line 182)
    getitem___5496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 16), sources_5495, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 182)
    subscript_call_result_5497 = invoke(stypy.reporting.localization.Localization(__file__, 182, 16), getitem___5496, int_5494)
    
    # Assigning a type to the variable 'types_api' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'types_api', subscript_call_result_5497)
    
    # Assigning a Subscript to a Name (line 183):
    
    # Assigning a Subscript to a Name (line 183):
    
    # Obtaining the type of the subscript
    int_5498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 31), 'int')
    # Getting the type of 'sources' (line 183)
    sources_5499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 23), 'sources')
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___5500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 23), sources_5499, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
    subscript_call_result_5501 = invoke(stypy.reporting.localization.Localization(__file__, 183, 23), getitem___5500, int_5498)
    
    # Assigning a type to the variable 'multiarray_funcs' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'multiarray_funcs', subscript_call_result_5501)
    
    # Assigning a Subscript to a Name (line 185):
    
    # Assigning a Subscript to a Name (line 185):
    
    # Obtaining the type of the subscript
    slice_5502 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 185, 21), None, None, None)
    # Getting the type of 'sources' (line 185)
    sources_5503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 21), 'sources')
    # Obtaining the member '__getitem__' of a type (line 185)
    getitem___5504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 21), sources_5503, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 185)
    subscript_call_result_5505 = invoke(stypy.reporting.localization.Localization(__file__, 185, 21), getitem___5504, slice_5502)
    
    # Assigning a type to the variable 'multiarray_api' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'multiarray_api', subscript_call_result_5505)
    
    # Assigning a List to a Name (line 187):
    
    # Assigning a List to a Name (line 187):
    
    # Obtaining an instance of the builtin type 'list' (line 187)
    list_5506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 187)
    
    # Assigning a type to the variable 'module_list' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'module_list', list_5506)
    
    # Assigning a List to a Name (line 188):
    
    # Assigning a List to a Name (line 188):
    
    # Obtaining an instance of the builtin type 'list' (line 188)
    list_5507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 188)
    
    # Assigning a type to the variable 'extension_list' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'extension_list', list_5507)
    
    # Assigning a List to a Name (line 189):
    
    # Assigning a List to a Name (line 189):
    
    # Obtaining an instance of the builtin type 'list' (line 189)
    list_5508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 189)
    
    # Assigning a type to the variable 'init_list' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'init_list', list_5508)
    
    # Assigning a Call to a Name (line 192):
    
    # Assigning a Call to a Name (line 192):
    
    # Call to merge_api_dicts(...): (line 192)
    # Processing the call arguments (line 192)
    # Getting the type of 'multiarray_api' (line 192)
    multiarray_api_5511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 50), 'multiarray_api', False)
    # Processing the call keyword arguments (line 192)
    kwargs_5512 = {}
    # Getting the type of 'genapi' (line 192)
    genapi_5509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 27), 'genapi', False)
    # Obtaining the member 'merge_api_dicts' of a type (line 192)
    merge_api_dicts_5510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 27), genapi_5509, 'merge_api_dicts')
    # Calling merge_api_dicts(args, kwargs) (line 192)
    merge_api_dicts_call_result_5513 = invoke(stypy.reporting.localization.Localization(__file__, 192, 27), merge_api_dicts_5510, *[multiarray_api_5511], **kwargs_5512)
    
    # Assigning a type to the variable 'multiarray_api_index' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'multiarray_api_index', merge_api_dicts_call_result_5513)
    
    # Call to check_api_dict(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'multiarray_api_index' (line 193)
    multiarray_api_index_5516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 26), 'multiarray_api_index', False)
    # Processing the call keyword arguments (line 193)
    kwargs_5517 = {}
    # Getting the type of 'genapi' (line 193)
    genapi_5514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'genapi', False)
    # Obtaining the member 'check_api_dict' of a type (line 193)
    check_api_dict_5515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 4), genapi_5514, 'check_api_dict')
    # Calling check_api_dict(args, kwargs) (line 193)
    check_api_dict_call_result_5518 = invoke(stypy.reporting.localization.Localization(__file__, 193, 4), check_api_dict_5515, *[multiarray_api_index_5516], **kwargs_5517)
    
    
    # Assigning a Call to a Name (line 195):
    
    # Assigning a Call to a Name (line 195):
    
    # Call to get_api_functions(...): (line 195)
    # Processing the call arguments (line 195)
    str_5521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 45), 'str', 'NUMPY_API')
    # Getting the type of 'multiarray_funcs' (line 196)
    multiarray_funcs_5522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 46), 'multiarray_funcs', False)
    # Processing the call keyword arguments (line 195)
    kwargs_5523 = {}
    # Getting the type of 'genapi' (line 195)
    genapi_5519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 20), 'genapi', False)
    # Obtaining the member 'get_api_functions' of a type (line 195)
    get_api_functions_5520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 20), genapi_5519, 'get_api_functions')
    # Calling get_api_functions(args, kwargs) (line 195)
    get_api_functions_call_result_5524 = invoke(stypy.reporting.localization.Localization(__file__, 195, 20), get_api_functions_5520, *[str_5521, multiarray_funcs_5522], **kwargs_5523)
    
    # Assigning a type to the variable 'numpyapi_list' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'numpyapi_list', get_api_functions_call_result_5524)
    
    # Assigning a Call to a Name (line 197):
    
    # Assigning a Call to a Name (line 197):
    
    # Call to order_dict(...): (line 197)
    # Processing the call arguments (line 197)
    # Getting the type of 'multiarray_funcs' (line 197)
    multiarray_funcs_5527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 42), 'multiarray_funcs', False)
    # Processing the call keyword arguments (line 197)
    kwargs_5528 = {}
    # Getting the type of 'genapi' (line 197)
    genapi_5525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 24), 'genapi', False)
    # Obtaining the member 'order_dict' of a type (line 197)
    order_dict_5526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 24), genapi_5525, 'order_dict')
    # Calling order_dict(args, kwargs) (line 197)
    order_dict_call_result_5529 = invoke(stypy.reporting.localization.Localization(__file__, 197, 24), order_dict_5526, *[multiarray_funcs_5527], **kwargs_5528)
    
    # Assigning a type to the variable 'ordered_funcs_api' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'ordered_funcs_api', order_dict_call_result_5529)
    
    # Assigning a Str to a Name (line 200):
    
    # Assigning a Str to a Name (line 200):
    str_5530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 15), 'str', 'PyArray_API')
    # Assigning a type to the variable 'api_name' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'api_name', str_5530)
    
    # Assigning a Dict to a Name (line 201):
    
    # Assigning a Dict to a Name (line 201):
    
    # Obtaining an instance of the builtin type 'dict' (line 201)
    dict_5531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 26), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 201)
    
    # Assigning a type to the variable 'multiarray_api_dict' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'multiarray_api_dict', dict_5531)
    
    # Getting the type of 'numpyapi_list' (line 202)
    numpyapi_list_5532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 13), 'numpyapi_list')
    # Testing the type of a for loop iterable (line 202)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 202, 4), numpyapi_list_5532)
    # Getting the type of the for loop variable (line 202)
    for_loop_var_5533 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 202, 4), numpyapi_list_5532)
    # Assigning a type to the variable 'f' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'f', for_loop_var_5533)
    # SSA begins for a for statement (line 202)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Attribute to a Name (line 203):
    
    # Assigning a Attribute to a Name (line 203):
    # Getting the type of 'f' (line 203)
    f_5534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'f')
    # Obtaining the member 'name' of a type (line 203)
    name_5535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 15), f_5534, 'name')
    # Assigning a type to the variable 'name' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'name', name_5535)
    
    # Assigning a Subscript to a Name (line 204):
    
    # Assigning a Subscript to a Name (line 204):
    
    # Obtaining the type of the subscript
    int_5536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 39), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 204)
    name_5537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 33), 'name')
    # Getting the type of 'multiarray_funcs' (line 204)
    multiarray_funcs_5538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'multiarray_funcs')
    # Obtaining the member '__getitem__' of a type (line 204)
    getitem___5539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 16), multiarray_funcs_5538, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 204)
    subscript_call_result_5540 = invoke(stypy.reporting.localization.Localization(__file__, 204, 16), getitem___5539, name_5537)
    
    # Obtaining the member '__getitem__' of a type (line 204)
    getitem___5541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 16), subscript_call_result_5540, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 204)
    subscript_call_result_5542 = invoke(stypy.reporting.localization.Localization(__file__, 204, 16), getitem___5541, int_5536)
    
    # Assigning a type to the variable 'index' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'index', subscript_call_result_5542)
    
    # Assigning a Subscript to a Name (line 205):
    
    # Assigning a Subscript to a Name (line 205):
    
    # Obtaining the type of the subscript
    int_5543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 45), 'int')
    slice_5544 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 205, 22), int_5543, None, None)
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 205)
    name_5545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 39), 'name')
    # Getting the type of 'multiarray_funcs' (line 205)
    multiarray_funcs_5546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 22), 'multiarray_funcs')
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___5547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 22), multiarray_funcs_5546, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_5548 = invoke(stypy.reporting.localization.Localization(__file__, 205, 22), getitem___5547, name_5545)
    
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___5549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 22), subscript_call_result_5548, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_5550 = invoke(stypy.reporting.localization.Localization(__file__, 205, 22), getitem___5549, slice_5544)
    
    # Assigning a type to the variable 'annotations' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'annotations', subscript_call_result_5550)
    
    # Assigning a Call to a Subscript (line 206):
    
    # Assigning a Call to a Subscript (line 206):
    
    # Call to FunctionApi(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'f' (line 206)
    f_5552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 50), 'f', False)
    # Obtaining the member 'name' of a type (line 206)
    name_5553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 50), f_5552, 'name')
    # Getting the type of 'index' (line 206)
    index_5554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 58), 'index', False)
    # Getting the type of 'annotations' (line 206)
    annotations_5555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 65), 'annotations', False)
    # Getting the type of 'f' (line 207)
    f_5556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 50), 'f', False)
    # Obtaining the member 'return_type' of a type (line 207)
    return_type_5557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 50), f_5556, 'return_type')
    # Getting the type of 'f' (line 208)
    f_5558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 50), 'f', False)
    # Obtaining the member 'args' of a type (line 208)
    args_5559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 50), f_5558, 'args')
    # Getting the type of 'api_name' (line 208)
    api_name_5560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 58), 'api_name', False)
    # Processing the call keyword arguments (line 206)
    kwargs_5561 = {}
    # Getting the type of 'FunctionApi' (line 206)
    FunctionApi_5551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 38), 'FunctionApi', False)
    # Calling FunctionApi(args, kwargs) (line 206)
    FunctionApi_call_result_5562 = invoke(stypy.reporting.localization.Localization(__file__, 206, 38), FunctionApi_5551, *[name_5553, index_5554, annotations_5555, return_type_5557, args_5559, api_name_5560], **kwargs_5561)
    
    # Getting the type of 'multiarray_api_dict' (line 206)
    multiarray_api_dict_5563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'multiarray_api_dict')
    # Getting the type of 'f' (line 206)
    f_5564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 28), 'f')
    # Obtaining the member 'name' of a type (line 206)
    name_5565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 28), f_5564, 'name')
    # Storing an element on a container (line 206)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 8), multiarray_api_dict_5563, (name_5565, FunctionApi_call_result_5562))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to items(...): (line 210)
    # Processing the call keyword arguments (line 210)
    kwargs_5568 = {}
    # Getting the type of 'global_vars' (line 210)
    global_vars_5566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 21), 'global_vars', False)
    # Obtaining the member 'items' of a type (line 210)
    items_5567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 21), global_vars_5566, 'items')
    # Calling items(args, kwargs) (line 210)
    items_call_result_5569 = invoke(stypy.reporting.localization.Localization(__file__, 210, 21), items_5567, *[], **kwargs_5568)
    
    # Testing the type of a for loop iterable (line 210)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 210, 4), items_call_result_5569)
    # Getting the type of the for loop variable (line 210)
    for_loop_var_5570 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 210, 4), items_call_result_5569)
    # Assigning a type to the variable 'name' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 4), for_loop_var_5570))
    # Assigning a type to the variable 'val' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 4), for_loop_var_5570))
    # SSA begins for a for statement (line 210)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Tuple (line 211):
    
    # Assigning a Subscript to a Name (line 211):
    
    # Obtaining the type of the subscript
    int_5571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 8), 'int')
    # Getting the type of 'val' (line 211)
    val_5572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 22), 'val')
    # Obtaining the member '__getitem__' of a type (line 211)
    getitem___5573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), val_5572, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 211)
    subscript_call_result_5574 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), getitem___5573, int_5571)
    
    # Assigning a type to the variable 'tuple_var_assignment_5406' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'tuple_var_assignment_5406', subscript_call_result_5574)
    
    # Assigning a Subscript to a Name (line 211):
    
    # Obtaining the type of the subscript
    int_5575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 8), 'int')
    # Getting the type of 'val' (line 211)
    val_5576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 22), 'val')
    # Obtaining the member '__getitem__' of a type (line 211)
    getitem___5577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), val_5576, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 211)
    subscript_call_result_5578 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), getitem___5577, int_5575)
    
    # Assigning a type to the variable 'tuple_var_assignment_5407' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'tuple_var_assignment_5407', subscript_call_result_5578)
    
    # Assigning a Name to a Name (line 211):
    # Getting the type of 'tuple_var_assignment_5406' (line 211)
    tuple_var_assignment_5406_5579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'tuple_var_assignment_5406')
    # Assigning a type to the variable 'index' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'index', tuple_var_assignment_5406_5579)
    
    # Assigning a Name to a Name (line 211):
    # Getting the type of 'tuple_var_assignment_5407' (line 211)
    tuple_var_assignment_5407_5580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'tuple_var_assignment_5407')
    # Assigning a type to the variable 'type' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 15), 'type', tuple_var_assignment_5407_5580)
    
    # Assigning a Call to a Subscript (line 212):
    
    # Assigning a Call to a Subscript (line 212):
    
    # Call to GlobalVarApi(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'name' (line 212)
    name_5582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 49), 'name', False)
    # Getting the type of 'index' (line 212)
    index_5583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 55), 'index', False)
    # Getting the type of 'type' (line 212)
    type_5584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 62), 'type', False)
    # Getting the type of 'api_name' (line 212)
    api_name_5585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 68), 'api_name', False)
    # Processing the call keyword arguments (line 212)
    kwargs_5586 = {}
    # Getting the type of 'GlobalVarApi' (line 212)
    GlobalVarApi_5581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), 'GlobalVarApi', False)
    # Calling GlobalVarApi(args, kwargs) (line 212)
    GlobalVarApi_call_result_5587 = invoke(stypy.reporting.localization.Localization(__file__, 212, 36), GlobalVarApi_5581, *[name_5582, index_5583, type_5584, api_name_5585], **kwargs_5586)
    
    # Getting the type of 'multiarray_api_dict' (line 212)
    multiarray_api_dict_5588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'multiarray_api_dict')
    # Getting the type of 'name' (line 212)
    name_5589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 28), 'name')
    # Storing an element on a container (line 212)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 8), multiarray_api_dict_5588, (name_5589, GlobalVarApi_call_result_5587))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to items(...): (line 214)
    # Processing the call keyword arguments (line 214)
    kwargs_5592 = {}
    # Getting the type of 'scalar_bool_values' (line 214)
    scalar_bool_values_5590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 21), 'scalar_bool_values', False)
    # Obtaining the member 'items' of a type (line 214)
    items_5591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 21), scalar_bool_values_5590, 'items')
    # Calling items(args, kwargs) (line 214)
    items_call_result_5593 = invoke(stypy.reporting.localization.Localization(__file__, 214, 21), items_5591, *[], **kwargs_5592)
    
    # Testing the type of a for loop iterable (line 214)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 214, 4), items_call_result_5593)
    # Getting the type of the for loop variable (line 214)
    for_loop_var_5594 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 214, 4), items_call_result_5593)
    # Assigning a type to the variable 'name' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 4), for_loop_var_5594))
    # Assigning a type to the variable 'val' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 4), for_loop_var_5594))
    # SSA begins for a for statement (line 214)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 215):
    
    # Assigning a Subscript to a Name (line 215):
    
    # Obtaining the type of the subscript
    int_5595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 20), 'int')
    # Getting the type of 'val' (line 215)
    val_5596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 16), 'val')
    # Obtaining the member '__getitem__' of a type (line 215)
    getitem___5597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 16), val_5596, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 215)
    subscript_call_result_5598 = invoke(stypy.reporting.localization.Localization(__file__, 215, 16), getitem___5597, int_5595)
    
    # Assigning a type to the variable 'index' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'index', subscript_call_result_5598)
    
    # Assigning a Call to a Subscript (line 216):
    
    # Assigning a Call to a Subscript (line 216):
    
    # Call to BoolValuesApi(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'name' (line 216)
    name_5600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 50), 'name', False)
    # Getting the type of 'index' (line 216)
    index_5601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 56), 'index', False)
    # Getting the type of 'api_name' (line 216)
    api_name_5602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 63), 'api_name', False)
    # Processing the call keyword arguments (line 216)
    kwargs_5603 = {}
    # Getting the type of 'BoolValuesApi' (line 216)
    BoolValuesApi_5599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 36), 'BoolValuesApi', False)
    # Calling BoolValuesApi(args, kwargs) (line 216)
    BoolValuesApi_call_result_5604 = invoke(stypy.reporting.localization.Localization(__file__, 216, 36), BoolValuesApi_5599, *[name_5600, index_5601, api_name_5602], **kwargs_5603)
    
    # Getting the type of 'multiarray_api_dict' (line 216)
    multiarray_api_dict_5605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'multiarray_api_dict')
    # Getting the type of 'name' (line 216)
    name_5606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 28), 'name')
    # Storing an element on a container (line 216)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 8), multiarray_api_dict_5605, (name_5606, BoolValuesApi_call_result_5604))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to items(...): (line 218)
    # Processing the call keyword arguments (line 218)
    kwargs_5609 = {}
    # Getting the type of 'types_api' (line 218)
    types_api_5607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 21), 'types_api', False)
    # Obtaining the member 'items' of a type (line 218)
    items_5608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 21), types_api_5607, 'items')
    # Calling items(args, kwargs) (line 218)
    items_call_result_5610 = invoke(stypy.reporting.localization.Localization(__file__, 218, 21), items_5608, *[], **kwargs_5609)
    
    # Testing the type of a for loop iterable (line 218)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 218, 4), items_call_result_5610)
    # Getting the type of the for loop variable (line 218)
    for_loop_var_5611 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 218, 4), items_call_result_5610)
    # Assigning a type to the variable 'name' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 4), for_loop_var_5611))
    # Assigning a type to the variable 'val' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 4), for_loop_var_5611))
    # SSA begins for a for statement (line 218)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 219):
    
    # Assigning a Subscript to a Name (line 219):
    
    # Obtaining the type of the subscript
    int_5612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 20), 'int')
    # Getting the type of 'val' (line 219)
    val_5613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'val')
    # Obtaining the member '__getitem__' of a type (line 219)
    getitem___5614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 16), val_5613, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 219)
    subscript_call_result_5615 = invoke(stypy.reporting.localization.Localization(__file__, 219, 16), getitem___5614, int_5612)
    
    # Assigning a type to the variable 'index' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'index', subscript_call_result_5615)
    
    # Assigning a Call to a Subscript (line 220):
    
    # Assigning a Call to a Subscript (line 220):
    
    # Call to TypeApi(...): (line 220)
    # Processing the call arguments (line 220)
    # Getting the type of 'name' (line 220)
    name_5617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 44), 'name', False)
    # Getting the type of 'index' (line 220)
    index_5618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 50), 'index', False)
    str_5619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 57), 'str', 'PyTypeObject')
    # Getting the type of 'api_name' (line 220)
    api_name_5620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 73), 'api_name', False)
    # Processing the call keyword arguments (line 220)
    kwargs_5621 = {}
    # Getting the type of 'TypeApi' (line 220)
    TypeApi_5616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 36), 'TypeApi', False)
    # Calling TypeApi(args, kwargs) (line 220)
    TypeApi_call_result_5622 = invoke(stypy.reporting.localization.Localization(__file__, 220, 36), TypeApi_5616, *[name_5617, index_5618, str_5619, api_name_5620], **kwargs_5621)
    
    # Getting the type of 'multiarray_api_dict' (line 220)
    multiarray_api_dict_5623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'multiarray_api_dict')
    # Getting the type of 'name' (line 220)
    name_5624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 28), 'name')
    # Storing an element on a container (line 220)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 8), multiarray_api_dict_5623, (name_5624, TypeApi_call_result_5622))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 222)
    # Processing the call arguments (line 222)
    # Getting the type of 'multiarray_api_dict' (line 222)
    multiarray_api_dict_5626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 11), 'multiarray_api_dict', False)
    # Processing the call keyword arguments (line 222)
    kwargs_5627 = {}
    # Getting the type of 'len' (line 222)
    len_5625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 7), 'len', False)
    # Calling len(args, kwargs) (line 222)
    len_call_result_5628 = invoke(stypy.reporting.localization.Localization(__file__, 222, 7), len_5625, *[multiarray_api_dict_5626], **kwargs_5627)
    
    
    # Call to len(...): (line 222)
    # Processing the call arguments (line 222)
    # Getting the type of 'multiarray_api_index' (line 222)
    multiarray_api_index_5630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 39), 'multiarray_api_index', False)
    # Processing the call keyword arguments (line 222)
    kwargs_5631 = {}
    # Getting the type of 'len' (line 222)
    len_5629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 35), 'len', False)
    # Calling len(args, kwargs) (line 222)
    len_call_result_5632 = invoke(stypy.reporting.localization.Localization(__file__, 222, 35), len_5629, *[multiarray_api_index_5630], **kwargs_5631)
    
    # Applying the binary operator '!=' (line 222)
    result_ne_5633 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 7), '!=', len_call_result_5628, len_call_result_5632)
    
    # Testing the type of an if condition (line 222)
    if_condition_5634 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 222, 4), result_ne_5633)
    # Assigning a type to the variable 'if_condition_5634' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'if_condition_5634', if_condition_5634)
    # SSA begins for if statement (line 222)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to AssertionError(...): (line 223)
    # Processing the call arguments (line 223)
    str_5636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 29), 'str', 'Multiarray API size mismatch %d %d')
    
    # Obtaining an instance of the builtin type 'tuple' (line 224)
    tuple_5637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 224)
    # Adding element type (line 224)
    
    # Call to len(...): (line 224)
    # Processing the call arguments (line 224)
    # Getting the type of 'multiarray_api_dict' (line 224)
    multiarray_api_dict_5639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 29), 'multiarray_api_dict', False)
    # Processing the call keyword arguments (line 224)
    kwargs_5640 = {}
    # Getting the type of 'len' (line 224)
    len_5638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 25), 'len', False)
    # Calling len(args, kwargs) (line 224)
    len_call_result_5641 = invoke(stypy.reporting.localization.Localization(__file__, 224, 25), len_5638, *[multiarray_api_dict_5639], **kwargs_5640)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 25), tuple_5637, len_call_result_5641)
    # Adding element type (line 224)
    
    # Call to len(...): (line 224)
    # Processing the call arguments (line 224)
    # Getting the type of 'multiarray_api_index' (line 224)
    multiarray_api_index_5643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 55), 'multiarray_api_index', False)
    # Processing the call keyword arguments (line 224)
    kwargs_5644 = {}
    # Getting the type of 'len' (line 224)
    len_5642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 51), 'len', False)
    # Calling len(args, kwargs) (line 224)
    len_call_result_5645 = invoke(stypy.reporting.localization.Localization(__file__, 224, 51), len_5642, *[multiarray_api_index_5643], **kwargs_5644)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 25), tuple_5637, len_call_result_5645)
    
    # Applying the binary operator '%' (line 223)
    result_mod_5646 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 29), '%', str_5636, tuple_5637)
    
    # Processing the call keyword arguments (line 223)
    kwargs_5647 = {}
    # Getting the type of 'AssertionError' (line 223)
    AssertionError_5635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 14), 'AssertionError', False)
    # Calling AssertionError(args, kwargs) (line 223)
    AssertionError_call_result_5648 = invoke(stypy.reporting.localization.Localization(__file__, 223, 14), AssertionError_5635, *[result_mod_5646], **kwargs_5647)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 223, 8), AssertionError_call_result_5648, 'raise parameter', BaseException)
    # SSA join for if statement (line 222)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 226):
    
    # Assigning a List to a Name (line 226):
    
    # Obtaining an instance of the builtin type 'list' (line 226)
    list_5649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 226)
    
    # Assigning a type to the variable 'extension_list' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'extension_list', list_5649)
    
    
    # Call to order_dict(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'multiarray_api_index' (line 227)
    multiarray_api_index_5652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 41), 'multiarray_api_index', False)
    # Processing the call keyword arguments (line 227)
    kwargs_5653 = {}
    # Getting the type of 'genapi' (line 227)
    genapi_5650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 23), 'genapi', False)
    # Obtaining the member 'order_dict' of a type (line 227)
    order_dict_5651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 23), genapi_5650, 'order_dict')
    # Calling order_dict(args, kwargs) (line 227)
    order_dict_call_result_5654 = invoke(stypy.reporting.localization.Localization(__file__, 227, 23), order_dict_5651, *[multiarray_api_index_5652], **kwargs_5653)
    
    # Testing the type of a for loop iterable (line 227)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 227, 4), order_dict_call_result_5654)
    # Getting the type of the for loop variable (line 227)
    for_loop_var_5655 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 227, 4), order_dict_call_result_5654)
    # Assigning a type to the variable 'name' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 4), for_loop_var_5655))
    # Assigning a type to the variable 'index' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'index', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 4), for_loop_var_5655))
    # SSA begins for a for statement (line 227)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 228):
    
    # Assigning a Subscript to a Name (line 228):
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 228)
    name_5656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 39), 'name')
    # Getting the type of 'multiarray_api_dict' (line 228)
    multiarray_api_dict_5657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 19), 'multiarray_api_dict')
    # Obtaining the member '__getitem__' of a type (line 228)
    getitem___5658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 19), multiarray_api_dict_5657, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 228)
    subscript_call_result_5659 = invoke(stypy.reporting.localization.Localization(__file__, 228, 19), getitem___5658, name_5656)
    
    # Assigning a type to the variable 'api_item' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'api_item', subscript_call_result_5659)
    
    # Call to append(...): (line 229)
    # Processing the call arguments (line 229)
    
    # Call to define_from_array_api_string(...): (line 229)
    # Processing the call keyword arguments (line 229)
    kwargs_5664 = {}
    # Getting the type of 'api_item' (line 229)
    api_item_5662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 30), 'api_item', False)
    # Obtaining the member 'define_from_array_api_string' of a type (line 229)
    define_from_array_api_string_5663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 30), api_item_5662, 'define_from_array_api_string')
    # Calling define_from_array_api_string(args, kwargs) (line 229)
    define_from_array_api_string_call_result_5665 = invoke(stypy.reporting.localization.Localization(__file__, 229, 30), define_from_array_api_string_5663, *[], **kwargs_5664)
    
    # Processing the call keyword arguments (line 229)
    kwargs_5666 = {}
    # Getting the type of 'extension_list' (line 229)
    extension_list_5660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'extension_list', False)
    # Obtaining the member 'append' of a type (line 229)
    append_5661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), extension_list_5660, 'append')
    # Calling append(args, kwargs) (line 229)
    append_call_result_5667 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), append_5661, *[define_from_array_api_string_call_result_5665], **kwargs_5666)
    
    
    # Call to append(...): (line 230)
    # Processing the call arguments (line 230)
    
    # Call to array_api_define(...): (line 230)
    # Processing the call keyword arguments (line 230)
    kwargs_5672 = {}
    # Getting the type of 'api_item' (line 230)
    api_item_5670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 25), 'api_item', False)
    # Obtaining the member 'array_api_define' of a type (line 230)
    array_api_define_5671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 25), api_item_5670, 'array_api_define')
    # Calling array_api_define(args, kwargs) (line 230)
    array_api_define_call_result_5673 = invoke(stypy.reporting.localization.Localization(__file__, 230, 25), array_api_define_5671, *[], **kwargs_5672)
    
    # Processing the call keyword arguments (line 230)
    kwargs_5674 = {}
    # Getting the type of 'init_list' (line 230)
    init_list_5668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'init_list', False)
    # Obtaining the member 'append' of a type (line 230)
    append_5669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), init_list_5668, 'append')
    # Calling append(args, kwargs) (line 230)
    append_call_result_5675 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), append_5669, *[array_api_define_call_result_5673], **kwargs_5674)
    
    
    # Call to append(...): (line 231)
    # Processing the call arguments (line 231)
    
    # Call to internal_define(...): (line 231)
    # Processing the call keyword arguments (line 231)
    kwargs_5680 = {}
    # Getting the type of 'api_item' (line 231)
    api_item_5678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 27), 'api_item', False)
    # Obtaining the member 'internal_define' of a type (line 231)
    internal_define_5679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 27), api_item_5678, 'internal_define')
    # Calling internal_define(args, kwargs) (line 231)
    internal_define_call_result_5681 = invoke(stypy.reporting.localization.Localization(__file__, 231, 27), internal_define_5679, *[], **kwargs_5680)
    
    # Processing the call keyword arguments (line 231)
    kwargs_5682 = {}
    # Getting the type of 'module_list' (line 231)
    module_list_5676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'module_list', False)
    # Obtaining the member 'append' of a type (line 231)
    append_5677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), module_list_5676, 'append')
    # Calling append(args, kwargs) (line 231)
    append_call_result_5683 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), append_5677, *[internal_define_call_result_5681], **kwargs_5682)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 234):
    
    # Assigning a Call to a Name (line 234):
    
    # Call to open(...): (line 234)
    # Processing the call arguments (line 234)
    # Getting the type of 'header_file' (line 234)
    header_file_5685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 15), 'header_file', False)
    str_5686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 28), 'str', 'w')
    # Processing the call keyword arguments (line 234)
    kwargs_5687 = {}
    # Getting the type of 'open' (line 234)
    open_5684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 10), 'open', False)
    # Calling open(args, kwargs) (line 234)
    open_call_result_5688 = invoke(stypy.reporting.localization.Localization(__file__, 234, 10), open_5684, *[header_file_5685, str_5686], **kwargs_5687)
    
    # Assigning a type to the variable 'fid' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'fid', open_call_result_5688)
    
    # Assigning a BinOp to a Name (line 235):
    
    # Assigning a BinOp to a Name (line 235):
    # Getting the type of 'h_template' (line 235)
    h_template_5689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'h_template')
    
    # Obtaining an instance of the builtin type 'tuple' (line 235)
    tuple_5690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 235)
    # Adding element type (line 235)
    
    # Call to join(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'module_list' (line 235)
    module_list_5693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 32), 'module_list', False)
    # Processing the call keyword arguments (line 235)
    kwargs_5694 = {}
    str_5691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 22), 'str', '\n')
    # Obtaining the member 'join' of a type (line 235)
    join_5692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 22), str_5691, 'join')
    # Calling join(args, kwargs) (line 235)
    join_call_result_5695 = invoke(stypy.reporting.localization.Localization(__file__, 235, 22), join_5692, *[module_list_5693], **kwargs_5694)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 22), tuple_5690, join_call_result_5695)
    # Adding element type (line 235)
    
    # Call to join(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'extension_list' (line 235)
    extension_list_5698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 56), 'extension_list', False)
    # Processing the call keyword arguments (line 235)
    kwargs_5699 = {}
    str_5696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 46), 'str', '\n')
    # Obtaining the member 'join' of a type (line 235)
    join_5697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 46), str_5696, 'join')
    # Calling join(args, kwargs) (line 235)
    join_call_result_5700 = invoke(stypy.reporting.localization.Localization(__file__, 235, 46), join_5697, *[extension_list_5698], **kwargs_5699)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 22), tuple_5690, join_call_result_5700)
    
    # Applying the binary operator '%' (line 235)
    result_mod_5701 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 8), '%', h_template_5689, tuple_5690)
    
    # Assigning a type to the variable 's' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 's', result_mod_5701)
    
    # Call to write(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 's' (line 236)
    s_5704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 14), 's', False)
    # Processing the call keyword arguments (line 236)
    kwargs_5705 = {}
    # Getting the type of 'fid' (line 236)
    fid_5702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'fid', False)
    # Obtaining the member 'write' of a type (line 236)
    write_5703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 4), fid_5702, 'write')
    # Calling write(args, kwargs) (line 236)
    write_call_result_5706 = invoke(stypy.reporting.localization.Localization(__file__, 236, 4), write_5703, *[s_5704], **kwargs_5705)
    
    
    # Call to close(...): (line 237)
    # Processing the call keyword arguments (line 237)
    kwargs_5709 = {}
    # Getting the type of 'fid' (line 237)
    fid_5707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'fid', False)
    # Obtaining the member 'close' of a type (line 237)
    close_5708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 4), fid_5707, 'close')
    # Calling close(args, kwargs) (line 237)
    close_call_result_5710 = invoke(stypy.reporting.localization.Localization(__file__, 237, 4), close_5708, *[], **kwargs_5709)
    
    
    # Assigning a Call to a Name (line 240):
    
    # Assigning a Call to a Name (line 240):
    
    # Call to open(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'c_file' (line 240)
    c_file_5712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 15), 'c_file', False)
    str_5713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 23), 'str', 'w')
    # Processing the call keyword arguments (line 240)
    kwargs_5714 = {}
    # Getting the type of 'open' (line 240)
    open_5711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 10), 'open', False)
    # Calling open(args, kwargs) (line 240)
    open_call_result_5715 = invoke(stypy.reporting.localization.Localization(__file__, 240, 10), open_5711, *[c_file_5712, str_5713], **kwargs_5714)
    
    # Assigning a type to the variable 'fid' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'fid', open_call_result_5715)
    
    # Assigning a BinOp to a Name (line 241):
    
    # Assigning a BinOp to a Name (line 241):
    # Getting the type of 'c_template' (line 241)
    c_template_5716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'c_template')
    
    # Call to join(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'init_list' (line 241)
    init_list_5719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 32), 'init_list', False)
    # Processing the call keyword arguments (line 241)
    kwargs_5720 = {}
    str_5717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 21), 'str', ',\n')
    # Obtaining the member 'join' of a type (line 241)
    join_5718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 21), str_5717, 'join')
    # Calling join(args, kwargs) (line 241)
    join_call_result_5721 = invoke(stypy.reporting.localization.Localization(__file__, 241, 21), join_5718, *[init_list_5719], **kwargs_5720)
    
    # Applying the binary operator '%' (line 241)
    result_mod_5722 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 8), '%', c_template_5716, join_call_result_5721)
    
    # Assigning a type to the variable 's' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 's', result_mod_5722)
    
    # Call to write(...): (line 242)
    # Processing the call arguments (line 242)
    # Getting the type of 's' (line 242)
    s_5725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 14), 's', False)
    # Processing the call keyword arguments (line 242)
    kwargs_5726 = {}
    # Getting the type of 'fid' (line 242)
    fid_5723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'fid', False)
    # Obtaining the member 'write' of a type (line 242)
    write_5724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 4), fid_5723, 'write')
    # Calling write(args, kwargs) (line 242)
    write_call_result_5727 = invoke(stypy.reporting.localization.Localization(__file__, 242, 4), write_5724, *[s_5725], **kwargs_5726)
    
    
    # Call to close(...): (line 243)
    # Processing the call keyword arguments (line 243)
    kwargs_5730 = {}
    # Getting the type of 'fid' (line 243)
    fid_5728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'fid', False)
    # Obtaining the member 'close' of a type (line 243)
    close_5729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 4), fid_5728, 'close')
    # Calling close(args, kwargs) (line 243)
    close_call_result_5731 = invoke(stypy.reporting.localization.Localization(__file__, 243, 4), close_5729, *[], **kwargs_5730)
    
    
    # Assigning a Call to a Name (line 246):
    
    # Assigning a Call to a Name (line 246):
    
    # Call to open(...): (line 246)
    # Processing the call arguments (line 246)
    # Getting the type of 'doc_file' (line 246)
    doc_file_5733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 15), 'doc_file', False)
    str_5734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 25), 'str', 'w')
    # Processing the call keyword arguments (line 246)
    kwargs_5735 = {}
    # Getting the type of 'open' (line 246)
    open_5732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 10), 'open', False)
    # Calling open(args, kwargs) (line 246)
    open_call_result_5736 = invoke(stypy.reporting.localization.Localization(__file__, 246, 10), open_5732, *[doc_file_5733, str_5734], **kwargs_5735)
    
    # Assigning a type to the variable 'fid' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'fid', open_call_result_5736)
    
    # Call to write(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'c_api_header' (line 247)
    c_api_header_5739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 14), 'c_api_header', False)
    # Processing the call keyword arguments (line 247)
    kwargs_5740 = {}
    # Getting the type of 'fid' (line 247)
    fid_5737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'fid', False)
    # Obtaining the member 'write' of a type (line 247)
    write_5738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 4), fid_5737, 'write')
    # Calling write(args, kwargs) (line 247)
    write_call_result_5741 = invoke(stypy.reporting.localization.Localization(__file__, 247, 4), write_5738, *[c_api_header_5739], **kwargs_5740)
    
    
    # Getting the type of 'numpyapi_list' (line 248)
    numpyapi_list_5742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'numpyapi_list')
    # Testing the type of a for loop iterable (line 248)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 248, 4), numpyapi_list_5742)
    # Getting the type of the for loop variable (line 248)
    for_loop_var_5743 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 248, 4), numpyapi_list_5742)
    # Assigning a type to the variable 'func' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'func', for_loop_var_5743)
    # SSA begins for a for statement (line 248)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to write(...): (line 249)
    # Processing the call arguments (line 249)
    
    # Call to to_ReST(...): (line 249)
    # Processing the call keyword arguments (line 249)
    kwargs_5748 = {}
    # Getting the type of 'func' (line 249)
    func_5746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 18), 'func', False)
    # Obtaining the member 'to_ReST' of a type (line 249)
    to_ReST_5747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 18), func_5746, 'to_ReST')
    # Calling to_ReST(args, kwargs) (line 249)
    to_ReST_call_result_5749 = invoke(stypy.reporting.localization.Localization(__file__, 249, 18), to_ReST_5747, *[], **kwargs_5748)
    
    # Processing the call keyword arguments (line 249)
    kwargs_5750 = {}
    # Getting the type of 'fid' (line 249)
    fid_5744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'fid', False)
    # Obtaining the member 'write' of a type (line 249)
    write_5745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), fid_5744, 'write')
    # Calling write(args, kwargs) (line 249)
    write_call_result_5751 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), write_5745, *[to_ReST_call_result_5749], **kwargs_5750)
    
    
    # Call to write(...): (line 250)
    # Processing the call arguments (line 250)
    str_5754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 18), 'str', '\n\n')
    # Processing the call keyword arguments (line 250)
    kwargs_5755 = {}
    # Getting the type of 'fid' (line 250)
    fid_5752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'fid', False)
    # Obtaining the member 'write' of a type (line 250)
    write_5753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), fid_5752, 'write')
    # Calling write(args, kwargs) (line 250)
    write_call_result_5756 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), write_5753, *[str_5754], **kwargs_5755)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to close(...): (line 251)
    # Processing the call keyword arguments (line 251)
    kwargs_5759 = {}
    # Getting the type of 'fid' (line 251)
    fid_5757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'fid', False)
    # Obtaining the member 'close' of a type (line 251)
    close_5758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 4), fid_5757, 'close')
    # Calling close(args, kwargs) (line 251)
    close_call_result_5760 = invoke(stypy.reporting.localization.Localization(__file__, 251, 4), close_5758, *[], **kwargs_5759)
    
    # Getting the type of 'targets' (line 253)
    targets_5761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 11), 'targets')
    # Assigning a type to the variable 'stypy_return_type' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'stypy_return_type', targets_5761)
    
    # ################# End of 'do_generate_api(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'do_generate_api' in the type store
    # Getting the type of 'stypy_return_type' (line 175)
    stypy_return_type_5762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5762)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'do_generate_api'
    return stypy_return_type_5762

# Assigning a type to the variable 'do_generate_api' (line 175)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'do_generate_api', do_generate_api)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
