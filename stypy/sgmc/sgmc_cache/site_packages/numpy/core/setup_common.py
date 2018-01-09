
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: # Code common to build tools
4: import sys
5: import warnings
6: import copy
7: import binascii
8: 
9: from numpy.distutils.misc_util import mingw32
10: 
11: 
12: #-------------------
13: # Versioning support
14: #-------------------
15: # How to change C_API_VERSION ?
16: #   - increase C_API_VERSION value
17: #   - record the hash for the new C API with the script cversions.py
18: #   and add the hash to cversions.txt
19: # The hash values are used to remind developers when the C API number was not
20: # updated - generates a MismatchCAPIWarning warning which is turned into an
21: # exception for released version.
22: 
23: # Binary compatibility version number. This number is increased whenever the
24: # C-API is changed such that binary compatibility is broken, i.e. whenever a
25: # recompile of extension modules is needed.
26: C_ABI_VERSION = 0x01000009
27: 
28: # Minor API version.  This number is increased whenever a change is made to the
29: # C-API -- whether it breaks binary compatibility or not.  Some changes, such
30: # as adding a function pointer to the end of the function table, can be made
31: # without breaking binary compatibility.  In this case, only the C_API_VERSION
32: # (*not* C_ABI_VERSION) would be increased.  Whenever binary compatibility is
33: # broken, both C_API_VERSION and C_ABI_VERSION should be increased.
34: #
35: # 0x00000008 - 1.7.x
36: # 0x00000009 - 1.8.x
37: # 0x00000009 - 1.9.x
38: # 0x0000000a - 1.10.x
39: # 0x0000000a - 1.11.x
40: C_API_VERSION = 0x0000000a
41: 
42: class MismatchCAPIWarning(Warning):
43:     pass
44: 
45: def is_released(config):
46:     '''Return True if a released version of numpy is detected.'''
47:     from distutils.version import LooseVersion
48: 
49:     v = config.get_version('../version.py')
50:     if v is None:
51:         raise ValueError("Could not get version")
52:     pv = LooseVersion(vstring=v).version
53:     if len(pv) > 3:
54:         return False
55:     return True
56: 
57: def get_api_versions(apiversion, codegen_dir):
58:     '''
59:     Return current C API checksum and the recorded checksum.
60: 
61:     Return current C API checksum and the recorded checksum for the given
62:     version of the C API version.
63: 
64:     '''
65:     # Compute the hash of the current API as defined in the .txt files in
66:     # code_generators
67:     sys.path.insert(0, codegen_dir)
68:     try:
69:         m = __import__('genapi')
70:         numpy_api = __import__('numpy_api')
71:         curapi_hash = m.fullapi_hash(numpy_api.full_api)
72:         apis_hash = m.get_versions_hash()
73:     finally:
74:         del sys.path[0]
75: 
76:     return curapi_hash, apis_hash[apiversion]
77: 
78: def check_api_version(apiversion, codegen_dir):
79:     '''Emits a MismacthCAPIWarning if the C API version needs updating.'''
80:     curapi_hash, api_hash = get_api_versions(apiversion, codegen_dir)
81: 
82:     # If different hash, it means that the api .txt files in
83:     # codegen_dir have been updated without the API version being
84:     # updated. Any modification in those .txt files should be reflected
85:     # in the api and eventually abi versions.
86:     # To compute the checksum of the current API, use
87:     # code_generators/cversions.py script
88:     if not curapi_hash == api_hash:
89:         msg = ("API mismatch detected, the C API version "
90:                "numbers have to be updated. Current C api version is %d, "
91:                "with checksum %s, but recorded checksum for C API version %d in "
92:                "codegen_dir/cversions.txt is %s. If functions were added in the "
93:                "C API, you have to update C_API_VERSION  in %s."
94:                )
95:         warnings.warn(msg % (apiversion, curapi_hash, apiversion, api_hash,
96:                              __file__),
97:                       MismatchCAPIWarning)
98: # Mandatory functions: if not found, fail the build
99: MANDATORY_FUNCS = ["sin", "cos", "tan", "sinh", "cosh", "tanh", "fabs",
100:         "floor", "ceil", "sqrt", "log10", "log", "exp", "asin",
101:         "acos", "atan", "fmod", 'modf', 'frexp', 'ldexp']
102: 
103: # Standard functions which may not be available and for which we have a
104: # replacement implementation. Note that some of these are C99 functions.
105: OPTIONAL_STDFUNCS = ["expm1", "log1p", "acosh", "asinh", "atanh",
106:         "rint", "trunc", "exp2", "log2", "hypot", "atan2", "pow",
107:         "copysign", "nextafter", "ftello", "fseeko",
108:         "strtoll", "strtoull", "cbrt", "strtold_l", "fallocate"]
109: 
110: 
111: OPTIONAL_HEADERS = [
112: # sse headers only enabled automatically on amd64/x32 builds
113:                 "xmmintrin.h",  # SSE
114:                 "emmintrin.h",  # SSE2
115:                 "features.h",  # for glibc version linux
116: ]
117: 
118: # optional gcc compiler builtins and their call arguments and optional a
119: # required header
120: # call arguments are required as the compiler will do strict signature checking
121: OPTIONAL_INTRINSICS = [("__builtin_isnan", '5.'),
122:                        ("__builtin_isinf", '5.'),
123:                        ("__builtin_isfinite", '5.'),
124:                        ("__builtin_bswap32", '5u'),
125:                        ("__builtin_bswap64", '5u'),
126:                        ("__builtin_expect", '5, 0'),
127:                        ("__builtin_mul_overflow", '5, 5, (int*)5'),
128:                        ("_mm_load_ps", '(float*)0', "xmmintrin.h"),  # SSE
129:                        ("_mm_prefetch", '(float*)0, _MM_HINT_NTA',
130:                         "xmmintrin.h"),  # SSE
131:                        ("_mm_load_pd", '(double*)0', "emmintrin.h"),  # SSE2
132:                        ("__builtin_prefetch", "(float*)0, 0, 3"),
133:                        ]
134: 
135: # function attributes
136: # tested via "int %s %s(void *);" % (attribute, name)
137: # function name will be converted to HAVE_<upper-case-name> preprocessor macro
138: OPTIONAL_FUNCTION_ATTRIBUTES = [('__attribute__((optimize("unroll-loops")))',
139:                                 'attribute_optimize_unroll_loops'),
140:                                 ('__attribute__((optimize("O3")))',
141:                                  'attribute_optimize_opt_3'),
142:                                 ('__attribute__((nonnull (1)))',
143:                                  'attribute_nonnull'),
144:                                 ]
145: 
146: # variable attributes tested via "int %s a" % attribute
147: OPTIONAL_VARIABLE_ATTRIBUTES = ["__thread", "__declspec(thread)"]
148: 
149: # Subset of OPTIONAL_STDFUNCS which may alreay have HAVE_* defined by Python.h
150: OPTIONAL_STDFUNCS_MAYBE = [
151:     "expm1", "log1p", "acosh", "atanh", "asinh", "hypot", "copysign",
152:     "ftello", "fseeko"
153:     ]
154: 
155: # C99 functions: float and long double versions
156: C99_FUNCS = [
157:     "sin", "cos", "tan", "sinh", "cosh", "tanh", "fabs", "floor", "ceil",
158:     "rint", "trunc", "sqrt", "log10", "log", "log1p", "exp", "expm1",
159:     "asin", "acos", "atan", "asinh", "acosh", "atanh", "hypot", "atan2",
160:     "pow", "fmod", "modf", 'frexp', 'ldexp', "exp2", "log2", "copysign",
161:     "nextafter", "cbrt"
162:     ]
163: C99_FUNCS_SINGLE = [f + 'f' for f in C99_FUNCS]
164: C99_FUNCS_EXTENDED = [f + 'l' for f in C99_FUNCS]
165: C99_COMPLEX_TYPES = [
166:     'complex double', 'complex float', 'complex long double'
167:     ]
168: C99_COMPLEX_FUNCS = [
169:     "cabs", "cacos", "cacosh", "carg", "casin", "casinh", "catan",
170:     "catanh", "ccos", "ccosh", "cexp", "cimag", "clog", "conj", "cpow",
171:     "cproj", "creal", "csin", "csinh", "csqrt", "ctan", "ctanh"
172:     ]
173: 
174: def fname2def(name):
175:     return "HAVE_%s" % name.upper()
176: 
177: def sym2def(symbol):
178:     define = symbol.replace(' ', '')
179:     return define.upper()
180: 
181: def type2def(symbol):
182:     define = symbol.replace(' ', '_')
183:     return define.upper()
184: 
185: # Code to detect long double representation taken from MPFR m4 macro
186: def check_long_double_representation(cmd):
187:     cmd._check_compiler()
188:     body = LONG_DOUBLE_REPRESENTATION_SRC % {'type': 'long double'}
189: 
190:     # Disable whole program optimization (the default on vs2015, with python 3.5+)
191:     # which generates intermediary object files and prevents checking the
192:     # float representation.
193:     if sys.platform == "win32" and not mingw32():
194:         try:
195:             cmd.compiler.compile_options.remove("/GL")
196:         except (AttributeError, ValueError):
197:             pass
198: 
199:     # We need to use _compile because we need the object filename
200:     src, obj = cmd._compile(body, None, None, 'c')
201:     try:
202:         ltype = long_double_representation(pyod(obj))
203:         return ltype
204:     except ValueError:
205:         # try linking to support CC="gcc -flto" or icc -ipo
206:         # struct needs to be volatile so it isn't optimized away
207:         body = body.replace('struct', 'volatile struct')
208:         body += "int main(void) { return 0; }\n"
209:         src, obj = cmd._compile(body, None, None, 'c')
210:         cmd.temp_files.append("_configtest")
211:         cmd.compiler.link_executable([obj], "_configtest")
212:         ltype = long_double_representation(pyod("_configtest"))
213:         return ltype
214:     finally:
215:         cmd._clean()
216: 
217: LONG_DOUBLE_REPRESENTATION_SRC = r'''
218: /* "before" is 16 bytes to ensure there's no padding between it and "x".
219:  *    We're not expecting any "long double" bigger than 16 bytes or with
220:  *       alignment requirements stricter than 16 bytes.  */
221: typedef %(type)s test_type;
222: 
223: struct {
224:         char         before[16];
225:         test_type    x;
226:         char         after[8];
227: } foo = {
228:         { '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
229:           '\001', '\043', '\105', '\147', '\211', '\253', '\315', '\357' },
230:         -123456789.0,
231:         { '\376', '\334', '\272', '\230', '\166', '\124', '\062', '\020' }
232: };
233: '''
234: 
235: def pyod(filename):
236:     '''Python implementation of the od UNIX utility (od -b, more exactly).
237: 
238:     Parameters
239:     ----------
240:     filename : str
241:         name of the file to get the dump from.
242: 
243:     Returns
244:     -------
245:     out : seq
246:         list of lines of od output
247: 
248:     Note
249:     ----
250:     We only implement enough to get the necessary information for long double
251:     representation, this is not intended as a compatible replacement for od.
252:     '''
253:     def _pyod2():
254:         out = []
255: 
256:         fid = open(filename, 'rb')
257:         try:
258:             yo = [int(oct(int(binascii.b2a_hex(o), 16))) for o in fid.read()]
259:             for i in range(0, len(yo), 16):
260:                 line = ['%07d' % int(oct(i))]
261:                 line.extend(['%03d' % c for c in yo[i:i+16]])
262:                 out.append(" ".join(line))
263:             return out
264:         finally:
265:             fid.close()
266: 
267:     def _pyod3():
268:         out = []
269: 
270:         fid = open(filename, 'rb')
271:         try:
272:             yo2 = [oct(o)[2:] for o in fid.read()]
273:             for i in range(0, len(yo2), 16):
274:                 line = ['%07d' % int(oct(i)[2:])]
275:                 line.extend(['%03d' % int(c) for c in yo2[i:i+16]])
276:                 out.append(" ".join(line))
277:             return out
278:         finally:
279:             fid.close()
280: 
281:     if sys.version_info[0] < 3:
282:         return _pyod2()
283:     else:
284:         return _pyod3()
285: 
286: _BEFORE_SEQ = ['000', '000', '000', '000', '000', '000', '000', '000',
287:               '001', '043', '105', '147', '211', '253', '315', '357']
288: _AFTER_SEQ = ['376', '334', '272', '230', '166', '124', '062', '020']
289: 
290: _IEEE_DOUBLE_BE = ['301', '235', '157', '064', '124', '000', '000', '000']
291: _IEEE_DOUBLE_LE = _IEEE_DOUBLE_BE[::-1]
292: _INTEL_EXTENDED_12B = ['000', '000', '000', '000', '240', '242', '171', '353',
293:                        '031', '300', '000', '000']
294: _INTEL_EXTENDED_16B = ['000', '000', '000', '000', '240', '242', '171', '353',
295:                        '031', '300', '000', '000', '000', '000', '000', '000']
296: _MOTOROLA_EXTENDED_12B = ['300', '031', '000', '000', '353', '171',
297:                           '242', '240', '000', '000', '000', '000']
298: _IEEE_QUAD_PREC_BE = ['300', '031', '326', '363', '105', '100', '000', '000',
299:                       '000', '000', '000', '000', '000', '000', '000', '000']
300: _IEEE_QUAD_PREC_LE = _IEEE_QUAD_PREC_BE[::-1]
301: _DOUBLE_DOUBLE_BE = (['301', '235', '157', '064', '124', '000', '000', '000'] +
302:                      ['000'] * 8)
303: _DOUBLE_DOUBLE_LE = (['000', '000', '000', '124', '064', '157', '235', '301'] +
304:                      ['000'] * 8)
305: 
306: def long_double_representation(lines):
307:     '''Given a binary dump as given by GNU od -b, look for long double
308:     representation.'''
309: 
310:     # Read contains a list of 32 items, each item is a byte (in octal
311:     # representation, as a string). We 'slide' over the output until read is of
312:     # the form before_seq + content + after_sequence, where content is the long double
313:     # representation:
314:     #  - content is 12 bytes: 80 bits Intel representation
315:     #  - content is 16 bytes: 80 bits Intel representation (64 bits) or quad precision
316:     #  - content is 8 bytes: same as double (not implemented yet)
317:     read = [''] * 32
318:     saw = None
319:     for line in lines:
320:         # we skip the first word, as od -b output an index at the beginning of
321:         # each line
322:         for w in line.split()[1:]:
323:             read.pop(0)
324:             read.append(w)
325: 
326:             # If the end of read is equal to the after_sequence, read contains
327:             # the long double
328:             if read[-8:] == _AFTER_SEQ:
329:                 saw = copy.copy(read)
330:                 if read[:12] == _BEFORE_SEQ[4:]:
331:                     if read[12:-8] == _INTEL_EXTENDED_12B:
332:                         return 'INTEL_EXTENDED_12_BYTES_LE'
333:                     if read[12:-8] == _MOTOROLA_EXTENDED_12B:
334:                         return 'MOTOROLA_EXTENDED_12_BYTES_BE'
335:                 elif read[:8] == _BEFORE_SEQ[8:]:
336:                     if read[8:-8] == _INTEL_EXTENDED_16B:
337:                         return 'INTEL_EXTENDED_16_BYTES_LE'
338:                     elif read[8:-8] == _IEEE_QUAD_PREC_BE:
339:                         return 'IEEE_QUAD_BE'
340:                     elif read[8:-8] == _IEEE_QUAD_PREC_LE:
341:                         return 'IEEE_QUAD_LE'
342:                     elif read[8:-8] == _DOUBLE_DOUBLE_BE:
343:                         return 'DOUBLE_DOUBLE_BE'
344:                     elif read[8:-8] == _DOUBLE_DOUBLE_LE:
345:                         return 'DOUBLE_DOUBLE_LE'
346:                 elif read[:16] == _BEFORE_SEQ:
347:                     if read[16:-8] == _IEEE_DOUBLE_LE:
348:                         return 'IEEE_DOUBLE_LE'
349:                     elif read[16:-8] == _IEEE_DOUBLE_BE:
350:                         return 'IEEE_DOUBLE_BE'
351: 
352:     if saw is not None:
353:         raise ValueError("Unrecognized format (%s)" % saw)
354:     else:
355:         # We never detected the after_sequence
356:         raise ValueError("Could not lock sequences (%s)" % saw)
357: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import warnings' statement (line 5)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import copy' statement (line 6)
import copy

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'copy', copy, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import binascii' statement (line 7)
import binascii

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'binascii', binascii, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.distutils.misc_util import mingw32' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_17518 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.distutils.misc_util')

if (type(import_17518) is not StypyTypeError):

    if (import_17518 != 'pyd_module'):
        __import__(import_17518)
        sys_modules_17519 = sys.modules[import_17518]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.distutils.misc_util', sys_modules_17519.module_type_store, module_type_store, ['mingw32'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_17519, sys_modules_17519.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import mingw32

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.distutils.misc_util', None, module_type_store, ['mingw32'], [mingw32])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.distutils.misc_util', import_17518)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')


# Assigning a Num to a Name (line 26):

# Assigning a Num to a Name (line 26):
int_17520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 16), 'int')
# Assigning a type to the variable 'C_ABI_VERSION' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'C_ABI_VERSION', int_17520)

# Assigning a Num to a Name (line 40):

# Assigning a Num to a Name (line 40):
int_17521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 16), 'int')
# Assigning a type to the variable 'C_API_VERSION' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'C_API_VERSION', int_17521)
# Declaration of the 'MismatchCAPIWarning' class
# Getting the type of 'Warning' (line 42)
Warning_17522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 26), 'Warning')

class MismatchCAPIWarning(Warning_17522, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 42, 0, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MismatchCAPIWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MismatchCAPIWarning' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'MismatchCAPIWarning', MismatchCAPIWarning)

@norecursion
def is_released(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_released'
    module_type_store = module_type_store.open_function_context('is_released', 45, 0, False)
    
    # Passed parameters checking function
    is_released.stypy_localization = localization
    is_released.stypy_type_of_self = None
    is_released.stypy_type_store = module_type_store
    is_released.stypy_function_name = 'is_released'
    is_released.stypy_param_names_list = ['config']
    is_released.stypy_varargs_param_name = None
    is_released.stypy_kwargs_param_name = None
    is_released.stypy_call_defaults = defaults
    is_released.stypy_call_varargs = varargs
    is_released.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_released', ['config'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_released', localization, ['config'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_released(...)' code ##################

    str_17523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 4), 'str', 'Return True if a released version of numpy is detected.')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 47, 4))
    
    # 'from distutils.version import LooseVersion' statement (line 47)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
    import_17524 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 47, 4), 'distutils.version')

    if (type(import_17524) is not StypyTypeError):

        if (import_17524 != 'pyd_module'):
            __import__(import_17524)
            sys_modules_17525 = sys.modules[import_17524]
            import_from_module(stypy.reporting.localization.Localization(__file__, 47, 4), 'distutils.version', sys_modules_17525.module_type_store, module_type_store, ['LooseVersion'])
            nest_module(stypy.reporting.localization.Localization(__file__, 47, 4), __file__, sys_modules_17525, sys_modules_17525.module_type_store, module_type_store)
        else:
            from distutils.version import LooseVersion

            import_from_module(stypy.reporting.localization.Localization(__file__, 47, 4), 'distutils.version', None, module_type_store, ['LooseVersion'], [LooseVersion])

    else:
        # Assigning a type to the variable 'distutils.version' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'distutils.version', import_17524)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')
    
    
    # Assigning a Call to a Name (line 49):
    
    # Assigning a Call to a Name (line 49):
    
    # Call to get_version(...): (line 49)
    # Processing the call arguments (line 49)
    str_17528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 27), 'str', '../version.py')
    # Processing the call keyword arguments (line 49)
    kwargs_17529 = {}
    # Getting the type of 'config' (line 49)
    config_17526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'config', False)
    # Obtaining the member 'get_version' of a type (line 49)
    get_version_17527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), config_17526, 'get_version')
    # Calling get_version(args, kwargs) (line 49)
    get_version_call_result_17530 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), get_version_17527, *[str_17528], **kwargs_17529)
    
    # Assigning a type to the variable 'v' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'v', get_version_call_result_17530)
    
    # Type idiom detected: calculating its left and rigth part (line 50)
    # Getting the type of 'v' (line 50)
    v_17531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 7), 'v')
    # Getting the type of 'None' (line 50)
    None_17532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'None')
    
    (may_be_17533, more_types_in_union_17534) = may_be_none(v_17531, None_17532)

    if may_be_17533:

        if more_types_in_union_17534:
            # Runtime conditional SSA (line 50)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 51)
        # Processing the call arguments (line 51)
        str_17536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 25), 'str', 'Could not get version')
        # Processing the call keyword arguments (line 51)
        kwargs_17537 = {}
        # Getting the type of 'ValueError' (line 51)
        ValueError_17535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 51)
        ValueError_call_result_17538 = invoke(stypy.reporting.localization.Localization(__file__, 51, 14), ValueError_17535, *[str_17536], **kwargs_17537)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 51, 8), ValueError_call_result_17538, 'raise parameter', BaseException)

        if more_types_in_union_17534:
            # SSA join for if statement (line 50)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Attribute to a Name (line 52):
    
    # Assigning a Attribute to a Name (line 52):
    
    # Call to LooseVersion(...): (line 52)
    # Processing the call keyword arguments (line 52)
    # Getting the type of 'v' (line 52)
    v_17540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 30), 'v', False)
    keyword_17541 = v_17540
    kwargs_17542 = {'vstring': keyword_17541}
    # Getting the type of 'LooseVersion' (line 52)
    LooseVersion_17539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 9), 'LooseVersion', False)
    # Calling LooseVersion(args, kwargs) (line 52)
    LooseVersion_call_result_17543 = invoke(stypy.reporting.localization.Localization(__file__, 52, 9), LooseVersion_17539, *[], **kwargs_17542)
    
    # Obtaining the member 'version' of a type (line 52)
    version_17544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 9), LooseVersion_call_result_17543, 'version')
    # Assigning a type to the variable 'pv' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'pv', version_17544)
    
    
    
    # Call to len(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'pv' (line 53)
    pv_17546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'pv', False)
    # Processing the call keyword arguments (line 53)
    kwargs_17547 = {}
    # Getting the type of 'len' (line 53)
    len_17545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 7), 'len', False)
    # Calling len(args, kwargs) (line 53)
    len_call_result_17548 = invoke(stypy.reporting.localization.Localization(__file__, 53, 7), len_17545, *[pv_17546], **kwargs_17547)
    
    int_17549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 17), 'int')
    # Applying the binary operator '>' (line 53)
    result_gt_17550 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 7), '>', len_call_result_17548, int_17549)
    
    # Testing the type of an if condition (line 53)
    if_condition_17551 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 4), result_gt_17550)
    # Assigning a type to the variable 'if_condition_17551' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'if_condition_17551', if_condition_17551)
    # SSA begins for if statement (line 53)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'False' (line 54)
    False_17552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'stypy_return_type', False_17552)
    # SSA join for if statement (line 53)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'True' (line 55)
    True_17553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type', True_17553)
    
    # ################# End of 'is_released(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_released' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_17554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17554)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_released'
    return stypy_return_type_17554

# Assigning a type to the variable 'is_released' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'is_released', is_released)

@norecursion
def get_api_versions(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_api_versions'
    module_type_store = module_type_store.open_function_context('get_api_versions', 57, 0, False)
    
    # Passed parameters checking function
    get_api_versions.stypy_localization = localization
    get_api_versions.stypy_type_of_self = None
    get_api_versions.stypy_type_store = module_type_store
    get_api_versions.stypy_function_name = 'get_api_versions'
    get_api_versions.stypy_param_names_list = ['apiversion', 'codegen_dir']
    get_api_versions.stypy_varargs_param_name = None
    get_api_versions.stypy_kwargs_param_name = None
    get_api_versions.stypy_call_defaults = defaults
    get_api_versions.stypy_call_varargs = varargs
    get_api_versions.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_api_versions', ['apiversion', 'codegen_dir'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_api_versions', localization, ['apiversion', 'codegen_dir'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_api_versions(...)' code ##################

    str_17555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, (-1)), 'str', '\n    Return current C API checksum and the recorded checksum.\n\n    Return current C API checksum and the recorded checksum for the given\n    version of the C API version.\n\n    ')
    
    # Call to insert(...): (line 67)
    # Processing the call arguments (line 67)
    int_17559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 20), 'int')
    # Getting the type of 'codegen_dir' (line 67)
    codegen_dir_17560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 23), 'codegen_dir', False)
    # Processing the call keyword arguments (line 67)
    kwargs_17561 = {}
    # Getting the type of 'sys' (line 67)
    sys_17556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'sys', False)
    # Obtaining the member 'path' of a type (line 67)
    path_17557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 4), sys_17556, 'path')
    # Obtaining the member 'insert' of a type (line 67)
    insert_17558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 4), path_17557, 'insert')
    # Calling insert(args, kwargs) (line 67)
    insert_call_result_17562 = invoke(stypy.reporting.localization.Localization(__file__, 67, 4), insert_17558, *[int_17559, codegen_dir_17560], **kwargs_17561)
    
    
    # Try-finally block (line 68)
    
    # Assigning a Call to a Name (line 69):
    
    # Assigning a Call to a Name (line 69):
    
    # Call to __import__(...): (line 69)
    # Processing the call arguments (line 69)
    str_17564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 23), 'str', 'genapi')
    # Processing the call keyword arguments (line 69)
    kwargs_17565 = {}
    # Getting the type of '__import__' (line 69)
    import___17563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), '__import__', False)
    # Calling __import__(args, kwargs) (line 69)
    import___call_result_17566 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), import___17563, *[str_17564], **kwargs_17565)
    
    # Assigning a type to the variable 'm' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'm', import___call_result_17566)
    
    # Assigning a Call to a Name (line 70):
    
    # Assigning a Call to a Name (line 70):
    
    # Call to __import__(...): (line 70)
    # Processing the call arguments (line 70)
    str_17568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 31), 'str', 'numpy_api')
    # Processing the call keyword arguments (line 70)
    kwargs_17569 = {}
    # Getting the type of '__import__' (line 70)
    import___17567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 20), '__import__', False)
    # Calling __import__(args, kwargs) (line 70)
    import___call_result_17570 = invoke(stypy.reporting.localization.Localization(__file__, 70, 20), import___17567, *[str_17568], **kwargs_17569)
    
    # Assigning a type to the variable 'numpy_api' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'numpy_api', import___call_result_17570)
    
    # Assigning a Call to a Name (line 71):
    
    # Assigning a Call to a Name (line 71):
    
    # Call to fullapi_hash(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'numpy_api' (line 71)
    numpy_api_17573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 37), 'numpy_api', False)
    # Obtaining the member 'full_api' of a type (line 71)
    full_api_17574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 37), numpy_api_17573, 'full_api')
    # Processing the call keyword arguments (line 71)
    kwargs_17575 = {}
    # Getting the type of 'm' (line 71)
    m_17571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'm', False)
    # Obtaining the member 'fullapi_hash' of a type (line 71)
    fullapi_hash_17572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 22), m_17571, 'fullapi_hash')
    # Calling fullapi_hash(args, kwargs) (line 71)
    fullapi_hash_call_result_17576 = invoke(stypy.reporting.localization.Localization(__file__, 71, 22), fullapi_hash_17572, *[full_api_17574], **kwargs_17575)
    
    # Assigning a type to the variable 'curapi_hash' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'curapi_hash', fullapi_hash_call_result_17576)
    
    # Assigning a Call to a Name (line 72):
    
    # Assigning a Call to a Name (line 72):
    
    # Call to get_versions_hash(...): (line 72)
    # Processing the call keyword arguments (line 72)
    kwargs_17579 = {}
    # Getting the type of 'm' (line 72)
    m_17577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'm', False)
    # Obtaining the member 'get_versions_hash' of a type (line 72)
    get_versions_hash_17578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 20), m_17577, 'get_versions_hash')
    # Calling get_versions_hash(args, kwargs) (line 72)
    get_versions_hash_call_result_17580 = invoke(stypy.reporting.localization.Localization(__file__, 72, 20), get_versions_hash_17578, *[], **kwargs_17579)
    
    # Assigning a type to the variable 'apis_hash' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'apis_hash', get_versions_hash_call_result_17580)
    
    # finally branch of the try-finally block (line 68)
    # Deleting a member
    # Getting the type of 'sys' (line 74)
    sys_17581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'sys')
    # Obtaining the member 'path' of a type (line 74)
    path_17582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 12), sys_17581, 'path')
    
    # Obtaining the type of the subscript
    int_17583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 21), 'int')
    # Getting the type of 'sys' (line 74)
    sys_17584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'sys')
    # Obtaining the member 'path' of a type (line 74)
    path_17585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 12), sys_17584, 'path')
    # Obtaining the member '__getitem__' of a type (line 74)
    getitem___17586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 12), path_17585, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 74)
    subscript_call_result_17587 = invoke(stypy.reporting.localization.Localization(__file__, 74, 12), getitem___17586, int_17583)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 8), path_17582, subscript_call_result_17587)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 76)
    tuple_17588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 76)
    # Adding element type (line 76)
    # Getting the type of 'curapi_hash' (line 76)
    curapi_hash_17589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'curapi_hash')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 11), tuple_17588, curapi_hash_17589)
    # Adding element type (line 76)
    
    # Obtaining the type of the subscript
    # Getting the type of 'apiversion' (line 76)
    apiversion_17590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 34), 'apiversion')
    # Getting the type of 'apis_hash' (line 76)
    apis_hash_17591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'apis_hash')
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___17592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 24), apis_hash_17591, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_17593 = invoke(stypy.reporting.localization.Localization(__file__, 76, 24), getitem___17592, apiversion_17590)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 11), tuple_17588, subscript_call_result_17593)
    
    # Assigning a type to the variable 'stypy_return_type' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'stypy_return_type', tuple_17588)
    
    # ################# End of 'get_api_versions(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_api_versions' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_17594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17594)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_api_versions'
    return stypy_return_type_17594

# Assigning a type to the variable 'get_api_versions' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'get_api_versions', get_api_versions)

@norecursion
def check_api_version(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_api_version'
    module_type_store = module_type_store.open_function_context('check_api_version', 78, 0, False)
    
    # Passed parameters checking function
    check_api_version.stypy_localization = localization
    check_api_version.stypy_type_of_self = None
    check_api_version.stypy_type_store = module_type_store
    check_api_version.stypy_function_name = 'check_api_version'
    check_api_version.stypy_param_names_list = ['apiversion', 'codegen_dir']
    check_api_version.stypy_varargs_param_name = None
    check_api_version.stypy_kwargs_param_name = None
    check_api_version.stypy_call_defaults = defaults
    check_api_version.stypy_call_varargs = varargs
    check_api_version.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_api_version', ['apiversion', 'codegen_dir'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_api_version', localization, ['apiversion', 'codegen_dir'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_api_version(...)' code ##################

    str_17595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 4), 'str', 'Emits a MismacthCAPIWarning if the C API version needs updating.')
    
    # Assigning a Call to a Tuple (line 80):
    
    # Assigning a Call to a Name:
    
    # Call to get_api_versions(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'apiversion' (line 80)
    apiversion_17597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 45), 'apiversion', False)
    # Getting the type of 'codegen_dir' (line 80)
    codegen_dir_17598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 57), 'codegen_dir', False)
    # Processing the call keyword arguments (line 80)
    kwargs_17599 = {}
    # Getting the type of 'get_api_versions' (line 80)
    get_api_versions_17596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 28), 'get_api_versions', False)
    # Calling get_api_versions(args, kwargs) (line 80)
    get_api_versions_call_result_17600 = invoke(stypy.reporting.localization.Localization(__file__, 80, 28), get_api_versions_17596, *[apiversion_17597, codegen_dir_17598], **kwargs_17599)
    
    # Assigning a type to the variable 'call_assignment_17509' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'call_assignment_17509', get_api_versions_call_result_17600)
    
    # Assigning a Call to a Name (line 80):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_17603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 4), 'int')
    # Processing the call keyword arguments
    kwargs_17604 = {}
    # Getting the type of 'call_assignment_17509' (line 80)
    call_assignment_17509_17601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'call_assignment_17509', False)
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___17602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 4), call_assignment_17509_17601, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_17605 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___17602, *[int_17603], **kwargs_17604)
    
    # Assigning a type to the variable 'call_assignment_17510' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'call_assignment_17510', getitem___call_result_17605)
    
    # Assigning a Name to a Name (line 80):
    # Getting the type of 'call_assignment_17510' (line 80)
    call_assignment_17510_17606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'call_assignment_17510')
    # Assigning a type to the variable 'curapi_hash' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'curapi_hash', call_assignment_17510_17606)
    
    # Assigning a Call to a Name (line 80):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_17609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 4), 'int')
    # Processing the call keyword arguments
    kwargs_17610 = {}
    # Getting the type of 'call_assignment_17509' (line 80)
    call_assignment_17509_17607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'call_assignment_17509', False)
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___17608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 4), call_assignment_17509_17607, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_17611 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___17608, *[int_17609], **kwargs_17610)
    
    # Assigning a type to the variable 'call_assignment_17511' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'call_assignment_17511', getitem___call_result_17611)
    
    # Assigning a Name to a Name (line 80):
    # Getting the type of 'call_assignment_17511' (line 80)
    call_assignment_17511_17612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'call_assignment_17511')
    # Assigning a type to the variable 'api_hash' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'api_hash', call_assignment_17511_17612)
    
    
    
    # Getting the type of 'curapi_hash' (line 88)
    curapi_hash_17613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'curapi_hash')
    # Getting the type of 'api_hash' (line 88)
    api_hash_17614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 26), 'api_hash')
    # Applying the binary operator '==' (line 88)
    result_eq_17615 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 11), '==', curapi_hash_17613, api_hash_17614)
    
    # Applying the 'not' unary operator (line 88)
    result_not__17616 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 7), 'not', result_eq_17615)
    
    # Testing the type of an if condition (line 88)
    if_condition_17617 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 4), result_not__17616)
    # Assigning a type to the variable 'if_condition_17617' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'if_condition_17617', if_condition_17617)
    # SSA begins for if statement (line 88)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 89):
    
    # Assigning a Str to a Name (line 89):
    str_17618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 15), 'str', 'API mismatch detected, the C API version numbers have to be updated. Current C api version is %d, with checksum %s, but recorded checksum for C API version %d in codegen_dir/cversions.txt is %s. If functions were added in the C API, you have to update C_API_VERSION  in %s.')
    # Assigning a type to the variable 'msg' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'msg', str_17618)
    
    # Call to warn(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'msg' (line 95)
    msg_17621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 22), 'msg', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 95)
    tuple_17622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 95)
    # Adding element type (line 95)
    # Getting the type of 'apiversion' (line 95)
    apiversion_17623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 29), 'apiversion', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 29), tuple_17622, apiversion_17623)
    # Adding element type (line 95)
    # Getting the type of 'curapi_hash' (line 95)
    curapi_hash_17624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 41), 'curapi_hash', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 29), tuple_17622, curapi_hash_17624)
    # Adding element type (line 95)
    # Getting the type of 'apiversion' (line 95)
    apiversion_17625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 54), 'apiversion', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 29), tuple_17622, apiversion_17625)
    # Adding element type (line 95)
    # Getting the type of 'api_hash' (line 95)
    api_hash_17626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 66), 'api_hash', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 29), tuple_17622, api_hash_17626)
    # Adding element type (line 95)
    # Getting the type of '__file__' (line 96)
    file___17627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 29), '__file__', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 29), tuple_17622, file___17627)
    
    # Applying the binary operator '%' (line 95)
    result_mod_17628 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 22), '%', msg_17621, tuple_17622)
    
    # Getting the type of 'MismatchCAPIWarning' (line 97)
    MismatchCAPIWarning_17629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 22), 'MismatchCAPIWarning', False)
    # Processing the call keyword arguments (line 95)
    kwargs_17630 = {}
    # Getting the type of 'warnings' (line 95)
    warnings_17619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 95)
    warn_17620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), warnings_17619, 'warn')
    # Calling warn(args, kwargs) (line 95)
    warn_call_result_17631 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), warn_17620, *[result_mod_17628, MismatchCAPIWarning_17629], **kwargs_17630)
    
    # SSA join for if statement (line 88)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_api_version(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_api_version' in the type store
    # Getting the type of 'stypy_return_type' (line 78)
    stypy_return_type_17632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17632)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_api_version'
    return stypy_return_type_17632

# Assigning a type to the variable 'check_api_version' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'check_api_version', check_api_version)

# Assigning a List to a Name (line 99):

# Assigning a List to a Name (line 99):

# Obtaining an instance of the builtin type 'list' (line 99)
list_17633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 99)
# Adding element type (line 99)
str_17634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 19), 'str', 'sin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_17633, str_17634)
# Adding element type (line 99)
str_17635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 26), 'str', 'cos')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_17633, str_17635)
# Adding element type (line 99)
str_17636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 33), 'str', 'tan')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_17633, str_17636)
# Adding element type (line 99)
str_17637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 40), 'str', 'sinh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_17633, str_17637)
# Adding element type (line 99)
str_17638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 48), 'str', 'cosh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_17633, str_17638)
# Adding element type (line 99)
str_17639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 56), 'str', 'tanh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_17633, str_17639)
# Adding element type (line 99)
str_17640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 64), 'str', 'fabs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_17633, str_17640)
# Adding element type (line 99)
str_17641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 8), 'str', 'floor')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_17633, str_17641)
# Adding element type (line 99)
str_17642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 17), 'str', 'ceil')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_17633, str_17642)
# Adding element type (line 99)
str_17643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 25), 'str', 'sqrt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_17633, str_17643)
# Adding element type (line 99)
str_17644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 33), 'str', 'log10')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_17633, str_17644)
# Adding element type (line 99)
str_17645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 42), 'str', 'log')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_17633, str_17645)
# Adding element type (line 99)
str_17646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 49), 'str', 'exp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_17633, str_17646)
# Adding element type (line 99)
str_17647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 56), 'str', 'asin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_17633, str_17647)
# Adding element type (line 99)
str_17648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'str', 'acos')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_17633, str_17648)
# Adding element type (line 99)
str_17649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 16), 'str', 'atan')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_17633, str_17649)
# Adding element type (line 99)
str_17650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 24), 'str', 'fmod')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_17633, str_17650)
# Adding element type (line 99)
str_17651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 32), 'str', 'modf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_17633, str_17651)
# Adding element type (line 99)
str_17652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 40), 'str', 'frexp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_17633, str_17652)
# Adding element type (line 99)
str_17653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 49), 'str', 'ldexp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 18), list_17633, str_17653)

# Assigning a type to the variable 'MANDATORY_FUNCS' (line 99)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'MANDATORY_FUNCS', list_17633)

# Assigning a List to a Name (line 105):

# Assigning a List to a Name (line 105):

# Obtaining an instance of the builtin type 'list' (line 105)
list_17654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 105)
# Adding element type (line 105)
str_17655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 21), 'str', 'expm1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17655)
# Adding element type (line 105)
str_17656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 30), 'str', 'log1p')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17656)
# Adding element type (line 105)
str_17657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 39), 'str', 'acosh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17657)
# Adding element type (line 105)
str_17658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 48), 'str', 'asinh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17658)
# Adding element type (line 105)
str_17659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 57), 'str', 'atanh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17659)
# Adding element type (line 105)
str_17660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'str', 'rint')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17660)
# Adding element type (line 105)
str_17661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 16), 'str', 'trunc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17661)
# Adding element type (line 105)
str_17662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 25), 'str', 'exp2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17662)
# Adding element type (line 105)
str_17663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 33), 'str', 'log2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17663)
# Adding element type (line 105)
str_17664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 41), 'str', 'hypot')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17664)
# Adding element type (line 105)
str_17665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 50), 'str', 'atan2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17665)
# Adding element type (line 105)
str_17666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 59), 'str', 'pow')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17666)
# Adding element type (line 105)
str_17667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 8), 'str', 'copysign')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17667)
# Adding element type (line 105)
str_17668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 20), 'str', 'nextafter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17668)
# Adding element type (line 105)
str_17669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 33), 'str', 'ftello')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17669)
# Adding element type (line 105)
str_17670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 43), 'str', 'fseeko')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17670)
# Adding element type (line 105)
str_17671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 8), 'str', 'strtoll')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17671)
# Adding element type (line 105)
str_17672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 19), 'str', 'strtoull')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17672)
# Adding element type (line 105)
str_17673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 31), 'str', 'cbrt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17673)
# Adding element type (line 105)
str_17674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 39), 'str', 'strtold_l')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17674)
# Adding element type (line 105)
str_17675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 52), 'str', 'fallocate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 20), list_17654, str_17675)

# Assigning a type to the variable 'OPTIONAL_STDFUNCS' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'OPTIONAL_STDFUNCS', list_17654)

# Assigning a List to a Name (line 111):

# Assigning a List to a Name (line 111):

# Obtaining an instance of the builtin type 'list' (line 111)
list_17676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 111)
# Adding element type (line 111)
str_17677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 16), 'str', 'xmmintrin.h')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 19), list_17676, str_17677)
# Adding element type (line 111)
str_17678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 16), 'str', 'emmintrin.h')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 19), list_17676, str_17678)
# Adding element type (line 111)
str_17679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 16), 'str', 'features.h')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 19), list_17676, str_17679)

# Assigning a type to the variable 'OPTIONAL_HEADERS' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'OPTIONAL_HEADERS', list_17676)

# Assigning a List to a Name (line 121):

# Assigning a List to a Name (line 121):

# Obtaining an instance of the builtin type 'list' (line 121)
list_17680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 121)
# Adding element type (line 121)

# Obtaining an instance of the builtin type 'tuple' (line 121)
tuple_17681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 121)
# Adding element type (line 121)
str_17682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 24), 'str', '__builtin_isnan')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 24), tuple_17681, str_17682)
# Adding element type (line 121)
str_17683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 43), 'str', '5.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 24), tuple_17681, str_17683)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 22), list_17680, tuple_17681)
# Adding element type (line 121)

# Obtaining an instance of the builtin type 'tuple' (line 122)
tuple_17684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 122)
# Adding element type (line 122)
str_17685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 24), 'str', '__builtin_isinf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 24), tuple_17684, str_17685)
# Adding element type (line 122)
str_17686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 43), 'str', '5.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 24), tuple_17684, str_17686)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 22), list_17680, tuple_17684)
# Adding element type (line 121)

# Obtaining an instance of the builtin type 'tuple' (line 123)
tuple_17687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 123)
# Adding element type (line 123)
str_17688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 24), 'str', '__builtin_isfinite')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 24), tuple_17687, str_17688)
# Adding element type (line 123)
str_17689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 46), 'str', '5.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 24), tuple_17687, str_17689)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 22), list_17680, tuple_17687)
# Adding element type (line 121)

# Obtaining an instance of the builtin type 'tuple' (line 124)
tuple_17690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 124)
# Adding element type (line 124)
str_17691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 24), 'str', '__builtin_bswap32')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 24), tuple_17690, str_17691)
# Adding element type (line 124)
str_17692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 45), 'str', '5u')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 24), tuple_17690, str_17692)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 22), list_17680, tuple_17690)
# Adding element type (line 121)

# Obtaining an instance of the builtin type 'tuple' (line 125)
tuple_17693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 125)
# Adding element type (line 125)
str_17694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 24), 'str', '__builtin_bswap64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 24), tuple_17693, str_17694)
# Adding element type (line 125)
str_17695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 45), 'str', '5u')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 24), tuple_17693, str_17695)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 22), list_17680, tuple_17693)
# Adding element type (line 121)

# Obtaining an instance of the builtin type 'tuple' (line 126)
tuple_17696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 126)
# Adding element type (line 126)
str_17697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 24), 'str', '__builtin_expect')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 24), tuple_17696, str_17697)
# Adding element type (line 126)
str_17698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 44), 'str', '5, 0')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 24), tuple_17696, str_17698)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 22), list_17680, tuple_17696)
# Adding element type (line 121)

# Obtaining an instance of the builtin type 'tuple' (line 127)
tuple_17699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 127)
# Adding element type (line 127)
str_17700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 24), 'str', '__builtin_mul_overflow')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 24), tuple_17699, str_17700)
# Adding element type (line 127)
str_17701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 50), 'str', '5, 5, (int*)5')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 24), tuple_17699, str_17701)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 22), list_17680, tuple_17699)
# Adding element type (line 121)

# Obtaining an instance of the builtin type 'tuple' (line 128)
tuple_17702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 128)
# Adding element type (line 128)
str_17703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 24), 'str', '_mm_load_ps')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 24), tuple_17702, str_17703)
# Adding element type (line 128)
str_17704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 39), 'str', '(float*)0')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 24), tuple_17702, str_17704)
# Adding element type (line 128)
str_17705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 52), 'str', 'xmmintrin.h')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 24), tuple_17702, str_17705)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 22), list_17680, tuple_17702)
# Adding element type (line 121)

# Obtaining an instance of the builtin type 'tuple' (line 129)
tuple_17706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 129)
# Adding element type (line 129)
str_17707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 24), 'str', '_mm_prefetch')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 24), tuple_17706, str_17707)
# Adding element type (line 129)
str_17708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 40), 'str', '(float*)0, _MM_HINT_NTA')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 24), tuple_17706, str_17708)
# Adding element type (line 129)
str_17709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 24), 'str', 'xmmintrin.h')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 24), tuple_17706, str_17709)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 22), list_17680, tuple_17706)
# Adding element type (line 121)

# Obtaining an instance of the builtin type 'tuple' (line 131)
tuple_17710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 131)
# Adding element type (line 131)
str_17711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 24), 'str', '_mm_load_pd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 24), tuple_17710, str_17711)
# Adding element type (line 131)
str_17712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 39), 'str', '(double*)0')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 24), tuple_17710, str_17712)
# Adding element type (line 131)
str_17713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 53), 'str', 'emmintrin.h')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 24), tuple_17710, str_17713)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 22), list_17680, tuple_17710)
# Adding element type (line 121)

# Obtaining an instance of the builtin type 'tuple' (line 132)
tuple_17714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 132)
# Adding element type (line 132)
str_17715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 24), 'str', '__builtin_prefetch')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 24), tuple_17714, str_17715)
# Adding element type (line 132)
str_17716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 46), 'str', '(float*)0, 0, 3')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 24), tuple_17714, str_17716)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 22), list_17680, tuple_17714)

# Assigning a type to the variable 'OPTIONAL_INTRINSICS' (line 121)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'OPTIONAL_INTRINSICS', list_17680)

# Assigning a List to a Name (line 138):

# Assigning a List to a Name (line 138):

# Obtaining an instance of the builtin type 'list' (line 138)
list_17717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 31), 'list')
# Adding type elements to the builtin type 'list' instance (line 138)
# Adding element type (line 138)

# Obtaining an instance of the builtin type 'tuple' (line 138)
tuple_17718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 138)
# Adding element type (line 138)
str_17719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 33), 'str', '__attribute__((optimize("unroll-loops")))')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 33), tuple_17718, str_17719)
# Adding element type (line 138)
str_17720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 32), 'str', 'attribute_optimize_unroll_loops')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 33), tuple_17718, str_17720)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 31), list_17717, tuple_17718)
# Adding element type (line 138)

# Obtaining an instance of the builtin type 'tuple' (line 140)
tuple_17721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 140)
# Adding element type (line 140)
str_17722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 33), 'str', '__attribute__((optimize("O3")))')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 33), tuple_17721, str_17722)
# Adding element type (line 140)
str_17723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 33), 'str', 'attribute_optimize_opt_3')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 33), tuple_17721, str_17723)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 31), list_17717, tuple_17721)
# Adding element type (line 138)

# Obtaining an instance of the builtin type 'tuple' (line 142)
tuple_17724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 142)
# Adding element type (line 142)
str_17725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 33), 'str', '__attribute__((nonnull (1)))')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 33), tuple_17724, str_17725)
# Adding element type (line 142)
str_17726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 33), 'str', 'attribute_nonnull')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 33), tuple_17724, str_17726)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 31), list_17717, tuple_17724)

# Assigning a type to the variable 'OPTIONAL_FUNCTION_ATTRIBUTES' (line 138)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'OPTIONAL_FUNCTION_ATTRIBUTES', list_17717)

# Assigning a List to a Name (line 147):

# Assigning a List to a Name (line 147):

# Obtaining an instance of the builtin type 'list' (line 147)
list_17727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 31), 'list')
# Adding type elements to the builtin type 'list' instance (line 147)
# Adding element type (line 147)
str_17728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 32), 'str', '__thread')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 31), list_17727, str_17728)
# Adding element type (line 147)
str_17729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 44), 'str', '__declspec(thread)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 31), list_17727, str_17729)

# Assigning a type to the variable 'OPTIONAL_VARIABLE_ATTRIBUTES' (line 147)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 0), 'OPTIONAL_VARIABLE_ATTRIBUTES', list_17727)

# Assigning a List to a Name (line 150):

# Assigning a List to a Name (line 150):

# Obtaining an instance of the builtin type 'list' (line 150)
list_17730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 26), 'list')
# Adding type elements to the builtin type 'list' instance (line 150)
# Adding element type (line 150)
str_17731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 4), 'str', 'expm1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 26), list_17730, str_17731)
# Adding element type (line 150)
str_17732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 13), 'str', 'log1p')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 26), list_17730, str_17732)
# Adding element type (line 150)
str_17733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 22), 'str', 'acosh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 26), list_17730, str_17733)
# Adding element type (line 150)
str_17734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 31), 'str', 'atanh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 26), list_17730, str_17734)
# Adding element type (line 150)
str_17735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 40), 'str', 'asinh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 26), list_17730, str_17735)
# Adding element type (line 150)
str_17736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 49), 'str', 'hypot')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 26), list_17730, str_17736)
# Adding element type (line 150)
str_17737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 58), 'str', 'copysign')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 26), list_17730, str_17737)
# Adding element type (line 150)
str_17738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 4), 'str', 'ftello')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 26), list_17730, str_17738)
# Adding element type (line 150)
str_17739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 14), 'str', 'fseeko')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 26), list_17730, str_17739)

# Assigning a type to the variable 'OPTIONAL_STDFUNCS_MAYBE' (line 150)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 0), 'OPTIONAL_STDFUNCS_MAYBE', list_17730)

# Assigning a List to a Name (line 156):

# Assigning a List to a Name (line 156):

# Obtaining an instance of the builtin type 'list' (line 156)
list_17740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 156)
# Adding element type (line 156)
str_17741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 4), 'str', 'sin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17741)
# Adding element type (line 156)
str_17742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 11), 'str', 'cos')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17742)
# Adding element type (line 156)
str_17743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 18), 'str', 'tan')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17743)
# Adding element type (line 156)
str_17744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 25), 'str', 'sinh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17744)
# Adding element type (line 156)
str_17745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 33), 'str', 'cosh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17745)
# Adding element type (line 156)
str_17746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 41), 'str', 'tanh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17746)
# Adding element type (line 156)
str_17747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 49), 'str', 'fabs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17747)
# Adding element type (line 156)
str_17748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 57), 'str', 'floor')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17748)
# Adding element type (line 156)
str_17749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 66), 'str', 'ceil')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17749)
# Adding element type (line 156)
str_17750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 4), 'str', 'rint')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17750)
# Adding element type (line 156)
str_17751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 12), 'str', 'trunc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17751)
# Adding element type (line 156)
str_17752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 21), 'str', 'sqrt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17752)
# Adding element type (line 156)
str_17753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 29), 'str', 'log10')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17753)
# Adding element type (line 156)
str_17754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 38), 'str', 'log')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17754)
# Adding element type (line 156)
str_17755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 45), 'str', 'log1p')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17755)
# Adding element type (line 156)
str_17756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 54), 'str', 'exp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17756)
# Adding element type (line 156)
str_17757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 61), 'str', 'expm1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17757)
# Adding element type (line 156)
str_17758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 4), 'str', 'asin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17758)
# Adding element type (line 156)
str_17759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 12), 'str', 'acos')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17759)
# Adding element type (line 156)
str_17760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 20), 'str', 'atan')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17760)
# Adding element type (line 156)
str_17761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 28), 'str', 'asinh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17761)
# Adding element type (line 156)
str_17762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 37), 'str', 'acosh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17762)
# Adding element type (line 156)
str_17763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 46), 'str', 'atanh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17763)
# Adding element type (line 156)
str_17764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 55), 'str', 'hypot')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17764)
# Adding element type (line 156)
str_17765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 64), 'str', 'atan2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17765)
# Adding element type (line 156)
str_17766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 4), 'str', 'pow')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17766)
# Adding element type (line 156)
str_17767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 11), 'str', 'fmod')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17767)
# Adding element type (line 156)
str_17768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 19), 'str', 'modf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17768)
# Adding element type (line 156)
str_17769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 27), 'str', 'frexp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17769)
# Adding element type (line 156)
str_17770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 36), 'str', 'ldexp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17770)
# Adding element type (line 156)
str_17771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 45), 'str', 'exp2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17771)
# Adding element type (line 156)
str_17772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 53), 'str', 'log2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17772)
# Adding element type (line 156)
str_17773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 61), 'str', 'copysign')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17773)
# Adding element type (line 156)
str_17774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 4), 'str', 'nextafter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17774)
# Adding element type (line 156)
str_17775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 17), 'str', 'cbrt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), list_17740, str_17775)

# Assigning a type to the variable 'C99_FUNCS' (line 156)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 0), 'C99_FUNCS', list_17740)

# Assigning a ListComp to a Name (line 163):

# Assigning a ListComp to a Name (line 163):
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'C99_FUNCS' (line 163)
C99_FUNCS_17779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 37), 'C99_FUNCS')
comprehension_17780 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 20), C99_FUNCS_17779)
# Assigning a type to the variable 'f' (line 163)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'f', comprehension_17780)
# Getting the type of 'f' (line 163)
f_17776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'f')
str_17777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 24), 'str', 'f')
# Applying the binary operator '+' (line 163)
result_add_17778 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 20), '+', f_17776, str_17777)

list_17781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 20), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 20), list_17781, result_add_17778)
# Assigning a type to the variable 'C99_FUNCS_SINGLE' (line 163)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), 'C99_FUNCS_SINGLE', list_17781)

# Assigning a ListComp to a Name (line 164):

# Assigning a ListComp to a Name (line 164):
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'C99_FUNCS' (line 164)
C99_FUNCS_17785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 39), 'C99_FUNCS')
comprehension_17786 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 22), C99_FUNCS_17785)
# Assigning a type to the variable 'f' (line 164)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 22), 'f', comprehension_17786)
# Getting the type of 'f' (line 164)
f_17782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 22), 'f')
str_17783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 26), 'str', 'l')
# Applying the binary operator '+' (line 164)
result_add_17784 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 22), '+', f_17782, str_17783)

list_17787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 22), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 22), list_17787, result_add_17784)
# Assigning a type to the variable 'C99_FUNCS_EXTENDED' (line 164)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), 'C99_FUNCS_EXTENDED', list_17787)

# Assigning a List to a Name (line 165):

# Assigning a List to a Name (line 165):

# Obtaining an instance of the builtin type 'list' (line 165)
list_17788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 165)
# Adding element type (line 165)
str_17789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 4), 'str', 'complex double')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 20), list_17788, str_17789)
# Adding element type (line 165)
str_17790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 22), 'str', 'complex float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 20), list_17788, str_17790)
# Adding element type (line 165)
str_17791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 39), 'str', 'complex long double')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 20), list_17788, str_17791)

# Assigning a type to the variable 'C99_COMPLEX_TYPES' (line 165)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'C99_COMPLEX_TYPES', list_17788)

# Assigning a List to a Name (line 168):

# Assigning a List to a Name (line 168):

# Obtaining an instance of the builtin type 'list' (line 168)
list_17792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 168)
# Adding element type (line 168)
str_17793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 4), 'str', 'cabs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17793)
# Adding element type (line 168)
str_17794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 12), 'str', 'cacos')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17794)
# Adding element type (line 168)
str_17795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 21), 'str', 'cacosh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17795)
# Adding element type (line 168)
str_17796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 31), 'str', 'carg')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17796)
# Adding element type (line 168)
str_17797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 39), 'str', 'casin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17797)
# Adding element type (line 168)
str_17798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 48), 'str', 'casinh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17798)
# Adding element type (line 168)
str_17799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 58), 'str', 'catan')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17799)
# Adding element type (line 168)
str_17800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 4), 'str', 'catanh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17800)
# Adding element type (line 168)
str_17801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 14), 'str', 'ccos')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17801)
# Adding element type (line 168)
str_17802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 22), 'str', 'ccosh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17802)
# Adding element type (line 168)
str_17803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 31), 'str', 'cexp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17803)
# Adding element type (line 168)
str_17804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 39), 'str', 'cimag')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17804)
# Adding element type (line 168)
str_17805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 48), 'str', 'clog')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17805)
# Adding element type (line 168)
str_17806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 56), 'str', 'conj')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17806)
# Adding element type (line 168)
str_17807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 64), 'str', 'cpow')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17807)
# Adding element type (line 168)
str_17808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 4), 'str', 'cproj')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17808)
# Adding element type (line 168)
str_17809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 13), 'str', 'creal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17809)
# Adding element type (line 168)
str_17810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 22), 'str', 'csin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17810)
# Adding element type (line 168)
str_17811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 30), 'str', 'csinh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17811)
# Adding element type (line 168)
str_17812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 39), 'str', 'csqrt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17812)
# Adding element type (line 168)
str_17813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 48), 'str', 'ctan')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17813)
# Adding element type (line 168)
str_17814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 56), 'str', 'ctanh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 20), list_17792, str_17814)

# Assigning a type to the variable 'C99_COMPLEX_FUNCS' (line 168)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), 'C99_COMPLEX_FUNCS', list_17792)

@norecursion
def fname2def(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fname2def'
    module_type_store = module_type_store.open_function_context('fname2def', 174, 0, False)
    
    # Passed parameters checking function
    fname2def.stypy_localization = localization
    fname2def.stypy_type_of_self = None
    fname2def.stypy_type_store = module_type_store
    fname2def.stypy_function_name = 'fname2def'
    fname2def.stypy_param_names_list = ['name']
    fname2def.stypy_varargs_param_name = None
    fname2def.stypy_kwargs_param_name = None
    fname2def.stypy_call_defaults = defaults
    fname2def.stypy_call_varargs = varargs
    fname2def.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fname2def', ['name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fname2def', localization, ['name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fname2def(...)' code ##################

    str_17815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 11), 'str', 'HAVE_%s')
    
    # Call to upper(...): (line 175)
    # Processing the call keyword arguments (line 175)
    kwargs_17818 = {}
    # Getting the type of 'name' (line 175)
    name_17816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 23), 'name', False)
    # Obtaining the member 'upper' of a type (line 175)
    upper_17817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 23), name_17816, 'upper')
    # Calling upper(args, kwargs) (line 175)
    upper_call_result_17819 = invoke(stypy.reporting.localization.Localization(__file__, 175, 23), upper_17817, *[], **kwargs_17818)
    
    # Applying the binary operator '%' (line 175)
    result_mod_17820 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 11), '%', str_17815, upper_call_result_17819)
    
    # Assigning a type to the variable 'stypy_return_type' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'stypy_return_type', result_mod_17820)
    
    # ################# End of 'fname2def(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fname2def' in the type store
    # Getting the type of 'stypy_return_type' (line 174)
    stypy_return_type_17821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17821)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fname2def'
    return stypy_return_type_17821

# Assigning a type to the variable 'fname2def' (line 174)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 0), 'fname2def', fname2def)

@norecursion
def sym2def(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sym2def'
    module_type_store = module_type_store.open_function_context('sym2def', 177, 0, False)
    
    # Passed parameters checking function
    sym2def.stypy_localization = localization
    sym2def.stypy_type_of_self = None
    sym2def.stypy_type_store = module_type_store
    sym2def.stypy_function_name = 'sym2def'
    sym2def.stypy_param_names_list = ['symbol']
    sym2def.stypy_varargs_param_name = None
    sym2def.stypy_kwargs_param_name = None
    sym2def.stypy_call_defaults = defaults
    sym2def.stypy_call_varargs = varargs
    sym2def.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sym2def', ['symbol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sym2def', localization, ['symbol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sym2def(...)' code ##################

    
    # Assigning a Call to a Name (line 178):
    
    # Assigning a Call to a Name (line 178):
    
    # Call to replace(...): (line 178)
    # Processing the call arguments (line 178)
    str_17824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 28), 'str', ' ')
    str_17825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 33), 'str', '')
    # Processing the call keyword arguments (line 178)
    kwargs_17826 = {}
    # Getting the type of 'symbol' (line 178)
    symbol_17822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 13), 'symbol', False)
    # Obtaining the member 'replace' of a type (line 178)
    replace_17823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 13), symbol_17822, 'replace')
    # Calling replace(args, kwargs) (line 178)
    replace_call_result_17827 = invoke(stypy.reporting.localization.Localization(__file__, 178, 13), replace_17823, *[str_17824, str_17825], **kwargs_17826)
    
    # Assigning a type to the variable 'define' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'define', replace_call_result_17827)
    
    # Call to upper(...): (line 179)
    # Processing the call keyword arguments (line 179)
    kwargs_17830 = {}
    # Getting the type of 'define' (line 179)
    define_17828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 11), 'define', False)
    # Obtaining the member 'upper' of a type (line 179)
    upper_17829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 11), define_17828, 'upper')
    # Calling upper(args, kwargs) (line 179)
    upper_call_result_17831 = invoke(stypy.reporting.localization.Localization(__file__, 179, 11), upper_17829, *[], **kwargs_17830)
    
    # Assigning a type to the variable 'stypy_return_type' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'stypy_return_type', upper_call_result_17831)
    
    # ################# End of 'sym2def(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sym2def' in the type store
    # Getting the type of 'stypy_return_type' (line 177)
    stypy_return_type_17832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17832)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sym2def'
    return stypy_return_type_17832

# Assigning a type to the variable 'sym2def' (line 177)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'sym2def', sym2def)

@norecursion
def type2def(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'type2def'
    module_type_store = module_type_store.open_function_context('type2def', 181, 0, False)
    
    # Passed parameters checking function
    type2def.stypy_localization = localization
    type2def.stypy_type_of_self = None
    type2def.stypy_type_store = module_type_store
    type2def.stypy_function_name = 'type2def'
    type2def.stypy_param_names_list = ['symbol']
    type2def.stypy_varargs_param_name = None
    type2def.stypy_kwargs_param_name = None
    type2def.stypy_call_defaults = defaults
    type2def.stypy_call_varargs = varargs
    type2def.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'type2def', ['symbol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'type2def', localization, ['symbol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'type2def(...)' code ##################

    
    # Assigning a Call to a Name (line 182):
    
    # Assigning a Call to a Name (line 182):
    
    # Call to replace(...): (line 182)
    # Processing the call arguments (line 182)
    str_17835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 28), 'str', ' ')
    str_17836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 33), 'str', '_')
    # Processing the call keyword arguments (line 182)
    kwargs_17837 = {}
    # Getting the type of 'symbol' (line 182)
    symbol_17833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 13), 'symbol', False)
    # Obtaining the member 'replace' of a type (line 182)
    replace_17834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 13), symbol_17833, 'replace')
    # Calling replace(args, kwargs) (line 182)
    replace_call_result_17838 = invoke(stypy.reporting.localization.Localization(__file__, 182, 13), replace_17834, *[str_17835, str_17836], **kwargs_17837)
    
    # Assigning a type to the variable 'define' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'define', replace_call_result_17838)
    
    # Call to upper(...): (line 183)
    # Processing the call keyword arguments (line 183)
    kwargs_17841 = {}
    # Getting the type of 'define' (line 183)
    define_17839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 11), 'define', False)
    # Obtaining the member 'upper' of a type (line 183)
    upper_17840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 11), define_17839, 'upper')
    # Calling upper(args, kwargs) (line 183)
    upper_call_result_17842 = invoke(stypy.reporting.localization.Localization(__file__, 183, 11), upper_17840, *[], **kwargs_17841)
    
    # Assigning a type to the variable 'stypy_return_type' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type', upper_call_result_17842)
    
    # ################# End of 'type2def(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'type2def' in the type store
    # Getting the type of 'stypy_return_type' (line 181)
    stypy_return_type_17843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17843)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'type2def'
    return stypy_return_type_17843

# Assigning a type to the variable 'type2def' (line 181)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 0), 'type2def', type2def)

@norecursion
def check_long_double_representation(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_long_double_representation'
    module_type_store = module_type_store.open_function_context('check_long_double_representation', 186, 0, False)
    
    # Passed parameters checking function
    check_long_double_representation.stypy_localization = localization
    check_long_double_representation.stypy_type_of_self = None
    check_long_double_representation.stypy_type_store = module_type_store
    check_long_double_representation.stypy_function_name = 'check_long_double_representation'
    check_long_double_representation.stypy_param_names_list = ['cmd']
    check_long_double_representation.stypy_varargs_param_name = None
    check_long_double_representation.stypy_kwargs_param_name = None
    check_long_double_representation.stypy_call_defaults = defaults
    check_long_double_representation.stypy_call_varargs = varargs
    check_long_double_representation.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_long_double_representation', ['cmd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_long_double_representation', localization, ['cmd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_long_double_representation(...)' code ##################

    
    # Call to _check_compiler(...): (line 187)
    # Processing the call keyword arguments (line 187)
    kwargs_17846 = {}
    # Getting the type of 'cmd' (line 187)
    cmd_17844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'cmd', False)
    # Obtaining the member '_check_compiler' of a type (line 187)
    _check_compiler_17845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 4), cmd_17844, '_check_compiler')
    # Calling _check_compiler(args, kwargs) (line 187)
    _check_compiler_call_result_17847 = invoke(stypy.reporting.localization.Localization(__file__, 187, 4), _check_compiler_17845, *[], **kwargs_17846)
    
    
    # Assigning a BinOp to a Name (line 188):
    
    # Assigning a BinOp to a Name (line 188):
    # Getting the type of 'LONG_DOUBLE_REPRESENTATION_SRC' (line 188)
    LONG_DOUBLE_REPRESENTATION_SRC_17848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 11), 'LONG_DOUBLE_REPRESENTATION_SRC')
    
    # Obtaining an instance of the builtin type 'dict' (line 188)
    dict_17849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 44), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 188)
    # Adding element type (key, value) (line 188)
    str_17850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 45), 'str', 'type')
    str_17851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 53), 'str', 'long double')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 44), dict_17849, (str_17850, str_17851))
    
    # Applying the binary operator '%' (line 188)
    result_mod_17852 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 11), '%', LONG_DOUBLE_REPRESENTATION_SRC_17848, dict_17849)
    
    # Assigning a type to the variable 'body' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'body', result_mod_17852)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'sys' (line 193)
    sys_17853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 7), 'sys')
    # Obtaining the member 'platform' of a type (line 193)
    platform_17854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 7), sys_17853, 'platform')
    str_17855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 23), 'str', 'win32')
    # Applying the binary operator '==' (line 193)
    result_eq_17856 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 7), '==', platform_17854, str_17855)
    
    
    
    # Call to mingw32(...): (line 193)
    # Processing the call keyword arguments (line 193)
    kwargs_17858 = {}
    # Getting the type of 'mingw32' (line 193)
    mingw32_17857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 39), 'mingw32', False)
    # Calling mingw32(args, kwargs) (line 193)
    mingw32_call_result_17859 = invoke(stypy.reporting.localization.Localization(__file__, 193, 39), mingw32_17857, *[], **kwargs_17858)
    
    # Applying the 'not' unary operator (line 193)
    result_not__17860 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 35), 'not', mingw32_call_result_17859)
    
    # Applying the binary operator 'and' (line 193)
    result_and_keyword_17861 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 7), 'and', result_eq_17856, result_not__17860)
    
    # Testing the type of an if condition (line 193)
    if_condition_17862 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 193, 4), result_and_keyword_17861)
    # Assigning a type to the variable 'if_condition_17862' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'if_condition_17862', if_condition_17862)
    # SSA begins for if statement (line 193)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 194)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to remove(...): (line 195)
    # Processing the call arguments (line 195)
    str_17867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 48), 'str', '/GL')
    # Processing the call keyword arguments (line 195)
    kwargs_17868 = {}
    # Getting the type of 'cmd' (line 195)
    cmd_17863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'cmd', False)
    # Obtaining the member 'compiler' of a type (line 195)
    compiler_17864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), cmd_17863, 'compiler')
    # Obtaining the member 'compile_options' of a type (line 195)
    compile_options_17865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), compiler_17864, 'compile_options')
    # Obtaining the member 'remove' of a type (line 195)
    remove_17866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), compile_options_17865, 'remove')
    # Calling remove(args, kwargs) (line 195)
    remove_call_result_17869 = invoke(stypy.reporting.localization.Localization(__file__, 195, 12), remove_17866, *[str_17867], **kwargs_17868)
    
    # SSA branch for the except part of a try statement (line 194)
    # SSA branch for the except 'Tuple' branch of a try statement (line 194)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 194)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 193)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 200):
    
    # Assigning a Call to a Name:
    
    # Call to _compile(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 'body' (line 200)
    body_17872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 28), 'body', False)
    # Getting the type of 'None' (line 200)
    None_17873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 34), 'None', False)
    # Getting the type of 'None' (line 200)
    None_17874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 40), 'None', False)
    str_17875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 46), 'str', 'c')
    # Processing the call keyword arguments (line 200)
    kwargs_17876 = {}
    # Getting the type of 'cmd' (line 200)
    cmd_17870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'cmd', False)
    # Obtaining the member '_compile' of a type (line 200)
    _compile_17871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 15), cmd_17870, '_compile')
    # Calling _compile(args, kwargs) (line 200)
    _compile_call_result_17877 = invoke(stypy.reporting.localization.Localization(__file__, 200, 15), _compile_17871, *[body_17872, None_17873, None_17874, str_17875], **kwargs_17876)
    
    # Assigning a type to the variable 'call_assignment_17512' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'call_assignment_17512', _compile_call_result_17877)
    
    # Assigning a Call to a Name (line 200):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_17880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 4), 'int')
    # Processing the call keyword arguments
    kwargs_17881 = {}
    # Getting the type of 'call_assignment_17512' (line 200)
    call_assignment_17512_17878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'call_assignment_17512', False)
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___17879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 4), call_assignment_17512_17878, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_17882 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___17879, *[int_17880], **kwargs_17881)
    
    # Assigning a type to the variable 'call_assignment_17513' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'call_assignment_17513', getitem___call_result_17882)
    
    # Assigning a Name to a Name (line 200):
    # Getting the type of 'call_assignment_17513' (line 200)
    call_assignment_17513_17883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'call_assignment_17513')
    # Assigning a type to the variable 'src' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'src', call_assignment_17513_17883)
    
    # Assigning a Call to a Name (line 200):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_17886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 4), 'int')
    # Processing the call keyword arguments
    kwargs_17887 = {}
    # Getting the type of 'call_assignment_17512' (line 200)
    call_assignment_17512_17884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'call_assignment_17512', False)
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___17885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 4), call_assignment_17512_17884, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_17888 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___17885, *[int_17886], **kwargs_17887)
    
    # Assigning a type to the variable 'call_assignment_17514' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'call_assignment_17514', getitem___call_result_17888)
    
    # Assigning a Name to a Name (line 200):
    # Getting the type of 'call_assignment_17514' (line 200)
    call_assignment_17514_17889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'call_assignment_17514')
    # Assigning a type to the variable 'obj' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 9), 'obj', call_assignment_17514_17889)
    
    # Try-finally block (line 201)
    
    
    # SSA begins for try-except statement (line 201)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 202):
    
    # Assigning a Call to a Name (line 202):
    
    # Call to long_double_representation(...): (line 202)
    # Processing the call arguments (line 202)
    
    # Call to pyod(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'obj' (line 202)
    obj_17892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 48), 'obj', False)
    # Processing the call keyword arguments (line 202)
    kwargs_17893 = {}
    # Getting the type of 'pyod' (line 202)
    pyod_17891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 43), 'pyod', False)
    # Calling pyod(args, kwargs) (line 202)
    pyod_call_result_17894 = invoke(stypy.reporting.localization.Localization(__file__, 202, 43), pyod_17891, *[obj_17892], **kwargs_17893)
    
    # Processing the call keyword arguments (line 202)
    kwargs_17895 = {}
    # Getting the type of 'long_double_representation' (line 202)
    long_double_representation_17890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'long_double_representation', False)
    # Calling long_double_representation(args, kwargs) (line 202)
    long_double_representation_call_result_17896 = invoke(stypy.reporting.localization.Localization(__file__, 202, 16), long_double_representation_17890, *[pyod_call_result_17894], **kwargs_17895)
    
    # Assigning a type to the variable 'ltype' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'ltype', long_double_representation_call_result_17896)
    # Getting the type of 'ltype' (line 203)
    ltype_17897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'ltype')
    # Assigning a type to the variable 'stypy_return_type' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'stypy_return_type', ltype_17897)
    # SSA branch for the except part of a try statement (line 201)
    # SSA branch for the except 'ValueError' branch of a try statement (line 201)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 207):
    
    # Assigning a Call to a Name (line 207):
    
    # Call to replace(...): (line 207)
    # Processing the call arguments (line 207)
    str_17900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 28), 'str', 'struct')
    str_17901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 38), 'str', 'volatile struct')
    # Processing the call keyword arguments (line 207)
    kwargs_17902 = {}
    # Getting the type of 'body' (line 207)
    body_17898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), 'body', False)
    # Obtaining the member 'replace' of a type (line 207)
    replace_17899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 15), body_17898, 'replace')
    # Calling replace(args, kwargs) (line 207)
    replace_call_result_17903 = invoke(stypy.reporting.localization.Localization(__file__, 207, 15), replace_17899, *[str_17900, str_17901], **kwargs_17902)
    
    # Assigning a type to the variable 'body' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'body', replace_call_result_17903)
    
    # Getting the type of 'body' (line 208)
    body_17904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'body')
    str_17905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 16), 'str', 'int main(void) { return 0; }\n')
    # Applying the binary operator '+=' (line 208)
    result_iadd_17906 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 8), '+=', body_17904, str_17905)
    # Assigning a type to the variable 'body' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'body', result_iadd_17906)
    
    
    # Assigning a Call to a Tuple (line 209):
    
    # Assigning a Call to a Name:
    
    # Call to _compile(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'body' (line 209)
    body_17909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 32), 'body', False)
    # Getting the type of 'None' (line 209)
    None_17910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 38), 'None', False)
    # Getting the type of 'None' (line 209)
    None_17911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 44), 'None', False)
    str_17912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 50), 'str', 'c')
    # Processing the call keyword arguments (line 209)
    kwargs_17913 = {}
    # Getting the type of 'cmd' (line 209)
    cmd_17907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 19), 'cmd', False)
    # Obtaining the member '_compile' of a type (line 209)
    _compile_17908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 19), cmd_17907, '_compile')
    # Calling _compile(args, kwargs) (line 209)
    _compile_call_result_17914 = invoke(stypy.reporting.localization.Localization(__file__, 209, 19), _compile_17908, *[body_17909, None_17910, None_17911, str_17912], **kwargs_17913)
    
    # Assigning a type to the variable 'call_assignment_17515' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'call_assignment_17515', _compile_call_result_17914)
    
    # Assigning a Call to a Name (line 209):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_17917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 8), 'int')
    # Processing the call keyword arguments
    kwargs_17918 = {}
    # Getting the type of 'call_assignment_17515' (line 209)
    call_assignment_17515_17915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'call_assignment_17515', False)
    # Obtaining the member '__getitem__' of a type (line 209)
    getitem___17916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), call_assignment_17515_17915, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_17919 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___17916, *[int_17917], **kwargs_17918)
    
    # Assigning a type to the variable 'call_assignment_17516' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'call_assignment_17516', getitem___call_result_17919)
    
    # Assigning a Name to a Name (line 209):
    # Getting the type of 'call_assignment_17516' (line 209)
    call_assignment_17516_17920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'call_assignment_17516')
    # Assigning a type to the variable 'src' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'src', call_assignment_17516_17920)
    
    # Assigning a Call to a Name (line 209):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_17923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 8), 'int')
    # Processing the call keyword arguments
    kwargs_17924 = {}
    # Getting the type of 'call_assignment_17515' (line 209)
    call_assignment_17515_17921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'call_assignment_17515', False)
    # Obtaining the member '__getitem__' of a type (line 209)
    getitem___17922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), call_assignment_17515_17921, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_17925 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___17922, *[int_17923], **kwargs_17924)
    
    # Assigning a type to the variable 'call_assignment_17517' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'call_assignment_17517', getitem___call_result_17925)
    
    # Assigning a Name to a Name (line 209):
    # Getting the type of 'call_assignment_17517' (line 209)
    call_assignment_17517_17926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'call_assignment_17517')
    # Assigning a type to the variable 'obj' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 13), 'obj', call_assignment_17517_17926)
    
    # Call to append(...): (line 210)
    # Processing the call arguments (line 210)
    str_17930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 30), 'str', '_configtest')
    # Processing the call keyword arguments (line 210)
    kwargs_17931 = {}
    # Getting the type of 'cmd' (line 210)
    cmd_17927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'cmd', False)
    # Obtaining the member 'temp_files' of a type (line 210)
    temp_files_17928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), cmd_17927, 'temp_files')
    # Obtaining the member 'append' of a type (line 210)
    append_17929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), temp_files_17928, 'append')
    # Calling append(args, kwargs) (line 210)
    append_call_result_17932 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), append_17929, *[str_17930], **kwargs_17931)
    
    
    # Call to link_executable(...): (line 211)
    # Processing the call arguments (line 211)
    
    # Obtaining an instance of the builtin type 'list' (line 211)
    list_17936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 211)
    # Adding element type (line 211)
    # Getting the type of 'obj' (line 211)
    obj_17937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 38), 'obj', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 37), list_17936, obj_17937)
    
    str_17938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 44), 'str', '_configtest')
    # Processing the call keyword arguments (line 211)
    kwargs_17939 = {}
    # Getting the type of 'cmd' (line 211)
    cmd_17933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'cmd', False)
    # Obtaining the member 'compiler' of a type (line 211)
    compiler_17934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), cmd_17933, 'compiler')
    # Obtaining the member 'link_executable' of a type (line 211)
    link_executable_17935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), compiler_17934, 'link_executable')
    # Calling link_executable(args, kwargs) (line 211)
    link_executable_call_result_17940 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), link_executable_17935, *[list_17936, str_17938], **kwargs_17939)
    
    
    # Assigning a Call to a Name (line 212):
    
    # Assigning a Call to a Name (line 212):
    
    # Call to long_double_representation(...): (line 212)
    # Processing the call arguments (line 212)
    
    # Call to pyod(...): (line 212)
    # Processing the call arguments (line 212)
    str_17943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 48), 'str', '_configtest')
    # Processing the call keyword arguments (line 212)
    kwargs_17944 = {}
    # Getting the type of 'pyod' (line 212)
    pyod_17942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 43), 'pyod', False)
    # Calling pyod(args, kwargs) (line 212)
    pyod_call_result_17945 = invoke(stypy.reporting.localization.Localization(__file__, 212, 43), pyod_17942, *[str_17943], **kwargs_17944)
    
    # Processing the call keyword arguments (line 212)
    kwargs_17946 = {}
    # Getting the type of 'long_double_representation' (line 212)
    long_double_representation_17941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 16), 'long_double_representation', False)
    # Calling long_double_representation(args, kwargs) (line 212)
    long_double_representation_call_result_17947 = invoke(stypy.reporting.localization.Localization(__file__, 212, 16), long_double_representation_17941, *[pyod_call_result_17945], **kwargs_17946)
    
    # Assigning a type to the variable 'ltype' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'ltype', long_double_representation_call_result_17947)
    # Getting the type of 'ltype' (line 213)
    ltype_17948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 15), 'ltype')
    # Assigning a type to the variable 'stypy_return_type' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'stypy_return_type', ltype_17948)
    # SSA join for try-except statement (line 201)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 201)
    
    # Call to _clean(...): (line 215)
    # Processing the call keyword arguments (line 215)
    kwargs_17951 = {}
    # Getting the type of 'cmd' (line 215)
    cmd_17949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'cmd', False)
    # Obtaining the member '_clean' of a type (line 215)
    _clean_17950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), cmd_17949, '_clean')
    # Calling _clean(args, kwargs) (line 215)
    _clean_call_result_17952 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), _clean_17950, *[], **kwargs_17951)
    
    
    
    # ################# End of 'check_long_double_representation(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_long_double_representation' in the type store
    # Getting the type of 'stypy_return_type' (line 186)
    stypy_return_type_17953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17953)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_long_double_representation'
    return stypy_return_type_17953

# Assigning a type to the variable 'check_long_double_representation' (line 186)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 0), 'check_long_double_representation', check_long_double_representation)

# Assigning a Str to a Name (line 217):

# Assigning a Str to a Name (line 217):
str_17954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, (-1)), 'str', '\n/* "before" is 16 bytes to ensure there\'s no padding between it and "x".\n *    We\'re not expecting any "long double" bigger than 16 bytes or with\n *       alignment requirements stricter than 16 bytes.  */\ntypedef %(type)s test_type;\n\nstruct {\n        char         before[16];\n        test_type    x;\n        char         after[8];\n} foo = {\n        { \'\\0\', \'\\0\', \'\\0\', \'\\0\', \'\\0\', \'\\0\', \'\\0\', \'\\0\',\n          \'\\001\', \'\\043\', \'\\105\', \'\\147\', \'\\211\', \'\\253\', \'\\315\', \'\\357\' },\n        -123456789.0,\n        { \'\\376\', \'\\334\', \'\\272\', \'\\230\', \'\\166\', \'\\124\', \'\\062\', \'\\020\' }\n};\n')
# Assigning a type to the variable 'LONG_DOUBLE_REPRESENTATION_SRC' (line 217)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 0), 'LONG_DOUBLE_REPRESENTATION_SRC', str_17954)

@norecursion
def pyod(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pyod'
    module_type_store = module_type_store.open_function_context('pyod', 235, 0, False)
    
    # Passed parameters checking function
    pyod.stypy_localization = localization
    pyod.stypy_type_of_self = None
    pyod.stypy_type_store = module_type_store
    pyod.stypy_function_name = 'pyod'
    pyod.stypy_param_names_list = ['filename']
    pyod.stypy_varargs_param_name = None
    pyod.stypy_kwargs_param_name = None
    pyod.stypy_call_defaults = defaults
    pyod.stypy_call_varargs = varargs
    pyod.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pyod', ['filename'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pyod', localization, ['filename'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pyod(...)' code ##################

    str_17955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, (-1)), 'str', 'Python implementation of the od UNIX utility (od -b, more exactly).\n\n    Parameters\n    ----------\n    filename : str\n        name of the file to get the dump from.\n\n    Returns\n    -------\n    out : seq\n        list of lines of od output\n\n    Note\n    ----\n    We only implement enough to get the necessary information for long double\n    representation, this is not intended as a compatible replacement for od.\n    ')

    @norecursion
    def _pyod2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pyod2'
        module_type_store = module_type_store.open_function_context('_pyod2', 253, 4, False)
        
        # Passed parameters checking function
        _pyod2.stypy_localization = localization
        _pyod2.stypy_type_of_self = None
        _pyod2.stypy_type_store = module_type_store
        _pyod2.stypy_function_name = '_pyod2'
        _pyod2.stypy_param_names_list = []
        _pyod2.stypy_varargs_param_name = None
        _pyod2.stypy_kwargs_param_name = None
        _pyod2.stypy_call_defaults = defaults
        _pyod2.stypy_call_varargs = varargs
        _pyod2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_pyod2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pyod2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pyod2(...)' code ##################

        
        # Assigning a List to a Name (line 254):
        
        # Assigning a List to a Name (line 254):
        
        # Obtaining an instance of the builtin type 'list' (line 254)
        list_17956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 254)
        
        # Assigning a type to the variable 'out' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'out', list_17956)
        
        # Assigning a Call to a Name (line 256):
        
        # Assigning a Call to a Name (line 256):
        
        # Call to open(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'filename' (line 256)
        filename_17958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 19), 'filename', False)
        str_17959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 29), 'str', 'rb')
        # Processing the call keyword arguments (line 256)
        kwargs_17960 = {}
        # Getting the type of 'open' (line 256)
        open_17957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 14), 'open', False)
        # Calling open(args, kwargs) (line 256)
        open_call_result_17961 = invoke(stypy.reporting.localization.Localization(__file__, 256, 14), open_17957, *[filename_17958, str_17959], **kwargs_17960)
        
        # Assigning a type to the variable 'fid' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'fid', open_call_result_17961)
        
        # Try-finally block (line 257)
        
        # Assigning a ListComp to a Name (line 258):
        
        # Assigning a ListComp to a Name (line 258):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to read(...): (line 258)
        # Processing the call keyword arguments (line 258)
        kwargs_17979 = {}
        # Getting the type of 'fid' (line 258)
        fid_17977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 66), 'fid', False)
        # Obtaining the member 'read' of a type (line 258)
        read_17978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 66), fid_17977, 'read')
        # Calling read(args, kwargs) (line 258)
        read_call_result_17980 = invoke(stypy.reporting.localization.Localization(__file__, 258, 66), read_17978, *[], **kwargs_17979)
        
        comprehension_17981 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 18), read_call_result_17980)
        # Assigning a type to the variable 'o' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 18), 'o', comprehension_17981)
        
        # Call to int(...): (line 258)
        # Processing the call arguments (line 258)
        
        # Call to oct(...): (line 258)
        # Processing the call arguments (line 258)
        
        # Call to int(...): (line 258)
        # Processing the call arguments (line 258)
        
        # Call to b2a_hex(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'o' (line 258)
        o_17967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 47), 'o', False)
        # Processing the call keyword arguments (line 258)
        kwargs_17968 = {}
        # Getting the type of 'binascii' (line 258)
        binascii_17965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 30), 'binascii', False)
        # Obtaining the member 'b2a_hex' of a type (line 258)
        b2a_hex_17966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 30), binascii_17965, 'b2a_hex')
        # Calling b2a_hex(args, kwargs) (line 258)
        b2a_hex_call_result_17969 = invoke(stypy.reporting.localization.Localization(__file__, 258, 30), b2a_hex_17966, *[o_17967], **kwargs_17968)
        
        int_17970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 51), 'int')
        # Processing the call keyword arguments (line 258)
        kwargs_17971 = {}
        # Getting the type of 'int' (line 258)
        int_17964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 26), 'int', False)
        # Calling int(args, kwargs) (line 258)
        int_call_result_17972 = invoke(stypy.reporting.localization.Localization(__file__, 258, 26), int_17964, *[b2a_hex_call_result_17969, int_17970], **kwargs_17971)
        
        # Processing the call keyword arguments (line 258)
        kwargs_17973 = {}
        # Getting the type of 'oct' (line 258)
        oct_17963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 22), 'oct', False)
        # Calling oct(args, kwargs) (line 258)
        oct_call_result_17974 = invoke(stypy.reporting.localization.Localization(__file__, 258, 22), oct_17963, *[int_call_result_17972], **kwargs_17973)
        
        # Processing the call keyword arguments (line 258)
        kwargs_17975 = {}
        # Getting the type of 'int' (line 258)
        int_17962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 18), 'int', False)
        # Calling int(args, kwargs) (line 258)
        int_call_result_17976 = invoke(stypy.reporting.localization.Localization(__file__, 258, 18), int_17962, *[oct_call_result_17974], **kwargs_17975)
        
        list_17982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 18), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 18), list_17982, int_call_result_17976)
        # Assigning a type to the variable 'yo' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'yo', list_17982)
        
        
        # Call to range(...): (line 259)
        # Processing the call arguments (line 259)
        int_17984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 27), 'int')
        
        # Call to len(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'yo' (line 259)
        yo_17986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 34), 'yo', False)
        # Processing the call keyword arguments (line 259)
        kwargs_17987 = {}
        # Getting the type of 'len' (line 259)
        len_17985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 30), 'len', False)
        # Calling len(args, kwargs) (line 259)
        len_call_result_17988 = invoke(stypy.reporting.localization.Localization(__file__, 259, 30), len_17985, *[yo_17986], **kwargs_17987)
        
        int_17989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 39), 'int')
        # Processing the call keyword arguments (line 259)
        kwargs_17990 = {}
        # Getting the type of 'range' (line 259)
        range_17983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 21), 'range', False)
        # Calling range(args, kwargs) (line 259)
        range_call_result_17991 = invoke(stypy.reporting.localization.Localization(__file__, 259, 21), range_17983, *[int_17984, len_call_result_17988, int_17989], **kwargs_17990)
        
        # Testing the type of a for loop iterable (line 259)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 259, 12), range_call_result_17991)
        # Getting the type of the for loop variable (line 259)
        for_loop_var_17992 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 259, 12), range_call_result_17991)
        # Assigning a type to the variable 'i' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'i', for_loop_var_17992)
        # SSA begins for a for statement (line 259)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a List to a Name (line 260):
        
        # Assigning a List to a Name (line 260):
        
        # Obtaining an instance of the builtin type 'list' (line 260)
        list_17993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 260)
        # Adding element type (line 260)
        str_17994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 24), 'str', '%07d')
        
        # Call to int(...): (line 260)
        # Processing the call arguments (line 260)
        
        # Call to oct(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'i' (line 260)
        i_17997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 41), 'i', False)
        # Processing the call keyword arguments (line 260)
        kwargs_17998 = {}
        # Getting the type of 'oct' (line 260)
        oct_17996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 37), 'oct', False)
        # Calling oct(args, kwargs) (line 260)
        oct_call_result_17999 = invoke(stypy.reporting.localization.Localization(__file__, 260, 37), oct_17996, *[i_17997], **kwargs_17998)
        
        # Processing the call keyword arguments (line 260)
        kwargs_18000 = {}
        # Getting the type of 'int' (line 260)
        int_17995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 33), 'int', False)
        # Calling int(args, kwargs) (line 260)
        int_call_result_18001 = invoke(stypy.reporting.localization.Localization(__file__, 260, 33), int_17995, *[oct_call_result_17999], **kwargs_18000)
        
        # Applying the binary operator '%' (line 260)
        result_mod_18002 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 24), '%', str_17994, int_call_result_18001)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 23), list_17993, result_mod_18002)
        
        # Assigning a type to the variable 'line' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'line', list_17993)
        
        # Call to extend(...): (line 261)
        # Processing the call arguments (line 261)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 261)
        i_18008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 52), 'i', False)
        # Getting the type of 'i' (line 261)
        i_18009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 54), 'i', False)
        int_18010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 56), 'int')
        # Applying the binary operator '+' (line 261)
        result_add_18011 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 54), '+', i_18009, int_18010)
        
        slice_18012 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 261, 49), i_18008, result_add_18011, None)
        # Getting the type of 'yo' (line 261)
        yo_18013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 49), 'yo', False)
        # Obtaining the member '__getitem__' of a type (line 261)
        getitem___18014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 49), yo_18013, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 261)
        subscript_call_result_18015 = invoke(stypy.reporting.localization.Localization(__file__, 261, 49), getitem___18014, slice_18012)
        
        comprehension_18016 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 29), subscript_call_result_18015)
        # Assigning a type to the variable 'c' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 29), 'c', comprehension_18016)
        str_18005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 29), 'str', '%03d')
        # Getting the type of 'c' (line 261)
        c_18006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 38), 'c', False)
        # Applying the binary operator '%' (line 261)
        result_mod_18007 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 29), '%', str_18005, c_18006)
        
        list_18017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 29), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 29), list_18017, result_mod_18007)
        # Processing the call keyword arguments (line 261)
        kwargs_18018 = {}
        # Getting the type of 'line' (line 261)
        line_18003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 16), 'line', False)
        # Obtaining the member 'extend' of a type (line 261)
        extend_18004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 16), line_18003, 'extend')
        # Calling extend(args, kwargs) (line 261)
        extend_call_result_18019 = invoke(stypy.reporting.localization.Localization(__file__, 261, 16), extend_18004, *[list_18017], **kwargs_18018)
        
        
        # Call to append(...): (line 262)
        # Processing the call arguments (line 262)
        
        # Call to join(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'line' (line 262)
        line_18024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 36), 'line', False)
        # Processing the call keyword arguments (line 262)
        kwargs_18025 = {}
        str_18022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 27), 'str', ' ')
        # Obtaining the member 'join' of a type (line 262)
        join_18023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 27), str_18022, 'join')
        # Calling join(args, kwargs) (line 262)
        join_call_result_18026 = invoke(stypy.reporting.localization.Localization(__file__, 262, 27), join_18023, *[line_18024], **kwargs_18025)
        
        # Processing the call keyword arguments (line 262)
        kwargs_18027 = {}
        # Getting the type of 'out' (line 262)
        out_18020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 16), 'out', False)
        # Obtaining the member 'append' of a type (line 262)
        append_18021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 16), out_18020, 'append')
        # Calling append(args, kwargs) (line 262)
        append_call_result_18028 = invoke(stypy.reporting.localization.Localization(__file__, 262, 16), append_18021, *[join_call_result_18026], **kwargs_18027)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'out' (line 263)
        out_18029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 19), 'out')
        # Assigning a type to the variable 'stypy_return_type' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'stypy_return_type', out_18029)
        
        # finally branch of the try-finally block (line 257)
        
        # Call to close(...): (line 265)
        # Processing the call keyword arguments (line 265)
        kwargs_18032 = {}
        # Getting the type of 'fid' (line 265)
        fid_18030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'fid', False)
        # Obtaining the member 'close' of a type (line 265)
        close_18031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 12), fid_18030, 'close')
        # Calling close(args, kwargs) (line 265)
        close_call_result_18033 = invoke(stypy.reporting.localization.Localization(__file__, 265, 12), close_18031, *[], **kwargs_18032)
        
        
        
        # ################# End of '_pyod2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pyod2' in the type store
        # Getting the type of 'stypy_return_type' (line 253)
        stypy_return_type_18034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18034)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pyod2'
        return stypy_return_type_18034

    # Assigning a type to the variable '_pyod2' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), '_pyod2', _pyod2)

    @norecursion
    def _pyod3(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_pyod3'
        module_type_store = module_type_store.open_function_context('_pyod3', 267, 4, False)
        
        # Passed parameters checking function
        _pyod3.stypy_localization = localization
        _pyod3.stypy_type_of_self = None
        _pyod3.stypy_type_store = module_type_store
        _pyod3.stypy_function_name = '_pyod3'
        _pyod3.stypy_param_names_list = []
        _pyod3.stypy_varargs_param_name = None
        _pyod3.stypy_kwargs_param_name = None
        _pyod3.stypy_call_defaults = defaults
        _pyod3.stypy_call_varargs = varargs
        _pyod3.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_pyod3', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_pyod3', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_pyod3(...)' code ##################

        
        # Assigning a List to a Name (line 268):
        
        # Assigning a List to a Name (line 268):
        
        # Obtaining an instance of the builtin type 'list' (line 268)
        list_18035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 268)
        
        # Assigning a type to the variable 'out' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'out', list_18035)
        
        # Assigning a Call to a Name (line 270):
        
        # Assigning a Call to a Name (line 270):
        
        # Call to open(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'filename' (line 270)
        filename_18037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 19), 'filename', False)
        str_18038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 29), 'str', 'rb')
        # Processing the call keyword arguments (line 270)
        kwargs_18039 = {}
        # Getting the type of 'open' (line 270)
        open_18036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 14), 'open', False)
        # Calling open(args, kwargs) (line 270)
        open_call_result_18040 = invoke(stypy.reporting.localization.Localization(__file__, 270, 14), open_18036, *[filename_18037, str_18038], **kwargs_18039)
        
        # Assigning a type to the variable 'fid' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'fid', open_call_result_18040)
        
        # Try-finally block (line 271)
        
        # Assigning a ListComp to a Name (line 272):
        
        # Assigning a ListComp to a Name (line 272):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to read(...): (line 272)
        # Processing the call keyword arguments (line 272)
        kwargs_18051 = {}
        # Getting the type of 'fid' (line 272)
        fid_18049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 39), 'fid', False)
        # Obtaining the member 'read' of a type (line 272)
        read_18050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 39), fid_18049, 'read')
        # Calling read(args, kwargs) (line 272)
        read_call_result_18052 = invoke(stypy.reporting.localization.Localization(__file__, 272, 39), read_18050, *[], **kwargs_18051)
        
        comprehension_18053 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 19), read_call_result_18052)
        # Assigning a type to the variable 'o' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 19), 'o', comprehension_18053)
        
        # Obtaining the type of the subscript
        int_18041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 26), 'int')
        slice_18042 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 272, 19), int_18041, None, None)
        
        # Call to oct(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'o' (line 272)
        o_18044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 23), 'o', False)
        # Processing the call keyword arguments (line 272)
        kwargs_18045 = {}
        # Getting the type of 'oct' (line 272)
        oct_18043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 19), 'oct', False)
        # Calling oct(args, kwargs) (line 272)
        oct_call_result_18046 = invoke(stypy.reporting.localization.Localization(__file__, 272, 19), oct_18043, *[o_18044], **kwargs_18045)
        
        # Obtaining the member '__getitem__' of a type (line 272)
        getitem___18047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 19), oct_call_result_18046, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 272)
        subscript_call_result_18048 = invoke(stypy.reporting.localization.Localization(__file__, 272, 19), getitem___18047, slice_18042)
        
        list_18054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 19), list_18054, subscript_call_result_18048)
        # Assigning a type to the variable 'yo2' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'yo2', list_18054)
        
        
        # Call to range(...): (line 273)
        # Processing the call arguments (line 273)
        int_18056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 27), 'int')
        
        # Call to len(...): (line 273)
        # Processing the call arguments (line 273)
        # Getting the type of 'yo2' (line 273)
        yo2_18058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 34), 'yo2', False)
        # Processing the call keyword arguments (line 273)
        kwargs_18059 = {}
        # Getting the type of 'len' (line 273)
        len_18057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 30), 'len', False)
        # Calling len(args, kwargs) (line 273)
        len_call_result_18060 = invoke(stypy.reporting.localization.Localization(__file__, 273, 30), len_18057, *[yo2_18058], **kwargs_18059)
        
        int_18061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 40), 'int')
        # Processing the call keyword arguments (line 273)
        kwargs_18062 = {}
        # Getting the type of 'range' (line 273)
        range_18055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 21), 'range', False)
        # Calling range(args, kwargs) (line 273)
        range_call_result_18063 = invoke(stypy.reporting.localization.Localization(__file__, 273, 21), range_18055, *[int_18056, len_call_result_18060, int_18061], **kwargs_18062)
        
        # Testing the type of a for loop iterable (line 273)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 273, 12), range_call_result_18063)
        # Getting the type of the for loop variable (line 273)
        for_loop_var_18064 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 273, 12), range_call_result_18063)
        # Assigning a type to the variable 'i' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'i', for_loop_var_18064)
        # SSA begins for a for statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a List to a Name (line 274):
        
        # Assigning a List to a Name (line 274):
        
        # Obtaining an instance of the builtin type 'list' (line 274)
        list_18065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 274)
        # Adding element type (line 274)
        str_18066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 24), 'str', '%07d')
        
        # Call to int(...): (line 274)
        # Processing the call arguments (line 274)
        
        # Obtaining the type of the subscript
        int_18068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 44), 'int')
        slice_18069 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 274, 37), int_18068, None, None)
        
        # Call to oct(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'i' (line 274)
        i_18071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 41), 'i', False)
        # Processing the call keyword arguments (line 274)
        kwargs_18072 = {}
        # Getting the type of 'oct' (line 274)
        oct_18070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 37), 'oct', False)
        # Calling oct(args, kwargs) (line 274)
        oct_call_result_18073 = invoke(stypy.reporting.localization.Localization(__file__, 274, 37), oct_18070, *[i_18071], **kwargs_18072)
        
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___18074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 37), oct_call_result_18073, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_18075 = invoke(stypy.reporting.localization.Localization(__file__, 274, 37), getitem___18074, slice_18069)
        
        # Processing the call keyword arguments (line 274)
        kwargs_18076 = {}
        # Getting the type of 'int' (line 274)
        int_18067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 33), 'int', False)
        # Calling int(args, kwargs) (line 274)
        int_call_result_18077 = invoke(stypy.reporting.localization.Localization(__file__, 274, 33), int_18067, *[subscript_call_result_18075], **kwargs_18076)
        
        # Applying the binary operator '%' (line 274)
        result_mod_18078 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 24), '%', str_18066, int_call_result_18077)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 23), list_18065, result_mod_18078)
        
        # Assigning a type to the variable 'line' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'line', list_18065)
        
        # Call to extend(...): (line 275)
        # Processing the call arguments (line 275)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 275)
        i_18087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 58), 'i', False)
        # Getting the type of 'i' (line 275)
        i_18088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 60), 'i', False)
        int_18089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 62), 'int')
        # Applying the binary operator '+' (line 275)
        result_add_18090 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 60), '+', i_18088, int_18089)
        
        slice_18091 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 275, 54), i_18087, result_add_18090, None)
        # Getting the type of 'yo2' (line 275)
        yo2_18092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 54), 'yo2', False)
        # Obtaining the member '__getitem__' of a type (line 275)
        getitem___18093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 54), yo2_18092, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 275)
        subscript_call_result_18094 = invoke(stypy.reporting.localization.Localization(__file__, 275, 54), getitem___18093, slice_18091)
        
        comprehension_18095 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 29), subscript_call_result_18094)
        # Assigning a type to the variable 'c' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 29), 'c', comprehension_18095)
        str_18081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 29), 'str', '%03d')
        
        # Call to int(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'c' (line 275)
        c_18083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 42), 'c', False)
        # Processing the call keyword arguments (line 275)
        kwargs_18084 = {}
        # Getting the type of 'int' (line 275)
        int_18082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 38), 'int', False)
        # Calling int(args, kwargs) (line 275)
        int_call_result_18085 = invoke(stypy.reporting.localization.Localization(__file__, 275, 38), int_18082, *[c_18083], **kwargs_18084)
        
        # Applying the binary operator '%' (line 275)
        result_mod_18086 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 29), '%', str_18081, int_call_result_18085)
        
        list_18096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 29), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 29), list_18096, result_mod_18086)
        # Processing the call keyword arguments (line 275)
        kwargs_18097 = {}
        # Getting the type of 'line' (line 275)
        line_18079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'line', False)
        # Obtaining the member 'extend' of a type (line 275)
        extend_18080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 16), line_18079, 'extend')
        # Calling extend(args, kwargs) (line 275)
        extend_call_result_18098 = invoke(stypy.reporting.localization.Localization(__file__, 275, 16), extend_18080, *[list_18096], **kwargs_18097)
        
        
        # Call to append(...): (line 276)
        # Processing the call arguments (line 276)
        
        # Call to join(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'line' (line 276)
        line_18103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 36), 'line', False)
        # Processing the call keyword arguments (line 276)
        kwargs_18104 = {}
        str_18101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 27), 'str', ' ')
        # Obtaining the member 'join' of a type (line 276)
        join_18102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 27), str_18101, 'join')
        # Calling join(args, kwargs) (line 276)
        join_call_result_18105 = invoke(stypy.reporting.localization.Localization(__file__, 276, 27), join_18102, *[line_18103], **kwargs_18104)
        
        # Processing the call keyword arguments (line 276)
        kwargs_18106 = {}
        # Getting the type of 'out' (line 276)
        out_18099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 16), 'out', False)
        # Obtaining the member 'append' of a type (line 276)
        append_18100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 16), out_18099, 'append')
        # Calling append(args, kwargs) (line 276)
        append_call_result_18107 = invoke(stypy.reporting.localization.Localization(__file__, 276, 16), append_18100, *[join_call_result_18105], **kwargs_18106)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'out' (line 277)
        out_18108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 19), 'out')
        # Assigning a type to the variable 'stypy_return_type' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'stypy_return_type', out_18108)
        
        # finally branch of the try-finally block (line 271)
        
        # Call to close(...): (line 279)
        # Processing the call keyword arguments (line 279)
        kwargs_18111 = {}
        # Getting the type of 'fid' (line 279)
        fid_18109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'fid', False)
        # Obtaining the member 'close' of a type (line 279)
        close_18110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 12), fid_18109, 'close')
        # Calling close(args, kwargs) (line 279)
        close_call_result_18112 = invoke(stypy.reporting.localization.Localization(__file__, 279, 12), close_18110, *[], **kwargs_18111)
        
        
        
        # ################# End of '_pyod3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_pyod3' in the type store
        # Getting the type of 'stypy_return_type' (line 267)
        stypy_return_type_18113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18113)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_pyod3'
        return stypy_return_type_18113

    # Assigning a type to the variable '_pyod3' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), '_pyod3', _pyod3)
    
    
    
    # Obtaining the type of the subscript
    int_18114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 24), 'int')
    # Getting the type of 'sys' (line 281)
    sys_18115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 7), 'sys')
    # Obtaining the member 'version_info' of a type (line 281)
    version_info_18116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 7), sys_18115, 'version_info')
    # Obtaining the member '__getitem__' of a type (line 281)
    getitem___18117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 7), version_info_18116, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 281)
    subscript_call_result_18118 = invoke(stypy.reporting.localization.Localization(__file__, 281, 7), getitem___18117, int_18114)
    
    int_18119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 29), 'int')
    # Applying the binary operator '<' (line 281)
    result_lt_18120 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 7), '<', subscript_call_result_18118, int_18119)
    
    # Testing the type of an if condition (line 281)
    if_condition_18121 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 4), result_lt_18120)
    # Assigning a type to the variable 'if_condition_18121' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'if_condition_18121', if_condition_18121)
    # SSA begins for if statement (line 281)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _pyod2(...): (line 282)
    # Processing the call keyword arguments (line 282)
    kwargs_18123 = {}
    # Getting the type of '_pyod2' (line 282)
    _pyod2_18122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 15), '_pyod2', False)
    # Calling _pyod2(args, kwargs) (line 282)
    _pyod2_call_result_18124 = invoke(stypy.reporting.localization.Localization(__file__, 282, 15), _pyod2_18122, *[], **kwargs_18123)
    
    # Assigning a type to the variable 'stypy_return_type' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'stypy_return_type', _pyod2_call_result_18124)
    # SSA branch for the else part of an if statement (line 281)
    module_type_store.open_ssa_branch('else')
    
    # Call to _pyod3(...): (line 284)
    # Processing the call keyword arguments (line 284)
    kwargs_18126 = {}
    # Getting the type of '_pyod3' (line 284)
    _pyod3_18125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 15), '_pyod3', False)
    # Calling _pyod3(args, kwargs) (line 284)
    _pyod3_call_result_18127 = invoke(stypy.reporting.localization.Localization(__file__, 284, 15), _pyod3_18125, *[], **kwargs_18126)
    
    # Assigning a type to the variable 'stypy_return_type' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'stypy_return_type', _pyod3_call_result_18127)
    # SSA join for if statement (line 281)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'pyod(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pyod' in the type store
    # Getting the type of 'stypy_return_type' (line 235)
    stypy_return_type_18128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18128)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pyod'
    return stypy_return_type_18128

# Assigning a type to the variable 'pyod' (line 235)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), 'pyod', pyod)

# Assigning a List to a Name (line 286):

# Assigning a List to a Name (line 286):

# Obtaining an instance of the builtin type 'list' (line 286)
list_18129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 286)
# Adding element type (line 286)
str_18130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 15), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 14), list_18129, str_18130)
# Adding element type (line 286)
str_18131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 22), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 14), list_18129, str_18131)
# Adding element type (line 286)
str_18132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 29), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 14), list_18129, str_18132)
# Adding element type (line 286)
str_18133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 36), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 14), list_18129, str_18133)
# Adding element type (line 286)
str_18134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 43), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 14), list_18129, str_18134)
# Adding element type (line 286)
str_18135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 50), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 14), list_18129, str_18135)
# Adding element type (line 286)
str_18136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 57), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 14), list_18129, str_18136)
# Adding element type (line 286)
str_18137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 64), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 14), list_18129, str_18137)
# Adding element type (line 286)
str_18138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 14), 'str', '001')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 14), list_18129, str_18138)
# Adding element type (line 286)
str_18139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 21), 'str', '043')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 14), list_18129, str_18139)
# Adding element type (line 286)
str_18140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 28), 'str', '105')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 14), list_18129, str_18140)
# Adding element type (line 286)
str_18141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 35), 'str', '147')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 14), list_18129, str_18141)
# Adding element type (line 286)
str_18142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 42), 'str', '211')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 14), list_18129, str_18142)
# Adding element type (line 286)
str_18143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 49), 'str', '253')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 14), list_18129, str_18143)
# Adding element type (line 286)
str_18144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 56), 'str', '315')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 14), list_18129, str_18144)
# Adding element type (line 286)
str_18145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 63), 'str', '357')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 14), list_18129, str_18145)

# Assigning a type to the variable '_BEFORE_SEQ' (line 286)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 0), '_BEFORE_SEQ', list_18129)

# Assigning a List to a Name (line 288):

# Assigning a List to a Name (line 288):

# Obtaining an instance of the builtin type 'list' (line 288)
list_18146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 288)
# Adding element type (line 288)
str_18147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 14), 'str', '376')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 13), list_18146, str_18147)
# Adding element type (line 288)
str_18148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 21), 'str', '334')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 13), list_18146, str_18148)
# Adding element type (line 288)
str_18149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 28), 'str', '272')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 13), list_18146, str_18149)
# Adding element type (line 288)
str_18150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 35), 'str', '230')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 13), list_18146, str_18150)
# Adding element type (line 288)
str_18151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 42), 'str', '166')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 13), list_18146, str_18151)
# Adding element type (line 288)
str_18152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 49), 'str', '124')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 13), list_18146, str_18152)
# Adding element type (line 288)
str_18153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 56), 'str', '062')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 13), list_18146, str_18153)
# Adding element type (line 288)
str_18154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 63), 'str', '020')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 13), list_18146, str_18154)

# Assigning a type to the variable '_AFTER_SEQ' (line 288)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 0), '_AFTER_SEQ', list_18146)

# Assigning a List to a Name (line 290):

# Assigning a List to a Name (line 290):

# Obtaining an instance of the builtin type 'list' (line 290)
list_18155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 290)
# Adding element type (line 290)
str_18156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 19), 'str', '301')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 18), list_18155, str_18156)
# Adding element type (line 290)
str_18157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 26), 'str', '235')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 18), list_18155, str_18157)
# Adding element type (line 290)
str_18158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 33), 'str', '157')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 18), list_18155, str_18158)
# Adding element type (line 290)
str_18159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 40), 'str', '064')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 18), list_18155, str_18159)
# Adding element type (line 290)
str_18160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 47), 'str', '124')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 18), list_18155, str_18160)
# Adding element type (line 290)
str_18161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 54), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 18), list_18155, str_18161)
# Adding element type (line 290)
str_18162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 61), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 18), list_18155, str_18162)
# Adding element type (line 290)
str_18163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 68), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 18), list_18155, str_18163)

# Assigning a type to the variable '_IEEE_DOUBLE_BE' (line 290)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 0), '_IEEE_DOUBLE_BE', list_18155)

# Assigning a Subscript to a Name (line 291):

# Assigning a Subscript to a Name (line 291):

# Obtaining the type of the subscript
int_18164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 36), 'int')
slice_18165 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 291, 18), None, None, int_18164)
# Getting the type of '_IEEE_DOUBLE_BE' (line 291)
_IEEE_DOUBLE_BE_18166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 18), '_IEEE_DOUBLE_BE')
# Obtaining the member '__getitem__' of a type (line 291)
getitem___18167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 18), _IEEE_DOUBLE_BE_18166, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 291)
subscript_call_result_18168 = invoke(stypy.reporting.localization.Localization(__file__, 291, 18), getitem___18167, slice_18165)

# Assigning a type to the variable '_IEEE_DOUBLE_LE' (line 291)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 0), '_IEEE_DOUBLE_LE', subscript_call_result_18168)

# Assigning a List to a Name (line 292):

# Assigning a List to a Name (line 292):

# Obtaining an instance of the builtin type 'list' (line 292)
list_18169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 292)
# Adding element type (line 292)
str_18170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 23), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 22), list_18169, str_18170)
# Adding element type (line 292)
str_18171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 30), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 22), list_18169, str_18171)
# Adding element type (line 292)
str_18172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 37), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 22), list_18169, str_18172)
# Adding element type (line 292)
str_18173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 44), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 22), list_18169, str_18173)
# Adding element type (line 292)
str_18174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 51), 'str', '240')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 22), list_18169, str_18174)
# Adding element type (line 292)
str_18175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 58), 'str', '242')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 22), list_18169, str_18175)
# Adding element type (line 292)
str_18176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 65), 'str', '171')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 22), list_18169, str_18176)
# Adding element type (line 292)
str_18177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 72), 'str', '353')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 22), list_18169, str_18177)
# Adding element type (line 292)
str_18178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 23), 'str', '031')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 22), list_18169, str_18178)
# Adding element type (line 292)
str_18179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 30), 'str', '300')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 22), list_18169, str_18179)
# Adding element type (line 292)
str_18180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 37), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 22), list_18169, str_18180)
# Adding element type (line 292)
str_18181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 44), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 22), list_18169, str_18181)

# Assigning a type to the variable '_INTEL_EXTENDED_12B' (line 292)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 0), '_INTEL_EXTENDED_12B', list_18169)

# Assigning a List to a Name (line 294):

# Assigning a List to a Name (line 294):

# Obtaining an instance of the builtin type 'list' (line 294)
list_18182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 294)
# Adding element type (line 294)
str_18183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 23), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 22), list_18182, str_18183)
# Adding element type (line 294)
str_18184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 30), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 22), list_18182, str_18184)
# Adding element type (line 294)
str_18185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 37), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 22), list_18182, str_18185)
# Adding element type (line 294)
str_18186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 44), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 22), list_18182, str_18186)
# Adding element type (line 294)
str_18187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 51), 'str', '240')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 22), list_18182, str_18187)
# Adding element type (line 294)
str_18188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 58), 'str', '242')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 22), list_18182, str_18188)
# Adding element type (line 294)
str_18189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 65), 'str', '171')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 22), list_18182, str_18189)
# Adding element type (line 294)
str_18190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 72), 'str', '353')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 22), list_18182, str_18190)
# Adding element type (line 294)
str_18191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 23), 'str', '031')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 22), list_18182, str_18191)
# Adding element type (line 294)
str_18192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 30), 'str', '300')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 22), list_18182, str_18192)
# Adding element type (line 294)
str_18193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 37), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 22), list_18182, str_18193)
# Adding element type (line 294)
str_18194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 44), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 22), list_18182, str_18194)
# Adding element type (line 294)
str_18195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 51), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 22), list_18182, str_18195)
# Adding element type (line 294)
str_18196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 58), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 22), list_18182, str_18196)
# Adding element type (line 294)
str_18197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 65), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 22), list_18182, str_18197)
# Adding element type (line 294)
str_18198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 72), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 22), list_18182, str_18198)

# Assigning a type to the variable '_INTEL_EXTENDED_16B' (line 294)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 0), '_INTEL_EXTENDED_16B', list_18182)

# Assigning a List to a Name (line 296):

# Assigning a List to a Name (line 296):

# Obtaining an instance of the builtin type 'list' (line 296)
list_18199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 296)
# Adding element type (line 296)
str_18200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 26), 'str', '300')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 25), list_18199, str_18200)
# Adding element type (line 296)
str_18201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 33), 'str', '031')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 25), list_18199, str_18201)
# Adding element type (line 296)
str_18202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 40), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 25), list_18199, str_18202)
# Adding element type (line 296)
str_18203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 47), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 25), list_18199, str_18203)
# Adding element type (line 296)
str_18204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 54), 'str', '353')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 25), list_18199, str_18204)
# Adding element type (line 296)
str_18205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 61), 'str', '171')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 25), list_18199, str_18205)
# Adding element type (line 296)
str_18206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 26), 'str', '242')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 25), list_18199, str_18206)
# Adding element type (line 296)
str_18207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 33), 'str', '240')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 25), list_18199, str_18207)
# Adding element type (line 296)
str_18208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 40), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 25), list_18199, str_18208)
# Adding element type (line 296)
str_18209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 47), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 25), list_18199, str_18209)
# Adding element type (line 296)
str_18210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 54), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 25), list_18199, str_18210)
# Adding element type (line 296)
str_18211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 61), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 25), list_18199, str_18211)

# Assigning a type to the variable '_MOTOROLA_EXTENDED_12B' (line 296)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 0), '_MOTOROLA_EXTENDED_12B', list_18199)

# Assigning a List to a Name (line 298):

# Assigning a List to a Name (line 298):

# Obtaining an instance of the builtin type 'list' (line 298)
list_18212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 298)
# Adding element type (line 298)
str_18213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 22), 'str', '300')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 21), list_18212, str_18213)
# Adding element type (line 298)
str_18214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 29), 'str', '031')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 21), list_18212, str_18214)
# Adding element type (line 298)
str_18215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 36), 'str', '326')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 21), list_18212, str_18215)
# Adding element type (line 298)
str_18216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 43), 'str', '363')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 21), list_18212, str_18216)
# Adding element type (line 298)
str_18217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 50), 'str', '105')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 21), list_18212, str_18217)
# Adding element type (line 298)
str_18218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 57), 'str', '100')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 21), list_18212, str_18218)
# Adding element type (line 298)
str_18219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 64), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 21), list_18212, str_18219)
# Adding element type (line 298)
str_18220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 71), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 21), list_18212, str_18220)
# Adding element type (line 298)
str_18221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 22), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 21), list_18212, str_18221)
# Adding element type (line 298)
str_18222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 29), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 21), list_18212, str_18222)
# Adding element type (line 298)
str_18223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 36), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 21), list_18212, str_18223)
# Adding element type (line 298)
str_18224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 43), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 21), list_18212, str_18224)
# Adding element type (line 298)
str_18225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 50), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 21), list_18212, str_18225)
# Adding element type (line 298)
str_18226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 57), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 21), list_18212, str_18226)
# Adding element type (line 298)
str_18227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 64), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 21), list_18212, str_18227)
# Adding element type (line 298)
str_18228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 71), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 21), list_18212, str_18228)

# Assigning a type to the variable '_IEEE_QUAD_PREC_BE' (line 298)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 0), '_IEEE_QUAD_PREC_BE', list_18212)

# Assigning a Subscript to a Name (line 300):

# Assigning a Subscript to a Name (line 300):

# Obtaining the type of the subscript
int_18229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 42), 'int')
slice_18230 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 300, 21), None, None, int_18229)
# Getting the type of '_IEEE_QUAD_PREC_BE' (line 300)
_IEEE_QUAD_PREC_BE_18231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 21), '_IEEE_QUAD_PREC_BE')
# Obtaining the member '__getitem__' of a type (line 300)
getitem___18232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 21), _IEEE_QUAD_PREC_BE_18231, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 300)
subscript_call_result_18233 = invoke(stypy.reporting.localization.Localization(__file__, 300, 21), getitem___18232, slice_18230)

# Assigning a type to the variable '_IEEE_QUAD_PREC_LE' (line 300)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 0), '_IEEE_QUAD_PREC_LE', subscript_call_result_18233)

# Assigning a BinOp to a Name (line 301):

# Assigning a BinOp to a Name (line 301):

# Obtaining an instance of the builtin type 'list' (line 301)
list_18234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 301)
# Adding element type (line 301)
str_18235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 22), 'str', '301')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 21), list_18234, str_18235)
# Adding element type (line 301)
str_18236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 29), 'str', '235')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 21), list_18234, str_18236)
# Adding element type (line 301)
str_18237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 36), 'str', '157')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 21), list_18234, str_18237)
# Adding element type (line 301)
str_18238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 43), 'str', '064')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 21), list_18234, str_18238)
# Adding element type (line 301)
str_18239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 50), 'str', '124')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 21), list_18234, str_18239)
# Adding element type (line 301)
str_18240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 57), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 21), list_18234, str_18240)
# Adding element type (line 301)
str_18241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 64), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 21), list_18234, str_18241)
# Adding element type (line 301)
str_18242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 71), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 21), list_18234, str_18242)


# Obtaining an instance of the builtin type 'list' (line 302)
list_18243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 302)
# Adding element type (line 302)
str_18244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 22), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 21), list_18243, str_18244)

int_18245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 31), 'int')
# Applying the binary operator '*' (line 302)
result_mul_18246 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 21), '*', list_18243, int_18245)

# Applying the binary operator '+' (line 301)
result_add_18247 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 21), '+', list_18234, result_mul_18246)

# Assigning a type to the variable '_DOUBLE_DOUBLE_BE' (line 301)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 0), '_DOUBLE_DOUBLE_BE', result_add_18247)

# Assigning a BinOp to a Name (line 303):

# Assigning a BinOp to a Name (line 303):

# Obtaining an instance of the builtin type 'list' (line 303)
list_18248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 303)
# Adding element type (line 303)
str_18249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 22), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 21), list_18248, str_18249)
# Adding element type (line 303)
str_18250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 29), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 21), list_18248, str_18250)
# Adding element type (line 303)
str_18251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 36), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 21), list_18248, str_18251)
# Adding element type (line 303)
str_18252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 43), 'str', '124')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 21), list_18248, str_18252)
# Adding element type (line 303)
str_18253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 50), 'str', '064')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 21), list_18248, str_18253)
# Adding element type (line 303)
str_18254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 57), 'str', '157')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 21), list_18248, str_18254)
# Adding element type (line 303)
str_18255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 64), 'str', '235')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 21), list_18248, str_18255)
# Adding element type (line 303)
str_18256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 71), 'str', '301')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 21), list_18248, str_18256)


# Obtaining an instance of the builtin type 'list' (line 304)
list_18257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 304)
# Adding element type (line 304)
str_18258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 22), 'str', '000')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 21), list_18257, str_18258)

int_18259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 31), 'int')
# Applying the binary operator '*' (line 304)
result_mul_18260 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 21), '*', list_18257, int_18259)

# Applying the binary operator '+' (line 303)
result_add_18261 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 21), '+', list_18248, result_mul_18260)

# Assigning a type to the variable '_DOUBLE_DOUBLE_LE' (line 303)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 0), '_DOUBLE_DOUBLE_LE', result_add_18261)

@norecursion
def long_double_representation(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'long_double_representation'
    module_type_store = module_type_store.open_function_context('long_double_representation', 306, 0, False)
    
    # Passed parameters checking function
    long_double_representation.stypy_localization = localization
    long_double_representation.stypy_type_of_self = None
    long_double_representation.stypy_type_store = module_type_store
    long_double_representation.stypy_function_name = 'long_double_representation'
    long_double_representation.stypy_param_names_list = ['lines']
    long_double_representation.stypy_varargs_param_name = None
    long_double_representation.stypy_kwargs_param_name = None
    long_double_representation.stypy_call_defaults = defaults
    long_double_representation.stypy_call_varargs = varargs
    long_double_representation.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'long_double_representation', ['lines'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'long_double_representation', localization, ['lines'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'long_double_representation(...)' code ##################

    str_18262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, (-1)), 'str', 'Given a binary dump as given by GNU od -b, look for long double\n    representation.')
    
    # Assigning a BinOp to a Name (line 317):
    
    # Assigning a BinOp to a Name (line 317):
    
    # Obtaining an instance of the builtin type 'list' (line 317)
    list_18263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 317)
    # Adding element type (line 317)
    str_18264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 12), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 11), list_18263, str_18264)
    
    int_18265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 18), 'int')
    # Applying the binary operator '*' (line 317)
    result_mul_18266 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 11), '*', list_18263, int_18265)
    
    # Assigning a type to the variable 'read' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'read', result_mul_18266)
    
    # Assigning a Name to a Name (line 318):
    
    # Assigning a Name to a Name (line 318):
    # Getting the type of 'None' (line 318)
    None_18267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 10), 'None')
    # Assigning a type to the variable 'saw' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'saw', None_18267)
    
    # Getting the type of 'lines' (line 319)
    lines_18268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 16), 'lines')
    # Testing the type of a for loop iterable (line 319)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 319, 4), lines_18268)
    # Getting the type of the for loop variable (line 319)
    for_loop_var_18269 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 319, 4), lines_18268)
    # Assigning a type to the variable 'line' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'line', for_loop_var_18269)
    # SSA begins for a for statement (line 319)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining the type of the subscript
    int_18270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 30), 'int')
    slice_18271 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 322, 17), int_18270, None, None)
    
    # Call to split(...): (line 322)
    # Processing the call keyword arguments (line 322)
    kwargs_18274 = {}
    # Getting the type of 'line' (line 322)
    line_18272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 17), 'line', False)
    # Obtaining the member 'split' of a type (line 322)
    split_18273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 17), line_18272, 'split')
    # Calling split(args, kwargs) (line 322)
    split_call_result_18275 = invoke(stypy.reporting.localization.Localization(__file__, 322, 17), split_18273, *[], **kwargs_18274)
    
    # Obtaining the member '__getitem__' of a type (line 322)
    getitem___18276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 17), split_call_result_18275, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 322)
    subscript_call_result_18277 = invoke(stypy.reporting.localization.Localization(__file__, 322, 17), getitem___18276, slice_18271)
    
    # Testing the type of a for loop iterable (line 322)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 322, 8), subscript_call_result_18277)
    # Getting the type of the for loop variable (line 322)
    for_loop_var_18278 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 322, 8), subscript_call_result_18277)
    # Assigning a type to the variable 'w' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'w', for_loop_var_18278)
    # SSA begins for a for statement (line 322)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to pop(...): (line 323)
    # Processing the call arguments (line 323)
    int_18281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 21), 'int')
    # Processing the call keyword arguments (line 323)
    kwargs_18282 = {}
    # Getting the type of 'read' (line 323)
    read_18279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'read', False)
    # Obtaining the member 'pop' of a type (line 323)
    pop_18280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 12), read_18279, 'pop')
    # Calling pop(args, kwargs) (line 323)
    pop_call_result_18283 = invoke(stypy.reporting.localization.Localization(__file__, 323, 12), pop_18280, *[int_18281], **kwargs_18282)
    
    
    # Call to append(...): (line 324)
    # Processing the call arguments (line 324)
    # Getting the type of 'w' (line 324)
    w_18286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 24), 'w', False)
    # Processing the call keyword arguments (line 324)
    kwargs_18287 = {}
    # Getting the type of 'read' (line 324)
    read_18284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'read', False)
    # Obtaining the member 'append' of a type (line 324)
    append_18285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 12), read_18284, 'append')
    # Calling append(args, kwargs) (line 324)
    append_call_result_18288 = invoke(stypy.reporting.localization.Localization(__file__, 324, 12), append_18285, *[w_18286], **kwargs_18287)
    
    
    
    
    # Obtaining the type of the subscript
    int_18289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 20), 'int')
    slice_18290 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 328, 15), int_18289, None, None)
    # Getting the type of 'read' (line 328)
    read_18291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 15), 'read')
    # Obtaining the member '__getitem__' of a type (line 328)
    getitem___18292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 15), read_18291, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 328)
    subscript_call_result_18293 = invoke(stypy.reporting.localization.Localization(__file__, 328, 15), getitem___18292, slice_18290)
    
    # Getting the type of '_AFTER_SEQ' (line 328)
    _AFTER_SEQ_18294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 28), '_AFTER_SEQ')
    # Applying the binary operator '==' (line 328)
    result_eq_18295 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 15), '==', subscript_call_result_18293, _AFTER_SEQ_18294)
    
    # Testing the type of an if condition (line 328)
    if_condition_18296 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 328, 12), result_eq_18295)
    # Assigning a type to the variable 'if_condition_18296' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'if_condition_18296', if_condition_18296)
    # SSA begins for if statement (line 328)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 329):
    
    # Assigning a Call to a Name (line 329):
    
    # Call to copy(...): (line 329)
    # Processing the call arguments (line 329)
    # Getting the type of 'read' (line 329)
    read_18299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 32), 'read', False)
    # Processing the call keyword arguments (line 329)
    kwargs_18300 = {}
    # Getting the type of 'copy' (line 329)
    copy_18297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 22), 'copy', False)
    # Obtaining the member 'copy' of a type (line 329)
    copy_18298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 22), copy_18297, 'copy')
    # Calling copy(args, kwargs) (line 329)
    copy_call_result_18301 = invoke(stypy.reporting.localization.Localization(__file__, 329, 22), copy_18298, *[read_18299], **kwargs_18300)
    
    # Assigning a type to the variable 'saw' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'saw', copy_call_result_18301)
    
    
    
    # Obtaining the type of the subscript
    int_18302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 25), 'int')
    slice_18303 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 330, 19), None, int_18302, None)
    # Getting the type of 'read' (line 330)
    read_18304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 19), 'read')
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___18305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 19), read_18304, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_18306 = invoke(stypy.reporting.localization.Localization(__file__, 330, 19), getitem___18305, slice_18303)
    
    
    # Obtaining the type of the subscript
    int_18307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 44), 'int')
    slice_18308 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 330, 32), int_18307, None, None)
    # Getting the type of '_BEFORE_SEQ' (line 330)
    _BEFORE_SEQ_18309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 32), '_BEFORE_SEQ')
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___18310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 32), _BEFORE_SEQ_18309, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_18311 = invoke(stypy.reporting.localization.Localization(__file__, 330, 32), getitem___18310, slice_18308)
    
    # Applying the binary operator '==' (line 330)
    result_eq_18312 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 19), '==', subscript_call_result_18306, subscript_call_result_18311)
    
    # Testing the type of an if condition (line 330)
    if_condition_18313 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 16), result_eq_18312)
    # Assigning a type to the variable 'if_condition_18313' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'if_condition_18313', if_condition_18313)
    # SSA begins for if statement (line 330)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_18314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 28), 'int')
    int_18315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 31), 'int')
    slice_18316 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 331, 23), int_18314, int_18315, None)
    # Getting the type of 'read' (line 331)
    read_18317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 23), 'read')
    # Obtaining the member '__getitem__' of a type (line 331)
    getitem___18318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 23), read_18317, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 331)
    subscript_call_result_18319 = invoke(stypy.reporting.localization.Localization(__file__, 331, 23), getitem___18318, slice_18316)
    
    # Getting the type of '_INTEL_EXTENDED_12B' (line 331)
    _INTEL_EXTENDED_12B_18320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 38), '_INTEL_EXTENDED_12B')
    # Applying the binary operator '==' (line 331)
    result_eq_18321 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 23), '==', subscript_call_result_18319, _INTEL_EXTENDED_12B_18320)
    
    # Testing the type of an if condition (line 331)
    if_condition_18322 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 331, 20), result_eq_18321)
    # Assigning a type to the variable 'if_condition_18322' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 20), 'if_condition_18322', if_condition_18322)
    # SSA begins for if statement (line 331)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_18323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 31), 'str', 'INTEL_EXTENDED_12_BYTES_LE')
    # Assigning a type to the variable 'stypy_return_type' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 24), 'stypy_return_type', str_18323)
    # SSA join for if statement (line 331)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_18324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 28), 'int')
    int_18325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 31), 'int')
    slice_18326 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 333, 23), int_18324, int_18325, None)
    # Getting the type of 'read' (line 333)
    read_18327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 23), 'read')
    # Obtaining the member '__getitem__' of a type (line 333)
    getitem___18328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 23), read_18327, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 333)
    subscript_call_result_18329 = invoke(stypy.reporting.localization.Localization(__file__, 333, 23), getitem___18328, slice_18326)
    
    # Getting the type of '_MOTOROLA_EXTENDED_12B' (line 333)
    _MOTOROLA_EXTENDED_12B_18330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 38), '_MOTOROLA_EXTENDED_12B')
    # Applying the binary operator '==' (line 333)
    result_eq_18331 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 23), '==', subscript_call_result_18329, _MOTOROLA_EXTENDED_12B_18330)
    
    # Testing the type of an if condition (line 333)
    if_condition_18332 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 333, 20), result_eq_18331)
    # Assigning a type to the variable 'if_condition_18332' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 20), 'if_condition_18332', if_condition_18332)
    # SSA begins for if statement (line 333)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_18333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 31), 'str', 'MOTOROLA_EXTENDED_12_BYTES_BE')
    # Assigning a type to the variable 'stypy_return_type' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 24), 'stypy_return_type', str_18333)
    # SSA join for if statement (line 333)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 330)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_18334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 27), 'int')
    slice_18335 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 335, 21), None, int_18334, None)
    # Getting the type of 'read' (line 335)
    read_18336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 21), 'read')
    # Obtaining the member '__getitem__' of a type (line 335)
    getitem___18337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 21), read_18336, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 335)
    subscript_call_result_18338 = invoke(stypy.reporting.localization.Localization(__file__, 335, 21), getitem___18337, slice_18335)
    
    
    # Obtaining the type of the subscript
    int_18339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 45), 'int')
    slice_18340 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 335, 33), int_18339, None, None)
    # Getting the type of '_BEFORE_SEQ' (line 335)
    _BEFORE_SEQ_18341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 33), '_BEFORE_SEQ')
    # Obtaining the member '__getitem__' of a type (line 335)
    getitem___18342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 33), _BEFORE_SEQ_18341, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 335)
    subscript_call_result_18343 = invoke(stypy.reporting.localization.Localization(__file__, 335, 33), getitem___18342, slice_18340)
    
    # Applying the binary operator '==' (line 335)
    result_eq_18344 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 21), '==', subscript_call_result_18338, subscript_call_result_18343)
    
    # Testing the type of an if condition (line 335)
    if_condition_18345 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 335, 21), result_eq_18344)
    # Assigning a type to the variable 'if_condition_18345' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 21), 'if_condition_18345', if_condition_18345)
    # SSA begins for if statement (line 335)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_18346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 28), 'int')
    int_18347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 30), 'int')
    slice_18348 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 336, 23), int_18346, int_18347, None)
    # Getting the type of 'read' (line 336)
    read_18349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 23), 'read')
    # Obtaining the member '__getitem__' of a type (line 336)
    getitem___18350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 23), read_18349, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 336)
    subscript_call_result_18351 = invoke(stypy.reporting.localization.Localization(__file__, 336, 23), getitem___18350, slice_18348)
    
    # Getting the type of '_INTEL_EXTENDED_16B' (line 336)
    _INTEL_EXTENDED_16B_18352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 37), '_INTEL_EXTENDED_16B')
    # Applying the binary operator '==' (line 336)
    result_eq_18353 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 23), '==', subscript_call_result_18351, _INTEL_EXTENDED_16B_18352)
    
    # Testing the type of an if condition (line 336)
    if_condition_18354 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 336, 20), result_eq_18353)
    # Assigning a type to the variable 'if_condition_18354' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 20), 'if_condition_18354', if_condition_18354)
    # SSA begins for if statement (line 336)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_18355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 31), 'str', 'INTEL_EXTENDED_16_BYTES_LE')
    # Assigning a type to the variable 'stypy_return_type' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 24), 'stypy_return_type', str_18355)
    # SSA branch for the else part of an if statement (line 336)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_18356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 30), 'int')
    int_18357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 32), 'int')
    slice_18358 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 338, 25), int_18356, int_18357, None)
    # Getting the type of 'read' (line 338)
    read_18359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 25), 'read')
    # Obtaining the member '__getitem__' of a type (line 338)
    getitem___18360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 25), read_18359, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 338)
    subscript_call_result_18361 = invoke(stypy.reporting.localization.Localization(__file__, 338, 25), getitem___18360, slice_18358)
    
    # Getting the type of '_IEEE_QUAD_PREC_BE' (line 338)
    _IEEE_QUAD_PREC_BE_18362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 39), '_IEEE_QUAD_PREC_BE')
    # Applying the binary operator '==' (line 338)
    result_eq_18363 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 25), '==', subscript_call_result_18361, _IEEE_QUAD_PREC_BE_18362)
    
    # Testing the type of an if condition (line 338)
    if_condition_18364 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 338, 25), result_eq_18363)
    # Assigning a type to the variable 'if_condition_18364' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 25), 'if_condition_18364', if_condition_18364)
    # SSA begins for if statement (line 338)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_18365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 31), 'str', 'IEEE_QUAD_BE')
    # Assigning a type to the variable 'stypy_return_type' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 24), 'stypy_return_type', str_18365)
    # SSA branch for the else part of an if statement (line 338)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_18366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 30), 'int')
    int_18367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 32), 'int')
    slice_18368 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 340, 25), int_18366, int_18367, None)
    # Getting the type of 'read' (line 340)
    read_18369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 25), 'read')
    # Obtaining the member '__getitem__' of a type (line 340)
    getitem___18370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 25), read_18369, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 340)
    subscript_call_result_18371 = invoke(stypy.reporting.localization.Localization(__file__, 340, 25), getitem___18370, slice_18368)
    
    # Getting the type of '_IEEE_QUAD_PREC_LE' (line 340)
    _IEEE_QUAD_PREC_LE_18372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 39), '_IEEE_QUAD_PREC_LE')
    # Applying the binary operator '==' (line 340)
    result_eq_18373 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 25), '==', subscript_call_result_18371, _IEEE_QUAD_PREC_LE_18372)
    
    # Testing the type of an if condition (line 340)
    if_condition_18374 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 25), result_eq_18373)
    # Assigning a type to the variable 'if_condition_18374' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 25), 'if_condition_18374', if_condition_18374)
    # SSA begins for if statement (line 340)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_18375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 31), 'str', 'IEEE_QUAD_LE')
    # Assigning a type to the variable 'stypy_return_type' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 24), 'stypy_return_type', str_18375)
    # SSA branch for the else part of an if statement (line 340)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_18376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 30), 'int')
    int_18377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 32), 'int')
    slice_18378 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 342, 25), int_18376, int_18377, None)
    # Getting the type of 'read' (line 342)
    read_18379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 25), 'read')
    # Obtaining the member '__getitem__' of a type (line 342)
    getitem___18380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 25), read_18379, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 342)
    subscript_call_result_18381 = invoke(stypy.reporting.localization.Localization(__file__, 342, 25), getitem___18380, slice_18378)
    
    # Getting the type of '_DOUBLE_DOUBLE_BE' (line 342)
    _DOUBLE_DOUBLE_BE_18382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 39), '_DOUBLE_DOUBLE_BE')
    # Applying the binary operator '==' (line 342)
    result_eq_18383 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 25), '==', subscript_call_result_18381, _DOUBLE_DOUBLE_BE_18382)
    
    # Testing the type of an if condition (line 342)
    if_condition_18384 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 342, 25), result_eq_18383)
    # Assigning a type to the variable 'if_condition_18384' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 25), 'if_condition_18384', if_condition_18384)
    # SSA begins for if statement (line 342)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_18385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 31), 'str', 'DOUBLE_DOUBLE_BE')
    # Assigning a type to the variable 'stypy_return_type' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 24), 'stypy_return_type', str_18385)
    # SSA branch for the else part of an if statement (line 342)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_18386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 30), 'int')
    int_18387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 32), 'int')
    slice_18388 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 344, 25), int_18386, int_18387, None)
    # Getting the type of 'read' (line 344)
    read_18389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 25), 'read')
    # Obtaining the member '__getitem__' of a type (line 344)
    getitem___18390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 25), read_18389, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 344)
    subscript_call_result_18391 = invoke(stypy.reporting.localization.Localization(__file__, 344, 25), getitem___18390, slice_18388)
    
    # Getting the type of '_DOUBLE_DOUBLE_LE' (line 344)
    _DOUBLE_DOUBLE_LE_18392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 39), '_DOUBLE_DOUBLE_LE')
    # Applying the binary operator '==' (line 344)
    result_eq_18393 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 25), '==', subscript_call_result_18391, _DOUBLE_DOUBLE_LE_18392)
    
    # Testing the type of an if condition (line 344)
    if_condition_18394 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 344, 25), result_eq_18393)
    # Assigning a type to the variable 'if_condition_18394' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 25), 'if_condition_18394', if_condition_18394)
    # SSA begins for if statement (line 344)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_18395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 31), 'str', 'DOUBLE_DOUBLE_LE')
    # Assigning a type to the variable 'stypy_return_type' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 24), 'stypy_return_type', str_18395)
    # SSA join for if statement (line 344)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 342)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 340)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 338)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 336)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 335)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_18396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 27), 'int')
    slice_18397 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 346, 21), None, int_18396, None)
    # Getting the type of 'read' (line 346)
    read_18398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 21), 'read')
    # Obtaining the member '__getitem__' of a type (line 346)
    getitem___18399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 21), read_18398, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 346)
    subscript_call_result_18400 = invoke(stypy.reporting.localization.Localization(__file__, 346, 21), getitem___18399, slice_18397)
    
    # Getting the type of '_BEFORE_SEQ' (line 346)
    _BEFORE_SEQ_18401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 34), '_BEFORE_SEQ')
    # Applying the binary operator '==' (line 346)
    result_eq_18402 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 21), '==', subscript_call_result_18400, _BEFORE_SEQ_18401)
    
    # Testing the type of an if condition (line 346)
    if_condition_18403 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 346, 21), result_eq_18402)
    # Assigning a type to the variable 'if_condition_18403' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 21), 'if_condition_18403', if_condition_18403)
    # SSA begins for if statement (line 346)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_18404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 28), 'int')
    int_18405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 31), 'int')
    slice_18406 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 347, 23), int_18404, int_18405, None)
    # Getting the type of 'read' (line 347)
    read_18407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 23), 'read')
    # Obtaining the member '__getitem__' of a type (line 347)
    getitem___18408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 23), read_18407, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 347)
    subscript_call_result_18409 = invoke(stypy.reporting.localization.Localization(__file__, 347, 23), getitem___18408, slice_18406)
    
    # Getting the type of '_IEEE_DOUBLE_LE' (line 347)
    _IEEE_DOUBLE_LE_18410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 38), '_IEEE_DOUBLE_LE')
    # Applying the binary operator '==' (line 347)
    result_eq_18411 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 23), '==', subscript_call_result_18409, _IEEE_DOUBLE_LE_18410)
    
    # Testing the type of an if condition (line 347)
    if_condition_18412 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 347, 20), result_eq_18411)
    # Assigning a type to the variable 'if_condition_18412' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 20), 'if_condition_18412', if_condition_18412)
    # SSA begins for if statement (line 347)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_18413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 31), 'str', 'IEEE_DOUBLE_LE')
    # Assigning a type to the variable 'stypy_return_type' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 24), 'stypy_return_type', str_18413)
    # SSA branch for the else part of an if statement (line 347)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_18414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 30), 'int')
    int_18415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 33), 'int')
    slice_18416 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 349, 25), int_18414, int_18415, None)
    # Getting the type of 'read' (line 349)
    read_18417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 25), 'read')
    # Obtaining the member '__getitem__' of a type (line 349)
    getitem___18418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 25), read_18417, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 349)
    subscript_call_result_18419 = invoke(stypy.reporting.localization.Localization(__file__, 349, 25), getitem___18418, slice_18416)
    
    # Getting the type of '_IEEE_DOUBLE_BE' (line 349)
    _IEEE_DOUBLE_BE_18420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 40), '_IEEE_DOUBLE_BE')
    # Applying the binary operator '==' (line 349)
    result_eq_18421 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 25), '==', subscript_call_result_18419, _IEEE_DOUBLE_BE_18420)
    
    # Testing the type of an if condition (line 349)
    if_condition_18422 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 349, 25), result_eq_18421)
    # Assigning a type to the variable 'if_condition_18422' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 25), 'if_condition_18422', if_condition_18422)
    # SSA begins for if statement (line 349)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_18423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 31), 'str', 'IEEE_DOUBLE_BE')
    # Assigning a type to the variable 'stypy_return_type' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 24), 'stypy_return_type', str_18423)
    # SSA join for if statement (line 349)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 347)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 346)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 335)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 330)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 328)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 352)
    # Getting the type of 'saw' (line 352)
    saw_18424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'saw')
    # Getting the type of 'None' (line 352)
    None_18425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 18), 'None')
    
    (may_be_18426, more_types_in_union_18427) = may_not_be_none(saw_18424, None_18425)

    if may_be_18426:

        if more_types_in_union_18427:
            # Runtime conditional SSA (line 352)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 353)
        # Processing the call arguments (line 353)
        str_18429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 25), 'str', 'Unrecognized format (%s)')
        # Getting the type of 'saw' (line 353)
        saw_18430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 54), 'saw', False)
        # Applying the binary operator '%' (line 353)
        result_mod_18431 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 25), '%', str_18429, saw_18430)
        
        # Processing the call keyword arguments (line 353)
        kwargs_18432 = {}
        # Getting the type of 'ValueError' (line 353)
        ValueError_18428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 353)
        ValueError_call_result_18433 = invoke(stypy.reporting.localization.Localization(__file__, 353, 14), ValueError_18428, *[result_mod_18431], **kwargs_18432)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 353, 8), ValueError_call_result_18433, 'raise parameter', BaseException)

        if more_types_in_union_18427:
            # Runtime conditional SSA for else branch (line 352)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_18426) or more_types_in_union_18427):
        
        # Call to ValueError(...): (line 356)
        # Processing the call arguments (line 356)
        str_18435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 25), 'str', 'Could not lock sequences (%s)')
        # Getting the type of 'saw' (line 356)
        saw_18436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 59), 'saw', False)
        # Applying the binary operator '%' (line 356)
        result_mod_18437 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 25), '%', str_18435, saw_18436)
        
        # Processing the call keyword arguments (line 356)
        kwargs_18438 = {}
        # Getting the type of 'ValueError' (line 356)
        ValueError_18434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 356)
        ValueError_call_result_18439 = invoke(stypy.reporting.localization.Localization(__file__, 356, 14), ValueError_18434, *[result_mod_18437], **kwargs_18438)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 356, 8), ValueError_call_result_18439, 'raise parameter', BaseException)

        if (may_be_18426 and more_types_in_union_18427):
            # SSA join for if statement (line 352)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'long_double_representation(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'long_double_representation' in the type store
    # Getting the type of 'stypy_return_type' (line 306)
    stypy_return_type_18440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18440)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'long_double_representation'
    return stypy_return_type_18440

# Assigning a type to the variable 'long_double_representation' (line 306)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 0), 'long_double_representation', long_double_representation)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
