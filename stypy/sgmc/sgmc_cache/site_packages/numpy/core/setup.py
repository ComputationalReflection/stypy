
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function
2: 
3: import imp
4: import os
5: import sys
6: import pickle
7: import copy
8: import warnings
9: from os.path import join
10: from numpy.distutils import log
11: from distutils.dep_util import newer
12: from distutils.sysconfig import get_config_var
13: from numpy._build_utils.apple_accelerate import (uses_accelerate_framework,
14:                                                  get_sgemv_fix)
15: 
16: from setup_common import *
17: 
18: # Set to True to enable relaxed strides checking. This (mostly) means
19: # that `strides[dim]` is ignored if `shape[dim] == 1` when setting flags.
20: NPY_RELAXED_STRIDES_CHECKING = (os.environ.get('NPY_RELAXED_STRIDES_CHECKING', "0") != "0")
21: 
22: # XXX: ugly, we use a class to avoid calling twice some expensive functions in
23: # config.h/numpyconfig.h. I don't see a better way because distutils force
24: # config.h generation inside an Extension class, and as such sharing
25: # configuration informations between extensions is not easy.
26: # Using a pickled-based memoize does not work because config_cmd is an instance
27: # method, which cPickle does not like.
28: #
29: # Use pickle in all cases, as cPickle is gone in python3 and the difference
30: # in time is only in build. -- Charles Harris, 2013-03-30
31: 
32: class CallOnceOnly(object):
33:     def __init__(self):
34:         self._check_types = None
35:         self._check_ieee_macros = None
36:         self._check_complex = None
37: 
38:     def check_types(self, *a, **kw):
39:         if self._check_types is None:
40:             out = check_types(*a, **kw)
41:             self._check_types = pickle.dumps(out)
42:         else:
43:             out = copy.deepcopy(pickle.loads(self._check_types))
44:         return out
45: 
46:     def check_ieee_macros(self, *a, **kw):
47:         if self._check_ieee_macros is None:
48:             out = check_ieee_macros(*a, **kw)
49:             self._check_ieee_macros = pickle.dumps(out)
50:         else:
51:             out = copy.deepcopy(pickle.loads(self._check_ieee_macros))
52:         return out
53: 
54:     def check_complex(self, *a, **kw):
55:         if self._check_complex is None:
56:             out = check_complex(*a, **kw)
57:             self._check_complex = pickle.dumps(out)
58:         else:
59:             out = copy.deepcopy(pickle.loads(self._check_complex))
60:         return out
61: 
62: PYTHON_HAS_UNICODE_WIDE = True
63: 
64: def pythonlib_dir():
65:     '''return path where libpython* is.'''
66:     if sys.platform == 'win32':
67:         return os.path.join(sys.prefix, "libs")
68:     else:
69:         return get_config_var('LIBDIR')
70: 
71: def is_npy_no_signal():
72:     '''Return True if the NPY_NO_SIGNAL symbol must be defined in configuration
73:     header.'''
74:     return sys.platform == 'win32'
75: 
76: def is_npy_no_smp():
77:     '''Return True if the NPY_NO_SMP symbol must be defined in public
78:     header (when SMP support cannot be reliably enabled).'''
79:     # Perhaps a fancier check is in order here.
80:     #  so that threads are only enabled if there
81:     #  are actually multiple CPUS? -- but
82:     #  threaded code can be nice even on a single
83:     #  CPU so that long-calculating code doesn't
84:     #  block.
85:     return 'NPY_NOSMP' in os.environ
86: 
87: def win32_checks(deflist):
88:     from numpy.distutils.misc_util import get_build_architecture
89:     a = get_build_architecture()
90: 
91:     # Distutils hack on AMD64 on windows
92:     print('BUILD_ARCHITECTURE: %r, os.name=%r, sys.platform=%r' %
93:           (a, os.name, sys.platform))
94:     if a == 'AMD64':
95:         deflist.append('DISTUTILS_USE_SDK')
96: 
97:     # On win32, force long double format string to be 'g', not
98:     # 'Lg', since the MS runtime does not support long double whose
99:     # size is > sizeof(double)
100:     if a == "Intel" or a == "AMD64":
101:         deflist.append('FORCE_NO_LONG_DOUBLE_FORMATTING')
102: 
103: def check_math_capabilities(config, moredefs, mathlibs):
104:     def check_func(func_name):
105:         return config.check_func(func_name, libraries=mathlibs,
106:                                  decl=True, call=True)
107: 
108:     def check_funcs_once(funcs_name):
109:         decl = dict([(f, True) for f in funcs_name])
110:         st = config.check_funcs_once(funcs_name, libraries=mathlibs,
111:                                      decl=decl, call=decl)
112:         if st:
113:             moredefs.extend([(fname2def(f), 1) for f in funcs_name])
114:         return st
115: 
116:     def check_funcs(funcs_name):
117:         # Use check_funcs_once first, and if it does not work, test func per
118:         # func. Return success only if all the functions are available
119:         if not check_funcs_once(funcs_name):
120:             # Global check failed, check func per func
121:             for f in funcs_name:
122:                 if check_func(f):
123:                     moredefs.append((fname2def(f), 1))
124:             return 0
125:         else:
126:             return 1
127: 
128:     #use_msvc = config.check_decl("_MSC_VER")
129: 
130:     if not check_funcs_once(MANDATORY_FUNCS):
131:         raise SystemError("One of the required function to build numpy is not"
132:                 " available (the list is %s)." % str(MANDATORY_FUNCS))
133: 
134:     # Standard functions which may not be available and for which we have a
135:     # replacement implementation. Note that some of these are C99 functions.
136: 
137:     # XXX: hack to circumvent cpp pollution from python: python put its
138:     # config.h in the public namespace, so we have a clash for the common
139:     # functions we test. We remove every function tested by python's
140:     # autoconf, hoping their own test are correct
141:     for f in OPTIONAL_STDFUNCS_MAYBE:
142:         if config.check_decl(fname2def(f),
143:                     headers=["Python.h", "math.h"]):
144:             OPTIONAL_STDFUNCS.remove(f)
145: 
146:     check_funcs(OPTIONAL_STDFUNCS)
147: 
148:     for h in OPTIONAL_HEADERS:
149:         if config.check_func("", decl=False, call=False, headers=[h]):
150:             moredefs.append((fname2def(h).replace(".", "_"), 1))
151: 
152:     for tup in OPTIONAL_INTRINSICS:
153:         headers = None
154:         if len(tup) == 2:
155:             f, args = tup
156:         else:
157:             f, args, headers = tup[0], tup[1], [tup[2]]
158:         if config.check_func(f, decl=False, call=True, call_args=args,
159:                              headers=headers):
160:             moredefs.append((fname2def(f), 1))
161: 
162:     for dec, fn in OPTIONAL_FUNCTION_ATTRIBUTES:
163:         if config.check_gcc_function_attribute(dec, fn):
164:             moredefs.append((fname2def(fn), 1))
165: 
166:     for fn in OPTIONAL_VARIABLE_ATTRIBUTES:
167:         if config.check_gcc_variable_attribute(fn):
168:             m = fn.replace("(", "_").replace(")", "_")
169:             moredefs.append((fname2def(m), 1))
170: 
171:     # C99 functions: float and long double versions
172:     check_funcs(C99_FUNCS_SINGLE)
173:     check_funcs(C99_FUNCS_EXTENDED)
174: 
175: def check_complex(config, mathlibs):
176:     priv = []
177:     pub = []
178: 
179:     try:
180:         if os.uname()[0] == "Interix":
181:             warnings.warn("Disabling broken complex support. See #1365")
182:             return priv, pub
183:     except:
184:         # os.uname not available on all platforms. blanket except ugly but safe
185:         pass
186: 
187:     # Check for complex support
188:     st = config.check_header('complex.h')
189:     if st:
190:         priv.append(('HAVE_COMPLEX_H', 1))
191:         pub.append(('NPY_USE_C99_COMPLEX', 1))
192: 
193:         for t in C99_COMPLEX_TYPES:
194:             st = config.check_type(t, headers=["complex.h"])
195:             if st:
196:                 pub.append(('NPY_HAVE_%s' % type2def(t), 1))
197: 
198:         def check_prec(prec):
199:             flist = [f + prec for f in C99_COMPLEX_FUNCS]
200:             decl = dict([(f, True) for f in flist])
201:             if not config.check_funcs_once(flist, call=decl, decl=decl,
202:                                            libraries=mathlibs):
203:                 for f in flist:
204:                     if config.check_func(f, call=True, decl=True,
205:                                          libraries=mathlibs):
206:                         priv.append((fname2def(f), 1))
207:             else:
208:                 priv.extend([(fname2def(f), 1) for f in flist])
209: 
210:         check_prec('')
211:         check_prec('f')
212:         check_prec('l')
213: 
214:     return priv, pub
215: 
216: def check_ieee_macros(config):
217:     priv = []
218:     pub = []
219: 
220:     macros = []
221: 
222:     def _add_decl(f):
223:         priv.append(fname2def("decl_%s" % f))
224:         pub.append('NPY_%s' % fname2def("decl_%s" % f))
225: 
226:     # XXX: hack to circumvent cpp pollution from python: python put its
227:     # config.h in the public namespace, so we have a clash for the common
228:     # functions we test. We remove every function tested by python's
229:     # autoconf, hoping their own test are correct
230:     _macros = ["isnan", "isinf", "signbit", "isfinite"]
231:     for f in _macros:
232:         py_symbol = fname2def("decl_%s" % f)
233:         already_declared = config.check_decl(py_symbol,
234:                 headers=["Python.h", "math.h"])
235:         if already_declared:
236:             if config.check_macro_true(py_symbol,
237:                     headers=["Python.h", "math.h"]):
238:                 pub.append('NPY_%s' % fname2def("decl_%s" % f))
239:         else:
240:             macros.append(f)
241:     # Normally, isnan and isinf are macro (C99), but some platforms only have
242:     # func, or both func and macro version. Check for macro only, and define
243:     # replacement ones if not found.
244:     # Note: including Python.h is necessary because it modifies some math.h
245:     # definitions
246:     for f in macros:
247:         st = config.check_decl(f, headers=["Python.h", "math.h"])
248:         if st:
249:             _add_decl(f)
250: 
251:     return priv, pub
252: 
253: def check_types(config_cmd, ext, build_dir):
254:     private_defines = []
255:     public_defines = []
256: 
257:     # Expected size (in number of bytes) for each type. This is an
258:     # optimization: those are only hints, and an exhaustive search for the size
259:     # is done if the hints are wrong.
260:     expected = {'short': [2], 'int': [4], 'long': [8, 4],
261:                 'float': [4], 'double': [8], 'long double': [16, 12, 8],
262:                 'Py_intptr_t': [8, 4], 'PY_LONG_LONG': [8], 'long long': [8],
263:                 'off_t': [8, 4]}
264: 
265:     # Check we have the python header (-dev* packages on Linux)
266:     result = config_cmd.check_header('Python.h')
267:     if not result:
268:         raise SystemError(
269:                 "Cannot compile 'Python.h'. Perhaps you need to "
270:                 "install python-dev|python-devel.")
271:     res = config_cmd.check_header("endian.h")
272:     if res:
273:         private_defines.append(('HAVE_ENDIAN_H', 1))
274:         public_defines.append(('NPY_HAVE_ENDIAN_H', 1))
275: 
276:     # Check basic types sizes
277:     for type in ('short', 'int', 'long'):
278:         res = config_cmd.check_decl("SIZEOF_%s" % sym2def(type), headers=["Python.h"])
279:         if res:
280:             public_defines.append(('NPY_SIZEOF_%s' % sym2def(type), "SIZEOF_%s" % sym2def(type)))
281:         else:
282:             res = config_cmd.check_type_size(type, expected=expected[type])
283:             if res >= 0:
284:                 public_defines.append(('NPY_SIZEOF_%s' % sym2def(type), '%d' % res))
285:             else:
286:                 raise SystemError("Checking sizeof (%s) failed !" % type)
287: 
288:     for type in ('float', 'double', 'long double'):
289:         already_declared = config_cmd.check_decl("SIZEOF_%s" % sym2def(type),
290:                                                  headers=["Python.h"])
291:         res = config_cmd.check_type_size(type, expected=expected[type])
292:         if res >= 0:
293:             public_defines.append(('NPY_SIZEOF_%s' % sym2def(type), '%d' % res))
294:             if not already_declared and not type == 'long double':
295:                 private_defines.append(('SIZEOF_%s' % sym2def(type), '%d' % res))
296:         else:
297:             raise SystemError("Checking sizeof (%s) failed !" % type)
298: 
299:         # Compute size of corresponding complex type: used to check that our
300:         # definition is binary compatible with C99 complex type (check done at
301:         # build time in npy_common.h)
302:         complex_def = "struct {%s __x; %s __y;}" % (type, type)
303:         res = config_cmd.check_type_size(complex_def,
304:                                          expected=[2 * x for x in expected[type]])
305:         if res >= 0:
306:             public_defines.append(('NPY_SIZEOF_COMPLEX_%s' % sym2def(type), '%d' % res))
307:         else:
308:             raise SystemError("Checking sizeof (%s) failed !" % complex_def)
309: 
310:     for type in ('Py_intptr_t', 'off_t'):
311:         res = config_cmd.check_type_size(type, headers=["Python.h"],
312:                 library_dirs=[pythonlib_dir()],
313:                 expected=expected[type])
314: 
315:         if res >= 0:
316:             private_defines.append(('SIZEOF_%s' % sym2def(type), '%d' % res))
317:             public_defines.append(('NPY_SIZEOF_%s' % sym2def(type), '%d' % res))
318:         else:
319:             raise SystemError("Checking sizeof (%s) failed !" % type)
320: 
321:     # We check declaration AND type because that's how distutils does it.
322:     if config_cmd.check_decl('PY_LONG_LONG', headers=['Python.h']):
323:         res = config_cmd.check_type_size('PY_LONG_LONG',  headers=['Python.h'],
324:                 library_dirs=[pythonlib_dir()],
325:                 expected=expected['PY_LONG_LONG'])
326:         if res >= 0:
327:             private_defines.append(('SIZEOF_%s' % sym2def('PY_LONG_LONG'), '%d' % res))
328:             public_defines.append(('NPY_SIZEOF_%s' % sym2def('PY_LONG_LONG'), '%d' % res))
329:         else:
330:             raise SystemError("Checking sizeof (%s) failed !" % 'PY_LONG_LONG')
331: 
332:         res = config_cmd.check_type_size('long long',
333:                 expected=expected['long long'])
334:         if res >= 0:
335:             #private_defines.append(('SIZEOF_%s' % sym2def('long long'), '%d' % res))
336:             public_defines.append(('NPY_SIZEOF_%s' % sym2def('long long'), '%d' % res))
337:         else:
338:             raise SystemError("Checking sizeof (%s) failed !" % 'long long')
339: 
340:     if not config_cmd.check_decl('CHAR_BIT', headers=['Python.h']):
341:         raise RuntimeError(
342:             "Config wo CHAR_BIT is not supported"
343:             ", please contact the maintainers")
344: 
345:     return private_defines, public_defines
346: 
347: def check_mathlib(config_cmd):
348:     # Testing the C math library
349:     mathlibs = []
350:     mathlibs_choices = [[], ['m'], ['cpml']]
351:     mathlib = os.environ.get('MATHLIB')
352:     if mathlib:
353:         mathlibs_choices.insert(0, mathlib.split(','))
354:     for libs in mathlibs_choices:
355:         if config_cmd.check_func("exp", libraries=libs, decl=True, call=True):
356:             mathlibs = libs
357:             break
358:     else:
359:         raise EnvironmentError("math library missing; rerun "
360:                                "setup.py after setting the "
361:                                "MATHLIB env variable")
362:     return mathlibs
363: 
364: def visibility_define(config):
365:     '''Return the define value to use for NPY_VISIBILITY_HIDDEN (may be empty
366:     string).'''
367:     if config.check_compiler_gcc4():
368:         return '__attribute__((visibility("hidden")))'
369:     else:
370:         return ''
371: 
372: def configuration(parent_package='',top_path=None):
373:     from numpy.distutils.misc_util import Configuration, dot_join
374:     from numpy.distutils.system_info import get_info
375: 
376:     config = Configuration('core', parent_package, top_path)
377:     local_dir = config.local_path
378:     codegen_dir = join(local_dir, 'code_generators')
379: 
380:     if is_released(config):
381:         warnings.simplefilter('error', MismatchCAPIWarning)
382: 
383:     # Check whether we have a mismatch between the set C API VERSION and the
384:     # actual C API VERSION
385:     check_api_version(C_API_VERSION, codegen_dir)
386: 
387:     generate_umath_py = join(codegen_dir, 'generate_umath.py')
388:     n = dot_join(config.name, 'generate_umath')
389:     generate_umath = imp.load_module('_'.join(n.split('.')),
390:                                      open(generate_umath_py, 'U'), generate_umath_py,
391:                                      ('.py', 'U', 1))
392: 
393:     header_dir = 'include/numpy'  # this is relative to config.path_in_package
394: 
395:     cocache = CallOnceOnly()
396: 
397:     def generate_config_h(ext, build_dir):
398:         target = join(build_dir, header_dir, 'config.h')
399:         d = os.path.dirname(target)
400:         if not os.path.exists(d):
401:             os.makedirs(d)
402: 
403:         if newer(__file__, target):
404:             config_cmd = config.get_config_cmd()
405:             log.info('Generating %s', target)
406: 
407:             # Check sizeof
408:             moredefs, ignored = cocache.check_types(config_cmd, ext, build_dir)
409: 
410:             # Check math library and C99 math funcs availability
411:             mathlibs = check_mathlib(config_cmd)
412:             moredefs.append(('MATHLIB', ','.join(mathlibs)))
413: 
414:             check_math_capabilities(config_cmd, moredefs, mathlibs)
415:             moredefs.extend(cocache.check_ieee_macros(config_cmd)[0])
416:             moredefs.extend(cocache.check_complex(config_cmd, mathlibs)[0])
417: 
418:             # Signal check
419:             if is_npy_no_signal():
420:                 moredefs.append('__NPY_PRIVATE_NO_SIGNAL')
421: 
422:             # Windows checks
423:             if sys.platform == 'win32' or os.name == 'nt':
424:                 win32_checks(moredefs)
425: 
426:             # C99 restrict keyword
427:             moredefs.append(('NPY_RESTRICT', config_cmd.check_restrict()))
428: 
429:             # Inline check
430:             inline = config_cmd.check_inline()
431: 
432:             # Check whether we need our own wide character support
433:             if not config_cmd.check_decl('Py_UNICODE_WIDE', headers=['Python.h']):
434:                 PYTHON_HAS_UNICODE_WIDE = True
435:             else:
436:                 PYTHON_HAS_UNICODE_WIDE = False
437: 
438:             if NPY_RELAXED_STRIDES_CHECKING:
439:                 moredefs.append(('NPY_RELAXED_STRIDES_CHECKING', 1))
440: 
441:             # Get long double representation
442:             if sys.platform != 'darwin':
443:                 rep = check_long_double_representation(config_cmd)
444:                 if rep in ['INTEL_EXTENDED_12_BYTES_LE',
445:                            'INTEL_EXTENDED_16_BYTES_LE',
446:                            'MOTOROLA_EXTENDED_12_BYTES_BE',
447:                            'IEEE_QUAD_LE', 'IEEE_QUAD_BE',
448:                            'IEEE_DOUBLE_LE', 'IEEE_DOUBLE_BE',
449:                            'DOUBLE_DOUBLE_BE', 'DOUBLE_DOUBLE_LE']:
450:                     moredefs.append(('HAVE_LDOUBLE_%s' % rep, 1))
451:                 else:
452:                     raise ValueError("Unrecognized long double format: %s" % rep)
453: 
454:             # Py3K check
455:             if sys.version_info[0] == 3:
456:                 moredefs.append(('NPY_PY3K', 1))
457: 
458:             # Generate the config.h file from moredefs
459:             target_f = open(target, 'w')
460:             for d in moredefs:
461:                 if isinstance(d, str):
462:                     target_f.write('#define %s\n' % (d))
463:                 else:
464:                     target_f.write('#define %s %s\n' % (d[0], d[1]))
465: 
466:             # define inline to our keyword, or nothing
467:             target_f.write('#ifndef __cplusplus\n')
468:             if inline == 'inline':
469:                 target_f.write('/* #undef inline */\n')
470:             else:
471:                 target_f.write('#define inline %s\n' % inline)
472:             target_f.write('#endif\n')
473: 
474:             # add the guard to make sure config.h is never included directly,
475:             # but always through npy_config.h
476:             target_f.write('''
477: #ifndef _NPY_NPY_CONFIG_H_
478: #error config.h should never be included directly, include npy_config.h instead
479: #endif
480: ''')
481: 
482:             target_f.close()
483:             print('File:', target)
484:             target_f = open(target)
485:             print(target_f.read())
486:             target_f.close()
487:             print('EOF')
488:         else:
489:             mathlibs = []
490:             target_f = open(target)
491:             for line in target_f:
492:                 s = '#define MATHLIB'
493:                 if line.startswith(s):
494:                     value = line[len(s):].strip()
495:                     if value:
496:                         mathlibs.extend(value.split(','))
497:             target_f.close()
498: 
499:         # Ugly: this can be called within a library and not an extension,
500:         # in which case there is no libraries attributes (and none is
501:         # needed).
502:         if hasattr(ext, 'libraries'):
503:             ext.libraries.extend(mathlibs)
504: 
505:         incl_dir = os.path.dirname(target)
506:         if incl_dir not in config.numpy_include_dirs:
507:             config.numpy_include_dirs.append(incl_dir)
508: 
509:         return target
510: 
511:     def generate_numpyconfig_h(ext, build_dir):
512:         '''Depends on config.h: generate_config_h has to be called before !'''
513:         # put private include directory in build_dir on search path
514:         # allows using code generation in headers headers
515:         config.add_include_dirs(join(build_dir, "src", "private"))
516: 
517:         target = join(build_dir, header_dir, '_numpyconfig.h')
518:         d = os.path.dirname(target)
519:         if not os.path.exists(d):
520:             os.makedirs(d)
521:         if newer(__file__, target):
522:             config_cmd = config.get_config_cmd()
523:             log.info('Generating %s', target)
524: 
525:             # Check sizeof
526:             ignored, moredefs = cocache.check_types(config_cmd, ext, build_dir)
527: 
528:             if is_npy_no_signal():
529:                 moredefs.append(('NPY_NO_SIGNAL', 1))
530: 
531:             if is_npy_no_smp():
532:                 moredefs.append(('NPY_NO_SMP', 1))
533:             else:
534:                 moredefs.append(('NPY_NO_SMP', 0))
535: 
536:             mathlibs = check_mathlib(config_cmd)
537:             moredefs.extend(cocache.check_ieee_macros(config_cmd)[1])
538:             moredefs.extend(cocache.check_complex(config_cmd, mathlibs)[1])
539: 
540:             if NPY_RELAXED_STRIDES_CHECKING:
541:                 moredefs.append(('NPY_RELAXED_STRIDES_CHECKING', 1))
542: 
543:             # Check wether we can use inttypes (C99) formats
544:             if config_cmd.check_decl('PRIdPTR', headers=['inttypes.h']):
545:                 moredefs.append(('NPY_USE_C99_FORMATS', 1))
546: 
547:             # visibility check
548:             hidden_visibility = visibility_define(config_cmd)
549:             moredefs.append(('NPY_VISIBILITY_HIDDEN', hidden_visibility))
550: 
551:             # Add the C API/ABI versions
552:             moredefs.append(('NPY_ABI_VERSION', '0x%.8X' % C_ABI_VERSION))
553:             moredefs.append(('NPY_API_VERSION', '0x%.8X' % C_API_VERSION))
554: 
555:             # Add moredefs to header
556:             target_f = open(target, 'w')
557:             for d in moredefs:
558:                 if isinstance(d, str):
559:                     target_f.write('#define %s\n' % (d))
560:                 else:
561:                     target_f.write('#define %s %s\n' % (d[0], d[1]))
562: 
563:             # Define __STDC_FORMAT_MACROS
564:             target_f.write('''
565: #ifndef __STDC_FORMAT_MACROS
566: #define __STDC_FORMAT_MACROS 1
567: #endif
568: ''')
569:             target_f.close()
570: 
571:             # Dump the numpyconfig.h header to stdout
572:             print('File: %s' % target)
573:             target_f = open(target)
574:             print(target_f.read())
575:             target_f.close()
576:             print('EOF')
577:         config.add_data_files((header_dir, target))
578:         return target
579: 
580:     def generate_api_func(module_name):
581:         def generate_api(ext, build_dir):
582:             script = join(codegen_dir, module_name + '.py')
583:             sys.path.insert(0, codegen_dir)
584:             try:
585:                 m = __import__(module_name)
586:                 log.info('executing %s', script)
587:                 h_file, c_file, doc_file = m.generate_api(os.path.join(build_dir, header_dir))
588:             finally:
589:                 del sys.path[0]
590:             config.add_data_files((header_dir, h_file),
591:                                   (header_dir, doc_file))
592:             return (h_file,)
593:         return generate_api
594: 
595:     generate_numpy_api = generate_api_func('generate_numpy_api')
596:     generate_ufunc_api = generate_api_func('generate_ufunc_api')
597: 
598:     config.add_include_dirs(join(local_dir, "src", "private"))
599:     config.add_include_dirs(join(local_dir, "src"))
600:     config.add_include_dirs(join(local_dir))
601: 
602:     config.add_data_files('include/numpy/*.h')
603:     config.add_include_dirs(join('src', 'npymath'))
604:     config.add_include_dirs(join('src', 'multiarray'))
605:     config.add_include_dirs(join('src', 'umath'))
606:     config.add_include_dirs(join('src', 'npysort'))
607: 
608:     config.add_define_macros([("HAVE_NPY_CONFIG_H", "1")])
609:     config.add_define_macros([("_FILE_OFFSET_BITS", "64")])
610:     config.add_define_macros([('_LARGEFILE_SOURCE', '1')])
611:     config.add_define_macros([('_LARGEFILE64_SOURCE', '1')])
612: 
613:     config.numpy_include_dirs.extend(config.paths('include'))
614: 
615:     deps = [join('src', 'npymath', '_signbit.c'),
616:             join('include', 'numpy', '*object.h'),
617:             join(codegen_dir, 'genapi.py'),
618:             ]
619: 
620:     #######################################################################
621:     #                            dummy module                             #
622:     #######################################################################
623: 
624:     # npymath needs the config.h and numpyconfig.h files to be generated, but
625:     # build_clib cannot handle generate_config_h and generate_numpyconfig_h
626:     # (don't ask). Because clib are generated before extensions, we have to
627:     # explicitly add an extension which has generate_config_h and
628:     # generate_numpyconfig_h as sources *before* adding npymath.
629: 
630:     config.add_extension('_dummy',
631:                          sources=[join('src', 'dummymodule.c'),
632:                                   generate_config_h,
633:                                   generate_numpyconfig_h,
634:                                   generate_numpy_api]
635:                          )
636: 
637:     #######################################################################
638:     #                          npymath library                            #
639:     #######################################################################
640: 
641:     subst_dict = dict([("sep", os.path.sep), ("pkgname", "numpy.core")])
642: 
643:     def get_mathlib_info(*args):
644:         # Another ugly hack: the mathlib info is known once build_src is run,
645:         # but we cannot use add_installed_pkg_config here either, so we only
646:         # update the substition dictionary during npymath build
647:         config_cmd = config.get_config_cmd()
648: 
649:         # Check that the toolchain works, to fail early if it doesn't
650:         # (avoid late errors with MATHLIB which are confusing if the
651:         # compiler does not work).
652:         st = config_cmd.try_link('int main(void) { return 0;}')
653:         if not st:
654:             raise RuntimeError("Broken toolchain: cannot link a simple C program")
655:         mlibs = check_mathlib(config_cmd)
656: 
657:         posix_mlib = ' '.join(['-l%s' % l for l in mlibs])
658:         msvc_mlib = ' '.join(['%s.lib' % l for l in mlibs])
659:         subst_dict["posix_mathlib"] = posix_mlib
660:         subst_dict["msvc_mathlib"] = msvc_mlib
661: 
662:     npymath_sources = [join('src', 'npymath', 'npy_math.c.src'),
663:                        join('src', 'npymath', 'ieee754.c.src'),
664:                        join('src', 'npymath', 'npy_math_complex.c.src'),
665:                        join('src', 'npymath', 'halffloat.c')
666:                        ]
667:     config.add_installed_library('npymath',
668:             sources=npymath_sources + [get_mathlib_info],
669:             install_dir='lib')
670:     config.add_npy_pkg_config("npymath.ini.in", "lib/npy-pkg-config",
671:             subst_dict)
672:     config.add_npy_pkg_config("mlib.ini.in", "lib/npy-pkg-config",
673:             subst_dict)
674: 
675:     #######################################################################
676:     #                         npysort library                             #
677:     #######################################################################
678: 
679:     # This library is created for the build but it is not installed
680:     npysort_sources = [join('src', 'npysort', 'quicksort.c.src'),
681:                        join('src', 'npysort', 'mergesort.c.src'),
682:                        join('src', 'npysort', 'heapsort.c.src'),
683:                        join('src', 'private', 'npy_partition.h.src'),
684:                        join('src', 'npysort', 'selection.c.src'),
685:                        join('src', 'private', 'npy_binsearch.h.src'),
686:                        join('src', 'npysort', 'binsearch.c.src'),
687:                        ]
688:     config.add_library('npysort',
689:                        sources=npysort_sources,
690:                        include_dirs=[])
691: 
692:     #######################################################################
693:     #                        multiarray module                            #
694:     #######################################################################
695: 
696:     # Multiarray version: this function is needed to build foo.c from foo.c.src
697:     # when foo.c is included in another file and as such not in the src
698:     # argument of build_ext command
699:     def generate_multiarray_templated_sources(ext, build_dir):
700:         from numpy.distutils.misc_util import get_cmd
701: 
702:         subpath = join('src', 'multiarray')
703:         sources = [join(local_dir, subpath, 'scalartypes.c.src'),
704:                    join(local_dir, subpath, 'arraytypes.c.src'),
705:                    join(local_dir, subpath, 'nditer_templ.c.src'),
706:                    join(local_dir, subpath, 'lowlevel_strided_loops.c.src'),
707:                    join(local_dir, subpath, 'einsum.c.src'),
708:                    join(local_dir, 'src', 'private', 'templ_common.h.src')
709:                    ]
710: 
711:         # numpy.distutils generate .c from .c.src in weird directories, we have
712:         # to add them there as they depend on the build_dir
713:         config.add_include_dirs(join(build_dir, subpath))
714:         cmd = get_cmd('build_src')
715:         cmd.ensure_finalized()
716:         cmd.template_sources(sources, ext)
717: 
718:     multiarray_deps = [
719:             join('src', 'multiarray', 'arrayobject.h'),
720:             join('src', 'multiarray', 'arraytypes.h'),
721:             join('src', 'multiarray', 'array_assign.h'),
722:             join('src', 'multiarray', 'buffer.h'),
723:             join('src', 'multiarray', 'calculation.h'),
724:             join('src', 'multiarray', 'cblasfuncs.h'),
725:             join('src', 'multiarray', 'common.h'),
726:             join('src', 'multiarray', 'convert_datatype.h'),
727:             join('src', 'multiarray', 'convert.h'),
728:             join('src', 'multiarray', 'conversion_utils.h'),
729:             join('src', 'multiarray', 'ctors.h'),
730:             join('src', 'multiarray', 'descriptor.h'),
731:             join('src', 'multiarray', 'getset.h'),
732:             join('src', 'multiarray', 'hashdescr.h'),
733:             join('src', 'multiarray', 'iterators.h'),
734:             join('src', 'multiarray', 'mapping.h'),
735:             join('src', 'multiarray', 'methods.h'),
736:             join('src', 'multiarray', 'multiarraymodule.h'),
737:             join('src', 'multiarray', 'nditer_impl.h'),
738:             join('src', 'multiarray', 'numpymemoryview.h'),
739:             join('src', 'multiarray', 'number.h'),
740:             join('src', 'multiarray', 'numpyos.h'),
741:             join('src', 'multiarray', 'refcount.h'),
742:             join('src', 'multiarray', 'scalartypes.h'),
743:             join('src', 'multiarray', 'sequence.h'),
744:             join('src', 'multiarray', 'shape.h'),
745:             join('src', 'multiarray', 'ucsnarrow.h'),
746:             join('src', 'multiarray', 'usertypes.h'),
747:             join('src', 'multiarray', 'vdot.h'),
748:             join('src', 'private', 'npy_config.h'),
749:             join('src', 'private', 'templ_common.h.src'),
750:             join('src', 'private', 'lowlevel_strided_loops.h'),
751:             join('src', 'private', 'mem_overlap.h'),
752:             join('src', 'private', 'npy_extint128.h'),
753:             join('include', 'numpy', 'arrayobject.h'),
754:             join('include', 'numpy', '_neighborhood_iterator_imp.h'),
755:             join('include', 'numpy', 'npy_endian.h'),
756:             join('include', 'numpy', 'arrayscalars.h'),
757:             join('include', 'numpy', 'noprefix.h'),
758:             join('include', 'numpy', 'npy_interrupt.h'),
759:             join('include', 'numpy', 'npy_3kcompat.h'),
760:             join('include', 'numpy', 'npy_math.h'),
761:             join('include', 'numpy', 'halffloat.h'),
762:             join('include', 'numpy', 'npy_common.h'),
763:             join('include', 'numpy', 'npy_os.h'),
764:             join('include', 'numpy', 'utils.h'),
765:             join('include', 'numpy', 'ndarrayobject.h'),
766:             join('include', 'numpy', 'npy_cpu.h'),
767:             join('include', 'numpy', 'numpyconfig.h'),
768:             join('include', 'numpy', 'ndarraytypes.h'),
769:             join('include', 'numpy', 'npy_1_7_deprecated_api.h'),
770:             join('include', 'numpy', '_numpyconfig.h.in'),
771:             # add library sources as distuils does not consider libraries
772:             # dependencies
773:             ] + npysort_sources + npymath_sources
774: 
775:     multiarray_src = [
776:             join('src', 'multiarray', 'alloc.c'),
777:             join('src', 'multiarray', 'arrayobject.c'),
778:             join('src', 'multiarray', 'arraytypes.c.src'),
779:             join('src', 'multiarray', 'array_assign.c'),
780:             join('src', 'multiarray', 'array_assign_scalar.c'),
781:             join('src', 'multiarray', 'array_assign_array.c'),
782:             join('src', 'multiarray', 'buffer.c'),
783:             join('src', 'multiarray', 'calculation.c'),
784:             join('src', 'multiarray', 'compiled_base.c'),
785:             join('src', 'multiarray', 'common.c'),
786:             join('src', 'multiarray', 'convert.c'),
787:             join('src', 'multiarray', 'convert_datatype.c'),
788:             join('src', 'multiarray', 'conversion_utils.c'),
789:             join('src', 'multiarray', 'ctors.c'),
790:             join('src', 'multiarray', 'datetime.c'),
791:             join('src', 'multiarray', 'datetime_strings.c'),
792:             join('src', 'multiarray', 'datetime_busday.c'),
793:             join('src', 'multiarray', 'datetime_busdaycal.c'),
794:             join('src', 'multiarray', 'descriptor.c'),
795:             join('src', 'multiarray', 'dtype_transfer.c'),
796:             join('src', 'multiarray', 'einsum.c.src'),
797:             join('src', 'multiarray', 'flagsobject.c'),
798:             join('src', 'multiarray', 'getset.c'),
799:             join('src', 'multiarray', 'hashdescr.c'),
800:             join('src', 'multiarray', 'item_selection.c'),
801:             join('src', 'multiarray', 'iterators.c'),
802:             join('src', 'multiarray', 'lowlevel_strided_loops.c.src'),
803:             join('src', 'multiarray', 'mapping.c'),
804:             join('src', 'multiarray', 'methods.c'),
805:             join('src', 'multiarray', 'multiarraymodule.c'),
806:             join('src', 'multiarray', 'nditer_templ.c.src'),
807:             join('src', 'multiarray', 'nditer_api.c'),
808:             join('src', 'multiarray', 'nditer_constr.c'),
809:             join('src', 'multiarray', 'nditer_pywrap.c'),
810:             join('src', 'multiarray', 'number.c'),
811:             join('src', 'multiarray', 'numpymemoryview.c'),
812:             join('src', 'multiarray', 'numpyos.c'),
813:             join('src', 'multiarray', 'refcount.c'),
814:             join('src', 'multiarray', 'sequence.c'),
815:             join('src', 'multiarray', 'shape.c'),
816:             join('src', 'multiarray', 'scalarapi.c'),
817:             join('src', 'multiarray', 'scalartypes.c.src'),
818:             join('src', 'multiarray', 'usertypes.c'),
819:             join('src', 'multiarray', 'ucsnarrow.c'),
820:             join('src', 'multiarray', 'vdot.c'),
821:             join('src', 'private', 'templ_common.h.src'),
822:             join('src', 'private', 'mem_overlap.c'),
823:             ]
824: 
825:     blas_info = get_info('blas_opt', 0)
826:     if blas_info and ('HAVE_CBLAS', None) in blas_info.get('define_macros', []):
827:         extra_info = blas_info
828:         # These files are also in MANIFEST.in so that they are always in
829:         # the source distribution independently of HAVE_CBLAS.
830:         multiarray_src.extend([join('src', 'multiarray', 'cblasfuncs.c'),
831:                                join('src', 'multiarray', 'python_xerbla.c'),
832:                                ])
833:         if uses_accelerate_framework(blas_info):
834:             multiarray_src.extend(get_sgemv_fix())
835:     else:
836:         extra_info = {}
837: 
838:     config.add_extension('multiarray',
839:                          sources=multiarray_src +
840:                                  [generate_config_h,
841:                                   generate_numpyconfig_h,
842:                                   generate_numpy_api,
843:                                   join(codegen_dir, 'generate_numpy_api.py'),
844:                                   join('*.py')],
845:                          depends=deps + multiarray_deps,
846:                          libraries=['npymath', 'npysort'],
847:                          extra_info=extra_info)
848: 
849:     #######################################################################
850:     #                           umath module                              #
851:     #######################################################################
852: 
853:     # umath version: this function is needed to build foo.c from foo.c.src
854:     # when foo.c is included in another file and as such not in the src
855:     # argument of build_ext command
856:     def generate_umath_templated_sources(ext, build_dir):
857:         from numpy.distutils.misc_util import get_cmd
858: 
859:         subpath = join('src', 'umath')
860:         sources = [
861:             join(local_dir, subpath, 'loops.h.src'),
862:             join(local_dir, subpath, 'loops.c.src'),
863:             join(local_dir, subpath, 'scalarmath.c.src'),
864:             join(local_dir, subpath, 'simd.inc.src')]
865: 
866:         # numpy.distutils generate .c from .c.src in weird directories, we have
867:         # to add them there as they depend on the build_dir
868:         config.add_include_dirs(join(build_dir, subpath))
869:         cmd = get_cmd('build_src')
870:         cmd.ensure_finalized()
871:         cmd.template_sources(sources, ext)
872: 
873:     def generate_umath_c(ext, build_dir):
874:         target = join(build_dir, header_dir, '__umath_generated.c')
875:         dir = os.path.dirname(target)
876:         if not os.path.exists(dir):
877:             os.makedirs(dir)
878:         script = generate_umath_py
879:         if newer(script, target):
880:             f = open(target, 'w')
881:             f.write(generate_umath.make_code(generate_umath.defdict,
882:                                              generate_umath.__file__))
883:             f.close()
884:         return []
885: 
886:     umath_src = [
887:             join('src', 'umath', 'umathmodule.c'),
888:             join('src', 'umath', 'reduction.c'),
889:             join('src', 'umath', 'funcs.inc.src'),
890:             join('src', 'umath', 'simd.inc.src'),
891:             join('src', 'umath', 'loops.h.src'),
892:             join('src', 'umath', 'loops.c.src'),
893:             join('src', 'umath', 'ufunc_object.c'),
894:             join('src', 'umath', 'scalarmath.c.src'),
895:             join('src', 'umath', 'ufunc_type_resolution.c')]
896: 
897:     umath_deps = [
898:             generate_umath_py,
899:             join('include', 'numpy', 'npy_math.h'),
900:             join('include', 'numpy', 'halffloat.h'),
901:             join('src', 'multiarray', 'common.h'),
902:             join('src', 'private', 'templ_common.h.src'),
903:             join('src', 'umath', 'simd.inc.src'),
904:             join(codegen_dir, 'generate_ufunc_api.py'),
905:             join('src', 'private', 'ufunc_override.h')] + npymath_sources
906: 
907:     config.add_extension('umath',
908:                          sources=umath_src +
909:                                  [generate_config_h,
910:                                  generate_numpyconfig_h,
911:                                  generate_umath_c,
912:                                  generate_ufunc_api],
913:                          depends=deps + umath_deps,
914:                          libraries=['npymath'],
915:                          )
916: 
917:     #######################################################################
918:     #                        umath_tests module                           #
919:     #######################################################################
920: 
921:     config.add_extension('umath_tests',
922:                     sources=[join('src', 'umath', 'umath_tests.c.src')])
923: 
924:     #######################################################################
925:     #                   custom rational dtype module                      #
926:     #######################################################################
927: 
928:     config.add_extension('test_rational',
929:                     sources=[join('src', 'umath', 'test_rational.c.src')])
930: 
931:     #######################################################################
932:     #                        struct_ufunc_test module                     #
933:     #######################################################################
934: 
935:     config.add_extension('struct_ufunc_test',
936:                     sources=[join('src', 'umath', 'struct_ufunc_test.c.src')])
937: 
938:     #######################################################################
939:     #                     multiarray_tests module                         #
940:     #######################################################################
941: 
942:     config.add_extension('multiarray_tests',
943:                     sources=[join('src', 'multiarray', 'multiarray_tests.c.src'),
944:                              join('src', 'private', 'mem_overlap.c')],
945:                     depends=[join('src', 'private', 'mem_overlap.h'),
946:                              join('src', 'private', 'npy_extint128.h')])
947: 
948:     #######################################################################
949:     #                        operand_flag_tests module                    #
950:     #######################################################################
951: 
952:     config.add_extension('operand_flag_tests',
953:                     sources=[join('src', 'umath', 'operand_flag_tests.c.src')])
954: 
955:     config.add_data_dir('tests')
956:     config.add_data_dir('tests/data')
957: 
958:     config.make_svn_version_py()
959: 
960:     return config
961: 
962: if __name__ == '__main__':
963:     from numpy.distutils.core import setup
964:     setup(configuration=configuration)
965: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import imp' statement (line 3)
import imp

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'imp', imp, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import sys' statement (line 5)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import pickle' statement (line 6)
import pickle

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pickle', pickle, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import copy' statement (line 7)
import copy

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'copy', copy, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import warnings' statement (line 8)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from os.path import join' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_14144 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os.path')

if (type(import_14144) is not StypyTypeError):

    if (import_14144 != 'pyd_module'):
        __import__(import_14144)
        sys_modules_14145 = sys.modules[import_14144]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os.path', sys_modules_14145.module_type_store, module_type_store, ['join'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_14145, sys_modules_14145.module_type_store, module_type_store)
    else:
        from os.path import join

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os.path', None, module_type_store, ['join'], [join])

else:
    # Assigning a type to the variable 'os.path' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'os.path', import_14144)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy.distutils import log' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_14146 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.distutils')

if (type(import_14146) is not StypyTypeError):

    if (import_14146 != 'pyd_module'):
        __import__(import_14146)
        sys_modules_14147 = sys.modules[import_14146]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.distutils', sys_modules_14147.module_type_store, module_type_store, ['log'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_14147, sys_modules_14147.module_type_store, module_type_store)
    else:
        from numpy.distutils import log

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.distutils', None, module_type_store, ['log'], [log])

else:
    # Assigning a type to the variable 'numpy.distutils' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.distutils', import_14146)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.dep_util import newer' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_14148 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.dep_util')

if (type(import_14148) is not StypyTypeError):

    if (import_14148 != 'pyd_module'):
        __import__(import_14148)
        sys_modules_14149 = sys.modules[import_14148]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.dep_util', sys_modules_14149.module_type_store, module_type_store, ['newer'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_14149, sys_modules_14149.module_type_store, module_type_store)
    else:
        from distutils.dep_util import newer

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.dep_util', None, module_type_store, ['newer'], [newer])

else:
    # Assigning a type to the variable 'distutils.dep_util' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.dep_util', import_14148)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.sysconfig import get_config_var' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_14150 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.sysconfig')

if (type(import_14150) is not StypyTypeError):

    if (import_14150 != 'pyd_module'):
        __import__(import_14150)
        sys_modules_14151 = sys.modules[import_14150]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.sysconfig', sys_modules_14151.module_type_store, module_type_store, ['get_config_var'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_14151, sys_modules_14151.module_type_store, module_type_store)
    else:
        from distutils.sysconfig import get_config_var

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.sysconfig', None, module_type_store, ['get_config_var'], [get_config_var])

else:
    # Assigning a type to the variable 'distutils.sysconfig' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.sysconfig', import_14150)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy._build_utils.apple_accelerate import uses_accelerate_framework, get_sgemv_fix' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_14152 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy._build_utils.apple_accelerate')

if (type(import_14152) is not StypyTypeError):

    if (import_14152 != 'pyd_module'):
        __import__(import_14152)
        sys_modules_14153 = sys.modules[import_14152]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy._build_utils.apple_accelerate', sys_modules_14153.module_type_store, module_type_store, ['uses_accelerate_framework', 'get_sgemv_fix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_14153, sys_modules_14153.module_type_store, module_type_store)
    else:
        from numpy._build_utils.apple_accelerate import uses_accelerate_framework, get_sgemv_fix

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy._build_utils.apple_accelerate', None, module_type_store, ['uses_accelerate_framework', 'get_sgemv_fix'], [uses_accelerate_framework, get_sgemv_fix])

else:
    # Assigning a type to the variable 'numpy._build_utils.apple_accelerate' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy._build_utils.apple_accelerate', import_14152)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from setup_common import ' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_14154 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'setup_common')

if (type(import_14154) is not StypyTypeError):

    if (import_14154 != 'pyd_module'):
        __import__(import_14154)
        sys_modules_14155 = sys.modules[import_14154]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'setup_common', sys_modules_14155.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_14155, sys_modules_14155.module_type_store, module_type_store)
    else:
        from setup_common import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'setup_common', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'setup_common' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'setup_common', import_14154)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')


# Assigning a Compare to a Name (line 20):

# Assigning a Compare to a Name (line 20):


# Call to get(...): (line 20)
# Processing the call arguments (line 20)
str_14159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 47), 'str', 'NPY_RELAXED_STRIDES_CHECKING')
str_14160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 79), 'str', '0')
# Processing the call keyword arguments (line 20)
kwargs_14161 = {}
# Getting the type of 'os' (line 20)
os_14156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 32), 'os', False)
# Obtaining the member 'environ' of a type (line 20)
environ_14157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 32), os_14156, 'environ')
# Obtaining the member 'get' of a type (line 20)
get_14158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 32), environ_14157, 'get')
# Calling get(args, kwargs) (line 20)
get_call_result_14162 = invoke(stypy.reporting.localization.Localization(__file__, 20, 32), get_14158, *[str_14159, str_14160], **kwargs_14161)

str_14163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 87), 'str', '0')
# Applying the binary operator '!=' (line 20)
result_ne_14164 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 32), '!=', get_call_result_14162, str_14163)

# Assigning a type to the variable 'NPY_RELAXED_STRIDES_CHECKING' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'NPY_RELAXED_STRIDES_CHECKING', result_ne_14164)
# Declaration of the 'CallOnceOnly' class

class CallOnceOnly(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CallOnceOnly.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 34):
        
        # Assigning a Name to a Attribute (line 34):
        # Getting the type of 'None' (line 34)
        None_14165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 28), 'None')
        # Getting the type of 'self' (line 34)
        self_14166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member '_check_types' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_14166, '_check_types', None_14165)
        
        # Assigning a Name to a Attribute (line 35):
        
        # Assigning a Name to a Attribute (line 35):
        # Getting the type of 'None' (line 35)
        None_14167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 34), 'None')
        # Getting the type of 'self' (line 35)
        self_14168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self')
        # Setting the type of the member '_check_ieee_macros' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_14168, '_check_ieee_macros', None_14167)
        
        # Assigning a Name to a Attribute (line 36):
        
        # Assigning a Name to a Attribute (line 36):
        # Getting the type of 'None' (line 36)
        None_14169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 30), 'None')
        # Getting the type of 'self' (line 36)
        self_14170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self')
        # Setting the type of the member '_check_complex' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_14170, '_check_complex', None_14169)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def check_types(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_types'
        module_type_store = module_type_store.open_function_context('check_types', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CallOnceOnly.check_types.__dict__.__setitem__('stypy_localization', localization)
        CallOnceOnly.check_types.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CallOnceOnly.check_types.__dict__.__setitem__('stypy_type_store', module_type_store)
        CallOnceOnly.check_types.__dict__.__setitem__('stypy_function_name', 'CallOnceOnly.check_types')
        CallOnceOnly.check_types.__dict__.__setitem__('stypy_param_names_list', [])
        CallOnceOnly.check_types.__dict__.__setitem__('stypy_varargs_param_name', 'a')
        CallOnceOnly.check_types.__dict__.__setitem__('stypy_kwargs_param_name', 'kw')
        CallOnceOnly.check_types.__dict__.__setitem__('stypy_call_defaults', defaults)
        CallOnceOnly.check_types.__dict__.__setitem__('stypy_call_varargs', varargs)
        CallOnceOnly.check_types.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CallOnceOnly.check_types.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CallOnceOnly.check_types', [], 'a', 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_types', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_types(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 39)
        # Getting the type of 'self' (line 39)
        self_14171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'self')
        # Obtaining the member '_check_types' of a type (line 39)
        _check_types_14172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 11), self_14171, '_check_types')
        # Getting the type of 'None' (line 39)
        None_14173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 32), 'None')
        
        (may_be_14174, more_types_in_union_14175) = may_be_none(_check_types_14172, None_14173)

        if may_be_14174:

            if more_types_in_union_14175:
                # Runtime conditional SSA (line 39)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 40):
            
            # Assigning a Call to a Name (line 40):
            
            # Call to check_types(...): (line 40)
            # Getting the type of 'a' (line 40)
            a_14177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 31), 'a', False)
            # Processing the call keyword arguments (line 40)
            # Getting the type of 'kw' (line 40)
            kw_14178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 36), 'kw', False)
            kwargs_14179 = {'kw_14178': kw_14178}
            # Getting the type of 'check_types' (line 40)
            check_types_14176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 18), 'check_types', False)
            # Calling check_types(args, kwargs) (line 40)
            check_types_call_result_14180 = invoke(stypy.reporting.localization.Localization(__file__, 40, 18), check_types_14176, *[a_14177], **kwargs_14179)
            
            # Assigning a type to the variable 'out' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'out', check_types_call_result_14180)
            
            # Assigning a Call to a Attribute (line 41):
            
            # Assigning a Call to a Attribute (line 41):
            
            # Call to dumps(...): (line 41)
            # Processing the call arguments (line 41)
            # Getting the type of 'out' (line 41)
            out_14183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 45), 'out', False)
            # Processing the call keyword arguments (line 41)
            kwargs_14184 = {}
            # Getting the type of 'pickle' (line 41)
            pickle_14181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 32), 'pickle', False)
            # Obtaining the member 'dumps' of a type (line 41)
            dumps_14182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 32), pickle_14181, 'dumps')
            # Calling dumps(args, kwargs) (line 41)
            dumps_call_result_14185 = invoke(stypy.reporting.localization.Localization(__file__, 41, 32), dumps_14182, *[out_14183], **kwargs_14184)
            
            # Getting the type of 'self' (line 41)
            self_14186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'self')
            # Setting the type of the member '_check_types' of a type (line 41)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), self_14186, '_check_types', dumps_call_result_14185)

            if more_types_in_union_14175:
                # Runtime conditional SSA for else branch (line 39)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_14174) or more_types_in_union_14175):
            
            # Assigning a Call to a Name (line 43):
            
            # Assigning a Call to a Name (line 43):
            
            # Call to deepcopy(...): (line 43)
            # Processing the call arguments (line 43)
            
            # Call to loads(...): (line 43)
            # Processing the call arguments (line 43)
            # Getting the type of 'self' (line 43)
            self_14191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 45), 'self', False)
            # Obtaining the member '_check_types' of a type (line 43)
            _check_types_14192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 45), self_14191, '_check_types')
            # Processing the call keyword arguments (line 43)
            kwargs_14193 = {}
            # Getting the type of 'pickle' (line 43)
            pickle_14189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 32), 'pickle', False)
            # Obtaining the member 'loads' of a type (line 43)
            loads_14190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 32), pickle_14189, 'loads')
            # Calling loads(args, kwargs) (line 43)
            loads_call_result_14194 = invoke(stypy.reporting.localization.Localization(__file__, 43, 32), loads_14190, *[_check_types_14192], **kwargs_14193)
            
            # Processing the call keyword arguments (line 43)
            kwargs_14195 = {}
            # Getting the type of 'copy' (line 43)
            copy_14187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 18), 'copy', False)
            # Obtaining the member 'deepcopy' of a type (line 43)
            deepcopy_14188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 18), copy_14187, 'deepcopy')
            # Calling deepcopy(args, kwargs) (line 43)
            deepcopy_call_result_14196 = invoke(stypy.reporting.localization.Localization(__file__, 43, 18), deepcopy_14188, *[loads_call_result_14194], **kwargs_14195)
            
            # Assigning a type to the variable 'out' (line 43)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'out', deepcopy_call_result_14196)

            if (may_be_14174 and more_types_in_union_14175):
                # SSA join for if statement (line 39)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'out' (line 44)
        out_14197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'out')
        # Assigning a type to the variable 'stypy_return_type' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'stypy_return_type', out_14197)
        
        # ################# End of 'check_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_types' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_14198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14198)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_types'
        return stypy_return_type_14198


    @norecursion
    def check_ieee_macros(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_ieee_macros'
        module_type_store = module_type_store.open_function_context('check_ieee_macros', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CallOnceOnly.check_ieee_macros.__dict__.__setitem__('stypy_localization', localization)
        CallOnceOnly.check_ieee_macros.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CallOnceOnly.check_ieee_macros.__dict__.__setitem__('stypy_type_store', module_type_store)
        CallOnceOnly.check_ieee_macros.__dict__.__setitem__('stypy_function_name', 'CallOnceOnly.check_ieee_macros')
        CallOnceOnly.check_ieee_macros.__dict__.__setitem__('stypy_param_names_list', [])
        CallOnceOnly.check_ieee_macros.__dict__.__setitem__('stypy_varargs_param_name', 'a')
        CallOnceOnly.check_ieee_macros.__dict__.__setitem__('stypy_kwargs_param_name', 'kw')
        CallOnceOnly.check_ieee_macros.__dict__.__setitem__('stypy_call_defaults', defaults)
        CallOnceOnly.check_ieee_macros.__dict__.__setitem__('stypy_call_varargs', varargs)
        CallOnceOnly.check_ieee_macros.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CallOnceOnly.check_ieee_macros.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CallOnceOnly.check_ieee_macros', [], 'a', 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_ieee_macros', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_ieee_macros(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 47)
        # Getting the type of 'self' (line 47)
        self_14199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'self')
        # Obtaining the member '_check_ieee_macros' of a type (line 47)
        _check_ieee_macros_14200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 11), self_14199, '_check_ieee_macros')
        # Getting the type of 'None' (line 47)
        None_14201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 38), 'None')
        
        (may_be_14202, more_types_in_union_14203) = may_be_none(_check_ieee_macros_14200, None_14201)

        if may_be_14202:

            if more_types_in_union_14203:
                # Runtime conditional SSA (line 47)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 48):
            
            # Assigning a Call to a Name (line 48):
            
            # Call to check_ieee_macros(...): (line 48)
            # Getting the type of 'a' (line 48)
            a_14205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 37), 'a', False)
            # Processing the call keyword arguments (line 48)
            # Getting the type of 'kw' (line 48)
            kw_14206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 42), 'kw', False)
            kwargs_14207 = {'kw_14206': kw_14206}
            # Getting the type of 'check_ieee_macros' (line 48)
            check_ieee_macros_14204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 18), 'check_ieee_macros', False)
            # Calling check_ieee_macros(args, kwargs) (line 48)
            check_ieee_macros_call_result_14208 = invoke(stypy.reporting.localization.Localization(__file__, 48, 18), check_ieee_macros_14204, *[a_14205], **kwargs_14207)
            
            # Assigning a type to the variable 'out' (line 48)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'out', check_ieee_macros_call_result_14208)
            
            # Assigning a Call to a Attribute (line 49):
            
            # Assigning a Call to a Attribute (line 49):
            
            # Call to dumps(...): (line 49)
            # Processing the call arguments (line 49)
            # Getting the type of 'out' (line 49)
            out_14211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 51), 'out', False)
            # Processing the call keyword arguments (line 49)
            kwargs_14212 = {}
            # Getting the type of 'pickle' (line 49)
            pickle_14209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 38), 'pickle', False)
            # Obtaining the member 'dumps' of a type (line 49)
            dumps_14210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 38), pickle_14209, 'dumps')
            # Calling dumps(args, kwargs) (line 49)
            dumps_call_result_14213 = invoke(stypy.reporting.localization.Localization(__file__, 49, 38), dumps_14210, *[out_14211], **kwargs_14212)
            
            # Getting the type of 'self' (line 49)
            self_14214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'self')
            # Setting the type of the member '_check_ieee_macros' of a type (line 49)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), self_14214, '_check_ieee_macros', dumps_call_result_14213)

            if more_types_in_union_14203:
                # Runtime conditional SSA for else branch (line 47)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_14202) or more_types_in_union_14203):
            
            # Assigning a Call to a Name (line 51):
            
            # Assigning a Call to a Name (line 51):
            
            # Call to deepcopy(...): (line 51)
            # Processing the call arguments (line 51)
            
            # Call to loads(...): (line 51)
            # Processing the call arguments (line 51)
            # Getting the type of 'self' (line 51)
            self_14219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 45), 'self', False)
            # Obtaining the member '_check_ieee_macros' of a type (line 51)
            _check_ieee_macros_14220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 45), self_14219, '_check_ieee_macros')
            # Processing the call keyword arguments (line 51)
            kwargs_14221 = {}
            # Getting the type of 'pickle' (line 51)
            pickle_14217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 32), 'pickle', False)
            # Obtaining the member 'loads' of a type (line 51)
            loads_14218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 32), pickle_14217, 'loads')
            # Calling loads(args, kwargs) (line 51)
            loads_call_result_14222 = invoke(stypy.reporting.localization.Localization(__file__, 51, 32), loads_14218, *[_check_ieee_macros_14220], **kwargs_14221)
            
            # Processing the call keyword arguments (line 51)
            kwargs_14223 = {}
            # Getting the type of 'copy' (line 51)
            copy_14215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'copy', False)
            # Obtaining the member 'deepcopy' of a type (line 51)
            deepcopy_14216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 18), copy_14215, 'deepcopy')
            # Calling deepcopy(args, kwargs) (line 51)
            deepcopy_call_result_14224 = invoke(stypy.reporting.localization.Localization(__file__, 51, 18), deepcopy_14216, *[loads_call_result_14222], **kwargs_14223)
            
            # Assigning a type to the variable 'out' (line 51)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'out', deepcopy_call_result_14224)

            if (may_be_14202 and more_types_in_union_14203):
                # SSA join for if statement (line 47)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'out' (line 52)
        out_14225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'out')
        # Assigning a type to the variable 'stypy_return_type' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'stypy_return_type', out_14225)
        
        # ################# End of 'check_ieee_macros(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_ieee_macros' in the type store
        # Getting the type of 'stypy_return_type' (line 46)
        stypy_return_type_14226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14226)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_ieee_macros'
        return stypy_return_type_14226


    @norecursion
    def check_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_complex'
        module_type_store = module_type_store.open_function_context('check_complex', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CallOnceOnly.check_complex.__dict__.__setitem__('stypy_localization', localization)
        CallOnceOnly.check_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CallOnceOnly.check_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        CallOnceOnly.check_complex.__dict__.__setitem__('stypy_function_name', 'CallOnceOnly.check_complex')
        CallOnceOnly.check_complex.__dict__.__setitem__('stypy_param_names_list', [])
        CallOnceOnly.check_complex.__dict__.__setitem__('stypy_varargs_param_name', 'a')
        CallOnceOnly.check_complex.__dict__.__setitem__('stypy_kwargs_param_name', 'kw')
        CallOnceOnly.check_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        CallOnceOnly.check_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        CallOnceOnly.check_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CallOnceOnly.check_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CallOnceOnly.check_complex', [], 'a', 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_complex(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 55)
        # Getting the type of 'self' (line 55)
        self_14227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'self')
        # Obtaining the member '_check_complex' of a type (line 55)
        _check_complex_14228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 11), self_14227, '_check_complex')
        # Getting the type of 'None' (line 55)
        None_14229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 34), 'None')
        
        (may_be_14230, more_types_in_union_14231) = may_be_none(_check_complex_14228, None_14229)

        if may_be_14230:

            if more_types_in_union_14231:
                # Runtime conditional SSA (line 55)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 56):
            
            # Assigning a Call to a Name (line 56):
            
            # Call to check_complex(...): (line 56)
            # Getting the type of 'a' (line 56)
            a_14233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'a', False)
            # Processing the call keyword arguments (line 56)
            # Getting the type of 'kw' (line 56)
            kw_14234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 38), 'kw', False)
            kwargs_14235 = {'kw_14234': kw_14234}
            # Getting the type of 'check_complex' (line 56)
            check_complex_14232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'check_complex', False)
            # Calling check_complex(args, kwargs) (line 56)
            check_complex_call_result_14236 = invoke(stypy.reporting.localization.Localization(__file__, 56, 18), check_complex_14232, *[a_14233], **kwargs_14235)
            
            # Assigning a type to the variable 'out' (line 56)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'out', check_complex_call_result_14236)
            
            # Assigning a Call to a Attribute (line 57):
            
            # Assigning a Call to a Attribute (line 57):
            
            # Call to dumps(...): (line 57)
            # Processing the call arguments (line 57)
            # Getting the type of 'out' (line 57)
            out_14239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 47), 'out', False)
            # Processing the call keyword arguments (line 57)
            kwargs_14240 = {}
            # Getting the type of 'pickle' (line 57)
            pickle_14237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 34), 'pickle', False)
            # Obtaining the member 'dumps' of a type (line 57)
            dumps_14238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 34), pickle_14237, 'dumps')
            # Calling dumps(args, kwargs) (line 57)
            dumps_call_result_14241 = invoke(stypy.reporting.localization.Localization(__file__, 57, 34), dumps_14238, *[out_14239], **kwargs_14240)
            
            # Getting the type of 'self' (line 57)
            self_14242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'self')
            # Setting the type of the member '_check_complex' of a type (line 57)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), self_14242, '_check_complex', dumps_call_result_14241)

            if more_types_in_union_14231:
                # Runtime conditional SSA for else branch (line 55)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_14230) or more_types_in_union_14231):
            
            # Assigning a Call to a Name (line 59):
            
            # Assigning a Call to a Name (line 59):
            
            # Call to deepcopy(...): (line 59)
            # Processing the call arguments (line 59)
            
            # Call to loads(...): (line 59)
            # Processing the call arguments (line 59)
            # Getting the type of 'self' (line 59)
            self_14247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 45), 'self', False)
            # Obtaining the member '_check_complex' of a type (line 59)
            _check_complex_14248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 45), self_14247, '_check_complex')
            # Processing the call keyword arguments (line 59)
            kwargs_14249 = {}
            # Getting the type of 'pickle' (line 59)
            pickle_14245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 32), 'pickle', False)
            # Obtaining the member 'loads' of a type (line 59)
            loads_14246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 32), pickle_14245, 'loads')
            # Calling loads(args, kwargs) (line 59)
            loads_call_result_14250 = invoke(stypy.reporting.localization.Localization(__file__, 59, 32), loads_14246, *[_check_complex_14248], **kwargs_14249)
            
            # Processing the call keyword arguments (line 59)
            kwargs_14251 = {}
            # Getting the type of 'copy' (line 59)
            copy_14243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 18), 'copy', False)
            # Obtaining the member 'deepcopy' of a type (line 59)
            deepcopy_14244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 18), copy_14243, 'deepcopy')
            # Calling deepcopy(args, kwargs) (line 59)
            deepcopy_call_result_14252 = invoke(stypy.reporting.localization.Localization(__file__, 59, 18), deepcopy_14244, *[loads_call_result_14250], **kwargs_14251)
            
            # Assigning a type to the variable 'out' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'out', deepcopy_call_result_14252)

            if (may_be_14230 and more_types_in_union_14231):
                # SSA join for if statement (line 55)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'out' (line 60)
        out_14253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'out')
        # Assigning a type to the variable 'stypy_return_type' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'stypy_return_type', out_14253)
        
        # ################# End of 'check_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_14254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14254)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_complex'
        return stypy_return_type_14254


# Assigning a type to the variable 'CallOnceOnly' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'CallOnceOnly', CallOnceOnly)

# Assigning a Name to a Name (line 62):

# Assigning a Name to a Name (line 62):
# Getting the type of 'True' (line 62)
True_14255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 26), 'True')
# Assigning a type to the variable 'PYTHON_HAS_UNICODE_WIDE' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'PYTHON_HAS_UNICODE_WIDE', True_14255)

@norecursion
def pythonlib_dir(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pythonlib_dir'
    module_type_store = module_type_store.open_function_context('pythonlib_dir', 64, 0, False)
    
    # Passed parameters checking function
    pythonlib_dir.stypy_localization = localization
    pythonlib_dir.stypy_type_of_self = None
    pythonlib_dir.stypy_type_store = module_type_store
    pythonlib_dir.stypy_function_name = 'pythonlib_dir'
    pythonlib_dir.stypy_param_names_list = []
    pythonlib_dir.stypy_varargs_param_name = None
    pythonlib_dir.stypy_kwargs_param_name = None
    pythonlib_dir.stypy_call_defaults = defaults
    pythonlib_dir.stypy_call_varargs = varargs
    pythonlib_dir.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pythonlib_dir', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pythonlib_dir', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pythonlib_dir(...)' code ##################

    str_14256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'str', 'return path where libpython* is.')
    
    
    # Getting the type of 'sys' (line 66)
    sys_14257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 7), 'sys')
    # Obtaining the member 'platform' of a type (line 66)
    platform_14258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 7), sys_14257, 'platform')
    str_14259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 23), 'str', 'win32')
    # Applying the binary operator '==' (line 66)
    result_eq_14260 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 7), '==', platform_14258, str_14259)
    
    # Testing the type of an if condition (line 66)
    if_condition_14261 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 4), result_eq_14260)
    # Assigning a type to the variable 'if_condition_14261' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'if_condition_14261', if_condition_14261)
    # SSA begins for if statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to join(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'sys' (line 67)
    sys_14265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 28), 'sys', False)
    # Obtaining the member 'prefix' of a type (line 67)
    prefix_14266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 28), sys_14265, 'prefix')
    str_14267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 40), 'str', 'libs')
    # Processing the call keyword arguments (line 67)
    kwargs_14268 = {}
    # Getting the type of 'os' (line 67)
    os_14262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 67)
    path_14263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 15), os_14262, 'path')
    # Obtaining the member 'join' of a type (line 67)
    join_14264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 15), path_14263, 'join')
    # Calling join(args, kwargs) (line 67)
    join_call_result_14269 = invoke(stypy.reporting.localization.Localization(__file__, 67, 15), join_14264, *[prefix_14266, str_14267], **kwargs_14268)
    
    # Assigning a type to the variable 'stypy_return_type' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'stypy_return_type', join_call_result_14269)
    # SSA branch for the else part of an if statement (line 66)
    module_type_store.open_ssa_branch('else')
    
    # Call to get_config_var(...): (line 69)
    # Processing the call arguments (line 69)
    str_14271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 30), 'str', 'LIBDIR')
    # Processing the call keyword arguments (line 69)
    kwargs_14272 = {}
    # Getting the type of 'get_config_var' (line 69)
    get_config_var_14270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'get_config_var', False)
    # Calling get_config_var(args, kwargs) (line 69)
    get_config_var_call_result_14273 = invoke(stypy.reporting.localization.Localization(__file__, 69, 15), get_config_var_14270, *[str_14271], **kwargs_14272)
    
    # Assigning a type to the variable 'stypy_return_type' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'stypy_return_type', get_config_var_call_result_14273)
    # SSA join for if statement (line 66)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'pythonlib_dir(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pythonlib_dir' in the type store
    # Getting the type of 'stypy_return_type' (line 64)
    stypy_return_type_14274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14274)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pythonlib_dir'
    return stypy_return_type_14274

# Assigning a type to the variable 'pythonlib_dir' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'pythonlib_dir', pythonlib_dir)

@norecursion
def is_npy_no_signal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_npy_no_signal'
    module_type_store = module_type_store.open_function_context('is_npy_no_signal', 71, 0, False)
    
    # Passed parameters checking function
    is_npy_no_signal.stypy_localization = localization
    is_npy_no_signal.stypy_type_of_self = None
    is_npy_no_signal.stypy_type_store = module_type_store
    is_npy_no_signal.stypy_function_name = 'is_npy_no_signal'
    is_npy_no_signal.stypy_param_names_list = []
    is_npy_no_signal.stypy_varargs_param_name = None
    is_npy_no_signal.stypy_kwargs_param_name = None
    is_npy_no_signal.stypy_call_defaults = defaults
    is_npy_no_signal.stypy_call_varargs = varargs
    is_npy_no_signal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_npy_no_signal', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_npy_no_signal', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_npy_no_signal(...)' code ##################

    str_14275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, (-1)), 'str', 'Return True if the NPY_NO_SIGNAL symbol must be defined in configuration\n    header.')
    
    # Getting the type of 'sys' (line 74)
    sys_14276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 11), 'sys')
    # Obtaining the member 'platform' of a type (line 74)
    platform_14277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 11), sys_14276, 'platform')
    str_14278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 27), 'str', 'win32')
    # Applying the binary operator '==' (line 74)
    result_eq_14279 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 11), '==', platform_14277, str_14278)
    
    # Assigning a type to the variable 'stypy_return_type' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type', result_eq_14279)
    
    # ################# End of 'is_npy_no_signal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_npy_no_signal' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_14280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14280)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_npy_no_signal'
    return stypy_return_type_14280

# Assigning a type to the variable 'is_npy_no_signal' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'is_npy_no_signal', is_npy_no_signal)

@norecursion
def is_npy_no_smp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_npy_no_smp'
    module_type_store = module_type_store.open_function_context('is_npy_no_smp', 76, 0, False)
    
    # Passed parameters checking function
    is_npy_no_smp.stypy_localization = localization
    is_npy_no_smp.stypy_type_of_self = None
    is_npy_no_smp.stypy_type_store = module_type_store
    is_npy_no_smp.stypy_function_name = 'is_npy_no_smp'
    is_npy_no_smp.stypy_param_names_list = []
    is_npy_no_smp.stypy_varargs_param_name = None
    is_npy_no_smp.stypy_kwargs_param_name = None
    is_npy_no_smp.stypy_call_defaults = defaults
    is_npy_no_smp.stypy_call_varargs = varargs
    is_npy_no_smp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_npy_no_smp', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_npy_no_smp', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_npy_no_smp(...)' code ##################

    str_14281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, (-1)), 'str', 'Return True if the NPY_NO_SMP symbol must be defined in public\n    header (when SMP support cannot be reliably enabled).')
    
    str_14282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 11), 'str', 'NPY_NOSMP')
    # Getting the type of 'os' (line 85)
    os_14283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 26), 'os')
    # Obtaining the member 'environ' of a type (line 85)
    environ_14284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 26), os_14283, 'environ')
    # Applying the binary operator 'in' (line 85)
    result_contains_14285 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 11), 'in', str_14282, environ_14284)
    
    # Assigning a type to the variable 'stypy_return_type' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type', result_contains_14285)
    
    # ################# End of 'is_npy_no_smp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_npy_no_smp' in the type store
    # Getting the type of 'stypy_return_type' (line 76)
    stypy_return_type_14286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14286)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_npy_no_smp'
    return stypy_return_type_14286

# Assigning a type to the variable 'is_npy_no_smp' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'is_npy_no_smp', is_npy_no_smp)

@norecursion
def win32_checks(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'win32_checks'
    module_type_store = module_type_store.open_function_context('win32_checks', 87, 0, False)
    
    # Passed parameters checking function
    win32_checks.stypy_localization = localization
    win32_checks.stypy_type_of_self = None
    win32_checks.stypy_type_store = module_type_store
    win32_checks.stypy_function_name = 'win32_checks'
    win32_checks.stypy_param_names_list = ['deflist']
    win32_checks.stypy_varargs_param_name = None
    win32_checks.stypy_kwargs_param_name = None
    win32_checks.stypy_call_defaults = defaults
    win32_checks.stypy_call_varargs = varargs
    win32_checks.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'win32_checks', ['deflist'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'win32_checks', localization, ['deflist'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'win32_checks(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 88, 4))
    
    # 'from numpy.distutils.misc_util import get_build_architecture' statement (line 88)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
    import_14287 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 88, 4), 'numpy.distutils.misc_util')

    if (type(import_14287) is not StypyTypeError):

        if (import_14287 != 'pyd_module'):
            __import__(import_14287)
            sys_modules_14288 = sys.modules[import_14287]
            import_from_module(stypy.reporting.localization.Localization(__file__, 88, 4), 'numpy.distutils.misc_util', sys_modules_14288.module_type_store, module_type_store, ['get_build_architecture'])
            nest_module(stypy.reporting.localization.Localization(__file__, 88, 4), __file__, sys_modules_14288, sys_modules_14288.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import get_build_architecture

            import_from_module(stypy.reporting.localization.Localization(__file__, 88, 4), 'numpy.distutils.misc_util', None, module_type_store, ['get_build_architecture'], [get_build_architecture])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'numpy.distutils.misc_util', import_14287)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')
    
    
    # Assigning a Call to a Name (line 89):
    
    # Assigning a Call to a Name (line 89):
    
    # Call to get_build_architecture(...): (line 89)
    # Processing the call keyword arguments (line 89)
    kwargs_14290 = {}
    # Getting the type of 'get_build_architecture' (line 89)
    get_build_architecture_14289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'get_build_architecture', False)
    # Calling get_build_architecture(args, kwargs) (line 89)
    get_build_architecture_call_result_14291 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), get_build_architecture_14289, *[], **kwargs_14290)
    
    # Assigning a type to the variable 'a' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'a', get_build_architecture_call_result_14291)
    
    # Call to print(...): (line 92)
    # Processing the call arguments (line 92)
    str_14293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 10), 'str', 'BUILD_ARCHITECTURE: %r, os.name=%r, sys.platform=%r')
    
    # Obtaining an instance of the builtin type 'tuple' (line 93)
    tuple_14294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 93)
    # Adding element type (line 93)
    # Getting the type of 'a' (line 93)
    a_14295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 11), tuple_14294, a_14295)
    # Adding element type (line 93)
    # Getting the type of 'os' (line 93)
    os_14296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 14), 'os', False)
    # Obtaining the member 'name' of a type (line 93)
    name_14297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 14), os_14296, 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 11), tuple_14294, name_14297)
    # Adding element type (line 93)
    # Getting the type of 'sys' (line 93)
    sys_14298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 23), 'sys', False)
    # Obtaining the member 'platform' of a type (line 93)
    platform_14299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 23), sys_14298, 'platform')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 11), tuple_14294, platform_14299)
    
    # Applying the binary operator '%' (line 92)
    result_mod_14300 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 10), '%', str_14293, tuple_14294)
    
    # Processing the call keyword arguments (line 92)
    kwargs_14301 = {}
    # Getting the type of 'print' (line 92)
    print_14292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'print', False)
    # Calling print(args, kwargs) (line 92)
    print_call_result_14302 = invoke(stypy.reporting.localization.Localization(__file__, 92, 4), print_14292, *[result_mod_14300], **kwargs_14301)
    
    
    
    # Getting the type of 'a' (line 94)
    a_14303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 7), 'a')
    str_14304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 12), 'str', 'AMD64')
    # Applying the binary operator '==' (line 94)
    result_eq_14305 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 7), '==', a_14303, str_14304)
    
    # Testing the type of an if condition (line 94)
    if_condition_14306 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 4), result_eq_14305)
    # Assigning a type to the variable 'if_condition_14306' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'if_condition_14306', if_condition_14306)
    # SSA begins for if statement (line 94)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 95)
    # Processing the call arguments (line 95)
    str_14309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 23), 'str', 'DISTUTILS_USE_SDK')
    # Processing the call keyword arguments (line 95)
    kwargs_14310 = {}
    # Getting the type of 'deflist' (line 95)
    deflist_14307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'deflist', False)
    # Obtaining the member 'append' of a type (line 95)
    append_14308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), deflist_14307, 'append')
    # Calling append(args, kwargs) (line 95)
    append_call_result_14311 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), append_14308, *[str_14309], **kwargs_14310)
    
    # SSA join for if statement (line 94)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'a' (line 100)
    a_14312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 7), 'a')
    str_14313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 12), 'str', 'Intel')
    # Applying the binary operator '==' (line 100)
    result_eq_14314 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 7), '==', a_14312, str_14313)
    
    
    # Getting the type of 'a' (line 100)
    a_14315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 23), 'a')
    str_14316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 28), 'str', 'AMD64')
    # Applying the binary operator '==' (line 100)
    result_eq_14317 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 23), '==', a_14315, str_14316)
    
    # Applying the binary operator 'or' (line 100)
    result_or_keyword_14318 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 7), 'or', result_eq_14314, result_eq_14317)
    
    # Testing the type of an if condition (line 100)
    if_condition_14319 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 4), result_or_keyword_14318)
    # Assigning a type to the variable 'if_condition_14319' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'if_condition_14319', if_condition_14319)
    # SSA begins for if statement (line 100)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 101)
    # Processing the call arguments (line 101)
    str_14322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 23), 'str', 'FORCE_NO_LONG_DOUBLE_FORMATTING')
    # Processing the call keyword arguments (line 101)
    kwargs_14323 = {}
    # Getting the type of 'deflist' (line 101)
    deflist_14320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'deflist', False)
    # Obtaining the member 'append' of a type (line 101)
    append_14321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), deflist_14320, 'append')
    # Calling append(args, kwargs) (line 101)
    append_call_result_14324 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), append_14321, *[str_14322], **kwargs_14323)
    
    # SSA join for if statement (line 100)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'win32_checks(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'win32_checks' in the type store
    # Getting the type of 'stypy_return_type' (line 87)
    stypy_return_type_14325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14325)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'win32_checks'
    return stypy_return_type_14325

# Assigning a type to the variable 'win32_checks' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'win32_checks', win32_checks)

@norecursion
def check_math_capabilities(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_math_capabilities'
    module_type_store = module_type_store.open_function_context('check_math_capabilities', 103, 0, False)
    
    # Passed parameters checking function
    check_math_capabilities.stypy_localization = localization
    check_math_capabilities.stypy_type_of_self = None
    check_math_capabilities.stypy_type_store = module_type_store
    check_math_capabilities.stypy_function_name = 'check_math_capabilities'
    check_math_capabilities.stypy_param_names_list = ['config', 'moredefs', 'mathlibs']
    check_math_capabilities.stypy_varargs_param_name = None
    check_math_capabilities.stypy_kwargs_param_name = None
    check_math_capabilities.stypy_call_defaults = defaults
    check_math_capabilities.stypy_call_varargs = varargs
    check_math_capabilities.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_math_capabilities', ['config', 'moredefs', 'mathlibs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_math_capabilities', localization, ['config', 'moredefs', 'mathlibs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_math_capabilities(...)' code ##################


    @norecursion
    def check_func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_func'
        module_type_store = module_type_store.open_function_context('check_func', 104, 4, False)
        
        # Passed parameters checking function
        check_func.stypy_localization = localization
        check_func.stypy_type_of_self = None
        check_func.stypy_type_store = module_type_store
        check_func.stypy_function_name = 'check_func'
        check_func.stypy_param_names_list = ['func_name']
        check_func.stypy_varargs_param_name = None
        check_func.stypy_kwargs_param_name = None
        check_func.stypy_call_defaults = defaults
        check_func.stypy_call_varargs = varargs
        check_func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check_func', ['func_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_func', localization, ['func_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_func(...)' code ##################

        
        # Call to check_func(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'func_name' (line 105)
        func_name_14328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 33), 'func_name', False)
        # Processing the call keyword arguments (line 105)
        # Getting the type of 'mathlibs' (line 105)
        mathlibs_14329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 54), 'mathlibs', False)
        keyword_14330 = mathlibs_14329
        # Getting the type of 'True' (line 106)
        True_14331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 38), 'True', False)
        keyword_14332 = True_14331
        # Getting the type of 'True' (line 106)
        True_14333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 49), 'True', False)
        keyword_14334 = True_14333
        kwargs_14335 = {'libraries': keyword_14330, 'decl': keyword_14332, 'call': keyword_14334}
        # Getting the type of 'config' (line 105)
        config_14326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'config', False)
        # Obtaining the member 'check_func' of a type (line 105)
        check_func_14327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 15), config_14326, 'check_func')
        # Calling check_func(args, kwargs) (line 105)
        check_func_call_result_14336 = invoke(stypy.reporting.localization.Localization(__file__, 105, 15), check_func_14327, *[func_name_14328], **kwargs_14335)
        
        # Assigning a type to the variable 'stypy_return_type' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'stypy_return_type', check_func_call_result_14336)
        
        # ################# End of 'check_func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_func' in the type store
        # Getting the type of 'stypy_return_type' (line 104)
        stypy_return_type_14337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14337)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_func'
        return stypy_return_type_14337

    # Assigning a type to the variable 'check_func' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'check_func', check_func)

    @norecursion
    def check_funcs_once(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_funcs_once'
        module_type_store = module_type_store.open_function_context('check_funcs_once', 108, 4, False)
        
        # Passed parameters checking function
        check_funcs_once.stypy_localization = localization
        check_funcs_once.stypy_type_of_self = None
        check_funcs_once.stypy_type_store = module_type_store
        check_funcs_once.stypy_function_name = 'check_funcs_once'
        check_funcs_once.stypy_param_names_list = ['funcs_name']
        check_funcs_once.stypy_varargs_param_name = None
        check_funcs_once.stypy_kwargs_param_name = None
        check_funcs_once.stypy_call_defaults = defaults
        check_funcs_once.stypy_call_varargs = varargs
        check_funcs_once.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check_funcs_once', ['funcs_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_funcs_once', localization, ['funcs_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_funcs_once(...)' code ##################

        
        # Assigning a Call to a Name (line 109):
        
        # Assigning a Call to a Name (line 109):
        
        # Call to dict(...): (line 109)
        # Processing the call arguments (line 109)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'funcs_name' (line 109)
        funcs_name_14342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 40), 'funcs_name', False)
        comprehension_14343 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 21), funcs_name_14342)
        # Assigning a type to the variable 'f' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'f', comprehension_14343)
        
        # Obtaining an instance of the builtin type 'tuple' (line 109)
        tuple_14339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 109)
        # Adding element type (line 109)
        # Getting the type of 'f' (line 109)
        f_14340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 22), 'f', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 22), tuple_14339, f_14340)
        # Adding element type (line 109)
        # Getting the type of 'True' (line 109)
        True_14341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 25), 'True', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 22), tuple_14339, True_14341)
        
        list_14344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 21), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 21), list_14344, tuple_14339)
        # Processing the call keyword arguments (line 109)
        kwargs_14345 = {}
        # Getting the type of 'dict' (line 109)
        dict_14338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 15), 'dict', False)
        # Calling dict(args, kwargs) (line 109)
        dict_call_result_14346 = invoke(stypy.reporting.localization.Localization(__file__, 109, 15), dict_14338, *[list_14344], **kwargs_14345)
        
        # Assigning a type to the variable 'decl' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'decl', dict_call_result_14346)
        
        # Assigning a Call to a Name (line 110):
        
        # Assigning a Call to a Name (line 110):
        
        # Call to check_funcs_once(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'funcs_name' (line 110)
        funcs_name_14349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 37), 'funcs_name', False)
        # Processing the call keyword arguments (line 110)
        # Getting the type of 'mathlibs' (line 110)
        mathlibs_14350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 59), 'mathlibs', False)
        keyword_14351 = mathlibs_14350
        # Getting the type of 'decl' (line 111)
        decl_14352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 42), 'decl', False)
        keyword_14353 = decl_14352
        # Getting the type of 'decl' (line 111)
        decl_14354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 53), 'decl', False)
        keyword_14355 = decl_14354
        kwargs_14356 = {'libraries': keyword_14351, 'decl': keyword_14353, 'call': keyword_14355}
        # Getting the type of 'config' (line 110)
        config_14347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 13), 'config', False)
        # Obtaining the member 'check_funcs_once' of a type (line 110)
        check_funcs_once_14348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 13), config_14347, 'check_funcs_once')
        # Calling check_funcs_once(args, kwargs) (line 110)
        check_funcs_once_call_result_14357 = invoke(stypy.reporting.localization.Localization(__file__, 110, 13), check_funcs_once_14348, *[funcs_name_14349], **kwargs_14356)
        
        # Assigning a type to the variable 'st' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'st', check_funcs_once_call_result_14357)
        
        # Getting the type of 'st' (line 112)
        st_14358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'st')
        # Testing the type of an if condition (line 112)
        if_condition_14359 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 8), st_14358)
        # Assigning a type to the variable 'if_condition_14359' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'if_condition_14359', if_condition_14359)
        # SSA begins for if statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 113)
        # Processing the call arguments (line 113)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'funcs_name' (line 113)
        funcs_name_14368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 56), 'funcs_name', False)
        comprehension_14369 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 29), funcs_name_14368)
        # Assigning a type to the variable 'f' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 29), 'f', comprehension_14369)
        
        # Obtaining an instance of the builtin type 'tuple' (line 113)
        tuple_14362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 113)
        # Adding element type (line 113)
        
        # Call to fname2def(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'f' (line 113)
        f_14364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 40), 'f', False)
        # Processing the call keyword arguments (line 113)
        kwargs_14365 = {}
        # Getting the type of 'fname2def' (line 113)
        fname2def_14363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 30), 'fname2def', False)
        # Calling fname2def(args, kwargs) (line 113)
        fname2def_call_result_14366 = invoke(stypy.reporting.localization.Localization(__file__, 113, 30), fname2def_14363, *[f_14364], **kwargs_14365)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 30), tuple_14362, fname2def_call_result_14366)
        # Adding element type (line 113)
        int_14367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 30), tuple_14362, int_14367)
        
        list_14370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 29), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 29), list_14370, tuple_14362)
        # Processing the call keyword arguments (line 113)
        kwargs_14371 = {}
        # Getting the type of 'moredefs' (line 113)
        moredefs_14360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'moredefs', False)
        # Obtaining the member 'extend' of a type (line 113)
        extend_14361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 12), moredefs_14360, 'extend')
        # Calling extend(args, kwargs) (line 113)
        extend_call_result_14372 = invoke(stypy.reporting.localization.Localization(__file__, 113, 12), extend_14361, *[list_14370], **kwargs_14371)
        
        # SSA join for if statement (line 112)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'st' (line 114)
        st_14373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 15), 'st')
        # Assigning a type to the variable 'stypy_return_type' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'stypy_return_type', st_14373)
        
        # ################# End of 'check_funcs_once(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_funcs_once' in the type store
        # Getting the type of 'stypy_return_type' (line 108)
        stypy_return_type_14374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14374)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_funcs_once'
        return stypy_return_type_14374

    # Assigning a type to the variable 'check_funcs_once' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'check_funcs_once', check_funcs_once)

    @norecursion
    def check_funcs(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_funcs'
        module_type_store = module_type_store.open_function_context('check_funcs', 116, 4, False)
        
        # Passed parameters checking function
        check_funcs.stypy_localization = localization
        check_funcs.stypy_type_of_self = None
        check_funcs.stypy_type_store = module_type_store
        check_funcs.stypy_function_name = 'check_funcs'
        check_funcs.stypy_param_names_list = ['funcs_name']
        check_funcs.stypy_varargs_param_name = None
        check_funcs.stypy_kwargs_param_name = None
        check_funcs.stypy_call_defaults = defaults
        check_funcs.stypy_call_varargs = varargs
        check_funcs.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check_funcs', ['funcs_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_funcs', localization, ['funcs_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_funcs(...)' code ##################

        
        
        
        # Call to check_funcs_once(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'funcs_name' (line 119)
        funcs_name_14376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 32), 'funcs_name', False)
        # Processing the call keyword arguments (line 119)
        kwargs_14377 = {}
        # Getting the type of 'check_funcs_once' (line 119)
        check_funcs_once_14375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'check_funcs_once', False)
        # Calling check_funcs_once(args, kwargs) (line 119)
        check_funcs_once_call_result_14378 = invoke(stypy.reporting.localization.Localization(__file__, 119, 15), check_funcs_once_14375, *[funcs_name_14376], **kwargs_14377)
        
        # Applying the 'not' unary operator (line 119)
        result_not__14379 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 11), 'not', check_funcs_once_call_result_14378)
        
        # Testing the type of an if condition (line 119)
        if_condition_14380 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 8), result_not__14379)
        # Assigning a type to the variable 'if_condition_14380' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'if_condition_14380', if_condition_14380)
        # SSA begins for if statement (line 119)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'funcs_name' (line 121)
        funcs_name_14381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 21), 'funcs_name')
        # Testing the type of a for loop iterable (line 121)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 121, 12), funcs_name_14381)
        # Getting the type of the for loop variable (line 121)
        for_loop_var_14382 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 121, 12), funcs_name_14381)
        # Assigning a type to the variable 'f' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'f', for_loop_var_14382)
        # SSA begins for a for statement (line 121)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to check_func(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'f' (line 122)
        f_14384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 30), 'f', False)
        # Processing the call keyword arguments (line 122)
        kwargs_14385 = {}
        # Getting the type of 'check_func' (line 122)
        check_func_14383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 19), 'check_func', False)
        # Calling check_func(args, kwargs) (line 122)
        check_func_call_result_14386 = invoke(stypy.reporting.localization.Localization(__file__, 122, 19), check_func_14383, *[f_14384], **kwargs_14385)
        
        # Testing the type of an if condition (line 122)
        if_condition_14387 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 16), check_func_call_result_14386)
        # Assigning a type to the variable 'if_condition_14387' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'if_condition_14387', if_condition_14387)
        # SSA begins for if statement (line 122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Obtaining an instance of the builtin type 'tuple' (line 123)
        tuple_14390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 123)
        # Adding element type (line 123)
        
        # Call to fname2def(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'f' (line 123)
        f_14392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 47), 'f', False)
        # Processing the call keyword arguments (line 123)
        kwargs_14393 = {}
        # Getting the type of 'fname2def' (line 123)
        fname2def_14391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 37), 'fname2def', False)
        # Calling fname2def(args, kwargs) (line 123)
        fname2def_call_result_14394 = invoke(stypy.reporting.localization.Localization(__file__, 123, 37), fname2def_14391, *[f_14392], **kwargs_14393)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 37), tuple_14390, fname2def_call_result_14394)
        # Adding element type (line 123)
        int_14395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 37), tuple_14390, int_14395)
        
        # Processing the call keyword arguments (line 123)
        kwargs_14396 = {}
        # Getting the type of 'moredefs' (line 123)
        moredefs_14388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 20), 'moredefs', False)
        # Obtaining the member 'append' of a type (line 123)
        append_14389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 20), moredefs_14388, 'append')
        # Calling append(args, kwargs) (line 123)
        append_call_result_14397 = invoke(stypy.reporting.localization.Localization(__file__, 123, 20), append_14389, *[tuple_14390], **kwargs_14396)
        
        # SSA join for if statement (line 122)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        int_14398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 19), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'stypy_return_type', int_14398)
        # SSA branch for the else part of an if statement (line 119)
        module_type_store.open_ssa_branch('else')
        int_14399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 19), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'stypy_return_type', int_14399)
        # SSA join for if statement (line 119)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check_funcs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_funcs' in the type store
        # Getting the type of 'stypy_return_type' (line 116)
        stypy_return_type_14400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14400)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_funcs'
        return stypy_return_type_14400

    # Assigning a type to the variable 'check_funcs' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'check_funcs', check_funcs)
    
    
    
    # Call to check_funcs_once(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'MANDATORY_FUNCS' (line 130)
    MANDATORY_FUNCS_14402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 28), 'MANDATORY_FUNCS', False)
    # Processing the call keyword arguments (line 130)
    kwargs_14403 = {}
    # Getting the type of 'check_funcs_once' (line 130)
    check_funcs_once_14401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), 'check_funcs_once', False)
    # Calling check_funcs_once(args, kwargs) (line 130)
    check_funcs_once_call_result_14404 = invoke(stypy.reporting.localization.Localization(__file__, 130, 11), check_funcs_once_14401, *[MANDATORY_FUNCS_14402], **kwargs_14403)
    
    # Applying the 'not' unary operator (line 130)
    result_not__14405 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 7), 'not', check_funcs_once_call_result_14404)
    
    # Testing the type of an if condition (line 130)
    if_condition_14406 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 4), result_not__14405)
    # Assigning a type to the variable 'if_condition_14406' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'if_condition_14406', if_condition_14406)
    # SSA begins for if statement (line 130)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to SystemError(...): (line 131)
    # Processing the call arguments (line 131)
    str_14408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 26), 'str', 'One of the required function to build numpy is not available (the list is %s).')
    
    # Call to str(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'MANDATORY_FUNCS' (line 132)
    MANDATORY_FUNCS_14410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 53), 'MANDATORY_FUNCS', False)
    # Processing the call keyword arguments (line 132)
    kwargs_14411 = {}
    # Getting the type of 'str' (line 132)
    str_14409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 49), 'str', False)
    # Calling str(args, kwargs) (line 132)
    str_call_result_14412 = invoke(stypy.reporting.localization.Localization(__file__, 132, 49), str_14409, *[MANDATORY_FUNCS_14410], **kwargs_14411)
    
    # Applying the binary operator '%' (line 131)
    result_mod_14413 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 26), '%', str_14408, str_call_result_14412)
    
    # Processing the call keyword arguments (line 131)
    kwargs_14414 = {}
    # Getting the type of 'SystemError' (line 131)
    SystemError_14407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 14), 'SystemError', False)
    # Calling SystemError(args, kwargs) (line 131)
    SystemError_call_result_14415 = invoke(stypy.reporting.localization.Localization(__file__, 131, 14), SystemError_14407, *[result_mod_14413], **kwargs_14414)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 131, 8), SystemError_call_result_14415, 'raise parameter', BaseException)
    # SSA join for if statement (line 130)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'OPTIONAL_STDFUNCS_MAYBE' (line 141)
    OPTIONAL_STDFUNCS_MAYBE_14416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 13), 'OPTIONAL_STDFUNCS_MAYBE')
    # Testing the type of a for loop iterable (line 141)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 141, 4), OPTIONAL_STDFUNCS_MAYBE_14416)
    # Getting the type of the for loop variable (line 141)
    for_loop_var_14417 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 141, 4), OPTIONAL_STDFUNCS_MAYBE_14416)
    # Assigning a type to the variable 'f' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'f', for_loop_var_14417)
    # SSA begins for a for statement (line 141)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to check_decl(...): (line 142)
    # Processing the call arguments (line 142)
    
    # Call to fname2def(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'f' (line 142)
    f_14421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 39), 'f', False)
    # Processing the call keyword arguments (line 142)
    kwargs_14422 = {}
    # Getting the type of 'fname2def' (line 142)
    fname2def_14420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 29), 'fname2def', False)
    # Calling fname2def(args, kwargs) (line 142)
    fname2def_call_result_14423 = invoke(stypy.reporting.localization.Localization(__file__, 142, 29), fname2def_14420, *[f_14421], **kwargs_14422)
    
    # Processing the call keyword arguments (line 142)
    
    # Obtaining an instance of the builtin type 'list' (line 143)
    list_14424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 143)
    # Adding element type (line 143)
    str_14425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 29), 'str', 'Python.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 28), list_14424, str_14425)
    # Adding element type (line 143)
    str_14426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 41), 'str', 'math.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 28), list_14424, str_14426)
    
    keyword_14427 = list_14424
    kwargs_14428 = {'headers': keyword_14427}
    # Getting the type of 'config' (line 142)
    config_14418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 11), 'config', False)
    # Obtaining the member 'check_decl' of a type (line 142)
    check_decl_14419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 11), config_14418, 'check_decl')
    # Calling check_decl(args, kwargs) (line 142)
    check_decl_call_result_14429 = invoke(stypy.reporting.localization.Localization(__file__, 142, 11), check_decl_14419, *[fname2def_call_result_14423], **kwargs_14428)
    
    # Testing the type of an if condition (line 142)
    if_condition_14430 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 8), check_decl_call_result_14429)
    # Assigning a type to the variable 'if_condition_14430' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'if_condition_14430', if_condition_14430)
    # SSA begins for if statement (line 142)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to remove(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'f' (line 144)
    f_14433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 37), 'f', False)
    # Processing the call keyword arguments (line 144)
    kwargs_14434 = {}
    # Getting the type of 'OPTIONAL_STDFUNCS' (line 144)
    OPTIONAL_STDFUNCS_14431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'OPTIONAL_STDFUNCS', False)
    # Obtaining the member 'remove' of a type (line 144)
    remove_14432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 12), OPTIONAL_STDFUNCS_14431, 'remove')
    # Calling remove(args, kwargs) (line 144)
    remove_call_result_14435 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), remove_14432, *[f_14433], **kwargs_14434)
    
    # SSA join for if statement (line 142)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to check_funcs(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'OPTIONAL_STDFUNCS' (line 146)
    OPTIONAL_STDFUNCS_14437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'OPTIONAL_STDFUNCS', False)
    # Processing the call keyword arguments (line 146)
    kwargs_14438 = {}
    # Getting the type of 'check_funcs' (line 146)
    check_funcs_14436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'check_funcs', False)
    # Calling check_funcs(args, kwargs) (line 146)
    check_funcs_call_result_14439 = invoke(stypy.reporting.localization.Localization(__file__, 146, 4), check_funcs_14436, *[OPTIONAL_STDFUNCS_14437], **kwargs_14438)
    
    
    # Getting the type of 'OPTIONAL_HEADERS' (line 148)
    OPTIONAL_HEADERS_14440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 13), 'OPTIONAL_HEADERS')
    # Testing the type of a for loop iterable (line 148)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 148, 4), OPTIONAL_HEADERS_14440)
    # Getting the type of the for loop variable (line 148)
    for_loop_var_14441 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 148, 4), OPTIONAL_HEADERS_14440)
    # Assigning a type to the variable 'h' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'h', for_loop_var_14441)
    # SSA begins for a for statement (line 148)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to check_func(...): (line 149)
    # Processing the call arguments (line 149)
    str_14444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 29), 'str', '')
    # Processing the call keyword arguments (line 149)
    # Getting the type of 'False' (line 149)
    False_14445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 38), 'False', False)
    keyword_14446 = False_14445
    # Getting the type of 'False' (line 149)
    False_14447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 50), 'False', False)
    keyword_14448 = False_14447
    
    # Obtaining an instance of the builtin type 'list' (line 149)
    list_14449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 65), 'list')
    # Adding type elements to the builtin type 'list' instance (line 149)
    # Adding element type (line 149)
    # Getting the type of 'h' (line 149)
    h_14450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 66), 'h', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 65), list_14449, h_14450)
    
    keyword_14451 = list_14449
    kwargs_14452 = {'decl': keyword_14446, 'headers': keyword_14451, 'call': keyword_14448}
    # Getting the type of 'config' (line 149)
    config_14442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'config', False)
    # Obtaining the member 'check_func' of a type (line 149)
    check_func_14443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 11), config_14442, 'check_func')
    # Calling check_func(args, kwargs) (line 149)
    check_func_call_result_14453 = invoke(stypy.reporting.localization.Localization(__file__, 149, 11), check_func_14443, *[str_14444], **kwargs_14452)
    
    # Testing the type of an if condition (line 149)
    if_condition_14454 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 8), check_func_call_result_14453)
    # Assigning a type to the variable 'if_condition_14454' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'if_condition_14454', if_condition_14454)
    # SSA begins for if statement (line 149)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 150)
    # Processing the call arguments (line 150)
    
    # Obtaining an instance of the builtin type 'tuple' (line 150)
    tuple_14457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 150)
    # Adding element type (line 150)
    
    # Call to replace(...): (line 150)
    # Processing the call arguments (line 150)
    str_14463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 50), 'str', '.')
    str_14464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 55), 'str', '_')
    # Processing the call keyword arguments (line 150)
    kwargs_14465 = {}
    
    # Call to fname2def(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'h' (line 150)
    h_14459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 39), 'h', False)
    # Processing the call keyword arguments (line 150)
    kwargs_14460 = {}
    # Getting the type of 'fname2def' (line 150)
    fname2def_14458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 29), 'fname2def', False)
    # Calling fname2def(args, kwargs) (line 150)
    fname2def_call_result_14461 = invoke(stypy.reporting.localization.Localization(__file__, 150, 29), fname2def_14458, *[h_14459], **kwargs_14460)
    
    # Obtaining the member 'replace' of a type (line 150)
    replace_14462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 29), fname2def_call_result_14461, 'replace')
    # Calling replace(args, kwargs) (line 150)
    replace_call_result_14466 = invoke(stypy.reporting.localization.Localization(__file__, 150, 29), replace_14462, *[str_14463, str_14464], **kwargs_14465)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 29), tuple_14457, replace_call_result_14466)
    # Adding element type (line 150)
    int_14467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 61), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 29), tuple_14457, int_14467)
    
    # Processing the call keyword arguments (line 150)
    kwargs_14468 = {}
    # Getting the type of 'moredefs' (line 150)
    moredefs_14455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'moredefs', False)
    # Obtaining the member 'append' of a type (line 150)
    append_14456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 12), moredefs_14455, 'append')
    # Calling append(args, kwargs) (line 150)
    append_call_result_14469 = invoke(stypy.reporting.localization.Localization(__file__, 150, 12), append_14456, *[tuple_14457], **kwargs_14468)
    
    # SSA join for if statement (line 149)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'OPTIONAL_INTRINSICS' (line 152)
    OPTIONAL_INTRINSICS_14470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'OPTIONAL_INTRINSICS')
    # Testing the type of a for loop iterable (line 152)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 152, 4), OPTIONAL_INTRINSICS_14470)
    # Getting the type of the for loop variable (line 152)
    for_loop_var_14471 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 152, 4), OPTIONAL_INTRINSICS_14470)
    # Assigning a type to the variable 'tup' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'tup', for_loop_var_14471)
    # SSA begins for a for statement (line 152)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 153):
    
    # Assigning a Name to a Name (line 153):
    # Getting the type of 'None' (line 153)
    None_14472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 18), 'None')
    # Assigning a type to the variable 'headers' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'headers', None_14472)
    
    
    
    # Call to len(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'tup' (line 154)
    tup_14474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'tup', False)
    # Processing the call keyword arguments (line 154)
    kwargs_14475 = {}
    # Getting the type of 'len' (line 154)
    len_14473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 11), 'len', False)
    # Calling len(args, kwargs) (line 154)
    len_call_result_14476 = invoke(stypy.reporting.localization.Localization(__file__, 154, 11), len_14473, *[tup_14474], **kwargs_14475)
    
    int_14477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 23), 'int')
    # Applying the binary operator '==' (line 154)
    result_eq_14478 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 11), '==', len_call_result_14476, int_14477)
    
    # Testing the type of an if condition (line 154)
    if_condition_14479 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 8), result_eq_14478)
    # Assigning a type to the variable 'if_condition_14479' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'if_condition_14479', if_condition_14479)
    # SSA begins for if statement (line 154)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Tuple (line 155):
    
    # Assigning a Subscript to a Name (line 155):
    
    # Obtaining the type of the subscript
    int_14480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 12), 'int')
    # Getting the type of 'tup' (line 155)
    tup_14481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 22), 'tup')
    # Obtaining the member '__getitem__' of a type (line 155)
    getitem___14482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), tup_14481, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 155)
    subscript_call_result_14483 = invoke(stypy.reporting.localization.Localization(__file__, 155, 12), getitem___14482, int_14480)
    
    # Assigning a type to the variable 'tuple_var_assignment_14129' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'tuple_var_assignment_14129', subscript_call_result_14483)
    
    # Assigning a Subscript to a Name (line 155):
    
    # Obtaining the type of the subscript
    int_14484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 12), 'int')
    # Getting the type of 'tup' (line 155)
    tup_14485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 22), 'tup')
    # Obtaining the member '__getitem__' of a type (line 155)
    getitem___14486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), tup_14485, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 155)
    subscript_call_result_14487 = invoke(stypy.reporting.localization.Localization(__file__, 155, 12), getitem___14486, int_14484)
    
    # Assigning a type to the variable 'tuple_var_assignment_14130' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'tuple_var_assignment_14130', subscript_call_result_14487)
    
    # Assigning a Name to a Name (line 155):
    # Getting the type of 'tuple_var_assignment_14129' (line 155)
    tuple_var_assignment_14129_14488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'tuple_var_assignment_14129')
    # Assigning a type to the variable 'f' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'f', tuple_var_assignment_14129_14488)
    
    # Assigning a Name to a Name (line 155):
    # Getting the type of 'tuple_var_assignment_14130' (line 155)
    tuple_var_assignment_14130_14489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'tuple_var_assignment_14130')
    # Assigning a type to the variable 'args' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 15), 'args', tuple_var_assignment_14130_14489)
    # SSA branch for the else part of an if statement (line 154)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Tuple (line 157):
    
    # Assigning a Subscript to a Name (line 157):
    
    # Obtaining the type of the subscript
    int_14490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 35), 'int')
    # Getting the type of 'tup' (line 157)
    tup_14491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 31), 'tup')
    # Obtaining the member '__getitem__' of a type (line 157)
    getitem___14492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 31), tup_14491, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 157)
    subscript_call_result_14493 = invoke(stypy.reporting.localization.Localization(__file__, 157, 31), getitem___14492, int_14490)
    
    # Assigning a type to the variable 'tuple_assignment_14131' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'tuple_assignment_14131', subscript_call_result_14493)
    
    # Assigning a Subscript to a Name (line 157):
    
    # Obtaining the type of the subscript
    int_14494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 43), 'int')
    # Getting the type of 'tup' (line 157)
    tup_14495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 39), 'tup')
    # Obtaining the member '__getitem__' of a type (line 157)
    getitem___14496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 39), tup_14495, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 157)
    subscript_call_result_14497 = invoke(stypy.reporting.localization.Localization(__file__, 157, 39), getitem___14496, int_14494)
    
    # Assigning a type to the variable 'tuple_assignment_14132' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'tuple_assignment_14132', subscript_call_result_14497)
    
    # Assigning a List to a Name (line 157):
    
    # Obtaining an instance of the builtin type 'list' (line 157)
    list_14498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 47), 'list')
    # Adding type elements to the builtin type 'list' instance (line 157)
    # Adding element type (line 157)
    
    # Obtaining the type of the subscript
    int_14499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 52), 'int')
    # Getting the type of 'tup' (line 157)
    tup_14500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 48), 'tup')
    # Obtaining the member '__getitem__' of a type (line 157)
    getitem___14501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 48), tup_14500, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 157)
    subscript_call_result_14502 = invoke(stypy.reporting.localization.Localization(__file__, 157, 48), getitem___14501, int_14499)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 47), list_14498, subscript_call_result_14502)
    
    # Assigning a type to the variable 'tuple_assignment_14133' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'tuple_assignment_14133', list_14498)
    
    # Assigning a Name to a Name (line 157):
    # Getting the type of 'tuple_assignment_14131' (line 157)
    tuple_assignment_14131_14503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'tuple_assignment_14131')
    # Assigning a type to the variable 'f' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'f', tuple_assignment_14131_14503)
    
    # Assigning a Name to a Name (line 157):
    # Getting the type of 'tuple_assignment_14132' (line 157)
    tuple_assignment_14132_14504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'tuple_assignment_14132')
    # Assigning a type to the variable 'args' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 15), 'args', tuple_assignment_14132_14504)
    
    # Assigning a Name to a Name (line 157):
    # Getting the type of 'tuple_assignment_14133' (line 157)
    tuple_assignment_14133_14505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'tuple_assignment_14133')
    # Assigning a type to the variable 'headers' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 21), 'headers', tuple_assignment_14133_14505)
    # SSA join for if statement (line 154)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to check_func(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'f' (line 158)
    f_14508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 29), 'f', False)
    # Processing the call keyword arguments (line 158)
    # Getting the type of 'False' (line 158)
    False_14509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 37), 'False', False)
    keyword_14510 = False_14509
    # Getting the type of 'True' (line 158)
    True_14511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 49), 'True', False)
    keyword_14512 = True_14511
    # Getting the type of 'args' (line 158)
    args_14513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 65), 'args', False)
    keyword_14514 = args_14513
    # Getting the type of 'headers' (line 159)
    headers_14515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 37), 'headers', False)
    keyword_14516 = headers_14515
    kwargs_14517 = {'decl': keyword_14510, 'headers': keyword_14516, 'call': keyword_14512, 'call_args': keyword_14514}
    # Getting the type of 'config' (line 158)
    config_14506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), 'config', False)
    # Obtaining the member 'check_func' of a type (line 158)
    check_func_14507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 11), config_14506, 'check_func')
    # Calling check_func(args, kwargs) (line 158)
    check_func_call_result_14518 = invoke(stypy.reporting.localization.Localization(__file__, 158, 11), check_func_14507, *[f_14508], **kwargs_14517)
    
    # Testing the type of an if condition (line 158)
    if_condition_14519 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 8), check_func_call_result_14518)
    # Assigning a type to the variable 'if_condition_14519' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'if_condition_14519', if_condition_14519)
    # SSA begins for if statement (line 158)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 160)
    # Processing the call arguments (line 160)
    
    # Obtaining an instance of the builtin type 'tuple' (line 160)
    tuple_14522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 160)
    # Adding element type (line 160)
    
    # Call to fname2def(...): (line 160)
    # Processing the call arguments (line 160)
    # Getting the type of 'f' (line 160)
    f_14524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 39), 'f', False)
    # Processing the call keyword arguments (line 160)
    kwargs_14525 = {}
    # Getting the type of 'fname2def' (line 160)
    fname2def_14523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 29), 'fname2def', False)
    # Calling fname2def(args, kwargs) (line 160)
    fname2def_call_result_14526 = invoke(stypy.reporting.localization.Localization(__file__, 160, 29), fname2def_14523, *[f_14524], **kwargs_14525)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 29), tuple_14522, fname2def_call_result_14526)
    # Adding element type (line 160)
    int_14527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 29), tuple_14522, int_14527)
    
    # Processing the call keyword arguments (line 160)
    kwargs_14528 = {}
    # Getting the type of 'moredefs' (line 160)
    moredefs_14520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'moredefs', False)
    # Obtaining the member 'append' of a type (line 160)
    append_14521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 12), moredefs_14520, 'append')
    # Calling append(args, kwargs) (line 160)
    append_call_result_14529 = invoke(stypy.reporting.localization.Localization(__file__, 160, 12), append_14521, *[tuple_14522], **kwargs_14528)
    
    # SSA join for if statement (line 158)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'OPTIONAL_FUNCTION_ATTRIBUTES' (line 162)
    OPTIONAL_FUNCTION_ATTRIBUTES_14530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), 'OPTIONAL_FUNCTION_ATTRIBUTES')
    # Testing the type of a for loop iterable (line 162)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 162, 4), OPTIONAL_FUNCTION_ATTRIBUTES_14530)
    # Getting the type of the for loop variable (line 162)
    for_loop_var_14531 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 162, 4), OPTIONAL_FUNCTION_ATTRIBUTES_14530)
    # Assigning a type to the variable 'dec' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'dec', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 4), for_loop_var_14531))
    # Assigning a type to the variable 'fn' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'fn', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 4), for_loop_var_14531))
    # SSA begins for a for statement (line 162)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to check_gcc_function_attribute(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'dec' (line 163)
    dec_14534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 47), 'dec', False)
    # Getting the type of 'fn' (line 163)
    fn_14535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 52), 'fn', False)
    # Processing the call keyword arguments (line 163)
    kwargs_14536 = {}
    # Getting the type of 'config' (line 163)
    config_14532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'config', False)
    # Obtaining the member 'check_gcc_function_attribute' of a type (line 163)
    check_gcc_function_attribute_14533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 11), config_14532, 'check_gcc_function_attribute')
    # Calling check_gcc_function_attribute(args, kwargs) (line 163)
    check_gcc_function_attribute_call_result_14537 = invoke(stypy.reporting.localization.Localization(__file__, 163, 11), check_gcc_function_attribute_14533, *[dec_14534, fn_14535], **kwargs_14536)
    
    # Testing the type of an if condition (line 163)
    if_condition_14538 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 8), check_gcc_function_attribute_call_result_14537)
    # Assigning a type to the variable 'if_condition_14538' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'if_condition_14538', if_condition_14538)
    # SSA begins for if statement (line 163)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 164)
    # Processing the call arguments (line 164)
    
    # Obtaining an instance of the builtin type 'tuple' (line 164)
    tuple_14541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 164)
    # Adding element type (line 164)
    
    # Call to fname2def(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'fn' (line 164)
    fn_14543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 39), 'fn', False)
    # Processing the call keyword arguments (line 164)
    kwargs_14544 = {}
    # Getting the type of 'fname2def' (line 164)
    fname2def_14542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 29), 'fname2def', False)
    # Calling fname2def(args, kwargs) (line 164)
    fname2def_call_result_14545 = invoke(stypy.reporting.localization.Localization(__file__, 164, 29), fname2def_14542, *[fn_14543], **kwargs_14544)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 29), tuple_14541, fname2def_call_result_14545)
    # Adding element type (line 164)
    int_14546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 29), tuple_14541, int_14546)
    
    # Processing the call keyword arguments (line 164)
    kwargs_14547 = {}
    # Getting the type of 'moredefs' (line 164)
    moredefs_14539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'moredefs', False)
    # Obtaining the member 'append' of a type (line 164)
    append_14540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 12), moredefs_14539, 'append')
    # Calling append(args, kwargs) (line 164)
    append_call_result_14548 = invoke(stypy.reporting.localization.Localization(__file__, 164, 12), append_14540, *[tuple_14541], **kwargs_14547)
    
    # SSA join for if statement (line 163)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'OPTIONAL_VARIABLE_ATTRIBUTES' (line 166)
    OPTIONAL_VARIABLE_ATTRIBUTES_14549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 14), 'OPTIONAL_VARIABLE_ATTRIBUTES')
    # Testing the type of a for loop iterable (line 166)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 166, 4), OPTIONAL_VARIABLE_ATTRIBUTES_14549)
    # Getting the type of the for loop variable (line 166)
    for_loop_var_14550 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 166, 4), OPTIONAL_VARIABLE_ATTRIBUTES_14549)
    # Assigning a type to the variable 'fn' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'fn', for_loop_var_14550)
    # SSA begins for a for statement (line 166)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to check_gcc_variable_attribute(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'fn' (line 167)
    fn_14553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 47), 'fn', False)
    # Processing the call keyword arguments (line 167)
    kwargs_14554 = {}
    # Getting the type of 'config' (line 167)
    config_14551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'config', False)
    # Obtaining the member 'check_gcc_variable_attribute' of a type (line 167)
    check_gcc_variable_attribute_14552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 11), config_14551, 'check_gcc_variable_attribute')
    # Calling check_gcc_variable_attribute(args, kwargs) (line 167)
    check_gcc_variable_attribute_call_result_14555 = invoke(stypy.reporting.localization.Localization(__file__, 167, 11), check_gcc_variable_attribute_14552, *[fn_14553], **kwargs_14554)
    
    # Testing the type of an if condition (line 167)
    if_condition_14556 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 8), check_gcc_variable_attribute_call_result_14555)
    # Assigning a type to the variable 'if_condition_14556' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'if_condition_14556', if_condition_14556)
    # SSA begins for if statement (line 167)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 168):
    
    # Assigning a Call to a Name (line 168):
    
    # Call to replace(...): (line 168)
    # Processing the call arguments (line 168)
    str_14564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 45), 'str', ')')
    str_14565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 50), 'str', '_')
    # Processing the call keyword arguments (line 168)
    kwargs_14566 = {}
    
    # Call to replace(...): (line 168)
    # Processing the call arguments (line 168)
    str_14559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 27), 'str', '(')
    str_14560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 32), 'str', '_')
    # Processing the call keyword arguments (line 168)
    kwargs_14561 = {}
    # Getting the type of 'fn' (line 168)
    fn_14557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'fn', False)
    # Obtaining the member 'replace' of a type (line 168)
    replace_14558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 16), fn_14557, 'replace')
    # Calling replace(args, kwargs) (line 168)
    replace_call_result_14562 = invoke(stypy.reporting.localization.Localization(__file__, 168, 16), replace_14558, *[str_14559, str_14560], **kwargs_14561)
    
    # Obtaining the member 'replace' of a type (line 168)
    replace_14563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 16), replace_call_result_14562, 'replace')
    # Calling replace(args, kwargs) (line 168)
    replace_call_result_14567 = invoke(stypy.reporting.localization.Localization(__file__, 168, 16), replace_14563, *[str_14564, str_14565], **kwargs_14566)
    
    # Assigning a type to the variable 'm' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'm', replace_call_result_14567)
    
    # Call to append(...): (line 169)
    # Processing the call arguments (line 169)
    
    # Obtaining an instance of the builtin type 'tuple' (line 169)
    tuple_14570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 169)
    # Adding element type (line 169)
    
    # Call to fname2def(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'm' (line 169)
    m_14572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 39), 'm', False)
    # Processing the call keyword arguments (line 169)
    kwargs_14573 = {}
    # Getting the type of 'fname2def' (line 169)
    fname2def_14571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 29), 'fname2def', False)
    # Calling fname2def(args, kwargs) (line 169)
    fname2def_call_result_14574 = invoke(stypy.reporting.localization.Localization(__file__, 169, 29), fname2def_14571, *[m_14572], **kwargs_14573)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 29), tuple_14570, fname2def_call_result_14574)
    # Adding element type (line 169)
    int_14575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 29), tuple_14570, int_14575)
    
    # Processing the call keyword arguments (line 169)
    kwargs_14576 = {}
    # Getting the type of 'moredefs' (line 169)
    moredefs_14568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'moredefs', False)
    # Obtaining the member 'append' of a type (line 169)
    append_14569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 12), moredefs_14568, 'append')
    # Calling append(args, kwargs) (line 169)
    append_call_result_14577 = invoke(stypy.reporting.localization.Localization(__file__, 169, 12), append_14569, *[tuple_14570], **kwargs_14576)
    
    # SSA join for if statement (line 167)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to check_funcs(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'C99_FUNCS_SINGLE' (line 172)
    C99_FUNCS_SINGLE_14579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'C99_FUNCS_SINGLE', False)
    # Processing the call keyword arguments (line 172)
    kwargs_14580 = {}
    # Getting the type of 'check_funcs' (line 172)
    check_funcs_14578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'check_funcs', False)
    # Calling check_funcs(args, kwargs) (line 172)
    check_funcs_call_result_14581 = invoke(stypy.reporting.localization.Localization(__file__, 172, 4), check_funcs_14578, *[C99_FUNCS_SINGLE_14579], **kwargs_14580)
    
    
    # Call to check_funcs(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'C99_FUNCS_EXTENDED' (line 173)
    C99_FUNCS_EXTENDED_14583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'C99_FUNCS_EXTENDED', False)
    # Processing the call keyword arguments (line 173)
    kwargs_14584 = {}
    # Getting the type of 'check_funcs' (line 173)
    check_funcs_14582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'check_funcs', False)
    # Calling check_funcs(args, kwargs) (line 173)
    check_funcs_call_result_14585 = invoke(stypy.reporting.localization.Localization(__file__, 173, 4), check_funcs_14582, *[C99_FUNCS_EXTENDED_14583], **kwargs_14584)
    
    
    # ################# End of 'check_math_capabilities(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_math_capabilities' in the type store
    # Getting the type of 'stypy_return_type' (line 103)
    stypy_return_type_14586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14586)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_math_capabilities'
    return stypy_return_type_14586

# Assigning a type to the variable 'check_math_capabilities' (line 103)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'check_math_capabilities', check_math_capabilities)

@norecursion
def check_complex(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_complex'
    module_type_store = module_type_store.open_function_context('check_complex', 175, 0, False)
    
    # Passed parameters checking function
    check_complex.stypy_localization = localization
    check_complex.stypy_type_of_self = None
    check_complex.stypy_type_store = module_type_store
    check_complex.stypy_function_name = 'check_complex'
    check_complex.stypy_param_names_list = ['config', 'mathlibs']
    check_complex.stypy_varargs_param_name = None
    check_complex.stypy_kwargs_param_name = None
    check_complex.stypy_call_defaults = defaults
    check_complex.stypy_call_varargs = varargs
    check_complex.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_complex', ['config', 'mathlibs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_complex', localization, ['config', 'mathlibs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_complex(...)' code ##################

    
    # Assigning a List to a Name (line 176):
    
    # Assigning a List to a Name (line 176):
    
    # Obtaining an instance of the builtin type 'list' (line 176)
    list_14587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 176)
    
    # Assigning a type to the variable 'priv' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'priv', list_14587)
    
    # Assigning a List to a Name (line 177):
    
    # Assigning a List to a Name (line 177):
    
    # Obtaining an instance of the builtin type 'list' (line 177)
    list_14588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 177)
    
    # Assigning a type to the variable 'pub' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'pub', list_14588)
    
    
    # SSA begins for try-except statement (line 179)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    
    
    # Obtaining the type of the subscript
    int_14589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 22), 'int')
    
    # Call to uname(...): (line 180)
    # Processing the call keyword arguments (line 180)
    kwargs_14592 = {}
    # Getting the type of 'os' (line 180)
    os_14590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 11), 'os', False)
    # Obtaining the member 'uname' of a type (line 180)
    uname_14591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 11), os_14590, 'uname')
    # Calling uname(args, kwargs) (line 180)
    uname_call_result_14593 = invoke(stypy.reporting.localization.Localization(__file__, 180, 11), uname_14591, *[], **kwargs_14592)
    
    # Obtaining the member '__getitem__' of a type (line 180)
    getitem___14594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 11), uname_call_result_14593, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 180)
    subscript_call_result_14595 = invoke(stypy.reporting.localization.Localization(__file__, 180, 11), getitem___14594, int_14589)
    
    str_14596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 28), 'str', 'Interix')
    # Applying the binary operator '==' (line 180)
    result_eq_14597 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 11), '==', subscript_call_result_14595, str_14596)
    
    # Testing the type of an if condition (line 180)
    if_condition_14598 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 8), result_eq_14597)
    # Assigning a type to the variable 'if_condition_14598' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'if_condition_14598', if_condition_14598)
    # SSA begins for if statement (line 180)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 181)
    # Processing the call arguments (line 181)
    str_14601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 26), 'str', 'Disabling broken complex support. See #1365')
    # Processing the call keyword arguments (line 181)
    kwargs_14602 = {}
    # Getting the type of 'warnings' (line 181)
    warnings_14599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 181)
    warn_14600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 12), warnings_14599, 'warn')
    # Calling warn(args, kwargs) (line 181)
    warn_call_result_14603 = invoke(stypy.reporting.localization.Localization(__file__, 181, 12), warn_14600, *[str_14601], **kwargs_14602)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 182)
    tuple_14604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 182)
    # Adding element type (line 182)
    # Getting the type of 'priv' (line 182)
    priv_14605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 19), 'priv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 19), tuple_14604, priv_14605)
    # Adding element type (line 182)
    # Getting the type of 'pub' (line 182)
    pub_14606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 25), 'pub')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 19), tuple_14604, pub_14606)
    
    # Assigning a type to the variable 'stypy_return_type' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'stypy_return_type', tuple_14604)
    # SSA join for if statement (line 180)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 179)
    # SSA branch for the except '<any exception>' branch of a try statement (line 179)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 179)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 188):
    
    # Assigning a Call to a Name (line 188):
    
    # Call to check_header(...): (line 188)
    # Processing the call arguments (line 188)
    str_14609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 29), 'str', 'complex.h')
    # Processing the call keyword arguments (line 188)
    kwargs_14610 = {}
    # Getting the type of 'config' (line 188)
    config_14607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 9), 'config', False)
    # Obtaining the member 'check_header' of a type (line 188)
    check_header_14608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 9), config_14607, 'check_header')
    # Calling check_header(args, kwargs) (line 188)
    check_header_call_result_14611 = invoke(stypy.reporting.localization.Localization(__file__, 188, 9), check_header_14608, *[str_14609], **kwargs_14610)
    
    # Assigning a type to the variable 'st' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'st', check_header_call_result_14611)
    
    # Getting the type of 'st' (line 189)
    st_14612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 7), 'st')
    # Testing the type of an if condition (line 189)
    if_condition_14613 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 4), st_14612)
    # Assigning a type to the variable 'if_condition_14613' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'if_condition_14613', if_condition_14613)
    # SSA begins for if statement (line 189)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 190)
    # Processing the call arguments (line 190)
    
    # Obtaining an instance of the builtin type 'tuple' (line 190)
    tuple_14616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 190)
    # Adding element type (line 190)
    str_14617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 21), 'str', 'HAVE_COMPLEX_H')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 21), tuple_14616, str_14617)
    # Adding element type (line 190)
    int_14618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 21), tuple_14616, int_14618)
    
    # Processing the call keyword arguments (line 190)
    kwargs_14619 = {}
    # Getting the type of 'priv' (line 190)
    priv_14614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'priv', False)
    # Obtaining the member 'append' of a type (line 190)
    append_14615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), priv_14614, 'append')
    # Calling append(args, kwargs) (line 190)
    append_call_result_14620 = invoke(stypy.reporting.localization.Localization(__file__, 190, 8), append_14615, *[tuple_14616], **kwargs_14619)
    
    
    # Call to append(...): (line 191)
    # Processing the call arguments (line 191)
    
    # Obtaining an instance of the builtin type 'tuple' (line 191)
    tuple_14623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 191)
    # Adding element type (line 191)
    str_14624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 20), 'str', 'NPY_USE_C99_COMPLEX')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 20), tuple_14623, str_14624)
    # Adding element type (line 191)
    int_14625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 20), tuple_14623, int_14625)
    
    # Processing the call keyword arguments (line 191)
    kwargs_14626 = {}
    # Getting the type of 'pub' (line 191)
    pub_14621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'pub', False)
    # Obtaining the member 'append' of a type (line 191)
    append_14622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), pub_14621, 'append')
    # Calling append(args, kwargs) (line 191)
    append_call_result_14627 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), append_14622, *[tuple_14623], **kwargs_14626)
    
    
    # Getting the type of 'C99_COMPLEX_TYPES' (line 193)
    C99_COMPLEX_TYPES_14628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 17), 'C99_COMPLEX_TYPES')
    # Testing the type of a for loop iterable (line 193)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 193, 8), C99_COMPLEX_TYPES_14628)
    # Getting the type of the for loop variable (line 193)
    for_loop_var_14629 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 193, 8), C99_COMPLEX_TYPES_14628)
    # Assigning a type to the variable 't' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 't', for_loop_var_14629)
    # SSA begins for a for statement (line 193)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 194):
    
    # Assigning a Call to a Name (line 194):
    
    # Call to check_type(...): (line 194)
    # Processing the call arguments (line 194)
    # Getting the type of 't' (line 194)
    t_14632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 35), 't', False)
    # Processing the call keyword arguments (line 194)
    
    # Obtaining an instance of the builtin type 'list' (line 194)
    list_14633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 46), 'list')
    # Adding type elements to the builtin type 'list' instance (line 194)
    # Adding element type (line 194)
    str_14634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 47), 'str', 'complex.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 46), list_14633, str_14634)
    
    keyword_14635 = list_14633
    kwargs_14636 = {'headers': keyword_14635}
    # Getting the type of 'config' (line 194)
    config_14630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'config', False)
    # Obtaining the member 'check_type' of a type (line 194)
    check_type_14631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 17), config_14630, 'check_type')
    # Calling check_type(args, kwargs) (line 194)
    check_type_call_result_14637 = invoke(stypy.reporting.localization.Localization(__file__, 194, 17), check_type_14631, *[t_14632], **kwargs_14636)
    
    # Assigning a type to the variable 'st' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'st', check_type_call_result_14637)
    
    # Getting the type of 'st' (line 195)
    st_14638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'st')
    # Testing the type of an if condition (line 195)
    if_condition_14639 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 12), st_14638)
    # Assigning a type to the variable 'if_condition_14639' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'if_condition_14639', if_condition_14639)
    # SSA begins for if statement (line 195)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 196)
    # Processing the call arguments (line 196)
    
    # Obtaining an instance of the builtin type 'tuple' (line 196)
    tuple_14642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 196)
    # Adding element type (line 196)
    str_14643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 28), 'str', 'NPY_HAVE_%s')
    
    # Call to type2def(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of 't' (line 196)
    t_14645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 53), 't', False)
    # Processing the call keyword arguments (line 196)
    kwargs_14646 = {}
    # Getting the type of 'type2def' (line 196)
    type2def_14644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 44), 'type2def', False)
    # Calling type2def(args, kwargs) (line 196)
    type2def_call_result_14647 = invoke(stypy.reporting.localization.Localization(__file__, 196, 44), type2def_14644, *[t_14645], **kwargs_14646)
    
    # Applying the binary operator '%' (line 196)
    result_mod_14648 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 28), '%', str_14643, type2def_call_result_14647)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 28), tuple_14642, result_mod_14648)
    # Adding element type (line 196)
    int_14649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 28), tuple_14642, int_14649)
    
    # Processing the call keyword arguments (line 196)
    kwargs_14650 = {}
    # Getting the type of 'pub' (line 196)
    pub_14640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'pub', False)
    # Obtaining the member 'append' of a type (line 196)
    append_14641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 16), pub_14640, 'append')
    # Calling append(args, kwargs) (line 196)
    append_call_result_14651 = invoke(stypy.reporting.localization.Localization(__file__, 196, 16), append_14641, *[tuple_14642], **kwargs_14650)
    
    # SSA join for if statement (line 195)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def check_prec(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_prec'
        module_type_store = module_type_store.open_function_context('check_prec', 198, 8, False)
        
        # Passed parameters checking function
        check_prec.stypy_localization = localization
        check_prec.stypy_type_of_self = None
        check_prec.stypy_type_store = module_type_store
        check_prec.stypy_function_name = 'check_prec'
        check_prec.stypy_param_names_list = ['prec']
        check_prec.stypy_varargs_param_name = None
        check_prec.stypy_kwargs_param_name = None
        check_prec.stypy_call_defaults = defaults
        check_prec.stypy_call_varargs = varargs
        check_prec.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check_prec', ['prec'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_prec', localization, ['prec'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_prec(...)' code ##################

        
        # Assigning a ListComp to a Name (line 199):
        
        # Assigning a ListComp to a Name (line 199):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'C99_COMPLEX_FUNCS' (line 199)
        C99_COMPLEX_FUNCS_14655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 39), 'C99_COMPLEX_FUNCS')
        comprehension_14656 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 21), C99_COMPLEX_FUNCS_14655)
        # Assigning a type to the variable 'f' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 21), 'f', comprehension_14656)
        # Getting the type of 'f' (line 199)
        f_14652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 21), 'f')
        # Getting the type of 'prec' (line 199)
        prec_14653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 25), 'prec')
        # Applying the binary operator '+' (line 199)
        result_add_14654 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 21), '+', f_14652, prec_14653)
        
        list_14657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 21), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 21), list_14657, result_add_14654)
        # Assigning a type to the variable 'flist' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'flist', list_14657)
        
        # Assigning a Call to a Name (line 200):
        
        # Assigning a Call to a Name (line 200):
        
        # Call to dict(...): (line 200)
        # Processing the call arguments (line 200)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'flist' (line 200)
        flist_14662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 44), 'flist', False)
        comprehension_14663 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 25), flist_14662)
        # Assigning a type to the variable 'f' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 25), 'f', comprehension_14663)
        
        # Obtaining an instance of the builtin type 'tuple' (line 200)
        tuple_14659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 200)
        # Adding element type (line 200)
        # Getting the type of 'f' (line 200)
        f_14660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 26), 'f', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 26), tuple_14659, f_14660)
        # Adding element type (line 200)
        # Getting the type of 'True' (line 200)
        True_14661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 29), 'True', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 26), tuple_14659, True_14661)
        
        list_14664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 25), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 25), list_14664, tuple_14659)
        # Processing the call keyword arguments (line 200)
        kwargs_14665 = {}
        # Getting the type of 'dict' (line 200)
        dict_14658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 19), 'dict', False)
        # Calling dict(args, kwargs) (line 200)
        dict_call_result_14666 = invoke(stypy.reporting.localization.Localization(__file__, 200, 19), dict_14658, *[list_14664], **kwargs_14665)
        
        # Assigning a type to the variable 'decl' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'decl', dict_call_result_14666)
        
        
        
        # Call to check_funcs_once(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'flist' (line 201)
        flist_14669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 43), 'flist', False)
        # Processing the call keyword arguments (line 201)
        # Getting the type of 'decl' (line 201)
        decl_14670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 55), 'decl', False)
        keyword_14671 = decl_14670
        # Getting the type of 'decl' (line 201)
        decl_14672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 66), 'decl', False)
        keyword_14673 = decl_14672
        # Getting the type of 'mathlibs' (line 202)
        mathlibs_14674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 53), 'mathlibs', False)
        keyword_14675 = mathlibs_14674
        kwargs_14676 = {'decl': keyword_14673, 'libraries': keyword_14675, 'call': keyword_14671}
        # Getting the type of 'config' (line 201)
        config_14667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 19), 'config', False)
        # Obtaining the member 'check_funcs_once' of a type (line 201)
        check_funcs_once_14668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 19), config_14667, 'check_funcs_once')
        # Calling check_funcs_once(args, kwargs) (line 201)
        check_funcs_once_call_result_14677 = invoke(stypy.reporting.localization.Localization(__file__, 201, 19), check_funcs_once_14668, *[flist_14669], **kwargs_14676)
        
        # Applying the 'not' unary operator (line 201)
        result_not__14678 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 15), 'not', check_funcs_once_call_result_14677)
        
        # Testing the type of an if condition (line 201)
        if_condition_14679 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 12), result_not__14678)
        # Assigning a type to the variable 'if_condition_14679' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'if_condition_14679', if_condition_14679)
        # SSA begins for if statement (line 201)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'flist' (line 203)
        flist_14680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 25), 'flist')
        # Testing the type of a for loop iterable (line 203)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 203, 16), flist_14680)
        # Getting the type of the for loop variable (line 203)
        for_loop_var_14681 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 203, 16), flist_14680)
        # Assigning a type to the variable 'f' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'f', for_loop_var_14681)
        # SSA begins for a for statement (line 203)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to check_func(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'f' (line 204)
        f_14684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 41), 'f', False)
        # Processing the call keyword arguments (line 204)
        # Getting the type of 'True' (line 204)
        True_14685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 49), 'True', False)
        keyword_14686 = True_14685
        # Getting the type of 'True' (line 204)
        True_14687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 60), 'True', False)
        keyword_14688 = True_14687
        # Getting the type of 'mathlibs' (line 205)
        mathlibs_14689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 51), 'mathlibs', False)
        keyword_14690 = mathlibs_14689
        kwargs_14691 = {'decl': keyword_14688, 'libraries': keyword_14690, 'call': keyword_14686}
        # Getting the type of 'config' (line 204)
        config_14682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 23), 'config', False)
        # Obtaining the member 'check_func' of a type (line 204)
        check_func_14683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 23), config_14682, 'check_func')
        # Calling check_func(args, kwargs) (line 204)
        check_func_call_result_14692 = invoke(stypy.reporting.localization.Localization(__file__, 204, 23), check_func_14683, *[f_14684], **kwargs_14691)
        
        # Testing the type of an if condition (line 204)
        if_condition_14693 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 204, 20), check_func_call_result_14692)
        # Assigning a type to the variable 'if_condition_14693' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 20), 'if_condition_14693', if_condition_14693)
        # SSA begins for if statement (line 204)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 206)
        # Processing the call arguments (line 206)
        
        # Obtaining an instance of the builtin type 'tuple' (line 206)
        tuple_14696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 206)
        # Adding element type (line 206)
        
        # Call to fname2def(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'f' (line 206)
        f_14698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 47), 'f', False)
        # Processing the call keyword arguments (line 206)
        kwargs_14699 = {}
        # Getting the type of 'fname2def' (line 206)
        fname2def_14697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 37), 'fname2def', False)
        # Calling fname2def(args, kwargs) (line 206)
        fname2def_call_result_14700 = invoke(stypy.reporting.localization.Localization(__file__, 206, 37), fname2def_14697, *[f_14698], **kwargs_14699)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 37), tuple_14696, fname2def_call_result_14700)
        # Adding element type (line 206)
        int_14701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 37), tuple_14696, int_14701)
        
        # Processing the call keyword arguments (line 206)
        kwargs_14702 = {}
        # Getting the type of 'priv' (line 206)
        priv_14694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 24), 'priv', False)
        # Obtaining the member 'append' of a type (line 206)
        append_14695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 24), priv_14694, 'append')
        # Calling append(args, kwargs) (line 206)
        append_call_result_14703 = invoke(stypy.reporting.localization.Localization(__file__, 206, 24), append_14695, *[tuple_14696], **kwargs_14702)
        
        # SSA join for if statement (line 204)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 201)
        module_type_store.open_ssa_branch('else')
        
        # Call to extend(...): (line 208)
        # Processing the call arguments (line 208)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'flist' (line 208)
        flist_14712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 56), 'flist', False)
        comprehension_14713 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 29), flist_14712)
        # Assigning a type to the variable 'f' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 29), 'f', comprehension_14713)
        
        # Obtaining an instance of the builtin type 'tuple' (line 208)
        tuple_14706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 208)
        # Adding element type (line 208)
        
        # Call to fname2def(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'f' (line 208)
        f_14708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 40), 'f', False)
        # Processing the call keyword arguments (line 208)
        kwargs_14709 = {}
        # Getting the type of 'fname2def' (line 208)
        fname2def_14707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 30), 'fname2def', False)
        # Calling fname2def(args, kwargs) (line 208)
        fname2def_call_result_14710 = invoke(stypy.reporting.localization.Localization(__file__, 208, 30), fname2def_14707, *[f_14708], **kwargs_14709)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 30), tuple_14706, fname2def_call_result_14710)
        # Adding element type (line 208)
        int_14711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 30), tuple_14706, int_14711)
        
        list_14714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 29), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 29), list_14714, tuple_14706)
        # Processing the call keyword arguments (line 208)
        kwargs_14715 = {}
        # Getting the type of 'priv' (line 208)
        priv_14704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 16), 'priv', False)
        # Obtaining the member 'extend' of a type (line 208)
        extend_14705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 16), priv_14704, 'extend')
        # Calling extend(args, kwargs) (line 208)
        extend_call_result_14716 = invoke(stypy.reporting.localization.Localization(__file__, 208, 16), extend_14705, *[list_14714], **kwargs_14715)
        
        # SSA join for if statement (line 201)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check_prec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_prec' in the type store
        # Getting the type of 'stypy_return_type' (line 198)
        stypy_return_type_14717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14717)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_prec'
        return stypy_return_type_14717

    # Assigning a type to the variable 'check_prec' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'check_prec', check_prec)
    
    # Call to check_prec(...): (line 210)
    # Processing the call arguments (line 210)
    str_14719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 19), 'str', '')
    # Processing the call keyword arguments (line 210)
    kwargs_14720 = {}
    # Getting the type of 'check_prec' (line 210)
    check_prec_14718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'check_prec', False)
    # Calling check_prec(args, kwargs) (line 210)
    check_prec_call_result_14721 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), check_prec_14718, *[str_14719], **kwargs_14720)
    
    
    # Call to check_prec(...): (line 211)
    # Processing the call arguments (line 211)
    str_14723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 19), 'str', 'f')
    # Processing the call keyword arguments (line 211)
    kwargs_14724 = {}
    # Getting the type of 'check_prec' (line 211)
    check_prec_14722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'check_prec', False)
    # Calling check_prec(args, kwargs) (line 211)
    check_prec_call_result_14725 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), check_prec_14722, *[str_14723], **kwargs_14724)
    
    
    # Call to check_prec(...): (line 212)
    # Processing the call arguments (line 212)
    str_14727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 19), 'str', 'l')
    # Processing the call keyword arguments (line 212)
    kwargs_14728 = {}
    # Getting the type of 'check_prec' (line 212)
    check_prec_14726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'check_prec', False)
    # Calling check_prec(args, kwargs) (line 212)
    check_prec_call_result_14729 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), check_prec_14726, *[str_14727], **kwargs_14728)
    
    # SSA join for if statement (line 189)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 214)
    tuple_14730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 214)
    # Adding element type (line 214)
    # Getting the type of 'priv' (line 214)
    priv_14731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 11), 'priv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 11), tuple_14730, priv_14731)
    # Adding element type (line 214)
    # Getting the type of 'pub' (line 214)
    pub_14732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 17), 'pub')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 11), tuple_14730, pub_14732)
    
    # Assigning a type to the variable 'stypy_return_type' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'stypy_return_type', tuple_14730)
    
    # ################# End of 'check_complex(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_complex' in the type store
    # Getting the type of 'stypy_return_type' (line 175)
    stypy_return_type_14733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14733)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_complex'
    return stypy_return_type_14733

# Assigning a type to the variable 'check_complex' (line 175)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'check_complex', check_complex)

@norecursion
def check_ieee_macros(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_ieee_macros'
    module_type_store = module_type_store.open_function_context('check_ieee_macros', 216, 0, False)
    
    # Passed parameters checking function
    check_ieee_macros.stypy_localization = localization
    check_ieee_macros.stypy_type_of_self = None
    check_ieee_macros.stypy_type_store = module_type_store
    check_ieee_macros.stypy_function_name = 'check_ieee_macros'
    check_ieee_macros.stypy_param_names_list = ['config']
    check_ieee_macros.stypy_varargs_param_name = None
    check_ieee_macros.stypy_kwargs_param_name = None
    check_ieee_macros.stypy_call_defaults = defaults
    check_ieee_macros.stypy_call_varargs = varargs
    check_ieee_macros.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_ieee_macros', ['config'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_ieee_macros', localization, ['config'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_ieee_macros(...)' code ##################

    
    # Assigning a List to a Name (line 217):
    
    # Assigning a List to a Name (line 217):
    
    # Obtaining an instance of the builtin type 'list' (line 217)
    list_14734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 217)
    
    # Assigning a type to the variable 'priv' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'priv', list_14734)
    
    # Assigning a List to a Name (line 218):
    
    # Assigning a List to a Name (line 218):
    
    # Obtaining an instance of the builtin type 'list' (line 218)
    list_14735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 218)
    
    # Assigning a type to the variable 'pub' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'pub', list_14735)
    
    # Assigning a List to a Name (line 220):
    
    # Assigning a List to a Name (line 220):
    
    # Obtaining an instance of the builtin type 'list' (line 220)
    list_14736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 220)
    
    # Assigning a type to the variable 'macros' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'macros', list_14736)

    @norecursion
    def _add_decl(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_add_decl'
        module_type_store = module_type_store.open_function_context('_add_decl', 222, 4, False)
        
        # Passed parameters checking function
        _add_decl.stypy_localization = localization
        _add_decl.stypy_type_of_self = None
        _add_decl.stypy_type_store = module_type_store
        _add_decl.stypy_function_name = '_add_decl'
        _add_decl.stypy_param_names_list = ['f']
        _add_decl.stypy_varargs_param_name = None
        _add_decl.stypy_kwargs_param_name = None
        _add_decl.stypy_call_defaults = defaults
        _add_decl.stypy_call_varargs = varargs
        _add_decl.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_add_decl', ['f'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_add_decl', localization, ['f'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_add_decl(...)' code ##################

        
        # Call to append(...): (line 223)
        # Processing the call arguments (line 223)
        
        # Call to fname2def(...): (line 223)
        # Processing the call arguments (line 223)
        str_14740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 30), 'str', 'decl_%s')
        # Getting the type of 'f' (line 223)
        f_14741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 42), 'f', False)
        # Applying the binary operator '%' (line 223)
        result_mod_14742 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 30), '%', str_14740, f_14741)
        
        # Processing the call keyword arguments (line 223)
        kwargs_14743 = {}
        # Getting the type of 'fname2def' (line 223)
        fname2def_14739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 20), 'fname2def', False)
        # Calling fname2def(args, kwargs) (line 223)
        fname2def_call_result_14744 = invoke(stypy.reporting.localization.Localization(__file__, 223, 20), fname2def_14739, *[result_mod_14742], **kwargs_14743)
        
        # Processing the call keyword arguments (line 223)
        kwargs_14745 = {}
        # Getting the type of 'priv' (line 223)
        priv_14737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'priv', False)
        # Obtaining the member 'append' of a type (line 223)
        append_14738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), priv_14737, 'append')
        # Calling append(args, kwargs) (line 223)
        append_call_result_14746 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), append_14738, *[fname2def_call_result_14744], **kwargs_14745)
        
        
        # Call to append(...): (line 224)
        # Processing the call arguments (line 224)
        str_14749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 19), 'str', 'NPY_%s')
        
        # Call to fname2def(...): (line 224)
        # Processing the call arguments (line 224)
        str_14751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 40), 'str', 'decl_%s')
        # Getting the type of 'f' (line 224)
        f_14752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 52), 'f', False)
        # Applying the binary operator '%' (line 224)
        result_mod_14753 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 40), '%', str_14751, f_14752)
        
        # Processing the call keyword arguments (line 224)
        kwargs_14754 = {}
        # Getting the type of 'fname2def' (line 224)
        fname2def_14750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 30), 'fname2def', False)
        # Calling fname2def(args, kwargs) (line 224)
        fname2def_call_result_14755 = invoke(stypy.reporting.localization.Localization(__file__, 224, 30), fname2def_14750, *[result_mod_14753], **kwargs_14754)
        
        # Applying the binary operator '%' (line 224)
        result_mod_14756 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 19), '%', str_14749, fname2def_call_result_14755)
        
        # Processing the call keyword arguments (line 224)
        kwargs_14757 = {}
        # Getting the type of 'pub' (line 224)
        pub_14747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'pub', False)
        # Obtaining the member 'append' of a type (line 224)
        append_14748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), pub_14747, 'append')
        # Calling append(args, kwargs) (line 224)
        append_call_result_14758 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), append_14748, *[result_mod_14756], **kwargs_14757)
        
        
        # ################# End of '_add_decl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_add_decl' in the type store
        # Getting the type of 'stypy_return_type' (line 222)
        stypy_return_type_14759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14759)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_add_decl'
        return stypy_return_type_14759

    # Assigning a type to the variable '_add_decl' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), '_add_decl', _add_decl)
    
    # Assigning a List to a Name (line 230):
    
    # Assigning a List to a Name (line 230):
    
    # Obtaining an instance of the builtin type 'list' (line 230)
    list_14760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 230)
    # Adding element type (line 230)
    str_14761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 15), 'str', 'isnan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 14), list_14760, str_14761)
    # Adding element type (line 230)
    str_14762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 24), 'str', 'isinf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 14), list_14760, str_14762)
    # Adding element type (line 230)
    str_14763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 33), 'str', 'signbit')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 14), list_14760, str_14763)
    # Adding element type (line 230)
    str_14764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 44), 'str', 'isfinite')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 14), list_14760, str_14764)
    
    # Assigning a type to the variable '_macros' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), '_macros', list_14760)
    
    # Getting the type of '_macros' (line 231)
    _macros_14765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 13), '_macros')
    # Testing the type of a for loop iterable (line 231)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 231, 4), _macros_14765)
    # Getting the type of the for loop variable (line 231)
    for_loop_var_14766 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 231, 4), _macros_14765)
    # Assigning a type to the variable 'f' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'f', for_loop_var_14766)
    # SSA begins for a for statement (line 231)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 232):
    
    # Assigning a Call to a Name (line 232):
    
    # Call to fname2def(...): (line 232)
    # Processing the call arguments (line 232)
    str_14768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 30), 'str', 'decl_%s')
    # Getting the type of 'f' (line 232)
    f_14769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 42), 'f', False)
    # Applying the binary operator '%' (line 232)
    result_mod_14770 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 30), '%', str_14768, f_14769)
    
    # Processing the call keyword arguments (line 232)
    kwargs_14771 = {}
    # Getting the type of 'fname2def' (line 232)
    fname2def_14767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 20), 'fname2def', False)
    # Calling fname2def(args, kwargs) (line 232)
    fname2def_call_result_14772 = invoke(stypy.reporting.localization.Localization(__file__, 232, 20), fname2def_14767, *[result_mod_14770], **kwargs_14771)
    
    # Assigning a type to the variable 'py_symbol' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'py_symbol', fname2def_call_result_14772)
    
    # Assigning a Call to a Name (line 233):
    
    # Assigning a Call to a Name (line 233):
    
    # Call to check_decl(...): (line 233)
    # Processing the call arguments (line 233)
    # Getting the type of 'py_symbol' (line 233)
    py_symbol_14775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 45), 'py_symbol', False)
    # Processing the call keyword arguments (line 233)
    
    # Obtaining an instance of the builtin type 'list' (line 234)
    list_14776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 234)
    # Adding element type (line 234)
    str_14777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 25), 'str', 'Python.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 24), list_14776, str_14777)
    # Adding element type (line 234)
    str_14778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 37), 'str', 'math.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 24), list_14776, str_14778)
    
    keyword_14779 = list_14776
    kwargs_14780 = {'headers': keyword_14779}
    # Getting the type of 'config' (line 233)
    config_14773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 27), 'config', False)
    # Obtaining the member 'check_decl' of a type (line 233)
    check_decl_14774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 27), config_14773, 'check_decl')
    # Calling check_decl(args, kwargs) (line 233)
    check_decl_call_result_14781 = invoke(stypy.reporting.localization.Localization(__file__, 233, 27), check_decl_14774, *[py_symbol_14775], **kwargs_14780)
    
    # Assigning a type to the variable 'already_declared' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'already_declared', check_decl_call_result_14781)
    
    # Getting the type of 'already_declared' (line 235)
    already_declared_14782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 11), 'already_declared')
    # Testing the type of an if condition (line 235)
    if_condition_14783 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 8), already_declared_14782)
    # Assigning a type to the variable 'if_condition_14783' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'if_condition_14783', if_condition_14783)
    # SSA begins for if statement (line 235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to check_macro_true(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'py_symbol' (line 236)
    py_symbol_14786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 39), 'py_symbol', False)
    # Processing the call keyword arguments (line 236)
    
    # Obtaining an instance of the builtin type 'list' (line 237)
    list_14787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 237)
    # Adding element type (line 237)
    str_14788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 29), 'str', 'Python.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 28), list_14787, str_14788)
    # Adding element type (line 237)
    str_14789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 41), 'str', 'math.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 28), list_14787, str_14789)
    
    keyword_14790 = list_14787
    kwargs_14791 = {'headers': keyword_14790}
    # Getting the type of 'config' (line 236)
    config_14784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 15), 'config', False)
    # Obtaining the member 'check_macro_true' of a type (line 236)
    check_macro_true_14785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 15), config_14784, 'check_macro_true')
    # Calling check_macro_true(args, kwargs) (line 236)
    check_macro_true_call_result_14792 = invoke(stypy.reporting.localization.Localization(__file__, 236, 15), check_macro_true_14785, *[py_symbol_14786], **kwargs_14791)
    
    # Testing the type of an if condition (line 236)
    if_condition_14793 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 12), check_macro_true_call_result_14792)
    # Assigning a type to the variable 'if_condition_14793' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'if_condition_14793', if_condition_14793)
    # SSA begins for if statement (line 236)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 238)
    # Processing the call arguments (line 238)
    str_14796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 27), 'str', 'NPY_%s')
    
    # Call to fname2def(...): (line 238)
    # Processing the call arguments (line 238)
    str_14798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 48), 'str', 'decl_%s')
    # Getting the type of 'f' (line 238)
    f_14799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 60), 'f', False)
    # Applying the binary operator '%' (line 238)
    result_mod_14800 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 48), '%', str_14798, f_14799)
    
    # Processing the call keyword arguments (line 238)
    kwargs_14801 = {}
    # Getting the type of 'fname2def' (line 238)
    fname2def_14797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 38), 'fname2def', False)
    # Calling fname2def(args, kwargs) (line 238)
    fname2def_call_result_14802 = invoke(stypy.reporting.localization.Localization(__file__, 238, 38), fname2def_14797, *[result_mod_14800], **kwargs_14801)
    
    # Applying the binary operator '%' (line 238)
    result_mod_14803 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 27), '%', str_14796, fname2def_call_result_14802)
    
    # Processing the call keyword arguments (line 238)
    kwargs_14804 = {}
    # Getting the type of 'pub' (line 238)
    pub_14794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'pub', False)
    # Obtaining the member 'append' of a type (line 238)
    append_14795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 16), pub_14794, 'append')
    # Calling append(args, kwargs) (line 238)
    append_call_result_14805 = invoke(stypy.reporting.localization.Localization(__file__, 238, 16), append_14795, *[result_mod_14803], **kwargs_14804)
    
    # SSA join for if statement (line 236)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 235)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'f' (line 240)
    f_14808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 26), 'f', False)
    # Processing the call keyword arguments (line 240)
    kwargs_14809 = {}
    # Getting the type of 'macros' (line 240)
    macros_14806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'macros', False)
    # Obtaining the member 'append' of a type (line 240)
    append_14807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 12), macros_14806, 'append')
    # Calling append(args, kwargs) (line 240)
    append_call_result_14810 = invoke(stypy.reporting.localization.Localization(__file__, 240, 12), append_14807, *[f_14808], **kwargs_14809)
    
    # SSA join for if statement (line 235)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'macros' (line 246)
    macros_14811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 13), 'macros')
    # Testing the type of a for loop iterable (line 246)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 246, 4), macros_14811)
    # Getting the type of the for loop variable (line 246)
    for_loop_var_14812 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 246, 4), macros_14811)
    # Assigning a type to the variable 'f' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'f', for_loop_var_14812)
    # SSA begins for a for statement (line 246)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 247):
    
    # Assigning a Call to a Name (line 247):
    
    # Call to check_decl(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'f' (line 247)
    f_14815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 31), 'f', False)
    # Processing the call keyword arguments (line 247)
    
    # Obtaining an instance of the builtin type 'list' (line 247)
    list_14816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 42), 'list')
    # Adding type elements to the builtin type 'list' instance (line 247)
    # Adding element type (line 247)
    str_14817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 43), 'str', 'Python.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 42), list_14816, str_14817)
    # Adding element type (line 247)
    str_14818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 55), 'str', 'math.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 42), list_14816, str_14818)
    
    keyword_14819 = list_14816
    kwargs_14820 = {'headers': keyword_14819}
    # Getting the type of 'config' (line 247)
    config_14813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 13), 'config', False)
    # Obtaining the member 'check_decl' of a type (line 247)
    check_decl_14814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 13), config_14813, 'check_decl')
    # Calling check_decl(args, kwargs) (line 247)
    check_decl_call_result_14821 = invoke(stypy.reporting.localization.Localization(__file__, 247, 13), check_decl_14814, *[f_14815], **kwargs_14820)
    
    # Assigning a type to the variable 'st' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'st', check_decl_call_result_14821)
    
    # Getting the type of 'st' (line 248)
    st_14822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 11), 'st')
    # Testing the type of an if condition (line 248)
    if_condition_14823 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 248, 8), st_14822)
    # Assigning a type to the variable 'if_condition_14823' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'if_condition_14823', if_condition_14823)
    # SSA begins for if statement (line 248)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _add_decl(...): (line 249)
    # Processing the call arguments (line 249)
    # Getting the type of 'f' (line 249)
    f_14825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 22), 'f', False)
    # Processing the call keyword arguments (line 249)
    kwargs_14826 = {}
    # Getting the type of '_add_decl' (line 249)
    _add_decl_14824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), '_add_decl', False)
    # Calling _add_decl(args, kwargs) (line 249)
    _add_decl_call_result_14827 = invoke(stypy.reporting.localization.Localization(__file__, 249, 12), _add_decl_14824, *[f_14825], **kwargs_14826)
    
    # SSA join for if statement (line 248)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 251)
    tuple_14828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 251)
    # Adding element type (line 251)
    # Getting the type of 'priv' (line 251)
    priv_14829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 11), 'priv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 11), tuple_14828, priv_14829)
    # Adding element type (line 251)
    # Getting the type of 'pub' (line 251)
    pub_14830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 17), 'pub')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 11), tuple_14828, pub_14830)
    
    # Assigning a type to the variable 'stypy_return_type' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'stypy_return_type', tuple_14828)
    
    # ################# End of 'check_ieee_macros(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_ieee_macros' in the type store
    # Getting the type of 'stypy_return_type' (line 216)
    stypy_return_type_14831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14831)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_ieee_macros'
    return stypy_return_type_14831

# Assigning a type to the variable 'check_ieee_macros' (line 216)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 0), 'check_ieee_macros', check_ieee_macros)

@norecursion
def check_types(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_types'
    module_type_store = module_type_store.open_function_context('check_types', 253, 0, False)
    
    # Passed parameters checking function
    check_types.stypy_localization = localization
    check_types.stypy_type_of_self = None
    check_types.stypy_type_store = module_type_store
    check_types.stypy_function_name = 'check_types'
    check_types.stypy_param_names_list = ['config_cmd', 'ext', 'build_dir']
    check_types.stypy_varargs_param_name = None
    check_types.stypy_kwargs_param_name = None
    check_types.stypy_call_defaults = defaults
    check_types.stypy_call_varargs = varargs
    check_types.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_types', ['config_cmd', 'ext', 'build_dir'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_types', localization, ['config_cmd', 'ext', 'build_dir'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_types(...)' code ##################

    
    # Assigning a List to a Name (line 254):
    
    # Assigning a List to a Name (line 254):
    
    # Obtaining an instance of the builtin type 'list' (line 254)
    list_14832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 254)
    
    # Assigning a type to the variable 'private_defines' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'private_defines', list_14832)
    
    # Assigning a List to a Name (line 255):
    
    # Assigning a List to a Name (line 255):
    
    # Obtaining an instance of the builtin type 'list' (line 255)
    list_14833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 255)
    
    # Assigning a type to the variable 'public_defines' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'public_defines', list_14833)
    
    # Assigning a Dict to a Name (line 260):
    
    # Assigning a Dict to a Name (line 260):
    
    # Obtaining an instance of the builtin type 'dict' (line 260)
    dict_14834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 260)
    # Adding element type (key, value) (line 260)
    str_14835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 16), 'str', 'short')
    
    # Obtaining an instance of the builtin type 'list' (line 260)
    list_14836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 260)
    # Adding element type (line 260)
    int_14837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 25), list_14836, int_14837)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 15), dict_14834, (str_14835, list_14836))
    # Adding element type (key, value) (line 260)
    str_14838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 30), 'str', 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 260)
    list_14839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 260)
    # Adding element type (line 260)
    int_14840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 37), list_14839, int_14840)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 15), dict_14834, (str_14838, list_14839))
    # Adding element type (key, value) (line 260)
    str_14841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 42), 'str', 'long')
    
    # Obtaining an instance of the builtin type 'list' (line 260)
    list_14842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 50), 'list')
    # Adding type elements to the builtin type 'list' instance (line 260)
    # Adding element type (line 260)
    int_14843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 50), list_14842, int_14843)
    # Adding element type (line 260)
    int_14844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 50), list_14842, int_14844)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 15), dict_14834, (str_14841, list_14842))
    # Adding element type (key, value) (line 260)
    str_14845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 16), 'str', 'float')
    
    # Obtaining an instance of the builtin type 'list' (line 261)
    list_14846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 261)
    # Adding element type (line 261)
    int_14847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 25), list_14846, int_14847)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 15), dict_14834, (str_14845, list_14846))
    # Adding element type (key, value) (line 260)
    str_14848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 30), 'str', 'double')
    
    # Obtaining an instance of the builtin type 'list' (line 261)
    list_14849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 261)
    # Adding element type (line 261)
    int_14850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 40), list_14849, int_14850)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 15), dict_14834, (str_14848, list_14849))
    # Adding element type (key, value) (line 260)
    str_14851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 45), 'str', 'long double')
    
    # Obtaining an instance of the builtin type 'list' (line 261)
    list_14852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 60), 'list')
    # Adding type elements to the builtin type 'list' instance (line 261)
    # Adding element type (line 261)
    int_14853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 61), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 60), list_14852, int_14853)
    # Adding element type (line 261)
    int_14854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 65), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 60), list_14852, int_14854)
    # Adding element type (line 261)
    int_14855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 69), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 60), list_14852, int_14855)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 15), dict_14834, (str_14851, list_14852))
    # Adding element type (key, value) (line 260)
    str_14856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 16), 'str', 'Py_intptr_t')
    
    # Obtaining an instance of the builtin type 'list' (line 262)
    list_14857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 262)
    # Adding element type (line 262)
    int_14858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 31), list_14857, int_14858)
    # Adding element type (line 262)
    int_14859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 31), list_14857, int_14859)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 15), dict_14834, (str_14856, list_14857))
    # Adding element type (key, value) (line 260)
    str_14860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 39), 'str', 'PY_LONG_LONG')
    
    # Obtaining an instance of the builtin type 'list' (line 262)
    list_14861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 55), 'list')
    # Adding type elements to the builtin type 'list' instance (line 262)
    # Adding element type (line 262)
    int_14862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 56), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 55), list_14861, int_14862)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 15), dict_14834, (str_14860, list_14861))
    # Adding element type (key, value) (line 260)
    str_14863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 60), 'str', 'long long')
    
    # Obtaining an instance of the builtin type 'list' (line 262)
    list_14864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 73), 'list')
    # Adding type elements to the builtin type 'list' instance (line 262)
    # Adding element type (line 262)
    int_14865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 74), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 73), list_14864, int_14865)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 15), dict_14834, (str_14863, list_14864))
    # Adding element type (key, value) (line 260)
    str_14866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 16), 'str', 'off_t')
    
    # Obtaining an instance of the builtin type 'list' (line 263)
    list_14867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 263)
    # Adding element type (line 263)
    int_14868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 25), list_14867, int_14868)
    # Adding element type (line 263)
    int_14869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 25), list_14867, int_14869)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 15), dict_14834, (str_14866, list_14867))
    
    # Assigning a type to the variable 'expected' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'expected', dict_14834)
    
    # Assigning a Call to a Name (line 266):
    
    # Assigning a Call to a Name (line 266):
    
    # Call to check_header(...): (line 266)
    # Processing the call arguments (line 266)
    str_14872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 37), 'str', 'Python.h')
    # Processing the call keyword arguments (line 266)
    kwargs_14873 = {}
    # Getting the type of 'config_cmd' (line 266)
    config_cmd_14870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 13), 'config_cmd', False)
    # Obtaining the member 'check_header' of a type (line 266)
    check_header_14871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 13), config_cmd_14870, 'check_header')
    # Calling check_header(args, kwargs) (line 266)
    check_header_call_result_14874 = invoke(stypy.reporting.localization.Localization(__file__, 266, 13), check_header_14871, *[str_14872], **kwargs_14873)
    
    # Assigning a type to the variable 'result' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'result', check_header_call_result_14874)
    
    
    # Getting the type of 'result' (line 267)
    result_14875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 11), 'result')
    # Applying the 'not' unary operator (line 267)
    result_not__14876 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 7), 'not', result_14875)
    
    # Testing the type of an if condition (line 267)
    if_condition_14877 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 267, 4), result_not__14876)
    # Assigning a type to the variable 'if_condition_14877' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'if_condition_14877', if_condition_14877)
    # SSA begins for if statement (line 267)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to SystemError(...): (line 268)
    # Processing the call arguments (line 268)
    str_14879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 16), 'str', "Cannot compile 'Python.h'. Perhaps you need to install python-dev|python-devel.")
    # Processing the call keyword arguments (line 268)
    kwargs_14880 = {}
    # Getting the type of 'SystemError' (line 268)
    SystemError_14878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 14), 'SystemError', False)
    # Calling SystemError(args, kwargs) (line 268)
    SystemError_call_result_14881 = invoke(stypy.reporting.localization.Localization(__file__, 268, 14), SystemError_14878, *[str_14879], **kwargs_14880)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 268, 8), SystemError_call_result_14881, 'raise parameter', BaseException)
    # SSA join for if statement (line 267)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 271):
    
    # Assigning a Call to a Name (line 271):
    
    # Call to check_header(...): (line 271)
    # Processing the call arguments (line 271)
    str_14884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 34), 'str', 'endian.h')
    # Processing the call keyword arguments (line 271)
    kwargs_14885 = {}
    # Getting the type of 'config_cmd' (line 271)
    config_cmd_14882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 10), 'config_cmd', False)
    # Obtaining the member 'check_header' of a type (line 271)
    check_header_14883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 10), config_cmd_14882, 'check_header')
    # Calling check_header(args, kwargs) (line 271)
    check_header_call_result_14886 = invoke(stypy.reporting.localization.Localization(__file__, 271, 10), check_header_14883, *[str_14884], **kwargs_14885)
    
    # Assigning a type to the variable 'res' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'res', check_header_call_result_14886)
    
    # Getting the type of 'res' (line 272)
    res_14887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 7), 'res')
    # Testing the type of an if condition (line 272)
    if_condition_14888 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 4), res_14887)
    # Assigning a type to the variable 'if_condition_14888' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'if_condition_14888', if_condition_14888)
    # SSA begins for if statement (line 272)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 273)
    # Processing the call arguments (line 273)
    
    # Obtaining an instance of the builtin type 'tuple' (line 273)
    tuple_14891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 273)
    # Adding element type (line 273)
    str_14892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 32), 'str', 'HAVE_ENDIAN_H')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 32), tuple_14891, str_14892)
    # Adding element type (line 273)
    int_14893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 32), tuple_14891, int_14893)
    
    # Processing the call keyword arguments (line 273)
    kwargs_14894 = {}
    # Getting the type of 'private_defines' (line 273)
    private_defines_14889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'private_defines', False)
    # Obtaining the member 'append' of a type (line 273)
    append_14890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), private_defines_14889, 'append')
    # Calling append(args, kwargs) (line 273)
    append_call_result_14895 = invoke(stypy.reporting.localization.Localization(__file__, 273, 8), append_14890, *[tuple_14891], **kwargs_14894)
    
    
    # Call to append(...): (line 274)
    # Processing the call arguments (line 274)
    
    # Obtaining an instance of the builtin type 'tuple' (line 274)
    tuple_14898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 274)
    # Adding element type (line 274)
    str_14899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 31), 'str', 'NPY_HAVE_ENDIAN_H')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 31), tuple_14898, str_14899)
    # Adding element type (line 274)
    int_14900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 31), tuple_14898, int_14900)
    
    # Processing the call keyword arguments (line 274)
    kwargs_14901 = {}
    # Getting the type of 'public_defines' (line 274)
    public_defines_14896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'public_defines', False)
    # Obtaining the member 'append' of a type (line 274)
    append_14897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), public_defines_14896, 'append')
    # Calling append(args, kwargs) (line 274)
    append_call_result_14902 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), append_14897, *[tuple_14898], **kwargs_14901)
    
    # SSA join for if statement (line 272)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 277)
    tuple_14903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 277)
    # Adding element type (line 277)
    str_14904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 17), 'str', 'short')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 17), tuple_14903, str_14904)
    # Adding element type (line 277)
    str_14905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 26), 'str', 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 17), tuple_14903, str_14905)
    # Adding element type (line 277)
    str_14906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 33), 'str', 'long')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 17), tuple_14903, str_14906)
    
    # Testing the type of a for loop iterable (line 277)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 277, 4), tuple_14903)
    # Getting the type of the for loop variable (line 277)
    for_loop_var_14907 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 277, 4), tuple_14903)
    # Assigning a type to the variable 'type' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'type', for_loop_var_14907)
    # SSA begins for a for statement (line 277)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 278):
    
    # Assigning a Call to a Name (line 278):
    
    # Call to check_decl(...): (line 278)
    # Processing the call arguments (line 278)
    str_14910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 36), 'str', 'SIZEOF_%s')
    
    # Call to sym2def(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'type' (line 278)
    type_14912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 58), 'type', False)
    # Processing the call keyword arguments (line 278)
    kwargs_14913 = {}
    # Getting the type of 'sym2def' (line 278)
    sym2def_14911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 50), 'sym2def', False)
    # Calling sym2def(args, kwargs) (line 278)
    sym2def_call_result_14914 = invoke(stypy.reporting.localization.Localization(__file__, 278, 50), sym2def_14911, *[type_14912], **kwargs_14913)
    
    # Applying the binary operator '%' (line 278)
    result_mod_14915 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 36), '%', str_14910, sym2def_call_result_14914)
    
    # Processing the call keyword arguments (line 278)
    
    # Obtaining an instance of the builtin type 'list' (line 278)
    list_14916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 73), 'list')
    # Adding type elements to the builtin type 'list' instance (line 278)
    # Adding element type (line 278)
    str_14917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 74), 'str', 'Python.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 73), list_14916, str_14917)
    
    keyword_14918 = list_14916
    kwargs_14919 = {'headers': keyword_14918}
    # Getting the type of 'config_cmd' (line 278)
    config_cmd_14908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 14), 'config_cmd', False)
    # Obtaining the member 'check_decl' of a type (line 278)
    check_decl_14909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 14), config_cmd_14908, 'check_decl')
    # Calling check_decl(args, kwargs) (line 278)
    check_decl_call_result_14920 = invoke(stypy.reporting.localization.Localization(__file__, 278, 14), check_decl_14909, *[result_mod_14915], **kwargs_14919)
    
    # Assigning a type to the variable 'res' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'res', check_decl_call_result_14920)
    
    # Getting the type of 'res' (line 279)
    res_14921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 11), 'res')
    # Testing the type of an if condition (line 279)
    if_condition_14922 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 8), res_14921)
    # Assigning a type to the variable 'if_condition_14922' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'if_condition_14922', if_condition_14922)
    # SSA begins for if statement (line 279)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 280)
    # Processing the call arguments (line 280)
    
    # Obtaining an instance of the builtin type 'tuple' (line 280)
    tuple_14925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 280)
    # Adding element type (line 280)
    str_14926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 35), 'str', 'NPY_SIZEOF_%s')
    
    # Call to sym2def(...): (line 280)
    # Processing the call arguments (line 280)
    # Getting the type of 'type' (line 280)
    type_14928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 61), 'type', False)
    # Processing the call keyword arguments (line 280)
    kwargs_14929 = {}
    # Getting the type of 'sym2def' (line 280)
    sym2def_14927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 53), 'sym2def', False)
    # Calling sym2def(args, kwargs) (line 280)
    sym2def_call_result_14930 = invoke(stypy.reporting.localization.Localization(__file__, 280, 53), sym2def_14927, *[type_14928], **kwargs_14929)
    
    # Applying the binary operator '%' (line 280)
    result_mod_14931 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 35), '%', str_14926, sym2def_call_result_14930)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 35), tuple_14925, result_mod_14931)
    # Adding element type (line 280)
    str_14932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 68), 'str', 'SIZEOF_%s')
    
    # Call to sym2def(...): (line 280)
    # Processing the call arguments (line 280)
    # Getting the type of 'type' (line 280)
    type_14934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 90), 'type', False)
    # Processing the call keyword arguments (line 280)
    kwargs_14935 = {}
    # Getting the type of 'sym2def' (line 280)
    sym2def_14933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 82), 'sym2def', False)
    # Calling sym2def(args, kwargs) (line 280)
    sym2def_call_result_14936 = invoke(stypy.reporting.localization.Localization(__file__, 280, 82), sym2def_14933, *[type_14934], **kwargs_14935)
    
    # Applying the binary operator '%' (line 280)
    result_mod_14937 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 68), '%', str_14932, sym2def_call_result_14936)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 35), tuple_14925, result_mod_14937)
    
    # Processing the call keyword arguments (line 280)
    kwargs_14938 = {}
    # Getting the type of 'public_defines' (line 280)
    public_defines_14923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'public_defines', False)
    # Obtaining the member 'append' of a type (line 280)
    append_14924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 12), public_defines_14923, 'append')
    # Calling append(args, kwargs) (line 280)
    append_call_result_14939 = invoke(stypy.reporting.localization.Localization(__file__, 280, 12), append_14924, *[tuple_14925], **kwargs_14938)
    
    # SSA branch for the else part of an if statement (line 279)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 282):
    
    # Assigning a Call to a Name (line 282):
    
    # Call to check_type_size(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'type' (line 282)
    type_14942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 45), 'type', False)
    # Processing the call keyword arguments (line 282)
    
    # Obtaining the type of the subscript
    # Getting the type of 'type' (line 282)
    type_14943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 69), 'type', False)
    # Getting the type of 'expected' (line 282)
    expected_14944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 60), 'expected', False)
    # Obtaining the member '__getitem__' of a type (line 282)
    getitem___14945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 60), expected_14944, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 282)
    subscript_call_result_14946 = invoke(stypy.reporting.localization.Localization(__file__, 282, 60), getitem___14945, type_14943)
    
    keyword_14947 = subscript_call_result_14946
    kwargs_14948 = {'expected': keyword_14947}
    # Getting the type of 'config_cmd' (line 282)
    config_cmd_14940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 18), 'config_cmd', False)
    # Obtaining the member 'check_type_size' of a type (line 282)
    check_type_size_14941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 18), config_cmd_14940, 'check_type_size')
    # Calling check_type_size(args, kwargs) (line 282)
    check_type_size_call_result_14949 = invoke(stypy.reporting.localization.Localization(__file__, 282, 18), check_type_size_14941, *[type_14942], **kwargs_14948)
    
    # Assigning a type to the variable 'res' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'res', check_type_size_call_result_14949)
    
    
    # Getting the type of 'res' (line 283)
    res_14950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 15), 'res')
    int_14951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 22), 'int')
    # Applying the binary operator '>=' (line 283)
    result_ge_14952 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 15), '>=', res_14950, int_14951)
    
    # Testing the type of an if condition (line 283)
    if_condition_14953 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 283, 12), result_ge_14952)
    # Assigning a type to the variable 'if_condition_14953' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'if_condition_14953', if_condition_14953)
    # SSA begins for if statement (line 283)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 284)
    # Processing the call arguments (line 284)
    
    # Obtaining an instance of the builtin type 'tuple' (line 284)
    tuple_14956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 284)
    # Adding element type (line 284)
    str_14957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 39), 'str', 'NPY_SIZEOF_%s')
    
    # Call to sym2def(...): (line 284)
    # Processing the call arguments (line 284)
    # Getting the type of 'type' (line 284)
    type_14959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 65), 'type', False)
    # Processing the call keyword arguments (line 284)
    kwargs_14960 = {}
    # Getting the type of 'sym2def' (line 284)
    sym2def_14958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 57), 'sym2def', False)
    # Calling sym2def(args, kwargs) (line 284)
    sym2def_call_result_14961 = invoke(stypy.reporting.localization.Localization(__file__, 284, 57), sym2def_14958, *[type_14959], **kwargs_14960)
    
    # Applying the binary operator '%' (line 284)
    result_mod_14962 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 39), '%', str_14957, sym2def_call_result_14961)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 39), tuple_14956, result_mod_14962)
    # Adding element type (line 284)
    str_14963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 72), 'str', '%d')
    # Getting the type of 'res' (line 284)
    res_14964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 79), 'res', False)
    # Applying the binary operator '%' (line 284)
    result_mod_14965 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 72), '%', str_14963, res_14964)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 39), tuple_14956, result_mod_14965)
    
    # Processing the call keyword arguments (line 284)
    kwargs_14966 = {}
    # Getting the type of 'public_defines' (line 284)
    public_defines_14954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'public_defines', False)
    # Obtaining the member 'append' of a type (line 284)
    append_14955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 16), public_defines_14954, 'append')
    # Calling append(args, kwargs) (line 284)
    append_call_result_14967 = invoke(stypy.reporting.localization.Localization(__file__, 284, 16), append_14955, *[tuple_14956], **kwargs_14966)
    
    # SSA branch for the else part of an if statement (line 283)
    module_type_store.open_ssa_branch('else')
    
    # Call to SystemError(...): (line 286)
    # Processing the call arguments (line 286)
    str_14969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 34), 'str', 'Checking sizeof (%s) failed !')
    # Getting the type of 'type' (line 286)
    type_14970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 68), 'type', False)
    # Applying the binary operator '%' (line 286)
    result_mod_14971 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 34), '%', str_14969, type_14970)
    
    # Processing the call keyword arguments (line 286)
    kwargs_14972 = {}
    # Getting the type of 'SystemError' (line 286)
    SystemError_14968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 22), 'SystemError', False)
    # Calling SystemError(args, kwargs) (line 286)
    SystemError_call_result_14973 = invoke(stypy.reporting.localization.Localization(__file__, 286, 22), SystemError_14968, *[result_mod_14971], **kwargs_14972)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 286, 16), SystemError_call_result_14973, 'raise parameter', BaseException)
    # SSA join for if statement (line 283)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 279)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 288)
    tuple_14974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 288)
    # Adding element type (line 288)
    str_14975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 17), 'str', 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 17), tuple_14974, str_14975)
    # Adding element type (line 288)
    str_14976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 26), 'str', 'double')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 17), tuple_14974, str_14976)
    # Adding element type (line 288)
    str_14977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 36), 'str', 'long double')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 17), tuple_14974, str_14977)
    
    # Testing the type of a for loop iterable (line 288)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 288, 4), tuple_14974)
    # Getting the type of the for loop variable (line 288)
    for_loop_var_14978 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 288, 4), tuple_14974)
    # Assigning a type to the variable 'type' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'type', for_loop_var_14978)
    # SSA begins for a for statement (line 288)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 289):
    
    # Assigning a Call to a Name (line 289):
    
    # Call to check_decl(...): (line 289)
    # Processing the call arguments (line 289)
    str_14981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 49), 'str', 'SIZEOF_%s')
    
    # Call to sym2def(...): (line 289)
    # Processing the call arguments (line 289)
    # Getting the type of 'type' (line 289)
    type_14983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 71), 'type', False)
    # Processing the call keyword arguments (line 289)
    kwargs_14984 = {}
    # Getting the type of 'sym2def' (line 289)
    sym2def_14982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 63), 'sym2def', False)
    # Calling sym2def(args, kwargs) (line 289)
    sym2def_call_result_14985 = invoke(stypy.reporting.localization.Localization(__file__, 289, 63), sym2def_14982, *[type_14983], **kwargs_14984)
    
    # Applying the binary operator '%' (line 289)
    result_mod_14986 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 49), '%', str_14981, sym2def_call_result_14985)
    
    # Processing the call keyword arguments (line 289)
    
    # Obtaining an instance of the builtin type 'list' (line 290)
    list_14987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 57), 'list')
    # Adding type elements to the builtin type 'list' instance (line 290)
    # Adding element type (line 290)
    str_14988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 58), 'str', 'Python.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 57), list_14987, str_14988)
    
    keyword_14989 = list_14987
    kwargs_14990 = {'headers': keyword_14989}
    # Getting the type of 'config_cmd' (line 289)
    config_cmd_14979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 27), 'config_cmd', False)
    # Obtaining the member 'check_decl' of a type (line 289)
    check_decl_14980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 27), config_cmd_14979, 'check_decl')
    # Calling check_decl(args, kwargs) (line 289)
    check_decl_call_result_14991 = invoke(stypy.reporting.localization.Localization(__file__, 289, 27), check_decl_14980, *[result_mod_14986], **kwargs_14990)
    
    # Assigning a type to the variable 'already_declared' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'already_declared', check_decl_call_result_14991)
    
    # Assigning a Call to a Name (line 291):
    
    # Assigning a Call to a Name (line 291):
    
    # Call to check_type_size(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'type' (line 291)
    type_14994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 41), 'type', False)
    # Processing the call keyword arguments (line 291)
    
    # Obtaining the type of the subscript
    # Getting the type of 'type' (line 291)
    type_14995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 65), 'type', False)
    # Getting the type of 'expected' (line 291)
    expected_14996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 56), 'expected', False)
    # Obtaining the member '__getitem__' of a type (line 291)
    getitem___14997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 56), expected_14996, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 291)
    subscript_call_result_14998 = invoke(stypy.reporting.localization.Localization(__file__, 291, 56), getitem___14997, type_14995)
    
    keyword_14999 = subscript_call_result_14998
    kwargs_15000 = {'expected': keyword_14999}
    # Getting the type of 'config_cmd' (line 291)
    config_cmd_14992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 14), 'config_cmd', False)
    # Obtaining the member 'check_type_size' of a type (line 291)
    check_type_size_14993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 14), config_cmd_14992, 'check_type_size')
    # Calling check_type_size(args, kwargs) (line 291)
    check_type_size_call_result_15001 = invoke(stypy.reporting.localization.Localization(__file__, 291, 14), check_type_size_14993, *[type_14994], **kwargs_15000)
    
    # Assigning a type to the variable 'res' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'res', check_type_size_call_result_15001)
    
    
    # Getting the type of 'res' (line 292)
    res_15002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 11), 'res')
    int_15003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 18), 'int')
    # Applying the binary operator '>=' (line 292)
    result_ge_15004 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 11), '>=', res_15002, int_15003)
    
    # Testing the type of an if condition (line 292)
    if_condition_15005 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 8), result_ge_15004)
    # Assigning a type to the variable 'if_condition_15005' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'if_condition_15005', if_condition_15005)
    # SSA begins for if statement (line 292)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 293)
    # Processing the call arguments (line 293)
    
    # Obtaining an instance of the builtin type 'tuple' (line 293)
    tuple_15008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 293)
    # Adding element type (line 293)
    str_15009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 35), 'str', 'NPY_SIZEOF_%s')
    
    # Call to sym2def(...): (line 293)
    # Processing the call arguments (line 293)
    # Getting the type of 'type' (line 293)
    type_15011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 61), 'type', False)
    # Processing the call keyword arguments (line 293)
    kwargs_15012 = {}
    # Getting the type of 'sym2def' (line 293)
    sym2def_15010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 53), 'sym2def', False)
    # Calling sym2def(args, kwargs) (line 293)
    sym2def_call_result_15013 = invoke(stypy.reporting.localization.Localization(__file__, 293, 53), sym2def_15010, *[type_15011], **kwargs_15012)
    
    # Applying the binary operator '%' (line 293)
    result_mod_15014 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 35), '%', str_15009, sym2def_call_result_15013)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 35), tuple_15008, result_mod_15014)
    # Adding element type (line 293)
    str_15015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 68), 'str', '%d')
    # Getting the type of 'res' (line 293)
    res_15016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 75), 'res', False)
    # Applying the binary operator '%' (line 293)
    result_mod_15017 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 68), '%', str_15015, res_15016)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 35), tuple_15008, result_mod_15017)
    
    # Processing the call keyword arguments (line 293)
    kwargs_15018 = {}
    # Getting the type of 'public_defines' (line 293)
    public_defines_15006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'public_defines', False)
    # Obtaining the member 'append' of a type (line 293)
    append_15007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 12), public_defines_15006, 'append')
    # Calling append(args, kwargs) (line 293)
    append_call_result_15019 = invoke(stypy.reporting.localization.Localization(__file__, 293, 12), append_15007, *[tuple_15008], **kwargs_15018)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'already_declared' (line 294)
    already_declared_15020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 19), 'already_declared')
    # Applying the 'not' unary operator (line 294)
    result_not__15021 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 15), 'not', already_declared_15020)
    
    
    
    # Getting the type of 'type' (line 294)
    type_15022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 44), 'type')
    str_15023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 52), 'str', 'long double')
    # Applying the binary operator '==' (line 294)
    result_eq_15024 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 44), '==', type_15022, str_15023)
    
    # Applying the 'not' unary operator (line 294)
    result_not__15025 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 40), 'not', result_eq_15024)
    
    # Applying the binary operator 'and' (line 294)
    result_and_keyword_15026 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 15), 'and', result_not__15021, result_not__15025)
    
    # Testing the type of an if condition (line 294)
    if_condition_15027 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 12), result_and_keyword_15026)
    # Assigning a type to the variable 'if_condition_15027' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'if_condition_15027', if_condition_15027)
    # SSA begins for if statement (line 294)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 295)
    # Processing the call arguments (line 295)
    
    # Obtaining an instance of the builtin type 'tuple' (line 295)
    tuple_15030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 295)
    # Adding element type (line 295)
    str_15031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 40), 'str', 'SIZEOF_%s')
    
    # Call to sym2def(...): (line 295)
    # Processing the call arguments (line 295)
    # Getting the type of 'type' (line 295)
    type_15033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 62), 'type', False)
    # Processing the call keyword arguments (line 295)
    kwargs_15034 = {}
    # Getting the type of 'sym2def' (line 295)
    sym2def_15032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 54), 'sym2def', False)
    # Calling sym2def(args, kwargs) (line 295)
    sym2def_call_result_15035 = invoke(stypy.reporting.localization.Localization(__file__, 295, 54), sym2def_15032, *[type_15033], **kwargs_15034)
    
    # Applying the binary operator '%' (line 295)
    result_mod_15036 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 40), '%', str_15031, sym2def_call_result_15035)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 40), tuple_15030, result_mod_15036)
    # Adding element type (line 295)
    str_15037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 69), 'str', '%d')
    # Getting the type of 'res' (line 295)
    res_15038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 76), 'res', False)
    # Applying the binary operator '%' (line 295)
    result_mod_15039 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 69), '%', str_15037, res_15038)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 40), tuple_15030, result_mod_15039)
    
    # Processing the call keyword arguments (line 295)
    kwargs_15040 = {}
    # Getting the type of 'private_defines' (line 295)
    private_defines_15028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'private_defines', False)
    # Obtaining the member 'append' of a type (line 295)
    append_15029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 16), private_defines_15028, 'append')
    # Calling append(args, kwargs) (line 295)
    append_call_result_15041 = invoke(stypy.reporting.localization.Localization(__file__, 295, 16), append_15029, *[tuple_15030], **kwargs_15040)
    
    # SSA join for if statement (line 294)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 292)
    module_type_store.open_ssa_branch('else')
    
    # Call to SystemError(...): (line 297)
    # Processing the call arguments (line 297)
    str_15043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 30), 'str', 'Checking sizeof (%s) failed !')
    # Getting the type of 'type' (line 297)
    type_15044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 64), 'type', False)
    # Applying the binary operator '%' (line 297)
    result_mod_15045 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 30), '%', str_15043, type_15044)
    
    # Processing the call keyword arguments (line 297)
    kwargs_15046 = {}
    # Getting the type of 'SystemError' (line 297)
    SystemError_15042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 18), 'SystemError', False)
    # Calling SystemError(args, kwargs) (line 297)
    SystemError_call_result_15047 = invoke(stypy.reporting.localization.Localization(__file__, 297, 18), SystemError_15042, *[result_mod_15045], **kwargs_15046)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 297, 12), SystemError_call_result_15047, 'raise parameter', BaseException)
    # SSA join for if statement (line 292)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 302):
    
    # Assigning a BinOp to a Name (line 302):
    str_15048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 22), 'str', 'struct {%s __x; %s __y;}')
    
    # Obtaining an instance of the builtin type 'tuple' (line 302)
    tuple_15049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 52), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 302)
    # Adding element type (line 302)
    # Getting the type of 'type' (line 302)
    type_15050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 52), 'type')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 52), tuple_15049, type_15050)
    # Adding element type (line 302)
    # Getting the type of 'type' (line 302)
    type_15051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 58), 'type')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 52), tuple_15049, type_15051)
    
    # Applying the binary operator '%' (line 302)
    result_mod_15052 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 22), '%', str_15048, tuple_15049)
    
    # Assigning a type to the variable 'complex_def' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'complex_def', result_mod_15052)
    
    # Assigning a Call to a Name (line 303):
    
    # Assigning a Call to a Name (line 303):
    
    # Call to check_type_size(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'complex_def' (line 303)
    complex_def_15055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 41), 'complex_def', False)
    # Processing the call keyword arguments (line 303)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    # Getting the type of 'type' (line 304)
    type_15059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 75), 'type', False)
    # Getting the type of 'expected' (line 304)
    expected_15060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 66), 'expected', False)
    # Obtaining the member '__getitem__' of a type (line 304)
    getitem___15061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 66), expected_15060, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 304)
    subscript_call_result_15062 = invoke(stypy.reporting.localization.Localization(__file__, 304, 66), getitem___15061, type_15059)
    
    comprehension_15063 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 51), subscript_call_result_15062)
    # Assigning a type to the variable 'x' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 51), 'x', comprehension_15063)
    int_15056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 51), 'int')
    # Getting the type of 'x' (line 304)
    x_15057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 55), 'x', False)
    # Applying the binary operator '*' (line 304)
    result_mul_15058 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 51), '*', int_15056, x_15057)
    
    list_15064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 51), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 51), list_15064, result_mul_15058)
    keyword_15065 = list_15064
    kwargs_15066 = {'expected': keyword_15065}
    # Getting the type of 'config_cmd' (line 303)
    config_cmd_15053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 14), 'config_cmd', False)
    # Obtaining the member 'check_type_size' of a type (line 303)
    check_type_size_15054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 14), config_cmd_15053, 'check_type_size')
    # Calling check_type_size(args, kwargs) (line 303)
    check_type_size_call_result_15067 = invoke(stypy.reporting.localization.Localization(__file__, 303, 14), check_type_size_15054, *[complex_def_15055], **kwargs_15066)
    
    # Assigning a type to the variable 'res' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'res', check_type_size_call_result_15067)
    
    
    # Getting the type of 'res' (line 305)
    res_15068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 11), 'res')
    int_15069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 18), 'int')
    # Applying the binary operator '>=' (line 305)
    result_ge_15070 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 11), '>=', res_15068, int_15069)
    
    # Testing the type of an if condition (line 305)
    if_condition_15071 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 8), result_ge_15070)
    # Assigning a type to the variable 'if_condition_15071' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'if_condition_15071', if_condition_15071)
    # SSA begins for if statement (line 305)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 306)
    # Processing the call arguments (line 306)
    
    # Obtaining an instance of the builtin type 'tuple' (line 306)
    tuple_15074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 306)
    # Adding element type (line 306)
    str_15075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 35), 'str', 'NPY_SIZEOF_COMPLEX_%s')
    
    # Call to sym2def(...): (line 306)
    # Processing the call arguments (line 306)
    # Getting the type of 'type' (line 306)
    type_15077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 69), 'type', False)
    # Processing the call keyword arguments (line 306)
    kwargs_15078 = {}
    # Getting the type of 'sym2def' (line 306)
    sym2def_15076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 61), 'sym2def', False)
    # Calling sym2def(args, kwargs) (line 306)
    sym2def_call_result_15079 = invoke(stypy.reporting.localization.Localization(__file__, 306, 61), sym2def_15076, *[type_15077], **kwargs_15078)
    
    # Applying the binary operator '%' (line 306)
    result_mod_15080 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 35), '%', str_15075, sym2def_call_result_15079)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 35), tuple_15074, result_mod_15080)
    # Adding element type (line 306)
    str_15081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 76), 'str', '%d')
    # Getting the type of 'res' (line 306)
    res_15082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 83), 'res', False)
    # Applying the binary operator '%' (line 306)
    result_mod_15083 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 76), '%', str_15081, res_15082)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 35), tuple_15074, result_mod_15083)
    
    # Processing the call keyword arguments (line 306)
    kwargs_15084 = {}
    # Getting the type of 'public_defines' (line 306)
    public_defines_15072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'public_defines', False)
    # Obtaining the member 'append' of a type (line 306)
    append_15073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 12), public_defines_15072, 'append')
    # Calling append(args, kwargs) (line 306)
    append_call_result_15085 = invoke(stypy.reporting.localization.Localization(__file__, 306, 12), append_15073, *[tuple_15074], **kwargs_15084)
    
    # SSA branch for the else part of an if statement (line 305)
    module_type_store.open_ssa_branch('else')
    
    # Call to SystemError(...): (line 308)
    # Processing the call arguments (line 308)
    str_15087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 30), 'str', 'Checking sizeof (%s) failed !')
    # Getting the type of 'complex_def' (line 308)
    complex_def_15088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 64), 'complex_def', False)
    # Applying the binary operator '%' (line 308)
    result_mod_15089 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 30), '%', str_15087, complex_def_15088)
    
    # Processing the call keyword arguments (line 308)
    kwargs_15090 = {}
    # Getting the type of 'SystemError' (line 308)
    SystemError_15086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 18), 'SystemError', False)
    # Calling SystemError(args, kwargs) (line 308)
    SystemError_call_result_15091 = invoke(stypy.reporting.localization.Localization(__file__, 308, 18), SystemError_15086, *[result_mod_15089], **kwargs_15090)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 308, 12), SystemError_call_result_15091, 'raise parameter', BaseException)
    # SSA join for if statement (line 305)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 310)
    tuple_15092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 310)
    # Adding element type (line 310)
    str_15093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 17), 'str', 'Py_intptr_t')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 17), tuple_15092, str_15093)
    # Adding element type (line 310)
    str_15094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 32), 'str', 'off_t')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 17), tuple_15092, str_15094)
    
    # Testing the type of a for loop iterable (line 310)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 310, 4), tuple_15092)
    # Getting the type of the for loop variable (line 310)
    for_loop_var_15095 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 310, 4), tuple_15092)
    # Assigning a type to the variable 'type' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'type', for_loop_var_15095)
    # SSA begins for a for statement (line 310)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 311):
    
    # Assigning a Call to a Name (line 311):
    
    # Call to check_type_size(...): (line 311)
    # Processing the call arguments (line 311)
    # Getting the type of 'type' (line 311)
    type_15098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 41), 'type', False)
    # Processing the call keyword arguments (line 311)
    
    # Obtaining an instance of the builtin type 'list' (line 311)
    list_15099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 55), 'list')
    # Adding type elements to the builtin type 'list' instance (line 311)
    # Adding element type (line 311)
    str_15100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 56), 'str', 'Python.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 55), list_15099, str_15100)
    
    keyword_15101 = list_15099
    
    # Obtaining an instance of the builtin type 'list' (line 312)
    list_15102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 312)
    # Adding element type (line 312)
    
    # Call to pythonlib_dir(...): (line 312)
    # Processing the call keyword arguments (line 312)
    kwargs_15104 = {}
    # Getting the type of 'pythonlib_dir' (line 312)
    pythonlib_dir_15103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 30), 'pythonlib_dir', False)
    # Calling pythonlib_dir(args, kwargs) (line 312)
    pythonlib_dir_call_result_15105 = invoke(stypy.reporting.localization.Localization(__file__, 312, 30), pythonlib_dir_15103, *[], **kwargs_15104)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 29), list_15102, pythonlib_dir_call_result_15105)
    
    keyword_15106 = list_15102
    
    # Obtaining the type of the subscript
    # Getting the type of 'type' (line 313)
    type_15107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 34), 'type', False)
    # Getting the type of 'expected' (line 313)
    expected_15108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 25), 'expected', False)
    # Obtaining the member '__getitem__' of a type (line 313)
    getitem___15109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 25), expected_15108, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 313)
    subscript_call_result_15110 = invoke(stypy.reporting.localization.Localization(__file__, 313, 25), getitem___15109, type_15107)
    
    keyword_15111 = subscript_call_result_15110
    kwargs_15112 = {'expected': keyword_15111, 'headers': keyword_15101, 'library_dirs': keyword_15106}
    # Getting the type of 'config_cmd' (line 311)
    config_cmd_15096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 14), 'config_cmd', False)
    # Obtaining the member 'check_type_size' of a type (line 311)
    check_type_size_15097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 14), config_cmd_15096, 'check_type_size')
    # Calling check_type_size(args, kwargs) (line 311)
    check_type_size_call_result_15113 = invoke(stypy.reporting.localization.Localization(__file__, 311, 14), check_type_size_15097, *[type_15098], **kwargs_15112)
    
    # Assigning a type to the variable 'res' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'res', check_type_size_call_result_15113)
    
    
    # Getting the type of 'res' (line 315)
    res_15114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 11), 'res')
    int_15115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 18), 'int')
    # Applying the binary operator '>=' (line 315)
    result_ge_15116 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 11), '>=', res_15114, int_15115)
    
    # Testing the type of an if condition (line 315)
    if_condition_15117 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 315, 8), result_ge_15116)
    # Assigning a type to the variable 'if_condition_15117' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'if_condition_15117', if_condition_15117)
    # SSA begins for if statement (line 315)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 316)
    # Processing the call arguments (line 316)
    
    # Obtaining an instance of the builtin type 'tuple' (line 316)
    tuple_15120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 316)
    # Adding element type (line 316)
    str_15121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 36), 'str', 'SIZEOF_%s')
    
    # Call to sym2def(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'type' (line 316)
    type_15123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 58), 'type', False)
    # Processing the call keyword arguments (line 316)
    kwargs_15124 = {}
    # Getting the type of 'sym2def' (line 316)
    sym2def_15122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 50), 'sym2def', False)
    # Calling sym2def(args, kwargs) (line 316)
    sym2def_call_result_15125 = invoke(stypy.reporting.localization.Localization(__file__, 316, 50), sym2def_15122, *[type_15123], **kwargs_15124)
    
    # Applying the binary operator '%' (line 316)
    result_mod_15126 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 36), '%', str_15121, sym2def_call_result_15125)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 36), tuple_15120, result_mod_15126)
    # Adding element type (line 316)
    str_15127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 65), 'str', '%d')
    # Getting the type of 'res' (line 316)
    res_15128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 72), 'res', False)
    # Applying the binary operator '%' (line 316)
    result_mod_15129 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 65), '%', str_15127, res_15128)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 36), tuple_15120, result_mod_15129)
    
    # Processing the call keyword arguments (line 316)
    kwargs_15130 = {}
    # Getting the type of 'private_defines' (line 316)
    private_defines_15118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'private_defines', False)
    # Obtaining the member 'append' of a type (line 316)
    append_15119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 12), private_defines_15118, 'append')
    # Calling append(args, kwargs) (line 316)
    append_call_result_15131 = invoke(stypy.reporting.localization.Localization(__file__, 316, 12), append_15119, *[tuple_15120], **kwargs_15130)
    
    
    # Call to append(...): (line 317)
    # Processing the call arguments (line 317)
    
    # Obtaining an instance of the builtin type 'tuple' (line 317)
    tuple_15134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 317)
    # Adding element type (line 317)
    str_15135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 35), 'str', 'NPY_SIZEOF_%s')
    
    # Call to sym2def(...): (line 317)
    # Processing the call arguments (line 317)
    # Getting the type of 'type' (line 317)
    type_15137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 61), 'type', False)
    # Processing the call keyword arguments (line 317)
    kwargs_15138 = {}
    # Getting the type of 'sym2def' (line 317)
    sym2def_15136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 53), 'sym2def', False)
    # Calling sym2def(args, kwargs) (line 317)
    sym2def_call_result_15139 = invoke(stypy.reporting.localization.Localization(__file__, 317, 53), sym2def_15136, *[type_15137], **kwargs_15138)
    
    # Applying the binary operator '%' (line 317)
    result_mod_15140 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 35), '%', str_15135, sym2def_call_result_15139)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 35), tuple_15134, result_mod_15140)
    # Adding element type (line 317)
    str_15141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 68), 'str', '%d')
    # Getting the type of 'res' (line 317)
    res_15142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 75), 'res', False)
    # Applying the binary operator '%' (line 317)
    result_mod_15143 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 68), '%', str_15141, res_15142)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 35), tuple_15134, result_mod_15143)
    
    # Processing the call keyword arguments (line 317)
    kwargs_15144 = {}
    # Getting the type of 'public_defines' (line 317)
    public_defines_15132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'public_defines', False)
    # Obtaining the member 'append' of a type (line 317)
    append_15133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 12), public_defines_15132, 'append')
    # Calling append(args, kwargs) (line 317)
    append_call_result_15145 = invoke(stypy.reporting.localization.Localization(__file__, 317, 12), append_15133, *[tuple_15134], **kwargs_15144)
    
    # SSA branch for the else part of an if statement (line 315)
    module_type_store.open_ssa_branch('else')
    
    # Call to SystemError(...): (line 319)
    # Processing the call arguments (line 319)
    str_15147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 30), 'str', 'Checking sizeof (%s) failed !')
    # Getting the type of 'type' (line 319)
    type_15148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 64), 'type', False)
    # Applying the binary operator '%' (line 319)
    result_mod_15149 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 30), '%', str_15147, type_15148)
    
    # Processing the call keyword arguments (line 319)
    kwargs_15150 = {}
    # Getting the type of 'SystemError' (line 319)
    SystemError_15146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 18), 'SystemError', False)
    # Calling SystemError(args, kwargs) (line 319)
    SystemError_call_result_15151 = invoke(stypy.reporting.localization.Localization(__file__, 319, 18), SystemError_15146, *[result_mod_15149], **kwargs_15150)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 319, 12), SystemError_call_result_15151, 'raise parameter', BaseException)
    # SSA join for if statement (line 315)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to check_decl(...): (line 322)
    # Processing the call arguments (line 322)
    str_15154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 29), 'str', 'PY_LONG_LONG')
    # Processing the call keyword arguments (line 322)
    
    # Obtaining an instance of the builtin type 'list' (line 322)
    list_15155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 53), 'list')
    # Adding type elements to the builtin type 'list' instance (line 322)
    # Adding element type (line 322)
    str_15156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 54), 'str', 'Python.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 53), list_15155, str_15156)
    
    keyword_15157 = list_15155
    kwargs_15158 = {'headers': keyword_15157}
    # Getting the type of 'config_cmd' (line 322)
    config_cmd_15152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 7), 'config_cmd', False)
    # Obtaining the member 'check_decl' of a type (line 322)
    check_decl_15153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 7), config_cmd_15152, 'check_decl')
    # Calling check_decl(args, kwargs) (line 322)
    check_decl_call_result_15159 = invoke(stypy.reporting.localization.Localization(__file__, 322, 7), check_decl_15153, *[str_15154], **kwargs_15158)
    
    # Testing the type of an if condition (line 322)
    if_condition_15160 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 4), check_decl_call_result_15159)
    # Assigning a type to the variable 'if_condition_15160' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'if_condition_15160', if_condition_15160)
    # SSA begins for if statement (line 322)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 323):
    
    # Assigning a Call to a Name (line 323):
    
    # Call to check_type_size(...): (line 323)
    # Processing the call arguments (line 323)
    str_15163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 41), 'str', 'PY_LONG_LONG')
    # Processing the call keyword arguments (line 323)
    
    # Obtaining an instance of the builtin type 'list' (line 323)
    list_15164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 66), 'list')
    # Adding type elements to the builtin type 'list' instance (line 323)
    # Adding element type (line 323)
    str_15165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 67), 'str', 'Python.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 66), list_15164, str_15165)
    
    keyword_15166 = list_15164
    
    # Obtaining an instance of the builtin type 'list' (line 324)
    list_15167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 324)
    # Adding element type (line 324)
    
    # Call to pythonlib_dir(...): (line 324)
    # Processing the call keyword arguments (line 324)
    kwargs_15169 = {}
    # Getting the type of 'pythonlib_dir' (line 324)
    pythonlib_dir_15168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 30), 'pythonlib_dir', False)
    # Calling pythonlib_dir(args, kwargs) (line 324)
    pythonlib_dir_call_result_15170 = invoke(stypy.reporting.localization.Localization(__file__, 324, 30), pythonlib_dir_15168, *[], **kwargs_15169)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 29), list_15167, pythonlib_dir_call_result_15170)
    
    keyword_15171 = list_15167
    
    # Obtaining the type of the subscript
    str_15172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 34), 'str', 'PY_LONG_LONG')
    # Getting the type of 'expected' (line 325)
    expected_15173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 25), 'expected', False)
    # Obtaining the member '__getitem__' of a type (line 325)
    getitem___15174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 25), expected_15173, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 325)
    subscript_call_result_15175 = invoke(stypy.reporting.localization.Localization(__file__, 325, 25), getitem___15174, str_15172)
    
    keyword_15176 = subscript_call_result_15175
    kwargs_15177 = {'expected': keyword_15176, 'headers': keyword_15166, 'library_dirs': keyword_15171}
    # Getting the type of 'config_cmd' (line 323)
    config_cmd_15161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 14), 'config_cmd', False)
    # Obtaining the member 'check_type_size' of a type (line 323)
    check_type_size_15162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 14), config_cmd_15161, 'check_type_size')
    # Calling check_type_size(args, kwargs) (line 323)
    check_type_size_call_result_15178 = invoke(stypy.reporting.localization.Localization(__file__, 323, 14), check_type_size_15162, *[str_15163], **kwargs_15177)
    
    # Assigning a type to the variable 'res' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'res', check_type_size_call_result_15178)
    
    
    # Getting the type of 'res' (line 326)
    res_15179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 11), 'res')
    int_15180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 18), 'int')
    # Applying the binary operator '>=' (line 326)
    result_ge_15181 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 11), '>=', res_15179, int_15180)
    
    # Testing the type of an if condition (line 326)
    if_condition_15182 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 326, 8), result_ge_15181)
    # Assigning a type to the variable 'if_condition_15182' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'if_condition_15182', if_condition_15182)
    # SSA begins for if statement (line 326)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 327)
    # Processing the call arguments (line 327)
    
    # Obtaining an instance of the builtin type 'tuple' (line 327)
    tuple_15185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 327)
    # Adding element type (line 327)
    str_15186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 36), 'str', 'SIZEOF_%s')
    
    # Call to sym2def(...): (line 327)
    # Processing the call arguments (line 327)
    str_15188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 58), 'str', 'PY_LONG_LONG')
    # Processing the call keyword arguments (line 327)
    kwargs_15189 = {}
    # Getting the type of 'sym2def' (line 327)
    sym2def_15187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 50), 'sym2def', False)
    # Calling sym2def(args, kwargs) (line 327)
    sym2def_call_result_15190 = invoke(stypy.reporting.localization.Localization(__file__, 327, 50), sym2def_15187, *[str_15188], **kwargs_15189)
    
    # Applying the binary operator '%' (line 327)
    result_mod_15191 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 36), '%', str_15186, sym2def_call_result_15190)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 36), tuple_15185, result_mod_15191)
    # Adding element type (line 327)
    str_15192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 75), 'str', '%d')
    # Getting the type of 'res' (line 327)
    res_15193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 82), 'res', False)
    # Applying the binary operator '%' (line 327)
    result_mod_15194 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 75), '%', str_15192, res_15193)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 36), tuple_15185, result_mod_15194)
    
    # Processing the call keyword arguments (line 327)
    kwargs_15195 = {}
    # Getting the type of 'private_defines' (line 327)
    private_defines_15183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'private_defines', False)
    # Obtaining the member 'append' of a type (line 327)
    append_15184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 12), private_defines_15183, 'append')
    # Calling append(args, kwargs) (line 327)
    append_call_result_15196 = invoke(stypy.reporting.localization.Localization(__file__, 327, 12), append_15184, *[tuple_15185], **kwargs_15195)
    
    
    # Call to append(...): (line 328)
    # Processing the call arguments (line 328)
    
    # Obtaining an instance of the builtin type 'tuple' (line 328)
    tuple_15199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 328)
    # Adding element type (line 328)
    str_15200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 35), 'str', 'NPY_SIZEOF_%s')
    
    # Call to sym2def(...): (line 328)
    # Processing the call arguments (line 328)
    str_15202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 61), 'str', 'PY_LONG_LONG')
    # Processing the call keyword arguments (line 328)
    kwargs_15203 = {}
    # Getting the type of 'sym2def' (line 328)
    sym2def_15201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 53), 'sym2def', False)
    # Calling sym2def(args, kwargs) (line 328)
    sym2def_call_result_15204 = invoke(stypy.reporting.localization.Localization(__file__, 328, 53), sym2def_15201, *[str_15202], **kwargs_15203)
    
    # Applying the binary operator '%' (line 328)
    result_mod_15205 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 35), '%', str_15200, sym2def_call_result_15204)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 35), tuple_15199, result_mod_15205)
    # Adding element type (line 328)
    str_15206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 78), 'str', '%d')
    # Getting the type of 'res' (line 328)
    res_15207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 85), 'res', False)
    # Applying the binary operator '%' (line 328)
    result_mod_15208 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 78), '%', str_15206, res_15207)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 35), tuple_15199, result_mod_15208)
    
    # Processing the call keyword arguments (line 328)
    kwargs_15209 = {}
    # Getting the type of 'public_defines' (line 328)
    public_defines_15197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'public_defines', False)
    # Obtaining the member 'append' of a type (line 328)
    append_15198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 12), public_defines_15197, 'append')
    # Calling append(args, kwargs) (line 328)
    append_call_result_15210 = invoke(stypy.reporting.localization.Localization(__file__, 328, 12), append_15198, *[tuple_15199], **kwargs_15209)
    
    # SSA branch for the else part of an if statement (line 326)
    module_type_store.open_ssa_branch('else')
    
    # Call to SystemError(...): (line 330)
    # Processing the call arguments (line 330)
    str_15212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 30), 'str', 'Checking sizeof (%s) failed !')
    str_15213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 64), 'str', 'PY_LONG_LONG')
    # Applying the binary operator '%' (line 330)
    result_mod_15214 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 30), '%', str_15212, str_15213)
    
    # Processing the call keyword arguments (line 330)
    kwargs_15215 = {}
    # Getting the type of 'SystemError' (line 330)
    SystemError_15211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 18), 'SystemError', False)
    # Calling SystemError(args, kwargs) (line 330)
    SystemError_call_result_15216 = invoke(stypy.reporting.localization.Localization(__file__, 330, 18), SystemError_15211, *[result_mod_15214], **kwargs_15215)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 330, 12), SystemError_call_result_15216, 'raise parameter', BaseException)
    # SSA join for if statement (line 326)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 332):
    
    # Assigning a Call to a Name (line 332):
    
    # Call to check_type_size(...): (line 332)
    # Processing the call arguments (line 332)
    str_15219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 41), 'str', 'long long')
    # Processing the call keyword arguments (line 332)
    
    # Obtaining the type of the subscript
    str_15220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 34), 'str', 'long long')
    # Getting the type of 'expected' (line 333)
    expected_15221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 25), 'expected', False)
    # Obtaining the member '__getitem__' of a type (line 333)
    getitem___15222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 25), expected_15221, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 333)
    subscript_call_result_15223 = invoke(stypy.reporting.localization.Localization(__file__, 333, 25), getitem___15222, str_15220)
    
    keyword_15224 = subscript_call_result_15223
    kwargs_15225 = {'expected': keyword_15224}
    # Getting the type of 'config_cmd' (line 332)
    config_cmd_15217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 14), 'config_cmd', False)
    # Obtaining the member 'check_type_size' of a type (line 332)
    check_type_size_15218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 14), config_cmd_15217, 'check_type_size')
    # Calling check_type_size(args, kwargs) (line 332)
    check_type_size_call_result_15226 = invoke(stypy.reporting.localization.Localization(__file__, 332, 14), check_type_size_15218, *[str_15219], **kwargs_15225)
    
    # Assigning a type to the variable 'res' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'res', check_type_size_call_result_15226)
    
    
    # Getting the type of 'res' (line 334)
    res_15227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 11), 'res')
    int_15228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 18), 'int')
    # Applying the binary operator '>=' (line 334)
    result_ge_15229 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 11), '>=', res_15227, int_15228)
    
    # Testing the type of an if condition (line 334)
    if_condition_15230 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 8), result_ge_15229)
    # Assigning a type to the variable 'if_condition_15230' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'if_condition_15230', if_condition_15230)
    # SSA begins for if statement (line 334)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 336)
    # Processing the call arguments (line 336)
    
    # Obtaining an instance of the builtin type 'tuple' (line 336)
    tuple_15233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 336)
    # Adding element type (line 336)
    str_15234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 35), 'str', 'NPY_SIZEOF_%s')
    
    # Call to sym2def(...): (line 336)
    # Processing the call arguments (line 336)
    str_15236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 61), 'str', 'long long')
    # Processing the call keyword arguments (line 336)
    kwargs_15237 = {}
    # Getting the type of 'sym2def' (line 336)
    sym2def_15235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 53), 'sym2def', False)
    # Calling sym2def(args, kwargs) (line 336)
    sym2def_call_result_15238 = invoke(stypy.reporting.localization.Localization(__file__, 336, 53), sym2def_15235, *[str_15236], **kwargs_15237)
    
    # Applying the binary operator '%' (line 336)
    result_mod_15239 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 35), '%', str_15234, sym2def_call_result_15238)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 35), tuple_15233, result_mod_15239)
    # Adding element type (line 336)
    str_15240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 75), 'str', '%d')
    # Getting the type of 'res' (line 336)
    res_15241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 82), 'res', False)
    # Applying the binary operator '%' (line 336)
    result_mod_15242 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 75), '%', str_15240, res_15241)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 35), tuple_15233, result_mod_15242)
    
    # Processing the call keyword arguments (line 336)
    kwargs_15243 = {}
    # Getting the type of 'public_defines' (line 336)
    public_defines_15231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'public_defines', False)
    # Obtaining the member 'append' of a type (line 336)
    append_15232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 12), public_defines_15231, 'append')
    # Calling append(args, kwargs) (line 336)
    append_call_result_15244 = invoke(stypy.reporting.localization.Localization(__file__, 336, 12), append_15232, *[tuple_15233], **kwargs_15243)
    
    # SSA branch for the else part of an if statement (line 334)
    module_type_store.open_ssa_branch('else')
    
    # Call to SystemError(...): (line 338)
    # Processing the call arguments (line 338)
    str_15246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 30), 'str', 'Checking sizeof (%s) failed !')
    str_15247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 64), 'str', 'long long')
    # Applying the binary operator '%' (line 338)
    result_mod_15248 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 30), '%', str_15246, str_15247)
    
    # Processing the call keyword arguments (line 338)
    kwargs_15249 = {}
    # Getting the type of 'SystemError' (line 338)
    SystemError_15245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 18), 'SystemError', False)
    # Calling SystemError(args, kwargs) (line 338)
    SystemError_call_result_15250 = invoke(stypy.reporting.localization.Localization(__file__, 338, 18), SystemError_15245, *[result_mod_15248], **kwargs_15249)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 338, 12), SystemError_call_result_15250, 'raise parameter', BaseException)
    # SSA join for if statement (line 334)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 322)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to check_decl(...): (line 340)
    # Processing the call arguments (line 340)
    str_15253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 33), 'str', 'CHAR_BIT')
    # Processing the call keyword arguments (line 340)
    
    # Obtaining an instance of the builtin type 'list' (line 340)
    list_15254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 53), 'list')
    # Adding type elements to the builtin type 'list' instance (line 340)
    # Adding element type (line 340)
    str_15255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 54), 'str', 'Python.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 53), list_15254, str_15255)
    
    keyword_15256 = list_15254
    kwargs_15257 = {'headers': keyword_15256}
    # Getting the type of 'config_cmd' (line 340)
    config_cmd_15251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 11), 'config_cmd', False)
    # Obtaining the member 'check_decl' of a type (line 340)
    check_decl_15252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 11), config_cmd_15251, 'check_decl')
    # Calling check_decl(args, kwargs) (line 340)
    check_decl_call_result_15258 = invoke(stypy.reporting.localization.Localization(__file__, 340, 11), check_decl_15252, *[str_15253], **kwargs_15257)
    
    # Applying the 'not' unary operator (line 340)
    result_not__15259 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 7), 'not', check_decl_call_result_15258)
    
    # Testing the type of an if condition (line 340)
    if_condition_15260 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 4), result_not__15259)
    # Assigning a type to the variable 'if_condition_15260' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'if_condition_15260', if_condition_15260)
    # SSA begins for if statement (line 340)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 341)
    # Processing the call arguments (line 341)
    str_15262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 12), 'str', 'Config wo CHAR_BIT is not supported, please contact the maintainers')
    # Processing the call keyword arguments (line 341)
    kwargs_15263 = {}
    # Getting the type of 'RuntimeError' (line 341)
    RuntimeError_15261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 341)
    RuntimeError_call_result_15264 = invoke(stypy.reporting.localization.Localization(__file__, 341, 14), RuntimeError_15261, *[str_15262], **kwargs_15263)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 341, 8), RuntimeError_call_result_15264, 'raise parameter', BaseException)
    # SSA join for if statement (line 340)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 345)
    tuple_15265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 345)
    # Adding element type (line 345)
    # Getting the type of 'private_defines' (line 345)
    private_defines_15266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 11), 'private_defines')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 11), tuple_15265, private_defines_15266)
    # Adding element type (line 345)
    # Getting the type of 'public_defines' (line 345)
    public_defines_15267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 28), 'public_defines')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 11), tuple_15265, public_defines_15267)
    
    # Assigning a type to the variable 'stypy_return_type' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'stypy_return_type', tuple_15265)
    
    # ################# End of 'check_types(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_types' in the type store
    # Getting the type of 'stypy_return_type' (line 253)
    stypy_return_type_15268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15268)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_types'
    return stypy_return_type_15268

# Assigning a type to the variable 'check_types' (line 253)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 0), 'check_types', check_types)

@norecursion
def check_mathlib(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_mathlib'
    module_type_store = module_type_store.open_function_context('check_mathlib', 347, 0, False)
    
    # Passed parameters checking function
    check_mathlib.stypy_localization = localization
    check_mathlib.stypy_type_of_self = None
    check_mathlib.stypy_type_store = module_type_store
    check_mathlib.stypy_function_name = 'check_mathlib'
    check_mathlib.stypy_param_names_list = ['config_cmd']
    check_mathlib.stypy_varargs_param_name = None
    check_mathlib.stypy_kwargs_param_name = None
    check_mathlib.stypy_call_defaults = defaults
    check_mathlib.stypy_call_varargs = varargs
    check_mathlib.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_mathlib', ['config_cmd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_mathlib', localization, ['config_cmd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_mathlib(...)' code ##################

    
    # Assigning a List to a Name (line 349):
    
    # Assigning a List to a Name (line 349):
    
    # Obtaining an instance of the builtin type 'list' (line 349)
    list_15269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 349)
    
    # Assigning a type to the variable 'mathlibs' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'mathlibs', list_15269)
    
    # Assigning a List to a Name (line 350):
    
    # Assigning a List to a Name (line 350):
    
    # Obtaining an instance of the builtin type 'list' (line 350)
    list_15270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 350)
    # Adding element type (line 350)
    
    # Obtaining an instance of the builtin type 'list' (line 350)
    list_15271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 350)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 23), list_15270, list_15271)
    # Adding element type (line 350)
    
    # Obtaining an instance of the builtin type 'list' (line 350)
    list_15272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 350)
    # Adding element type (line 350)
    str_15273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 29), 'str', 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 28), list_15272, str_15273)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 23), list_15270, list_15272)
    # Adding element type (line 350)
    
    # Obtaining an instance of the builtin type 'list' (line 350)
    list_15274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 350)
    # Adding element type (line 350)
    str_15275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 36), 'str', 'cpml')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 35), list_15274, str_15275)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 23), list_15270, list_15274)
    
    # Assigning a type to the variable 'mathlibs_choices' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'mathlibs_choices', list_15270)
    
    # Assigning a Call to a Name (line 351):
    
    # Assigning a Call to a Name (line 351):
    
    # Call to get(...): (line 351)
    # Processing the call arguments (line 351)
    str_15279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 29), 'str', 'MATHLIB')
    # Processing the call keyword arguments (line 351)
    kwargs_15280 = {}
    # Getting the type of 'os' (line 351)
    os_15276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 14), 'os', False)
    # Obtaining the member 'environ' of a type (line 351)
    environ_15277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 14), os_15276, 'environ')
    # Obtaining the member 'get' of a type (line 351)
    get_15278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 14), environ_15277, 'get')
    # Calling get(args, kwargs) (line 351)
    get_call_result_15281 = invoke(stypy.reporting.localization.Localization(__file__, 351, 14), get_15278, *[str_15279], **kwargs_15280)
    
    # Assigning a type to the variable 'mathlib' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'mathlib', get_call_result_15281)
    
    # Getting the type of 'mathlib' (line 352)
    mathlib_15282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 7), 'mathlib')
    # Testing the type of an if condition (line 352)
    if_condition_15283 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 352, 4), mathlib_15282)
    # Assigning a type to the variable 'if_condition_15283' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'if_condition_15283', if_condition_15283)
    # SSA begins for if statement (line 352)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to insert(...): (line 353)
    # Processing the call arguments (line 353)
    int_15286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 32), 'int')
    
    # Call to split(...): (line 353)
    # Processing the call arguments (line 353)
    str_15289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 49), 'str', ',')
    # Processing the call keyword arguments (line 353)
    kwargs_15290 = {}
    # Getting the type of 'mathlib' (line 353)
    mathlib_15287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 35), 'mathlib', False)
    # Obtaining the member 'split' of a type (line 353)
    split_15288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 35), mathlib_15287, 'split')
    # Calling split(args, kwargs) (line 353)
    split_call_result_15291 = invoke(stypy.reporting.localization.Localization(__file__, 353, 35), split_15288, *[str_15289], **kwargs_15290)
    
    # Processing the call keyword arguments (line 353)
    kwargs_15292 = {}
    # Getting the type of 'mathlibs_choices' (line 353)
    mathlibs_choices_15284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'mathlibs_choices', False)
    # Obtaining the member 'insert' of a type (line 353)
    insert_15285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 8), mathlibs_choices_15284, 'insert')
    # Calling insert(args, kwargs) (line 353)
    insert_call_result_15293 = invoke(stypy.reporting.localization.Localization(__file__, 353, 8), insert_15285, *[int_15286, split_call_result_15291], **kwargs_15292)
    
    # SSA join for if statement (line 352)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'mathlibs_choices' (line 354)
    mathlibs_choices_15294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 16), 'mathlibs_choices')
    # Testing the type of a for loop iterable (line 354)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 354, 4), mathlibs_choices_15294)
    # Getting the type of the for loop variable (line 354)
    for_loop_var_15295 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 354, 4), mathlibs_choices_15294)
    # Assigning a type to the variable 'libs' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'libs', for_loop_var_15295)
    # SSA begins for a for statement (line 354)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to check_func(...): (line 355)
    # Processing the call arguments (line 355)
    str_15298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 33), 'str', 'exp')
    # Processing the call keyword arguments (line 355)
    # Getting the type of 'libs' (line 355)
    libs_15299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 50), 'libs', False)
    keyword_15300 = libs_15299
    # Getting the type of 'True' (line 355)
    True_15301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 61), 'True', False)
    keyword_15302 = True_15301
    # Getting the type of 'True' (line 355)
    True_15303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 72), 'True', False)
    keyword_15304 = True_15303
    kwargs_15305 = {'libraries': keyword_15300, 'decl': keyword_15302, 'call': keyword_15304}
    # Getting the type of 'config_cmd' (line 355)
    config_cmd_15296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 11), 'config_cmd', False)
    # Obtaining the member 'check_func' of a type (line 355)
    check_func_15297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 11), config_cmd_15296, 'check_func')
    # Calling check_func(args, kwargs) (line 355)
    check_func_call_result_15306 = invoke(stypy.reporting.localization.Localization(__file__, 355, 11), check_func_15297, *[str_15298], **kwargs_15305)
    
    # Testing the type of an if condition (line 355)
    if_condition_15307 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 355, 8), check_func_call_result_15306)
    # Assigning a type to the variable 'if_condition_15307' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'if_condition_15307', if_condition_15307)
    # SSA begins for if statement (line 355)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 356):
    
    # Assigning a Name to a Name (line 356):
    # Getting the type of 'libs' (line 356)
    libs_15308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 23), 'libs')
    # Assigning a type to the variable 'mathlibs' (line 356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'mathlibs', libs_15308)
    # SSA join for if statement (line 355)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of a for statement (line 354)
    module_type_store.open_ssa_branch('for loop else')
    
    # Call to EnvironmentError(...): (line 359)
    # Processing the call arguments (line 359)
    str_15310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 31), 'str', 'math library missing; rerun setup.py after setting the MATHLIB env variable')
    # Processing the call keyword arguments (line 359)
    kwargs_15311 = {}
    # Getting the type of 'EnvironmentError' (line 359)
    EnvironmentError_15309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 14), 'EnvironmentError', False)
    # Calling EnvironmentError(args, kwargs) (line 359)
    EnvironmentError_call_result_15312 = invoke(stypy.reporting.localization.Localization(__file__, 359, 14), EnvironmentError_15309, *[str_15310], **kwargs_15311)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 359, 8), EnvironmentError_call_result_15312, 'raise parameter', BaseException)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'mathlibs' (line 362)
    mathlibs_15313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 11), 'mathlibs')
    # Assigning a type to the variable 'stypy_return_type' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'stypy_return_type', mathlibs_15313)
    
    # ################# End of 'check_mathlib(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_mathlib' in the type store
    # Getting the type of 'stypy_return_type' (line 347)
    stypy_return_type_15314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15314)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_mathlib'
    return stypy_return_type_15314

# Assigning a type to the variable 'check_mathlib' (line 347)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 0), 'check_mathlib', check_mathlib)

@norecursion
def visibility_define(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'visibility_define'
    module_type_store = module_type_store.open_function_context('visibility_define', 364, 0, False)
    
    # Passed parameters checking function
    visibility_define.stypy_localization = localization
    visibility_define.stypy_type_of_self = None
    visibility_define.stypy_type_store = module_type_store
    visibility_define.stypy_function_name = 'visibility_define'
    visibility_define.stypy_param_names_list = ['config']
    visibility_define.stypy_varargs_param_name = None
    visibility_define.stypy_kwargs_param_name = None
    visibility_define.stypy_call_defaults = defaults
    visibility_define.stypy_call_varargs = varargs
    visibility_define.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'visibility_define', ['config'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'visibility_define', localization, ['config'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'visibility_define(...)' code ##################

    str_15315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, (-1)), 'str', 'Return the define value to use for NPY_VISIBILITY_HIDDEN (may be empty\n    string).')
    
    
    # Call to check_compiler_gcc4(...): (line 367)
    # Processing the call keyword arguments (line 367)
    kwargs_15318 = {}
    # Getting the type of 'config' (line 367)
    config_15316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 7), 'config', False)
    # Obtaining the member 'check_compiler_gcc4' of a type (line 367)
    check_compiler_gcc4_15317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 7), config_15316, 'check_compiler_gcc4')
    # Calling check_compiler_gcc4(args, kwargs) (line 367)
    check_compiler_gcc4_call_result_15319 = invoke(stypy.reporting.localization.Localization(__file__, 367, 7), check_compiler_gcc4_15317, *[], **kwargs_15318)
    
    # Testing the type of an if condition (line 367)
    if_condition_15320 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 367, 4), check_compiler_gcc4_call_result_15319)
    # Assigning a type to the variable 'if_condition_15320' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'if_condition_15320', if_condition_15320)
    # SSA begins for if statement (line 367)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_15321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 15), 'str', '__attribute__((visibility("hidden")))')
    # Assigning a type to the variable 'stypy_return_type' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'stypy_return_type', str_15321)
    # SSA branch for the else part of an if statement (line 367)
    module_type_store.open_ssa_branch('else')
    str_15322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 15), 'str', '')
    # Assigning a type to the variable 'stypy_return_type' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'stypy_return_type', str_15322)
    # SSA join for if statement (line 367)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'visibility_define(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'visibility_define' in the type store
    # Getting the type of 'stypy_return_type' (line 364)
    stypy_return_type_15323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15323)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'visibility_define'
    return stypy_return_type_15323

# Assigning a type to the variable 'visibility_define' (line 364)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 0), 'visibility_define', visibility_define)

@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_15324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 33), 'str', '')
    # Getting the type of 'None' (line 372)
    None_15325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 45), 'None')
    defaults = [str_15324, None_15325]
    # Create a new context for function 'configuration'
    module_type_store = module_type_store.open_function_context('configuration', 372, 0, False)
    
    # Passed parameters checking function
    configuration.stypy_localization = localization
    configuration.stypy_type_of_self = None
    configuration.stypy_type_store = module_type_store
    configuration.stypy_function_name = 'configuration'
    configuration.stypy_param_names_list = ['parent_package', 'top_path']
    configuration.stypy_varargs_param_name = None
    configuration.stypy_kwargs_param_name = None
    configuration.stypy_call_defaults = defaults
    configuration.stypy_call_varargs = varargs
    configuration.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'configuration', ['parent_package', 'top_path'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'configuration', localization, ['parent_package', 'top_path'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'configuration(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 373, 4))
    
    # 'from numpy.distutils.misc_util import Configuration, dot_join' statement (line 373)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
    import_15326 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 373, 4), 'numpy.distutils.misc_util')

    if (type(import_15326) is not StypyTypeError):

        if (import_15326 != 'pyd_module'):
            __import__(import_15326)
            sys_modules_15327 = sys.modules[import_15326]
            import_from_module(stypy.reporting.localization.Localization(__file__, 373, 4), 'numpy.distutils.misc_util', sys_modules_15327.module_type_store, module_type_store, ['Configuration', 'dot_join'])
            nest_module(stypy.reporting.localization.Localization(__file__, 373, 4), __file__, sys_modules_15327, sys_modules_15327.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration, dot_join

            import_from_module(stypy.reporting.localization.Localization(__file__, 373, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration', 'dot_join'], [Configuration, dot_join])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'numpy.distutils.misc_util', import_15326)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 374, 4))
    
    # 'from numpy.distutils.system_info import get_info' statement (line 374)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
    import_15328 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 374, 4), 'numpy.distutils.system_info')

    if (type(import_15328) is not StypyTypeError):

        if (import_15328 != 'pyd_module'):
            __import__(import_15328)
            sys_modules_15329 = sys.modules[import_15328]
            import_from_module(stypy.reporting.localization.Localization(__file__, 374, 4), 'numpy.distutils.system_info', sys_modules_15329.module_type_store, module_type_store, ['get_info'])
            nest_module(stypy.reporting.localization.Localization(__file__, 374, 4), __file__, sys_modules_15329, sys_modules_15329.module_type_store, module_type_store)
        else:
            from numpy.distutils.system_info import get_info

            import_from_module(stypy.reporting.localization.Localization(__file__, 374, 4), 'numpy.distutils.system_info', None, module_type_store, ['get_info'], [get_info])

    else:
        # Assigning a type to the variable 'numpy.distutils.system_info' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'numpy.distutils.system_info', import_15328)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')
    
    
    # Assigning a Call to a Name (line 376):
    
    # Assigning a Call to a Name (line 376):
    
    # Call to Configuration(...): (line 376)
    # Processing the call arguments (line 376)
    str_15331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 27), 'str', 'core')
    # Getting the type of 'parent_package' (line 376)
    parent_package_15332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 35), 'parent_package', False)
    # Getting the type of 'top_path' (line 376)
    top_path_15333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 51), 'top_path', False)
    # Processing the call keyword arguments (line 376)
    kwargs_15334 = {}
    # Getting the type of 'Configuration' (line 376)
    Configuration_15330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 376)
    Configuration_call_result_15335 = invoke(stypy.reporting.localization.Localization(__file__, 376, 13), Configuration_15330, *[str_15331, parent_package_15332, top_path_15333], **kwargs_15334)
    
    # Assigning a type to the variable 'config' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'config', Configuration_call_result_15335)
    
    # Assigning a Attribute to a Name (line 377):
    
    # Assigning a Attribute to a Name (line 377):
    # Getting the type of 'config' (line 377)
    config_15336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 16), 'config')
    # Obtaining the member 'local_path' of a type (line 377)
    local_path_15337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 16), config_15336, 'local_path')
    # Assigning a type to the variable 'local_dir' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'local_dir', local_path_15337)
    
    # Assigning a Call to a Name (line 378):
    
    # Assigning a Call to a Name (line 378):
    
    # Call to join(...): (line 378)
    # Processing the call arguments (line 378)
    # Getting the type of 'local_dir' (line 378)
    local_dir_15339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 23), 'local_dir', False)
    str_15340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 34), 'str', 'code_generators')
    # Processing the call keyword arguments (line 378)
    kwargs_15341 = {}
    # Getting the type of 'join' (line 378)
    join_15338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 18), 'join', False)
    # Calling join(args, kwargs) (line 378)
    join_call_result_15342 = invoke(stypy.reporting.localization.Localization(__file__, 378, 18), join_15338, *[local_dir_15339, str_15340], **kwargs_15341)
    
    # Assigning a type to the variable 'codegen_dir' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'codegen_dir', join_call_result_15342)
    
    
    # Call to is_released(...): (line 380)
    # Processing the call arguments (line 380)
    # Getting the type of 'config' (line 380)
    config_15344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 19), 'config', False)
    # Processing the call keyword arguments (line 380)
    kwargs_15345 = {}
    # Getting the type of 'is_released' (line 380)
    is_released_15343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 7), 'is_released', False)
    # Calling is_released(args, kwargs) (line 380)
    is_released_call_result_15346 = invoke(stypy.reporting.localization.Localization(__file__, 380, 7), is_released_15343, *[config_15344], **kwargs_15345)
    
    # Testing the type of an if condition (line 380)
    if_condition_15347 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 380, 4), is_released_call_result_15346)
    # Assigning a type to the variable 'if_condition_15347' (line 380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'if_condition_15347', if_condition_15347)
    # SSA begins for if statement (line 380)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to simplefilter(...): (line 381)
    # Processing the call arguments (line 381)
    str_15350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 30), 'str', 'error')
    # Getting the type of 'MismatchCAPIWarning' (line 381)
    MismatchCAPIWarning_15351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 39), 'MismatchCAPIWarning', False)
    # Processing the call keyword arguments (line 381)
    kwargs_15352 = {}
    # Getting the type of 'warnings' (line 381)
    warnings_15348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'warnings', False)
    # Obtaining the member 'simplefilter' of a type (line 381)
    simplefilter_15349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 8), warnings_15348, 'simplefilter')
    # Calling simplefilter(args, kwargs) (line 381)
    simplefilter_call_result_15353 = invoke(stypy.reporting.localization.Localization(__file__, 381, 8), simplefilter_15349, *[str_15350, MismatchCAPIWarning_15351], **kwargs_15352)
    
    # SSA join for if statement (line 380)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to check_api_version(...): (line 385)
    # Processing the call arguments (line 385)
    # Getting the type of 'C_API_VERSION' (line 385)
    C_API_VERSION_15355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 22), 'C_API_VERSION', False)
    # Getting the type of 'codegen_dir' (line 385)
    codegen_dir_15356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 37), 'codegen_dir', False)
    # Processing the call keyword arguments (line 385)
    kwargs_15357 = {}
    # Getting the type of 'check_api_version' (line 385)
    check_api_version_15354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'check_api_version', False)
    # Calling check_api_version(args, kwargs) (line 385)
    check_api_version_call_result_15358 = invoke(stypy.reporting.localization.Localization(__file__, 385, 4), check_api_version_15354, *[C_API_VERSION_15355, codegen_dir_15356], **kwargs_15357)
    
    
    # Assigning a Call to a Name (line 387):
    
    # Assigning a Call to a Name (line 387):
    
    # Call to join(...): (line 387)
    # Processing the call arguments (line 387)
    # Getting the type of 'codegen_dir' (line 387)
    codegen_dir_15360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 29), 'codegen_dir', False)
    str_15361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 42), 'str', 'generate_umath.py')
    # Processing the call keyword arguments (line 387)
    kwargs_15362 = {}
    # Getting the type of 'join' (line 387)
    join_15359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 24), 'join', False)
    # Calling join(args, kwargs) (line 387)
    join_call_result_15363 = invoke(stypy.reporting.localization.Localization(__file__, 387, 24), join_15359, *[codegen_dir_15360, str_15361], **kwargs_15362)
    
    # Assigning a type to the variable 'generate_umath_py' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'generate_umath_py', join_call_result_15363)
    
    # Assigning a Call to a Name (line 388):
    
    # Assigning a Call to a Name (line 388):
    
    # Call to dot_join(...): (line 388)
    # Processing the call arguments (line 388)
    # Getting the type of 'config' (line 388)
    config_15365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 17), 'config', False)
    # Obtaining the member 'name' of a type (line 388)
    name_15366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 17), config_15365, 'name')
    str_15367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 30), 'str', 'generate_umath')
    # Processing the call keyword arguments (line 388)
    kwargs_15368 = {}
    # Getting the type of 'dot_join' (line 388)
    dot_join_15364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'dot_join', False)
    # Calling dot_join(args, kwargs) (line 388)
    dot_join_call_result_15369 = invoke(stypy.reporting.localization.Localization(__file__, 388, 8), dot_join_15364, *[name_15366, str_15367], **kwargs_15368)
    
    # Assigning a type to the variable 'n' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'n', dot_join_call_result_15369)
    
    # Assigning a Call to a Name (line 389):
    
    # Assigning a Call to a Name (line 389):
    
    # Call to load_module(...): (line 389)
    # Processing the call arguments (line 389)
    
    # Call to join(...): (line 389)
    # Processing the call arguments (line 389)
    
    # Call to split(...): (line 389)
    # Processing the call arguments (line 389)
    str_15376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 54), 'str', '.')
    # Processing the call keyword arguments (line 389)
    kwargs_15377 = {}
    # Getting the type of 'n' (line 389)
    n_15374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 46), 'n', False)
    # Obtaining the member 'split' of a type (line 389)
    split_15375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 46), n_15374, 'split')
    # Calling split(args, kwargs) (line 389)
    split_call_result_15378 = invoke(stypy.reporting.localization.Localization(__file__, 389, 46), split_15375, *[str_15376], **kwargs_15377)
    
    # Processing the call keyword arguments (line 389)
    kwargs_15379 = {}
    str_15372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 37), 'str', '_')
    # Obtaining the member 'join' of a type (line 389)
    join_15373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 37), str_15372, 'join')
    # Calling join(args, kwargs) (line 389)
    join_call_result_15380 = invoke(stypy.reporting.localization.Localization(__file__, 389, 37), join_15373, *[split_call_result_15378], **kwargs_15379)
    
    
    # Call to open(...): (line 390)
    # Processing the call arguments (line 390)
    # Getting the type of 'generate_umath_py' (line 390)
    generate_umath_py_15382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 42), 'generate_umath_py', False)
    str_15383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 61), 'str', 'U')
    # Processing the call keyword arguments (line 390)
    kwargs_15384 = {}
    # Getting the type of 'open' (line 390)
    open_15381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 37), 'open', False)
    # Calling open(args, kwargs) (line 390)
    open_call_result_15385 = invoke(stypy.reporting.localization.Localization(__file__, 390, 37), open_15381, *[generate_umath_py_15382, str_15383], **kwargs_15384)
    
    # Getting the type of 'generate_umath_py' (line 390)
    generate_umath_py_15386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 67), 'generate_umath_py', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 391)
    tuple_15387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 391)
    # Adding element type (line 391)
    str_15388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 38), 'str', '.py')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 38), tuple_15387, str_15388)
    # Adding element type (line 391)
    str_15389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 45), 'str', 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 38), tuple_15387, str_15389)
    # Adding element type (line 391)
    int_15390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 50), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 38), tuple_15387, int_15390)
    
    # Processing the call keyword arguments (line 389)
    kwargs_15391 = {}
    # Getting the type of 'imp' (line 389)
    imp_15370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 21), 'imp', False)
    # Obtaining the member 'load_module' of a type (line 389)
    load_module_15371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 21), imp_15370, 'load_module')
    # Calling load_module(args, kwargs) (line 389)
    load_module_call_result_15392 = invoke(stypy.reporting.localization.Localization(__file__, 389, 21), load_module_15371, *[join_call_result_15380, open_call_result_15385, generate_umath_py_15386, tuple_15387], **kwargs_15391)
    
    # Assigning a type to the variable 'generate_umath' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'generate_umath', load_module_call_result_15392)
    
    # Assigning a Str to a Name (line 393):
    
    # Assigning a Str to a Name (line 393):
    str_15393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 17), 'str', 'include/numpy')
    # Assigning a type to the variable 'header_dir' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'header_dir', str_15393)
    
    # Assigning a Call to a Name (line 395):
    
    # Assigning a Call to a Name (line 395):
    
    # Call to CallOnceOnly(...): (line 395)
    # Processing the call keyword arguments (line 395)
    kwargs_15395 = {}
    # Getting the type of 'CallOnceOnly' (line 395)
    CallOnceOnly_15394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 14), 'CallOnceOnly', False)
    # Calling CallOnceOnly(args, kwargs) (line 395)
    CallOnceOnly_call_result_15396 = invoke(stypy.reporting.localization.Localization(__file__, 395, 14), CallOnceOnly_15394, *[], **kwargs_15395)
    
    # Assigning a type to the variable 'cocache' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'cocache', CallOnceOnly_call_result_15396)

    @norecursion
    def generate_config_h(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'generate_config_h'
        module_type_store = module_type_store.open_function_context('generate_config_h', 397, 4, False)
        
        # Passed parameters checking function
        generate_config_h.stypy_localization = localization
        generate_config_h.stypy_type_of_self = None
        generate_config_h.stypy_type_store = module_type_store
        generate_config_h.stypy_function_name = 'generate_config_h'
        generate_config_h.stypy_param_names_list = ['ext', 'build_dir']
        generate_config_h.stypy_varargs_param_name = None
        generate_config_h.stypy_kwargs_param_name = None
        generate_config_h.stypy_call_defaults = defaults
        generate_config_h.stypy_call_varargs = varargs
        generate_config_h.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'generate_config_h', ['ext', 'build_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'generate_config_h', localization, ['ext', 'build_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'generate_config_h(...)' code ##################

        
        # Assigning a Call to a Name (line 398):
        
        # Assigning a Call to a Name (line 398):
        
        # Call to join(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'build_dir' (line 398)
        build_dir_15398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 22), 'build_dir', False)
        # Getting the type of 'header_dir' (line 398)
        header_dir_15399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 33), 'header_dir', False)
        str_15400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 45), 'str', 'config.h')
        # Processing the call keyword arguments (line 398)
        kwargs_15401 = {}
        # Getting the type of 'join' (line 398)
        join_15397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 17), 'join', False)
        # Calling join(args, kwargs) (line 398)
        join_call_result_15402 = invoke(stypy.reporting.localization.Localization(__file__, 398, 17), join_15397, *[build_dir_15398, header_dir_15399, str_15400], **kwargs_15401)
        
        # Assigning a type to the variable 'target' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'target', join_call_result_15402)
        
        # Assigning a Call to a Name (line 399):
        
        # Assigning a Call to a Name (line 399):
        
        # Call to dirname(...): (line 399)
        # Processing the call arguments (line 399)
        # Getting the type of 'target' (line 399)
        target_15406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 28), 'target', False)
        # Processing the call keyword arguments (line 399)
        kwargs_15407 = {}
        # Getting the type of 'os' (line 399)
        os_15403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'os', False)
        # Obtaining the member 'path' of a type (line 399)
        path_15404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 12), os_15403, 'path')
        # Obtaining the member 'dirname' of a type (line 399)
        dirname_15405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 12), path_15404, 'dirname')
        # Calling dirname(args, kwargs) (line 399)
        dirname_call_result_15408 = invoke(stypy.reporting.localization.Localization(__file__, 399, 12), dirname_15405, *[target_15406], **kwargs_15407)
        
        # Assigning a type to the variable 'd' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'd', dirname_call_result_15408)
        
        
        
        # Call to exists(...): (line 400)
        # Processing the call arguments (line 400)
        # Getting the type of 'd' (line 400)
        d_15412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 30), 'd', False)
        # Processing the call keyword arguments (line 400)
        kwargs_15413 = {}
        # Getting the type of 'os' (line 400)
        os_15409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 400)
        path_15410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 15), os_15409, 'path')
        # Obtaining the member 'exists' of a type (line 400)
        exists_15411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 15), path_15410, 'exists')
        # Calling exists(args, kwargs) (line 400)
        exists_call_result_15414 = invoke(stypy.reporting.localization.Localization(__file__, 400, 15), exists_15411, *[d_15412], **kwargs_15413)
        
        # Applying the 'not' unary operator (line 400)
        result_not__15415 = python_operator(stypy.reporting.localization.Localization(__file__, 400, 11), 'not', exists_call_result_15414)
        
        # Testing the type of an if condition (line 400)
        if_condition_15416 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 400, 8), result_not__15415)
        # Assigning a type to the variable 'if_condition_15416' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'if_condition_15416', if_condition_15416)
        # SSA begins for if statement (line 400)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to makedirs(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'd' (line 401)
        d_15419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 24), 'd', False)
        # Processing the call keyword arguments (line 401)
        kwargs_15420 = {}
        # Getting the type of 'os' (line 401)
        os_15417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'os', False)
        # Obtaining the member 'makedirs' of a type (line 401)
        makedirs_15418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 12), os_15417, 'makedirs')
        # Calling makedirs(args, kwargs) (line 401)
        makedirs_call_result_15421 = invoke(stypy.reporting.localization.Localization(__file__, 401, 12), makedirs_15418, *[d_15419], **kwargs_15420)
        
        # SSA join for if statement (line 400)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to newer(...): (line 403)
        # Processing the call arguments (line 403)
        # Getting the type of '__file__' (line 403)
        file___15423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 17), '__file__', False)
        # Getting the type of 'target' (line 403)
        target_15424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 27), 'target', False)
        # Processing the call keyword arguments (line 403)
        kwargs_15425 = {}
        # Getting the type of 'newer' (line 403)
        newer_15422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 11), 'newer', False)
        # Calling newer(args, kwargs) (line 403)
        newer_call_result_15426 = invoke(stypy.reporting.localization.Localization(__file__, 403, 11), newer_15422, *[file___15423, target_15424], **kwargs_15425)
        
        # Testing the type of an if condition (line 403)
        if_condition_15427 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 403, 8), newer_call_result_15426)
        # Assigning a type to the variable 'if_condition_15427' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'if_condition_15427', if_condition_15427)
        # SSA begins for if statement (line 403)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 404):
        
        # Assigning a Call to a Name (line 404):
        
        # Call to get_config_cmd(...): (line 404)
        # Processing the call keyword arguments (line 404)
        kwargs_15430 = {}
        # Getting the type of 'config' (line 404)
        config_15428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 25), 'config', False)
        # Obtaining the member 'get_config_cmd' of a type (line 404)
        get_config_cmd_15429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 25), config_15428, 'get_config_cmd')
        # Calling get_config_cmd(args, kwargs) (line 404)
        get_config_cmd_call_result_15431 = invoke(stypy.reporting.localization.Localization(__file__, 404, 25), get_config_cmd_15429, *[], **kwargs_15430)
        
        # Assigning a type to the variable 'config_cmd' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 12), 'config_cmd', get_config_cmd_call_result_15431)
        
        # Call to info(...): (line 405)
        # Processing the call arguments (line 405)
        str_15434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 21), 'str', 'Generating %s')
        # Getting the type of 'target' (line 405)
        target_15435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 38), 'target', False)
        # Processing the call keyword arguments (line 405)
        kwargs_15436 = {}
        # Getting the type of 'log' (line 405)
        log_15432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 405)
        info_15433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 12), log_15432, 'info')
        # Calling info(args, kwargs) (line 405)
        info_call_result_15437 = invoke(stypy.reporting.localization.Localization(__file__, 405, 12), info_15433, *[str_15434, target_15435], **kwargs_15436)
        
        
        # Assigning a Call to a Tuple (line 408):
        
        # Assigning a Call to a Name:
        
        # Call to check_types(...): (line 408)
        # Processing the call arguments (line 408)
        # Getting the type of 'config_cmd' (line 408)
        config_cmd_15440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 52), 'config_cmd', False)
        # Getting the type of 'ext' (line 408)
        ext_15441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 64), 'ext', False)
        # Getting the type of 'build_dir' (line 408)
        build_dir_15442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 69), 'build_dir', False)
        # Processing the call keyword arguments (line 408)
        kwargs_15443 = {}
        # Getting the type of 'cocache' (line 408)
        cocache_15438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 32), 'cocache', False)
        # Obtaining the member 'check_types' of a type (line 408)
        check_types_15439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 32), cocache_15438, 'check_types')
        # Calling check_types(args, kwargs) (line 408)
        check_types_call_result_15444 = invoke(stypy.reporting.localization.Localization(__file__, 408, 32), check_types_15439, *[config_cmd_15440, ext_15441, build_dir_15442], **kwargs_15443)
        
        # Assigning a type to the variable 'call_assignment_14134' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'call_assignment_14134', check_types_call_result_15444)
        
        # Assigning a Call to a Name (line 408):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_15447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 12), 'int')
        # Processing the call keyword arguments
        kwargs_15448 = {}
        # Getting the type of 'call_assignment_14134' (line 408)
        call_assignment_14134_15445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'call_assignment_14134', False)
        # Obtaining the member '__getitem__' of a type (line 408)
        getitem___15446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 12), call_assignment_14134_15445, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_15449 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___15446, *[int_15447], **kwargs_15448)
        
        # Assigning a type to the variable 'call_assignment_14135' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'call_assignment_14135', getitem___call_result_15449)
        
        # Assigning a Name to a Name (line 408):
        # Getting the type of 'call_assignment_14135' (line 408)
        call_assignment_14135_15450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'call_assignment_14135')
        # Assigning a type to the variable 'moredefs' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'moredefs', call_assignment_14135_15450)
        
        # Assigning a Call to a Name (line 408):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_15453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 12), 'int')
        # Processing the call keyword arguments
        kwargs_15454 = {}
        # Getting the type of 'call_assignment_14134' (line 408)
        call_assignment_14134_15451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'call_assignment_14134', False)
        # Obtaining the member '__getitem__' of a type (line 408)
        getitem___15452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 12), call_assignment_14134_15451, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_15455 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___15452, *[int_15453], **kwargs_15454)
        
        # Assigning a type to the variable 'call_assignment_14136' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'call_assignment_14136', getitem___call_result_15455)
        
        # Assigning a Name to a Name (line 408):
        # Getting the type of 'call_assignment_14136' (line 408)
        call_assignment_14136_15456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'call_assignment_14136')
        # Assigning a type to the variable 'ignored' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 22), 'ignored', call_assignment_14136_15456)
        
        # Assigning a Call to a Name (line 411):
        
        # Assigning a Call to a Name (line 411):
        
        # Call to check_mathlib(...): (line 411)
        # Processing the call arguments (line 411)
        # Getting the type of 'config_cmd' (line 411)
        config_cmd_15458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 37), 'config_cmd', False)
        # Processing the call keyword arguments (line 411)
        kwargs_15459 = {}
        # Getting the type of 'check_mathlib' (line 411)
        check_mathlib_15457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 23), 'check_mathlib', False)
        # Calling check_mathlib(args, kwargs) (line 411)
        check_mathlib_call_result_15460 = invoke(stypy.reporting.localization.Localization(__file__, 411, 23), check_mathlib_15457, *[config_cmd_15458], **kwargs_15459)
        
        # Assigning a type to the variable 'mathlibs' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'mathlibs', check_mathlib_call_result_15460)
        
        # Call to append(...): (line 412)
        # Processing the call arguments (line 412)
        
        # Obtaining an instance of the builtin type 'tuple' (line 412)
        tuple_15463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 412)
        # Adding element type (line 412)
        str_15464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 29), 'str', 'MATHLIB')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 29), tuple_15463, str_15464)
        # Adding element type (line 412)
        
        # Call to join(...): (line 412)
        # Processing the call arguments (line 412)
        # Getting the type of 'mathlibs' (line 412)
        mathlibs_15467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 49), 'mathlibs', False)
        # Processing the call keyword arguments (line 412)
        kwargs_15468 = {}
        str_15465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 40), 'str', ',')
        # Obtaining the member 'join' of a type (line 412)
        join_15466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 40), str_15465, 'join')
        # Calling join(args, kwargs) (line 412)
        join_call_result_15469 = invoke(stypy.reporting.localization.Localization(__file__, 412, 40), join_15466, *[mathlibs_15467], **kwargs_15468)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 29), tuple_15463, join_call_result_15469)
        
        # Processing the call keyword arguments (line 412)
        kwargs_15470 = {}
        # Getting the type of 'moredefs' (line 412)
        moredefs_15461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 12), 'moredefs', False)
        # Obtaining the member 'append' of a type (line 412)
        append_15462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 12), moredefs_15461, 'append')
        # Calling append(args, kwargs) (line 412)
        append_call_result_15471 = invoke(stypy.reporting.localization.Localization(__file__, 412, 12), append_15462, *[tuple_15463], **kwargs_15470)
        
        
        # Call to check_math_capabilities(...): (line 414)
        # Processing the call arguments (line 414)
        # Getting the type of 'config_cmd' (line 414)
        config_cmd_15473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 36), 'config_cmd', False)
        # Getting the type of 'moredefs' (line 414)
        moredefs_15474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 48), 'moredefs', False)
        # Getting the type of 'mathlibs' (line 414)
        mathlibs_15475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 58), 'mathlibs', False)
        # Processing the call keyword arguments (line 414)
        kwargs_15476 = {}
        # Getting the type of 'check_math_capabilities' (line 414)
        check_math_capabilities_15472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'check_math_capabilities', False)
        # Calling check_math_capabilities(args, kwargs) (line 414)
        check_math_capabilities_call_result_15477 = invoke(stypy.reporting.localization.Localization(__file__, 414, 12), check_math_capabilities_15472, *[config_cmd_15473, moredefs_15474, mathlibs_15475], **kwargs_15476)
        
        
        # Call to extend(...): (line 415)
        # Processing the call arguments (line 415)
        
        # Obtaining the type of the subscript
        int_15480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 66), 'int')
        
        # Call to check_ieee_macros(...): (line 415)
        # Processing the call arguments (line 415)
        # Getting the type of 'config_cmd' (line 415)
        config_cmd_15483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 54), 'config_cmd', False)
        # Processing the call keyword arguments (line 415)
        kwargs_15484 = {}
        # Getting the type of 'cocache' (line 415)
        cocache_15481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 28), 'cocache', False)
        # Obtaining the member 'check_ieee_macros' of a type (line 415)
        check_ieee_macros_15482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 28), cocache_15481, 'check_ieee_macros')
        # Calling check_ieee_macros(args, kwargs) (line 415)
        check_ieee_macros_call_result_15485 = invoke(stypy.reporting.localization.Localization(__file__, 415, 28), check_ieee_macros_15482, *[config_cmd_15483], **kwargs_15484)
        
        # Obtaining the member '__getitem__' of a type (line 415)
        getitem___15486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 28), check_ieee_macros_call_result_15485, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 415)
        subscript_call_result_15487 = invoke(stypy.reporting.localization.Localization(__file__, 415, 28), getitem___15486, int_15480)
        
        # Processing the call keyword arguments (line 415)
        kwargs_15488 = {}
        # Getting the type of 'moredefs' (line 415)
        moredefs_15478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 12), 'moredefs', False)
        # Obtaining the member 'extend' of a type (line 415)
        extend_15479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 12), moredefs_15478, 'extend')
        # Calling extend(args, kwargs) (line 415)
        extend_call_result_15489 = invoke(stypy.reporting.localization.Localization(__file__, 415, 12), extend_15479, *[subscript_call_result_15487], **kwargs_15488)
        
        
        # Call to extend(...): (line 416)
        # Processing the call arguments (line 416)
        
        # Obtaining the type of the subscript
        int_15492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 72), 'int')
        
        # Call to check_complex(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'config_cmd' (line 416)
        config_cmd_15495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 50), 'config_cmd', False)
        # Getting the type of 'mathlibs' (line 416)
        mathlibs_15496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 62), 'mathlibs', False)
        # Processing the call keyword arguments (line 416)
        kwargs_15497 = {}
        # Getting the type of 'cocache' (line 416)
        cocache_15493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 28), 'cocache', False)
        # Obtaining the member 'check_complex' of a type (line 416)
        check_complex_15494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 28), cocache_15493, 'check_complex')
        # Calling check_complex(args, kwargs) (line 416)
        check_complex_call_result_15498 = invoke(stypy.reporting.localization.Localization(__file__, 416, 28), check_complex_15494, *[config_cmd_15495, mathlibs_15496], **kwargs_15497)
        
        # Obtaining the member '__getitem__' of a type (line 416)
        getitem___15499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 28), check_complex_call_result_15498, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 416)
        subscript_call_result_15500 = invoke(stypy.reporting.localization.Localization(__file__, 416, 28), getitem___15499, int_15492)
        
        # Processing the call keyword arguments (line 416)
        kwargs_15501 = {}
        # Getting the type of 'moredefs' (line 416)
        moredefs_15490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 12), 'moredefs', False)
        # Obtaining the member 'extend' of a type (line 416)
        extend_15491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 12), moredefs_15490, 'extend')
        # Calling extend(args, kwargs) (line 416)
        extend_call_result_15502 = invoke(stypy.reporting.localization.Localization(__file__, 416, 12), extend_15491, *[subscript_call_result_15500], **kwargs_15501)
        
        
        
        # Call to is_npy_no_signal(...): (line 419)
        # Processing the call keyword arguments (line 419)
        kwargs_15504 = {}
        # Getting the type of 'is_npy_no_signal' (line 419)
        is_npy_no_signal_15503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 15), 'is_npy_no_signal', False)
        # Calling is_npy_no_signal(args, kwargs) (line 419)
        is_npy_no_signal_call_result_15505 = invoke(stypy.reporting.localization.Localization(__file__, 419, 15), is_npy_no_signal_15503, *[], **kwargs_15504)
        
        # Testing the type of an if condition (line 419)
        if_condition_15506 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 419, 12), is_npy_no_signal_call_result_15505)
        # Assigning a type to the variable 'if_condition_15506' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 12), 'if_condition_15506', if_condition_15506)
        # SSA begins for if statement (line 419)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 420)
        # Processing the call arguments (line 420)
        str_15509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 32), 'str', '__NPY_PRIVATE_NO_SIGNAL')
        # Processing the call keyword arguments (line 420)
        kwargs_15510 = {}
        # Getting the type of 'moredefs' (line 420)
        moredefs_15507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 16), 'moredefs', False)
        # Obtaining the member 'append' of a type (line 420)
        append_15508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 16), moredefs_15507, 'append')
        # Calling append(args, kwargs) (line 420)
        append_call_result_15511 = invoke(stypy.reporting.localization.Localization(__file__, 420, 16), append_15508, *[str_15509], **kwargs_15510)
        
        # SSA join for if statement (line 419)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'sys' (line 423)
        sys_15512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 15), 'sys')
        # Obtaining the member 'platform' of a type (line 423)
        platform_15513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 15), sys_15512, 'platform')
        str_15514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 31), 'str', 'win32')
        # Applying the binary operator '==' (line 423)
        result_eq_15515 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 15), '==', platform_15513, str_15514)
        
        
        # Getting the type of 'os' (line 423)
        os_15516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 42), 'os')
        # Obtaining the member 'name' of a type (line 423)
        name_15517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 42), os_15516, 'name')
        str_15518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 53), 'str', 'nt')
        # Applying the binary operator '==' (line 423)
        result_eq_15519 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 42), '==', name_15517, str_15518)
        
        # Applying the binary operator 'or' (line 423)
        result_or_keyword_15520 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 15), 'or', result_eq_15515, result_eq_15519)
        
        # Testing the type of an if condition (line 423)
        if_condition_15521 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 423, 12), result_or_keyword_15520)
        # Assigning a type to the variable 'if_condition_15521' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'if_condition_15521', if_condition_15521)
        # SSA begins for if statement (line 423)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to win32_checks(...): (line 424)
        # Processing the call arguments (line 424)
        # Getting the type of 'moredefs' (line 424)
        moredefs_15523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 29), 'moredefs', False)
        # Processing the call keyword arguments (line 424)
        kwargs_15524 = {}
        # Getting the type of 'win32_checks' (line 424)
        win32_checks_15522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 16), 'win32_checks', False)
        # Calling win32_checks(args, kwargs) (line 424)
        win32_checks_call_result_15525 = invoke(stypy.reporting.localization.Localization(__file__, 424, 16), win32_checks_15522, *[moredefs_15523], **kwargs_15524)
        
        # SSA join for if statement (line 423)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 427)
        # Processing the call arguments (line 427)
        
        # Obtaining an instance of the builtin type 'tuple' (line 427)
        tuple_15528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 427)
        # Adding element type (line 427)
        str_15529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 29), 'str', 'NPY_RESTRICT')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 29), tuple_15528, str_15529)
        # Adding element type (line 427)
        
        # Call to check_restrict(...): (line 427)
        # Processing the call keyword arguments (line 427)
        kwargs_15532 = {}
        # Getting the type of 'config_cmd' (line 427)
        config_cmd_15530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 45), 'config_cmd', False)
        # Obtaining the member 'check_restrict' of a type (line 427)
        check_restrict_15531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 45), config_cmd_15530, 'check_restrict')
        # Calling check_restrict(args, kwargs) (line 427)
        check_restrict_call_result_15533 = invoke(stypy.reporting.localization.Localization(__file__, 427, 45), check_restrict_15531, *[], **kwargs_15532)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 29), tuple_15528, check_restrict_call_result_15533)
        
        # Processing the call keyword arguments (line 427)
        kwargs_15534 = {}
        # Getting the type of 'moredefs' (line 427)
        moredefs_15526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'moredefs', False)
        # Obtaining the member 'append' of a type (line 427)
        append_15527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 12), moredefs_15526, 'append')
        # Calling append(args, kwargs) (line 427)
        append_call_result_15535 = invoke(stypy.reporting.localization.Localization(__file__, 427, 12), append_15527, *[tuple_15528], **kwargs_15534)
        
        
        # Assigning a Call to a Name (line 430):
        
        # Assigning a Call to a Name (line 430):
        
        # Call to check_inline(...): (line 430)
        # Processing the call keyword arguments (line 430)
        kwargs_15538 = {}
        # Getting the type of 'config_cmd' (line 430)
        config_cmd_15536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 21), 'config_cmd', False)
        # Obtaining the member 'check_inline' of a type (line 430)
        check_inline_15537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 21), config_cmd_15536, 'check_inline')
        # Calling check_inline(args, kwargs) (line 430)
        check_inline_call_result_15539 = invoke(stypy.reporting.localization.Localization(__file__, 430, 21), check_inline_15537, *[], **kwargs_15538)
        
        # Assigning a type to the variable 'inline' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'inline', check_inline_call_result_15539)
        
        
        
        # Call to check_decl(...): (line 433)
        # Processing the call arguments (line 433)
        str_15542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 41), 'str', 'Py_UNICODE_WIDE')
        # Processing the call keyword arguments (line 433)
        
        # Obtaining an instance of the builtin type 'list' (line 433)
        list_15543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 68), 'list')
        # Adding type elements to the builtin type 'list' instance (line 433)
        # Adding element type (line 433)
        str_15544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 69), 'str', 'Python.h')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 68), list_15543, str_15544)
        
        keyword_15545 = list_15543
        kwargs_15546 = {'headers': keyword_15545}
        # Getting the type of 'config_cmd' (line 433)
        config_cmd_15540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 19), 'config_cmd', False)
        # Obtaining the member 'check_decl' of a type (line 433)
        check_decl_15541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 19), config_cmd_15540, 'check_decl')
        # Calling check_decl(args, kwargs) (line 433)
        check_decl_call_result_15547 = invoke(stypy.reporting.localization.Localization(__file__, 433, 19), check_decl_15541, *[str_15542], **kwargs_15546)
        
        # Applying the 'not' unary operator (line 433)
        result_not__15548 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 15), 'not', check_decl_call_result_15547)
        
        # Testing the type of an if condition (line 433)
        if_condition_15549 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 433, 12), result_not__15548)
        # Assigning a type to the variable 'if_condition_15549' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 12), 'if_condition_15549', if_condition_15549)
        # SSA begins for if statement (line 433)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 434):
        
        # Assigning a Name to a Name (line 434):
        # Getting the type of 'True' (line 434)
        True_15550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 42), 'True')
        # Assigning a type to the variable 'PYTHON_HAS_UNICODE_WIDE' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 16), 'PYTHON_HAS_UNICODE_WIDE', True_15550)
        # SSA branch for the else part of an if statement (line 433)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 436):
        
        # Assigning a Name to a Name (line 436):
        # Getting the type of 'False' (line 436)
        False_15551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 42), 'False')
        # Assigning a type to the variable 'PYTHON_HAS_UNICODE_WIDE' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'PYTHON_HAS_UNICODE_WIDE', False_15551)
        # SSA join for if statement (line 433)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'NPY_RELAXED_STRIDES_CHECKING' (line 438)
        NPY_RELAXED_STRIDES_CHECKING_15552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 15), 'NPY_RELAXED_STRIDES_CHECKING')
        # Testing the type of an if condition (line 438)
        if_condition_15553 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 438, 12), NPY_RELAXED_STRIDES_CHECKING_15552)
        # Assigning a type to the variable 'if_condition_15553' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'if_condition_15553', if_condition_15553)
        # SSA begins for if statement (line 438)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 439)
        # Processing the call arguments (line 439)
        
        # Obtaining an instance of the builtin type 'tuple' (line 439)
        tuple_15556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 439)
        # Adding element type (line 439)
        str_15557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 33), 'str', 'NPY_RELAXED_STRIDES_CHECKING')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 33), tuple_15556, str_15557)
        # Adding element type (line 439)
        int_15558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 33), tuple_15556, int_15558)
        
        # Processing the call keyword arguments (line 439)
        kwargs_15559 = {}
        # Getting the type of 'moredefs' (line 439)
        moredefs_15554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 16), 'moredefs', False)
        # Obtaining the member 'append' of a type (line 439)
        append_15555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 16), moredefs_15554, 'append')
        # Calling append(args, kwargs) (line 439)
        append_call_result_15560 = invoke(stypy.reporting.localization.Localization(__file__, 439, 16), append_15555, *[tuple_15556], **kwargs_15559)
        
        # SSA join for if statement (line 438)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'sys' (line 442)
        sys_15561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 15), 'sys')
        # Obtaining the member 'platform' of a type (line 442)
        platform_15562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 15), sys_15561, 'platform')
        str_15563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 31), 'str', 'darwin')
        # Applying the binary operator '!=' (line 442)
        result_ne_15564 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 15), '!=', platform_15562, str_15563)
        
        # Testing the type of an if condition (line 442)
        if_condition_15565 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 442, 12), result_ne_15564)
        # Assigning a type to the variable 'if_condition_15565' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'if_condition_15565', if_condition_15565)
        # SSA begins for if statement (line 442)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 443):
        
        # Assigning a Call to a Name (line 443):
        
        # Call to check_long_double_representation(...): (line 443)
        # Processing the call arguments (line 443)
        # Getting the type of 'config_cmd' (line 443)
        config_cmd_15567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 55), 'config_cmd', False)
        # Processing the call keyword arguments (line 443)
        kwargs_15568 = {}
        # Getting the type of 'check_long_double_representation' (line 443)
        check_long_double_representation_15566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 22), 'check_long_double_representation', False)
        # Calling check_long_double_representation(args, kwargs) (line 443)
        check_long_double_representation_call_result_15569 = invoke(stypy.reporting.localization.Localization(__file__, 443, 22), check_long_double_representation_15566, *[config_cmd_15567], **kwargs_15568)
        
        # Assigning a type to the variable 'rep' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 16), 'rep', check_long_double_representation_call_result_15569)
        
        
        # Getting the type of 'rep' (line 444)
        rep_15570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 19), 'rep')
        
        # Obtaining an instance of the builtin type 'list' (line 444)
        list_15571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 444)
        # Adding element type (line 444)
        str_15572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 27), 'str', 'INTEL_EXTENDED_12_BYTES_LE')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 26), list_15571, str_15572)
        # Adding element type (line 444)
        str_15573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 27), 'str', 'INTEL_EXTENDED_16_BYTES_LE')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 26), list_15571, str_15573)
        # Adding element type (line 444)
        str_15574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 27), 'str', 'MOTOROLA_EXTENDED_12_BYTES_BE')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 26), list_15571, str_15574)
        # Adding element type (line 444)
        str_15575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 27), 'str', 'IEEE_QUAD_LE')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 26), list_15571, str_15575)
        # Adding element type (line 444)
        str_15576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 43), 'str', 'IEEE_QUAD_BE')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 26), list_15571, str_15576)
        # Adding element type (line 444)
        str_15577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 27), 'str', 'IEEE_DOUBLE_LE')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 26), list_15571, str_15577)
        # Adding element type (line 444)
        str_15578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 45), 'str', 'IEEE_DOUBLE_BE')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 26), list_15571, str_15578)
        # Adding element type (line 444)
        str_15579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 27), 'str', 'DOUBLE_DOUBLE_BE')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 26), list_15571, str_15579)
        # Adding element type (line 444)
        str_15580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 47), 'str', 'DOUBLE_DOUBLE_LE')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 26), list_15571, str_15580)
        
        # Applying the binary operator 'in' (line 444)
        result_contains_15581 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 19), 'in', rep_15570, list_15571)
        
        # Testing the type of an if condition (line 444)
        if_condition_15582 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 444, 16), result_contains_15581)
        # Assigning a type to the variable 'if_condition_15582' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 16), 'if_condition_15582', if_condition_15582)
        # SSA begins for if statement (line 444)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 450)
        # Processing the call arguments (line 450)
        
        # Obtaining an instance of the builtin type 'tuple' (line 450)
        tuple_15585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 450)
        # Adding element type (line 450)
        str_15586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 37), 'str', 'HAVE_LDOUBLE_%s')
        # Getting the type of 'rep' (line 450)
        rep_15587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 57), 'rep', False)
        # Applying the binary operator '%' (line 450)
        result_mod_15588 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 37), '%', str_15586, rep_15587)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 450, 37), tuple_15585, result_mod_15588)
        # Adding element type (line 450)
        int_15589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 450, 37), tuple_15585, int_15589)
        
        # Processing the call keyword arguments (line 450)
        kwargs_15590 = {}
        # Getting the type of 'moredefs' (line 450)
        moredefs_15583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 20), 'moredefs', False)
        # Obtaining the member 'append' of a type (line 450)
        append_15584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 20), moredefs_15583, 'append')
        # Calling append(args, kwargs) (line 450)
        append_call_result_15591 = invoke(stypy.reporting.localization.Localization(__file__, 450, 20), append_15584, *[tuple_15585], **kwargs_15590)
        
        # SSA branch for the else part of an if statement (line 444)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 452)
        # Processing the call arguments (line 452)
        str_15593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 37), 'str', 'Unrecognized long double format: %s')
        # Getting the type of 'rep' (line 452)
        rep_15594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 77), 'rep', False)
        # Applying the binary operator '%' (line 452)
        result_mod_15595 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 37), '%', str_15593, rep_15594)
        
        # Processing the call keyword arguments (line 452)
        kwargs_15596 = {}
        # Getting the type of 'ValueError' (line 452)
        ValueError_15592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 452)
        ValueError_call_result_15597 = invoke(stypy.reporting.localization.Localization(__file__, 452, 26), ValueError_15592, *[result_mod_15595], **kwargs_15596)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 452, 20), ValueError_call_result_15597, 'raise parameter', BaseException)
        # SSA join for if statement (line 444)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 442)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        int_15598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 32), 'int')
        # Getting the type of 'sys' (line 455)
        sys_15599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 15), 'sys')
        # Obtaining the member 'version_info' of a type (line 455)
        version_info_15600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 15), sys_15599, 'version_info')
        # Obtaining the member '__getitem__' of a type (line 455)
        getitem___15601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 15), version_info_15600, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 455)
        subscript_call_result_15602 = invoke(stypy.reporting.localization.Localization(__file__, 455, 15), getitem___15601, int_15598)
        
        int_15603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 38), 'int')
        # Applying the binary operator '==' (line 455)
        result_eq_15604 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 15), '==', subscript_call_result_15602, int_15603)
        
        # Testing the type of an if condition (line 455)
        if_condition_15605 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 455, 12), result_eq_15604)
        # Assigning a type to the variable 'if_condition_15605' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 12), 'if_condition_15605', if_condition_15605)
        # SSA begins for if statement (line 455)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 456)
        # Processing the call arguments (line 456)
        
        # Obtaining an instance of the builtin type 'tuple' (line 456)
        tuple_15608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 456)
        # Adding element type (line 456)
        str_15609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 33), 'str', 'NPY_PY3K')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 456, 33), tuple_15608, str_15609)
        # Adding element type (line 456)
        int_15610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 456, 33), tuple_15608, int_15610)
        
        # Processing the call keyword arguments (line 456)
        kwargs_15611 = {}
        # Getting the type of 'moredefs' (line 456)
        moredefs_15606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'moredefs', False)
        # Obtaining the member 'append' of a type (line 456)
        append_15607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 16), moredefs_15606, 'append')
        # Calling append(args, kwargs) (line 456)
        append_call_result_15612 = invoke(stypy.reporting.localization.Localization(__file__, 456, 16), append_15607, *[tuple_15608], **kwargs_15611)
        
        # SSA join for if statement (line 455)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 459):
        
        # Assigning a Call to a Name (line 459):
        
        # Call to open(...): (line 459)
        # Processing the call arguments (line 459)
        # Getting the type of 'target' (line 459)
        target_15614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 28), 'target', False)
        str_15615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 36), 'str', 'w')
        # Processing the call keyword arguments (line 459)
        kwargs_15616 = {}
        # Getting the type of 'open' (line 459)
        open_15613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 23), 'open', False)
        # Calling open(args, kwargs) (line 459)
        open_call_result_15617 = invoke(stypy.reporting.localization.Localization(__file__, 459, 23), open_15613, *[target_15614, str_15615], **kwargs_15616)
        
        # Assigning a type to the variable 'target_f' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'target_f', open_call_result_15617)
        
        # Getting the type of 'moredefs' (line 460)
        moredefs_15618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 21), 'moredefs')
        # Testing the type of a for loop iterable (line 460)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 460, 12), moredefs_15618)
        # Getting the type of the for loop variable (line 460)
        for_loop_var_15619 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 460, 12), moredefs_15618)
        # Assigning a type to the variable 'd' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'd', for_loop_var_15619)
        # SSA begins for a for statement (line 460)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 461)
        # Getting the type of 'str' (line 461)
        str_15620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 33), 'str')
        # Getting the type of 'd' (line 461)
        d_15621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 30), 'd')
        
        (may_be_15622, more_types_in_union_15623) = may_be_subtype(str_15620, d_15621)

        if may_be_15622:

            if more_types_in_union_15623:
                # Runtime conditional SSA (line 461)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'd' (line 461)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 16), 'd', remove_not_subtype_from_union(d_15621, str))
            
            # Call to write(...): (line 462)
            # Processing the call arguments (line 462)
            str_15626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 35), 'str', '#define %s\n')
            # Getting the type of 'd' (line 462)
            d_15627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 53), 'd', False)
            # Applying the binary operator '%' (line 462)
            result_mod_15628 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 35), '%', str_15626, d_15627)
            
            # Processing the call keyword arguments (line 462)
            kwargs_15629 = {}
            # Getting the type of 'target_f' (line 462)
            target_f_15624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 20), 'target_f', False)
            # Obtaining the member 'write' of a type (line 462)
            write_15625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 20), target_f_15624, 'write')
            # Calling write(args, kwargs) (line 462)
            write_call_result_15630 = invoke(stypy.reporting.localization.Localization(__file__, 462, 20), write_15625, *[result_mod_15628], **kwargs_15629)
            

            if more_types_in_union_15623:
                # Runtime conditional SSA for else branch (line 461)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_15622) or more_types_in_union_15623):
            # Assigning a type to the variable 'd' (line 461)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 16), 'd', remove_subtype_from_union(d_15621, str))
            
            # Call to write(...): (line 464)
            # Processing the call arguments (line 464)
            str_15633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 35), 'str', '#define %s %s\n')
            
            # Obtaining an instance of the builtin type 'tuple' (line 464)
            tuple_15634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 56), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 464)
            # Adding element type (line 464)
            
            # Obtaining the type of the subscript
            int_15635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 58), 'int')
            # Getting the type of 'd' (line 464)
            d_15636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 56), 'd', False)
            # Obtaining the member '__getitem__' of a type (line 464)
            getitem___15637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 56), d_15636, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 464)
            subscript_call_result_15638 = invoke(stypy.reporting.localization.Localization(__file__, 464, 56), getitem___15637, int_15635)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 56), tuple_15634, subscript_call_result_15638)
            # Adding element type (line 464)
            
            # Obtaining the type of the subscript
            int_15639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 64), 'int')
            # Getting the type of 'd' (line 464)
            d_15640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 62), 'd', False)
            # Obtaining the member '__getitem__' of a type (line 464)
            getitem___15641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 62), d_15640, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 464)
            subscript_call_result_15642 = invoke(stypy.reporting.localization.Localization(__file__, 464, 62), getitem___15641, int_15639)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 56), tuple_15634, subscript_call_result_15642)
            
            # Applying the binary operator '%' (line 464)
            result_mod_15643 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 35), '%', str_15633, tuple_15634)
            
            # Processing the call keyword arguments (line 464)
            kwargs_15644 = {}
            # Getting the type of 'target_f' (line 464)
            target_f_15631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 20), 'target_f', False)
            # Obtaining the member 'write' of a type (line 464)
            write_15632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 20), target_f_15631, 'write')
            # Calling write(args, kwargs) (line 464)
            write_call_result_15645 = invoke(stypy.reporting.localization.Localization(__file__, 464, 20), write_15632, *[result_mod_15643], **kwargs_15644)
            

            if (may_be_15622 and more_types_in_union_15623):
                # SSA join for if statement (line 461)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 467)
        # Processing the call arguments (line 467)
        str_15648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 27), 'str', '#ifndef __cplusplus\n')
        # Processing the call keyword arguments (line 467)
        kwargs_15649 = {}
        # Getting the type of 'target_f' (line 467)
        target_f_15646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'target_f', False)
        # Obtaining the member 'write' of a type (line 467)
        write_15647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 12), target_f_15646, 'write')
        # Calling write(args, kwargs) (line 467)
        write_call_result_15650 = invoke(stypy.reporting.localization.Localization(__file__, 467, 12), write_15647, *[str_15648], **kwargs_15649)
        
        
        
        # Getting the type of 'inline' (line 468)
        inline_15651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 15), 'inline')
        str_15652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 25), 'str', 'inline')
        # Applying the binary operator '==' (line 468)
        result_eq_15653 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 15), '==', inline_15651, str_15652)
        
        # Testing the type of an if condition (line 468)
        if_condition_15654 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 468, 12), result_eq_15653)
        # Assigning a type to the variable 'if_condition_15654' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), 'if_condition_15654', if_condition_15654)
        # SSA begins for if statement (line 468)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 469)
        # Processing the call arguments (line 469)
        str_15657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 31), 'str', '/* #undef inline */\n')
        # Processing the call keyword arguments (line 469)
        kwargs_15658 = {}
        # Getting the type of 'target_f' (line 469)
        target_f_15655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 16), 'target_f', False)
        # Obtaining the member 'write' of a type (line 469)
        write_15656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 16), target_f_15655, 'write')
        # Calling write(args, kwargs) (line 469)
        write_call_result_15659 = invoke(stypy.reporting.localization.Localization(__file__, 469, 16), write_15656, *[str_15657], **kwargs_15658)
        
        # SSA branch for the else part of an if statement (line 468)
        module_type_store.open_ssa_branch('else')
        
        # Call to write(...): (line 471)
        # Processing the call arguments (line 471)
        str_15662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 31), 'str', '#define inline %s\n')
        # Getting the type of 'inline' (line 471)
        inline_15663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 55), 'inline', False)
        # Applying the binary operator '%' (line 471)
        result_mod_15664 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 31), '%', str_15662, inline_15663)
        
        # Processing the call keyword arguments (line 471)
        kwargs_15665 = {}
        # Getting the type of 'target_f' (line 471)
        target_f_15660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 16), 'target_f', False)
        # Obtaining the member 'write' of a type (line 471)
        write_15661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 16), target_f_15660, 'write')
        # Calling write(args, kwargs) (line 471)
        write_call_result_15666 = invoke(stypy.reporting.localization.Localization(__file__, 471, 16), write_15661, *[result_mod_15664], **kwargs_15665)
        
        # SSA join for if statement (line 468)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 472)
        # Processing the call arguments (line 472)
        str_15669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 27), 'str', '#endif\n')
        # Processing the call keyword arguments (line 472)
        kwargs_15670 = {}
        # Getting the type of 'target_f' (line 472)
        target_f_15667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 12), 'target_f', False)
        # Obtaining the member 'write' of a type (line 472)
        write_15668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 12), target_f_15667, 'write')
        # Calling write(args, kwargs) (line 472)
        write_call_result_15671 = invoke(stypy.reporting.localization.Localization(__file__, 472, 12), write_15668, *[str_15669], **kwargs_15670)
        
        
        # Call to write(...): (line 476)
        # Processing the call arguments (line 476)
        str_15674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, (-1)), 'str', '\n#ifndef _NPY_NPY_CONFIG_H_\n#error config.h should never be included directly, include npy_config.h instead\n#endif\n')
        # Processing the call keyword arguments (line 476)
        kwargs_15675 = {}
        # Getting the type of 'target_f' (line 476)
        target_f_15672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), 'target_f', False)
        # Obtaining the member 'write' of a type (line 476)
        write_15673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 12), target_f_15672, 'write')
        # Calling write(args, kwargs) (line 476)
        write_call_result_15676 = invoke(stypy.reporting.localization.Localization(__file__, 476, 12), write_15673, *[str_15674], **kwargs_15675)
        
        
        # Call to close(...): (line 482)
        # Processing the call keyword arguments (line 482)
        kwargs_15679 = {}
        # Getting the type of 'target_f' (line 482)
        target_f_15677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 12), 'target_f', False)
        # Obtaining the member 'close' of a type (line 482)
        close_15678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 12), target_f_15677, 'close')
        # Calling close(args, kwargs) (line 482)
        close_call_result_15680 = invoke(stypy.reporting.localization.Localization(__file__, 482, 12), close_15678, *[], **kwargs_15679)
        
        
        # Call to print(...): (line 483)
        # Processing the call arguments (line 483)
        str_15682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 18), 'str', 'File:')
        # Getting the type of 'target' (line 483)
        target_15683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 27), 'target', False)
        # Processing the call keyword arguments (line 483)
        kwargs_15684 = {}
        # Getting the type of 'print' (line 483)
        print_15681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'print', False)
        # Calling print(args, kwargs) (line 483)
        print_call_result_15685 = invoke(stypy.reporting.localization.Localization(__file__, 483, 12), print_15681, *[str_15682, target_15683], **kwargs_15684)
        
        
        # Assigning a Call to a Name (line 484):
        
        # Assigning a Call to a Name (line 484):
        
        # Call to open(...): (line 484)
        # Processing the call arguments (line 484)
        # Getting the type of 'target' (line 484)
        target_15687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 28), 'target', False)
        # Processing the call keyword arguments (line 484)
        kwargs_15688 = {}
        # Getting the type of 'open' (line 484)
        open_15686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 23), 'open', False)
        # Calling open(args, kwargs) (line 484)
        open_call_result_15689 = invoke(stypy.reporting.localization.Localization(__file__, 484, 23), open_15686, *[target_15687], **kwargs_15688)
        
        # Assigning a type to the variable 'target_f' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 12), 'target_f', open_call_result_15689)
        
        # Call to print(...): (line 485)
        # Processing the call arguments (line 485)
        
        # Call to read(...): (line 485)
        # Processing the call keyword arguments (line 485)
        kwargs_15693 = {}
        # Getting the type of 'target_f' (line 485)
        target_f_15691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 18), 'target_f', False)
        # Obtaining the member 'read' of a type (line 485)
        read_15692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 18), target_f_15691, 'read')
        # Calling read(args, kwargs) (line 485)
        read_call_result_15694 = invoke(stypy.reporting.localization.Localization(__file__, 485, 18), read_15692, *[], **kwargs_15693)
        
        # Processing the call keyword arguments (line 485)
        kwargs_15695 = {}
        # Getting the type of 'print' (line 485)
        print_15690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 12), 'print', False)
        # Calling print(args, kwargs) (line 485)
        print_call_result_15696 = invoke(stypy.reporting.localization.Localization(__file__, 485, 12), print_15690, *[read_call_result_15694], **kwargs_15695)
        
        
        # Call to close(...): (line 486)
        # Processing the call keyword arguments (line 486)
        kwargs_15699 = {}
        # Getting the type of 'target_f' (line 486)
        target_f_15697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 12), 'target_f', False)
        # Obtaining the member 'close' of a type (line 486)
        close_15698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 12), target_f_15697, 'close')
        # Calling close(args, kwargs) (line 486)
        close_call_result_15700 = invoke(stypy.reporting.localization.Localization(__file__, 486, 12), close_15698, *[], **kwargs_15699)
        
        
        # Call to print(...): (line 487)
        # Processing the call arguments (line 487)
        str_15702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 18), 'str', 'EOF')
        # Processing the call keyword arguments (line 487)
        kwargs_15703 = {}
        # Getting the type of 'print' (line 487)
        print_15701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'print', False)
        # Calling print(args, kwargs) (line 487)
        print_call_result_15704 = invoke(stypy.reporting.localization.Localization(__file__, 487, 12), print_15701, *[str_15702], **kwargs_15703)
        
        # SSA branch for the else part of an if statement (line 403)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 489):
        
        # Assigning a List to a Name (line 489):
        
        # Obtaining an instance of the builtin type 'list' (line 489)
        list_15705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 489)
        
        # Assigning a type to the variable 'mathlibs' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 12), 'mathlibs', list_15705)
        
        # Assigning a Call to a Name (line 490):
        
        # Assigning a Call to a Name (line 490):
        
        # Call to open(...): (line 490)
        # Processing the call arguments (line 490)
        # Getting the type of 'target' (line 490)
        target_15707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 28), 'target', False)
        # Processing the call keyword arguments (line 490)
        kwargs_15708 = {}
        # Getting the type of 'open' (line 490)
        open_15706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 23), 'open', False)
        # Calling open(args, kwargs) (line 490)
        open_call_result_15709 = invoke(stypy.reporting.localization.Localization(__file__, 490, 23), open_15706, *[target_15707], **kwargs_15708)
        
        # Assigning a type to the variable 'target_f' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'target_f', open_call_result_15709)
        
        # Getting the type of 'target_f' (line 491)
        target_f_15710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 24), 'target_f')
        # Testing the type of a for loop iterable (line 491)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 491, 12), target_f_15710)
        # Getting the type of the for loop variable (line 491)
        for_loop_var_15711 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 491, 12), target_f_15710)
        # Assigning a type to the variable 'line' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'line', for_loop_var_15711)
        # SSA begins for a for statement (line 491)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Str to a Name (line 492):
        
        # Assigning a Str to a Name (line 492):
        str_15712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 20), 'str', '#define MATHLIB')
        # Assigning a type to the variable 's' (line 492)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 16), 's', str_15712)
        
        
        # Call to startswith(...): (line 493)
        # Processing the call arguments (line 493)
        # Getting the type of 's' (line 493)
        s_15715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 35), 's', False)
        # Processing the call keyword arguments (line 493)
        kwargs_15716 = {}
        # Getting the type of 'line' (line 493)
        line_15713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 19), 'line', False)
        # Obtaining the member 'startswith' of a type (line 493)
        startswith_15714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 19), line_15713, 'startswith')
        # Calling startswith(args, kwargs) (line 493)
        startswith_call_result_15717 = invoke(stypy.reporting.localization.Localization(__file__, 493, 19), startswith_15714, *[s_15715], **kwargs_15716)
        
        # Testing the type of an if condition (line 493)
        if_condition_15718 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 493, 16), startswith_call_result_15717)
        # Assigning a type to the variable 'if_condition_15718' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 16), 'if_condition_15718', if_condition_15718)
        # SSA begins for if statement (line 493)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 494):
        
        # Assigning a Call to a Name (line 494):
        
        # Call to strip(...): (line 494)
        # Processing the call keyword arguments (line 494)
        kwargs_15728 = {}
        
        # Obtaining the type of the subscript
        
        # Call to len(...): (line 494)
        # Processing the call arguments (line 494)
        # Getting the type of 's' (line 494)
        s_15720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 37), 's', False)
        # Processing the call keyword arguments (line 494)
        kwargs_15721 = {}
        # Getting the type of 'len' (line 494)
        len_15719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 33), 'len', False)
        # Calling len(args, kwargs) (line 494)
        len_call_result_15722 = invoke(stypy.reporting.localization.Localization(__file__, 494, 33), len_15719, *[s_15720], **kwargs_15721)
        
        slice_15723 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 494, 28), len_call_result_15722, None, None)
        # Getting the type of 'line' (line 494)
        line_15724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 28), 'line', False)
        # Obtaining the member '__getitem__' of a type (line 494)
        getitem___15725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 28), line_15724, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 494)
        subscript_call_result_15726 = invoke(stypy.reporting.localization.Localization(__file__, 494, 28), getitem___15725, slice_15723)
        
        # Obtaining the member 'strip' of a type (line 494)
        strip_15727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 28), subscript_call_result_15726, 'strip')
        # Calling strip(args, kwargs) (line 494)
        strip_call_result_15729 = invoke(stypy.reporting.localization.Localization(__file__, 494, 28), strip_15727, *[], **kwargs_15728)
        
        # Assigning a type to the variable 'value' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 20), 'value', strip_call_result_15729)
        
        # Getting the type of 'value' (line 495)
        value_15730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 23), 'value')
        # Testing the type of an if condition (line 495)
        if_condition_15731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 495, 20), value_15730)
        # Assigning a type to the variable 'if_condition_15731' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 20), 'if_condition_15731', if_condition_15731)
        # SSA begins for if statement (line 495)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 496)
        # Processing the call arguments (line 496)
        
        # Call to split(...): (line 496)
        # Processing the call arguments (line 496)
        str_15736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 52), 'str', ',')
        # Processing the call keyword arguments (line 496)
        kwargs_15737 = {}
        # Getting the type of 'value' (line 496)
        value_15734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 40), 'value', False)
        # Obtaining the member 'split' of a type (line 496)
        split_15735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 40), value_15734, 'split')
        # Calling split(args, kwargs) (line 496)
        split_call_result_15738 = invoke(stypy.reporting.localization.Localization(__file__, 496, 40), split_15735, *[str_15736], **kwargs_15737)
        
        # Processing the call keyword arguments (line 496)
        kwargs_15739 = {}
        # Getting the type of 'mathlibs' (line 496)
        mathlibs_15732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 24), 'mathlibs', False)
        # Obtaining the member 'extend' of a type (line 496)
        extend_15733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 24), mathlibs_15732, 'extend')
        # Calling extend(args, kwargs) (line 496)
        extend_call_result_15740 = invoke(stypy.reporting.localization.Localization(__file__, 496, 24), extend_15733, *[split_call_result_15738], **kwargs_15739)
        
        # SSA join for if statement (line 495)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 493)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to close(...): (line 497)
        # Processing the call keyword arguments (line 497)
        kwargs_15743 = {}
        # Getting the type of 'target_f' (line 497)
        target_f_15741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 12), 'target_f', False)
        # Obtaining the member 'close' of a type (line 497)
        close_15742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 12), target_f_15741, 'close')
        # Calling close(args, kwargs) (line 497)
        close_call_result_15744 = invoke(stypy.reporting.localization.Localization(__file__, 497, 12), close_15742, *[], **kwargs_15743)
        
        # SSA join for if statement (line 403)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 502)
        str_15745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 24), 'str', 'libraries')
        # Getting the type of 'ext' (line 502)
        ext_15746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 19), 'ext')
        
        (may_be_15747, more_types_in_union_15748) = may_provide_member(str_15745, ext_15746)

        if may_be_15747:

            if more_types_in_union_15748:
                # Runtime conditional SSA (line 502)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'ext' (line 502)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'ext', remove_not_member_provider_from_union(ext_15746, 'libraries'))
            
            # Call to extend(...): (line 503)
            # Processing the call arguments (line 503)
            # Getting the type of 'mathlibs' (line 503)
            mathlibs_15752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 33), 'mathlibs', False)
            # Processing the call keyword arguments (line 503)
            kwargs_15753 = {}
            # Getting the type of 'ext' (line 503)
            ext_15749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 12), 'ext', False)
            # Obtaining the member 'libraries' of a type (line 503)
            libraries_15750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 12), ext_15749, 'libraries')
            # Obtaining the member 'extend' of a type (line 503)
            extend_15751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 12), libraries_15750, 'extend')
            # Calling extend(args, kwargs) (line 503)
            extend_call_result_15754 = invoke(stypy.reporting.localization.Localization(__file__, 503, 12), extend_15751, *[mathlibs_15752], **kwargs_15753)
            

            if more_types_in_union_15748:
                # SSA join for if statement (line 502)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 505):
        
        # Assigning a Call to a Name (line 505):
        
        # Call to dirname(...): (line 505)
        # Processing the call arguments (line 505)
        # Getting the type of 'target' (line 505)
        target_15758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 35), 'target', False)
        # Processing the call keyword arguments (line 505)
        kwargs_15759 = {}
        # Getting the type of 'os' (line 505)
        os_15755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 505)
        path_15756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 19), os_15755, 'path')
        # Obtaining the member 'dirname' of a type (line 505)
        dirname_15757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 19), path_15756, 'dirname')
        # Calling dirname(args, kwargs) (line 505)
        dirname_call_result_15760 = invoke(stypy.reporting.localization.Localization(__file__, 505, 19), dirname_15757, *[target_15758], **kwargs_15759)
        
        # Assigning a type to the variable 'incl_dir' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'incl_dir', dirname_call_result_15760)
        
        
        # Getting the type of 'incl_dir' (line 506)
        incl_dir_15761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 11), 'incl_dir')
        # Getting the type of 'config' (line 506)
        config_15762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 27), 'config')
        # Obtaining the member 'numpy_include_dirs' of a type (line 506)
        numpy_include_dirs_15763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 27), config_15762, 'numpy_include_dirs')
        # Applying the binary operator 'notin' (line 506)
        result_contains_15764 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 11), 'notin', incl_dir_15761, numpy_include_dirs_15763)
        
        # Testing the type of an if condition (line 506)
        if_condition_15765 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 506, 8), result_contains_15764)
        # Assigning a type to the variable 'if_condition_15765' (line 506)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'if_condition_15765', if_condition_15765)
        # SSA begins for if statement (line 506)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 507)
        # Processing the call arguments (line 507)
        # Getting the type of 'incl_dir' (line 507)
        incl_dir_15769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 45), 'incl_dir', False)
        # Processing the call keyword arguments (line 507)
        kwargs_15770 = {}
        # Getting the type of 'config' (line 507)
        config_15766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 'config', False)
        # Obtaining the member 'numpy_include_dirs' of a type (line 507)
        numpy_include_dirs_15767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 12), config_15766, 'numpy_include_dirs')
        # Obtaining the member 'append' of a type (line 507)
        append_15768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 12), numpy_include_dirs_15767, 'append')
        # Calling append(args, kwargs) (line 507)
        append_call_result_15771 = invoke(stypy.reporting.localization.Localization(__file__, 507, 12), append_15768, *[incl_dir_15769], **kwargs_15770)
        
        # SSA join for if statement (line 506)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'target' (line 509)
        target_15772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 15), 'target')
        # Assigning a type to the variable 'stypy_return_type' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'stypy_return_type', target_15772)
        
        # ################# End of 'generate_config_h(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generate_config_h' in the type store
        # Getting the type of 'stypy_return_type' (line 397)
        stypy_return_type_15773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15773)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generate_config_h'
        return stypy_return_type_15773

    # Assigning a type to the variable 'generate_config_h' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'generate_config_h', generate_config_h)

    @norecursion
    def generate_numpyconfig_h(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'generate_numpyconfig_h'
        module_type_store = module_type_store.open_function_context('generate_numpyconfig_h', 511, 4, False)
        
        # Passed parameters checking function
        generate_numpyconfig_h.stypy_localization = localization
        generate_numpyconfig_h.stypy_type_of_self = None
        generate_numpyconfig_h.stypy_type_store = module_type_store
        generate_numpyconfig_h.stypy_function_name = 'generate_numpyconfig_h'
        generate_numpyconfig_h.stypy_param_names_list = ['ext', 'build_dir']
        generate_numpyconfig_h.stypy_varargs_param_name = None
        generate_numpyconfig_h.stypy_kwargs_param_name = None
        generate_numpyconfig_h.stypy_call_defaults = defaults
        generate_numpyconfig_h.stypy_call_varargs = varargs
        generate_numpyconfig_h.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'generate_numpyconfig_h', ['ext', 'build_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'generate_numpyconfig_h', localization, ['ext', 'build_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'generate_numpyconfig_h(...)' code ##################

        str_15774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 8), 'str', 'Depends on config.h: generate_config_h has to be called before !')
        
        # Call to add_include_dirs(...): (line 515)
        # Processing the call arguments (line 515)
        
        # Call to join(...): (line 515)
        # Processing the call arguments (line 515)
        # Getting the type of 'build_dir' (line 515)
        build_dir_15778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 37), 'build_dir', False)
        str_15779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 48), 'str', 'src')
        str_15780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 55), 'str', 'private')
        # Processing the call keyword arguments (line 515)
        kwargs_15781 = {}
        # Getting the type of 'join' (line 515)
        join_15777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 32), 'join', False)
        # Calling join(args, kwargs) (line 515)
        join_call_result_15782 = invoke(stypy.reporting.localization.Localization(__file__, 515, 32), join_15777, *[build_dir_15778, str_15779, str_15780], **kwargs_15781)
        
        # Processing the call keyword arguments (line 515)
        kwargs_15783 = {}
        # Getting the type of 'config' (line 515)
        config_15775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'config', False)
        # Obtaining the member 'add_include_dirs' of a type (line 515)
        add_include_dirs_15776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 8), config_15775, 'add_include_dirs')
        # Calling add_include_dirs(args, kwargs) (line 515)
        add_include_dirs_call_result_15784 = invoke(stypy.reporting.localization.Localization(__file__, 515, 8), add_include_dirs_15776, *[join_call_result_15782], **kwargs_15783)
        
        
        # Assigning a Call to a Name (line 517):
        
        # Assigning a Call to a Name (line 517):
        
        # Call to join(...): (line 517)
        # Processing the call arguments (line 517)
        # Getting the type of 'build_dir' (line 517)
        build_dir_15786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 22), 'build_dir', False)
        # Getting the type of 'header_dir' (line 517)
        header_dir_15787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 33), 'header_dir', False)
        str_15788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 45), 'str', '_numpyconfig.h')
        # Processing the call keyword arguments (line 517)
        kwargs_15789 = {}
        # Getting the type of 'join' (line 517)
        join_15785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 17), 'join', False)
        # Calling join(args, kwargs) (line 517)
        join_call_result_15790 = invoke(stypy.reporting.localization.Localization(__file__, 517, 17), join_15785, *[build_dir_15786, header_dir_15787, str_15788], **kwargs_15789)
        
        # Assigning a type to the variable 'target' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'target', join_call_result_15790)
        
        # Assigning a Call to a Name (line 518):
        
        # Assigning a Call to a Name (line 518):
        
        # Call to dirname(...): (line 518)
        # Processing the call arguments (line 518)
        # Getting the type of 'target' (line 518)
        target_15794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 28), 'target', False)
        # Processing the call keyword arguments (line 518)
        kwargs_15795 = {}
        # Getting the type of 'os' (line 518)
        os_15791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'os', False)
        # Obtaining the member 'path' of a type (line 518)
        path_15792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 12), os_15791, 'path')
        # Obtaining the member 'dirname' of a type (line 518)
        dirname_15793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 12), path_15792, 'dirname')
        # Calling dirname(args, kwargs) (line 518)
        dirname_call_result_15796 = invoke(stypy.reporting.localization.Localization(__file__, 518, 12), dirname_15793, *[target_15794], **kwargs_15795)
        
        # Assigning a type to the variable 'd' (line 518)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'd', dirname_call_result_15796)
        
        
        
        # Call to exists(...): (line 519)
        # Processing the call arguments (line 519)
        # Getting the type of 'd' (line 519)
        d_15800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 30), 'd', False)
        # Processing the call keyword arguments (line 519)
        kwargs_15801 = {}
        # Getting the type of 'os' (line 519)
        os_15797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 519)
        path_15798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 15), os_15797, 'path')
        # Obtaining the member 'exists' of a type (line 519)
        exists_15799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 15), path_15798, 'exists')
        # Calling exists(args, kwargs) (line 519)
        exists_call_result_15802 = invoke(stypy.reporting.localization.Localization(__file__, 519, 15), exists_15799, *[d_15800], **kwargs_15801)
        
        # Applying the 'not' unary operator (line 519)
        result_not__15803 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 11), 'not', exists_call_result_15802)
        
        # Testing the type of an if condition (line 519)
        if_condition_15804 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 519, 8), result_not__15803)
        # Assigning a type to the variable 'if_condition_15804' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'if_condition_15804', if_condition_15804)
        # SSA begins for if statement (line 519)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to makedirs(...): (line 520)
        # Processing the call arguments (line 520)
        # Getting the type of 'd' (line 520)
        d_15807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 24), 'd', False)
        # Processing the call keyword arguments (line 520)
        kwargs_15808 = {}
        # Getting the type of 'os' (line 520)
        os_15805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'os', False)
        # Obtaining the member 'makedirs' of a type (line 520)
        makedirs_15806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), os_15805, 'makedirs')
        # Calling makedirs(args, kwargs) (line 520)
        makedirs_call_result_15809 = invoke(stypy.reporting.localization.Localization(__file__, 520, 12), makedirs_15806, *[d_15807], **kwargs_15808)
        
        # SSA join for if statement (line 519)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to newer(...): (line 521)
        # Processing the call arguments (line 521)
        # Getting the type of '__file__' (line 521)
        file___15811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 17), '__file__', False)
        # Getting the type of 'target' (line 521)
        target_15812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 27), 'target', False)
        # Processing the call keyword arguments (line 521)
        kwargs_15813 = {}
        # Getting the type of 'newer' (line 521)
        newer_15810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 11), 'newer', False)
        # Calling newer(args, kwargs) (line 521)
        newer_call_result_15814 = invoke(stypy.reporting.localization.Localization(__file__, 521, 11), newer_15810, *[file___15811, target_15812], **kwargs_15813)
        
        # Testing the type of an if condition (line 521)
        if_condition_15815 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 521, 8), newer_call_result_15814)
        # Assigning a type to the variable 'if_condition_15815' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'if_condition_15815', if_condition_15815)
        # SSA begins for if statement (line 521)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 522):
        
        # Assigning a Call to a Name (line 522):
        
        # Call to get_config_cmd(...): (line 522)
        # Processing the call keyword arguments (line 522)
        kwargs_15818 = {}
        # Getting the type of 'config' (line 522)
        config_15816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 25), 'config', False)
        # Obtaining the member 'get_config_cmd' of a type (line 522)
        get_config_cmd_15817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 25), config_15816, 'get_config_cmd')
        # Calling get_config_cmd(args, kwargs) (line 522)
        get_config_cmd_call_result_15819 = invoke(stypy.reporting.localization.Localization(__file__, 522, 25), get_config_cmd_15817, *[], **kwargs_15818)
        
        # Assigning a type to the variable 'config_cmd' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'config_cmd', get_config_cmd_call_result_15819)
        
        # Call to info(...): (line 523)
        # Processing the call arguments (line 523)
        str_15822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 21), 'str', 'Generating %s')
        # Getting the type of 'target' (line 523)
        target_15823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 38), 'target', False)
        # Processing the call keyword arguments (line 523)
        kwargs_15824 = {}
        # Getting the type of 'log' (line 523)
        log_15820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 523)
        info_15821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 12), log_15820, 'info')
        # Calling info(args, kwargs) (line 523)
        info_call_result_15825 = invoke(stypy.reporting.localization.Localization(__file__, 523, 12), info_15821, *[str_15822, target_15823], **kwargs_15824)
        
        
        # Assigning a Call to a Tuple (line 526):
        
        # Assigning a Call to a Name:
        
        # Call to check_types(...): (line 526)
        # Processing the call arguments (line 526)
        # Getting the type of 'config_cmd' (line 526)
        config_cmd_15828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 52), 'config_cmd', False)
        # Getting the type of 'ext' (line 526)
        ext_15829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 64), 'ext', False)
        # Getting the type of 'build_dir' (line 526)
        build_dir_15830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 69), 'build_dir', False)
        # Processing the call keyword arguments (line 526)
        kwargs_15831 = {}
        # Getting the type of 'cocache' (line 526)
        cocache_15826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 32), 'cocache', False)
        # Obtaining the member 'check_types' of a type (line 526)
        check_types_15827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 32), cocache_15826, 'check_types')
        # Calling check_types(args, kwargs) (line 526)
        check_types_call_result_15832 = invoke(stypy.reporting.localization.Localization(__file__, 526, 32), check_types_15827, *[config_cmd_15828, ext_15829, build_dir_15830], **kwargs_15831)
        
        # Assigning a type to the variable 'call_assignment_14137' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 12), 'call_assignment_14137', check_types_call_result_15832)
        
        # Assigning a Call to a Name (line 526):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_15835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 12), 'int')
        # Processing the call keyword arguments
        kwargs_15836 = {}
        # Getting the type of 'call_assignment_14137' (line 526)
        call_assignment_14137_15833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 12), 'call_assignment_14137', False)
        # Obtaining the member '__getitem__' of a type (line 526)
        getitem___15834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 12), call_assignment_14137_15833, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_15837 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___15834, *[int_15835], **kwargs_15836)
        
        # Assigning a type to the variable 'call_assignment_14138' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 12), 'call_assignment_14138', getitem___call_result_15837)
        
        # Assigning a Name to a Name (line 526):
        # Getting the type of 'call_assignment_14138' (line 526)
        call_assignment_14138_15838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 12), 'call_assignment_14138')
        # Assigning a type to the variable 'ignored' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 12), 'ignored', call_assignment_14138_15838)
        
        # Assigning a Call to a Name (line 526):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_15841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 12), 'int')
        # Processing the call keyword arguments
        kwargs_15842 = {}
        # Getting the type of 'call_assignment_14137' (line 526)
        call_assignment_14137_15839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 12), 'call_assignment_14137', False)
        # Obtaining the member '__getitem__' of a type (line 526)
        getitem___15840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 12), call_assignment_14137_15839, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_15843 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___15840, *[int_15841], **kwargs_15842)
        
        # Assigning a type to the variable 'call_assignment_14139' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 12), 'call_assignment_14139', getitem___call_result_15843)
        
        # Assigning a Name to a Name (line 526):
        # Getting the type of 'call_assignment_14139' (line 526)
        call_assignment_14139_15844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 12), 'call_assignment_14139')
        # Assigning a type to the variable 'moredefs' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 21), 'moredefs', call_assignment_14139_15844)
        
        
        # Call to is_npy_no_signal(...): (line 528)
        # Processing the call keyword arguments (line 528)
        kwargs_15846 = {}
        # Getting the type of 'is_npy_no_signal' (line 528)
        is_npy_no_signal_15845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 15), 'is_npy_no_signal', False)
        # Calling is_npy_no_signal(args, kwargs) (line 528)
        is_npy_no_signal_call_result_15847 = invoke(stypy.reporting.localization.Localization(__file__, 528, 15), is_npy_no_signal_15845, *[], **kwargs_15846)
        
        # Testing the type of an if condition (line 528)
        if_condition_15848 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 528, 12), is_npy_no_signal_call_result_15847)
        # Assigning a type to the variable 'if_condition_15848' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'if_condition_15848', if_condition_15848)
        # SSA begins for if statement (line 528)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 529)
        # Processing the call arguments (line 529)
        
        # Obtaining an instance of the builtin type 'tuple' (line 529)
        tuple_15851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 529)
        # Adding element type (line 529)
        str_15852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 33), 'str', 'NPY_NO_SIGNAL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 33), tuple_15851, str_15852)
        # Adding element type (line 529)
        int_15853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 33), tuple_15851, int_15853)
        
        # Processing the call keyword arguments (line 529)
        kwargs_15854 = {}
        # Getting the type of 'moredefs' (line 529)
        moredefs_15849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 16), 'moredefs', False)
        # Obtaining the member 'append' of a type (line 529)
        append_15850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 16), moredefs_15849, 'append')
        # Calling append(args, kwargs) (line 529)
        append_call_result_15855 = invoke(stypy.reporting.localization.Localization(__file__, 529, 16), append_15850, *[tuple_15851], **kwargs_15854)
        
        # SSA join for if statement (line 528)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to is_npy_no_smp(...): (line 531)
        # Processing the call keyword arguments (line 531)
        kwargs_15857 = {}
        # Getting the type of 'is_npy_no_smp' (line 531)
        is_npy_no_smp_15856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 15), 'is_npy_no_smp', False)
        # Calling is_npy_no_smp(args, kwargs) (line 531)
        is_npy_no_smp_call_result_15858 = invoke(stypy.reporting.localization.Localization(__file__, 531, 15), is_npy_no_smp_15856, *[], **kwargs_15857)
        
        # Testing the type of an if condition (line 531)
        if_condition_15859 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 531, 12), is_npy_no_smp_call_result_15858)
        # Assigning a type to the variable 'if_condition_15859' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 12), 'if_condition_15859', if_condition_15859)
        # SSA begins for if statement (line 531)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 532)
        # Processing the call arguments (line 532)
        
        # Obtaining an instance of the builtin type 'tuple' (line 532)
        tuple_15862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 532)
        # Adding element type (line 532)
        str_15863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 33), 'str', 'NPY_NO_SMP')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 33), tuple_15862, str_15863)
        # Adding element type (line 532)
        int_15864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 33), tuple_15862, int_15864)
        
        # Processing the call keyword arguments (line 532)
        kwargs_15865 = {}
        # Getting the type of 'moredefs' (line 532)
        moredefs_15860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 16), 'moredefs', False)
        # Obtaining the member 'append' of a type (line 532)
        append_15861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 16), moredefs_15860, 'append')
        # Calling append(args, kwargs) (line 532)
        append_call_result_15866 = invoke(stypy.reporting.localization.Localization(__file__, 532, 16), append_15861, *[tuple_15862], **kwargs_15865)
        
        # SSA branch for the else part of an if statement (line 531)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 534)
        # Processing the call arguments (line 534)
        
        # Obtaining an instance of the builtin type 'tuple' (line 534)
        tuple_15869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 534)
        # Adding element type (line 534)
        str_15870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 33), 'str', 'NPY_NO_SMP')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 33), tuple_15869, str_15870)
        # Adding element type (line 534)
        int_15871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 534, 33), tuple_15869, int_15871)
        
        # Processing the call keyword arguments (line 534)
        kwargs_15872 = {}
        # Getting the type of 'moredefs' (line 534)
        moredefs_15867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 16), 'moredefs', False)
        # Obtaining the member 'append' of a type (line 534)
        append_15868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 16), moredefs_15867, 'append')
        # Calling append(args, kwargs) (line 534)
        append_call_result_15873 = invoke(stypy.reporting.localization.Localization(__file__, 534, 16), append_15868, *[tuple_15869], **kwargs_15872)
        
        # SSA join for if statement (line 531)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 536):
        
        # Assigning a Call to a Name (line 536):
        
        # Call to check_mathlib(...): (line 536)
        # Processing the call arguments (line 536)
        # Getting the type of 'config_cmd' (line 536)
        config_cmd_15875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 37), 'config_cmd', False)
        # Processing the call keyword arguments (line 536)
        kwargs_15876 = {}
        # Getting the type of 'check_mathlib' (line 536)
        check_mathlib_15874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 23), 'check_mathlib', False)
        # Calling check_mathlib(args, kwargs) (line 536)
        check_mathlib_call_result_15877 = invoke(stypy.reporting.localization.Localization(__file__, 536, 23), check_mathlib_15874, *[config_cmd_15875], **kwargs_15876)
        
        # Assigning a type to the variable 'mathlibs' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 12), 'mathlibs', check_mathlib_call_result_15877)
        
        # Call to extend(...): (line 537)
        # Processing the call arguments (line 537)
        
        # Obtaining the type of the subscript
        int_15880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 66), 'int')
        
        # Call to check_ieee_macros(...): (line 537)
        # Processing the call arguments (line 537)
        # Getting the type of 'config_cmd' (line 537)
        config_cmd_15883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 54), 'config_cmd', False)
        # Processing the call keyword arguments (line 537)
        kwargs_15884 = {}
        # Getting the type of 'cocache' (line 537)
        cocache_15881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 28), 'cocache', False)
        # Obtaining the member 'check_ieee_macros' of a type (line 537)
        check_ieee_macros_15882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 28), cocache_15881, 'check_ieee_macros')
        # Calling check_ieee_macros(args, kwargs) (line 537)
        check_ieee_macros_call_result_15885 = invoke(stypy.reporting.localization.Localization(__file__, 537, 28), check_ieee_macros_15882, *[config_cmd_15883], **kwargs_15884)
        
        # Obtaining the member '__getitem__' of a type (line 537)
        getitem___15886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 28), check_ieee_macros_call_result_15885, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 537)
        subscript_call_result_15887 = invoke(stypy.reporting.localization.Localization(__file__, 537, 28), getitem___15886, int_15880)
        
        # Processing the call keyword arguments (line 537)
        kwargs_15888 = {}
        # Getting the type of 'moredefs' (line 537)
        moredefs_15878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), 'moredefs', False)
        # Obtaining the member 'extend' of a type (line 537)
        extend_15879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 12), moredefs_15878, 'extend')
        # Calling extend(args, kwargs) (line 537)
        extend_call_result_15889 = invoke(stypy.reporting.localization.Localization(__file__, 537, 12), extend_15879, *[subscript_call_result_15887], **kwargs_15888)
        
        
        # Call to extend(...): (line 538)
        # Processing the call arguments (line 538)
        
        # Obtaining the type of the subscript
        int_15892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 72), 'int')
        
        # Call to check_complex(...): (line 538)
        # Processing the call arguments (line 538)
        # Getting the type of 'config_cmd' (line 538)
        config_cmd_15895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 50), 'config_cmd', False)
        # Getting the type of 'mathlibs' (line 538)
        mathlibs_15896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 62), 'mathlibs', False)
        # Processing the call keyword arguments (line 538)
        kwargs_15897 = {}
        # Getting the type of 'cocache' (line 538)
        cocache_15893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 28), 'cocache', False)
        # Obtaining the member 'check_complex' of a type (line 538)
        check_complex_15894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 28), cocache_15893, 'check_complex')
        # Calling check_complex(args, kwargs) (line 538)
        check_complex_call_result_15898 = invoke(stypy.reporting.localization.Localization(__file__, 538, 28), check_complex_15894, *[config_cmd_15895, mathlibs_15896], **kwargs_15897)
        
        # Obtaining the member '__getitem__' of a type (line 538)
        getitem___15899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 28), check_complex_call_result_15898, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 538)
        subscript_call_result_15900 = invoke(stypy.reporting.localization.Localization(__file__, 538, 28), getitem___15899, int_15892)
        
        # Processing the call keyword arguments (line 538)
        kwargs_15901 = {}
        # Getting the type of 'moredefs' (line 538)
        moredefs_15890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'moredefs', False)
        # Obtaining the member 'extend' of a type (line 538)
        extend_15891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 12), moredefs_15890, 'extend')
        # Calling extend(args, kwargs) (line 538)
        extend_call_result_15902 = invoke(stypy.reporting.localization.Localization(__file__, 538, 12), extend_15891, *[subscript_call_result_15900], **kwargs_15901)
        
        
        # Getting the type of 'NPY_RELAXED_STRIDES_CHECKING' (line 540)
        NPY_RELAXED_STRIDES_CHECKING_15903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 15), 'NPY_RELAXED_STRIDES_CHECKING')
        # Testing the type of an if condition (line 540)
        if_condition_15904 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 540, 12), NPY_RELAXED_STRIDES_CHECKING_15903)
        # Assigning a type to the variable 'if_condition_15904' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 12), 'if_condition_15904', if_condition_15904)
        # SSA begins for if statement (line 540)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 541)
        # Processing the call arguments (line 541)
        
        # Obtaining an instance of the builtin type 'tuple' (line 541)
        tuple_15907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 541)
        # Adding element type (line 541)
        str_15908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 33), 'str', 'NPY_RELAXED_STRIDES_CHECKING')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 541, 33), tuple_15907, str_15908)
        # Adding element type (line 541)
        int_15909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 541, 33), tuple_15907, int_15909)
        
        # Processing the call keyword arguments (line 541)
        kwargs_15910 = {}
        # Getting the type of 'moredefs' (line 541)
        moredefs_15905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 16), 'moredefs', False)
        # Obtaining the member 'append' of a type (line 541)
        append_15906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 16), moredefs_15905, 'append')
        # Calling append(args, kwargs) (line 541)
        append_call_result_15911 = invoke(stypy.reporting.localization.Localization(__file__, 541, 16), append_15906, *[tuple_15907], **kwargs_15910)
        
        # SSA join for if statement (line 540)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to check_decl(...): (line 544)
        # Processing the call arguments (line 544)
        str_15914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 37), 'str', 'PRIdPTR')
        # Processing the call keyword arguments (line 544)
        
        # Obtaining an instance of the builtin type 'list' (line 544)
        list_15915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 544)
        # Adding element type (line 544)
        str_15916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 57), 'str', 'inttypes.h')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 544, 56), list_15915, str_15916)
        
        keyword_15917 = list_15915
        kwargs_15918 = {'headers': keyword_15917}
        # Getting the type of 'config_cmd' (line 544)
        config_cmd_15912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 15), 'config_cmd', False)
        # Obtaining the member 'check_decl' of a type (line 544)
        check_decl_15913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 15), config_cmd_15912, 'check_decl')
        # Calling check_decl(args, kwargs) (line 544)
        check_decl_call_result_15919 = invoke(stypy.reporting.localization.Localization(__file__, 544, 15), check_decl_15913, *[str_15914], **kwargs_15918)
        
        # Testing the type of an if condition (line 544)
        if_condition_15920 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 544, 12), check_decl_call_result_15919)
        # Assigning a type to the variable 'if_condition_15920' (line 544)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 12), 'if_condition_15920', if_condition_15920)
        # SSA begins for if statement (line 544)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 545)
        # Processing the call arguments (line 545)
        
        # Obtaining an instance of the builtin type 'tuple' (line 545)
        tuple_15923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 545)
        # Adding element type (line 545)
        str_15924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 33), 'str', 'NPY_USE_C99_FORMATS')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 33), tuple_15923, str_15924)
        # Adding element type (line 545)
        int_15925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 33), tuple_15923, int_15925)
        
        # Processing the call keyword arguments (line 545)
        kwargs_15926 = {}
        # Getting the type of 'moredefs' (line 545)
        moredefs_15921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'moredefs', False)
        # Obtaining the member 'append' of a type (line 545)
        append_15922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 16), moredefs_15921, 'append')
        # Calling append(args, kwargs) (line 545)
        append_call_result_15927 = invoke(stypy.reporting.localization.Localization(__file__, 545, 16), append_15922, *[tuple_15923], **kwargs_15926)
        
        # SSA join for if statement (line 544)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 548):
        
        # Assigning a Call to a Name (line 548):
        
        # Call to visibility_define(...): (line 548)
        # Processing the call arguments (line 548)
        # Getting the type of 'config_cmd' (line 548)
        config_cmd_15929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 50), 'config_cmd', False)
        # Processing the call keyword arguments (line 548)
        kwargs_15930 = {}
        # Getting the type of 'visibility_define' (line 548)
        visibility_define_15928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 32), 'visibility_define', False)
        # Calling visibility_define(args, kwargs) (line 548)
        visibility_define_call_result_15931 = invoke(stypy.reporting.localization.Localization(__file__, 548, 32), visibility_define_15928, *[config_cmd_15929], **kwargs_15930)
        
        # Assigning a type to the variable 'hidden_visibility' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 12), 'hidden_visibility', visibility_define_call_result_15931)
        
        # Call to append(...): (line 549)
        # Processing the call arguments (line 549)
        
        # Obtaining an instance of the builtin type 'tuple' (line 549)
        tuple_15934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 549)
        # Adding element type (line 549)
        str_15935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 29), 'str', 'NPY_VISIBILITY_HIDDEN')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 549, 29), tuple_15934, str_15935)
        # Adding element type (line 549)
        # Getting the type of 'hidden_visibility' (line 549)
        hidden_visibility_15936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 54), 'hidden_visibility', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 549, 29), tuple_15934, hidden_visibility_15936)
        
        # Processing the call keyword arguments (line 549)
        kwargs_15937 = {}
        # Getting the type of 'moredefs' (line 549)
        moredefs_15932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'moredefs', False)
        # Obtaining the member 'append' of a type (line 549)
        append_15933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 12), moredefs_15932, 'append')
        # Calling append(args, kwargs) (line 549)
        append_call_result_15938 = invoke(stypy.reporting.localization.Localization(__file__, 549, 12), append_15933, *[tuple_15934], **kwargs_15937)
        
        
        # Call to append(...): (line 552)
        # Processing the call arguments (line 552)
        
        # Obtaining an instance of the builtin type 'tuple' (line 552)
        tuple_15941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 552)
        # Adding element type (line 552)
        str_15942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 29), 'str', 'NPY_ABI_VERSION')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 29), tuple_15941, str_15942)
        # Adding element type (line 552)
        str_15943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 48), 'str', '0x%.8X')
        # Getting the type of 'C_ABI_VERSION' (line 552)
        C_ABI_VERSION_15944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 59), 'C_ABI_VERSION', False)
        # Applying the binary operator '%' (line 552)
        result_mod_15945 = python_operator(stypy.reporting.localization.Localization(__file__, 552, 48), '%', str_15943, C_ABI_VERSION_15944)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 29), tuple_15941, result_mod_15945)
        
        # Processing the call keyword arguments (line 552)
        kwargs_15946 = {}
        # Getting the type of 'moredefs' (line 552)
        moredefs_15939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 12), 'moredefs', False)
        # Obtaining the member 'append' of a type (line 552)
        append_15940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 12), moredefs_15939, 'append')
        # Calling append(args, kwargs) (line 552)
        append_call_result_15947 = invoke(stypy.reporting.localization.Localization(__file__, 552, 12), append_15940, *[tuple_15941], **kwargs_15946)
        
        
        # Call to append(...): (line 553)
        # Processing the call arguments (line 553)
        
        # Obtaining an instance of the builtin type 'tuple' (line 553)
        tuple_15950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 553)
        # Adding element type (line 553)
        str_15951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 29), 'str', 'NPY_API_VERSION')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 553, 29), tuple_15950, str_15951)
        # Adding element type (line 553)
        str_15952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 48), 'str', '0x%.8X')
        # Getting the type of 'C_API_VERSION' (line 553)
        C_API_VERSION_15953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 59), 'C_API_VERSION', False)
        # Applying the binary operator '%' (line 553)
        result_mod_15954 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 48), '%', str_15952, C_API_VERSION_15953)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 553, 29), tuple_15950, result_mod_15954)
        
        # Processing the call keyword arguments (line 553)
        kwargs_15955 = {}
        # Getting the type of 'moredefs' (line 553)
        moredefs_15948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 12), 'moredefs', False)
        # Obtaining the member 'append' of a type (line 553)
        append_15949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 12), moredefs_15948, 'append')
        # Calling append(args, kwargs) (line 553)
        append_call_result_15956 = invoke(stypy.reporting.localization.Localization(__file__, 553, 12), append_15949, *[tuple_15950], **kwargs_15955)
        
        
        # Assigning a Call to a Name (line 556):
        
        # Assigning a Call to a Name (line 556):
        
        # Call to open(...): (line 556)
        # Processing the call arguments (line 556)
        # Getting the type of 'target' (line 556)
        target_15958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 28), 'target', False)
        str_15959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 36), 'str', 'w')
        # Processing the call keyword arguments (line 556)
        kwargs_15960 = {}
        # Getting the type of 'open' (line 556)
        open_15957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 23), 'open', False)
        # Calling open(args, kwargs) (line 556)
        open_call_result_15961 = invoke(stypy.reporting.localization.Localization(__file__, 556, 23), open_15957, *[target_15958, str_15959], **kwargs_15960)
        
        # Assigning a type to the variable 'target_f' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 12), 'target_f', open_call_result_15961)
        
        # Getting the type of 'moredefs' (line 557)
        moredefs_15962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 21), 'moredefs')
        # Testing the type of a for loop iterable (line 557)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 557, 12), moredefs_15962)
        # Getting the type of the for loop variable (line 557)
        for_loop_var_15963 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 557, 12), moredefs_15962)
        # Assigning a type to the variable 'd' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'd', for_loop_var_15963)
        # SSA begins for a for statement (line 557)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 558)
        # Getting the type of 'str' (line 558)
        str_15964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 33), 'str')
        # Getting the type of 'd' (line 558)
        d_15965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 30), 'd')
        
        (may_be_15966, more_types_in_union_15967) = may_be_subtype(str_15964, d_15965)

        if may_be_15966:

            if more_types_in_union_15967:
                # Runtime conditional SSA (line 558)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'd' (line 558)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 16), 'd', remove_not_subtype_from_union(d_15965, str))
            
            # Call to write(...): (line 559)
            # Processing the call arguments (line 559)
            str_15970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 35), 'str', '#define %s\n')
            # Getting the type of 'd' (line 559)
            d_15971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 53), 'd', False)
            # Applying the binary operator '%' (line 559)
            result_mod_15972 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 35), '%', str_15970, d_15971)
            
            # Processing the call keyword arguments (line 559)
            kwargs_15973 = {}
            # Getting the type of 'target_f' (line 559)
            target_f_15968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 20), 'target_f', False)
            # Obtaining the member 'write' of a type (line 559)
            write_15969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 20), target_f_15968, 'write')
            # Calling write(args, kwargs) (line 559)
            write_call_result_15974 = invoke(stypy.reporting.localization.Localization(__file__, 559, 20), write_15969, *[result_mod_15972], **kwargs_15973)
            

            if more_types_in_union_15967:
                # Runtime conditional SSA for else branch (line 558)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_15966) or more_types_in_union_15967):
            # Assigning a type to the variable 'd' (line 558)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 16), 'd', remove_subtype_from_union(d_15965, str))
            
            # Call to write(...): (line 561)
            # Processing the call arguments (line 561)
            str_15977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 35), 'str', '#define %s %s\n')
            
            # Obtaining an instance of the builtin type 'tuple' (line 561)
            tuple_15978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 56), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 561)
            # Adding element type (line 561)
            
            # Obtaining the type of the subscript
            int_15979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 58), 'int')
            # Getting the type of 'd' (line 561)
            d_15980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 56), 'd', False)
            # Obtaining the member '__getitem__' of a type (line 561)
            getitem___15981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 56), d_15980, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 561)
            subscript_call_result_15982 = invoke(stypy.reporting.localization.Localization(__file__, 561, 56), getitem___15981, int_15979)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 56), tuple_15978, subscript_call_result_15982)
            # Adding element type (line 561)
            
            # Obtaining the type of the subscript
            int_15983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 64), 'int')
            # Getting the type of 'd' (line 561)
            d_15984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 62), 'd', False)
            # Obtaining the member '__getitem__' of a type (line 561)
            getitem___15985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 62), d_15984, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 561)
            subscript_call_result_15986 = invoke(stypy.reporting.localization.Localization(__file__, 561, 62), getitem___15985, int_15983)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 56), tuple_15978, subscript_call_result_15986)
            
            # Applying the binary operator '%' (line 561)
            result_mod_15987 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 35), '%', str_15977, tuple_15978)
            
            # Processing the call keyword arguments (line 561)
            kwargs_15988 = {}
            # Getting the type of 'target_f' (line 561)
            target_f_15975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 20), 'target_f', False)
            # Obtaining the member 'write' of a type (line 561)
            write_15976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 20), target_f_15975, 'write')
            # Calling write(args, kwargs) (line 561)
            write_call_result_15989 = invoke(stypy.reporting.localization.Localization(__file__, 561, 20), write_15976, *[result_mod_15987], **kwargs_15988)
            

            if (may_be_15966 and more_types_in_union_15967):
                # SSA join for if statement (line 558)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 564)
        # Processing the call arguments (line 564)
        str_15992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, (-1)), 'str', '\n#ifndef __STDC_FORMAT_MACROS\n#define __STDC_FORMAT_MACROS 1\n#endif\n')
        # Processing the call keyword arguments (line 564)
        kwargs_15993 = {}
        # Getting the type of 'target_f' (line 564)
        target_f_15990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 12), 'target_f', False)
        # Obtaining the member 'write' of a type (line 564)
        write_15991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 12), target_f_15990, 'write')
        # Calling write(args, kwargs) (line 564)
        write_call_result_15994 = invoke(stypy.reporting.localization.Localization(__file__, 564, 12), write_15991, *[str_15992], **kwargs_15993)
        
        
        # Call to close(...): (line 569)
        # Processing the call keyword arguments (line 569)
        kwargs_15997 = {}
        # Getting the type of 'target_f' (line 569)
        target_f_15995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 12), 'target_f', False)
        # Obtaining the member 'close' of a type (line 569)
        close_15996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 12), target_f_15995, 'close')
        # Calling close(args, kwargs) (line 569)
        close_call_result_15998 = invoke(stypy.reporting.localization.Localization(__file__, 569, 12), close_15996, *[], **kwargs_15997)
        
        
        # Call to print(...): (line 572)
        # Processing the call arguments (line 572)
        str_16000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 18), 'str', 'File: %s')
        # Getting the type of 'target' (line 572)
        target_16001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 31), 'target', False)
        # Applying the binary operator '%' (line 572)
        result_mod_16002 = python_operator(stypy.reporting.localization.Localization(__file__, 572, 18), '%', str_16000, target_16001)
        
        # Processing the call keyword arguments (line 572)
        kwargs_16003 = {}
        # Getting the type of 'print' (line 572)
        print_15999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'print', False)
        # Calling print(args, kwargs) (line 572)
        print_call_result_16004 = invoke(stypy.reporting.localization.Localization(__file__, 572, 12), print_15999, *[result_mod_16002], **kwargs_16003)
        
        
        # Assigning a Call to a Name (line 573):
        
        # Assigning a Call to a Name (line 573):
        
        # Call to open(...): (line 573)
        # Processing the call arguments (line 573)
        # Getting the type of 'target' (line 573)
        target_16006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 28), 'target', False)
        # Processing the call keyword arguments (line 573)
        kwargs_16007 = {}
        # Getting the type of 'open' (line 573)
        open_16005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 23), 'open', False)
        # Calling open(args, kwargs) (line 573)
        open_call_result_16008 = invoke(stypy.reporting.localization.Localization(__file__, 573, 23), open_16005, *[target_16006], **kwargs_16007)
        
        # Assigning a type to the variable 'target_f' (line 573)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 12), 'target_f', open_call_result_16008)
        
        # Call to print(...): (line 574)
        # Processing the call arguments (line 574)
        
        # Call to read(...): (line 574)
        # Processing the call keyword arguments (line 574)
        kwargs_16012 = {}
        # Getting the type of 'target_f' (line 574)
        target_f_16010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 18), 'target_f', False)
        # Obtaining the member 'read' of a type (line 574)
        read_16011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 18), target_f_16010, 'read')
        # Calling read(args, kwargs) (line 574)
        read_call_result_16013 = invoke(stypy.reporting.localization.Localization(__file__, 574, 18), read_16011, *[], **kwargs_16012)
        
        # Processing the call keyword arguments (line 574)
        kwargs_16014 = {}
        # Getting the type of 'print' (line 574)
        print_16009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 12), 'print', False)
        # Calling print(args, kwargs) (line 574)
        print_call_result_16015 = invoke(stypy.reporting.localization.Localization(__file__, 574, 12), print_16009, *[read_call_result_16013], **kwargs_16014)
        
        
        # Call to close(...): (line 575)
        # Processing the call keyword arguments (line 575)
        kwargs_16018 = {}
        # Getting the type of 'target_f' (line 575)
        target_f_16016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'target_f', False)
        # Obtaining the member 'close' of a type (line 575)
        close_16017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 12), target_f_16016, 'close')
        # Calling close(args, kwargs) (line 575)
        close_call_result_16019 = invoke(stypy.reporting.localization.Localization(__file__, 575, 12), close_16017, *[], **kwargs_16018)
        
        
        # Call to print(...): (line 576)
        # Processing the call arguments (line 576)
        str_16021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 18), 'str', 'EOF')
        # Processing the call keyword arguments (line 576)
        kwargs_16022 = {}
        # Getting the type of 'print' (line 576)
        print_16020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'print', False)
        # Calling print(args, kwargs) (line 576)
        print_call_result_16023 = invoke(stypy.reporting.localization.Localization(__file__, 576, 12), print_16020, *[str_16021], **kwargs_16022)
        
        # SSA join for if statement (line 521)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to add_data_files(...): (line 577)
        # Processing the call arguments (line 577)
        
        # Obtaining an instance of the builtin type 'tuple' (line 577)
        tuple_16026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 577)
        # Adding element type (line 577)
        # Getting the type of 'header_dir' (line 577)
        header_dir_16027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 31), 'header_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 31), tuple_16026, header_dir_16027)
        # Adding element type (line 577)
        # Getting the type of 'target' (line 577)
        target_16028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 43), 'target', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 31), tuple_16026, target_16028)
        
        # Processing the call keyword arguments (line 577)
        kwargs_16029 = {}
        # Getting the type of 'config' (line 577)
        config_16024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'config', False)
        # Obtaining the member 'add_data_files' of a type (line 577)
        add_data_files_16025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 8), config_16024, 'add_data_files')
        # Calling add_data_files(args, kwargs) (line 577)
        add_data_files_call_result_16030 = invoke(stypy.reporting.localization.Localization(__file__, 577, 8), add_data_files_16025, *[tuple_16026], **kwargs_16029)
        
        # Getting the type of 'target' (line 578)
        target_16031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 15), 'target')
        # Assigning a type to the variable 'stypy_return_type' (line 578)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 8), 'stypy_return_type', target_16031)
        
        # ################# End of 'generate_numpyconfig_h(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generate_numpyconfig_h' in the type store
        # Getting the type of 'stypy_return_type' (line 511)
        stypy_return_type_16032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16032)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generate_numpyconfig_h'
        return stypy_return_type_16032

    # Assigning a type to the variable 'generate_numpyconfig_h' (line 511)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 4), 'generate_numpyconfig_h', generate_numpyconfig_h)

    @norecursion
    def generate_api_func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'generate_api_func'
        module_type_store = module_type_store.open_function_context('generate_api_func', 580, 4, False)
        
        # Passed parameters checking function
        generate_api_func.stypy_localization = localization
        generate_api_func.stypy_type_of_self = None
        generate_api_func.stypy_type_store = module_type_store
        generate_api_func.stypy_function_name = 'generate_api_func'
        generate_api_func.stypy_param_names_list = ['module_name']
        generate_api_func.stypy_varargs_param_name = None
        generate_api_func.stypy_kwargs_param_name = None
        generate_api_func.stypy_call_defaults = defaults
        generate_api_func.stypy_call_varargs = varargs
        generate_api_func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'generate_api_func', ['module_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'generate_api_func', localization, ['module_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'generate_api_func(...)' code ##################


        @norecursion
        def generate_api(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'generate_api'
            module_type_store = module_type_store.open_function_context('generate_api', 581, 8, False)
            
            # Passed parameters checking function
            generate_api.stypy_localization = localization
            generate_api.stypy_type_of_self = None
            generate_api.stypy_type_store = module_type_store
            generate_api.stypy_function_name = 'generate_api'
            generate_api.stypy_param_names_list = ['ext', 'build_dir']
            generate_api.stypy_varargs_param_name = None
            generate_api.stypy_kwargs_param_name = None
            generate_api.stypy_call_defaults = defaults
            generate_api.stypy_call_varargs = varargs
            generate_api.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'generate_api', ['ext', 'build_dir'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'generate_api', localization, ['ext', 'build_dir'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'generate_api(...)' code ##################

            
            # Assigning a Call to a Name (line 582):
            
            # Assigning a Call to a Name (line 582):
            
            # Call to join(...): (line 582)
            # Processing the call arguments (line 582)
            # Getting the type of 'codegen_dir' (line 582)
            codegen_dir_16034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 26), 'codegen_dir', False)
            # Getting the type of 'module_name' (line 582)
            module_name_16035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 39), 'module_name', False)
            str_16036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 53), 'str', '.py')
            # Applying the binary operator '+' (line 582)
            result_add_16037 = python_operator(stypy.reporting.localization.Localization(__file__, 582, 39), '+', module_name_16035, str_16036)
            
            # Processing the call keyword arguments (line 582)
            kwargs_16038 = {}
            # Getting the type of 'join' (line 582)
            join_16033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 21), 'join', False)
            # Calling join(args, kwargs) (line 582)
            join_call_result_16039 = invoke(stypy.reporting.localization.Localization(__file__, 582, 21), join_16033, *[codegen_dir_16034, result_add_16037], **kwargs_16038)
            
            # Assigning a type to the variable 'script' (line 582)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 12), 'script', join_call_result_16039)
            
            # Call to insert(...): (line 583)
            # Processing the call arguments (line 583)
            int_16043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 28), 'int')
            # Getting the type of 'codegen_dir' (line 583)
            codegen_dir_16044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 31), 'codegen_dir', False)
            # Processing the call keyword arguments (line 583)
            kwargs_16045 = {}
            # Getting the type of 'sys' (line 583)
            sys_16040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'sys', False)
            # Obtaining the member 'path' of a type (line 583)
            path_16041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 12), sys_16040, 'path')
            # Obtaining the member 'insert' of a type (line 583)
            insert_16042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 12), path_16041, 'insert')
            # Calling insert(args, kwargs) (line 583)
            insert_call_result_16046 = invoke(stypy.reporting.localization.Localization(__file__, 583, 12), insert_16042, *[int_16043, codegen_dir_16044], **kwargs_16045)
            
            
            # Try-finally block (line 584)
            
            # Assigning a Call to a Name (line 585):
            
            # Assigning a Call to a Name (line 585):
            
            # Call to __import__(...): (line 585)
            # Processing the call arguments (line 585)
            # Getting the type of 'module_name' (line 585)
            module_name_16048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 31), 'module_name', False)
            # Processing the call keyword arguments (line 585)
            kwargs_16049 = {}
            # Getting the type of '__import__' (line 585)
            import___16047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 20), '__import__', False)
            # Calling __import__(args, kwargs) (line 585)
            import___call_result_16050 = invoke(stypy.reporting.localization.Localization(__file__, 585, 20), import___16047, *[module_name_16048], **kwargs_16049)
            
            # Assigning a type to the variable 'm' (line 585)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 16), 'm', import___call_result_16050)
            
            # Call to info(...): (line 586)
            # Processing the call arguments (line 586)
            str_16053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 25), 'str', 'executing %s')
            # Getting the type of 'script' (line 586)
            script_16054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 41), 'script', False)
            # Processing the call keyword arguments (line 586)
            kwargs_16055 = {}
            # Getting the type of 'log' (line 586)
            log_16051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 16), 'log', False)
            # Obtaining the member 'info' of a type (line 586)
            info_16052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 16), log_16051, 'info')
            # Calling info(args, kwargs) (line 586)
            info_call_result_16056 = invoke(stypy.reporting.localization.Localization(__file__, 586, 16), info_16052, *[str_16053, script_16054], **kwargs_16055)
            
            
            # Assigning a Call to a Tuple (line 587):
            
            # Assigning a Call to a Name:
            
            # Call to generate_api(...): (line 587)
            # Processing the call arguments (line 587)
            
            # Call to join(...): (line 587)
            # Processing the call arguments (line 587)
            # Getting the type of 'build_dir' (line 587)
            build_dir_16062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 71), 'build_dir', False)
            # Getting the type of 'header_dir' (line 587)
            header_dir_16063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 82), 'header_dir', False)
            # Processing the call keyword arguments (line 587)
            kwargs_16064 = {}
            # Getting the type of 'os' (line 587)
            os_16059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 58), 'os', False)
            # Obtaining the member 'path' of a type (line 587)
            path_16060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 58), os_16059, 'path')
            # Obtaining the member 'join' of a type (line 587)
            join_16061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 58), path_16060, 'join')
            # Calling join(args, kwargs) (line 587)
            join_call_result_16065 = invoke(stypy.reporting.localization.Localization(__file__, 587, 58), join_16061, *[build_dir_16062, header_dir_16063], **kwargs_16064)
            
            # Processing the call keyword arguments (line 587)
            kwargs_16066 = {}
            # Getting the type of 'm' (line 587)
            m_16057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 43), 'm', False)
            # Obtaining the member 'generate_api' of a type (line 587)
            generate_api_16058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 43), m_16057, 'generate_api')
            # Calling generate_api(args, kwargs) (line 587)
            generate_api_call_result_16067 = invoke(stypy.reporting.localization.Localization(__file__, 587, 43), generate_api_16058, *[join_call_result_16065], **kwargs_16066)
            
            # Assigning a type to the variable 'call_assignment_14140' (line 587)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'call_assignment_14140', generate_api_call_result_16067)
            
            # Assigning a Call to a Name (line 587):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_16070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 16), 'int')
            # Processing the call keyword arguments
            kwargs_16071 = {}
            # Getting the type of 'call_assignment_14140' (line 587)
            call_assignment_14140_16068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'call_assignment_14140', False)
            # Obtaining the member '__getitem__' of a type (line 587)
            getitem___16069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 16), call_assignment_14140_16068, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_16072 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___16069, *[int_16070], **kwargs_16071)
            
            # Assigning a type to the variable 'call_assignment_14141' (line 587)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'call_assignment_14141', getitem___call_result_16072)
            
            # Assigning a Name to a Name (line 587):
            # Getting the type of 'call_assignment_14141' (line 587)
            call_assignment_14141_16073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'call_assignment_14141')
            # Assigning a type to the variable 'h_file' (line 587)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'h_file', call_assignment_14141_16073)
            
            # Assigning a Call to a Name (line 587):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_16076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 16), 'int')
            # Processing the call keyword arguments
            kwargs_16077 = {}
            # Getting the type of 'call_assignment_14140' (line 587)
            call_assignment_14140_16074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'call_assignment_14140', False)
            # Obtaining the member '__getitem__' of a type (line 587)
            getitem___16075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 16), call_assignment_14140_16074, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_16078 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___16075, *[int_16076], **kwargs_16077)
            
            # Assigning a type to the variable 'call_assignment_14142' (line 587)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'call_assignment_14142', getitem___call_result_16078)
            
            # Assigning a Name to a Name (line 587):
            # Getting the type of 'call_assignment_14142' (line 587)
            call_assignment_14142_16079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'call_assignment_14142')
            # Assigning a type to the variable 'c_file' (line 587)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 24), 'c_file', call_assignment_14142_16079)
            
            # Assigning a Call to a Name (line 587):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_16082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 16), 'int')
            # Processing the call keyword arguments
            kwargs_16083 = {}
            # Getting the type of 'call_assignment_14140' (line 587)
            call_assignment_14140_16080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'call_assignment_14140', False)
            # Obtaining the member '__getitem__' of a type (line 587)
            getitem___16081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 16), call_assignment_14140_16080, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_16084 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___16081, *[int_16082], **kwargs_16083)
            
            # Assigning a type to the variable 'call_assignment_14143' (line 587)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'call_assignment_14143', getitem___call_result_16084)
            
            # Assigning a Name to a Name (line 587):
            # Getting the type of 'call_assignment_14143' (line 587)
            call_assignment_14143_16085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'call_assignment_14143')
            # Assigning a type to the variable 'doc_file' (line 587)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 32), 'doc_file', call_assignment_14143_16085)
            
            # finally branch of the try-finally block (line 584)
            # Deleting a member
            # Getting the type of 'sys' (line 589)
            sys_16086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 20), 'sys')
            # Obtaining the member 'path' of a type (line 589)
            path_16087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 20), sys_16086, 'path')
            
            # Obtaining the type of the subscript
            int_16088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 29), 'int')
            # Getting the type of 'sys' (line 589)
            sys_16089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 20), 'sys')
            # Obtaining the member 'path' of a type (line 589)
            path_16090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 20), sys_16089, 'path')
            # Obtaining the member '__getitem__' of a type (line 589)
            getitem___16091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 20), path_16090, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 589)
            subscript_call_result_16092 = invoke(stypy.reporting.localization.Localization(__file__, 589, 20), getitem___16091, int_16088)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 589, 16), path_16087, subscript_call_result_16092)
            
            
            # Call to add_data_files(...): (line 590)
            # Processing the call arguments (line 590)
            
            # Obtaining an instance of the builtin type 'tuple' (line 590)
            tuple_16095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 35), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 590)
            # Adding element type (line 590)
            # Getting the type of 'header_dir' (line 590)
            header_dir_16096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 35), 'header_dir', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 35), tuple_16095, header_dir_16096)
            # Adding element type (line 590)
            # Getting the type of 'h_file' (line 590)
            h_file_16097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 47), 'h_file', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 35), tuple_16095, h_file_16097)
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 591)
            tuple_16098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 35), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 591)
            # Adding element type (line 591)
            # Getting the type of 'header_dir' (line 591)
            header_dir_16099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 35), 'header_dir', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 35), tuple_16098, header_dir_16099)
            # Adding element type (line 591)
            # Getting the type of 'doc_file' (line 591)
            doc_file_16100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 47), 'doc_file', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 35), tuple_16098, doc_file_16100)
            
            # Processing the call keyword arguments (line 590)
            kwargs_16101 = {}
            # Getting the type of 'config' (line 590)
            config_16093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'config', False)
            # Obtaining the member 'add_data_files' of a type (line 590)
            add_data_files_16094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 12), config_16093, 'add_data_files')
            # Calling add_data_files(args, kwargs) (line 590)
            add_data_files_call_result_16102 = invoke(stypy.reporting.localization.Localization(__file__, 590, 12), add_data_files_16094, *[tuple_16095, tuple_16098], **kwargs_16101)
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 592)
            tuple_16103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 20), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 592)
            # Adding element type (line 592)
            # Getting the type of 'h_file' (line 592)
            h_file_16104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 20), 'h_file')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 592, 20), tuple_16103, h_file_16104)
            
            # Assigning a type to the variable 'stypy_return_type' (line 592)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 12), 'stypy_return_type', tuple_16103)
            
            # ################# End of 'generate_api(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'generate_api' in the type store
            # Getting the type of 'stypy_return_type' (line 581)
            stypy_return_type_16105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_16105)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'generate_api'
            return stypy_return_type_16105

        # Assigning a type to the variable 'generate_api' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'generate_api', generate_api)
        # Getting the type of 'generate_api' (line 593)
        generate_api_16106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 15), 'generate_api')
        # Assigning a type to the variable 'stypy_return_type' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'stypy_return_type', generate_api_16106)
        
        # ################# End of 'generate_api_func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generate_api_func' in the type store
        # Getting the type of 'stypy_return_type' (line 580)
        stypy_return_type_16107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16107)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generate_api_func'
        return stypy_return_type_16107

    # Assigning a type to the variable 'generate_api_func' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 4), 'generate_api_func', generate_api_func)
    
    # Assigning a Call to a Name (line 595):
    
    # Assigning a Call to a Name (line 595):
    
    # Call to generate_api_func(...): (line 595)
    # Processing the call arguments (line 595)
    str_16109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 43), 'str', 'generate_numpy_api')
    # Processing the call keyword arguments (line 595)
    kwargs_16110 = {}
    # Getting the type of 'generate_api_func' (line 595)
    generate_api_func_16108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 25), 'generate_api_func', False)
    # Calling generate_api_func(args, kwargs) (line 595)
    generate_api_func_call_result_16111 = invoke(stypy.reporting.localization.Localization(__file__, 595, 25), generate_api_func_16108, *[str_16109], **kwargs_16110)
    
    # Assigning a type to the variable 'generate_numpy_api' (line 595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'generate_numpy_api', generate_api_func_call_result_16111)
    
    # Assigning a Call to a Name (line 596):
    
    # Assigning a Call to a Name (line 596):
    
    # Call to generate_api_func(...): (line 596)
    # Processing the call arguments (line 596)
    str_16113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 43), 'str', 'generate_ufunc_api')
    # Processing the call keyword arguments (line 596)
    kwargs_16114 = {}
    # Getting the type of 'generate_api_func' (line 596)
    generate_api_func_16112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 25), 'generate_api_func', False)
    # Calling generate_api_func(args, kwargs) (line 596)
    generate_api_func_call_result_16115 = invoke(stypy.reporting.localization.Localization(__file__, 596, 25), generate_api_func_16112, *[str_16113], **kwargs_16114)
    
    # Assigning a type to the variable 'generate_ufunc_api' (line 596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'generate_ufunc_api', generate_api_func_call_result_16115)
    
    # Call to add_include_dirs(...): (line 598)
    # Processing the call arguments (line 598)
    
    # Call to join(...): (line 598)
    # Processing the call arguments (line 598)
    # Getting the type of 'local_dir' (line 598)
    local_dir_16119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 33), 'local_dir', False)
    str_16120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 44), 'str', 'src')
    str_16121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 51), 'str', 'private')
    # Processing the call keyword arguments (line 598)
    kwargs_16122 = {}
    # Getting the type of 'join' (line 598)
    join_16118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 28), 'join', False)
    # Calling join(args, kwargs) (line 598)
    join_call_result_16123 = invoke(stypy.reporting.localization.Localization(__file__, 598, 28), join_16118, *[local_dir_16119, str_16120, str_16121], **kwargs_16122)
    
    # Processing the call keyword arguments (line 598)
    kwargs_16124 = {}
    # Getting the type of 'config' (line 598)
    config_16116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'config', False)
    # Obtaining the member 'add_include_dirs' of a type (line 598)
    add_include_dirs_16117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 4), config_16116, 'add_include_dirs')
    # Calling add_include_dirs(args, kwargs) (line 598)
    add_include_dirs_call_result_16125 = invoke(stypy.reporting.localization.Localization(__file__, 598, 4), add_include_dirs_16117, *[join_call_result_16123], **kwargs_16124)
    
    
    # Call to add_include_dirs(...): (line 599)
    # Processing the call arguments (line 599)
    
    # Call to join(...): (line 599)
    # Processing the call arguments (line 599)
    # Getting the type of 'local_dir' (line 599)
    local_dir_16129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 33), 'local_dir', False)
    str_16130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 44), 'str', 'src')
    # Processing the call keyword arguments (line 599)
    kwargs_16131 = {}
    # Getting the type of 'join' (line 599)
    join_16128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 28), 'join', False)
    # Calling join(args, kwargs) (line 599)
    join_call_result_16132 = invoke(stypy.reporting.localization.Localization(__file__, 599, 28), join_16128, *[local_dir_16129, str_16130], **kwargs_16131)
    
    # Processing the call keyword arguments (line 599)
    kwargs_16133 = {}
    # Getting the type of 'config' (line 599)
    config_16126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 4), 'config', False)
    # Obtaining the member 'add_include_dirs' of a type (line 599)
    add_include_dirs_16127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 4), config_16126, 'add_include_dirs')
    # Calling add_include_dirs(args, kwargs) (line 599)
    add_include_dirs_call_result_16134 = invoke(stypy.reporting.localization.Localization(__file__, 599, 4), add_include_dirs_16127, *[join_call_result_16132], **kwargs_16133)
    
    
    # Call to add_include_dirs(...): (line 600)
    # Processing the call arguments (line 600)
    
    # Call to join(...): (line 600)
    # Processing the call arguments (line 600)
    # Getting the type of 'local_dir' (line 600)
    local_dir_16138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 33), 'local_dir', False)
    # Processing the call keyword arguments (line 600)
    kwargs_16139 = {}
    # Getting the type of 'join' (line 600)
    join_16137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 28), 'join', False)
    # Calling join(args, kwargs) (line 600)
    join_call_result_16140 = invoke(stypy.reporting.localization.Localization(__file__, 600, 28), join_16137, *[local_dir_16138], **kwargs_16139)
    
    # Processing the call keyword arguments (line 600)
    kwargs_16141 = {}
    # Getting the type of 'config' (line 600)
    config_16135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 4), 'config', False)
    # Obtaining the member 'add_include_dirs' of a type (line 600)
    add_include_dirs_16136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 4), config_16135, 'add_include_dirs')
    # Calling add_include_dirs(args, kwargs) (line 600)
    add_include_dirs_call_result_16142 = invoke(stypy.reporting.localization.Localization(__file__, 600, 4), add_include_dirs_16136, *[join_call_result_16140], **kwargs_16141)
    
    
    # Call to add_data_files(...): (line 602)
    # Processing the call arguments (line 602)
    str_16145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 26), 'str', 'include/numpy/*.h')
    # Processing the call keyword arguments (line 602)
    kwargs_16146 = {}
    # Getting the type of 'config' (line 602)
    config_16143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 4), 'config', False)
    # Obtaining the member 'add_data_files' of a type (line 602)
    add_data_files_16144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 4), config_16143, 'add_data_files')
    # Calling add_data_files(args, kwargs) (line 602)
    add_data_files_call_result_16147 = invoke(stypy.reporting.localization.Localization(__file__, 602, 4), add_data_files_16144, *[str_16145], **kwargs_16146)
    
    
    # Call to add_include_dirs(...): (line 603)
    # Processing the call arguments (line 603)
    
    # Call to join(...): (line 603)
    # Processing the call arguments (line 603)
    str_16151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 33), 'str', 'src')
    str_16152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 40), 'str', 'npymath')
    # Processing the call keyword arguments (line 603)
    kwargs_16153 = {}
    # Getting the type of 'join' (line 603)
    join_16150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 28), 'join', False)
    # Calling join(args, kwargs) (line 603)
    join_call_result_16154 = invoke(stypy.reporting.localization.Localization(__file__, 603, 28), join_16150, *[str_16151, str_16152], **kwargs_16153)
    
    # Processing the call keyword arguments (line 603)
    kwargs_16155 = {}
    # Getting the type of 'config' (line 603)
    config_16148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 4), 'config', False)
    # Obtaining the member 'add_include_dirs' of a type (line 603)
    add_include_dirs_16149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 4), config_16148, 'add_include_dirs')
    # Calling add_include_dirs(args, kwargs) (line 603)
    add_include_dirs_call_result_16156 = invoke(stypy.reporting.localization.Localization(__file__, 603, 4), add_include_dirs_16149, *[join_call_result_16154], **kwargs_16155)
    
    
    # Call to add_include_dirs(...): (line 604)
    # Processing the call arguments (line 604)
    
    # Call to join(...): (line 604)
    # Processing the call arguments (line 604)
    str_16160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 33), 'str', 'src')
    str_16161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 40), 'str', 'multiarray')
    # Processing the call keyword arguments (line 604)
    kwargs_16162 = {}
    # Getting the type of 'join' (line 604)
    join_16159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 28), 'join', False)
    # Calling join(args, kwargs) (line 604)
    join_call_result_16163 = invoke(stypy.reporting.localization.Localization(__file__, 604, 28), join_16159, *[str_16160, str_16161], **kwargs_16162)
    
    # Processing the call keyword arguments (line 604)
    kwargs_16164 = {}
    # Getting the type of 'config' (line 604)
    config_16157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 4), 'config', False)
    # Obtaining the member 'add_include_dirs' of a type (line 604)
    add_include_dirs_16158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 4), config_16157, 'add_include_dirs')
    # Calling add_include_dirs(args, kwargs) (line 604)
    add_include_dirs_call_result_16165 = invoke(stypy.reporting.localization.Localization(__file__, 604, 4), add_include_dirs_16158, *[join_call_result_16163], **kwargs_16164)
    
    
    # Call to add_include_dirs(...): (line 605)
    # Processing the call arguments (line 605)
    
    # Call to join(...): (line 605)
    # Processing the call arguments (line 605)
    str_16169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 33), 'str', 'src')
    str_16170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 40), 'str', 'umath')
    # Processing the call keyword arguments (line 605)
    kwargs_16171 = {}
    # Getting the type of 'join' (line 605)
    join_16168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 28), 'join', False)
    # Calling join(args, kwargs) (line 605)
    join_call_result_16172 = invoke(stypy.reporting.localization.Localization(__file__, 605, 28), join_16168, *[str_16169, str_16170], **kwargs_16171)
    
    # Processing the call keyword arguments (line 605)
    kwargs_16173 = {}
    # Getting the type of 'config' (line 605)
    config_16166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 4), 'config', False)
    # Obtaining the member 'add_include_dirs' of a type (line 605)
    add_include_dirs_16167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 4), config_16166, 'add_include_dirs')
    # Calling add_include_dirs(args, kwargs) (line 605)
    add_include_dirs_call_result_16174 = invoke(stypy.reporting.localization.Localization(__file__, 605, 4), add_include_dirs_16167, *[join_call_result_16172], **kwargs_16173)
    
    
    # Call to add_include_dirs(...): (line 606)
    # Processing the call arguments (line 606)
    
    # Call to join(...): (line 606)
    # Processing the call arguments (line 606)
    str_16178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 33), 'str', 'src')
    str_16179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 40), 'str', 'npysort')
    # Processing the call keyword arguments (line 606)
    kwargs_16180 = {}
    # Getting the type of 'join' (line 606)
    join_16177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 28), 'join', False)
    # Calling join(args, kwargs) (line 606)
    join_call_result_16181 = invoke(stypy.reporting.localization.Localization(__file__, 606, 28), join_16177, *[str_16178, str_16179], **kwargs_16180)
    
    # Processing the call keyword arguments (line 606)
    kwargs_16182 = {}
    # Getting the type of 'config' (line 606)
    config_16175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 4), 'config', False)
    # Obtaining the member 'add_include_dirs' of a type (line 606)
    add_include_dirs_16176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 4), config_16175, 'add_include_dirs')
    # Calling add_include_dirs(args, kwargs) (line 606)
    add_include_dirs_call_result_16183 = invoke(stypy.reporting.localization.Localization(__file__, 606, 4), add_include_dirs_16176, *[join_call_result_16181], **kwargs_16182)
    
    
    # Call to add_define_macros(...): (line 608)
    # Processing the call arguments (line 608)
    
    # Obtaining an instance of the builtin type 'list' (line 608)
    list_16186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 608)
    # Adding element type (line 608)
    
    # Obtaining an instance of the builtin type 'tuple' (line 608)
    tuple_16187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 608)
    # Adding element type (line 608)
    str_16188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 31), 'str', 'HAVE_NPY_CONFIG_H')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 31), tuple_16187, str_16188)
    # Adding element type (line 608)
    str_16189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 52), 'str', '1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 31), tuple_16187, str_16189)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 29), list_16186, tuple_16187)
    
    # Processing the call keyword arguments (line 608)
    kwargs_16190 = {}
    # Getting the type of 'config' (line 608)
    config_16184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 4), 'config', False)
    # Obtaining the member 'add_define_macros' of a type (line 608)
    add_define_macros_16185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 4), config_16184, 'add_define_macros')
    # Calling add_define_macros(args, kwargs) (line 608)
    add_define_macros_call_result_16191 = invoke(stypy.reporting.localization.Localization(__file__, 608, 4), add_define_macros_16185, *[list_16186], **kwargs_16190)
    
    
    # Call to add_define_macros(...): (line 609)
    # Processing the call arguments (line 609)
    
    # Obtaining an instance of the builtin type 'list' (line 609)
    list_16194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 609)
    # Adding element type (line 609)
    
    # Obtaining an instance of the builtin type 'tuple' (line 609)
    tuple_16195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 609)
    # Adding element type (line 609)
    str_16196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 31), 'str', '_FILE_OFFSET_BITS')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 609, 31), tuple_16195, str_16196)
    # Adding element type (line 609)
    str_16197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 52), 'str', '64')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 609, 31), tuple_16195, str_16197)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 609, 29), list_16194, tuple_16195)
    
    # Processing the call keyword arguments (line 609)
    kwargs_16198 = {}
    # Getting the type of 'config' (line 609)
    config_16192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 4), 'config', False)
    # Obtaining the member 'add_define_macros' of a type (line 609)
    add_define_macros_16193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 4), config_16192, 'add_define_macros')
    # Calling add_define_macros(args, kwargs) (line 609)
    add_define_macros_call_result_16199 = invoke(stypy.reporting.localization.Localization(__file__, 609, 4), add_define_macros_16193, *[list_16194], **kwargs_16198)
    
    
    # Call to add_define_macros(...): (line 610)
    # Processing the call arguments (line 610)
    
    # Obtaining an instance of the builtin type 'list' (line 610)
    list_16202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 610)
    # Adding element type (line 610)
    
    # Obtaining an instance of the builtin type 'tuple' (line 610)
    tuple_16203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 610)
    # Adding element type (line 610)
    str_16204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 31), 'str', '_LARGEFILE_SOURCE')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 31), tuple_16203, str_16204)
    # Adding element type (line 610)
    str_16205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 52), 'str', '1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 31), tuple_16203, str_16205)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 29), list_16202, tuple_16203)
    
    # Processing the call keyword arguments (line 610)
    kwargs_16206 = {}
    # Getting the type of 'config' (line 610)
    config_16200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 4), 'config', False)
    # Obtaining the member 'add_define_macros' of a type (line 610)
    add_define_macros_16201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 4), config_16200, 'add_define_macros')
    # Calling add_define_macros(args, kwargs) (line 610)
    add_define_macros_call_result_16207 = invoke(stypy.reporting.localization.Localization(__file__, 610, 4), add_define_macros_16201, *[list_16202], **kwargs_16206)
    
    
    # Call to add_define_macros(...): (line 611)
    # Processing the call arguments (line 611)
    
    # Obtaining an instance of the builtin type 'list' (line 611)
    list_16210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 611)
    # Adding element type (line 611)
    
    # Obtaining an instance of the builtin type 'tuple' (line 611)
    tuple_16211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 611)
    # Adding element type (line 611)
    str_16212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 31), 'str', '_LARGEFILE64_SOURCE')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 611, 31), tuple_16211, str_16212)
    # Adding element type (line 611)
    str_16213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 54), 'str', '1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 611, 31), tuple_16211, str_16213)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 611, 29), list_16210, tuple_16211)
    
    # Processing the call keyword arguments (line 611)
    kwargs_16214 = {}
    # Getting the type of 'config' (line 611)
    config_16208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 4), 'config', False)
    # Obtaining the member 'add_define_macros' of a type (line 611)
    add_define_macros_16209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 4), config_16208, 'add_define_macros')
    # Calling add_define_macros(args, kwargs) (line 611)
    add_define_macros_call_result_16215 = invoke(stypy.reporting.localization.Localization(__file__, 611, 4), add_define_macros_16209, *[list_16210], **kwargs_16214)
    
    
    # Call to extend(...): (line 613)
    # Processing the call arguments (line 613)
    
    # Call to paths(...): (line 613)
    # Processing the call arguments (line 613)
    str_16221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 50), 'str', 'include')
    # Processing the call keyword arguments (line 613)
    kwargs_16222 = {}
    # Getting the type of 'config' (line 613)
    config_16219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 37), 'config', False)
    # Obtaining the member 'paths' of a type (line 613)
    paths_16220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 37), config_16219, 'paths')
    # Calling paths(args, kwargs) (line 613)
    paths_call_result_16223 = invoke(stypy.reporting.localization.Localization(__file__, 613, 37), paths_16220, *[str_16221], **kwargs_16222)
    
    # Processing the call keyword arguments (line 613)
    kwargs_16224 = {}
    # Getting the type of 'config' (line 613)
    config_16216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'config', False)
    # Obtaining the member 'numpy_include_dirs' of a type (line 613)
    numpy_include_dirs_16217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 4), config_16216, 'numpy_include_dirs')
    # Obtaining the member 'extend' of a type (line 613)
    extend_16218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 4), numpy_include_dirs_16217, 'extend')
    # Calling extend(args, kwargs) (line 613)
    extend_call_result_16225 = invoke(stypy.reporting.localization.Localization(__file__, 613, 4), extend_16218, *[paths_call_result_16223], **kwargs_16224)
    
    
    # Assigning a List to a Name (line 615):
    
    # Assigning a List to a Name (line 615):
    
    # Obtaining an instance of the builtin type 'list' (line 615)
    list_16226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 615)
    # Adding element type (line 615)
    
    # Call to join(...): (line 615)
    # Processing the call arguments (line 615)
    str_16228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 17), 'str', 'src')
    str_16229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 24), 'str', 'npymath')
    str_16230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 35), 'str', '_signbit.c')
    # Processing the call keyword arguments (line 615)
    kwargs_16231 = {}
    # Getting the type of 'join' (line 615)
    join_16227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 12), 'join', False)
    # Calling join(args, kwargs) (line 615)
    join_call_result_16232 = invoke(stypy.reporting.localization.Localization(__file__, 615, 12), join_16227, *[str_16228, str_16229, str_16230], **kwargs_16231)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 11), list_16226, join_call_result_16232)
    # Adding element type (line 615)
    
    # Call to join(...): (line 616)
    # Processing the call arguments (line 616)
    str_16234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 17), 'str', 'include')
    str_16235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 28), 'str', 'numpy')
    str_16236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 37), 'str', '*object.h')
    # Processing the call keyword arguments (line 616)
    kwargs_16237 = {}
    # Getting the type of 'join' (line 616)
    join_16233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 12), 'join', False)
    # Calling join(args, kwargs) (line 616)
    join_call_result_16238 = invoke(stypy.reporting.localization.Localization(__file__, 616, 12), join_16233, *[str_16234, str_16235, str_16236], **kwargs_16237)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 11), list_16226, join_call_result_16238)
    # Adding element type (line 615)
    
    # Call to join(...): (line 617)
    # Processing the call arguments (line 617)
    # Getting the type of 'codegen_dir' (line 617)
    codegen_dir_16240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 17), 'codegen_dir', False)
    str_16241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 30), 'str', 'genapi.py')
    # Processing the call keyword arguments (line 617)
    kwargs_16242 = {}
    # Getting the type of 'join' (line 617)
    join_16239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'join', False)
    # Calling join(args, kwargs) (line 617)
    join_call_result_16243 = invoke(stypy.reporting.localization.Localization(__file__, 617, 12), join_16239, *[codegen_dir_16240, str_16241], **kwargs_16242)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 615, 11), list_16226, join_call_result_16243)
    
    # Assigning a type to the variable 'deps' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'deps', list_16226)
    
    # Call to add_extension(...): (line 630)
    # Processing the call arguments (line 630)
    str_16246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 25), 'str', '_dummy')
    # Processing the call keyword arguments (line 630)
    
    # Obtaining an instance of the builtin type 'list' (line 631)
    list_16247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 631)
    # Adding element type (line 631)
    
    # Call to join(...): (line 631)
    # Processing the call arguments (line 631)
    str_16249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 39), 'str', 'src')
    str_16250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 46), 'str', 'dummymodule.c')
    # Processing the call keyword arguments (line 631)
    kwargs_16251 = {}
    # Getting the type of 'join' (line 631)
    join_16248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 34), 'join', False)
    # Calling join(args, kwargs) (line 631)
    join_call_result_16252 = invoke(stypy.reporting.localization.Localization(__file__, 631, 34), join_16248, *[str_16249, str_16250], **kwargs_16251)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 631, 33), list_16247, join_call_result_16252)
    # Adding element type (line 631)
    # Getting the type of 'generate_config_h' (line 632)
    generate_config_h_16253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 34), 'generate_config_h', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 631, 33), list_16247, generate_config_h_16253)
    # Adding element type (line 631)
    # Getting the type of 'generate_numpyconfig_h' (line 633)
    generate_numpyconfig_h_16254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 34), 'generate_numpyconfig_h', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 631, 33), list_16247, generate_numpyconfig_h_16254)
    # Adding element type (line 631)
    # Getting the type of 'generate_numpy_api' (line 634)
    generate_numpy_api_16255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 34), 'generate_numpy_api', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 631, 33), list_16247, generate_numpy_api_16255)
    
    keyword_16256 = list_16247
    kwargs_16257 = {'sources': keyword_16256}
    # Getting the type of 'config' (line 630)
    config_16244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 630)
    add_extension_16245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 4), config_16244, 'add_extension')
    # Calling add_extension(args, kwargs) (line 630)
    add_extension_call_result_16258 = invoke(stypy.reporting.localization.Localization(__file__, 630, 4), add_extension_16245, *[str_16246], **kwargs_16257)
    
    
    # Assigning a Call to a Name (line 641):
    
    # Assigning a Call to a Name (line 641):
    
    # Call to dict(...): (line 641)
    # Processing the call arguments (line 641)
    
    # Obtaining an instance of the builtin type 'list' (line 641)
    list_16260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 641)
    # Adding element type (line 641)
    
    # Obtaining an instance of the builtin type 'tuple' (line 641)
    tuple_16261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 641)
    # Adding element type (line 641)
    str_16262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 24), 'str', 'sep')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 641, 24), tuple_16261, str_16262)
    # Adding element type (line 641)
    # Getting the type of 'os' (line 641)
    os_16263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 31), 'os', False)
    # Obtaining the member 'path' of a type (line 641)
    path_16264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 31), os_16263, 'path')
    # Obtaining the member 'sep' of a type (line 641)
    sep_16265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 31), path_16264, 'sep')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 641, 24), tuple_16261, sep_16265)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 641, 22), list_16260, tuple_16261)
    # Adding element type (line 641)
    
    # Obtaining an instance of the builtin type 'tuple' (line 641)
    tuple_16266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 641)
    # Adding element type (line 641)
    str_16267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 46), 'str', 'pkgname')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 641, 46), tuple_16266, str_16267)
    # Adding element type (line 641)
    str_16268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 57), 'str', 'numpy.core')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 641, 46), tuple_16266, str_16268)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 641, 22), list_16260, tuple_16266)
    
    # Processing the call keyword arguments (line 641)
    kwargs_16269 = {}
    # Getting the type of 'dict' (line 641)
    dict_16259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 17), 'dict', False)
    # Calling dict(args, kwargs) (line 641)
    dict_call_result_16270 = invoke(stypy.reporting.localization.Localization(__file__, 641, 17), dict_16259, *[list_16260], **kwargs_16269)
    
    # Assigning a type to the variable 'subst_dict' (line 641)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 4), 'subst_dict', dict_call_result_16270)

    @norecursion
    def get_mathlib_info(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_mathlib_info'
        module_type_store = module_type_store.open_function_context('get_mathlib_info', 643, 4, False)
        
        # Passed parameters checking function
        get_mathlib_info.stypy_localization = localization
        get_mathlib_info.stypy_type_of_self = None
        get_mathlib_info.stypy_type_store = module_type_store
        get_mathlib_info.stypy_function_name = 'get_mathlib_info'
        get_mathlib_info.stypy_param_names_list = []
        get_mathlib_info.stypy_varargs_param_name = 'args'
        get_mathlib_info.stypy_kwargs_param_name = None
        get_mathlib_info.stypy_call_defaults = defaults
        get_mathlib_info.stypy_call_varargs = varargs
        get_mathlib_info.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'get_mathlib_info', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_mathlib_info', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_mathlib_info(...)' code ##################

        
        # Assigning a Call to a Name (line 647):
        
        # Assigning a Call to a Name (line 647):
        
        # Call to get_config_cmd(...): (line 647)
        # Processing the call keyword arguments (line 647)
        kwargs_16273 = {}
        # Getting the type of 'config' (line 647)
        config_16271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 21), 'config', False)
        # Obtaining the member 'get_config_cmd' of a type (line 647)
        get_config_cmd_16272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 21), config_16271, 'get_config_cmd')
        # Calling get_config_cmd(args, kwargs) (line 647)
        get_config_cmd_call_result_16274 = invoke(stypy.reporting.localization.Localization(__file__, 647, 21), get_config_cmd_16272, *[], **kwargs_16273)
        
        # Assigning a type to the variable 'config_cmd' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'config_cmd', get_config_cmd_call_result_16274)
        
        # Assigning a Call to a Name (line 652):
        
        # Assigning a Call to a Name (line 652):
        
        # Call to try_link(...): (line 652)
        # Processing the call arguments (line 652)
        str_16277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 33), 'str', 'int main(void) { return 0;}')
        # Processing the call keyword arguments (line 652)
        kwargs_16278 = {}
        # Getting the type of 'config_cmd' (line 652)
        config_cmd_16275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 13), 'config_cmd', False)
        # Obtaining the member 'try_link' of a type (line 652)
        try_link_16276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 13), config_cmd_16275, 'try_link')
        # Calling try_link(args, kwargs) (line 652)
        try_link_call_result_16279 = invoke(stypy.reporting.localization.Localization(__file__, 652, 13), try_link_16276, *[str_16277], **kwargs_16278)
        
        # Assigning a type to the variable 'st' (line 652)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 8), 'st', try_link_call_result_16279)
        
        
        # Getting the type of 'st' (line 653)
        st_16280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 15), 'st')
        # Applying the 'not' unary operator (line 653)
        result_not__16281 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 11), 'not', st_16280)
        
        # Testing the type of an if condition (line 653)
        if_condition_16282 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 653, 8), result_not__16281)
        # Assigning a type to the variable 'if_condition_16282' (line 653)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'if_condition_16282', if_condition_16282)
        # SSA begins for if statement (line 653)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to RuntimeError(...): (line 654)
        # Processing the call arguments (line 654)
        str_16284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 31), 'str', 'Broken toolchain: cannot link a simple C program')
        # Processing the call keyword arguments (line 654)
        kwargs_16285 = {}
        # Getting the type of 'RuntimeError' (line 654)
        RuntimeError_16283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 654)
        RuntimeError_call_result_16286 = invoke(stypy.reporting.localization.Localization(__file__, 654, 18), RuntimeError_16283, *[str_16284], **kwargs_16285)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 654, 12), RuntimeError_call_result_16286, 'raise parameter', BaseException)
        # SSA join for if statement (line 653)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 655):
        
        # Assigning a Call to a Name (line 655):
        
        # Call to check_mathlib(...): (line 655)
        # Processing the call arguments (line 655)
        # Getting the type of 'config_cmd' (line 655)
        config_cmd_16288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 30), 'config_cmd', False)
        # Processing the call keyword arguments (line 655)
        kwargs_16289 = {}
        # Getting the type of 'check_mathlib' (line 655)
        check_mathlib_16287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 16), 'check_mathlib', False)
        # Calling check_mathlib(args, kwargs) (line 655)
        check_mathlib_call_result_16290 = invoke(stypy.reporting.localization.Localization(__file__, 655, 16), check_mathlib_16287, *[config_cmd_16288], **kwargs_16289)
        
        # Assigning a type to the variable 'mlibs' (line 655)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 8), 'mlibs', check_mathlib_call_result_16290)
        
        # Assigning a Call to a Name (line 657):
        
        # Assigning a Call to a Name (line 657):
        
        # Call to join(...): (line 657)
        # Processing the call arguments (line 657)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'mlibs' (line 657)
        mlibs_16296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 51), 'mlibs', False)
        comprehension_16297 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 657, 31), mlibs_16296)
        # Assigning a type to the variable 'l' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 31), 'l', comprehension_16297)
        str_16293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 31), 'str', '-l%s')
        # Getting the type of 'l' (line 657)
        l_16294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 40), 'l', False)
        # Applying the binary operator '%' (line 657)
        result_mod_16295 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 31), '%', str_16293, l_16294)
        
        list_16298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 31), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 657, 31), list_16298, result_mod_16295)
        # Processing the call keyword arguments (line 657)
        kwargs_16299 = {}
        str_16291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 21), 'str', ' ')
        # Obtaining the member 'join' of a type (line 657)
        join_16292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 21), str_16291, 'join')
        # Calling join(args, kwargs) (line 657)
        join_call_result_16300 = invoke(stypy.reporting.localization.Localization(__file__, 657, 21), join_16292, *[list_16298], **kwargs_16299)
        
        # Assigning a type to the variable 'posix_mlib' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 8), 'posix_mlib', join_call_result_16300)
        
        # Assigning a Call to a Name (line 658):
        
        # Assigning a Call to a Name (line 658):
        
        # Call to join(...): (line 658)
        # Processing the call arguments (line 658)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'mlibs' (line 658)
        mlibs_16306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 52), 'mlibs', False)
        comprehension_16307 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 658, 30), mlibs_16306)
        # Assigning a type to the variable 'l' (line 658)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 30), 'l', comprehension_16307)
        str_16303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 30), 'str', '%s.lib')
        # Getting the type of 'l' (line 658)
        l_16304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 41), 'l', False)
        # Applying the binary operator '%' (line 658)
        result_mod_16305 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 30), '%', str_16303, l_16304)
        
        list_16308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 30), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 658, 30), list_16308, result_mod_16305)
        # Processing the call keyword arguments (line 658)
        kwargs_16309 = {}
        str_16301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 20), 'str', ' ')
        # Obtaining the member 'join' of a type (line 658)
        join_16302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 20), str_16301, 'join')
        # Calling join(args, kwargs) (line 658)
        join_call_result_16310 = invoke(stypy.reporting.localization.Localization(__file__, 658, 20), join_16302, *[list_16308], **kwargs_16309)
        
        # Assigning a type to the variable 'msvc_mlib' (line 658)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 8), 'msvc_mlib', join_call_result_16310)
        
        # Assigning a Name to a Subscript (line 659):
        
        # Assigning a Name to a Subscript (line 659):
        # Getting the type of 'posix_mlib' (line 659)
        posix_mlib_16311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 38), 'posix_mlib')
        # Getting the type of 'subst_dict' (line 659)
        subst_dict_16312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 8), 'subst_dict')
        str_16313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 19), 'str', 'posix_mathlib')
        # Storing an element on a container (line 659)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 659, 8), subst_dict_16312, (str_16313, posix_mlib_16311))
        
        # Assigning a Name to a Subscript (line 660):
        
        # Assigning a Name to a Subscript (line 660):
        # Getting the type of 'msvc_mlib' (line 660)
        msvc_mlib_16314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 37), 'msvc_mlib')
        # Getting the type of 'subst_dict' (line 660)
        subst_dict_16315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 8), 'subst_dict')
        str_16316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 19), 'str', 'msvc_mathlib')
        # Storing an element on a container (line 660)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 660, 8), subst_dict_16315, (str_16316, msvc_mlib_16314))
        
        # ################# End of 'get_mathlib_info(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_mathlib_info' in the type store
        # Getting the type of 'stypy_return_type' (line 643)
        stypy_return_type_16317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16317)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_mathlib_info'
        return stypy_return_type_16317

    # Assigning a type to the variable 'get_mathlib_info' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 4), 'get_mathlib_info', get_mathlib_info)
    
    # Assigning a List to a Name (line 662):
    
    # Assigning a List to a Name (line 662):
    
    # Obtaining an instance of the builtin type 'list' (line 662)
    list_16318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 662)
    # Adding element type (line 662)
    
    # Call to join(...): (line 662)
    # Processing the call arguments (line 662)
    str_16320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 28), 'str', 'src')
    str_16321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 35), 'str', 'npymath')
    str_16322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 46), 'str', 'npy_math.c.src')
    # Processing the call keyword arguments (line 662)
    kwargs_16323 = {}
    # Getting the type of 'join' (line 662)
    join_16319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 23), 'join', False)
    # Calling join(args, kwargs) (line 662)
    join_call_result_16324 = invoke(stypy.reporting.localization.Localization(__file__, 662, 23), join_16319, *[str_16320, str_16321, str_16322], **kwargs_16323)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 662, 22), list_16318, join_call_result_16324)
    # Adding element type (line 662)
    
    # Call to join(...): (line 663)
    # Processing the call arguments (line 663)
    str_16326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 28), 'str', 'src')
    str_16327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 35), 'str', 'npymath')
    str_16328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 46), 'str', 'ieee754.c.src')
    # Processing the call keyword arguments (line 663)
    kwargs_16329 = {}
    # Getting the type of 'join' (line 663)
    join_16325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 23), 'join', False)
    # Calling join(args, kwargs) (line 663)
    join_call_result_16330 = invoke(stypy.reporting.localization.Localization(__file__, 663, 23), join_16325, *[str_16326, str_16327, str_16328], **kwargs_16329)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 662, 22), list_16318, join_call_result_16330)
    # Adding element type (line 662)
    
    # Call to join(...): (line 664)
    # Processing the call arguments (line 664)
    str_16332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 28), 'str', 'src')
    str_16333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 35), 'str', 'npymath')
    str_16334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 46), 'str', 'npy_math_complex.c.src')
    # Processing the call keyword arguments (line 664)
    kwargs_16335 = {}
    # Getting the type of 'join' (line 664)
    join_16331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 23), 'join', False)
    # Calling join(args, kwargs) (line 664)
    join_call_result_16336 = invoke(stypy.reporting.localization.Localization(__file__, 664, 23), join_16331, *[str_16332, str_16333, str_16334], **kwargs_16335)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 662, 22), list_16318, join_call_result_16336)
    # Adding element type (line 662)
    
    # Call to join(...): (line 665)
    # Processing the call arguments (line 665)
    str_16338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 28), 'str', 'src')
    str_16339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 35), 'str', 'npymath')
    str_16340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 46), 'str', 'halffloat.c')
    # Processing the call keyword arguments (line 665)
    kwargs_16341 = {}
    # Getting the type of 'join' (line 665)
    join_16337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 23), 'join', False)
    # Calling join(args, kwargs) (line 665)
    join_call_result_16342 = invoke(stypy.reporting.localization.Localization(__file__, 665, 23), join_16337, *[str_16338, str_16339, str_16340], **kwargs_16341)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 662, 22), list_16318, join_call_result_16342)
    
    # Assigning a type to the variable 'npymath_sources' (line 662)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 4), 'npymath_sources', list_16318)
    
    # Call to add_installed_library(...): (line 667)
    # Processing the call arguments (line 667)
    str_16345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 33), 'str', 'npymath')
    # Processing the call keyword arguments (line 667)
    # Getting the type of 'npymath_sources' (line 668)
    npymath_sources_16346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 20), 'npymath_sources', False)
    
    # Obtaining an instance of the builtin type 'list' (line 668)
    list_16347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 668)
    # Adding element type (line 668)
    # Getting the type of 'get_mathlib_info' (line 668)
    get_mathlib_info_16348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 39), 'get_mathlib_info', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 668, 38), list_16347, get_mathlib_info_16348)
    
    # Applying the binary operator '+' (line 668)
    result_add_16349 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 20), '+', npymath_sources_16346, list_16347)
    
    keyword_16350 = result_add_16349
    str_16351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 24), 'str', 'lib')
    keyword_16352 = str_16351
    kwargs_16353 = {'sources': keyword_16350, 'install_dir': keyword_16352}
    # Getting the type of 'config' (line 667)
    config_16343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 4), 'config', False)
    # Obtaining the member 'add_installed_library' of a type (line 667)
    add_installed_library_16344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 4), config_16343, 'add_installed_library')
    # Calling add_installed_library(args, kwargs) (line 667)
    add_installed_library_call_result_16354 = invoke(stypy.reporting.localization.Localization(__file__, 667, 4), add_installed_library_16344, *[str_16345], **kwargs_16353)
    
    
    # Call to add_npy_pkg_config(...): (line 670)
    # Processing the call arguments (line 670)
    str_16357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 30), 'str', 'npymath.ini.in')
    str_16358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 48), 'str', 'lib/npy-pkg-config')
    # Getting the type of 'subst_dict' (line 671)
    subst_dict_16359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 12), 'subst_dict', False)
    # Processing the call keyword arguments (line 670)
    kwargs_16360 = {}
    # Getting the type of 'config' (line 670)
    config_16355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 4), 'config', False)
    # Obtaining the member 'add_npy_pkg_config' of a type (line 670)
    add_npy_pkg_config_16356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 4), config_16355, 'add_npy_pkg_config')
    # Calling add_npy_pkg_config(args, kwargs) (line 670)
    add_npy_pkg_config_call_result_16361 = invoke(stypy.reporting.localization.Localization(__file__, 670, 4), add_npy_pkg_config_16356, *[str_16357, str_16358, subst_dict_16359], **kwargs_16360)
    
    
    # Call to add_npy_pkg_config(...): (line 672)
    # Processing the call arguments (line 672)
    str_16364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 30), 'str', 'mlib.ini.in')
    str_16365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 45), 'str', 'lib/npy-pkg-config')
    # Getting the type of 'subst_dict' (line 673)
    subst_dict_16366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 12), 'subst_dict', False)
    # Processing the call keyword arguments (line 672)
    kwargs_16367 = {}
    # Getting the type of 'config' (line 672)
    config_16362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 4), 'config', False)
    # Obtaining the member 'add_npy_pkg_config' of a type (line 672)
    add_npy_pkg_config_16363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 4), config_16362, 'add_npy_pkg_config')
    # Calling add_npy_pkg_config(args, kwargs) (line 672)
    add_npy_pkg_config_call_result_16368 = invoke(stypy.reporting.localization.Localization(__file__, 672, 4), add_npy_pkg_config_16363, *[str_16364, str_16365, subst_dict_16366], **kwargs_16367)
    
    
    # Assigning a List to a Name (line 680):
    
    # Assigning a List to a Name (line 680):
    
    # Obtaining an instance of the builtin type 'list' (line 680)
    list_16369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 680)
    # Adding element type (line 680)
    
    # Call to join(...): (line 680)
    # Processing the call arguments (line 680)
    str_16371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 28), 'str', 'src')
    str_16372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 35), 'str', 'npysort')
    str_16373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 46), 'str', 'quicksort.c.src')
    # Processing the call keyword arguments (line 680)
    kwargs_16374 = {}
    # Getting the type of 'join' (line 680)
    join_16370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 23), 'join', False)
    # Calling join(args, kwargs) (line 680)
    join_call_result_16375 = invoke(stypy.reporting.localization.Localization(__file__, 680, 23), join_16370, *[str_16371, str_16372, str_16373], **kwargs_16374)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 22), list_16369, join_call_result_16375)
    # Adding element type (line 680)
    
    # Call to join(...): (line 681)
    # Processing the call arguments (line 681)
    str_16377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 28), 'str', 'src')
    str_16378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 35), 'str', 'npysort')
    str_16379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 46), 'str', 'mergesort.c.src')
    # Processing the call keyword arguments (line 681)
    kwargs_16380 = {}
    # Getting the type of 'join' (line 681)
    join_16376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 23), 'join', False)
    # Calling join(args, kwargs) (line 681)
    join_call_result_16381 = invoke(stypy.reporting.localization.Localization(__file__, 681, 23), join_16376, *[str_16377, str_16378, str_16379], **kwargs_16380)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 22), list_16369, join_call_result_16381)
    # Adding element type (line 680)
    
    # Call to join(...): (line 682)
    # Processing the call arguments (line 682)
    str_16383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 28), 'str', 'src')
    str_16384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 35), 'str', 'npysort')
    str_16385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 46), 'str', 'heapsort.c.src')
    # Processing the call keyword arguments (line 682)
    kwargs_16386 = {}
    # Getting the type of 'join' (line 682)
    join_16382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 23), 'join', False)
    # Calling join(args, kwargs) (line 682)
    join_call_result_16387 = invoke(stypy.reporting.localization.Localization(__file__, 682, 23), join_16382, *[str_16383, str_16384, str_16385], **kwargs_16386)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 22), list_16369, join_call_result_16387)
    # Adding element type (line 680)
    
    # Call to join(...): (line 683)
    # Processing the call arguments (line 683)
    str_16389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 28), 'str', 'src')
    str_16390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 35), 'str', 'private')
    str_16391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 46), 'str', 'npy_partition.h.src')
    # Processing the call keyword arguments (line 683)
    kwargs_16392 = {}
    # Getting the type of 'join' (line 683)
    join_16388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 23), 'join', False)
    # Calling join(args, kwargs) (line 683)
    join_call_result_16393 = invoke(stypy.reporting.localization.Localization(__file__, 683, 23), join_16388, *[str_16389, str_16390, str_16391], **kwargs_16392)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 22), list_16369, join_call_result_16393)
    # Adding element type (line 680)
    
    # Call to join(...): (line 684)
    # Processing the call arguments (line 684)
    str_16395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 28), 'str', 'src')
    str_16396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 35), 'str', 'npysort')
    str_16397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 46), 'str', 'selection.c.src')
    # Processing the call keyword arguments (line 684)
    kwargs_16398 = {}
    # Getting the type of 'join' (line 684)
    join_16394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 23), 'join', False)
    # Calling join(args, kwargs) (line 684)
    join_call_result_16399 = invoke(stypy.reporting.localization.Localization(__file__, 684, 23), join_16394, *[str_16395, str_16396, str_16397], **kwargs_16398)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 22), list_16369, join_call_result_16399)
    # Adding element type (line 680)
    
    # Call to join(...): (line 685)
    # Processing the call arguments (line 685)
    str_16401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 28), 'str', 'src')
    str_16402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 35), 'str', 'private')
    str_16403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 46), 'str', 'npy_binsearch.h.src')
    # Processing the call keyword arguments (line 685)
    kwargs_16404 = {}
    # Getting the type of 'join' (line 685)
    join_16400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 23), 'join', False)
    # Calling join(args, kwargs) (line 685)
    join_call_result_16405 = invoke(stypy.reporting.localization.Localization(__file__, 685, 23), join_16400, *[str_16401, str_16402, str_16403], **kwargs_16404)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 22), list_16369, join_call_result_16405)
    # Adding element type (line 680)
    
    # Call to join(...): (line 686)
    # Processing the call arguments (line 686)
    str_16407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 28), 'str', 'src')
    str_16408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 35), 'str', 'npysort')
    str_16409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 46), 'str', 'binsearch.c.src')
    # Processing the call keyword arguments (line 686)
    kwargs_16410 = {}
    # Getting the type of 'join' (line 686)
    join_16406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 23), 'join', False)
    # Calling join(args, kwargs) (line 686)
    join_call_result_16411 = invoke(stypy.reporting.localization.Localization(__file__, 686, 23), join_16406, *[str_16407, str_16408, str_16409], **kwargs_16410)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 22), list_16369, join_call_result_16411)
    
    # Assigning a type to the variable 'npysort_sources' (line 680)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 4), 'npysort_sources', list_16369)
    
    # Call to add_library(...): (line 688)
    # Processing the call arguments (line 688)
    str_16414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 23), 'str', 'npysort')
    # Processing the call keyword arguments (line 688)
    # Getting the type of 'npysort_sources' (line 689)
    npysort_sources_16415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 31), 'npysort_sources', False)
    keyword_16416 = npysort_sources_16415
    
    # Obtaining an instance of the builtin type 'list' (line 690)
    list_16417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 690)
    
    keyword_16418 = list_16417
    kwargs_16419 = {'sources': keyword_16416, 'include_dirs': keyword_16418}
    # Getting the type of 'config' (line 688)
    config_16412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 688)
    add_library_16413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 4), config_16412, 'add_library')
    # Calling add_library(args, kwargs) (line 688)
    add_library_call_result_16420 = invoke(stypy.reporting.localization.Localization(__file__, 688, 4), add_library_16413, *[str_16414], **kwargs_16419)
    

    @norecursion
    def generate_multiarray_templated_sources(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'generate_multiarray_templated_sources'
        module_type_store = module_type_store.open_function_context('generate_multiarray_templated_sources', 699, 4, False)
        
        # Passed parameters checking function
        generate_multiarray_templated_sources.stypy_localization = localization
        generate_multiarray_templated_sources.stypy_type_of_self = None
        generate_multiarray_templated_sources.stypy_type_store = module_type_store
        generate_multiarray_templated_sources.stypy_function_name = 'generate_multiarray_templated_sources'
        generate_multiarray_templated_sources.stypy_param_names_list = ['ext', 'build_dir']
        generate_multiarray_templated_sources.stypy_varargs_param_name = None
        generate_multiarray_templated_sources.stypy_kwargs_param_name = None
        generate_multiarray_templated_sources.stypy_call_defaults = defaults
        generate_multiarray_templated_sources.stypy_call_varargs = varargs
        generate_multiarray_templated_sources.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'generate_multiarray_templated_sources', ['ext', 'build_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'generate_multiarray_templated_sources', localization, ['ext', 'build_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'generate_multiarray_templated_sources(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 700, 8))
        
        # 'from numpy.distutils.misc_util import get_cmd' statement (line 700)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
        import_16421 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 700, 8), 'numpy.distutils.misc_util')

        if (type(import_16421) is not StypyTypeError):

            if (import_16421 != 'pyd_module'):
                __import__(import_16421)
                sys_modules_16422 = sys.modules[import_16421]
                import_from_module(stypy.reporting.localization.Localization(__file__, 700, 8), 'numpy.distutils.misc_util', sys_modules_16422.module_type_store, module_type_store, ['get_cmd'])
                nest_module(stypy.reporting.localization.Localization(__file__, 700, 8), __file__, sys_modules_16422, sys_modules_16422.module_type_store, module_type_store)
            else:
                from numpy.distutils.misc_util import get_cmd

                import_from_module(stypy.reporting.localization.Localization(__file__, 700, 8), 'numpy.distutils.misc_util', None, module_type_store, ['get_cmd'], [get_cmd])

        else:
            # Assigning a type to the variable 'numpy.distutils.misc_util' (line 700)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 8), 'numpy.distutils.misc_util', import_16421)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')
        
        
        # Assigning a Call to a Name (line 702):
        
        # Assigning a Call to a Name (line 702):
        
        # Call to join(...): (line 702)
        # Processing the call arguments (line 702)
        str_16424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 23), 'str', 'src')
        str_16425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 30), 'str', 'multiarray')
        # Processing the call keyword arguments (line 702)
        kwargs_16426 = {}
        # Getting the type of 'join' (line 702)
        join_16423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 18), 'join', False)
        # Calling join(args, kwargs) (line 702)
        join_call_result_16427 = invoke(stypy.reporting.localization.Localization(__file__, 702, 18), join_16423, *[str_16424, str_16425], **kwargs_16426)
        
        # Assigning a type to the variable 'subpath' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'subpath', join_call_result_16427)
        
        # Assigning a List to a Name (line 703):
        
        # Assigning a List to a Name (line 703):
        
        # Obtaining an instance of the builtin type 'list' (line 703)
        list_16428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 703)
        # Adding element type (line 703)
        
        # Call to join(...): (line 703)
        # Processing the call arguments (line 703)
        # Getting the type of 'local_dir' (line 703)
        local_dir_16430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 24), 'local_dir', False)
        # Getting the type of 'subpath' (line 703)
        subpath_16431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 35), 'subpath', False)
        str_16432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 44), 'str', 'scalartypes.c.src')
        # Processing the call keyword arguments (line 703)
        kwargs_16433 = {}
        # Getting the type of 'join' (line 703)
        join_16429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 19), 'join', False)
        # Calling join(args, kwargs) (line 703)
        join_call_result_16434 = invoke(stypy.reporting.localization.Localization(__file__, 703, 19), join_16429, *[local_dir_16430, subpath_16431, str_16432], **kwargs_16433)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 703, 18), list_16428, join_call_result_16434)
        # Adding element type (line 703)
        
        # Call to join(...): (line 704)
        # Processing the call arguments (line 704)
        # Getting the type of 'local_dir' (line 704)
        local_dir_16436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 24), 'local_dir', False)
        # Getting the type of 'subpath' (line 704)
        subpath_16437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 35), 'subpath', False)
        str_16438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 44), 'str', 'arraytypes.c.src')
        # Processing the call keyword arguments (line 704)
        kwargs_16439 = {}
        # Getting the type of 'join' (line 704)
        join_16435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 19), 'join', False)
        # Calling join(args, kwargs) (line 704)
        join_call_result_16440 = invoke(stypy.reporting.localization.Localization(__file__, 704, 19), join_16435, *[local_dir_16436, subpath_16437, str_16438], **kwargs_16439)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 703, 18), list_16428, join_call_result_16440)
        # Adding element type (line 703)
        
        # Call to join(...): (line 705)
        # Processing the call arguments (line 705)
        # Getting the type of 'local_dir' (line 705)
        local_dir_16442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 24), 'local_dir', False)
        # Getting the type of 'subpath' (line 705)
        subpath_16443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 35), 'subpath', False)
        str_16444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 44), 'str', 'nditer_templ.c.src')
        # Processing the call keyword arguments (line 705)
        kwargs_16445 = {}
        # Getting the type of 'join' (line 705)
        join_16441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 19), 'join', False)
        # Calling join(args, kwargs) (line 705)
        join_call_result_16446 = invoke(stypy.reporting.localization.Localization(__file__, 705, 19), join_16441, *[local_dir_16442, subpath_16443, str_16444], **kwargs_16445)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 703, 18), list_16428, join_call_result_16446)
        # Adding element type (line 703)
        
        # Call to join(...): (line 706)
        # Processing the call arguments (line 706)
        # Getting the type of 'local_dir' (line 706)
        local_dir_16448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 24), 'local_dir', False)
        # Getting the type of 'subpath' (line 706)
        subpath_16449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 35), 'subpath', False)
        str_16450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 44), 'str', 'lowlevel_strided_loops.c.src')
        # Processing the call keyword arguments (line 706)
        kwargs_16451 = {}
        # Getting the type of 'join' (line 706)
        join_16447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 19), 'join', False)
        # Calling join(args, kwargs) (line 706)
        join_call_result_16452 = invoke(stypy.reporting.localization.Localization(__file__, 706, 19), join_16447, *[local_dir_16448, subpath_16449, str_16450], **kwargs_16451)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 703, 18), list_16428, join_call_result_16452)
        # Adding element type (line 703)
        
        # Call to join(...): (line 707)
        # Processing the call arguments (line 707)
        # Getting the type of 'local_dir' (line 707)
        local_dir_16454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 24), 'local_dir', False)
        # Getting the type of 'subpath' (line 707)
        subpath_16455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 35), 'subpath', False)
        str_16456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 44), 'str', 'einsum.c.src')
        # Processing the call keyword arguments (line 707)
        kwargs_16457 = {}
        # Getting the type of 'join' (line 707)
        join_16453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 19), 'join', False)
        # Calling join(args, kwargs) (line 707)
        join_call_result_16458 = invoke(stypy.reporting.localization.Localization(__file__, 707, 19), join_16453, *[local_dir_16454, subpath_16455, str_16456], **kwargs_16457)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 703, 18), list_16428, join_call_result_16458)
        # Adding element type (line 703)
        
        # Call to join(...): (line 708)
        # Processing the call arguments (line 708)
        # Getting the type of 'local_dir' (line 708)
        local_dir_16460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 24), 'local_dir', False)
        str_16461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 35), 'str', 'src')
        str_16462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 42), 'str', 'private')
        str_16463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 53), 'str', 'templ_common.h.src')
        # Processing the call keyword arguments (line 708)
        kwargs_16464 = {}
        # Getting the type of 'join' (line 708)
        join_16459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 19), 'join', False)
        # Calling join(args, kwargs) (line 708)
        join_call_result_16465 = invoke(stypy.reporting.localization.Localization(__file__, 708, 19), join_16459, *[local_dir_16460, str_16461, str_16462, str_16463], **kwargs_16464)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 703, 18), list_16428, join_call_result_16465)
        
        # Assigning a type to the variable 'sources' (line 703)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'sources', list_16428)
        
        # Call to add_include_dirs(...): (line 713)
        # Processing the call arguments (line 713)
        
        # Call to join(...): (line 713)
        # Processing the call arguments (line 713)
        # Getting the type of 'build_dir' (line 713)
        build_dir_16469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 37), 'build_dir', False)
        # Getting the type of 'subpath' (line 713)
        subpath_16470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 48), 'subpath', False)
        # Processing the call keyword arguments (line 713)
        kwargs_16471 = {}
        # Getting the type of 'join' (line 713)
        join_16468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 32), 'join', False)
        # Calling join(args, kwargs) (line 713)
        join_call_result_16472 = invoke(stypy.reporting.localization.Localization(__file__, 713, 32), join_16468, *[build_dir_16469, subpath_16470], **kwargs_16471)
        
        # Processing the call keyword arguments (line 713)
        kwargs_16473 = {}
        # Getting the type of 'config' (line 713)
        config_16466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 8), 'config', False)
        # Obtaining the member 'add_include_dirs' of a type (line 713)
        add_include_dirs_16467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 8), config_16466, 'add_include_dirs')
        # Calling add_include_dirs(args, kwargs) (line 713)
        add_include_dirs_call_result_16474 = invoke(stypy.reporting.localization.Localization(__file__, 713, 8), add_include_dirs_16467, *[join_call_result_16472], **kwargs_16473)
        
        
        # Assigning a Call to a Name (line 714):
        
        # Assigning a Call to a Name (line 714):
        
        # Call to get_cmd(...): (line 714)
        # Processing the call arguments (line 714)
        str_16476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 22), 'str', 'build_src')
        # Processing the call keyword arguments (line 714)
        kwargs_16477 = {}
        # Getting the type of 'get_cmd' (line 714)
        get_cmd_16475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 14), 'get_cmd', False)
        # Calling get_cmd(args, kwargs) (line 714)
        get_cmd_call_result_16478 = invoke(stypy.reporting.localization.Localization(__file__, 714, 14), get_cmd_16475, *[str_16476], **kwargs_16477)
        
        # Assigning a type to the variable 'cmd' (line 714)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 8), 'cmd', get_cmd_call_result_16478)
        
        # Call to ensure_finalized(...): (line 715)
        # Processing the call keyword arguments (line 715)
        kwargs_16481 = {}
        # Getting the type of 'cmd' (line 715)
        cmd_16479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 715)
        ensure_finalized_16480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 8), cmd_16479, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 715)
        ensure_finalized_call_result_16482 = invoke(stypy.reporting.localization.Localization(__file__, 715, 8), ensure_finalized_16480, *[], **kwargs_16481)
        
        
        # Call to template_sources(...): (line 716)
        # Processing the call arguments (line 716)
        # Getting the type of 'sources' (line 716)
        sources_16485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 29), 'sources', False)
        # Getting the type of 'ext' (line 716)
        ext_16486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 38), 'ext', False)
        # Processing the call keyword arguments (line 716)
        kwargs_16487 = {}
        # Getting the type of 'cmd' (line 716)
        cmd_16483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 8), 'cmd', False)
        # Obtaining the member 'template_sources' of a type (line 716)
        template_sources_16484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 8), cmd_16483, 'template_sources')
        # Calling template_sources(args, kwargs) (line 716)
        template_sources_call_result_16488 = invoke(stypy.reporting.localization.Localization(__file__, 716, 8), template_sources_16484, *[sources_16485, ext_16486], **kwargs_16487)
        
        
        # ################# End of 'generate_multiarray_templated_sources(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generate_multiarray_templated_sources' in the type store
        # Getting the type of 'stypy_return_type' (line 699)
        stypy_return_type_16489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16489)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generate_multiarray_templated_sources'
        return stypy_return_type_16489

    # Assigning a type to the variable 'generate_multiarray_templated_sources' (line 699)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 4), 'generate_multiarray_templated_sources', generate_multiarray_templated_sources)
    
    # Assigning a BinOp to a Name (line 718):
    
    # Assigning a BinOp to a Name (line 718):
    
    # Obtaining an instance of the builtin type 'list' (line 718)
    list_16490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 718)
    # Adding element type (line 718)
    
    # Call to join(...): (line 719)
    # Processing the call arguments (line 719)
    str_16492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 17), 'str', 'src')
    str_16493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 24), 'str', 'multiarray')
    str_16494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 38), 'str', 'arrayobject.h')
    # Processing the call keyword arguments (line 719)
    kwargs_16495 = {}
    # Getting the type of 'join' (line 719)
    join_16491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 12), 'join', False)
    # Calling join(args, kwargs) (line 719)
    join_call_result_16496 = invoke(stypy.reporting.localization.Localization(__file__, 719, 12), join_16491, *[str_16492, str_16493, str_16494], **kwargs_16495)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16496)
    # Adding element type (line 718)
    
    # Call to join(...): (line 720)
    # Processing the call arguments (line 720)
    str_16498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 17), 'str', 'src')
    str_16499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 24), 'str', 'multiarray')
    str_16500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 38), 'str', 'arraytypes.h')
    # Processing the call keyword arguments (line 720)
    kwargs_16501 = {}
    # Getting the type of 'join' (line 720)
    join_16497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 12), 'join', False)
    # Calling join(args, kwargs) (line 720)
    join_call_result_16502 = invoke(stypy.reporting.localization.Localization(__file__, 720, 12), join_16497, *[str_16498, str_16499, str_16500], **kwargs_16501)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16502)
    # Adding element type (line 718)
    
    # Call to join(...): (line 721)
    # Processing the call arguments (line 721)
    str_16504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 17), 'str', 'src')
    str_16505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 24), 'str', 'multiarray')
    str_16506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 38), 'str', 'array_assign.h')
    # Processing the call keyword arguments (line 721)
    kwargs_16507 = {}
    # Getting the type of 'join' (line 721)
    join_16503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 12), 'join', False)
    # Calling join(args, kwargs) (line 721)
    join_call_result_16508 = invoke(stypy.reporting.localization.Localization(__file__, 721, 12), join_16503, *[str_16504, str_16505, str_16506], **kwargs_16507)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16508)
    # Adding element type (line 718)
    
    # Call to join(...): (line 722)
    # Processing the call arguments (line 722)
    str_16510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 17), 'str', 'src')
    str_16511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 24), 'str', 'multiarray')
    str_16512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 38), 'str', 'buffer.h')
    # Processing the call keyword arguments (line 722)
    kwargs_16513 = {}
    # Getting the type of 'join' (line 722)
    join_16509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 12), 'join', False)
    # Calling join(args, kwargs) (line 722)
    join_call_result_16514 = invoke(stypy.reporting.localization.Localization(__file__, 722, 12), join_16509, *[str_16510, str_16511, str_16512], **kwargs_16513)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16514)
    # Adding element type (line 718)
    
    # Call to join(...): (line 723)
    # Processing the call arguments (line 723)
    str_16516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 17), 'str', 'src')
    str_16517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 24), 'str', 'multiarray')
    str_16518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 38), 'str', 'calculation.h')
    # Processing the call keyword arguments (line 723)
    kwargs_16519 = {}
    # Getting the type of 'join' (line 723)
    join_16515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 12), 'join', False)
    # Calling join(args, kwargs) (line 723)
    join_call_result_16520 = invoke(stypy.reporting.localization.Localization(__file__, 723, 12), join_16515, *[str_16516, str_16517, str_16518], **kwargs_16519)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16520)
    # Adding element type (line 718)
    
    # Call to join(...): (line 724)
    # Processing the call arguments (line 724)
    str_16522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 17), 'str', 'src')
    str_16523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 24), 'str', 'multiarray')
    str_16524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 38), 'str', 'cblasfuncs.h')
    # Processing the call keyword arguments (line 724)
    kwargs_16525 = {}
    # Getting the type of 'join' (line 724)
    join_16521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 12), 'join', False)
    # Calling join(args, kwargs) (line 724)
    join_call_result_16526 = invoke(stypy.reporting.localization.Localization(__file__, 724, 12), join_16521, *[str_16522, str_16523, str_16524], **kwargs_16525)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16526)
    # Adding element type (line 718)
    
    # Call to join(...): (line 725)
    # Processing the call arguments (line 725)
    str_16528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 17), 'str', 'src')
    str_16529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 24), 'str', 'multiarray')
    str_16530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 38), 'str', 'common.h')
    # Processing the call keyword arguments (line 725)
    kwargs_16531 = {}
    # Getting the type of 'join' (line 725)
    join_16527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 12), 'join', False)
    # Calling join(args, kwargs) (line 725)
    join_call_result_16532 = invoke(stypy.reporting.localization.Localization(__file__, 725, 12), join_16527, *[str_16528, str_16529, str_16530], **kwargs_16531)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16532)
    # Adding element type (line 718)
    
    # Call to join(...): (line 726)
    # Processing the call arguments (line 726)
    str_16534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 17), 'str', 'src')
    str_16535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 24), 'str', 'multiarray')
    str_16536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 38), 'str', 'convert_datatype.h')
    # Processing the call keyword arguments (line 726)
    kwargs_16537 = {}
    # Getting the type of 'join' (line 726)
    join_16533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 12), 'join', False)
    # Calling join(args, kwargs) (line 726)
    join_call_result_16538 = invoke(stypy.reporting.localization.Localization(__file__, 726, 12), join_16533, *[str_16534, str_16535, str_16536], **kwargs_16537)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16538)
    # Adding element type (line 718)
    
    # Call to join(...): (line 727)
    # Processing the call arguments (line 727)
    str_16540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 17), 'str', 'src')
    str_16541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 24), 'str', 'multiarray')
    str_16542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 38), 'str', 'convert.h')
    # Processing the call keyword arguments (line 727)
    kwargs_16543 = {}
    # Getting the type of 'join' (line 727)
    join_16539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 12), 'join', False)
    # Calling join(args, kwargs) (line 727)
    join_call_result_16544 = invoke(stypy.reporting.localization.Localization(__file__, 727, 12), join_16539, *[str_16540, str_16541, str_16542], **kwargs_16543)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16544)
    # Adding element type (line 718)
    
    # Call to join(...): (line 728)
    # Processing the call arguments (line 728)
    str_16546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 17), 'str', 'src')
    str_16547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 24), 'str', 'multiarray')
    str_16548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 38), 'str', 'conversion_utils.h')
    # Processing the call keyword arguments (line 728)
    kwargs_16549 = {}
    # Getting the type of 'join' (line 728)
    join_16545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 12), 'join', False)
    # Calling join(args, kwargs) (line 728)
    join_call_result_16550 = invoke(stypy.reporting.localization.Localization(__file__, 728, 12), join_16545, *[str_16546, str_16547, str_16548], **kwargs_16549)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16550)
    # Adding element type (line 718)
    
    # Call to join(...): (line 729)
    # Processing the call arguments (line 729)
    str_16552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 17), 'str', 'src')
    str_16553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 24), 'str', 'multiarray')
    str_16554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 38), 'str', 'ctors.h')
    # Processing the call keyword arguments (line 729)
    kwargs_16555 = {}
    # Getting the type of 'join' (line 729)
    join_16551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 12), 'join', False)
    # Calling join(args, kwargs) (line 729)
    join_call_result_16556 = invoke(stypy.reporting.localization.Localization(__file__, 729, 12), join_16551, *[str_16552, str_16553, str_16554], **kwargs_16555)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16556)
    # Adding element type (line 718)
    
    # Call to join(...): (line 730)
    # Processing the call arguments (line 730)
    str_16558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 17), 'str', 'src')
    str_16559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 24), 'str', 'multiarray')
    str_16560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 38), 'str', 'descriptor.h')
    # Processing the call keyword arguments (line 730)
    kwargs_16561 = {}
    # Getting the type of 'join' (line 730)
    join_16557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 12), 'join', False)
    # Calling join(args, kwargs) (line 730)
    join_call_result_16562 = invoke(stypy.reporting.localization.Localization(__file__, 730, 12), join_16557, *[str_16558, str_16559, str_16560], **kwargs_16561)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16562)
    # Adding element type (line 718)
    
    # Call to join(...): (line 731)
    # Processing the call arguments (line 731)
    str_16564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 17), 'str', 'src')
    str_16565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 24), 'str', 'multiarray')
    str_16566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 38), 'str', 'getset.h')
    # Processing the call keyword arguments (line 731)
    kwargs_16567 = {}
    # Getting the type of 'join' (line 731)
    join_16563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 12), 'join', False)
    # Calling join(args, kwargs) (line 731)
    join_call_result_16568 = invoke(stypy.reporting.localization.Localization(__file__, 731, 12), join_16563, *[str_16564, str_16565, str_16566], **kwargs_16567)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16568)
    # Adding element type (line 718)
    
    # Call to join(...): (line 732)
    # Processing the call arguments (line 732)
    str_16570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 17), 'str', 'src')
    str_16571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 24), 'str', 'multiarray')
    str_16572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 38), 'str', 'hashdescr.h')
    # Processing the call keyword arguments (line 732)
    kwargs_16573 = {}
    # Getting the type of 'join' (line 732)
    join_16569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 12), 'join', False)
    # Calling join(args, kwargs) (line 732)
    join_call_result_16574 = invoke(stypy.reporting.localization.Localization(__file__, 732, 12), join_16569, *[str_16570, str_16571, str_16572], **kwargs_16573)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16574)
    # Adding element type (line 718)
    
    # Call to join(...): (line 733)
    # Processing the call arguments (line 733)
    str_16576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 17), 'str', 'src')
    str_16577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 24), 'str', 'multiarray')
    str_16578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 38), 'str', 'iterators.h')
    # Processing the call keyword arguments (line 733)
    kwargs_16579 = {}
    # Getting the type of 'join' (line 733)
    join_16575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 12), 'join', False)
    # Calling join(args, kwargs) (line 733)
    join_call_result_16580 = invoke(stypy.reporting.localization.Localization(__file__, 733, 12), join_16575, *[str_16576, str_16577, str_16578], **kwargs_16579)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16580)
    # Adding element type (line 718)
    
    # Call to join(...): (line 734)
    # Processing the call arguments (line 734)
    str_16582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 17), 'str', 'src')
    str_16583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 24), 'str', 'multiarray')
    str_16584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 38), 'str', 'mapping.h')
    # Processing the call keyword arguments (line 734)
    kwargs_16585 = {}
    # Getting the type of 'join' (line 734)
    join_16581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 12), 'join', False)
    # Calling join(args, kwargs) (line 734)
    join_call_result_16586 = invoke(stypy.reporting.localization.Localization(__file__, 734, 12), join_16581, *[str_16582, str_16583, str_16584], **kwargs_16585)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16586)
    # Adding element type (line 718)
    
    # Call to join(...): (line 735)
    # Processing the call arguments (line 735)
    str_16588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 17), 'str', 'src')
    str_16589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 24), 'str', 'multiarray')
    str_16590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 38), 'str', 'methods.h')
    # Processing the call keyword arguments (line 735)
    kwargs_16591 = {}
    # Getting the type of 'join' (line 735)
    join_16587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 12), 'join', False)
    # Calling join(args, kwargs) (line 735)
    join_call_result_16592 = invoke(stypy.reporting.localization.Localization(__file__, 735, 12), join_16587, *[str_16588, str_16589, str_16590], **kwargs_16591)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16592)
    # Adding element type (line 718)
    
    # Call to join(...): (line 736)
    # Processing the call arguments (line 736)
    str_16594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 17), 'str', 'src')
    str_16595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 24), 'str', 'multiarray')
    str_16596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 38), 'str', 'multiarraymodule.h')
    # Processing the call keyword arguments (line 736)
    kwargs_16597 = {}
    # Getting the type of 'join' (line 736)
    join_16593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 12), 'join', False)
    # Calling join(args, kwargs) (line 736)
    join_call_result_16598 = invoke(stypy.reporting.localization.Localization(__file__, 736, 12), join_16593, *[str_16594, str_16595, str_16596], **kwargs_16597)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16598)
    # Adding element type (line 718)
    
    # Call to join(...): (line 737)
    # Processing the call arguments (line 737)
    str_16600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 17), 'str', 'src')
    str_16601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 24), 'str', 'multiarray')
    str_16602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 38), 'str', 'nditer_impl.h')
    # Processing the call keyword arguments (line 737)
    kwargs_16603 = {}
    # Getting the type of 'join' (line 737)
    join_16599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 12), 'join', False)
    # Calling join(args, kwargs) (line 737)
    join_call_result_16604 = invoke(stypy.reporting.localization.Localization(__file__, 737, 12), join_16599, *[str_16600, str_16601, str_16602], **kwargs_16603)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16604)
    # Adding element type (line 718)
    
    # Call to join(...): (line 738)
    # Processing the call arguments (line 738)
    str_16606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 17), 'str', 'src')
    str_16607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 24), 'str', 'multiarray')
    str_16608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 38), 'str', 'numpymemoryview.h')
    # Processing the call keyword arguments (line 738)
    kwargs_16609 = {}
    # Getting the type of 'join' (line 738)
    join_16605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'join', False)
    # Calling join(args, kwargs) (line 738)
    join_call_result_16610 = invoke(stypy.reporting.localization.Localization(__file__, 738, 12), join_16605, *[str_16606, str_16607, str_16608], **kwargs_16609)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16610)
    # Adding element type (line 718)
    
    # Call to join(...): (line 739)
    # Processing the call arguments (line 739)
    str_16612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 17), 'str', 'src')
    str_16613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 24), 'str', 'multiarray')
    str_16614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 38), 'str', 'number.h')
    # Processing the call keyword arguments (line 739)
    kwargs_16615 = {}
    # Getting the type of 'join' (line 739)
    join_16611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 12), 'join', False)
    # Calling join(args, kwargs) (line 739)
    join_call_result_16616 = invoke(stypy.reporting.localization.Localization(__file__, 739, 12), join_16611, *[str_16612, str_16613, str_16614], **kwargs_16615)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16616)
    # Adding element type (line 718)
    
    # Call to join(...): (line 740)
    # Processing the call arguments (line 740)
    str_16618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 17), 'str', 'src')
    str_16619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 24), 'str', 'multiarray')
    str_16620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 38), 'str', 'numpyos.h')
    # Processing the call keyword arguments (line 740)
    kwargs_16621 = {}
    # Getting the type of 'join' (line 740)
    join_16617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 12), 'join', False)
    # Calling join(args, kwargs) (line 740)
    join_call_result_16622 = invoke(stypy.reporting.localization.Localization(__file__, 740, 12), join_16617, *[str_16618, str_16619, str_16620], **kwargs_16621)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16622)
    # Adding element type (line 718)
    
    # Call to join(...): (line 741)
    # Processing the call arguments (line 741)
    str_16624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 17), 'str', 'src')
    str_16625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 24), 'str', 'multiarray')
    str_16626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 38), 'str', 'refcount.h')
    # Processing the call keyword arguments (line 741)
    kwargs_16627 = {}
    # Getting the type of 'join' (line 741)
    join_16623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 12), 'join', False)
    # Calling join(args, kwargs) (line 741)
    join_call_result_16628 = invoke(stypy.reporting.localization.Localization(__file__, 741, 12), join_16623, *[str_16624, str_16625, str_16626], **kwargs_16627)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16628)
    # Adding element type (line 718)
    
    # Call to join(...): (line 742)
    # Processing the call arguments (line 742)
    str_16630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 17), 'str', 'src')
    str_16631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 24), 'str', 'multiarray')
    str_16632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 38), 'str', 'scalartypes.h')
    # Processing the call keyword arguments (line 742)
    kwargs_16633 = {}
    # Getting the type of 'join' (line 742)
    join_16629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 12), 'join', False)
    # Calling join(args, kwargs) (line 742)
    join_call_result_16634 = invoke(stypy.reporting.localization.Localization(__file__, 742, 12), join_16629, *[str_16630, str_16631, str_16632], **kwargs_16633)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16634)
    # Adding element type (line 718)
    
    # Call to join(...): (line 743)
    # Processing the call arguments (line 743)
    str_16636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 17), 'str', 'src')
    str_16637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 24), 'str', 'multiarray')
    str_16638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 38), 'str', 'sequence.h')
    # Processing the call keyword arguments (line 743)
    kwargs_16639 = {}
    # Getting the type of 'join' (line 743)
    join_16635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 12), 'join', False)
    # Calling join(args, kwargs) (line 743)
    join_call_result_16640 = invoke(stypy.reporting.localization.Localization(__file__, 743, 12), join_16635, *[str_16636, str_16637, str_16638], **kwargs_16639)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16640)
    # Adding element type (line 718)
    
    # Call to join(...): (line 744)
    # Processing the call arguments (line 744)
    str_16642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 17), 'str', 'src')
    str_16643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 24), 'str', 'multiarray')
    str_16644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 38), 'str', 'shape.h')
    # Processing the call keyword arguments (line 744)
    kwargs_16645 = {}
    # Getting the type of 'join' (line 744)
    join_16641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 12), 'join', False)
    # Calling join(args, kwargs) (line 744)
    join_call_result_16646 = invoke(stypy.reporting.localization.Localization(__file__, 744, 12), join_16641, *[str_16642, str_16643, str_16644], **kwargs_16645)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16646)
    # Adding element type (line 718)
    
    # Call to join(...): (line 745)
    # Processing the call arguments (line 745)
    str_16648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 17), 'str', 'src')
    str_16649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 24), 'str', 'multiarray')
    str_16650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 38), 'str', 'ucsnarrow.h')
    # Processing the call keyword arguments (line 745)
    kwargs_16651 = {}
    # Getting the type of 'join' (line 745)
    join_16647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 12), 'join', False)
    # Calling join(args, kwargs) (line 745)
    join_call_result_16652 = invoke(stypy.reporting.localization.Localization(__file__, 745, 12), join_16647, *[str_16648, str_16649, str_16650], **kwargs_16651)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16652)
    # Adding element type (line 718)
    
    # Call to join(...): (line 746)
    # Processing the call arguments (line 746)
    str_16654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 17), 'str', 'src')
    str_16655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 24), 'str', 'multiarray')
    str_16656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 38), 'str', 'usertypes.h')
    # Processing the call keyword arguments (line 746)
    kwargs_16657 = {}
    # Getting the type of 'join' (line 746)
    join_16653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 12), 'join', False)
    # Calling join(args, kwargs) (line 746)
    join_call_result_16658 = invoke(stypy.reporting.localization.Localization(__file__, 746, 12), join_16653, *[str_16654, str_16655, str_16656], **kwargs_16657)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16658)
    # Adding element type (line 718)
    
    # Call to join(...): (line 747)
    # Processing the call arguments (line 747)
    str_16660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 17), 'str', 'src')
    str_16661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 24), 'str', 'multiarray')
    str_16662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 38), 'str', 'vdot.h')
    # Processing the call keyword arguments (line 747)
    kwargs_16663 = {}
    # Getting the type of 'join' (line 747)
    join_16659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 12), 'join', False)
    # Calling join(args, kwargs) (line 747)
    join_call_result_16664 = invoke(stypy.reporting.localization.Localization(__file__, 747, 12), join_16659, *[str_16660, str_16661, str_16662], **kwargs_16663)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16664)
    # Adding element type (line 718)
    
    # Call to join(...): (line 748)
    # Processing the call arguments (line 748)
    str_16666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 17), 'str', 'src')
    str_16667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 24), 'str', 'private')
    str_16668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 35), 'str', 'npy_config.h')
    # Processing the call keyword arguments (line 748)
    kwargs_16669 = {}
    # Getting the type of 'join' (line 748)
    join_16665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 12), 'join', False)
    # Calling join(args, kwargs) (line 748)
    join_call_result_16670 = invoke(stypy.reporting.localization.Localization(__file__, 748, 12), join_16665, *[str_16666, str_16667, str_16668], **kwargs_16669)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16670)
    # Adding element type (line 718)
    
    # Call to join(...): (line 749)
    # Processing the call arguments (line 749)
    str_16672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 17), 'str', 'src')
    str_16673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 24), 'str', 'private')
    str_16674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 35), 'str', 'templ_common.h.src')
    # Processing the call keyword arguments (line 749)
    kwargs_16675 = {}
    # Getting the type of 'join' (line 749)
    join_16671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 12), 'join', False)
    # Calling join(args, kwargs) (line 749)
    join_call_result_16676 = invoke(stypy.reporting.localization.Localization(__file__, 749, 12), join_16671, *[str_16672, str_16673, str_16674], **kwargs_16675)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16676)
    # Adding element type (line 718)
    
    # Call to join(...): (line 750)
    # Processing the call arguments (line 750)
    str_16678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 17), 'str', 'src')
    str_16679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 24), 'str', 'private')
    str_16680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 35), 'str', 'lowlevel_strided_loops.h')
    # Processing the call keyword arguments (line 750)
    kwargs_16681 = {}
    # Getting the type of 'join' (line 750)
    join_16677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 12), 'join', False)
    # Calling join(args, kwargs) (line 750)
    join_call_result_16682 = invoke(stypy.reporting.localization.Localization(__file__, 750, 12), join_16677, *[str_16678, str_16679, str_16680], **kwargs_16681)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16682)
    # Adding element type (line 718)
    
    # Call to join(...): (line 751)
    # Processing the call arguments (line 751)
    str_16684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 17), 'str', 'src')
    str_16685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 24), 'str', 'private')
    str_16686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 35), 'str', 'mem_overlap.h')
    # Processing the call keyword arguments (line 751)
    kwargs_16687 = {}
    # Getting the type of 'join' (line 751)
    join_16683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 12), 'join', False)
    # Calling join(args, kwargs) (line 751)
    join_call_result_16688 = invoke(stypy.reporting.localization.Localization(__file__, 751, 12), join_16683, *[str_16684, str_16685, str_16686], **kwargs_16687)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16688)
    # Adding element type (line 718)
    
    # Call to join(...): (line 752)
    # Processing the call arguments (line 752)
    str_16690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 17), 'str', 'src')
    str_16691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 24), 'str', 'private')
    str_16692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 35), 'str', 'npy_extint128.h')
    # Processing the call keyword arguments (line 752)
    kwargs_16693 = {}
    # Getting the type of 'join' (line 752)
    join_16689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 12), 'join', False)
    # Calling join(args, kwargs) (line 752)
    join_call_result_16694 = invoke(stypy.reporting.localization.Localization(__file__, 752, 12), join_16689, *[str_16690, str_16691, str_16692], **kwargs_16693)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16694)
    # Adding element type (line 718)
    
    # Call to join(...): (line 753)
    # Processing the call arguments (line 753)
    str_16696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 17), 'str', 'include')
    str_16697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 28), 'str', 'numpy')
    str_16698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 37), 'str', 'arrayobject.h')
    # Processing the call keyword arguments (line 753)
    kwargs_16699 = {}
    # Getting the type of 'join' (line 753)
    join_16695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 12), 'join', False)
    # Calling join(args, kwargs) (line 753)
    join_call_result_16700 = invoke(stypy.reporting.localization.Localization(__file__, 753, 12), join_16695, *[str_16696, str_16697, str_16698], **kwargs_16699)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16700)
    # Adding element type (line 718)
    
    # Call to join(...): (line 754)
    # Processing the call arguments (line 754)
    str_16702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 17), 'str', 'include')
    str_16703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 28), 'str', 'numpy')
    str_16704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 37), 'str', '_neighborhood_iterator_imp.h')
    # Processing the call keyword arguments (line 754)
    kwargs_16705 = {}
    # Getting the type of 'join' (line 754)
    join_16701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 12), 'join', False)
    # Calling join(args, kwargs) (line 754)
    join_call_result_16706 = invoke(stypy.reporting.localization.Localization(__file__, 754, 12), join_16701, *[str_16702, str_16703, str_16704], **kwargs_16705)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16706)
    # Adding element type (line 718)
    
    # Call to join(...): (line 755)
    # Processing the call arguments (line 755)
    str_16708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 17), 'str', 'include')
    str_16709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 28), 'str', 'numpy')
    str_16710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 37), 'str', 'npy_endian.h')
    # Processing the call keyword arguments (line 755)
    kwargs_16711 = {}
    # Getting the type of 'join' (line 755)
    join_16707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 12), 'join', False)
    # Calling join(args, kwargs) (line 755)
    join_call_result_16712 = invoke(stypy.reporting.localization.Localization(__file__, 755, 12), join_16707, *[str_16708, str_16709, str_16710], **kwargs_16711)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16712)
    # Adding element type (line 718)
    
    # Call to join(...): (line 756)
    # Processing the call arguments (line 756)
    str_16714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 17), 'str', 'include')
    str_16715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 28), 'str', 'numpy')
    str_16716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 37), 'str', 'arrayscalars.h')
    # Processing the call keyword arguments (line 756)
    kwargs_16717 = {}
    # Getting the type of 'join' (line 756)
    join_16713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 12), 'join', False)
    # Calling join(args, kwargs) (line 756)
    join_call_result_16718 = invoke(stypy.reporting.localization.Localization(__file__, 756, 12), join_16713, *[str_16714, str_16715, str_16716], **kwargs_16717)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16718)
    # Adding element type (line 718)
    
    # Call to join(...): (line 757)
    # Processing the call arguments (line 757)
    str_16720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 17), 'str', 'include')
    str_16721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 28), 'str', 'numpy')
    str_16722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 37), 'str', 'noprefix.h')
    # Processing the call keyword arguments (line 757)
    kwargs_16723 = {}
    # Getting the type of 'join' (line 757)
    join_16719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 12), 'join', False)
    # Calling join(args, kwargs) (line 757)
    join_call_result_16724 = invoke(stypy.reporting.localization.Localization(__file__, 757, 12), join_16719, *[str_16720, str_16721, str_16722], **kwargs_16723)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16724)
    # Adding element type (line 718)
    
    # Call to join(...): (line 758)
    # Processing the call arguments (line 758)
    str_16726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 17), 'str', 'include')
    str_16727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 28), 'str', 'numpy')
    str_16728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 37), 'str', 'npy_interrupt.h')
    # Processing the call keyword arguments (line 758)
    kwargs_16729 = {}
    # Getting the type of 'join' (line 758)
    join_16725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 12), 'join', False)
    # Calling join(args, kwargs) (line 758)
    join_call_result_16730 = invoke(stypy.reporting.localization.Localization(__file__, 758, 12), join_16725, *[str_16726, str_16727, str_16728], **kwargs_16729)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16730)
    # Adding element type (line 718)
    
    # Call to join(...): (line 759)
    # Processing the call arguments (line 759)
    str_16732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 17), 'str', 'include')
    str_16733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 28), 'str', 'numpy')
    str_16734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 37), 'str', 'npy_3kcompat.h')
    # Processing the call keyword arguments (line 759)
    kwargs_16735 = {}
    # Getting the type of 'join' (line 759)
    join_16731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 12), 'join', False)
    # Calling join(args, kwargs) (line 759)
    join_call_result_16736 = invoke(stypy.reporting.localization.Localization(__file__, 759, 12), join_16731, *[str_16732, str_16733, str_16734], **kwargs_16735)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16736)
    # Adding element type (line 718)
    
    # Call to join(...): (line 760)
    # Processing the call arguments (line 760)
    str_16738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 17), 'str', 'include')
    str_16739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 28), 'str', 'numpy')
    str_16740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 37), 'str', 'npy_math.h')
    # Processing the call keyword arguments (line 760)
    kwargs_16741 = {}
    # Getting the type of 'join' (line 760)
    join_16737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 12), 'join', False)
    # Calling join(args, kwargs) (line 760)
    join_call_result_16742 = invoke(stypy.reporting.localization.Localization(__file__, 760, 12), join_16737, *[str_16738, str_16739, str_16740], **kwargs_16741)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16742)
    # Adding element type (line 718)
    
    # Call to join(...): (line 761)
    # Processing the call arguments (line 761)
    str_16744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 17), 'str', 'include')
    str_16745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 28), 'str', 'numpy')
    str_16746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 37), 'str', 'halffloat.h')
    # Processing the call keyword arguments (line 761)
    kwargs_16747 = {}
    # Getting the type of 'join' (line 761)
    join_16743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'join', False)
    # Calling join(args, kwargs) (line 761)
    join_call_result_16748 = invoke(stypy.reporting.localization.Localization(__file__, 761, 12), join_16743, *[str_16744, str_16745, str_16746], **kwargs_16747)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16748)
    # Adding element type (line 718)
    
    # Call to join(...): (line 762)
    # Processing the call arguments (line 762)
    str_16750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 17), 'str', 'include')
    str_16751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 28), 'str', 'numpy')
    str_16752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 37), 'str', 'npy_common.h')
    # Processing the call keyword arguments (line 762)
    kwargs_16753 = {}
    # Getting the type of 'join' (line 762)
    join_16749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 12), 'join', False)
    # Calling join(args, kwargs) (line 762)
    join_call_result_16754 = invoke(stypy.reporting.localization.Localization(__file__, 762, 12), join_16749, *[str_16750, str_16751, str_16752], **kwargs_16753)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16754)
    # Adding element type (line 718)
    
    # Call to join(...): (line 763)
    # Processing the call arguments (line 763)
    str_16756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 17), 'str', 'include')
    str_16757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 28), 'str', 'numpy')
    str_16758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 37), 'str', 'npy_os.h')
    # Processing the call keyword arguments (line 763)
    kwargs_16759 = {}
    # Getting the type of 'join' (line 763)
    join_16755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 12), 'join', False)
    # Calling join(args, kwargs) (line 763)
    join_call_result_16760 = invoke(stypy.reporting.localization.Localization(__file__, 763, 12), join_16755, *[str_16756, str_16757, str_16758], **kwargs_16759)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16760)
    # Adding element type (line 718)
    
    # Call to join(...): (line 764)
    # Processing the call arguments (line 764)
    str_16762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 17), 'str', 'include')
    str_16763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 28), 'str', 'numpy')
    str_16764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 37), 'str', 'utils.h')
    # Processing the call keyword arguments (line 764)
    kwargs_16765 = {}
    # Getting the type of 'join' (line 764)
    join_16761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'join', False)
    # Calling join(args, kwargs) (line 764)
    join_call_result_16766 = invoke(stypy.reporting.localization.Localization(__file__, 764, 12), join_16761, *[str_16762, str_16763, str_16764], **kwargs_16765)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16766)
    # Adding element type (line 718)
    
    # Call to join(...): (line 765)
    # Processing the call arguments (line 765)
    str_16768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 17), 'str', 'include')
    str_16769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 28), 'str', 'numpy')
    str_16770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 37), 'str', 'ndarrayobject.h')
    # Processing the call keyword arguments (line 765)
    kwargs_16771 = {}
    # Getting the type of 'join' (line 765)
    join_16767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 12), 'join', False)
    # Calling join(args, kwargs) (line 765)
    join_call_result_16772 = invoke(stypy.reporting.localization.Localization(__file__, 765, 12), join_16767, *[str_16768, str_16769, str_16770], **kwargs_16771)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16772)
    # Adding element type (line 718)
    
    # Call to join(...): (line 766)
    # Processing the call arguments (line 766)
    str_16774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 17), 'str', 'include')
    str_16775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 28), 'str', 'numpy')
    str_16776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 37), 'str', 'npy_cpu.h')
    # Processing the call keyword arguments (line 766)
    kwargs_16777 = {}
    # Getting the type of 'join' (line 766)
    join_16773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 12), 'join', False)
    # Calling join(args, kwargs) (line 766)
    join_call_result_16778 = invoke(stypy.reporting.localization.Localization(__file__, 766, 12), join_16773, *[str_16774, str_16775, str_16776], **kwargs_16777)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16778)
    # Adding element type (line 718)
    
    # Call to join(...): (line 767)
    # Processing the call arguments (line 767)
    str_16780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 17), 'str', 'include')
    str_16781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 28), 'str', 'numpy')
    str_16782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 37), 'str', 'numpyconfig.h')
    # Processing the call keyword arguments (line 767)
    kwargs_16783 = {}
    # Getting the type of 'join' (line 767)
    join_16779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 12), 'join', False)
    # Calling join(args, kwargs) (line 767)
    join_call_result_16784 = invoke(stypy.reporting.localization.Localization(__file__, 767, 12), join_16779, *[str_16780, str_16781, str_16782], **kwargs_16783)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16784)
    # Adding element type (line 718)
    
    # Call to join(...): (line 768)
    # Processing the call arguments (line 768)
    str_16786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 17), 'str', 'include')
    str_16787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 28), 'str', 'numpy')
    str_16788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 37), 'str', 'ndarraytypes.h')
    # Processing the call keyword arguments (line 768)
    kwargs_16789 = {}
    # Getting the type of 'join' (line 768)
    join_16785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 12), 'join', False)
    # Calling join(args, kwargs) (line 768)
    join_call_result_16790 = invoke(stypy.reporting.localization.Localization(__file__, 768, 12), join_16785, *[str_16786, str_16787, str_16788], **kwargs_16789)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16790)
    # Adding element type (line 718)
    
    # Call to join(...): (line 769)
    # Processing the call arguments (line 769)
    str_16792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 17), 'str', 'include')
    str_16793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 28), 'str', 'numpy')
    str_16794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 37), 'str', 'npy_1_7_deprecated_api.h')
    # Processing the call keyword arguments (line 769)
    kwargs_16795 = {}
    # Getting the type of 'join' (line 769)
    join_16791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 12), 'join', False)
    # Calling join(args, kwargs) (line 769)
    join_call_result_16796 = invoke(stypy.reporting.localization.Localization(__file__, 769, 12), join_16791, *[str_16792, str_16793, str_16794], **kwargs_16795)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16796)
    # Adding element type (line 718)
    
    # Call to join(...): (line 770)
    # Processing the call arguments (line 770)
    str_16798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 17), 'str', 'include')
    str_16799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 28), 'str', 'numpy')
    str_16800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 37), 'str', '_numpyconfig.h.in')
    # Processing the call keyword arguments (line 770)
    kwargs_16801 = {}
    # Getting the type of 'join' (line 770)
    join_16797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 12), 'join', False)
    # Calling join(args, kwargs) (line 770)
    join_call_result_16802 = invoke(stypy.reporting.localization.Localization(__file__, 770, 12), join_16797, *[str_16798, str_16799, str_16800], **kwargs_16801)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 22), list_16490, join_call_result_16802)
    
    # Getting the type of 'npysort_sources' (line 773)
    npysort_sources_16803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 16), 'npysort_sources')
    # Applying the binary operator '+' (line 718)
    result_add_16804 = python_operator(stypy.reporting.localization.Localization(__file__, 718, 22), '+', list_16490, npysort_sources_16803)
    
    # Getting the type of 'npymath_sources' (line 773)
    npymath_sources_16805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 34), 'npymath_sources')
    # Applying the binary operator '+' (line 773)
    result_add_16806 = python_operator(stypy.reporting.localization.Localization(__file__, 773, 32), '+', result_add_16804, npymath_sources_16805)
    
    # Assigning a type to the variable 'multiarray_deps' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 4), 'multiarray_deps', result_add_16806)
    
    # Assigning a List to a Name (line 775):
    
    # Assigning a List to a Name (line 775):
    
    # Obtaining an instance of the builtin type 'list' (line 775)
    list_16807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 775)
    # Adding element type (line 775)
    
    # Call to join(...): (line 776)
    # Processing the call arguments (line 776)
    str_16809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 17), 'str', 'src')
    str_16810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 24), 'str', 'multiarray')
    str_16811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 38), 'str', 'alloc.c')
    # Processing the call keyword arguments (line 776)
    kwargs_16812 = {}
    # Getting the type of 'join' (line 776)
    join_16808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 12), 'join', False)
    # Calling join(args, kwargs) (line 776)
    join_call_result_16813 = invoke(stypy.reporting.localization.Localization(__file__, 776, 12), join_16808, *[str_16809, str_16810, str_16811], **kwargs_16812)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16813)
    # Adding element type (line 775)
    
    # Call to join(...): (line 777)
    # Processing the call arguments (line 777)
    str_16815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 17), 'str', 'src')
    str_16816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 24), 'str', 'multiarray')
    str_16817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 38), 'str', 'arrayobject.c')
    # Processing the call keyword arguments (line 777)
    kwargs_16818 = {}
    # Getting the type of 'join' (line 777)
    join_16814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 12), 'join', False)
    # Calling join(args, kwargs) (line 777)
    join_call_result_16819 = invoke(stypy.reporting.localization.Localization(__file__, 777, 12), join_16814, *[str_16815, str_16816, str_16817], **kwargs_16818)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16819)
    # Adding element type (line 775)
    
    # Call to join(...): (line 778)
    # Processing the call arguments (line 778)
    str_16821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 17), 'str', 'src')
    str_16822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 24), 'str', 'multiarray')
    str_16823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 38), 'str', 'arraytypes.c.src')
    # Processing the call keyword arguments (line 778)
    kwargs_16824 = {}
    # Getting the type of 'join' (line 778)
    join_16820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 12), 'join', False)
    # Calling join(args, kwargs) (line 778)
    join_call_result_16825 = invoke(stypy.reporting.localization.Localization(__file__, 778, 12), join_16820, *[str_16821, str_16822, str_16823], **kwargs_16824)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16825)
    # Adding element type (line 775)
    
    # Call to join(...): (line 779)
    # Processing the call arguments (line 779)
    str_16827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 17), 'str', 'src')
    str_16828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 24), 'str', 'multiarray')
    str_16829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 38), 'str', 'array_assign.c')
    # Processing the call keyword arguments (line 779)
    kwargs_16830 = {}
    # Getting the type of 'join' (line 779)
    join_16826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 12), 'join', False)
    # Calling join(args, kwargs) (line 779)
    join_call_result_16831 = invoke(stypy.reporting.localization.Localization(__file__, 779, 12), join_16826, *[str_16827, str_16828, str_16829], **kwargs_16830)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16831)
    # Adding element type (line 775)
    
    # Call to join(...): (line 780)
    # Processing the call arguments (line 780)
    str_16833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 17), 'str', 'src')
    str_16834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 24), 'str', 'multiarray')
    str_16835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 38), 'str', 'array_assign_scalar.c')
    # Processing the call keyword arguments (line 780)
    kwargs_16836 = {}
    # Getting the type of 'join' (line 780)
    join_16832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 12), 'join', False)
    # Calling join(args, kwargs) (line 780)
    join_call_result_16837 = invoke(stypy.reporting.localization.Localization(__file__, 780, 12), join_16832, *[str_16833, str_16834, str_16835], **kwargs_16836)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16837)
    # Adding element type (line 775)
    
    # Call to join(...): (line 781)
    # Processing the call arguments (line 781)
    str_16839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 17), 'str', 'src')
    str_16840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 24), 'str', 'multiarray')
    str_16841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 38), 'str', 'array_assign_array.c')
    # Processing the call keyword arguments (line 781)
    kwargs_16842 = {}
    # Getting the type of 'join' (line 781)
    join_16838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 12), 'join', False)
    # Calling join(args, kwargs) (line 781)
    join_call_result_16843 = invoke(stypy.reporting.localization.Localization(__file__, 781, 12), join_16838, *[str_16839, str_16840, str_16841], **kwargs_16842)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16843)
    # Adding element type (line 775)
    
    # Call to join(...): (line 782)
    # Processing the call arguments (line 782)
    str_16845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 17), 'str', 'src')
    str_16846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 24), 'str', 'multiarray')
    str_16847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 38), 'str', 'buffer.c')
    # Processing the call keyword arguments (line 782)
    kwargs_16848 = {}
    # Getting the type of 'join' (line 782)
    join_16844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 12), 'join', False)
    # Calling join(args, kwargs) (line 782)
    join_call_result_16849 = invoke(stypy.reporting.localization.Localization(__file__, 782, 12), join_16844, *[str_16845, str_16846, str_16847], **kwargs_16848)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16849)
    # Adding element type (line 775)
    
    # Call to join(...): (line 783)
    # Processing the call arguments (line 783)
    str_16851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 17), 'str', 'src')
    str_16852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 24), 'str', 'multiarray')
    str_16853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 38), 'str', 'calculation.c')
    # Processing the call keyword arguments (line 783)
    kwargs_16854 = {}
    # Getting the type of 'join' (line 783)
    join_16850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 12), 'join', False)
    # Calling join(args, kwargs) (line 783)
    join_call_result_16855 = invoke(stypy.reporting.localization.Localization(__file__, 783, 12), join_16850, *[str_16851, str_16852, str_16853], **kwargs_16854)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16855)
    # Adding element type (line 775)
    
    # Call to join(...): (line 784)
    # Processing the call arguments (line 784)
    str_16857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 17), 'str', 'src')
    str_16858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 24), 'str', 'multiarray')
    str_16859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 38), 'str', 'compiled_base.c')
    # Processing the call keyword arguments (line 784)
    kwargs_16860 = {}
    # Getting the type of 'join' (line 784)
    join_16856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 12), 'join', False)
    # Calling join(args, kwargs) (line 784)
    join_call_result_16861 = invoke(stypy.reporting.localization.Localization(__file__, 784, 12), join_16856, *[str_16857, str_16858, str_16859], **kwargs_16860)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16861)
    # Adding element type (line 775)
    
    # Call to join(...): (line 785)
    # Processing the call arguments (line 785)
    str_16863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 17), 'str', 'src')
    str_16864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 24), 'str', 'multiarray')
    str_16865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 38), 'str', 'common.c')
    # Processing the call keyword arguments (line 785)
    kwargs_16866 = {}
    # Getting the type of 'join' (line 785)
    join_16862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 12), 'join', False)
    # Calling join(args, kwargs) (line 785)
    join_call_result_16867 = invoke(stypy.reporting.localization.Localization(__file__, 785, 12), join_16862, *[str_16863, str_16864, str_16865], **kwargs_16866)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16867)
    # Adding element type (line 775)
    
    # Call to join(...): (line 786)
    # Processing the call arguments (line 786)
    str_16869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 17), 'str', 'src')
    str_16870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 24), 'str', 'multiarray')
    str_16871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 38), 'str', 'convert.c')
    # Processing the call keyword arguments (line 786)
    kwargs_16872 = {}
    # Getting the type of 'join' (line 786)
    join_16868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 12), 'join', False)
    # Calling join(args, kwargs) (line 786)
    join_call_result_16873 = invoke(stypy.reporting.localization.Localization(__file__, 786, 12), join_16868, *[str_16869, str_16870, str_16871], **kwargs_16872)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16873)
    # Adding element type (line 775)
    
    # Call to join(...): (line 787)
    # Processing the call arguments (line 787)
    str_16875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 17), 'str', 'src')
    str_16876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 24), 'str', 'multiarray')
    str_16877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 38), 'str', 'convert_datatype.c')
    # Processing the call keyword arguments (line 787)
    kwargs_16878 = {}
    # Getting the type of 'join' (line 787)
    join_16874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 12), 'join', False)
    # Calling join(args, kwargs) (line 787)
    join_call_result_16879 = invoke(stypy.reporting.localization.Localization(__file__, 787, 12), join_16874, *[str_16875, str_16876, str_16877], **kwargs_16878)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16879)
    # Adding element type (line 775)
    
    # Call to join(...): (line 788)
    # Processing the call arguments (line 788)
    str_16881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 17), 'str', 'src')
    str_16882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 24), 'str', 'multiarray')
    str_16883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 38), 'str', 'conversion_utils.c')
    # Processing the call keyword arguments (line 788)
    kwargs_16884 = {}
    # Getting the type of 'join' (line 788)
    join_16880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 12), 'join', False)
    # Calling join(args, kwargs) (line 788)
    join_call_result_16885 = invoke(stypy.reporting.localization.Localization(__file__, 788, 12), join_16880, *[str_16881, str_16882, str_16883], **kwargs_16884)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16885)
    # Adding element type (line 775)
    
    # Call to join(...): (line 789)
    # Processing the call arguments (line 789)
    str_16887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 789, 17), 'str', 'src')
    str_16888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 789, 24), 'str', 'multiarray')
    str_16889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 789, 38), 'str', 'ctors.c')
    # Processing the call keyword arguments (line 789)
    kwargs_16890 = {}
    # Getting the type of 'join' (line 789)
    join_16886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 12), 'join', False)
    # Calling join(args, kwargs) (line 789)
    join_call_result_16891 = invoke(stypy.reporting.localization.Localization(__file__, 789, 12), join_16886, *[str_16887, str_16888, str_16889], **kwargs_16890)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16891)
    # Adding element type (line 775)
    
    # Call to join(...): (line 790)
    # Processing the call arguments (line 790)
    str_16893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 790, 17), 'str', 'src')
    str_16894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 790, 24), 'str', 'multiarray')
    str_16895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 790, 38), 'str', 'datetime.c')
    # Processing the call keyword arguments (line 790)
    kwargs_16896 = {}
    # Getting the type of 'join' (line 790)
    join_16892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 12), 'join', False)
    # Calling join(args, kwargs) (line 790)
    join_call_result_16897 = invoke(stypy.reporting.localization.Localization(__file__, 790, 12), join_16892, *[str_16893, str_16894, str_16895], **kwargs_16896)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16897)
    # Adding element type (line 775)
    
    # Call to join(...): (line 791)
    # Processing the call arguments (line 791)
    str_16899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 17), 'str', 'src')
    str_16900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 24), 'str', 'multiarray')
    str_16901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 38), 'str', 'datetime_strings.c')
    # Processing the call keyword arguments (line 791)
    kwargs_16902 = {}
    # Getting the type of 'join' (line 791)
    join_16898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 12), 'join', False)
    # Calling join(args, kwargs) (line 791)
    join_call_result_16903 = invoke(stypy.reporting.localization.Localization(__file__, 791, 12), join_16898, *[str_16899, str_16900, str_16901], **kwargs_16902)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16903)
    # Adding element type (line 775)
    
    # Call to join(...): (line 792)
    # Processing the call arguments (line 792)
    str_16905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 17), 'str', 'src')
    str_16906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 24), 'str', 'multiarray')
    str_16907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 38), 'str', 'datetime_busday.c')
    # Processing the call keyword arguments (line 792)
    kwargs_16908 = {}
    # Getting the type of 'join' (line 792)
    join_16904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 12), 'join', False)
    # Calling join(args, kwargs) (line 792)
    join_call_result_16909 = invoke(stypy.reporting.localization.Localization(__file__, 792, 12), join_16904, *[str_16905, str_16906, str_16907], **kwargs_16908)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16909)
    # Adding element type (line 775)
    
    # Call to join(...): (line 793)
    # Processing the call arguments (line 793)
    str_16911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 793, 17), 'str', 'src')
    str_16912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 793, 24), 'str', 'multiarray')
    str_16913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 793, 38), 'str', 'datetime_busdaycal.c')
    # Processing the call keyword arguments (line 793)
    kwargs_16914 = {}
    # Getting the type of 'join' (line 793)
    join_16910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 12), 'join', False)
    # Calling join(args, kwargs) (line 793)
    join_call_result_16915 = invoke(stypy.reporting.localization.Localization(__file__, 793, 12), join_16910, *[str_16911, str_16912, str_16913], **kwargs_16914)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16915)
    # Adding element type (line 775)
    
    # Call to join(...): (line 794)
    # Processing the call arguments (line 794)
    str_16917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 17), 'str', 'src')
    str_16918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 24), 'str', 'multiarray')
    str_16919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 38), 'str', 'descriptor.c')
    # Processing the call keyword arguments (line 794)
    kwargs_16920 = {}
    # Getting the type of 'join' (line 794)
    join_16916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 12), 'join', False)
    # Calling join(args, kwargs) (line 794)
    join_call_result_16921 = invoke(stypy.reporting.localization.Localization(__file__, 794, 12), join_16916, *[str_16917, str_16918, str_16919], **kwargs_16920)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16921)
    # Adding element type (line 775)
    
    # Call to join(...): (line 795)
    # Processing the call arguments (line 795)
    str_16923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 17), 'str', 'src')
    str_16924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 24), 'str', 'multiarray')
    str_16925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 38), 'str', 'dtype_transfer.c')
    # Processing the call keyword arguments (line 795)
    kwargs_16926 = {}
    # Getting the type of 'join' (line 795)
    join_16922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 12), 'join', False)
    # Calling join(args, kwargs) (line 795)
    join_call_result_16927 = invoke(stypy.reporting.localization.Localization(__file__, 795, 12), join_16922, *[str_16923, str_16924, str_16925], **kwargs_16926)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16927)
    # Adding element type (line 775)
    
    # Call to join(...): (line 796)
    # Processing the call arguments (line 796)
    str_16929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 17), 'str', 'src')
    str_16930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 24), 'str', 'multiarray')
    str_16931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 38), 'str', 'einsum.c.src')
    # Processing the call keyword arguments (line 796)
    kwargs_16932 = {}
    # Getting the type of 'join' (line 796)
    join_16928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 12), 'join', False)
    # Calling join(args, kwargs) (line 796)
    join_call_result_16933 = invoke(stypy.reporting.localization.Localization(__file__, 796, 12), join_16928, *[str_16929, str_16930, str_16931], **kwargs_16932)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16933)
    # Adding element type (line 775)
    
    # Call to join(...): (line 797)
    # Processing the call arguments (line 797)
    str_16935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 17), 'str', 'src')
    str_16936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 24), 'str', 'multiarray')
    str_16937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 38), 'str', 'flagsobject.c')
    # Processing the call keyword arguments (line 797)
    kwargs_16938 = {}
    # Getting the type of 'join' (line 797)
    join_16934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 12), 'join', False)
    # Calling join(args, kwargs) (line 797)
    join_call_result_16939 = invoke(stypy.reporting.localization.Localization(__file__, 797, 12), join_16934, *[str_16935, str_16936, str_16937], **kwargs_16938)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16939)
    # Adding element type (line 775)
    
    # Call to join(...): (line 798)
    # Processing the call arguments (line 798)
    str_16941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 798, 17), 'str', 'src')
    str_16942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 798, 24), 'str', 'multiarray')
    str_16943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 798, 38), 'str', 'getset.c')
    # Processing the call keyword arguments (line 798)
    kwargs_16944 = {}
    # Getting the type of 'join' (line 798)
    join_16940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 12), 'join', False)
    # Calling join(args, kwargs) (line 798)
    join_call_result_16945 = invoke(stypy.reporting.localization.Localization(__file__, 798, 12), join_16940, *[str_16941, str_16942, str_16943], **kwargs_16944)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16945)
    # Adding element type (line 775)
    
    # Call to join(...): (line 799)
    # Processing the call arguments (line 799)
    str_16947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 799, 17), 'str', 'src')
    str_16948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 799, 24), 'str', 'multiarray')
    str_16949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 799, 38), 'str', 'hashdescr.c')
    # Processing the call keyword arguments (line 799)
    kwargs_16950 = {}
    # Getting the type of 'join' (line 799)
    join_16946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 12), 'join', False)
    # Calling join(args, kwargs) (line 799)
    join_call_result_16951 = invoke(stypy.reporting.localization.Localization(__file__, 799, 12), join_16946, *[str_16947, str_16948, str_16949], **kwargs_16950)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16951)
    # Adding element type (line 775)
    
    # Call to join(...): (line 800)
    # Processing the call arguments (line 800)
    str_16953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 17), 'str', 'src')
    str_16954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 24), 'str', 'multiarray')
    str_16955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 38), 'str', 'item_selection.c')
    # Processing the call keyword arguments (line 800)
    kwargs_16956 = {}
    # Getting the type of 'join' (line 800)
    join_16952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 12), 'join', False)
    # Calling join(args, kwargs) (line 800)
    join_call_result_16957 = invoke(stypy.reporting.localization.Localization(__file__, 800, 12), join_16952, *[str_16953, str_16954, str_16955], **kwargs_16956)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16957)
    # Adding element type (line 775)
    
    # Call to join(...): (line 801)
    # Processing the call arguments (line 801)
    str_16959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 17), 'str', 'src')
    str_16960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 24), 'str', 'multiarray')
    str_16961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 38), 'str', 'iterators.c')
    # Processing the call keyword arguments (line 801)
    kwargs_16962 = {}
    # Getting the type of 'join' (line 801)
    join_16958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 12), 'join', False)
    # Calling join(args, kwargs) (line 801)
    join_call_result_16963 = invoke(stypy.reporting.localization.Localization(__file__, 801, 12), join_16958, *[str_16959, str_16960, str_16961], **kwargs_16962)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16963)
    # Adding element type (line 775)
    
    # Call to join(...): (line 802)
    # Processing the call arguments (line 802)
    str_16965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 802, 17), 'str', 'src')
    str_16966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 802, 24), 'str', 'multiarray')
    str_16967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 802, 38), 'str', 'lowlevel_strided_loops.c.src')
    # Processing the call keyword arguments (line 802)
    kwargs_16968 = {}
    # Getting the type of 'join' (line 802)
    join_16964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 12), 'join', False)
    # Calling join(args, kwargs) (line 802)
    join_call_result_16969 = invoke(stypy.reporting.localization.Localization(__file__, 802, 12), join_16964, *[str_16965, str_16966, str_16967], **kwargs_16968)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16969)
    # Adding element type (line 775)
    
    # Call to join(...): (line 803)
    # Processing the call arguments (line 803)
    str_16971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 803, 17), 'str', 'src')
    str_16972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 803, 24), 'str', 'multiarray')
    str_16973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 803, 38), 'str', 'mapping.c')
    # Processing the call keyword arguments (line 803)
    kwargs_16974 = {}
    # Getting the type of 'join' (line 803)
    join_16970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 12), 'join', False)
    # Calling join(args, kwargs) (line 803)
    join_call_result_16975 = invoke(stypy.reporting.localization.Localization(__file__, 803, 12), join_16970, *[str_16971, str_16972, str_16973], **kwargs_16974)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16975)
    # Adding element type (line 775)
    
    # Call to join(...): (line 804)
    # Processing the call arguments (line 804)
    str_16977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 17), 'str', 'src')
    str_16978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 24), 'str', 'multiarray')
    str_16979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 38), 'str', 'methods.c')
    # Processing the call keyword arguments (line 804)
    kwargs_16980 = {}
    # Getting the type of 'join' (line 804)
    join_16976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 12), 'join', False)
    # Calling join(args, kwargs) (line 804)
    join_call_result_16981 = invoke(stypy.reporting.localization.Localization(__file__, 804, 12), join_16976, *[str_16977, str_16978, str_16979], **kwargs_16980)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16981)
    # Adding element type (line 775)
    
    # Call to join(...): (line 805)
    # Processing the call arguments (line 805)
    str_16983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 17), 'str', 'src')
    str_16984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 24), 'str', 'multiarray')
    str_16985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 38), 'str', 'multiarraymodule.c')
    # Processing the call keyword arguments (line 805)
    kwargs_16986 = {}
    # Getting the type of 'join' (line 805)
    join_16982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 12), 'join', False)
    # Calling join(args, kwargs) (line 805)
    join_call_result_16987 = invoke(stypy.reporting.localization.Localization(__file__, 805, 12), join_16982, *[str_16983, str_16984, str_16985], **kwargs_16986)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16987)
    # Adding element type (line 775)
    
    # Call to join(...): (line 806)
    # Processing the call arguments (line 806)
    str_16989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 17), 'str', 'src')
    str_16990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 24), 'str', 'multiarray')
    str_16991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 38), 'str', 'nditer_templ.c.src')
    # Processing the call keyword arguments (line 806)
    kwargs_16992 = {}
    # Getting the type of 'join' (line 806)
    join_16988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 12), 'join', False)
    # Calling join(args, kwargs) (line 806)
    join_call_result_16993 = invoke(stypy.reporting.localization.Localization(__file__, 806, 12), join_16988, *[str_16989, str_16990, str_16991], **kwargs_16992)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16993)
    # Adding element type (line 775)
    
    # Call to join(...): (line 807)
    # Processing the call arguments (line 807)
    str_16995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 17), 'str', 'src')
    str_16996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 24), 'str', 'multiarray')
    str_16997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 38), 'str', 'nditer_api.c')
    # Processing the call keyword arguments (line 807)
    kwargs_16998 = {}
    # Getting the type of 'join' (line 807)
    join_16994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 12), 'join', False)
    # Calling join(args, kwargs) (line 807)
    join_call_result_16999 = invoke(stypy.reporting.localization.Localization(__file__, 807, 12), join_16994, *[str_16995, str_16996, str_16997], **kwargs_16998)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_16999)
    # Adding element type (line 775)
    
    # Call to join(...): (line 808)
    # Processing the call arguments (line 808)
    str_17001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 17), 'str', 'src')
    str_17002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 24), 'str', 'multiarray')
    str_17003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 38), 'str', 'nditer_constr.c')
    # Processing the call keyword arguments (line 808)
    kwargs_17004 = {}
    # Getting the type of 'join' (line 808)
    join_17000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 12), 'join', False)
    # Calling join(args, kwargs) (line 808)
    join_call_result_17005 = invoke(stypy.reporting.localization.Localization(__file__, 808, 12), join_17000, *[str_17001, str_17002, str_17003], **kwargs_17004)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_17005)
    # Adding element type (line 775)
    
    # Call to join(...): (line 809)
    # Processing the call arguments (line 809)
    str_17007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, 17), 'str', 'src')
    str_17008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, 24), 'str', 'multiarray')
    str_17009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, 38), 'str', 'nditer_pywrap.c')
    # Processing the call keyword arguments (line 809)
    kwargs_17010 = {}
    # Getting the type of 'join' (line 809)
    join_17006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 12), 'join', False)
    # Calling join(args, kwargs) (line 809)
    join_call_result_17011 = invoke(stypy.reporting.localization.Localization(__file__, 809, 12), join_17006, *[str_17007, str_17008, str_17009], **kwargs_17010)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_17011)
    # Adding element type (line 775)
    
    # Call to join(...): (line 810)
    # Processing the call arguments (line 810)
    str_17013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 17), 'str', 'src')
    str_17014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 24), 'str', 'multiarray')
    str_17015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 38), 'str', 'number.c')
    # Processing the call keyword arguments (line 810)
    kwargs_17016 = {}
    # Getting the type of 'join' (line 810)
    join_17012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 12), 'join', False)
    # Calling join(args, kwargs) (line 810)
    join_call_result_17017 = invoke(stypy.reporting.localization.Localization(__file__, 810, 12), join_17012, *[str_17013, str_17014, str_17015], **kwargs_17016)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_17017)
    # Adding element type (line 775)
    
    # Call to join(...): (line 811)
    # Processing the call arguments (line 811)
    str_17019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 17), 'str', 'src')
    str_17020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 24), 'str', 'multiarray')
    str_17021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 38), 'str', 'numpymemoryview.c')
    # Processing the call keyword arguments (line 811)
    kwargs_17022 = {}
    # Getting the type of 'join' (line 811)
    join_17018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 12), 'join', False)
    # Calling join(args, kwargs) (line 811)
    join_call_result_17023 = invoke(stypy.reporting.localization.Localization(__file__, 811, 12), join_17018, *[str_17019, str_17020, str_17021], **kwargs_17022)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_17023)
    # Adding element type (line 775)
    
    # Call to join(...): (line 812)
    # Processing the call arguments (line 812)
    str_17025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 17), 'str', 'src')
    str_17026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 24), 'str', 'multiarray')
    str_17027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 38), 'str', 'numpyos.c')
    # Processing the call keyword arguments (line 812)
    kwargs_17028 = {}
    # Getting the type of 'join' (line 812)
    join_17024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 12), 'join', False)
    # Calling join(args, kwargs) (line 812)
    join_call_result_17029 = invoke(stypy.reporting.localization.Localization(__file__, 812, 12), join_17024, *[str_17025, str_17026, str_17027], **kwargs_17028)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_17029)
    # Adding element type (line 775)
    
    # Call to join(...): (line 813)
    # Processing the call arguments (line 813)
    str_17031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 17), 'str', 'src')
    str_17032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 24), 'str', 'multiarray')
    str_17033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 38), 'str', 'refcount.c')
    # Processing the call keyword arguments (line 813)
    kwargs_17034 = {}
    # Getting the type of 'join' (line 813)
    join_17030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 12), 'join', False)
    # Calling join(args, kwargs) (line 813)
    join_call_result_17035 = invoke(stypy.reporting.localization.Localization(__file__, 813, 12), join_17030, *[str_17031, str_17032, str_17033], **kwargs_17034)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_17035)
    # Adding element type (line 775)
    
    # Call to join(...): (line 814)
    # Processing the call arguments (line 814)
    str_17037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 814, 17), 'str', 'src')
    str_17038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 814, 24), 'str', 'multiarray')
    str_17039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 814, 38), 'str', 'sequence.c')
    # Processing the call keyword arguments (line 814)
    kwargs_17040 = {}
    # Getting the type of 'join' (line 814)
    join_17036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 12), 'join', False)
    # Calling join(args, kwargs) (line 814)
    join_call_result_17041 = invoke(stypy.reporting.localization.Localization(__file__, 814, 12), join_17036, *[str_17037, str_17038, str_17039], **kwargs_17040)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_17041)
    # Adding element type (line 775)
    
    # Call to join(...): (line 815)
    # Processing the call arguments (line 815)
    str_17043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 17), 'str', 'src')
    str_17044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 24), 'str', 'multiarray')
    str_17045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 38), 'str', 'shape.c')
    # Processing the call keyword arguments (line 815)
    kwargs_17046 = {}
    # Getting the type of 'join' (line 815)
    join_17042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 12), 'join', False)
    # Calling join(args, kwargs) (line 815)
    join_call_result_17047 = invoke(stypy.reporting.localization.Localization(__file__, 815, 12), join_17042, *[str_17043, str_17044, str_17045], **kwargs_17046)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_17047)
    # Adding element type (line 775)
    
    # Call to join(...): (line 816)
    # Processing the call arguments (line 816)
    str_17049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, 17), 'str', 'src')
    str_17050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, 24), 'str', 'multiarray')
    str_17051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, 38), 'str', 'scalarapi.c')
    # Processing the call keyword arguments (line 816)
    kwargs_17052 = {}
    # Getting the type of 'join' (line 816)
    join_17048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 12), 'join', False)
    # Calling join(args, kwargs) (line 816)
    join_call_result_17053 = invoke(stypy.reporting.localization.Localization(__file__, 816, 12), join_17048, *[str_17049, str_17050, str_17051], **kwargs_17052)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_17053)
    # Adding element type (line 775)
    
    # Call to join(...): (line 817)
    # Processing the call arguments (line 817)
    str_17055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 17), 'str', 'src')
    str_17056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 24), 'str', 'multiarray')
    str_17057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 38), 'str', 'scalartypes.c.src')
    # Processing the call keyword arguments (line 817)
    kwargs_17058 = {}
    # Getting the type of 'join' (line 817)
    join_17054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 12), 'join', False)
    # Calling join(args, kwargs) (line 817)
    join_call_result_17059 = invoke(stypy.reporting.localization.Localization(__file__, 817, 12), join_17054, *[str_17055, str_17056, str_17057], **kwargs_17058)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_17059)
    # Adding element type (line 775)
    
    # Call to join(...): (line 818)
    # Processing the call arguments (line 818)
    str_17061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 17), 'str', 'src')
    str_17062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 24), 'str', 'multiarray')
    str_17063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 38), 'str', 'usertypes.c')
    # Processing the call keyword arguments (line 818)
    kwargs_17064 = {}
    # Getting the type of 'join' (line 818)
    join_17060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 12), 'join', False)
    # Calling join(args, kwargs) (line 818)
    join_call_result_17065 = invoke(stypy.reporting.localization.Localization(__file__, 818, 12), join_17060, *[str_17061, str_17062, str_17063], **kwargs_17064)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_17065)
    # Adding element type (line 775)
    
    # Call to join(...): (line 819)
    # Processing the call arguments (line 819)
    str_17067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 17), 'str', 'src')
    str_17068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 24), 'str', 'multiarray')
    str_17069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 38), 'str', 'ucsnarrow.c')
    # Processing the call keyword arguments (line 819)
    kwargs_17070 = {}
    # Getting the type of 'join' (line 819)
    join_17066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 12), 'join', False)
    # Calling join(args, kwargs) (line 819)
    join_call_result_17071 = invoke(stypy.reporting.localization.Localization(__file__, 819, 12), join_17066, *[str_17067, str_17068, str_17069], **kwargs_17070)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_17071)
    # Adding element type (line 775)
    
    # Call to join(...): (line 820)
    # Processing the call arguments (line 820)
    str_17073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 17), 'str', 'src')
    str_17074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 24), 'str', 'multiarray')
    str_17075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 38), 'str', 'vdot.c')
    # Processing the call keyword arguments (line 820)
    kwargs_17076 = {}
    # Getting the type of 'join' (line 820)
    join_17072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 12), 'join', False)
    # Calling join(args, kwargs) (line 820)
    join_call_result_17077 = invoke(stypy.reporting.localization.Localization(__file__, 820, 12), join_17072, *[str_17073, str_17074, str_17075], **kwargs_17076)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_17077)
    # Adding element type (line 775)
    
    # Call to join(...): (line 821)
    # Processing the call arguments (line 821)
    str_17079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 17), 'str', 'src')
    str_17080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 24), 'str', 'private')
    str_17081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 35), 'str', 'templ_common.h.src')
    # Processing the call keyword arguments (line 821)
    kwargs_17082 = {}
    # Getting the type of 'join' (line 821)
    join_17078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 12), 'join', False)
    # Calling join(args, kwargs) (line 821)
    join_call_result_17083 = invoke(stypy.reporting.localization.Localization(__file__, 821, 12), join_17078, *[str_17079, str_17080, str_17081], **kwargs_17082)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_17083)
    # Adding element type (line 775)
    
    # Call to join(...): (line 822)
    # Processing the call arguments (line 822)
    str_17085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 822, 17), 'str', 'src')
    str_17086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 822, 24), 'str', 'private')
    str_17087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 822, 35), 'str', 'mem_overlap.c')
    # Processing the call keyword arguments (line 822)
    kwargs_17088 = {}
    # Getting the type of 'join' (line 822)
    join_17084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 12), 'join', False)
    # Calling join(args, kwargs) (line 822)
    join_call_result_17089 = invoke(stypy.reporting.localization.Localization(__file__, 822, 12), join_17084, *[str_17085, str_17086, str_17087], **kwargs_17088)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 21), list_16807, join_call_result_17089)
    
    # Assigning a type to the variable 'multiarray_src' (line 775)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 4), 'multiarray_src', list_16807)
    
    # Assigning a Call to a Name (line 825):
    
    # Assigning a Call to a Name (line 825):
    
    # Call to get_info(...): (line 825)
    # Processing the call arguments (line 825)
    str_17091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 825, 25), 'str', 'blas_opt')
    int_17092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 825, 37), 'int')
    # Processing the call keyword arguments (line 825)
    kwargs_17093 = {}
    # Getting the type of 'get_info' (line 825)
    get_info_17090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 16), 'get_info', False)
    # Calling get_info(args, kwargs) (line 825)
    get_info_call_result_17094 = invoke(stypy.reporting.localization.Localization(__file__, 825, 16), get_info_17090, *[str_17091, int_17092], **kwargs_17093)
    
    # Assigning a type to the variable 'blas_info' (line 825)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 825, 4), 'blas_info', get_info_call_result_17094)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'blas_info' (line 826)
    blas_info_17095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 7), 'blas_info')
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 826)
    tuple_17096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 826)
    # Adding element type (line 826)
    str_17097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 22), 'str', 'HAVE_CBLAS')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 826, 22), tuple_17096, str_17097)
    # Adding element type (line 826)
    # Getting the type of 'None' (line 826)
    None_17098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 36), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 826, 22), tuple_17096, None_17098)
    
    
    # Call to get(...): (line 826)
    # Processing the call arguments (line 826)
    str_17101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 59), 'str', 'define_macros')
    
    # Obtaining an instance of the builtin type 'list' (line 826)
    list_17102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 76), 'list')
    # Adding type elements to the builtin type 'list' instance (line 826)
    
    # Processing the call keyword arguments (line 826)
    kwargs_17103 = {}
    # Getting the type of 'blas_info' (line 826)
    blas_info_17099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 45), 'blas_info', False)
    # Obtaining the member 'get' of a type (line 826)
    get_17100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 826, 45), blas_info_17099, 'get')
    # Calling get(args, kwargs) (line 826)
    get_call_result_17104 = invoke(stypy.reporting.localization.Localization(__file__, 826, 45), get_17100, *[str_17101, list_17102], **kwargs_17103)
    
    # Applying the binary operator 'in' (line 826)
    result_contains_17105 = python_operator(stypy.reporting.localization.Localization(__file__, 826, 21), 'in', tuple_17096, get_call_result_17104)
    
    # Applying the binary operator 'and' (line 826)
    result_and_keyword_17106 = python_operator(stypy.reporting.localization.Localization(__file__, 826, 7), 'and', blas_info_17095, result_contains_17105)
    
    # Testing the type of an if condition (line 826)
    if_condition_17107 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 826, 4), result_and_keyword_17106)
    # Assigning a type to the variable 'if_condition_17107' (line 826)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 826, 4), 'if_condition_17107', if_condition_17107)
    # SSA begins for if statement (line 826)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 827):
    
    # Assigning a Name to a Name (line 827):
    # Getting the type of 'blas_info' (line 827)
    blas_info_17108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 21), 'blas_info')
    # Assigning a type to the variable 'extra_info' (line 827)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 827, 8), 'extra_info', blas_info_17108)
    
    # Call to extend(...): (line 830)
    # Processing the call arguments (line 830)
    
    # Obtaining an instance of the builtin type 'list' (line 830)
    list_17111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 830)
    # Adding element type (line 830)
    
    # Call to join(...): (line 830)
    # Processing the call arguments (line 830)
    str_17113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 36), 'str', 'src')
    str_17114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 43), 'str', 'multiarray')
    str_17115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 57), 'str', 'cblasfuncs.c')
    # Processing the call keyword arguments (line 830)
    kwargs_17116 = {}
    # Getting the type of 'join' (line 830)
    join_17112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 31), 'join', False)
    # Calling join(args, kwargs) (line 830)
    join_call_result_17117 = invoke(stypy.reporting.localization.Localization(__file__, 830, 31), join_17112, *[str_17113, str_17114, str_17115], **kwargs_17116)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 830, 30), list_17111, join_call_result_17117)
    # Adding element type (line 830)
    
    # Call to join(...): (line 831)
    # Processing the call arguments (line 831)
    str_17119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 36), 'str', 'src')
    str_17120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 43), 'str', 'multiarray')
    str_17121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 57), 'str', 'python_xerbla.c')
    # Processing the call keyword arguments (line 831)
    kwargs_17122 = {}
    # Getting the type of 'join' (line 831)
    join_17118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 31), 'join', False)
    # Calling join(args, kwargs) (line 831)
    join_call_result_17123 = invoke(stypy.reporting.localization.Localization(__file__, 831, 31), join_17118, *[str_17119, str_17120, str_17121], **kwargs_17122)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 830, 30), list_17111, join_call_result_17123)
    
    # Processing the call keyword arguments (line 830)
    kwargs_17124 = {}
    # Getting the type of 'multiarray_src' (line 830)
    multiarray_src_17109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 8), 'multiarray_src', False)
    # Obtaining the member 'extend' of a type (line 830)
    extend_17110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 830, 8), multiarray_src_17109, 'extend')
    # Calling extend(args, kwargs) (line 830)
    extend_call_result_17125 = invoke(stypy.reporting.localization.Localization(__file__, 830, 8), extend_17110, *[list_17111], **kwargs_17124)
    
    
    
    # Call to uses_accelerate_framework(...): (line 833)
    # Processing the call arguments (line 833)
    # Getting the type of 'blas_info' (line 833)
    blas_info_17127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 37), 'blas_info', False)
    # Processing the call keyword arguments (line 833)
    kwargs_17128 = {}
    # Getting the type of 'uses_accelerate_framework' (line 833)
    uses_accelerate_framework_17126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 11), 'uses_accelerate_framework', False)
    # Calling uses_accelerate_framework(args, kwargs) (line 833)
    uses_accelerate_framework_call_result_17129 = invoke(stypy.reporting.localization.Localization(__file__, 833, 11), uses_accelerate_framework_17126, *[blas_info_17127], **kwargs_17128)
    
    # Testing the type of an if condition (line 833)
    if_condition_17130 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 833, 8), uses_accelerate_framework_call_result_17129)
    # Assigning a type to the variable 'if_condition_17130' (line 833)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 8), 'if_condition_17130', if_condition_17130)
    # SSA begins for if statement (line 833)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to extend(...): (line 834)
    # Processing the call arguments (line 834)
    
    # Call to get_sgemv_fix(...): (line 834)
    # Processing the call keyword arguments (line 834)
    kwargs_17134 = {}
    # Getting the type of 'get_sgemv_fix' (line 834)
    get_sgemv_fix_17133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 34), 'get_sgemv_fix', False)
    # Calling get_sgemv_fix(args, kwargs) (line 834)
    get_sgemv_fix_call_result_17135 = invoke(stypy.reporting.localization.Localization(__file__, 834, 34), get_sgemv_fix_17133, *[], **kwargs_17134)
    
    # Processing the call keyword arguments (line 834)
    kwargs_17136 = {}
    # Getting the type of 'multiarray_src' (line 834)
    multiarray_src_17131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 12), 'multiarray_src', False)
    # Obtaining the member 'extend' of a type (line 834)
    extend_17132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 12), multiarray_src_17131, 'extend')
    # Calling extend(args, kwargs) (line 834)
    extend_call_result_17137 = invoke(stypy.reporting.localization.Localization(__file__, 834, 12), extend_17132, *[get_sgemv_fix_call_result_17135], **kwargs_17136)
    
    # SSA join for if statement (line 833)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 826)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Dict to a Name (line 836):
    
    # Assigning a Dict to a Name (line 836):
    
    # Obtaining an instance of the builtin type 'dict' (line 836)
    dict_17138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 836, 21), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 836)
    
    # Assigning a type to the variable 'extra_info' (line 836)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 836, 8), 'extra_info', dict_17138)
    # SSA join for if statement (line 826)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to add_extension(...): (line 838)
    # Processing the call arguments (line 838)
    str_17141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 25), 'str', 'multiarray')
    # Processing the call keyword arguments (line 838)
    # Getting the type of 'multiarray_src' (line 839)
    multiarray_src_17142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 33), 'multiarray_src', False)
    
    # Obtaining an instance of the builtin type 'list' (line 840)
    list_17143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 840)
    # Adding element type (line 840)
    # Getting the type of 'generate_config_h' (line 840)
    generate_config_h_17144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 34), 'generate_config_h', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 33), list_17143, generate_config_h_17144)
    # Adding element type (line 840)
    # Getting the type of 'generate_numpyconfig_h' (line 841)
    generate_numpyconfig_h_17145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 34), 'generate_numpyconfig_h', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 33), list_17143, generate_numpyconfig_h_17145)
    # Adding element type (line 840)
    # Getting the type of 'generate_numpy_api' (line 842)
    generate_numpy_api_17146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 34), 'generate_numpy_api', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 33), list_17143, generate_numpy_api_17146)
    # Adding element type (line 840)
    
    # Call to join(...): (line 843)
    # Processing the call arguments (line 843)
    # Getting the type of 'codegen_dir' (line 843)
    codegen_dir_17148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 39), 'codegen_dir', False)
    str_17149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 843, 52), 'str', 'generate_numpy_api.py')
    # Processing the call keyword arguments (line 843)
    kwargs_17150 = {}
    # Getting the type of 'join' (line 843)
    join_17147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 34), 'join', False)
    # Calling join(args, kwargs) (line 843)
    join_call_result_17151 = invoke(stypy.reporting.localization.Localization(__file__, 843, 34), join_17147, *[codegen_dir_17148, str_17149], **kwargs_17150)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 33), list_17143, join_call_result_17151)
    # Adding element type (line 840)
    
    # Call to join(...): (line 844)
    # Processing the call arguments (line 844)
    str_17153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 844, 39), 'str', '*.py')
    # Processing the call keyword arguments (line 844)
    kwargs_17154 = {}
    # Getting the type of 'join' (line 844)
    join_17152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 34), 'join', False)
    # Calling join(args, kwargs) (line 844)
    join_call_result_17155 = invoke(stypy.reporting.localization.Localization(__file__, 844, 34), join_17152, *[str_17153], **kwargs_17154)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 33), list_17143, join_call_result_17155)
    
    # Applying the binary operator '+' (line 839)
    result_add_17156 = python_operator(stypy.reporting.localization.Localization(__file__, 839, 33), '+', multiarray_src_17142, list_17143)
    
    keyword_17157 = result_add_17156
    # Getting the type of 'deps' (line 845)
    deps_17158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 33), 'deps', False)
    # Getting the type of 'multiarray_deps' (line 845)
    multiarray_deps_17159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 40), 'multiarray_deps', False)
    # Applying the binary operator '+' (line 845)
    result_add_17160 = python_operator(stypy.reporting.localization.Localization(__file__, 845, 33), '+', deps_17158, multiarray_deps_17159)
    
    keyword_17161 = result_add_17160
    
    # Obtaining an instance of the builtin type 'list' (line 846)
    list_17162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 846)
    # Adding element type (line 846)
    str_17163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 36), 'str', 'npymath')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 846, 35), list_17162, str_17163)
    # Adding element type (line 846)
    str_17164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 47), 'str', 'npysort')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 846, 35), list_17162, str_17164)
    
    keyword_17165 = list_17162
    # Getting the type of 'extra_info' (line 847)
    extra_info_17166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 36), 'extra_info', False)
    keyword_17167 = extra_info_17166
    kwargs_17168 = {'libraries': keyword_17165, 'sources': keyword_17157, 'depends': keyword_17161, 'extra_info': keyword_17167}
    # Getting the type of 'config' (line 838)
    config_17139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 838)
    add_extension_17140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 4), config_17139, 'add_extension')
    # Calling add_extension(args, kwargs) (line 838)
    add_extension_call_result_17169 = invoke(stypy.reporting.localization.Localization(__file__, 838, 4), add_extension_17140, *[str_17141], **kwargs_17168)
    

    @norecursion
    def generate_umath_templated_sources(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'generate_umath_templated_sources'
        module_type_store = module_type_store.open_function_context('generate_umath_templated_sources', 856, 4, False)
        
        # Passed parameters checking function
        generate_umath_templated_sources.stypy_localization = localization
        generate_umath_templated_sources.stypy_type_of_self = None
        generate_umath_templated_sources.stypy_type_store = module_type_store
        generate_umath_templated_sources.stypy_function_name = 'generate_umath_templated_sources'
        generate_umath_templated_sources.stypy_param_names_list = ['ext', 'build_dir']
        generate_umath_templated_sources.stypy_varargs_param_name = None
        generate_umath_templated_sources.stypy_kwargs_param_name = None
        generate_umath_templated_sources.stypy_call_defaults = defaults
        generate_umath_templated_sources.stypy_call_varargs = varargs
        generate_umath_templated_sources.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'generate_umath_templated_sources', ['ext', 'build_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'generate_umath_templated_sources', localization, ['ext', 'build_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'generate_umath_templated_sources(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 857, 8))
        
        # 'from numpy.distutils.misc_util import get_cmd' statement (line 857)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
        import_17170 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 857, 8), 'numpy.distutils.misc_util')

        if (type(import_17170) is not StypyTypeError):

            if (import_17170 != 'pyd_module'):
                __import__(import_17170)
                sys_modules_17171 = sys.modules[import_17170]
                import_from_module(stypy.reporting.localization.Localization(__file__, 857, 8), 'numpy.distutils.misc_util', sys_modules_17171.module_type_store, module_type_store, ['get_cmd'])
                nest_module(stypy.reporting.localization.Localization(__file__, 857, 8), __file__, sys_modules_17171, sys_modules_17171.module_type_store, module_type_store)
            else:
                from numpy.distutils.misc_util import get_cmd

                import_from_module(stypy.reporting.localization.Localization(__file__, 857, 8), 'numpy.distutils.misc_util', None, module_type_store, ['get_cmd'], [get_cmd])

        else:
            # Assigning a type to the variable 'numpy.distutils.misc_util' (line 857)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 857, 8), 'numpy.distutils.misc_util', import_17170)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')
        
        
        # Assigning a Call to a Name (line 859):
        
        # Assigning a Call to a Name (line 859):
        
        # Call to join(...): (line 859)
        # Processing the call arguments (line 859)
        str_17173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 859, 23), 'str', 'src')
        str_17174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 859, 30), 'str', 'umath')
        # Processing the call keyword arguments (line 859)
        kwargs_17175 = {}
        # Getting the type of 'join' (line 859)
        join_17172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 18), 'join', False)
        # Calling join(args, kwargs) (line 859)
        join_call_result_17176 = invoke(stypy.reporting.localization.Localization(__file__, 859, 18), join_17172, *[str_17173, str_17174], **kwargs_17175)
        
        # Assigning a type to the variable 'subpath' (line 859)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 859, 8), 'subpath', join_call_result_17176)
        
        # Assigning a List to a Name (line 860):
        
        # Assigning a List to a Name (line 860):
        
        # Obtaining an instance of the builtin type 'list' (line 860)
        list_17177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 860, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 860)
        # Adding element type (line 860)
        
        # Call to join(...): (line 861)
        # Processing the call arguments (line 861)
        # Getting the type of 'local_dir' (line 861)
        local_dir_17179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 17), 'local_dir', False)
        # Getting the type of 'subpath' (line 861)
        subpath_17180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 28), 'subpath', False)
        str_17181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 861, 37), 'str', 'loops.h.src')
        # Processing the call keyword arguments (line 861)
        kwargs_17182 = {}
        # Getting the type of 'join' (line 861)
        join_17178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 12), 'join', False)
        # Calling join(args, kwargs) (line 861)
        join_call_result_17183 = invoke(stypy.reporting.localization.Localization(__file__, 861, 12), join_17178, *[local_dir_17179, subpath_17180, str_17181], **kwargs_17182)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 860, 18), list_17177, join_call_result_17183)
        # Adding element type (line 860)
        
        # Call to join(...): (line 862)
        # Processing the call arguments (line 862)
        # Getting the type of 'local_dir' (line 862)
        local_dir_17185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 17), 'local_dir', False)
        # Getting the type of 'subpath' (line 862)
        subpath_17186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 28), 'subpath', False)
        str_17187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 862, 37), 'str', 'loops.c.src')
        # Processing the call keyword arguments (line 862)
        kwargs_17188 = {}
        # Getting the type of 'join' (line 862)
        join_17184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 12), 'join', False)
        # Calling join(args, kwargs) (line 862)
        join_call_result_17189 = invoke(stypy.reporting.localization.Localization(__file__, 862, 12), join_17184, *[local_dir_17185, subpath_17186, str_17187], **kwargs_17188)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 860, 18), list_17177, join_call_result_17189)
        # Adding element type (line 860)
        
        # Call to join(...): (line 863)
        # Processing the call arguments (line 863)
        # Getting the type of 'local_dir' (line 863)
        local_dir_17191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 17), 'local_dir', False)
        # Getting the type of 'subpath' (line 863)
        subpath_17192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 28), 'subpath', False)
        str_17193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 863, 37), 'str', 'scalarmath.c.src')
        # Processing the call keyword arguments (line 863)
        kwargs_17194 = {}
        # Getting the type of 'join' (line 863)
        join_17190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 12), 'join', False)
        # Calling join(args, kwargs) (line 863)
        join_call_result_17195 = invoke(stypy.reporting.localization.Localization(__file__, 863, 12), join_17190, *[local_dir_17191, subpath_17192, str_17193], **kwargs_17194)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 860, 18), list_17177, join_call_result_17195)
        # Adding element type (line 860)
        
        # Call to join(...): (line 864)
        # Processing the call arguments (line 864)
        # Getting the type of 'local_dir' (line 864)
        local_dir_17197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 17), 'local_dir', False)
        # Getting the type of 'subpath' (line 864)
        subpath_17198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 28), 'subpath', False)
        str_17199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 864, 37), 'str', 'simd.inc.src')
        # Processing the call keyword arguments (line 864)
        kwargs_17200 = {}
        # Getting the type of 'join' (line 864)
        join_17196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 12), 'join', False)
        # Calling join(args, kwargs) (line 864)
        join_call_result_17201 = invoke(stypy.reporting.localization.Localization(__file__, 864, 12), join_17196, *[local_dir_17197, subpath_17198, str_17199], **kwargs_17200)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 860, 18), list_17177, join_call_result_17201)
        
        # Assigning a type to the variable 'sources' (line 860)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 860, 8), 'sources', list_17177)
        
        # Call to add_include_dirs(...): (line 868)
        # Processing the call arguments (line 868)
        
        # Call to join(...): (line 868)
        # Processing the call arguments (line 868)
        # Getting the type of 'build_dir' (line 868)
        build_dir_17205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 37), 'build_dir', False)
        # Getting the type of 'subpath' (line 868)
        subpath_17206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 48), 'subpath', False)
        # Processing the call keyword arguments (line 868)
        kwargs_17207 = {}
        # Getting the type of 'join' (line 868)
        join_17204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 32), 'join', False)
        # Calling join(args, kwargs) (line 868)
        join_call_result_17208 = invoke(stypy.reporting.localization.Localization(__file__, 868, 32), join_17204, *[build_dir_17205, subpath_17206], **kwargs_17207)
        
        # Processing the call keyword arguments (line 868)
        kwargs_17209 = {}
        # Getting the type of 'config' (line 868)
        config_17202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 8), 'config', False)
        # Obtaining the member 'add_include_dirs' of a type (line 868)
        add_include_dirs_17203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 868, 8), config_17202, 'add_include_dirs')
        # Calling add_include_dirs(args, kwargs) (line 868)
        add_include_dirs_call_result_17210 = invoke(stypy.reporting.localization.Localization(__file__, 868, 8), add_include_dirs_17203, *[join_call_result_17208], **kwargs_17209)
        
        
        # Assigning a Call to a Name (line 869):
        
        # Assigning a Call to a Name (line 869):
        
        # Call to get_cmd(...): (line 869)
        # Processing the call arguments (line 869)
        str_17212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 869, 22), 'str', 'build_src')
        # Processing the call keyword arguments (line 869)
        kwargs_17213 = {}
        # Getting the type of 'get_cmd' (line 869)
        get_cmd_17211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 14), 'get_cmd', False)
        # Calling get_cmd(args, kwargs) (line 869)
        get_cmd_call_result_17214 = invoke(stypy.reporting.localization.Localization(__file__, 869, 14), get_cmd_17211, *[str_17212], **kwargs_17213)
        
        # Assigning a type to the variable 'cmd' (line 869)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 869, 8), 'cmd', get_cmd_call_result_17214)
        
        # Call to ensure_finalized(...): (line 870)
        # Processing the call keyword arguments (line 870)
        kwargs_17217 = {}
        # Getting the type of 'cmd' (line 870)
        cmd_17215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 870)
        ensure_finalized_17216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 870, 8), cmd_17215, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 870)
        ensure_finalized_call_result_17218 = invoke(stypy.reporting.localization.Localization(__file__, 870, 8), ensure_finalized_17216, *[], **kwargs_17217)
        
        
        # Call to template_sources(...): (line 871)
        # Processing the call arguments (line 871)
        # Getting the type of 'sources' (line 871)
        sources_17221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 29), 'sources', False)
        # Getting the type of 'ext' (line 871)
        ext_17222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 38), 'ext', False)
        # Processing the call keyword arguments (line 871)
        kwargs_17223 = {}
        # Getting the type of 'cmd' (line 871)
        cmd_17219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 8), 'cmd', False)
        # Obtaining the member 'template_sources' of a type (line 871)
        template_sources_17220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 8), cmd_17219, 'template_sources')
        # Calling template_sources(args, kwargs) (line 871)
        template_sources_call_result_17224 = invoke(stypy.reporting.localization.Localization(__file__, 871, 8), template_sources_17220, *[sources_17221, ext_17222], **kwargs_17223)
        
        
        # ################# End of 'generate_umath_templated_sources(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generate_umath_templated_sources' in the type store
        # Getting the type of 'stypy_return_type' (line 856)
        stypy_return_type_17225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17225)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generate_umath_templated_sources'
        return stypy_return_type_17225

    # Assigning a type to the variable 'generate_umath_templated_sources' (line 856)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 856, 4), 'generate_umath_templated_sources', generate_umath_templated_sources)

    @norecursion
    def generate_umath_c(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'generate_umath_c'
        module_type_store = module_type_store.open_function_context('generate_umath_c', 873, 4, False)
        
        # Passed parameters checking function
        generate_umath_c.stypy_localization = localization
        generate_umath_c.stypy_type_of_self = None
        generate_umath_c.stypy_type_store = module_type_store
        generate_umath_c.stypy_function_name = 'generate_umath_c'
        generate_umath_c.stypy_param_names_list = ['ext', 'build_dir']
        generate_umath_c.stypy_varargs_param_name = None
        generate_umath_c.stypy_kwargs_param_name = None
        generate_umath_c.stypy_call_defaults = defaults
        generate_umath_c.stypy_call_varargs = varargs
        generate_umath_c.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'generate_umath_c', ['ext', 'build_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'generate_umath_c', localization, ['ext', 'build_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'generate_umath_c(...)' code ##################

        
        # Assigning a Call to a Name (line 874):
        
        # Assigning a Call to a Name (line 874):
        
        # Call to join(...): (line 874)
        # Processing the call arguments (line 874)
        # Getting the type of 'build_dir' (line 874)
        build_dir_17227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 22), 'build_dir', False)
        # Getting the type of 'header_dir' (line 874)
        header_dir_17228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 33), 'header_dir', False)
        str_17229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 45), 'str', '__umath_generated.c')
        # Processing the call keyword arguments (line 874)
        kwargs_17230 = {}
        # Getting the type of 'join' (line 874)
        join_17226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 17), 'join', False)
        # Calling join(args, kwargs) (line 874)
        join_call_result_17231 = invoke(stypy.reporting.localization.Localization(__file__, 874, 17), join_17226, *[build_dir_17227, header_dir_17228, str_17229], **kwargs_17230)
        
        # Assigning a type to the variable 'target' (line 874)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 874, 8), 'target', join_call_result_17231)
        
        # Assigning a Call to a Name (line 875):
        
        # Assigning a Call to a Name (line 875):
        
        # Call to dirname(...): (line 875)
        # Processing the call arguments (line 875)
        # Getting the type of 'target' (line 875)
        target_17235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 30), 'target', False)
        # Processing the call keyword arguments (line 875)
        kwargs_17236 = {}
        # Getting the type of 'os' (line 875)
        os_17232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 14), 'os', False)
        # Obtaining the member 'path' of a type (line 875)
        path_17233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 875, 14), os_17232, 'path')
        # Obtaining the member 'dirname' of a type (line 875)
        dirname_17234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 875, 14), path_17233, 'dirname')
        # Calling dirname(args, kwargs) (line 875)
        dirname_call_result_17237 = invoke(stypy.reporting.localization.Localization(__file__, 875, 14), dirname_17234, *[target_17235], **kwargs_17236)
        
        # Assigning a type to the variable 'dir' (line 875)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 875, 8), 'dir', dirname_call_result_17237)
        
        
        
        # Call to exists(...): (line 876)
        # Processing the call arguments (line 876)
        # Getting the type of 'dir' (line 876)
        dir_17241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 30), 'dir', False)
        # Processing the call keyword arguments (line 876)
        kwargs_17242 = {}
        # Getting the type of 'os' (line 876)
        os_17238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 876)
        path_17239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 876, 15), os_17238, 'path')
        # Obtaining the member 'exists' of a type (line 876)
        exists_17240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 876, 15), path_17239, 'exists')
        # Calling exists(args, kwargs) (line 876)
        exists_call_result_17243 = invoke(stypy.reporting.localization.Localization(__file__, 876, 15), exists_17240, *[dir_17241], **kwargs_17242)
        
        # Applying the 'not' unary operator (line 876)
        result_not__17244 = python_operator(stypy.reporting.localization.Localization(__file__, 876, 11), 'not', exists_call_result_17243)
        
        # Testing the type of an if condition (line 876)
        if_condition_17245 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 876, 8), result_not__17244)
        # Assigning a type to the variable 'if_condition_17245' (line 876)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'if_condition_17245', if_condition_17245)
        # SSA begins for if statement (line 876)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to makedirs(...): (line 877)
        # Processing the call arguments (line 877)
        # Getting the type of 'dir' (line 877)
        dir_17248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 24), 'dir', False)
        # Processing the call keyword arguments (line 877)
        kwargs_17249 = {}
        # Getting the type of 'os' (line 877)
        os_17246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 12), 'os', False)
        # Obtaining the member 'makedirs' of a type (line 877)
        makedirs_17247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 12), os_17246, 'makedirs')
        # Calling makedirs(args, kwargs) (line 877)
        makedirs_call_result_17250 = invoke(stypy.reporting.localization.Localization(__file__, 877, 12), makedirs_17247, *[dir_17248], **kwargs_17249)
        
        # SSA join for if statement (line 876)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 878):
        
        # Assigning a Name to a Name (line 878):
        # Getting the type of 'generate_umath_py' (line 878)
        generate_umath_py_17251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 17), 'generate_umath_py')
        # Assigning a type to the variable 'script' (line 878)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 8), 'script', generate_umath_py_17251)
        
        
        # Call to newer(...): (line 879)
        # Processing the call arguments (line 879)
        # Getting the type of 'script' (line 879)
        script_17253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 17), 'script', False)
        # Getting the type of 'target' (line 879)
        target_17254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 25), 'target', False)
        # Processing the call keyword arguments (line 879)
        kwargs_17255 = {}
        # Getting the type of 'newer' (line 879)
        newer_17252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 11), 'newer', False)
        # Calling newer(args, kwargs) (line 879)
        newer_call_result_17256 = invoke(stypy.reporting.localization.Localization(__file__, 879, 11), newer_17252, *[script_17253, target_17254], **kwargs_17255)
        
        # Testing the type of an if condition (line 879)
        if_condition_17257 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 879, 8), newer_call_result_17256)
        # Assigning a type to the variable 'if_condition_17257' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'if_condition_17257', if_condition_17257)
        # SSA begins for if statement (line 879)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 880):
        
        # Assigning a Call to a Name (line 880):
        
        # Call to open(...): (line 880)
        # Processing the call arguments (line 880)
        # Getting the type of 'target' (line 880)
        target_17259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 21), 'target', False)
        str_17260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 29), 'str', 'w')
        # Processing the call keyword arguments (line 880)
        kwargs_17261 = {}
        # Getting the type of 'open' (line 880)
        open_17258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 16), 'open', False)
        # Calling open(args, kwargs) (line 880)
        open_call_result_17262 = invoke(stypy.reporting.localization.Localization(__file__, 880, 16), open_17258, *[target_17259, str_17260], **kwargs_17261)
        
        # Assigning a type to the variable 'f' (line 880)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 12), 'f', open_call_result_17262)
        
        # Call to write(...): (line 881)
        # Processing the call arguments (line 881)
        
        # Call to make_code(...): (line 881)
        # Processing the call arguments (line 881)
        # Getting the type of 'generate_umath' (line 881)
        generate_umath_17267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 45), 'generate_umath', False)
        # Obtaining the member 'defdict' of a type (line 881)
        defdict_17268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 881, 45), generate_umath_17267, 'defdict')
        # Getting the type of 'generate_umath' (line 882)
        generate_umath_17269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 45), 'generate_umath', False)
        # Obtaining the member '__file__' of a type (line 882)
        file___17270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 882, 45), generate_umath_17269, '__file__')
        # Processing the call keyword arguments (line 881)
        kwargs_17271 = {}
        # Getting the type of 'generate_umath' (line 881)
        generate_umath_17265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 20), 'generate_umath', False)
        # Obtaining the member 'make_code' of a type (line 881)
        make_code_17266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 881, 20), generate_umath_17265, 'make_code')
        # Calling make_code(args, kwargs) (line 881)
        make_code_call_result_17272 = invoke(stypy.reporting.localization.Localization(__file__, 881, 20), make_code_17266, *[defdict_17268, file___17270], **kwargs_17271)
        
        # Processing the call keyword arguments (line 881)
        kwargs_17273 = {}
        # Getting the type of 'f' (line 881)
        f_17263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 881)
        write_17264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 881, 12), f_17263, 'write')
        # Calling write(args, kwargs) (line 881)
        write_call_result_17274 = invoke(stypy.reporting.localization.Localization(__file__, 881, 12), write_17264, *[make_code_call_result_17272], **kwargs_17273)
        
        
        # Call to close(...): (line 883)
        # Processing the call keyword arguments (line 883)
        kwargs_17277 = {}
        # Getting the type of 'f' (line 883)
        f_17275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 883)
        close_17276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 883, 12), f_17275, 'close')
        # Calling close(args, kwargs) (line 883)
        close_call_result_17278 = invoke(stypy.reporting.localization.Localization(__file__, 883, 12), close_17276, *[], **kwargs_17277)
        
        # SSA join for if statement (line 879)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'list' (line 884)
        list_17279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 884, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 884)
        
        # Assigning a type to the variable 'stypy_return_type' (line 884)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 884, 8), 'stypy_return_type', list_17279)
        
        # ################# End of 'generate_umath_c(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generate_umath_c' in the type store
        # Getting the type of 'stypy_return_type' (line 873)
        stypy_return_type_17280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17280)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generate_umath_c'
        return stypy_return_type_17280

    # Assigning a type to the variable 'generate_umath_c' (line 873)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 873, 4), 'generate_umath_c', generate_umath_c)
    
    # Assigning a List to a Name (line 886):
    
    # Assigning a List to a Name (line 886):
    
    # Obtaining an instance of the builtin type 'list' (line 886)
    list_17281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 886, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 886)
    # Adding element type (line 886)
    
    # Call to join(...): (line 887)
    # Processing the call arguments (line 887)
    str_17283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, 17), 'str', 'src')
    str_17284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, 24), 'str', 'umath')
    str_17285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, 33), 'str', 'umathmodule.c')
    # Processing the call keyword arguments (line 887)
    kwargs_17286 = {}
    # Getting the type of 'join' (line 887)
    join_17282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 12), 'join', False)
    # Calling join(args, kwargs) (line 887)
    join_call_result_17287 = invoke(stypy.reporting.localization.Localization(__file__, 887, 12), join_17282, *[str_17283, str_17284, str_17285], **kwargs_17286)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 886, 16), list_17281, join_call_result_17287)
    # Adding element type (line 886)
    
    # Call to join(...): (line 888)
    # Processing the call arguments (line 888)
    str_17289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 17), 'str', 'src')
    str_17290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 24), 'str', 'umath')
    str_17291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 33), 'str', 'reduction.c')
    # Processing the call keyword arguments (line 888)
    kwargs_17292 = {}
    # Getting the type of 'join' (line 888)
    join_17288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 12), 'join', False)
    # Calling join(args, kwargs) (line 888)
    join_call_result_17293 = invoke(stypy.reporting.localization.Localization(__file__, 888, 12), join_17288, *[str_17289, str_17290, str_17291], **kwargs_17292)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 886, 16), list_17281, join_call_result_17293)
    # Adding element type (line 886)
    
    # Call to join(...): (line 889)
    # Processing the call arguments (line 889)
    str_17295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 889, 17), 'str', 'src')
    str_17296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 889, 24), 'str', 'umath')
    str_17297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 889, 33), 'str', 'funcs.inc.src')
    # Processing the call keyword arguments (line 889)
    kwargs_17298 = {}
    # Getting the type of 'join' (line 889)
    join_17294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 12), 'join', False)
    # Calling join(args, kwargs) (line 889)
    join_call_result_17299 = invoke(stypy.reporting.localization.Localization(__file__, 889, 12), join_17294, *[str_17295, str_17296, str_17297], **kwargs_17298)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 886, 16), list_17281, join_call_result_17299)
    # Adding element type (line 886)
    
    # Call to join(...): (line 890)
    # Processing the call arguments (line 890)
    str_17301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 890, 17), 'str', 'src')
    str_17302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 890, 24), 'str', 'umath')
    str_17303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 890, 33), 'str', 'simd.inc.src')
    # Processing the call keyword arguments (line 890)
    kwargs_17304 = {}
    # Getting the type of 'join' (line 890)
    join_17300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 12), 'join', False)
    # Calling join(args, kwargs) (line 890)
    join_call_result_17305 = invoke(stypy.reporting.localization.Localization(__file__, 890, 12), join_17300, *[str_17301, str_17302, str_17303], **kwargs_17304)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 886, 16), list_17281, join_call_result_17305)
    # Adding element type (line 886)
    
    # Call to join(...): (line 891)
    # Processing the call arguments (line 891)
    str_17307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 891, 17), 'str', 'src')
    str_17308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 891, 24), 'str', 'umath')
    str_17309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 891, 33), 'str', 'loops.h.src')
    # Processing the call keyword arguments (line 891)
    kwargs_17310 = {}
    # Getting the type of 'join' (line 891)
    join_17306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 12), 'join', False)
    # Calling join(args, kwargs) (line 891)
    join_call_result_17311 = invoke(stypy.reporting.localization.Localization(__file__, 891, 12), join_17306, *[str_17307, str_17308, str_17309], **kwargs_17310)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 886, 16), list_17281, join_call_result_17311)
    # Adding element type (line 886)
    
    # Call to join(...): (line 892)
    # Processing the call arguments (line 892)
    str_17313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 892, 17), 'str', 'src')
    str_17314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 892, 24), 'str', 'umath')
    str_17315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 892, 33), 'str', 'loops.c.src')
    # Processing the call keyword arguments (line 892)
    kwargs_17316 = {}
    # Getting the type of 'join' (line 892)
    join_17312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 12), 'join', False)
    # Calling join(args, kwargs) (line 892)
    join_call_result_17317 = invoke(stypy.reporting.localization.Localization(__file__, 892, 12), join_17312, *[str_17313, str_17314, str_17315], **kwargs_17316)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 886, 16), list_17281, join_call_result_17317)
    # Adding element type (line 886)
    
    # Call to join(...): (line 893)
    # Processing the call arguments (line 893)
    str_17319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 17), 'str', 'src')
    str_17320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 24), 'str', 'umath')
    str_17321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 33), 'str', 'ufunc_object.c')
    # Processing the call keyword arguments (line 893)
    kwargs_17322 = {}
    # Getting the type of 'join' (line 893)
    join_17318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 12), 'join', False)
    # Calling join(args, kwargs) (line 893)
    join_call_result_17323 = invoke(stypy.reporting.localization.Localization(__file__, 893, 12), join_17318, *[str_17319, str_17320, str_17321], **kwargs_17322)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 886, 16), list_17281, join_call_result_17323)
    # Adding element type (line 886)
    
    # Call to join(...): (line 894)
    # Processing the call arguments (line 894)
    str_17325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, 17), 'str', 'src')
    str_17326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, 24), 'str', 'umath')
    str_17327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, 33), 'str', 'scalarmath.c.src')
    # Processing the call keyword arguments (line 894)
    kwargs_17328 = {}
    # Getting the type of 'join' (line 894)
    join_17324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 12), 'join', False)
    # Calling join(args, kwargs) (line 894)
    join_call_result_17329 = invoke(stypy.reporting.localization.Localization(__file__, 894, 12), join_17324, *[str_17325, str_17326, str_17327], **kwargs_17328)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 886, 16), list_17281, join_call_result_17329)
    # Adding element type (line 886)
    
    # Call to join(...): (line 895)
    # Processing the call arguments (line 895)
    str_17331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 895, 17), 'str', 'src')
    str_17332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 895, 24), 'str', 'umath')
    str_17333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 895, 33), 'str', 'ufunc_type_resolution.c')
    # Processing the call keyword arguments (line 895)
    kwargs_17334 = {}
    # Getting the type of 'join' (line 895)
    join_17330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 12), 'join', False)
    # Calling join(args, kwargs) (line 895)
    join_call_result_17335 = invoke(stypy.reporting.localization.Localization(__file__, 895, 12), join_17330, *[str_17331, str_17332, str_17333], **kwargs_17334)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 886, 16), list_17281, join_call_result_17335)
    
    # Assigning a type to the variable 'umath_src' (line 886)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 886, 4), 'umath_src', list_17281)
    
    # Assigning a BinOp to a Name (line 897):
    
    # Assigning a BinOp to a Name (line 897):
    
    # Obtaining an instance of the builtin type 'list' (line 897)
    list_17336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 897)
    # Adding element type (line 897)
    # Getting the type of 'generate_umath_py' (line 898)
    generate_umath_py_17337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 12), 'generate_umath_py')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 17), list_17336, generate_umath_py_17337)
    # Adding element type (line 897)
    
    # Call to join(...): (line 899)
    # Processing the call arguments (line 899)
    str_17339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 17), 'str', 'include')
    str_17340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 28), 'str', 'numpy')
    str_17341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 37), 'str', 'npy_math.h')
    # Processing the call keyword arguments (line 899)
    kwargs_17342 = {}
    # Getting the type of 'join' (line 899)
    join_17338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 12), 'join', False)
    # Calling join(args, kwargs) (line 899)
    join_call_result_17343 = invoke(stypy.reporting.localization.Localization(__file__, 899, 12), join_17338, *[str_17339, str_17340, str_17341], **kwargs_17342)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 17), list_17336, join_call_result_17343)
    # Adding element type (line 897)
    
    # Call to join(...): (line 900)
    # Processing the call arguments (line 900)
    str_17345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 900, 17), 'str', 'include')
    str_17346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 900, 28), 'str', 'numpy')
    str_17347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 900, 37), 'str', 'halffloat.h')
    # Processing the call keyword arguments (line 900)
    kwargs_17348 = {}
    # Getting the type of 'join' (line 900)
    join_17344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'join', False)
    # Calling join(args, kwargs) (line 900)
    join_call_result_17349 = invoke(stypy.reporting.localization.Localization(__file__, 900, 12), join_17344, *[str_17345, str_17346, str_17347], **kwargs_17348)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 17), list_17336, join_call_result_17349)
    # Adding element type (line 897)
    
    # Call to join(...): (line 901)
    # Processing the call arguments (line 901)
    str_17351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 901, 17), 'str', 'src')
    str_17352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 901, 24), 'str', 'multiarray')
    str_17353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 901, 38), 'str', 'common.h')
    # Processing the call keyword arguments (line 901)
    kwargs_17354 = {}
    # Getting the type of 'join' (line 901)
    join_17350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 12), 'join', False)
    # Calling join(args, kwargs) (line 901)
    join_call_result_17355 = invoke(stypy.reporting.localization.Localization(__file__, 901, 12), join_17350, *[str_17351, str_17352, str_17353], **kwargs_17354)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 17), list_17336, join_call_result_17355)
    # Adding element type (line 897)
    
    # Call to join(...): (line 902)
    # Processing the call arguments (line 902)
    str_17357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, 17), 'str', 'src')
    str_17358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, 24), 'str', 'private')
    str_17359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, 35), 'str', 'templ_common.h.src')
    # Processing the call keyword arguments (line 902)
    kwargs_17360 = {}
    # Getting the type of 'join' (line 902)
    join_17356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 12), 'join', False)
    # Calling join(args, kwargs) (line 902)
    join_call_result_17361 = invoke(stypy.reporting.localization.Localization(__file__, 902, 12), join_17356, *[str_17357, str_17358, str_17359], **kwargs_17360)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 17), list_17336, join_call_result_17361)
    # Adding element type (line 897)
    
    # Call to join(...): (line 903)
    # Processing the call arguments (line 903)
    str_17363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 903, 17), 'str', 'src')
    str_17364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 903, 24), 'str', 'umath')
    str_17365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 903, 33), 'str', 'simd.inc.src')
    # Processing the call keyword arguments (line 903)
    kwargs_17366 = {}
    # Getting the type of 'join' (line 903)
    join_17362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 12), 'join', False)
    # Calling join(args, kwargs) (line 903)
    join_call_result_17367 = invoke(stypy.reporting.localization.Localization(__file__, 903, 12), join_17362, *[str_17363, str_17364, str_17365], **kwargs_17366)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 17), list_17336, join_call_result_17367)
    # Adding element type (line 897)
    
    # Call to join(...): (line 904)
    # Processing the call arguments (line 904)
    # Getting the type of 'codegen_dir' (line 904)
    codegen_dir_17369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 17), 'codegen_dir', False)
    str_17370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 904, 30), 'str', 'generate_ufunc_api.py')
    # Processing the call keyword arguments (line 904)
    kwargs_17371 = {}
    # Getting the type of 'join' (line 904)
    join_17368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 12), 'join', False)
    # Calling join(args, kwargs) (line 904)
    join_call_result_17372 = invoke(stypy.reporting.localization.Localization(__file__, 904, 12), join_17368, *[codegen_dir_17369, str_17370], **kwargs_17371)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 17), list_17336, join_call_result_17372)
    # Adding element type (line 897)
    
    # Call to join(...): (line 905)
    # Processing the call arguments (line 905)
    str_17374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 905, 17), 'str', 'src')
    str_17375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 905, 24), 'str', 'private')
    str_17376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 905, 35), 'str', 'ufunc_override.h')
    # Processing the call keyword arguments (line 905)
    kwargs_17377 = {}
    # Getting the type of 'join' (line 905)
    join_17373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 12), 'join', False)
    # Calling join(args, kwargs) (line 905)
    join_call_result_17378 = invoke(stypy.reporting.localization.Localization(__file__, 905, 12), join_17373, *[str_17374, str_17375, str_17376], **kwargs_17377)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 17), list_17336, join_call_result_17378)
    
    # Getting the type of 'npymath_sources' (line 905)
    npymath_sources_17379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 58), 'npymath_sources')
    # Applying the binary operator '+' (line 897)
    result_add_17380 = python_operator(stypy.reporting.localization.Localization(__file__, 897, 17), '+', list_17336, npymath_sources_17379)
    
    # Assigning a type to the variable 'umath_deps' (line 897)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 4), 'umath_deps', result_add_17380)
    
    # Call to add_extension(...): (line 907)
    # Processing the call arguments (line 907)
    str_17383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 907, 25), 'str', 'umath')
    # Processing the call keyword arguments (line 907)
    # Getting the type of 'umath_src' (line 908)
    umath_src_17384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 33), 'umath_src', False)
    
    # Obtaining an instance of the builtin type 'list' (line 909)
    list_17385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 909, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 909)
    # Adding element type (line 909)
    # Getting the type of 'generate_config_h' (line 909)
    generate_config_h_17386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 34), 'generate_config_h', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 909, 33), list_17385, generate_config_h_17386)
    # Adding element type (line 909)
    # Getting the type of 'generate_numpyconfig_h' (line 910)
    generate_numpyconfig_h_17387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 33), 'generate_numpyconfig_h', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 909, 33), list_17385, generate_numpyconfig_h_17387)
    # Adding element type (line 909)
    # Getting the type of 'generate_umath_c' (line 911)
    generate_umath_c_17388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 33), 'generate_umath_c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 909, 33), list_17385, generate_umath_c_17388)
    # Adding element type (line 909)
    # Getting the type of 'generate_ufunc_api' (line 912)
    generate_ufunc_api_17389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 33), 'generate_ufunc_api', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 909, 33), list_17385, generate_ufunc_api_17389)
    
    # Applying the binary operator '+' (line 908)
    result_add_17390 = python_operator(stypy.reporting.localization.Localization(__file__, 908, 33), '+', umath_src_17384, list_17385)
    
    keyword_17391 = result_add_17390
    # Getting the type of 'deps' (line 913)
    deps_17392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 33), 'deps', False)
    # Getting the type of 'umath_deps' (line 913)
    umath_deps_17393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 40), 'umath_deps', False)
    # Applying the binary operator '+' (line 913)
    result_add_17394 = python_operator(stypy.reporting.localization.Localization(__file__, 913, 33), '+', deps_17392, umath_deps_17393)
    
    keyword_17395 = result_add_17394
    
    # Obtaining an instance of the builtin type 'list' (line 914)
    list_17396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 914, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 914)
    # Adding element type (line 914)
    str_17397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 914, 36), 'str', 'npymath')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 914, 35), list_17396, str_17397)
    
    keyword_17398 = list_17396
    kwargs_17399 = {'libraries': keyword_17398, 'sources': keyword_17391, 'depends': keyword_17395}
    # Getting the type of 'config' (line 907)
    config_17381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 907)
    add_extension_17382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 907, 4), config_17381, 'add_extension')
    # Calling add_extension(args, kwargs) (line 907)
    add_extension_call_result_17400 = invoke(stypy.reporting.localization.Localization(__file__, 907, 4), add_extension_17382, *[str_17383], **kwargs_17399)
    
    
    # Call to add_extension(...): (line 921)
    # Processing the call arguments (line 921)
    str_17403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, 25), 'str', 'umath_tests')
    # Processing the call keyword arguments (line 921)
    
    # Obtaining an instance of the builtin type 'list' (line 922)
    list_17404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 922)
    # Adding element type (line 922)
    
    # Call to join(...): (line 922)
    # Processing the call arguments (line 922)
    str_17406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 34), 'str', 'src')
    str_17407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 41), 'str', 'umath')
    str_17408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 50), 'str', 'umath_tests.c.src')
    # Processing the call keyword arguments (line 922)
    kwargs_17409 = {}
    # Getting the type of 'join' (line 922)
    join_17405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 29), 'join', False)
    # Calling join(args, kwargs) (line 922)
    join_call_result_17410 = invoke(stypy.reporting.localization.Localization(__file__, 922, 29), join_17405, *[str_17406, str_17407, str_17408], **kwargs_17409)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 922, 28), list_17404, join_call_result_17410)
    
    keyword_17411 = list_17404
    kwargs_17412 = {'sources': keyword_17411}
    # Getting the type of 'config' (line 921)
    config_17401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 921)
    add_extension_17402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 4), config_17401, 'add_extension')
    # Calling add_extension(args, kwargs) (line 921)
    add_extension_call_result_17413 = invoke(stypy.reporting.localization.Localization(__file__, 921, 4), add_extension_17402, *[str_17403], **kwargs_17412)
    
    
    # Call to add_extension(...): (line 928)
    # Processing the call arguments (line 928)
    str_17416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 928, 25), 'str', 'test_rational')
    # Processing the call keyword arguments (line 928)
    
    # Obtaining an instance of the builtin type 'list' (line 929)
    list_17417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 929, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 929)
    # Adding element type (line 929)
    
    # Call to join(...): (line 929)
    # Processing the call arguments (line 929)
    str_17419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 929, 34), 'str', 'src')
    str_17420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 929, 41), 'str', 'umath')
    str_17421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 929, 50), 'str', 'test_rational.c.src')
    # Processing the call keyword arguments (line 929)
    kwargs_17422 = {}
    # Getting the type of 'join' (line 929)
    join_17418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 29), 'join', False)
    # Calling join(args, kwargs) (line 929)
    join_call_result_17423 = invoke(stypy.reporting.localization.Localization(__file__, 929, 29), join_17418, *[str_17419, str_17420, str_17421], **kwargs_17422)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 929, 28), list_17417, join_call_result_17423)
    
    keyword_17424 = list_17417
    kwargs_17425 = {'sources': keyword_17424}
    # Getting the type of 'config' (line 928)
    config_17414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 928)
    add_extension_17415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 4), config_17414, 'add_extension')
    # Calling add_extension(args, kwargs) (line 928)
    add_extension_call_result_17426 = invoke(stypy.reporting.localization.Localization(__file__, 928, 4), add_extension_17415, *[str_17416], **kwargs_17425)
    
    
    # Call to add_extension(...): (line 935)
    # Processing the call arguments (line 935)
    str_17429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 25), 'str', 'struct_ufunc_test')
    # Processing the call keyword arguments (line 935)
    
    # Obtaining an instance of the builtin type 'list' (line 936)
    list_17430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 936)
    # Adding element type (line 936)
    
    # Call to join(...): (line 936)
    # Processing the call arguments (line 936)
    str_17432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 34), 'str', 'src')
    str_17433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 41), 'str', 'umath')
    str_17434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 50), 'str', 'struct_ufunc_test.c.src')
    # Processing the call keyword arguments (line 936)
    kwargs_17435 = {}
    # Getting the type of 'join' (line 936)
    join_17431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 29), 'join', False)
    # Calling join(args, kwargs) (line 936)
    join_call_result_17436 = invoke(stypy.reporting.localization.Localization(__file__, 936, 29), join_17431, *[str_17432, str_17433, str_17434], **kwargs_17435)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 936, 28), list_17430, join_call_result_17436)
    
    keyword_17437 = list_17430
    kwargs_17438 = {'sources': keyword_17437}
    # Getting the type of 'config' (line 935)
    config_17427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 935)
    add_extension_17428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 935, 4), config_17427, 'add_extension')
    # Calling add_extension(args, kwargs) (line 935)
    add_extension_call_result_17439 = invoke(stypy.reporting.localization.Localization(__file__, 935, 4), add_extension_17428, *[str_17429], **kwargs_17438)
    
    
    # Call to add_extension(...): (line 942)
    # Processing the call arguments (line 942)
    str_17442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 942, 25), 'str', 'multiarray_tests')
    # Processing the call keyword arguments (line 942)
    
    # Obtaining an instance of the builtin type 'list' (line 943)
    list_17443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 943)
    # Adding element type (line 943)
    
    # Call to join(...): (line 943)
    # Processing the call arguments (line 943)
    str_17445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, 34), 'str', 'src')
    str_17446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, 41), 'str', 'multiarray')
    str_17447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, 55), 'str', 'multiarray_tests.c.src')
    # Processing the call keyword arguments (line 943)
    kwargs_17448 = {}
    # Getting the type of 'join' (line 943)
    join_17444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 29), 'join', False)
    # Calling join(args, kwargs) (line 943)
    join_call_result_17449 = invoke(stypy.reporting.localization.Localization(__file__, 943, 29), join_17444, *[str_17445, str_17446, str_17447], **kwargs_17448)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 943, 28), list_17443, join_call_result_17449)
    # Adding element type (line 943)
    
    # Call to join(...): (line 944)
    # Processing the call arguments (line 944)
    str_17451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 34), 'str', 'src')
    str_17452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 41), 'str', 'private')
    str_17453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 52), 'str', 'mem_overlap.c')
    # Processing the call keyword arguments (line 944)
    kwargs_17454 = {}
    # Getting the type of 'join' (line 944)
    join_17450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 29), 'join', False)
    # Calling join(args, kwargs) (line 944)
    join_call_result_17455 = invoke(stypy.reporting.localization.Localization(__file__, 944, 29), join_17450, *[str_17451, str_17452, str_17453], **kwargs_17454)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 943, 28), list_17443, join_call_result_17455)
    
    keyword_17456 = list_17443
    
    # Obtaining an instance of the builtin type 'list' (line 945)
    list_17457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 945, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 945)
    # Adding element type (line 945)
    
    # Call to join(...): (line 945)
    # Processing the call arguments (line 945)
    str_17459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 945, 34), 'str', 'src')
    str_17460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 945, 41), 'str', 'private')
    str_17461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 945, 52), 'str', 'mem_overlap.h')
    # Processing the call keyword arguments (line 945)
    kwargs_17462 = {}
    # Getting the type of 'join' (line 945)
    join_17458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 29), 'join', False)
    # Calling join(args, kwargs) (line 945)
    join_call_result_17463 = invoke(stypy.reporting.localization.Localization(__file__, 945, 29), join_17458, *[str_17459, str_17460, str_17461], **kwargs_17462)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 945, 28), list_17457, join_call_result_17463)
    # Adding element type (line 945)
    
    # Call to join(...): (line 946)
    # Processing the call arguments (line 946)
    str_17465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 34), 'str', 'src')
    str_17466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 41), 'str', 'private')
    str_17467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 52), 'str', 'npy_extint128.h')
    # Processing the call keyword arguments (line 946)
    kwargs_17468 = {}
    # Getting the type of 'join' (line 946)
    join_17464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 29), 'join', False)
    # Calling join(args, kwargs) (line 946)
    join_call_result_17469 = invoke(stypy.reporting.localization.Localization(__file__, 946, 29), join_17464, *[str_17465, str_17466, str_17467], **kwargs_17468)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 945, 28), list_17457, join_call_result_17469)
    
    keyword_17470 = list_17457
    kwargs_17471 = {'sources': keyword_17456, 'depends': keyword_17470}
    # Getting the type of 'config' (line 942)
    config_17440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 942)
    add_extension_17441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 942, 4), config_17440, 'add_extension')
    # Calling add_extension(args, kwargs) (line 942)
    add_extension_call_result_17472 = invoke(stypy.reporting.localization.Localization(__file__, 942, 4), add_extension_17441, *[str_17442], **kwargs_17471)
    
    
    # Call to add_extension(...): (line 952)
    # Processing the call arguments (line 952)
    str_17475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 952, 25), 'str', 'operand_flag_tests')
    # Processing the call keyword arguments (line 952)
    
    # Obtaining an instance of the builtin type 'list' (line 953)
    list_17476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 953, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 953)
    # Adding element type (line 953)
    
    # Call to join(...): (line 953)
    # Processing the call arguments (line 953)
    str_17478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 953, 34), 'str', 'src')
    str_17479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 953, 41), 'str', 'umath')
    str_17480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 953, 50), 'str', 'operand_flag_tests.c.src')
    # Processing the call keyword arguments (line 953)
    kwargs_17481 = {}
    # Getting the type of 'join' (line 953)
    join_17477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 29), 'join', False)
    # Calling join(args, kwargs) (line 953)
    join_call_result_17482 = invoke(stypy.reporting.localization.Localization(__file__, 953, 29), join_17477, *[str_17478, str_17479, str_17480], **kwargs_17481)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 953, 28), list_17476, join_call_result_17482)
    
    keyword_17483 = list_17476
    kwargs_17484 = {'sources': keyword_17483}
    # Getting the type of 'config' (line 952)
    config_17473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 952)
    add_extension_17474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 952, 4), config_17473, 'add_extension')
    # Calling add_extension(args, kwargs) (line 952)
    add_extension_call_result_17485 = invoke(stypy.reporting.localization.Localization(__file__, 952, 4), add_extension_17474, *[str_17475], **kwargs_17484)
    
    
    # Call to add_data_dir(...): (line 955)
    # Processing the call arguments (line 955)
    str_17488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 955, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 955)
    kwargs_17489 = {}
    # Getting the type of 'config' (line 955)
    config_17486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 955)
    add_data_dir_17487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 4), config_17486, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 955)
    add_data_dir_call_result_17490 = invoke(stypy.reporting.localization.Localization(__file__, 955, 4), add_data_dir_17487, *[str_17488], **kwargs_17489)
    
    
    # Call to add_data_dir(...): (line 956)
    # Processing the call arguments (line 956)
    str_17493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 956, 24), 'str', 'tests/data')
    # Processing the call keyword arguments (line 956)
    kwargs_17494 = {}
    # Getting the type of 'config' (line 956)
    config_17491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 956)
    add_data_dir_17492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 956, 4), config_17491, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 956)
    add_data_dir_call_result_17495 = invoke(stypy.reporting.localization.Localization(__file__, 956, 4), add_data_dir_17492, *[str_17493], **kwargs_17494)
    
    
    # Call to make_svn_version_py(...): (line 958)
    # Processing the call keyword arguments (line 958)
    kwargs_17498 = {}
    # Getting the type of 'config' (line 958)
    config_17496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 4), 'config', False)
    # Obtaining the member 'make_svn_version_py' of a type (line 958)
    make_svn_version_py_17497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 958, 4), config_17496, 'make_svn_version_py')
    # Calling make_svn_version_py(args, kwargs) (line 958)
    make_svn_version_py_call_result_17499 = invoke(stypy.reporting.localization.Localization(__file__, 958, 4), make_svn_version_py_17497, *[], **kwargs_17498)
    
    # Getting the type of 'config' (line 960)
    config_17500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 960)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 960, 4), 'stypy_return_type', config_17500)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 372)
    stypy_return_type_17501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17501)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_17501

# Assigning a type to the variable 'configuration' (line 372)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 963, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 963)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
    import_17502 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 963, 4), 'numpy.distutils.core')

    if (type(import_17502) is not StypyTypeError):

        if (import_17502 != 'pyd_module'):
            __import__(import_17502)
            sys_modules_17503 = sys.modules[import_17502]
            import_from_module(stypy.reporting.localization.Localization(__file__, 963, 4), 'numpy.distutils.core', sys_modules_17503.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 963, 4), __file__, sys_modules_17503, sys_modules_17503.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 963, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 963)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 4), 'numpy.distutils.core', import_17502)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')
    
    
    # Call to setup(...): (line 964)
    # Processing the call keyword arguments (line 964)
    # Getting the type of 'configuration' (line 964)
    configuration_17505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 24), 'configuration', False)
    keyword_17506 = configuration_17505
    kwargs_17507 = {'configuration': keyword_17506}
    # Getting the type of 'setup' (line 964)
    setup_17504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 964)
    setup_call_result_17508 = invoke(stypy.reporting.localization.Localization(__file__, 964, 4), setup_17504, *[], **kwargs_17507)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
