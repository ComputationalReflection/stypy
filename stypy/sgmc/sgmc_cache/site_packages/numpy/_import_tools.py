
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import os
4: import sys
5: import warnings
6: 
7: __all__ = ['PackageLoader']
8: 
9: class PackageLoader(object):
10:     def __init__(self, verbose=False, infunc=False):
11:         ''' Manages loading packages.
12:         '''
13: 
14:         if infunc:
15:             _level = 2
16:         else:
17:             _level = 1
18:         self.parent_frame = frame = sys._getframe(_level)
19:         self.parent_name = eval('__name__', frame.f_globals, frame.f_locals)
20:         parent_path = eval('__path__', frame.f_globals, frame.f_locals)
21:         if isinstance(parent_path, str):
22:             parent_path = [parent_path]
23:         self.parent_path = parent_path
24:         if '__all__' not in frame.f_locals:
25:             exec('__all__ = []', frame.f_globals, frame.f_locals)
26:         self.parent_export_names = eval('__all__', frame.f_globals, frame.f_locals)
27: 
28:         self.info_modules = {}
29:         self.imported_packages = []
30:         self.verbose = None
31: 
32:     def _get_info_files(self, package_dir, parent_path, parent_package=None):
33:         ''' Return list of (package name,info.py file) from parent_path subdirectories.
34:         '''
35:         from glob import glob
36:         files = glob(os.path.join(parent_path, package_dir, 'info.py'))
37:         for info_file in glob(os.path.join(parent_path, package_dir, 'info.pyc')):
38:             if info_file[:-1] not in files:
39:                 files.append(info_file)
40:         info_files = []
41:         for info_file in files:
42:             package_name = os.path.dirname(info_file[len(parent_path)+1:])\
43:                            .replace(os.sep, '.')
44:             if parent_package:
45:                 package_name = parent_package + '.' + package_name
46:             info_files.append((package_name, info_file))
47:             info_files.extend(self._get_info_files('*',
48:                                                    os.path.dirname(info_file),
49:                                                    package_name))
50:         return info_files
51: 
52:     def _init_info_modules(self, packages=None):
53:         '''Initialize info_modules = {<package_name>: <package info.py module>}.
54:         '''
55:         import imp
56:         info_files = []
57:         info_modules = self.info_modules
58: 
59:         if packages is None:
60:             for path in self.parent_path:
61:                 info_files.extend(self._get_info_files('*', path))
62:         else:
63:             for package_name in packages:
64:                 package_dir = os.path.join(*package_name.split('.'))
65:                 for path in self.parent_path:
66:                     names_files = self._get_info_files(package_dir, path)
67:                     if names_files:
68:                         info_files.extend(names_files)
69:                         break
70:                 else:
71:                     try:
72:                         exec('import %s.info as info' % (package_name))
73:                         info_modules[package_name] = info
74:                     except ImportError as msg:
75:                         self.warn('No scipy-style subpackage %r found in %s. '\
76:                                   'Ignoring: %s'\
77:                                   % (package_name, ':'.join(self.parent_path), msg))
78: 
79:         for package_name, info_file in info_files:
80:             if package_name in info_modules:
81:                 continue
82:             fullname = self.parent_name +'.'+ package_name
83:             if info_file[-1]=='c':
84:                 filedescriptor = ('.pyc', 'rb', 2)
85:             else:
86:                 filedescriptor = ('.py', 'U', 1)
87: 
88:             try:
89:                 info_module = imp.load_module(fullname+'.info',
90:                                               open(info_file, filedescriptor[1]),
91:                                               info_file,
92:                                               filedescriptor)
93:             except Exception as msg:
94:                 self.error(msg)
95:                 info_module = None
96: 
97:             if info_module is None or getattr(info_module, 'ignore', False):
98:                 info_modules.pop(package_name, None)
99:             else:
100:                 self._init_info_modules(getattr(info_module, 'depends', []))
101:                 info_modules[package_name] = info_module
102: 
103:         return
104: 
105:     def _get_sorted_names(self):
106:         ''' Return package names sorted in the order as they should be
107:         imported due to dependence relations between packages.
108:         '''
109: 
110:         depend_dict = {}
111:         for name, info_module in self.info_modules.items():
112:             depend_dict[name] = getattr(info_module, 'depends', [])
113:         package_names = []
114: 
115:         for name in list(depend_dict.keys()):
116:             if not depend_dict[name]:
117:                 package_names.append(name)
118:                 del depend_dict[name]
119: 
120:         while depend_dict:
121:             for name, lst in list(depend_dict.items()):
122:                 new_lst = [n for n in lst if n in depend_dict]
123:                 if not new_lst:
124:                     package_names.append(name)
125:                     del depend_dict[name]
126:                 else:
127:                     depend_dict[name] = new_lst
128: 
129:         return package_names
130: 
131:     def __call__(self,*packages, **options):
132:         '''Load one or more packages into parent package top-level namespace.
133: 
134:        This function is intended to shorten the need to import many
135:        subpackages, say of scipy, constantly with statements such as
136: 
137:          import scipy.linalg, scipy.fftpack, scipy.etc...
138: 
139:        Instead, you can say:
140: 
141:          import scipy
142:          scipy.pkgload('linalg','fftpack',...)
143: 
144:        or
145: 
146:          scipy.pkgload()
147: 
148:        to load all of them in one call.
149: 
150:        If a name which doesn't exist in scipy's namespace is
151:        given, a warning is shown.
152: 
153:        Parameters
154:        ----------
155:         *packages : arg-tuple
156:              the names (one or more strings) of all the modules one
157:              wishes to load into the top-level namespace.
158:         verbose= : integer
159:              verbosity level [default: -1].
160:              verbose=-1 will suspend also warnings.
161:         force= : bool
162:              when True, force reloading loaded packages [default: False].
163:         postpone= : bool
164:              when True, don't load packages [default: False]
165: 
166:         '''
167:         # 2014-10-29, 1.10
168:         warnings.warn('pkgload and PackageLoader are obsolete '
169:                 'and will be removed in a future version of numpy',
170:                 DeprecationWarning)
171:         frame = self.parent_frame
172:         self.info_modules = {}
173:         if options.get('force', False):
174:             self.imported_packages = []
175:         self.verbose = verbose = options.get('verbose', -1)
176:         postpone = options.get('postpone', None)
177:         self._init_info_modules(packages or None)
178: 
179:         self.log('Imports to %r namespace\n----------------------------'\
180:                  % self.parent_name)
181: 
182:         for package_name in self._get_sorted_names():
183:             if package_name in self.imported_packages:
184:                 continue
185:             info_module = self.info_modules[package_name]
186:             global_symbols = getattr(info_module, 'global_symbols', [])
187:             postpone_import = getattr(info_module, 'postpone_import', False)
188:             if (postpone and not global_symbols) \
189:                    or (postpone_import and postpone is not None):
190:                 continue
191: 
192:             old_object = frame.f_locals.get(package_name, None)
193: 
194:             cmdstr = 'import '+package_name
195:             if self._execcmd(cmdstr):
196:                 continue
197:             self.imported_packages.append(package_name)
198: 
199:             if verbose!=-1:
200:                 new_object = frame.f_locals.get(package_name)
201:                 if old_object is not None and old_object is not new_object:
202:                     self.warn('Overwriting %s=%s (was %s)' \
203:                               % (package_name, self._obj2repr(new_object),
204:                                  self._obj2repr(old_object)))
205: 
206:             if '.' not in package_name:
207:                 self.parent_export_names.append(package_name)
208: 
209:             for symbol in global_symbols:
210:                 if symbol=='*':
211:                     symbols = eval('getattr(%s,"__all__",None)'\
212:                                    % (package_name),
213:                                    frame.f_globals, frame.f_locals)
214:                     if symbols is None:
215:                         symbols = eval('dir(%s)' % (package_name),
216:                                        frame.f_globals, frame.f_locals)
217:                         symbols = [s for s in symbols if not s.startswith('_')]
218:                 else:
219:                     symbols = [symbol]
220: 
221:                 if verbose!=-1:
222:                     old_objects = {}
223:                     for s in symbols:
224:                         if s in frame.f_locals:
225:                             old_objects[s] = frame.f_locals[s]
226: 
227:                 cmdstr = 'from '+package_name+' import '+symbol
228:                 if self._execcmd(cmdstr):
229:                     continue
230: 
231:                 if verbose!=-1:
232:                     for s, old_object in old_objects.items():
233:                         new_object = frame.f_locals[s]
234:                         if new_object is not old_object:
235:                             self.warn('Overwriting %s=%s (was %s)' \
236:                                       % (s, self._obj2repr(new_object),
237:                                          self._obj2repr(old_object)))
238: 
239:                 if symbol=='*':
240:                     self.parent_export_names.extend(symbols)
241:                 else:
242:                     self.parent_export_names.append(symbol)
243: 
244:         return
245: 
246:     def _execcmd(self, cmdstr):
247:         ''' Execute command in parent_frame.'''
248:         frame = self.parent_frame
249:         try:
250:             exec (cmdstr, frame.f_globals, frame.f_locals)
251:         except Exception as msg:
252:             self.error('%s -> failed: %s' % (cmdstr, msg))
253:             return True
254:         else:
255:             self.log('%s -> success' % (cmdstr))
256:         return
257: 
258:     def _obj2repr(self, obj):
259:         ''' Return repr(obj) with'''
260:         module = getattr(obj, '__module__', None)
261:         file = getattr(obj, '__file__', None)
262:         if module is not None:
263:             return repr(obj) + ' from ' + module
264:         if file is not None:
265:             return repr(obj) + ' from ' + file
266:         return repr(obj)
267: 
268:     def log(self, mess):
269:         if self.verbose>1:
270:             print(str(mess), file=sys.stderr)
271:     def warn(self, mess):
272:         if self.verbose>=0:
273:             print(str(mess), file=sys.stderr)
274:     def error(self, mess):
275:         if self.verbose!=-1:
276:             print(str(mess), file=sys.stderr)
277: 
278:     def _get_doc_title(self, info_module):
279:         ''' Get the title from a package info.py file.
280:         '''
281:         title = getattr(info_module, '__doc_title__', None)
282:         if title is not None:
283:             return title
284:         title = getattr(info_module, '__doc__', None)
285:         if title is not None:
286:             title = title.lstrip().split('\n', 1)[0]
287:             return title
288:         return '* Not Available *'
289: 
290:     def _format_titles(self,titles,colsep='---'):
291:         display_window_width = 70 # How to determine the correct value in runtime??
292:         lengths = [len(name)-name.find('.')-1 for (name, title) in titles]+[0]
293:         max_length = max(lengths)
294:         lines = []
295:         for (name, title) in titles:
296:             name = name[name.find('.')+1:]
297:             w = max_length - len(name)
298:             words = title.split()
299:             line = '%s%s %s' % (name, w*' ', colsep)
300:             tab = len(line) * ' '
301:             while words:
302:                 word = words.pop(0)
303:                 if len(line)+len(word)>display_window_width:
304:                     lines.append(line)
305:                     line = tab
306:                 line += ' ' + word
307:             else:
308:                 lines.append(line)
309:         return '\n'.join(lines)
310: 
311:     def get_pkgdocs(self):
312:         ''' Return documentation summary of subpackages.
313:         '''
314:         import sys
315:         self.info_modules = {}
316:         self._init_info_modules(None)
317: 
318:         titles = []
319:         symbols = []
320:         for package_name, info_module in self.info_modules.items():
321:             global_symbols = getattr(info_module, 'global_symbols', [])
322:             fullname = self.parent_name +'.'+ package_name
323:             note = ''
324:             if fullname not in sys.modules:
325:                 note = ' [*]'
326:             titles.append((fullname, self._get_doc_title(info_module) + note))
327:             if global_symbols:
328:                 symbols.append((package_name, ', '.join(global_symbols)))
329: 
330:         retstr = self._format_titles(titles) +\
331:                '\n  [*] - using a package requires explicit import (see pkgload)'
332: 
333: 
334:         if symbols:
335:             retstr += '''\n\nGlobal symbols from subpackages'''\
336:                       '''\n-------------------------------\n''' +\
337:                       self._format_titles(symbols, '-->')
338: 
339:         return retstr
340: 
341: class PackageLoaderDebug(PackageLoader):
342:     def _execcmd(self, cmdstr):
343:         ''' Execute command in parent_frame.'''
344:         frame = self.parent_frame
345:         print('Executing', repr(cmdstr), '...', end=' ')
346:         sys.stdout.flush()
347:         exec (cmdstr, frame.f_globals, frame.f_locals)
348:         print('ok')
349:         sys.stdout.flush()
350:         return
351: 
352: if int(os.environ.get('NUMPY_IMPORT_DEBUG', '0')):
353:     PackageLoader = PackageLoaderDebug
354: 

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

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import warnings' statement (line 5)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'warnings', warnings, module_type_store)


# Assigning a List to a Name (line 7):
__all__ = ['PackageLoader']
module_type_store.set_exportable_members(['PackageLoader'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_24418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_24419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'PackageLoader')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_24418, str_24419)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_24418)
# Declaration of the 'PackageLoader' class

class PackageLoader(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 10)
        False_24420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 31), 'False')
        # Getting the type of 'False' (line 10)
        False_24421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 45), 'False')
        defaults = [False_24420, False_24421]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 10, 4, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PackageLoader.__init__', ['verbose', 'infunc'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['verbose', 'infunc'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_24422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, (-1)), 'str', ' Manages loading packages.\n        ')
        
        # Getting the type of 'infunc' (line 14)
        infunc_24423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'infunc')
        # Testing the type of an if condition (line 14)
        if_condition_24424 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 8), infunc_24423)
        # Assigning a type to the variable 'if_condition_24424' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'if_condition_24424', if_condition_24424)
        # SSA begins for if statement (line 14)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 15):
        int_24425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 21), 'int')
        # Assigning a type to the variable '_level' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), '_level', int_24425)
        # SSA branch for the else part of an if statement (line 14)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 17):
        int_24426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 21), 'int')
        # Assigning a type to the variable '_level' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), '_level', int_24426)
        # SSA join for if statement (line 14)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Multiple assignment of 2 elements.
        
        # Call to _getframe(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of '_level' (line 18)
        _level_24429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 50), '_level', False)
        # Processing the call keyword arguments (line 18)
        kwargs_24430 = {}
        # Getting the type of 'sys' (line 18)
        sys_24427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 36), 'sys', False)
        # Obtaining the member '_getframe' of a type (line 18)
        _getframe_24428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 36), sys_24427, '_getframe')
        # Calling _getframe(args, kwargs) (line 18)
        _getframe_call_result_24431 = invoke(stypy.reporting.localization.Localization(__file__, 18, 36), _getframe_24428, *[_level_24429], **kwargs_24430)
        
        # Assigning a type to the variable 'frame' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 28), 'frame', _getframe_call_result_24431)
        # Getting the type of 'frame' (line 18)
        frame_24432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 28), 'frame')
        # Getting the type of 'self' (line 18)
        self_24433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'self')
        # Setting the type of the member 'parent_frame' of a type (line 18)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), self_24433, 'parent_frame', frame_24432)
        
        # Assigning a Call to a Attribute (line 19):
        
        # Call to eval(...): (line 19)
        # Processing the call arguments (line 19)
        str_24435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 32), 'str', '__name__')
        # Getting the type of 'frame' (line 19)
        frame_24436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 44), 'frame', False)
        # Obtaining the member 'f_globals' of a type (line 19)
        f_globals_24437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 44), frame_24436, 'f_globals')
        # Getting the type of 'frame' (line 19)
        frame_24438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 61), 'frame', False)
        # Obtaining the member 'f_locals' of a type (line 19)
        f_locals_24439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 61), frame_24438, 'f_locals')
        # Processing the call keyword arguments (line 19)
        kwargs_24440 = {}
        # Getting the type of 'eval' (line 19)
        eval_24434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 27), 'eval', False)
        # Calling eval(args, kwargs) (line 19)
        eval_call_result_24441 = invoke(stypy.reporting.localization.Localization(__file__, 19, 27), eval_24434, *[str_24435, f_globals_24437, f_locals_24439], **kwargs_24440)
        
        # Getting the type of 'self' (line 19)
        self_24442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'self')
        # Setting the type of the member 'parent_name' of a type (line 19)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), self_24442, 'parent_name', eval_call_result_24441)
        
        # Assigning a Call to a Name (line 20):
        
        # Call to eval(...): (line 20)
        # Processing the call arguments (line 20)
        str_24444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 27), 'str', '__path__')
        # Getting the type of 'frame' (line 20)
        frame_24445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 39), 'frame', False)
        # Obtaining the member 'f_globals' of a type (line 20)
        f_globals_24446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 39), frame_24445, 'f_globals')
        # Getting the type of 'frame' (line 20)
        frame_24447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 56), 'frame', False)
        # Obtaining the member 'f_locals' of a type (line 20)
        f_locals_24448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 56), frame_24447, 'f_locals')
        # Processing the call keyword arguments (line 20)
        kwargs_24449 = {}
        # Getting the type of 'eval' (line 20)
        eval_24443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 22), 'eval', False)
        # Calling eval(args, kwargs) (line 20)
        eval_call_result_24450 = invoke(stypy.reporting.localization.Localization(__file__, 20, 22), eval_24443, *[str_24444, f_globals_24446, f_locals_24448], **kwargs_24449)
        
        # Assigning a type to the variable 'parent_path' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'parent_path', eval_call_result_24450)
        
        # Type idiom detected: calculating its left and rigth part (line 21)
        # Getting the type of 'str' (line 21)
        str_24451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 35), 'str')
        # Getting the type of 'parent_path' (line 21)
        parent_path_24452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'parent_path')
        
        (may_be_24453, more_types_in_union_24454) = may_be_subtype(str_24451, parent_path_24452)

        if may_be_24453:

            if more_types_in_union_24454:
                # Runtime conditional SSA (line 21)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'parent_path' (line 21)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'parent_path', remove_not_subtype_from_union(parent_path_24452, str))
            
            # Assigning a List to a Name (line 22):
            
            # Obtaining an instance of the builtin type 'list' (line 22)
            list_24455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 26), 'list')
            # Adding type elements to the builtin type 'list' instance (line 22)
            # Adding element type (line 22)
            # Getting the type of 'parent_path' (line 22)
            parent_path_24456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 27), 'parent_path')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 26), list_24455, parent_path_24456)
            
            # Assigning a type to the variable 'parent_path' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'parent_path', list_24455)

            if more_types_in_union_24454:
                # SSA join for if statement (line 21)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 23):
        # Getting the type of 'parent_path' (line 23)
        parent_path_24457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 27), 'parent_path')
        # Getting the type of 'self' (line 23)
        self_24458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self')
        # Setting the type of the member 'parent_path' of a type (line 23)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_24458, 'parent_path', parent_path_24457)
        
        
        str_24459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 11), 'str', '__all__')
        # Getting the type of 'frame' (line 24)
        frame_24460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 28), 'frame')
        # Obtaining the member 'f_locals' of a type (line 24)
        f_locals_24461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 28), frame_24460, 'f_locals')
        # Applying the binary operator 'notin' (line 24)
        result_contains_24462 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 11), 'notin', str_24459, f_locals_24461)
        
        # Testing the type of an if condition (line 24)
        if_condition_24463 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 8), result_contains_24462)
        # Assigning a type to the variable 'if_condition_24463' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'if_condition_24463', if_condition_24463)
        # SSA begins for if statement (line 24)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Dynamic code evaluation using an exec statement
        str_24464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 17), 'str', '__all__ = []')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 25, 12), str_24464, 'exec parameter', 'StringType', 'FileType', 'CodeType')
        enable_usage_of_dynamic_types_warning(stypy.reporting.localization.Localization(__file__, 25, 12))
        # SSA join for if statement (line 24)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 26):
        
        # Call to eval(...): (line 26)
        # Processing the call arguments (line 26)
        str_24466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 40), 'str', '__all__')
        # Getting the type of 'frame' (line 26)
        frame_24467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 51), 'frame', False)
        # Obtaining the member 'f_globals' of a type (line 26)
        f_globals_24468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 51), frame_24467, 'f_globals')
        # Getting the type of 'frame' (line 26)
        frame_24469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 68), 'frame', False)
        # Obtaining the member 'f_locals' of a type (line 26)
        f_locals_24470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 68), frame_24469, 'f_locals')
        # Processing the call keyword arguments (line 26)
        kwargs_24471 = {}
        # Getting the type of 'eval' (line 26)
        eval_24465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 35), 'eval', False)
        # Calling eval(args, kwargs) (line 26)
        eval_call_result_24472 = invoke(stypy.reporting.localization.Localization(__file__, 26, 35), eval_24465, *[str_24466, f_globals_24468, f_locals_24470], **kwargs_24471)
        
        # Getting the type of 'self' (line 26)
        self_24473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self')
        # Setting the type of the member 'parent_export_names' of a type (line 26)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_24473, 'parent_export_names', eval_call_result_24472)
        
        # Assigning a Dict to a Attribute (line 28):
        
        # Obtaining an instance of the builtin type 'dict' (line 28)
        dict_24474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 28)
        
        # Getting the type of 'self' (line 28)
        self_24475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Setting the type of the member 'info_modules' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_24475, 'info_modules', dict_24474)
        
        # Assigning a List to a Attribute (line 29):
        
        # Obtaining an instance of the builtin type 'list' (line 29)
        list_24476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 29)
        
        # Getting the type of 'self' (line 29)
        self_24477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Setting the type of the member 'imported_packages' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_24477, 'imported_packages', list_24476)
        
        # Assigning a Name to a Attribute (line 30):
        # Getting the type of 'None' (line 30)
        None_24478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 23), 'None')
        # Getting the type of 'self' (line 30)
        self_24479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member 'verbose' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_24479, 'verbose', None_24478)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _get_info_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 32)
        None_24480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 71), 'None')
        defaults = [None_24480]
        # Create a new context for function '_get_info_files'
        module_type_store = module_type_store.open_function_context('_get_info_files', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PackageLoader._get_info_files.__dict__.__setitem__('stypy_localization', localization)
        PackageLoader._get_info_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PackageLoader._get_info_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        PackageLoader._get_info_files.__dict__.__setitem__('stypy_function_name', 'PackageLoader._get_info_files')
        PackageLoader._get_info_files.__dict__.__setitem__('stypy_param_names_list', ['package_dir', 'parent_path', 'parent_package'])
        PackageLoader._get_info_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        PackageLoader._get_info_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PackageLoader._get_info_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        PackageLoader._get_info_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        PackageLoader._get_info_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PackageLoader._get_info_files.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PackageLoader._get_info_files', ['package_dir', 'parent_path', 'parent_package'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_info_files', localization, ['package_dir', 'parent_path', 'parent_package'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_info_files(...)' code ##################

        str_24481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, (-1)), 'str', ' Return list of (package name,info.py file) from parent_path subdirectories.\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 8))
        
        # 'from glob import glob' statement (line 35)
        from glob import glob

        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 8), 'glob', None, module_type_store, ['glob'], [glob])
        
        
        # Assigning a Call to a Name (line 36):
        
        # Call to glob(...): (line 36)
        # Processing the call arguments (line 36)
        
        # Call to join(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'parent_path' (line 36)
        parent_path_24486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 34), 'parent_path', False)
        # Getting the type of 'package_dir' (line 36)
        package_dir_24487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 47), 'package_dir', False)
        str_24488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 60), 'str', 'info.py')
        # Processing the call keyword arguments (line 36)
        kwargs_24489 = {}
        # Getting the type of 'os' (line 36)
        os_24483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 21), 'os', False)
        # Obtaining the member 'path' of a type (line 36)
        path_24484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 21), os_24483, 'path')
        # Obtaining the member 'join' of a type (line 36)
        join_24485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 21), path_24484, 'join')
        # Calling join(args, kwargs) (line 36)
        join_call_result_24490 = invoke(stypy.reporting.localization.Localization(__file__, 36, 21), join_24485, *[parent_path_24486, package_dir_24487, str_24488], **kwargs_24489)
        
        # Processing the call keyword arguments (line 36)
        kwargs_24491 = {}
        # Getting the type of 'glob' (line 36)
        glob_24482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'glob', False)
        # Calling glob(args, kwargs) (line 36)
        glob_call_result_24492 = invoke(stypy.reporting.localization.Localization(__file__, 36, 16), glob_24482, *[join_call_result_24490], **kwargs_24491)
        
        # Assigning a type to the variable 'files' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'files', glob_call_result_24492)
        
        
        # Call to glob(...): (line 37)
        # Processing the call arguments (line 37)
        
        # Call to join(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'parent_path' (line 37)
        parent_path_24497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 43), 'parent_path', False)
        # Getting the type of 'package_dir' (line 37)
        package_dir_24498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 56), 'package_dir', False)
        str_24499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 69), 'str', 'info.pyc')
        # Processing the call keyword arguments (line 37)
        kwargs_24500 = {}
        # Getting the type of 'os' (line 37)
        os_24494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 37)
        path_24495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 30), os_24494, 'path')
        # Obtaining the member 'join' of a type (line 37)
        join_24496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 30), path_24495, 'join')
        # Calling join(args, kwargs) (line 37)
        join_call_result_24501 = invoke(stypy.reporting.localization.Localization(__file__, 37, 30), join_24496, *[parent_path_24497, package_dir_24498, str_24499], **kwargs_24500)
        
        # Processing the call keyword arguments (line 37)
        kwargs_24502 = {}
        # Getting the type of 'glob' (line 37)
        glob_24493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 25), 'glob', False)
        # Calling glob(args, kwargs) (line 37)
        glob_call_result_24503 = invoke(stypy.reporting.localization.Localization(__file__, 37, 25), glob_24493, *[join_call_result_24501], **kwargs_24502)
        
        # Testing the type of a for loop iterable (line 37)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 37, 8), glob_call_result_24503)
        # Getting the type of the for loop variable (line 37)
        for_loop_var_24504 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 37, 8), glob_call_result_24503)
        # Assigning a type to the variable 'info_file' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'info_file', for_loop_var_24504)
        # SSA begins for a for statement (line 37)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Obtaining the type of the subscript
        int_24505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 26), 'int')
        slice_24506 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 38, 15), None, int_24505, None)
        # Getting the type of 'info_file' (line 38)
        info_file_24507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'info_file')
        # Obtaining the member '__getitem__' of a type (line 38)
        getitem___24508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 15), info_file_24507, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 38)
        subscript_call_result_24509 = invoke(stypy.reporting.localization.Localization(__file__, 38, 15), getitem___24508, slice_24506)
        
        # Getting the type of 'files' (line 38)
        files_24510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 37), 'files')
        # Applying the binary operator 'notin' (line 38)
        result_contains_24511 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 15), 'notin', subscript_call_result_24509, files_24510)
        
        # Testing the type of an if condition (line 38)
        if_condition_24512 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 12), result_contains_24511)
        # Assigning a type to the variable 'if_condition_24512' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'if_condition_24512', if_condition_24512)
        # SSA begins for if statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'info_file' (line 39)
        info_file_24515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'info_file', False)
        # Processing the call keyword arguments (line 39)
        kwargs_24516 = {}
        # Getting the type of 'files' (line 39)
        files_24513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'files', False)
        # Obtaining the member 'append' of a type (line 39)
        append_24514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 16), files_24513, 'append')
        # Calling append(args, kwargs) (line 39)
        append_call_result_24517 = invoke(stypy.reporting.localization.Localization(__file__, 39, 16), append_24514, *[info_file_24515], **kwargs_24516)
        
        # SSA join for if statement (line 38)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 40):
        
        # Obtaining an instance of the builtin type 'list' (line 40)
        list_24518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 40)
        
        # Assigning a type to the variable 'info_files' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'info_files', list_24518)
        
        # Getting the type of 'files' (line 41)
        files_24519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 25), 'files')
        # Testing the type of a for loop iterable (line 41)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 41, 8), files_24519)
        # Getting the type of the for loop variable (line 41)
        for_loop_var_24520 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 41, 8), files_24519)
        # Assigning a type to the variable 'info_file' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'info_file', for_loop_var_24520)
        # SSA begins for a for statement (line 41)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 42):
        
        # Call to replace(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'os' (line 43)
        os_24537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 36), 'os', False)
        # Obtaining the member 'sep' of a type (line 43)
        sep_24538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 36), os_24537, 'sep')
        str_24539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 44), 'str', '.')
        # Processing the call keyword arguments (line 42)
        kwargs_24540 = {}
        
        # Call to dirname(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Obtaining the type of the subscript
        
        # Call to len(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'parent_path' (line 42)
        parent_path_24525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 57), 'parent_path', False)
        # Processing the call keyword arguments (line 42)
        kwargs_24526 = {}
        # Getting the type of 'len' (line 42)
        len_24524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 53), 'len', False)
        # Calling len(args, kwargs) (line 42)
        len_call_result_24527 = invoke(stypy.reporting.localization.Localization(__file__, 42, 53), len_24524, *[parent_path_24525], **kwargs_24526)
        
        int_24528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 70), 'int')
        # Applying the binary operator '+' (line 42)
        result_add_24529 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 53), '+', len_call_result_24527, int_24528)
        
        slice_24530 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 42, 43), result_add_24529, None, None)
        # Getting the type of 'info_file' (line 42)
        info_file_24531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 43), 'info_file', False)
        # Obtaining the member '__getitem__' of a type (line 42)
        getitem___24532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 43), info_file_24531, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 42)
        subscript_call_result_24533 = invoke(stypy.reporting.localization.Localization(__file__, 42, 43), getitem___24532, slice_24530)
        
        # Processing the call keyword arguments (line 42)
        kwargs_24534 = {}
        # Getting the type of 'os' (line 42)
        os_24521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 42)
        path_24522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 27), os_24521, 'path')
        # Obtaining the member 'dirname' of a type (line 42)
        dirname_24523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 27), path_24522, 'dirname')
        # Calling dirname(args, kwargs) (line 42)
        dirname_call_result_24535 = invoke(stypy.reporting.localization.Localization(__file__, 42, 27), dirname_24523, *[subscript_call_result_24533], **kwargs_24534)
        
        # Obtaining the member 'replace' of a type (line 42)
        replace_24536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 27), dirname_call_result_24535, 'replace')
        # Calling replace(args, kwargs) (line 42)
        replace_call_result_24541 = invoke(stypy.reporting.localization.Localization(__file__, 42, 27), replace_24536, *[sep_24538, str_24539], **kwargs_24540)
        
        # Assigning a type to the variable 'package_name' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'package_name', replace_call_result_24541)
        
        # Getting the type of 'parent_package' (line 44)
        parent_package_24542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'parent_package')
        # Testing the type of an if condition (line 44)
        if_condition_24543 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 12), parent_package_24542)
        # Assigning a type to the variable 'if_condition_24543' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'if_condition_24543', if_condition_24543)
        # SSA begins for if statement (line 44)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 45):
        # Getting the type of 'parent_package' (line 45)
        parent_package_24544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 31), 'parent_package')
        str_24545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 48), 'str', '.')
        # Applying the binary operator '+' (line 45)
        result_add_24546 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 31), '+', parent_package_24544, str_24545)
        
        # Getting the type of 'package_name' (line 45)
        package_name_24547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 54), 'package_name')
        # Applying the binary operator '+' (line 45)
        result_add_24548 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 52), '+', result_add_24546, package_name_24547)
        
        # Assigning a type to the variable 'package_name' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'package_name', result_add_24548)
        # SSA join for if statement (line 44)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Obtaining an instance of the builtin type 'tuple' (line 46)
        tuple_24551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 46)
        # Adding element type (line 46)
        # Getting the type of 'package_name' (line 46)
        package_name_24552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 31), 'package_name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 31), tuple_24551, package_name_24552)
        # Adding element type (line 46)
        # Getting the type of 'info_file' (line 46)
        info_file_24553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 45), 'info_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 31), tuple_24551, info_file_24553)
        
        # Processing the call keyword arguments (line 46)
        kwargs_24554 = {}
        # Getting the type of 'info_files' (line 46)
        info_files_24549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'info_files', False)
        # Obtaining the member 'append' of a type (line 46)
        append_24550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 12), info_files_24549, 'append')
        # Calling append(args, kwargs) (line 46)
        append_call_result_24555 = invoke(stypy.reporting.localization.Localization(__file__, 46, 12), append_24550, *[tuple_24551], **kwargs_24554)
        
        
        # Call to extend(...): (line 47)
        # Processing the call arguments (line 47)
        
        # Call to _get_info_files(...): (line 47)
        # Processing the call arguments (line 47)
        str_24560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 51), 'str', '*')
        
        # Call to dirname(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'info_file' (line 48)
        info_file_24564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 67), 'info_file', False)
        # Processing the call keyword arguments (line 48)
        kwargs_24565 = {}
        # Getting the type of 'os' (line 48)
        os_24561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 51), 'os', False)
        # Obtaining the member 'path' of a type (line 48)
        path_24562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 51), os_24561, 'path')
        # Obtaining the member 'dirname' of a type (line 48)
        dirname_24563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 51), path_24562, 'dirname')
        # Calling dirname(args, kwargs) (line 48)
        dirname_call_result_24566 = invoke(stypy.reporting.localization.Localization(__file__, 48, 51), dirname_24563, *[info_file_24564], **kwargs_24565)
        
        # Getting the type of 'package_name' (line 49)
        package_name_24567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 51), 'package_name', False)
        # Processing the call keyword arguments (line 47)
        kwargs_24568 = {}
        # Getting the type of 'self' (line 47)
        self_24558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 30), 'self', False)
        # Obtaining the member '_get_info_files' of a type (line 47)
        _get_info_files_24559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 30), self_24558, '_get_info_files')
        # Calling _get_info_files(args, kwargs) (line 47)
        _get_info_files_call_result_24569 = invoke(stypy.reporting.localization.Localization(__file__, 47, 30), _get_info_files_24559, *[str_24560, dirname_call_result_24566, package_name_24567], **kwargs_24568)
        
        # Processing the call keyword arguments (line 47)
        kwargs_24570 = {}
        # Getting the type of 'info_files' (line 47)
        info_files_24556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'info_files', False)
        # Obtaining the member 'extend' of a type (line 47)
        extend_24557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), info_files_24556, 'extend')
        # Calling extend(args, kwargs) (line 47)
        extend_call_result_24571 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), extend_24557, *[_get_info_files_call_result_24569], **kwargs_24570)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'info_files' (line 50)
        info_files_24572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'info_files')
        # Assigning a type to the variable 'stypy_return_type' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'stypy_return_type', info_files_24572)
        
        # ################# End of '_get_info_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_info_files' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_24573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24573)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_info_files'
        return stypy_return_type_24573


    @norecursion
    def _init_info_modules(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 52)
        None_24574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 42), 'None')
        defaults = [None_24574]
        # Create a new context for function '_init_info_modules'
        module_type_store = module_type_store.open_function_context('_init_info_modules', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PackageLoader._init_info_modules.__dict__.__setitem__('stypy_localization', localization)
        PackageLoader._init_info_modules.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PackageLoader._init_info_modules.__dict__.__setitem__('stypy_type_store', module_type_store)
        PackageLoader._init_info_modules.__dict__.__setitem__('stypy_function_name', 'PackageLoader._init_info_modules')
        PackageLoader._init_info_modules.__dict__.__setitem__('stypy_param_names_list', ['packages'])
        PackageLoader._init_info_modules.__dict__.__setitem__('stypy_varargs_param_name', None)
        PackageLoader._init_info_modules.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PackageLoader._init_info_modules.__dict__.__setitem__('stypy_call_defaults', defaults)
        PackageLoader._init_info_modules.__dict__.__setitem__('stypy_call_varargs', varargs)
        PackageLoader._init_info_modules.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PackageLoader._init_info_modules.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PackageLoader._init_info_modules', ['packages'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_init_info_modules', localization, ['packages'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_init_info_modules(...)' code ##################

        str_24575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, (-1)), 'str', 'Initialize info_modules = {<package_name>: <package info.py module>}.\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 55, 8))
        
        # 'import imp' statement (line 55)
        import imp

        import_module(stypy.reporting.localization.Localization(__file__, 55, 8), 'imp', imp, module_type_store)
        
        
        # Assigning a List to a Name (line 56):
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_24576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        
        # Assigning a type to the variable 'info_files' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'info_files', list_24576)
        
        # Assigning a Attribute to a Name (line 57):
        # Getting the type of 'self' (line 57)
        self_24577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'self')
        # Obtaining the member 'info_modules' of a type (line 57)
        info_modules_24578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 23), self_24577, 'info_modules')
        # Assigning a type to the variable 'info_modules' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'info_modules', info_modules_24578)
        
        # Type idiom detected: calculating its left and rigth part (line 59)
        # Getting the type of 'packages' (line 59)
        packages_24579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'packages')
        # Getting the type of 'None' (line 59)
        None_24580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 23), 'None')
        
        (may_be_24581, more_types_in_union_24582) = may_be_none(packages_24579, None_24580)

        if may_be_24581:

            if more_types_in_union_24582:
                # Runtime conditional SSA (line 59)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'self' (line 60)
            self_24583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 24), 'self')
            # Obtaining the member 'parent_path' of a type (line 60)
            parent_path_24584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 24), self_24583, 'parent_path')
            # Testing the type of a for loop iterable (line 60)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 60, 12), parent_path_24584)
            # Getting the type of the for loop variable (line 60)
            for_loop_var_24585 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 60, 12), parent_path_24584)
            # Assigning a type to the variable 'path' (line 60)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'path', for_loop_var_24585)
            # SSA begins for a for statement (line 60)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to extend(...): (line 61)
            # Processing the call arguments (line 61)
            
            # Call to _get_info_files(...): (line 61)
            # Processing the call arguments (line 61)
            str_24590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 55), 'str', '*')
            # Getting the type of 'path' (line 61)
            path_24591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 60), 'path', False)
            # Processing the call keyword arguments (line 61)
            kwargs_24592 = {}
            # Getting the type of 'self' (line 61)
            self_24588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'self', False)
            # Obtaining the member '_get_info_files' of a type (line 61)
            _get_info_files_24589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 34), self_24588, '_get_info_files')
            # Calling _get_info_files(args, kwargs) (line 61)
            _get_info_files_call_result_24593 = invoke(stypy.reporting.localization.Localization(__file__, 61, 34), _get_info_files_24589, *[str_24590, path_24591], **kwargs_24592)
            
            # Processing the call keyword arguments (line 61)
            kwargs_24594 = {}
            # Getting the type of 'info_files' (line 61)
            info_files_24586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'info_files', False)
            # Obtaining the member 'extend' of a type (line 61)
            extend_24587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 16), info_files_24586, 'extend')
            # Calling extend(args, kwargs) (line 61)
            extend_call_result_24595 = invoke(stypy.reporting.localization.Localization(__file__, 61, 16), extend_24587, *[_get_info_files_call_result_24593], **kwargs_24594)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_24582:
                # Runtime conditional SSA for else branch (line 59)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_24581) or more_types_in_union_24582):
            
            # Getting the type of 'packages' (line 63)
            packages_24596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 32), 'packages')
            # Testing the type of a for loop iterable (line 63)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 63, 12), packages_24596)
            # Getting the type of the for loop variable (line 63)
            for_loop_var_24597 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 63, 12), packages_24596)
            # Assigning a type to the variable 'package_name' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'package_name', for_loop_var_24597)
            # SSA begins for a for statement (line 63)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 64):
            
            # Call to join(...): (line 64)
            
            # Call to split(...): (line 64)
            # Processing the call arguments (line 64)
            str_24603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 63), 'str', '.')
            # Processing the call keyword arguments (line 64)
            kwargs_24604 = {}
            # Getting the type of 'package_name' (line 64)
            package_name_24601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 44), 'package_name', False)
            # Obtaining the member 'split' of a type (line 64)
            split_24602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 44), package_name_24601, 'split')
            # Calling split(args, kwargs) (line 64)
            split_call_result_24605 = invoke(stypy.reporting.localization.Localization(__file__, 64, 44), split_24602, *[str_24603], **kwargs_24604)
            
            # Processing the call keyword arguments (line 64)
            kwargs_24606 = {}
            # Getting the type of 'os' (line 64)
            os_24598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 30), 'os', False)
            # Obtaining the member 'path' of a type (line 64)
            path_24599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 30), os_24598, 'path')
            # Obtaining the member 'join' of a type (line 64)
            join_24600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 30), path_24599, 'join')
            # Calling join(args, kwargs) (line 64)
            join_call_result_24607 = invoke(stypy.reporting.localization.Localization(__file__, 64, 30), join_24600, *[split_call_result_24605], **kwargs_24606)
            
            # Assigning a type to the variable 'package_dir' (line 64)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'package_dir', join_call_result_24607)
            
            # Getting the type of 'self' (line 65)
            self_24608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 28), 'self')
            # Obtaining the member 'parent_path' of a type (line 65)
            parent_path_24609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 28), self_24608, 'parent_path')
            # Testing the type of a for loop iterable (line 65)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 65, 16), parent_path_24609)
            # Getting the type of the for loop variable (line 65)
            for_loop_var_24610 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 65, 16), parent_path_24609)
            # Assigning a type to the variable 'path' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'path', for_loop_var_24610)
            # SSA begins for a for statement (line 65)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 66):
            
            # Call to _get_info_files(...): (line 66)
            # Processing the call arguments (line 66)
            # Getting the type of 'package_dir' (line 66)
            package_dir_24613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 55), 'package_dir', False)
            # Getting the type of 'path' (line 66)
            path_24614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 68), 'path', False)
            # Processing the call keyword arguments (line 66)
            kwargs_24615 = {}
            # Getting the type of 'self' (line 66)
            self_24611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 34), 'self', False)
            # Obtaining the member '_get_info_files' of a type (line 66)
            _get_info_files_24612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 34), self_24611, '_get_info_files')
            # Calling _get_info_files(args, kwargs) (line 66)
            _get_info_files_call_result_24616 = invoke(stypy.reporting.localization.Localization(__file__, 66, 34), _get_info_files_24612, *[package_dir_24613, path_24614], **kwargs_24615)
            
            # Assigning a type to the variable 'names_files' (line 66)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'names_files', _get_info_files_call_result_24616)
            
            # Getting the type of 'names_files' (line 67)
            names_files_24617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 23), 'names_files')
            # Testing the type of an if condition (line 67)
            if_condition_24618 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 20), names_files_24617)
            # Assigning a type to the variable 'if_condition_24618' (line 67)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), 'if_condition_24618', if_condition_24618)
            # SSA begins for if statement (line 67)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to extend(...): (line 68)
            # Processing the call arguments (line 68)
            # Getting the type of 'names_files' (line 68)
            names_files_24621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 42), 'names_files', False)
            # Processing the call keyword arguments (line 68)
            kwargs_24622 = {}
            # Getting the type of 'info_files' (line 68)
            info_files_24619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 24), 'info_files', False)
            # Obtaining the member 'extend' of a type (line 68)
            extend_24620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 24), info_files_24619, 'extend')
            # Calling extend(args, kwargs) (line 68)
            extend_call_result_24623 = invoke(stypy.reporting.localization.Localization(__file__, 68, 24), extend_24620, *[names_files_24621], **kwargs_24622)
            
            # SSA join for if statement (line 67)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of a for statement (line 65)
            module_type_store.open_ssa_branch('for loop else')
            
            
            # SSA begins for try-except statement (line 71)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            # Dynamic code evaluation using an exec statement
            str_24624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 29), 'str', 'import %s.info as info')
            # Getting the type of 'package_name' (line 72)
            package_name_24625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 57), 'package_name')
            # Applying the binary operator '%' (line 72)
            result_mod_24626 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 29), '%', str_24624, package_name_24625)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 72, 24), result_mod_24626, 'exec parameter', 'StringType', 'FileType', 'CodeType')
            enable_usage_of_dynamic_types_warning(stypy.reporting.localization.Localization(__file__, 72, 24))
            
            # Assigning a Name to a Subscript (line 73):
            # Getting the type of 'info' (line 73)
            info_24627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 53), 'info')
            # Getting the type of 'info_modules' (line 73)
            info_modules_24628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 24), 'info_modules')
            # Getting the type of 'package_name' (line 73)
            package_name_24629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 37), 'package_name')
            # Storing an element on a container (line 73)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 24), info_modules_24628, (package_name_24629, info_24627))
            # SSA branch for the except part of a try statement (line 71)
            # SSA branch for the except 'ImportError' branch of a try statement (line 71)
            # Storing handler type
            module_type_store.open_ssa_branch('except')
            # Getting the type of 'ImportError' (line 74)
            ImportError_24630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 27), 'ImportError')
            # Assigning a type to the variable 'msg' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'msg', ImportError_24630)
            
            # Call to warn(...): (line 75)
            # Processing the call arguments (line 75)
            str_24633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 34), 'str', 'No scipy-style subpackage %r found in %s. Ignoring: %s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 77)
            tuple_24634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 37), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 77)
            # Adding element type (line 77)
            # Getting the type of 'package_name' (line 77)
            package_name_24635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 37), 'package_name', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 37), tuple_24634, package_name_24635)
            # Adding element type (line 77)
            
            # Call to join(...): (line 77)
            # Processing the call arguments (line 77)
            # Getting the type of 'self' (line 77)
            self_24638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 60), 'self', False)
            # Obtaining the member 'parent_path' of a type (line 77)
            parent_path_24639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 60), self_24638, 'parent_path')
            # Processing the call keyword arguments (line 77)
            kwargs_24640 = {}
            str_24636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 51), 'str', ':')
            # Obtaining the member 'join' of a type (line 77)
            join_24637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 51), str_24636, 'join')
            # Calling join(args, kwargs) (line 77)
            join_call_result_24641 = invoke(stypy.reporting.localization.Localization(__file__, 77, 51), join_24637, *[parent_path_24639], **kwargs_24640)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 37), tuple_24634, join_call_result_24641)
            # Adding element type (line 77)
            # Getting the type of 'msg' (line 77)
            msg_24642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 79), 'msg', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 37), tuple_24634, msg_24642)
            
            # Applying the binary operator '%' (line 75)
            result_mod_24643 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 34), '%', str_24633, tuple_24634)
            
            # Processing the call keyword arguments (line 75)
            kwargs_24644 = {}
            # Getting the type of 'self' (line 75)
            self_24631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 24), 'self', False)
            # Obtaining the member 'warn' of a type (line 75)
            warn_24632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 24), self_24631, 'warn')
            # Calling warn(args, kwargs) (line 75)
            warn_call_result_24645 = invoke(stypy.reporting.localization.Localization(__file__, 75, 24), warn_24632, *[result_mod_24643], **kwargs_24644)
            
            # SSA join for try-except statement (line 71)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_24581 and more_types_in_union_24582):
                # SSA join for if statement (line 59)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'info_files' (line 79)
        info_files_24646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 39), 'info_files')
        # Testing the type of a for loop iterable (line 79)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 79, 8), info_files_24646)
        # Getting the type of the for loop variable (line 79)
        for_loop_var_24647 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 79, 8), info_files_24646)
        # Assigning a type to the variable 'package_name' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'package_name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 8), for_loop_var_24647))
        # Assigning a type to the variable 'info_file' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'info_file', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 8), for_loop_var_24647))
        # SSA begins for a for statement (line 79)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'package_name' (line 80)
        package_name_24648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'package_name')
        # Getting the type of 'info_modules' (line 80)
        info_modules_24649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 31), 'info_modules')
        # Applying the binary operator 'in' (line 80)
        result_contains_24650 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 15), 'in', package_name_24648, info_modules_24649)
        
        # Testing the type of an if condition (line 80)
        if_condition_24651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 12), result_contains_24650)
        # Assigning a type to the variable 'if_condition_24651' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'if_condition_24651', if_condition_24651)
        # SSA begins for if statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 80)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 82):
        # Getting the type of 'self' (line 82)
        self_24652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 23), 'self')
        # Obtaining the member 'parent_name' of a type (line 82)
        parent_name_24653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 23), self_24652, 'parent_name')
        str_24654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 41), 'str', '.')
        # Applying the binary operator '+' (line 82)
        result_add_24655 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 23), '+', parent_name_24653, str_24654)
        
        # Getting the type of 'package_name' (line 82)
        package_name_24656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 46), 'package_name')
        # Applying the binary operator '+' (line 82)
        result_add_24657 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 44), '+', result_add_24655, package_name_24656)
        
        # Assigning a type to the variable 'fullname' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'fullname', result_add_24657)
        
        
        
        # Obtaining the type of the subscript
        int_24658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 25), 'int')
        # Getting the type of 'info_file' (line 83)
        info_file_24659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'info_file')
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___24660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 15), info_file_24659, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_24661 = invoke(stypy.reporting.localization.Localization(__file__, 83, 15), getitem___24660, int_24658)
        
        str_24662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 30), 'str', 'c')
        # Applying the binary operator '==' (line 83)
        result_eq_24663 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 15), '==', subscript_call_result_24661, str_24662)
        
        # Testing the type of an if condition (line 83)
        if_condition_24664 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 12), result_eq_24663)
        # Assigning a type to the variable 'if_condition_24664' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'if_condition_24664', if_condition_24664)
        # SSA begins for if statement (line 83)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Name (line 84):
        
        # Obtaining an instance of the builtin type 'tuple' (line 84)
        tuple_24665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 84)
        # Adding element type (line 84)
        str_24666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 34), 'str', '.pyc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 34), tuple_24665, str_24666)
        # Adding element type (line 84)
        str_24667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 42), 'str', 'rb')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 34), tuple_24665, str_24667)
        # Adding element type (line 84)
        int_24668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 34), tuple_24665, int_24668)
        
        # Assigning a type to the variable 'filedescriptor' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'filedescriptor', tuple_24665)
        # SSA branch for the else part of an if statement (line 83)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Tuple to a Name (line 86):
        
        # Obtaining an instance of the builtin type 'tuple' (line 86)
        tuple_24669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 86)
        # Adding element type (line 86)
        str_24670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 34), 'str', '.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 34), tuple_24669, str_24670)
        # Adding element type (line 86)
        str_24671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 41), 'str', 'U')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 34), tuple_24669, str_24671)
        # Adding element type (line 86)
        int_24672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 34), tuple_24669, int_24672)
        
        # Assigning a type to the variable 'filedescriptor' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'filedescriptor', tuple_24669)
        # SSA join for if statement (line 83)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 89):
        
        # Call to load_module(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'fullname' (line 89)
        fullname_24675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 46), 'fullname', False)
        str_24676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 55), 'str', '.info')
        # Applying the binary operator '+' (line 89)
        result_add_24677 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 46), '+', fullname_24675, str_24676)
        
        
        # Call to open(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'info_file' (line 90)
        info_file_24679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 51), 'info_file', False)
        
        # Obtaining the type of the subscript
        int_24680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 77), 'int')
        # Getting the type of 'filedescriptor' (line 90)
        filedescriptor_24681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 62), 'filedescriptor', False)
        # Obtaining the member '__getitem__' of a type (line 90)
        getitem___24682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 62), filedescriptor_24681, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 90)
        subscript_call_result_24683 = invoke(stypy.reporting.localization.Localization(__file__, 90, 62), getitem___24682, int_24680)
        
        # Processing the call keyword arguments (line 90)
        kwargs_24684 = {}
        # Getting the type of 'open' (line 90)
        open_24678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 46), 'open', False)
        # Calling open(args, kwargs) (line 90)
        open_call_result_24685 = invoke(stypy.reporting.localization.Localization(__file__, 90, 46), open_24678, *[info_file_24679, subscript_call_result_24683], **kwargs_24684)
        
        # Getting the type of 'info_file' (line 91)
        info_file_24686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 46), 'info_file', False)
        # Getting the type of 'filedescriptor' (line 92)
        filedescriptor_24687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 46), 'filedescriptor', False)
        # Processing the call keyword arguments (line 89)
        kwargs_24688 = {}
        # Getting the type of 'imp' (line 89)
        imp_24673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 30), 'imp', False)
        # Obtaining the member 'load_module' of a type (line 89)
        load_module_24674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 30), imp_24673, 'load_module')
        # Calling load_module(args, kwargs) (line 89)
        load_module_call_result_24689 = invoke(stypy.reporting.localization.Localization(__file__, 89, 30), load_module_24674, *[result_add_24677, open_call_result_24685, info_file_24686, filedescriptor_24687], **kwargs_24688)
        
        # Assigning a type to the variable 'info_module' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'info_module', load_module_call_result_24689)
        # SSA branch for the except part of a try statement (line 88)
        # SSA branch for the except 'Exception' branch of a try statement (line 88)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'Exception' (line 93)
        Exception_24690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 19), 'Exception')
        # Assigning a type to the variable 'msg' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'msg', Exception_24690)
        
        # Call to error(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'msg' (line 94)
        msg_24693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 27), 'msg', False)
        # Processing the call keyword arguments (line 94)
        kwargs_24694 = {}
        # Getting the type of 'self' (line 94)
        self_24691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 16), 'self', False)
        # Obtaining the member 'error' of a type (line 94)
        error_24692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 16), self_24691, 'error')
        # Calling error(args, kwargs) (line 94)
        error_call_result_24695 = invoke(stypy.reporting.localization.Localization(__file__, 94, 16), error_24692, *[msg_24693], **kwargs_24694)
        
        
        # Assigning a Name to a Name (line 95):
        # Getting the type of 'None' (line 95)
        None_24696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 30), 'None')
        # Assigning a type to the variable 'info_module' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'info_module', None_24696)
        # SSA join for try-except statement (line 88)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'info_module' (line 97)
        info_module_24697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'info_module')
        # Getting the type of 'None' (line 97)
        None_24698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 30), 'None')
        # Applying the binary operator 'is' (line 97)
        result_is__24699 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 15), 'is', info_module_24697, None_24698)
        
        
        # Call to getattr(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'info_module' (line 97)
        info_module_24701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 46), 'info_module', False)
        str_24702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 59), 'str', 'ignore')
        # Getting the type of 'False' (line 97)
        False_24703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 69), 'False', False)
        # Processing the call keyword arguments (line 97)
        kwargs_24704 = {}
        # Getting the type of 'getattr' (line 97)
        getattr_24700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 38), 'getattr', False)
        # Calling getattr(args, kwargs) (line 97)
        getattr_call_result_24705 = invoke(stypy.reporting.localization.Localization(__file__, 97, 38), getattr_24700, *[info_module_24701, str_24702, False_24703], **kwargs_24704)
        
        # Applying the binary operator 'or' (line 97)
        result_or_keyword_24706 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 15), 'or', result_is__24699, getattr_call_result_24705)
        
        # Testing the type of an if condition (line 97)
        if_condition_24707 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 12), result_or_keyword_24706)
        # Assigning a type to the variable 'if_condition_24707' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'if_condition_24707', if_condition_24707)
        # SSA begins for if statement (line 97)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to pop(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'package_name' (line 98)
        package_name_24710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 'package_name', False)
        # Getting the type of 'None' (line 98)
        None_24711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 47), 'None', False)
        # Processing the call keyword arguments (line 98)
        kwargs_24712 = {}
        # Getting the type of 'info_modules' (line 98)
        info_modules_24708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'info_modules', False)
        # Obtaining the member 'pop' of a type (line 98)
        pop_24709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 16), info_modules_24708, 'pop')
        # Calling pop(args, kwargs) (line 98)
        pop_call_result_24713 = invoke(stypy.reporting.localization.Localization(__file__, 98, 16), pop_24709, *[package_name_24710, None_24711], **kwargs_24712)
        
        # SSA branch for the else part of an if statement (line 97)
        module_type_store.open_ssa_branch('else')
        
        # Call to _init_info_modules(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Call to getattr(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'info_module' (line 100)
        info_module_24717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 48), 'info_module', False)
        str_24718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 61), 'str', 'depends')
        
        # Obtaining an instance of the builtin type 'list' (line 100)
        list_24719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 72), 'list')
        # Adding type elements to the builtin type 'list' instance (line 100)
        
        # Processing the call keyword arguments (line 100)
        kwargs_24720 = {}
        # Getting the type of 'getattr' (line 100)
        getattr_24716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 40), 'getattr', False)
        # Calling getattr(args, kwargs) (line 100)
        getattr_call_result_24721 = invoke(stypy.reporting.localization.Localization(__file__, 100, 40), getattr_24716, *[info_module_24717, str_24718, list_24719], **kwargs_24720)
        
        # Processing the call keyword arguments (line 100)
        kwargs_24722 = {}
        # Getting the type of 'self' (line 100)
        self_24714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'self', False)
        # Obtaining the member '_init_info_modules' of a type (line 100)
        _init_info_modules_24715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 16), self_24714, '_init_info_modules')
        # Calling _init_info_modules(args, kwargs) (line 100)
        _init_info_modules_call_result_24723 = invoke(stypy.reporting.localization.Localization(__file__, 100, 16), _init_info_modules_24715, *[getattr_call_result_24721], **kwargs_24722)
        
        
        # Assigning a Name to a Subscript (line 101):
        # Getting the type of 'info_module' (line 101)
        info_module_24724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 45), 'info_module')
        # Getting the type of 'info_modules' (line 101)
        info_modules_24725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'info_modules')
        # Getting the type of 'package_name' (line 101)
        package_name_24726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 29), 'package_name')
        # Storing an element on a container (line 101)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 16), info_modules_24725, (package_name_24726, info_module_24724))
        # SSA join for if statement (line 97)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Assigning a type to the variable 'stypy_return_type' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of '_init_info_modules(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_init_info_modules' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_24727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24727)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_init_info_modules'
        return stypy_return_type_24727


    @norecursion
    def _get_sorted_names(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_sorted_names'
        module_type_store = module_type_store.open_function_context('_get_sorted_names', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PackageLoader._get_sorted_names.__dict__.__setitem__('stypy_localization', localization)
        PackageLoader._get_sorted_names.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PackageLoader._get_sorted_names.__dict__.__setitem__('stypy_type_store', module_type_store)
        PackageLoader._get_sorted_names.__dict__.__setitem__('stypy_function_name', 'PackageLoader._get_sorted_names')
        PackageLoader._get_sorted_names.__dict__.__setitem__('stypy_param_names_list', [])
        PackageLoader._get_sorted_names.__dict__.__setitem__('stypy_varargs_param_name', None)
        PackageLoader._get_sorted_names.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PackageLoader._get_sorted_names.__dict__.__setitem__('stypy_call_defaults', defaults)
        PackageLoader._get_sorted_names.__dict__.__setitem__('stypy_call_varargs', varargs)
        PackageLoader._get_sorted_names.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PackageLoader._get_sorted_names.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PackageLoader._get_sorted_names', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_sorted_names', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_sorted_names(...)' code ##################

        str_24728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, (-1)), 'str', ' Return package names sorted in the order as they should be\n        imported due to dependence relations between packages.\n        ')
        
        # Assigning a Dict to a Name (line 110):
        
        # Obtaining an instance of the builtin type 'dict' (line 110)
        dict_24729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 22), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 110)
        
        # Assigning a type to the variable 'depend_dict' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'depend_dict', dict_24729)
        
        
        # Call to items(...): (line 111)
        # Processing the call keyword arguments (line 111)
        kwargs_24733 = {}
        # Getting the type of 'self' (line 111)
        self_24730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 33), 'self', False)
        # Obtaining the member 'info_modules' of a type (line 111)
        info_modules_24731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 33), self_24730, 'info_modules')
        # Obtaining the member 'items' of a type (line 111)
        items_24732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 33), info_modules_24731, 'items')
        # Calling items(args, kwargs) (line 111)
        items_call_result_24734 = invoke(stypy.reporting.localization.Localization(__file__, 111, 33), items_24732, *[], **kwargs_24733)
        
        # Testing the type of a for loop iterable (line 111)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 111, 8), items_call_result_24734)
        # Getting the type of the for loop variable (line 111)
        for_loop_var_24735 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 111, 8), items_call_result_24734)
        # Assigning a type to the variable 'name' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 8), for_loop_var_24735))
        # Assigning a type to the variable 'info_module' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'info_module', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 8), for_loop_var_24735))
        # SSA begins for a for statement (line 111)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Subscript (line 112):
        
        # Call to getattr(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'info_module' (line 112)
        info_module_24737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 40), 'info_module', False)
        str_24738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 53), 'str', 'depends')
        
        # Obtaining an instance of the builtin type 'list' (line 112)
        list_24739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 64), 'list')
        # Adding type elements to the builtin type 'list' instance (line 112)
        
        # Processing the call keyword arguments (line 112)
        kwargs_24740 = {}
        # Getting the type of 'getattr' (line 112)
        getattr_24736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 32), 'getattr', False)
        # Calling getattr(args, kwargs) (line 112)
        getattr_call_result_24741 = invoke(stypy.reporting.localization.Localization(__file__, 112, 32), getattr_24736, *[info_module_24737, str_24738, list_24739], **kwargs_24740)
        
        # Getting the type of 'depend_dict' (line 112)
        depend_dict_24742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'depend_dict')
        # Getting the type of 'name' (line 112)
        name_24743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 24), 'name')
        # Storing an element on a container (line 112)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 12), depend_dict_24742, (name_24743, getattr_call_result_24741))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 113):
        
        # Obtaining an instance of the builtin type 'list' (line 113)
        list_24744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 113)
        
        # Assigning a type to the variable 'package_names' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'package_names', list_24744)
        
        
        # Call to list(...): (line 115)
        # Processing the call arguments (line 115)
        
        # Call to keys(...): (line 115)
        # Processing the call keyword arguments (line 115)
        kwargs_24748 = {}
        # Getting the type of 'depend_dict' (line 115)
        depend_dict_24746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 25), 'depend_dict', False)
        # Obtaining the member 'keys' of a type (line 115)
        keys_24747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 25), depend_dict_24746, 'keys')
        # Calling keys(args, kwargs) (line 115)
        keys_call_result_24749 = invoke(stypy.reporting.localization.Localization(__file__, 115, 25), keys_24747, *[], **kwargs_24748)
        
        # Processing the call keyword arguments (line 115)
        kwargs_24750 = {}
        # Getting the type of 'list' (line 115)
        list_24745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'list', False)
        # Calling list(args, kwargs) (line 115)
        list_call_result_24751 = invoke(stypy.reporting.localization.Localization(__file__, 115, 20), list_24745, *[keys_call_result_24749], **kwargs_24750)
        
        # Testing the type of a for loop iterable (line 115)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 115, 8), list_call_result_24751)
        # Getting the type of the for loop variable (line 115)
        for_loop_var_24752 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 115, 8), list_call_result_24751)
        # Assigning a type to the variable 'name' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'name', for_loop_var_24752)
        # SSA begins for a for statement (line 115)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 116)
        name_24753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), 'name')
        # Getting the type of 'depend_dict' (line 116)
        depend_dict_24754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 19), 'depend_dict')
        # Obtaining the member '__getitem__' of a type (line 116)
        getitem___24755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 19), depend_dict_24754, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 116)
        subscript_call_result_24756 = invoke(stypy.reporting.localization.Localization(__file__, 116, 19), getitem___24755, name_24753)
        
        # Applying the 'not' unary operator (line 116)
        result_not__24757 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 15), 'not', subscript_call_result_24756)
        
        # Testing the type of an if condition (line 116)
        if_condition_24758 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 12), result_not__24757)
        # Assigning a type to the variable 'if_condition_24758' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'if_condition_24758', if_condition_24758)
        # SSA begins for if statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'name' (line 117)
        name_24761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 37), 'name', False)
        # Processing the call keyword arguments (line 117)
        kwargs_24762 = {}
        # Getting the type of 'package_names' (line 117)
        package_names_24759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'package_names', False)
        # Obtaining the member 'append' of a type (line 117)
        append_24760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 16), package_names_24759, 'append')
        # Calling append(args, kwargs) (line 117)
        append_call_result_24763 = invoke(stypy.reporting.localization.Localization(__file__, 117, 16), append_24760, *[name_24761], **kwargs_24762)
        
        # Deleting a member
        # Getting the type of 'depend_dict' (line 118)
        depend_dict_24764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 20), 'depend_dict')
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 118)
        name_24765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 32), 'name')
        # Getting the type of 'depend_dict' (line 118)
        depend_dict_24766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 20), 'depend_dict')
        # Obtaining the member '__getitem__' of a type (line 118)
        getitem___24767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 20), depend_dict_24766, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 118)
        subscript_call_result_24768 = invoke(stypy.reporting.localization.Localization(__file__, 118, 20), getitem___24767, name_24765)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 16), depend_dict_24764, subscript_call_result_24768)
        # SSA join for if statement (line 116)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'depend_dict' (line 120)
        depend_dict_24769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 14), 'depend_dict')
        # Testing the type of an if condition (line 120)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 8), depend_dict_24769)
        # SSA begins for while statement (line 120)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        
        # Call to list(...): (line 121)
        # Processing the call arguments (line 121)
        
        # Call to items(...): (line 121)
        # Processing the call keyword arguments (line 121)
        kwargs_24773 = {}
        # Getting the type of 'depend_dict' (line 121)
        depend_dict_24771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 34), 'depend_dict', False)
        # Obtaining the member 'items' of a type (line 121)
        items_24772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 34), depend_dict_24771, 'items')
        # Calling items(args, kwargs) (line 121)
        items_call_result_24774 = invoke(stypy.reporting.localization.Localization(__file__, 121, 34), items_24772, *[], **kwargs_24773)
        
        # Processing the call keyword arguments (line 121)
        kwargs_24775 = {}
        # Getting the type of 'list' (line 121)
        list_24770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 29), 'list', False)
        # Calling list(args, kwargs) (line 121)
        list_call_result_24776 = invoke(stypy.reporting.localization.Localization(__file__, 121, 29), list_24770, *[items_call_result_24774], **kwargs_24775)
        
        # Testing the type of a for loop iterable (line 121)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 121, 12), list_call_result_24776)
        # Getting the type of the for loop variable (line 121)
        for_loop_var_24777 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 121, 12), list_call_result_24776)
        # Assigning a type to the variable 'name' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 12), for_loop_var_24777))
        # Assigning a type to the variable 'lst' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'lst', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 12), for_loop_var_24777))
        # SSA begins for a for statement (line 121)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a ListComp to a Name (line 122):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'lst' (line 122)
        lst_24782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 38), 'lst')
        comprehension_24783 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 27), lst_24782)
        # Assigning a type to the variable 'n' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 27), 'n', comprehension_24783)
        
        # Getting the type of 'n' (line 122)
        n_24779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 45), 'n')
        # Getting the type of 'depend_dict' (line 122)
        depend_dict_24780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 50), 'depend_dict')
        # Applying the binary operator 'in' (line 122)
        result_contains_24781 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 45), 'in', n_24779, depend_dict_24780)
        
        # Getting the type of 'n' (line 122)
        n_24778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 27), 'n')
        list_24784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 27), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 27), list_24784, n_24778)
        # Assigning a type to the variable 'new_lst' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'new_lst', list_24784)
        
        
        # Getting the type of 'new_lst' (line 123)
        new_lst_24785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 'new_lst')
        # Applying the 'not' unary operator (line 123)
        result_not__24786 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 19), 'not', new_lst_24785)
        
        # Testing the type of an if condition (line 123)
        if_condition_24787 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 16), result_not__24786)
        # Assigning a type to the variable 'if_condition_24787' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'if_condition_24787', if_condition_24787)
        # SSA begins for if statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'name' (line 124)
        name_24790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 41), 'name', False)
        # Processing the call keyword arguments (line 124)
        kwargs_24791 = {}
        # Getting the type of 'package_names' (line 124)
        package_names_24788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'package_names', False)
        # Obtaining the member 'append' of a type (line 124)
        append_24789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 20), package_names_24788, 'append')
        # Calling append(args, kwargs) (line 124)
        append_call_result_24792 = invoke(stypy.reporting.localization.Localization(__file__, 124, 20), append_24789, *[name_24790], **kwargs_24791)
        
        # Deleting a member
        # Getting the type of 'depend_dict' (line 125)
        depend_dict_24793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 24), 'depend_dict')
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 125)
        name_24794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 36), 'name')
        # Getting the type of 'depend_dict' (line 125)
        depend_dict_24795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 24), 'depend_dict')
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___24796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 24), depend_dict_24795, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 125)
        subscript_call_result_24797 = invoke(stypy.reporting.localization.Localization(__file__, 125, 24), getitem___24796, name_24794)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 20), depend_dict_24793, subscript_call_result_24797)
        # SSA branch for the else part of an if statement (line 123)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Subscript (line 127):
        # Getting the type of 'new_lst' (line 127)
        new_lst_24798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 40), 'new_lst')
        # Getting the type of 'depend_dict' (line 127)
        depend_dict_24799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'depend_dict')
        # Getting the type of 'name' (line 127)
        name_24800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 32), 'name')
        # Storing an element on a container (line 127)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 20), depend_dict_24799, (name_24800, new_lst_24798))
        # SSA join for if statement (line 123)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 120)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'package_names' (line 129)
        package_names_24801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 15), 'package_names')
        # Assigning a type to the variable 'stypy_return_type' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'stypy_return_type', package_names_24801)
        
        # ################# End of '_get_sorted_names(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_sorted_names' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_24802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24802)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_sorted_names'
        return stypy_return_type_24802


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PackageLoader.__call__.__dict__.__setitem__('stypy_localization', localization)
        PackageLoader.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PackageLoader.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        PackageLoader.__call__.__dict__.__setitem__('stypy_function_name', 'PackageLoader.__call__')
        PackageLoader.__call__.__dict__.__setitem__('stypy_param_names_list', [])
        PackageLoader.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'packages')
        PackageLoader.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'options')
        PackageLoader.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        PackageLoader.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        PackageLoader.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PackageLoader.__call__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PackageLoader.__call__', [], 'packages', 'options', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        str_24803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, (-1)), 'str', "Load one or more packages into parent package top-level namespace.\n\n       This function is intended to shorten the need to import many\n       subpackages, say of scipy, constantly with statements such as\n\n         import scipy.linalg, scipy.fftpack, scipy.etc...\n\n       Instead, you can say:\n\n         import scipy\n         scipy.pkgload('linalg','fftpack',...)\n\n       or\n\n         scipy.pkgload()\n\n       to load all of them in one call.\n\n       If a name which doesn't exist in scipy's namespace is\n       given, a warning is shown.\n\n       Parameters\n       ----------\n        *packages : arg-tuple\n             the names (one or more strings) of all the modules one\n             wishes to load into the top-level namespace.\n        verbose= : integer\n             verbosity level [default: -1].\n             verbose=-1 will suspend also warnings.\n        force= : bool\n             when True, force reloading loaded packages [default: False].\n        postpone= : bool\n             when True, don't load packages [default: False]\n\n        ")
        
        # Call to warn(...): (line 168)
        # Processing the call arguments (line 168)
        str_24806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 22), 'str', 'pkgload and PackageLoader are obsolete and will be removed in a future version of numpy')
        # Getting the type of 'DeprecationWarning' (line 170)
        DeprecationWarning_24807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'DeprecationWarning', False)
        # Processing the call keyword arguments (line 168)
        kwargs_24808 = {}
        # Getting the type of 'warnings' (line 168)
        warnings_24804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 168)
        warn_24805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), warnings_24804, 'warn')
        # Calling warn(args, kwargs) (line 168)
        warn_call_result_24809 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), warn_24805, *[str_24806, DeprecationWarning_24807], **kwargs_24808)
        
        
        # Assigning a Attribute to a Name (line 171):
        # Getting the type of 'self' (line 171)
        self_24810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'self')
        # Obtaining the member 'parent_frame' of a type (line 171)
        parent_frame_24811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 16), self_24810, 'parent_frame')
        # Assigning a type to the variable 'frame' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'frame', parent_frame_24811)
        
        # Assigning a Dict to a Attribute (line 172):
        
        # Obtaining an instance of the builtin type 'dict' (line 172)
        dict_24812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 172)
        
        # Getting the type of 'self' (line 172)
        self_24813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'self')
        # Setting the type of the member 'info_modules' of a type (line 172)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), self_24813, 'info_modules', dict_24812)
        
        
        # Call to get(...): (line 173)
        # Processing the call arguments (line 173)
        str_24816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 23), 'str', 'force')
        # Getting the type of 'False' (line 173)
        False_24817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 32), 'False', False)
        # Processing the call keyword arguments (line 173)
        kwargs_24818 = {}
        # Getting the type of 'options' (line 173)
        options_24814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 11), 'options', False)
        # Obtaining the member 'get' of a type (line 173)
        get_24815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 11), options_24814, 'get')
        # Calling get(args, kwargs) (line 173)
        get_call_result_24819 = invoke(stypy.reporting.localization.Localization(__file__, 173, 11), get_24815, *[str_24816, False_24817], **kwargs_24818)
        
        # Testing the type of an if condition (line 173)
        if_condition_24820 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 8), get_call_result_24819)
        # Assigning a type to the variable 'if_condition_24820' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'if_condition_24820', if_condition_24820)
        # SSA begins for if statement (line 173)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Attribute (line 174):
        
        # Obtaining an instance of the builtin type 'list' (line 174)
        list_24821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 174)
        
        # Getting the type of 'self' (line 174)
        self_24822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'self')
        # Setting the type of the member 'imported_packages' of a type (line 174)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 12), self_24822, 'imported_packages', list_24821)
        # SSA join for if statement (line 173)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Multiple assignment of 2 elements.
        
        # Call to get(...): (line 175)
        # Processing the call arguments (line 175)
        str_24825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 45), 'str', 'verbose')
        int_24826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 56), 'int')
        # Processing the call keyword arguments (line 175)
        kwargs_24827 = {}
        # Getting the type of 'options' (line 175)
        options_24823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 33), 'options', False)
        # Obtaining the member 'get' of a type (line 175)
        get_24824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 33), options_24823, 'get')
        # Calling get(args, kwargs) (line 175)
        get_call_result_24828 = invoke(stypy.reporting.localization.Localization(__file__, 175, 33), get_24824, *[str_24825, int_24826], **kwargs_24827)
        
        # Assigning a type to the variable 'verbose' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 23), 'verbose', get_call_result_24828)
        # Getting the type of 'verbose' (line 175)
        verbose_24829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 23), 'verbose')
        # Getting the type of 'self' (line 175)
        self_24830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'self')
        # Setting the type of the member 'verbose' of a type (line 175)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), self_24830, 'verbose', verbose_24829)
        
        # Assigning a Call to a Name (line 176):
        
        # Call to get(...): (line 176)
        # Processing the call arguments (line 176)
        str_24833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 31), 'str', 'postpone')
        # Getting the type of 'None' (line 176)
        None_24834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 43), 'None', False)
        # Processing the call keyword arguments (line 176)
        kwargs_24835 = {}
        # Getting the type of 'options' (line 176)
        options_24831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), 'options', False)
        # Obtaining the member 'get' of a type (line 176)
        get_24832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 19), options_24831, 'get')
        # Calling get(args, kwargs) (line 176)
        get_call_result_24836 = invoke(stypy.reporting.localization.Localization(__file__, 176, 19), get_24832, *[str_24833, None_24834], **kwargs_24835)
        
        # Assigning a type to the variable 'postpone' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'postpone', get_call_result_24836)
        
        # Call to _init_info_modules(...): (line 177)
        # Processing the call arguments (line 177)
        
        # Evaluating a boolean operation
        # Getting the type of 'packages' (line 177)
        packages_24839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 32), 'packages', False)
        # Getting the type of 'None' (line 177)
        None_24840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 44), 'None', False)
        # Applying the binary operator 'or' (line 177)
        result_or_keyword_24841 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 32), 'or', packages_24839, None_24840)
        
        # Processing the call keyword arguments (line 177)
        kwargs_24842 = {}
        # Getting the type of 'self' (line 177)
        self_24837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'self', False)
        # Obtaining the member '_init_info_modules' of a type (line 177)
        _init_info_modules_24838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), self_24837, '_init_info_modules')
        # Calling _init_info_modules(args, kwargs) (line 177)
        _init_info_modules_call_result_24843 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), _init_info_modules_24838, *[result_or_keyword_24841], **kwargs_24842)
        
        
        # Call to log(...): (line 179)
        # Processing the call arguments (line 179)
        str_24846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 17), 'str', 'Imports to %r namespace\n----------------------------')
        # Getting the type of 'self' (line 180)
        self_24847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 19), 'self', False)
        # Obtaining the member 'parent_name' of a type (line 180)
        parent_name_24848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 19), self_24847, 'parent_name')
        # Applying the binary operator '%' (line 179)
        result_mod_24849 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 17), '%', str_24846, parent_name_24848)
        
        # Processing the call keyword arguments (line 179)
        kwargs_24850 = {}
        # Getting the type of 'self' (line 179)
        self_24844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'self', False)
        # Obtaining the member 'log' of a type (line 179)
        log_24845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), self_24844, 'log')
        # Calling log(args, kwargs) (line 179)
        log_call_result_24851 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), log_24845, *[result_mod_24849], **kwargs_24850)
        
        
        
        # Call to _get_sorted_names(...): (line 182)
        # Processing the call keyword arguments (line 182)
        kwargs_24854 = {}
        # Getting the type of 'self' (line 182)
        self_24852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 28), 'self', False)
        # Obtaining the member '_get_sorted_names' of a type (line 182)
        _get_sorted_names_24853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 28), self_24852, '_get_sorted_names')
        # Calling _get_sorted_names(args, kwargs) (line 182)
        _get_sorted_names_call_result_24855 = invoke(stypy.reporting.localization.Localization(__file__, 182, 28), _get_sorted_names_24853, *[], **kwargs_24854)
        
        # Testing the type of a for loop iterable (line 182)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 182, 8), _get_sorted_names_call_result_24855)
        # Getting the type of the for loop variable (line 182)
        for_loop_var_24856 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 182, 8), _get_sorted_names_call_result_24855)
        # Assigning a type to the variable 'package_name' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'package_name', for_loop_var_24856)
        # SSA begins for a for statement (line 182)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'package_name' (line 183)
        package_name_24857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 15), 'package_name')
        # Getting the type of 'self' (line 183)
        self_24858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 31), 'self')
        # Obtaining the member 'imported_packages' of a type (line 183)
        imported_packages_24859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 31), self_24858, 'imported_packages')
        # Applying the binary operator 'in' (line 183)
        result_contains_24860 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 15), 'in', package_name_24857, imported_packages_24859)
        
        # Testing the type of an if condition (line 183)
        if_condition_24861 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 12), result_contains_24860)
        # Assigning a type to the variable 'if_condition_24861' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'if_condition_24861', if_condition_24861)
        # SSA begins for if statement (line 183)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 183)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 185):
        
        # Obtaining the type of the subscript
        # Getting the type of 'package_name' (line 185)
        package_name_24862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 44), 'package_name')
        # Getting the type of 'self' (line 185)
        self_24863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 26), 'self')
        # Obtaining the member 'info_modules' of a type (line 185)
        info_modules_24864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 26), self_24863, 'info_modules')
        # Obtaining the member '__getitem__' of a type (line 185)
        getitem___24865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 26), info_modules_24864, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 185)
        subscript_call_result_24866 = invoke(stypy.reporting.localization.Localization(__file__, 185, 26), getitem___24865, package_name_24862)
        
        # Assigning a type to the variable 'info_module' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'info_module', subscript_call_result_24866)
        
        # Assigning a Call to a Name (line 186):
        
        # Call to getattr(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'info_module' (line 186)
        info_module_24868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 37), 'info_module', False)
        str_24869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 50), 'str', 'global_symbols')
        
        # Obtaining an instance of the builtin type 'list' (line 186)
        list_24870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 68), 'list')
        # Adding type elements to the builtin type 'list' instance (line 186)
        
        # Processing the call keyword arguments (line 186)
        kwargs_24871 = {}
        # Getting the type of 'getattr' (line 186)
        getattr_24867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 29), 'getattr', False)
        # Calling getattr(args, kwargs) (line 186)
        getattr_call_result_24872 = invoke(stypy.reporting.localization.Localization(__file__, 186, 29), getattr_24867, *[info_module_24868, str_24869, list_24870], **kwargs_24871)
        
        # Assigning a type to the variable 'global_symbols' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'global_symbols', getattr_call_result_24872)
        
        # Assigning a Call to a Name (line 187):
        
        # Call to getattr(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'info_module' (line 187)
        info_module_24874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 38), 'info_module', False)
        str_24875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 51), 'str', 'postpone_import')
        # Getting the type of 'False' (line 187)
        False_24876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 70), 'False', False)
        # Processing the call keyword arguments (line 187)
        kwargs_24877 = {}
        # Getting the type of 'getattr' (line 187)
        getattr_24873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 30), 'getattr', False)
        # Calling getattr(args, kwargs) (line 187)
        getattr_call_result_24878 = invoke(stypy.reporting.localization.Localization(__file__, 187, 30), getattr_24873, *[info_module_24874, str_24875, False_24876], **kwargs_24877)
        
        # Assigning a type to the variable 'postpone_import' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'postpone_import', getattr_call_result_24878)
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'postpone' (line 188)
        postpone_24879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 16), 'postpone')
        
        # Getting the type of 'global_symbols' (line 188)
        global_symbols_24880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 33), 'global_symbols')
        # Applying the 'not' unary operator (line 188)
        result_not__24881 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 29), 'not', global_symbols_24880)
        
        # Applying the binary operator 'and' (line 188)
        result_and_keyword_24882 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 16), 'and', postpone_24879, result_not__24881)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'postpone_import' (line 189)
        postpone_import_24883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 23), 'postpone_import')
        
        # Getting the type of 'postpone' (line 189)
        postpone_24884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 43), 'postpone')
        # Getting the type of 'None' (line 189)
        None_24885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 59), 'None')
        # Applying the binary operator 'isnot' (line 189)
        result_is_not_24886 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 43), 'isnot', postpone_24884, None_24885)
        
        # Applying the binary operator 'and' (line 189)
        result_and_keyword_24887 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 23), 'and', postpone_import_24883, result_is_not_24886)
        
        # Applying the binary operator 'or' (line 188)
        result_or_keyword_24888 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 15), 'or', result_and_keyword_24882, result_and_keyword_24887)
        
        # Testing the type of an if condition (line 188)
        if_condition_24889 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 12), result_or_keyword_24888)
        # Assigning a type to the variable 'if_condition_24889' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'if_condition_24889', if_condition_24889)
        # SSA begins for if statement (line 188)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 188)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 192):
        
        # Call to get(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'package_name' (line 192)
        package_name_24893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 44), 'package_name', False)
        # Getting the type of 'None' (line 192)
        None_24894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 58), 'None', False)
        # Processing the call keyword arguments (line 192)
        kwargs_24895 = {}
        # Getting the type of 'frame' (line 192)
        frame_24890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 25), 'frame', False)
        # Obtaining the member 'f_locals' of a type (line 192)
        f_locals_24891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 25), frame_24890, 'f_locals')
        # Obtaining the member 'get' of a type (line 192)
        get_24892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 25), f_locals_24891, 'get')
        # Calling get(args, kwargs) (line 192)
        get_call_result_24896 = invoke(stypy.reporting.localization.Localization(__file__, 192, 25), get_24892, *[package_name_24893, None_24894], **kwargs_24895)
        
        # Assigning a type to the variable 'old_object' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'old_object', get_call_result_24896)
        
        # Assigning a BinOp to a Name (line 194):
        str_24897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 21), 'str', 'import ')
        # Getting the type of 'package_name' (line 194)
        package_name_24898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 31), 'package_name')
        # Applying the binary operator '+' (line 194)
        result_add_24899 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 21), '+', str_24897, package_name_24898)
        
        # Assigning a type to the variable 'cmdstr' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'cmdstr', result_add_24899)
        
        
        # Call to _execcmd(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'cmdstr' (line 195)
        cmdstr_24902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 29), 'cmdstr', False)
        # Processing the call keyword arguments (line 195)
        kwargs_24903 = {}
        # Getting the type of 'self' (line 195)
        self_24900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'self', False)
        # Obtaining the member '_execcmd' of a type (line 195)
        _execcmd_24901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 15), self_24900, '_execcmd')
        # Calling _execcmd(args, kwargs) (line 195)
        _execcmd_call_result_24904 = invoke(stypy.reporting.localization.Localization(__file__, 195, 15), _execcmd_24901, *[cmdstr_24902], **kwargs_24903)
        
        # Testing the type of an if condition (line 195)
        if_condition_24905 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 12), _execcmd_call_result_24904)
        # Assigning a type to the variable 'if_condition_24905' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'if_condition_24905', if_condition_24905)
        # SSA begins for if statement (line 195)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 195)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'package_name' (line 197)
        package_name_24909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 42), 'package_name', False)
        # Processing the call keyword arguments (line 197)
        kwargs_24910 = {}
        # Getting the type of 'self' (line 197)
        self_24906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'self', False)
        # Obtaining the member 'imported_packages' of a type (line 197)
        imported_packages_24907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 12), self_24906, 'imported_packages')
        # Obtaining the member 'append' of a type (line 197)
        append_24908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 12), imported_packages_24907, 'append')
        # Calling append(args, kwargs) (line 197)
        append_call_result_24911 = invoke(stypy.reporting.localization.Localization(__file__, 197, 12), append_24908, *[package_name_24909], **kwargs_24910)
        
        
        
        # Getting the type of 'verbose' (line 199)
        verbose_24912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 15), 'verbose')
        int_24913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 24), 'int')
        # Applying the binary operator '!=' (line 199)
        result_ne_24914 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 15), '!=', verbose_24912, int_24913)
        
        # Testing the type of an if condition (line 199)
        if_condition_24915 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 12), result_ne_24914)
        # Assigning a type to the variable 'if_condition_24915' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'if_condition_24915', if_condition_24915)
        # SSA begins for if statement (line 199)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 200):
        
        # Call to get(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'package_name' (line 200)
        package_name_24919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 48), 'package_name', False)
        # Processing the call keyword arguments (line 200)
        kwargs_24920 = {}
        # Getting the type of 'frame' (line 200)
        frame_24916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 29), 'frame', False)
        # Obtaining the member 'f_locals' of a type (line 200)
        f_locals_24917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 29), frame_24916, 'f_locals')
        # Obtaining the member 'get' of a type (line 200)
        get_24918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 29), f_locals_24917, 'get')
        # Calling get(args, kwargs) (line 200)
        get_call_result_24921 = invoke(stypy.reporting.localization.Localization(__file__, 200, 29), get_24918, *[package_name_24919], **kwargs_24920)
        
        # Assigning a type to the variable 'new_object' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 16), 'new_object', get_call_result_24921)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'old_object' (line 201)
        old_object_24922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 19), 'old_object')
        # Getting the type of 'None' (line 201)
        None_24923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 37), 'None')
        # Applying the binary operator 'isnot' (line 201)
        result_is_not_24924 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 19), 'isnot', old_object_24922, None_24923)
        
        
        # Getting the type of 'old_object' (line 201)
        old_object_24925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 46), 'old_object')
        # Getting the type of 'new_object' (line 201)
        new_object_24926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 64), 'new_object')
        # Applying the binary operator 'isnot' (line 201)
        result_is_not_24927 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 46), 'isnot', old_object_24925, new_object_24926)
        
        # Applying the binary operator 'and' (line 201)
        result_and_keyword_24928 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 19), 'and', result_is_not_24924, result_is_not_24927)
        
        # Testing the type of an if condition (line 201)
        if_condition_24929 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 16), result_and_keyword_24928)
        # Assigning a type to the variable 'if_condition_24929' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'if_condition_24929', if_condition_24929)
        # SSA begins for if statement (line 201)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 202)
        # Processing the call arguments (line 202)
        str_24932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 30), 'str', 'Overwriting %s=%s (was %s)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 203)
        tuple_24933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 203)
        # Adding element type (line 203)
        # Getting the type of 'package_name' (line 203)
        package_name_24934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 33), 'package_name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 33), tuple_24933, package_name_24934)
        # Adding element type (line 203)
        
        # Call to _obj2repr(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'new_object' (line 203)
        new_object_24937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 62), 'new_object', False)
        # Processing the call keyword arguments (line 203)
        kwargs_24938 = {}
        # Getting the type of 'self' (line 203)
        self_24935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 47), 'self', False)
        # Obtaining the member '_obj2repr' of a type (line 203)
        _obj2repr_24936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 47), self_24935, '_obj2repr')
        # Calling _obj2repr(args, kwargs) (line 203)
        _obj2repr_call_result_24939 = invoke(stypy.reporting.localization.Localization(__file__, 203, 47), _obj2repr_24936, *[new_object_24937], **kwargs_24938)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 33), tuple_24933, _obj2repr_call_result_24939)
        # Adding element type (line 203)
        
        # Call to _obj2repr(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'old_object' (line 204)
        old_object_24942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 48), 'old_object', False)
        # Processing the call keyword arguments (line 204)
        kwargs_24943 = {}
        # Getting the type of 'self' (line 204)
        self_24940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 33), 'self', False)
        # Obtaining the member '_obj2repr' of a type (line 204)
        _obj2repr_24941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 33), self_24940, '_obj2repr')
        # Calling _obj2repr(args, kwargs) (line 204)
        _obj2repr_call_result_24944 = invoke(stypy.reporting.localization.Localization(__file__, 204, 33), _obj2repr_24941, *[old_object_24942], **kwargs_24943)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 33), tuple_24933, _obj2repr_call_result_24944)
        
        # Applying the binary operator '%' (line 202)
        result_mod_24945 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 30), '%', str_24932, tuple_24933)
        
        # Processing the call keyword arguments (line 202)
        kwargs_24946 = {}
        # Getting the type of 'self' (line 202)
        self_24930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 20), 'self', False)
        # Obtaining the member 'warn' of a type (line 202)
        warn_24931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 20), self_24930, 'warn')
        # Calling warn(args, kwargs) (line 202)
        warn_call_result_24947 = invoke(stypy.reporting.localization.Localization(__file__, 202, 20), warn_24931, *[result_mod_24945], **kwargs_24946)
        
        # SSA join for if statement (line 201)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 199)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        str_24948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 15), 'str', '.')
        # Getting the type of 'package_name' (line 206)
        package_name_24949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'package_name')
        # Applying the binary operator 'notin' (line 206)
        result_contains_24950 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 15), 'notin', str_24948, package_name_24949)
        
        # Testing the type of an if condition (line 206)
        if_condition_24951 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 12), result_contains_24950)
        # Assigning a type to the variable 'if_condition_24951' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'if_condition_24951', if_condition_24951)
        # SSA begins for if statement (line 206)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'package_name' (line 207)
        package_name_24955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 48), 'package_name', False)
        # Processing the call keyword arguments (line 207)
        kwargs_24956 = {}
        # Getting the type of 'self' (line 207)
        self_24952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'self', False)
        # Obtaining the member 'parent_export_names' of a type (line 207)
        parent_export_names_24953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 16), self_24952, 'parent_export_names')
        # Obtaining the member 'append' of a type (line 207)
        append_24954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 16), parent_export_names_24953, 'append')
        # Calling append(args, kwargs) (line 207)
        append_call_result_24957 = invoke(stypy.reporting.localization.Localization(__file__, 207, 16), append_24954, *[package_name_24955], **kwargs_24956)
        
        # SSA join for if statement (line 206)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'global_symbols' (line 209)
        global_symbols_24958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 26), 'global_symbols')
        # Testing the type of a for loop iterable (line 209)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 209, 12), global_symbols_24958)
        # Getting the type of the for loop variable (line 209)
        for_loop_var_24959 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 209, 12), global_symbols_24958)
        # Assigning a type to the variable 'symbol' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'symbol', for_loop_var_24959)
        # SSA begins for a for statement (line 209)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'symbol' (line 210)
        symbol_24960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 19), 'symbol')
        str_24961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 27), 'str', '*')
        # Applying the binary operator '==' (line 210)
        result_eq_24962 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 19), '==', symbol_24960, str_24961)
        
        # Testing the type of an if condition (line 210)
        if_condition_24963 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 16), result_eq_24962)
        # Assigning a type to the variable 'if_condition_24963' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'if_condition_24963', if_condition_24963)
        # SSA begins for if statement (line 210)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 211):
        
        # Call to eval(...): (line 211)
        # Processing the call arguments (line 211)
        str_24965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 35), 'str', 'getattr(%s,"__all__",None)')
        # Getting the type of 'package_name' (line 212)
        package_name_24966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 38), 'package_name', False)
        # Applying the binary operator '%' (line 211)
        result_mod_24967 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 35), '%', str_24965, package_name_24966)
        
        # Getting the type of 'frame' (line 213)
        frame_24968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 35), 'frame', False)
        # Obtaining the member 'f_globals' of a type (line 213)
        f_globals_24969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 35), frame_24968, 'f_globals')
        # Getting the type of 'frame' (line 213)
        frame_24970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 52), 'frame', False)
        # Obtaining the member 'f_locals' of a type (line 213)
        f_locals_24971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 52), frame_24970, 'f_locals')
        # Processing the call keyword arguments (line 211)
        kwargs_24972 = {}
        # Getting the type of 'eval' (line 211)
        eval_24964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 30), 'eval', False)
        # Calling eval(args, kwargs) (line 211)
        eval_call_result_24973 = invoke(stypy.reporting.localization.Localization(__file__, 211, 30), eval_24964, *[result_mod_24967, f_globals_24969, f_locals_24971], **kwargs_24972)
        
        # Assigning a type to the variable 'symbols' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 20), 'symbols', eval_call_result_24973)
        
        # Type idiom detected: calculating its left and rigth part (line 214)
        # Getting the type of 'symbols' (line 214)
        symbols_24974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 23), 'symbols')
        # Getting the type of 'None' (line 214)
        None_24975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 34), 'None')
        
        (may_be_24976, more_types_in_union_24977) = may_be_none(symbols_24974, None_24975)

        if may_be_24976:

            if more_types_in_union_24977:
                # Runtime conditional SSA (line 214)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 215):
            
            # Call to eval(...): (line 215)
            # Processing the call arguments (line 215)
            str_24979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 39), 'str', 'dir(%s)')
            # Getting the type of 'package_name' (line 215)
            package_name_24980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 52), 'package_name', False)
            # Applying the binary operator '%' (line 215)
            result_mod_24981 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 39), '%', str_24979, package_name_24980)
            
            # Getting the type of 'frame' (line 216)
            frame_24982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 39), 'frame', False)
            # Obtaining the member 'f_globals' of a type (line 216)
            f_globals_24983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 39), frame_24982, 'f_globals')
            # Getting the type of 'frame' (line 216)
            frame_24984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 56), 'frame', False)
            # Obtaining the member 'f_locals' of a type (line 216)
            f_locals_24985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 56), frame_24984, 'f_locals')
            # Processing the call keyword arguments (line 215)
            kwargs_24986 = {}
            # Getting the type of 'eval' (line 215)
            eval_24978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 34), 'eval', False)
            # Calling eval(args, kwargs) (line 215)
            eval_call_result_24987 = invoke(stypy.reporting.localization.Localization(__file__, 215, 34), eval_24978, *[result_mod_24981, f_globals_24983, f_locals_24985], **kwargs_24986)
            
            # Assigning a type to the variable 'symbols' (line 215)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 24), 'symbols', eval_call_result_24987)
            
            # Assigning a ListComp to a Name (line 217):
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'symbols' (line 217)
            symbols_24995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 46), 'symbols')
            comprehension_24996 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 35), symbols_24995)
            # Assigning a type to the variable 's' (line 217)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 35), 's', comprehension_24996)
            
            
            # Call to startswith(...): (line 217)
            # Processing the call arguments (line 217)
            str_24991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 74), 'str', '_')
            # Processing the call keyword arguments (line 217)
            kwargs_24992 = {}
            # Getting the type of 's' (line 217)
            s_24989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 61), 's', False)
            # Obtaining the member 'startswith' of a type (line 217)
            startswith_24990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 61), s_24989, 'startswith')
            # Calling startswith(args, kwargs) (line 217)
            startswith_call_result_24993 = invoke(stypy.reporting.localization.Localization(__file__, 217, 61), startswith_24990, *[str_24991], **kwargs_24992)
            
            # Applying the 'not' unary operator (line 217)
            result_not__24994 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 57), 'not', startswith_call_result_24993)
            
            # Getting the type of 's' (line 217)
            s_24988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 35), 's')
            list_24997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 35), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 35), list_24997, s_24988)
            # Assigning a type to the variable 'symbols' (line 217)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 24), 'symbols', list_24997)

            if more_types_in_union_24977:
                # SSA join for if statement (line 214)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA branch for the else part of an if statement (line 210)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 219):
        
        # Obtaining an instance of the builtin type 'list' (line 219)
        list_24998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 219)
        # Adding element type (line 219)
        # Getting the type of 'symbol' (line 219)
        symbol_24999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 31), 'symbol')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 30), list_24998, symbol_24999)
        
        # Assigning a type to the variable 'symbols' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 20), 'symbols', list_24998)
        # SSA join for if statement (line 210)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'verbose' (line 221)
        verbose_25000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 19), 'verbose')
        int_25001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 28), 'int')
        # Applying the binary operator '!=' (line 221)
        result_ne_25002 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 19), '!=', verbose_25000, int_25001)
        
        # Testing the type of an if condition (line 221)
        if_condition_25003 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 16), result_ne_25002)
        # Assigning a type to the variable 'if_condition_25003' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'if_condition_25003', if_condition_25003)
        # SSA begins for if statement (line 221)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Dict to a Name (line 222):
        
        # Obtaining an instance of the builtin type 'dict' (line 222)
        dict_25004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 34), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 222)
        
        # Assigning a type to the variable 'old_objects' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'old_objects', dict_25004)
        
        # Getting the type of 'symbols' (line 223)
        symbols_25005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 29), 'symbols')
        # Testing the type of a for loop iterable (line 223)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 223, 20), symbols_25005)
        # Getting the type of the for loop variable (line 223)
        for_loop_var_25006 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 223, 20), symbols_25005)
        # Assigning a type to the variable 's' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 20), 's', for_loop_var_25006)
        # SSA begins for a for statement (line 223)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 's' (line 224)
        s_25007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 27), 's')
        # Getting the type of 'frame' (line 224)
        frame_25008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 32), 'frame')
        # Obtaining the member 'f_locals' of a type (line 224)
        f_locals_25009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 32), frame_25008, 'f_locals')
        # Applying the binary operator 'in' (line 224)
        result_contains_25010 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 27), 'in', s_25007, f_locals_25009)
        
        # Testing the type of an if condition (line 224)
        if_condition_25011 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 224, 24), result_contains_25010)
        # Assigning a type to the variable 'if_condition_25011' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 24), 'if_condition_25011', if_condition_25011)
        # SSA begins for if statement (line 224)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Subscript (line 225):
        
        # Obtaining the type of the subscript
        # Getting the type of 's' (line 225)
        s_25012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 60), 's')
        # Getting the type of 'frame' (line 225)
        frame_25013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 45), 'frame')
        # Obtaining the member 'f_locals' of a type (line 225)
        f_locals_25014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 45), frame_25013, 'f_locals')
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___25015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 45), f_locals_25014, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 225)
        subscript_call_result_25016 = invoke(stypy.reporting.localization.Localization(__file__, 225, 45), getitem___25015, s_25012)
        
        # Getting the type of 'old_objects' (line 225)
        old_objects_25017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 28), 'old_objects')
        # Getting the type of 's' (line 225)
        s_25018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 40), 's')
        # Storing an element on a container (line 225)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 28), old_objects_25017, (s_25018, subscript_call_result_25016))
        # SSA join for if statement (line 224)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 221)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 227):
        str_25019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 25), 'str', 'from ')
        # Getting the type of 'package_name' (line 227)
        package_name_25020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 33), 'package_name')
        # Applying the binary operator '+' (line 227)
        result_add_25021 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 25), '+', str_25019, package_name_25020)
        
        str_25022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 46), 'str', ' import ')
        # Applying the binary operator '+' (line 227)
        result_add_25023 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 45), '+', result_add_25021, str_25022)
        
        # Getting the type of 'symbol' (line 227)
        symbol_25024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 57), 'symbol')
        # Applying the binary operator '+' (line 227)
        result_add_25025 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 56), '+', result_add_25023, symbol_25024)
        
        # Assigning a type to the variable 'cmdstr' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'cmdstr', result_add_25025)
        
        
        # Call to _execcmd(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'cmdstr' (line 228)
        cmdstr_25028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 33), 'cmdstr', False)
        # Processing the call keyword arguments (line 228)
        kwargs_25029 = {}
        # Getting the type of 'self' (line 228)
        self_25026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 19), 'self', False)
        # Obtaining the member '_execcmd' of a type (line 228)
        _execcmd_25027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 19), self_25026, '_execcmd')
        # Calling _execcmd(args, kwargs) (line 228)
        _execcmd_call_result_25030 = invoke(stypy.reporting.localization.Localization(__file__, 228, 19), _execcmd_25027, *[cmdstr_25028], **kwargs_25029)
        
        # Testing the type of an if condition (line 228)
        if_condition_25031 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 16), _execcmd_call_result_25030)
        # Assigning a type to the variable 'if_condition_25031' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'if_condition_25031', if_condition_25031)
        # SSA begins for if statement (line 228)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 228)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'verbose' (line 231)
        verbose_25032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 19), 'verbose')
        int_25033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 28), 'int')
        # Applying the binary operator '!=' (line 231)
        result_ne_25034 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 19), '!=', verbose_25032, int_25033)
        
        # Testing the type of an if condition (line 231)
        if_condition_25035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 231, 16), result_ne_25034)
        # Assigning a type to the variable 'if_condition_25035' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'if_condition_25035', if_condition_25035)
        # SSA begins for if statement (line 231)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to items(...): (line 232)
        # Processing the call keyword arguments (line 232)
        kwargs_25038 = {}
        # Getting the type of 'old_objects' (line 232)
        old_objects_25036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 41), 'old_objects', False)
        # Obtaining the member 'items' of a type (line 232)
        items_25037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 41), old_objects_25036, 'items')
        # Calling items(args, kwargs) (line 232)
        items_call_result_25039 = invoke(stypy.reporting.localization.Localization(__file__, 232, 41), items_25037, *[], **kwargs_25038)
        
        # Testing the type of a for loop iterable (line 232)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 232, 20), items_call_result_25039)
        # Getting the type of the for loop variable (line 232)
        for_loop_var_25040 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 232, 20), items_call_result_25039)
        # Assigning a type to the variable 's' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 20), 's', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 20), for_loop_var_25040))
        # Assigning a type to the variable 'old_object' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 20), 'old_object', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 20), for_loop_var_25040))
        # SSA begins for a for statement (line 232)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 233):
        
        # Obtaining the type of the subscript
        # Getting the type of 's' (line 233)
        s_25041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 52), 's')
        # Getting the type of 'frame' (line 233)
        frame_25042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 37), 'frame')
        # Obtaining the member 'f_locals' of a type (line 233)
        f_locals_25043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 37), frame_25042, 'f_locals')
        # Obtaining the member '__getitem__' of a type (line 233)
        getitem___25044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 37), f_locals_25043, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 233)
        subscript_call_result_25045 = invoke(stypy.reporting.localization.Localization(__file__, 233, 37), getitem___25044, s_25041)
        
        # Assigning a type to the variable 'new_object' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 24), 'new_object', subscript_call_result_25045)
        
        
        # Getting the type of 'new_object' (line 234)
        new_object_25046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 27), 'new_object')
        # Getting the type of 'old_object' (line 234)
        old_object_25047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 45), 'old_object')
        # Applying the binary operator 'isnot' (line 234)
        result_is_not_25048 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 27), 'isnot', new_object_25046, old_object_25047)
        
        # Testing the type of an if condition (line 234)
        if_condition_25049 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 234, 24), result_is_not_25048)
        # Assigning a type to the variable 'if_condition_25049' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 24), 'if_condition_25049', if_condition_25049)
        # SSA begins for if statement (line 234)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 235)
        # Processing the call arguments (line 235)
        str_25052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 38), 'str', 'Overwriting %s=%s (was %s)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 236)
        tuple_25053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 236)
        # Adding element type (line 236)
        # Getting the type of 's' (line 236)
        s_25054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 41), 's', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 41), tuple_25053, s_25054)
        # Adding element type (line 236)
        
        # Call to _obj2repr(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'new_object' (line 236)
        new_object_25057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 59), 'new_object', False)
        # Processing the call keyword arguments (line 236)
        kwargs_25058 = {}
        # Getting the type of 'self' (line 236)
        self_25055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 44), 'self', False)
        # Obtaining the member '_obj2repr' of a type (line 236)
        _obj2repr_25056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 44), self_25055, '_obj2repr')
        # Calling _obj2repr(args, kwargs) (line 236)
        _obj2repr_call_result_25059 = invoke(stypy.reporting.localization.Localization(__file__, 236, 44), _obj2repr_25056, *[new_object_25057], **kwargs_25058)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 41), tuple_25053, _obj2repr_call_result_25059)
        # Adding element type (line 236)
        
        # Call to _obj2repr(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'old_object' (line 237)
        old_object_25062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 56), 'old_object', False)
        # Processing the call keyword arguments (line 237)
        kwargs_25063 = {}
        # Getting the type of 'self' (line 237)
        self_25060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 41), 'self', False)
        # Obtaining the member '_obj2repr' of a type (line 237)
        _obj2repr_25061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 41), self_25060, '_obj2repr')
        # Calling _obj2repr(args, kwargs) (line 237)
        _obj2repr_call_result_25064 = invoke(stypy.reporting.localization.Localization(__file__, 237, 41), _obj2repr_25061, *[old_object_25062], **kwargs_25063)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 41), tuple_25053, _obj2repr_call_result_25064)
        
        # Applying the binary operator '%' (line 235)
        result_mod_25065 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 38), '%', str_25052, tuple_25053)
        
        # Processing the call keyword arguments (line 235)
        kwargs_25066 = {}
        # Getting the type of 'self' (line 235)
        self_25050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 28), 'self', False)
        # Obtaining the member 'warn' of a type (line 235)
        warn_25051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 28), self_25050, 'warn')
        # Calling warn(args, kwargs) (line 235)
        warn_call_result_25067 = invoke(stypy.reporting.localization.Localization(__file__, 235, 28), warn_25051, *[result_mod_25065], **kwargs_25066)
        
        # SSA join for if statement (line 234)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 231)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'symbol' (line 239)
        symbol_25068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 19), 'symbol')
        str_25069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 27), 'str', '*')
        # Applying the binary operator '==' (line 239)
        result_eq_25070 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 19), '==', symbol_25068, str_25069)
        
        # Testing the type of an if condition (line 239)
        if_condition_25071 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 16), result_eq_25070)
        # Assigning a type to the variable 'if_condition_25071' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), 'if_condition_25071', if_condition_25071)
        # SSA begins for if statement (line 239)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 'symbols' (line 240)
        symbols_25075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 52), 'symbols', False)
        # Processing the call keyword arguments (line 240)
        kwargs_25076 = {}
        # Getting the type of 'self' (line 240)
        self_25072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), 'self', False)
        # Obtaining the member 'parent_export_names' of a type (line 240)
        parent_export_names_25073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 20), self_25072, 'parent_export_names')
        # Obtaining the member 'extend' of a type (line 240)
        extend_25074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 20), parent_export_names_25073, 'extend')
        # Calling extend(args, kwargs) (line 240)
        extend_call_result_25077 = invoke(stypy.reporting.localization.Localization(__file__, 240, 20), extend_25074, *[symbols_25075], **kwargs_25076)
        
        # SSA branch for the else part of an if statement (line 239)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'symbol' (line 242)
        symbol_25081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 52), 'symbol', False)
        # Processing the call keyword arguments (line 242)
        kwargs_25082 = {}
        # Getting the type of 'self' (line 242)
        self_25078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 20), 'self', False)
        # Obtaining the member 'parent_export_names' of a type (line 242)
        parent_export_names_25079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 20), self_25078, 'parent_export_names')
        # Obtaining the member 'append' of a type (line 242)
        append_25080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 20), parent_export_names_25079, 'append')
        # Calling append(args, kwargs) (line 242)
        append_call_result_25083 = invoke(stypy.reporting.localization.Localization(__file__, 242, 20), append_25080, *[symbol_25081], **kwargs_25082)
        
        # SSA join for if statement (line 239)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Assigning a type to the variable 'stypy_return_type' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_25084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25084)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_25084


    @norecursion
    def _execcmd(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_execcmd'
        module_type_store = module_type_store.open_function_context('_execcmd', 246, 4, False)
        # Assigning a type to the variable 'self' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PackageLoader._execcmd.__dict__.__setitem__('stypy_localization', localization)
        PackageLoader._execcmd.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PackageLoader._execcmd.__dict__.__setitem__('stypy_type_store', module_type_store)
        PackageLoader._execcmd.__dict__.__setitem__('stypy_function_name', 'PackageLoader._execcmd')
        PackageLoader._execcmd.__dict__.__setitem__('stypy_param_names_list', ['cmdstr'])
        PackageLoader._execcmd.__dict__.__setitem__('stypy_varargs_param_name', None)
        PackageLoader._execcmd.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PackageLoader._execcmd.__dict__.__setitem__('stypy_call_defaults', defaults)
        PackageLoader._execcmd.__dict__.__setitem__('stypy_call_varargs', varargs)
        PackageLoader._execcmd.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PackageLoader._execcmd.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PackageLoader._execcmd', ['cmdstr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_execcmd', localization, ['cmdstr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_execcmd(...)' code ##################

        str_25085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 8), 'str', ' Execute command in parent_frame.')
        
        # Assigning a Attribute to a Name (line 248):
        # Getting the type of 'self' (line 248)
        self_25086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'self')
        # Obtaining the member 'parent_frame' of a type (line 248)
        parent_frame_25087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 16), self_25086, 'parent_frame')
        # Assigning a type to the variable 'frame' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'frame', parent_frame_25087)
        
        
        # SSA begins for try-except statement (line 249)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        # Dynamic code evaluation using an exec statement
        # Getting the type of 'cmdstr' (line 250)
        cmdstr_25088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 18), 'cmdstr')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 250, 12), cmdstr_25088, 'exec parameter', 'StringType', 'FileType', 'CodeType')
        enable_usage_of_dynamic_types_warning(stypy.reporting.localization.Localization(__file__, 250, 12))
        # SSA branch for the except part of a try statement (line 249)
        # SSA branch for the except 'Exception' branch of a try statement (line 249)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'Exception' (line 251)
        Exception_25089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 'Exception')
        # Assigning a type to the variable 'msg' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'msg', Exception_25089)
        
        # Call to error(...): (line 252)
        # Processing the call arguments (line 252)
        str_25092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 23), 'str', '%s -> failed: %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 252)
        tuple_25093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 252)
        # Adding element type (line 252)
        # Getting the type of 'cmdstr' (line 252)
        cmdstr_25094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 45), 'cmdstr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 45), tuple_25093, cmdstr_25094)
        # Adding element type (line 252)
        # Getting the type of 'msg' (line 252)
        msg_25095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 53), 'msg', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 45), tuple_25093, msg_25095)
        
        # Applying the binary operator '%' (line 252)
        result_mod_25096 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 23), '%', str_25092, tuple_25093)
        
        # Processing the call keyword arguments (line 252)
        kwargs_25097 = {}
        # Getting the type of 'self' (line 252)
        self_25090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'self', False)
        # Obtaining the member 'error' of a type (line 252)
        error_25091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 12), self_25090, 'error')
        # Calling error(args, kwargs) (line 252)
        error_call_result_25098 = invoke(stypy.reporting.localization.Localization(__file__, 252, 12), error_25091, *[result_mod_25096], **kwargs_25097)
        
        # Getting the type of 'True' (line 253)
        True_25099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'stypy_return_type', True_25099)
        # SSA branch for the else branch of a try statement (line 249)
        module_type_store.open_ssa_branch('except else')
        
        # Call to log(...): (line 255)
        # Processing the call arguments (line 255)
        str_25102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 21), 'str', '%s -> success')
        # Getting the type of 'cmdstr' (line 255)
        cmdstr_25103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 40), 'cmdstr', False)
        # Applying the binary operator '%' (line 255)
        result_mod_25104 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 21), '%', str_25102, cmdstr_25103)
        
        # Processing the call keyword arguments (line 255)
        kwargs_25105 = {}
        # Getting the type of 'self' (line 255)
        self_25100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'self', False)
        # Obtaining the member 'log' of a type (line 255)
        log_25101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 12), self_25100, 'log')
        # Calling log(args, kwargs) (line 255)
        log_call_result_25106 = invoke(stypy.reporting.localization.Localization(__file__, 255, 12), log_25101, *[result_mod_25104], **kwargs_25105)
        
        # SSA join for try-except statement (line 249)
        module_type_store = module_type_store.join_ssa_context()
        
        # Assigning a type to the variable 'stypy_return_type' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of '_execcmd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_execcmd' in the type store
        # Getting the type of 'stypy_return_type' (line 246)
        stypy_return_type_25107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25107)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_execcmd'
        return stypy_return_type_25107


    @norecursion
    def _obj2repr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_obj2repr'
        module_type_store = module_type_store.open_function_context('_obj2repr', 258, 4, False)
        # Assigning a type to the variable 'self' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PackageLoader._obj2repr.__dict__.__setitem__('stypy_localization', localization)
        PackageLoader._obj2repr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PackageLoader._obj2repr.__dict__.__setitem__('stypy_type_store', module_type_store)
        PackageLoader._obj2repr.__dict__.__setitem__('stypy_function_name', 'PackageLoader._obj2repr')
        PackageLoader._obj2repr.__dict__.__setitem__('stypy_param_names_list', ['obj'])
        PackageLoader._obj2repr.__dict__.__setitem__('stypy_varargs_param_name', None)
        PackageLoader._obj2repr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PackageLoader._obj2repr.__dict__.__setitem__('stypy_call_defaults', defaults)
        PackageLoader._obj2repr.__dict__.__setitem__('stypy_call_varargs', varargs)
        PackageLoader._obj2repr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PackageLoader._obj2repr.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PackageLoader._obj2repr', ['obj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_obj2repr', localization, ['obj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_obj2repr(...)' code ##################

        str_25108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 8), 'str', ' Return repr(obj) with')
        
        # Assigning a Call to a Name (line 260):
        
        # Call to getattr(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'obj' (line 260)
        obj_25110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 25), 'obj', False)
        str_25111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 30), 'str', '__module__')
        # Getting the type of 'None' (line 260)
        None_25112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 44), 'None', False)
        # Processing the call keyword arguments (line 260)
        kwargs_25113 = {}
        # Getting the type of 'getattr' (line 260)
        getattr_25109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 17), 'getattr', False)
        # Calling getattr(args, kwargs) (line 260)
        getattr_call_result_25114 = invoke(stypy.reporting.localization.Localization(__file__, 260, 17), getattr_25109, *[obj_25110, str_25111, None_25112], **kwargs_25113)
        
        # Assigning a type to the variable 'module' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'module', getattr_call_result_25114)
        
        # Assigning a Call to a Name (line 261):
        
        # Call to getattr(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'obj' (line 261)
        obj_25116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 23), 'obj', False)
        str_25117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 28), 'str', '__file__')
        # Getting the type of 'None' (line 261)
        None_25118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 40), 'None', False)
        # Processing the call keyword arguments (line 261)
        kwargs_25119 = {}
        # Getting the type of 'getattr' (line 261)
        getattr_25115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 261)
        getattr_call_result_25120 = invoke(stypy.reporting.localization.Localization(__file__, 261, 15), getattr_25115, *[obj_25116, str_25117, None_25118], **kwargs_25119)
        
        # Assigning a type to the variable 'file' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'file', getattr_call_result_25120)
        
        # Type idiom detected: calculating its left and rigth part (line 262)
        # Getting the type of 'module' (line 262)
        module_25121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'module')
        # Getting the type of 'None' (line 262)
        None_25122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 25), 'None')
        
        (may_be_25123, more_types_in_union_25124) = may_not_be_none(module_25121, None_25122)

        if may_be_25123:

            if more_types_in_union_25124:
                # Runtime conditional SSA (line 262)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to repr(...): (line 263)
            # Processing the call arguments (line 263)
            # Getting the type of 'obj' (line 263)
            obj_25126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 24), 'obj', False)
            # Processing the call keyword arguments (line 263)
            kwargs_25127 = {}
            # Getting the type of 'repr' (line 263)
            repr_25125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 19), 'repr', False)
            # Calling repr(args, kwargs) (line 263)
            repr_call_result_25128 = invoke(stypy.reporting.localization.Localization(__file__, 263, 19), repr_25125, *[obj_25126], **kwargs_25127)
            
            str_25129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 31), 'str', ' from ')
            # Applying the binary operator '+' (line 263)
            result_add_25130 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 19), '+', repr_call_result_25128, str_25129)
            
            # Getting the type of 'module' (line 263)
            module_25131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 42), 'module')
            # Applying the binary operator '+' (line 263)
            result_add_25132 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 40), '+', result_add_25130, module_25131)
            
            # Assigning a type to the variable 'stypy_return_type' (line 263)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'stypy_return_type', result_add_25132)

            if more_types_in_union_25124:
                # SSA join for if statement (line 262)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 264)
        # Getting the type of 'file' (line 264)
        file_25133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'file')
        # Getting the type of 'None' (line 264)
        None_25134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 23), 'None')
        
        (may_be_25135, more_types_in_union_25136) = may_not_be_none(file_25133, None_25134)

        if may_be_25135:

            if more_types_in_union_25136:
                # Runtime conditional SSA (line 264)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to repr(...): (line 265)
            # Processing the call arguments (line 265)
            # Getting the type of 'obj' (line 265)
            obj_25138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 24), 'obj', False)
            # Processing the call keyword arguments (line 265)
            kwargs_25139 = {}
            # Getting the type of 'repr' (line 265)
            repr_25137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 19), 'repr', False)
            # Calling repr(args, kwargs) (line 265)
            repr_call_result_25140 = invoke(stypy.reporting.localization.Localization(__file__, 265, 19), repr_25137, *[obj_25138], **kwargs_25139)
            
            str_25141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 31), 'str', ' from ')
            # Applying the binary operator '+' (line 265)
            result_add_25142 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 19), '+', repr_call_result_25140, str_25141)
            
            # Getting the type of 'file' (line 265)
            file_25143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 42), 'file')
            # Applying the binary operator '+' (line 265)
            result_add_25144 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 40), '+', result_add_25142, file_25143)
            
            # Assigning a type to the variable 'stypy_return_type' (line 265)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'stypy_return_type', result_add_25144)

            if more_types_in_union_25136:
                # SSA join for if statement (line 264)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to repr(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'obj' (line 266)
        obj_25146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 20), 'obj', False)
        # Processing the call keyword arguments (line 266)
        kwargs_25147 = {}
        # Getting the type of 'repr' (line 266)
        repr_25145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 15), 'repr', False)
        # Calling repr(args, kwargs) (line 266)
        repr_call_result_25148 = invoke(stypy.reporting.localization.Localization(__file__, 266, 15), repr_25145, *[obj_25146], **kwargs_25147)
        
        # Assigning a type to the variable 'stypy_return_type' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'stypy_return_type', repr_call_result_25148)
        
        # ################# End of '_obj2repr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_obj2repr' in the type store
        # Getting the type of 'stypy_return_type' (line 258)
        stypy_return_type_25149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25149)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_obj2repr'
        return stypy_return_type_25149


    @norecursion
    def log(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'log'
        module_type_store = module_type_store.open_function_context('log', 268, 4, False)
        # Assigning a type to the variable 'self' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PackageLoader.log.__dict__.__setitem__('stypy_localization', localization)
        PackageLoader.log.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PackageLoader.log.__dict__.__setitem__('stypy_type_store', module_type_store)
        PackageLoader.log.__dict__.__setitem__('stypy_function_name', 'PackageLoader.log')
        PackageLoader.log.__dict__.__setitem__('stypy_param_names_list', ['mess'])
        PackageLoader.log.__dict__.__setitem__('stypy_varargs_param_name', None)
        PackageLoader.log.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PackageLoader.log.__dict__.__setitem__('stypy_call_defaults', defaults)
        PackageLoader.log.__dict__.__setitem__('stypy_call_varargs', varargs)
        PackageLoader.log.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PackageLoader.log.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PackageLoader.log', ['mess'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'log', localization, ['mess'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'log(...)' code ##################

        
        
        # Getting the type of 'self' (line 269)
        self_25150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 11), 'self')
        # Obtaining the member 'verbose' of a type (line 269)
        verbose_25151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 11), self_25150, 'verbose')
        int_25152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 24), 'int')
        # Applying the binary operator '>' (line 269)
        result_gt_25153 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 11), '>', verbose_25151, int_25152)
        
        # Testing the type of an if condition (line 269)
        if_condition_25154 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 8), result_gt_25153)
        # Assigning a type to the variable 'if_condition_25154' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'if_condition_25154', if_condition_25154)
        # SSA begins for if statement (line 269)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 270)
        # Processing the call arguments (line 270)
        
        # Call to str(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'mess' (line 270)
        mess_25157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 22), 'mess', False)
        # Processing the call keyword arguments (line 270)
        kwargs_25158 = {}
        # Getting the type of 'str' (line 270)
        str_25156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 18), 'str', False)
        # Calling str(args, kwargs) (line 270)
        str_call_result_25159 = invoke(stypy.reporting.localization.Localization(__file__, 270, 18), str_25156, *[mess_25157], **kwargs_25158)
        
        # Processing the call keyword arguments (line 270)
        # Getting the type of 'sys' (line 270)
        sys_25160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 34), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 270)
        stderr_25161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 34), sys_25160, 'stderr')
        keyword_25162 = stderr_25161
        kwargs_25163 = {'file': keyword_25162}
        # Getting the type of 'print' (line 270)
        print_25155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'print', False)
        # Calling print(args, kwargs) (line 270)
        print_call_result_25164 = invoke(stypy.reporting.localization.Localization(__file__, 270, 12), print_25155, *[str_call_result_25159], **kwargs_25163)
        
        # SSA join for if statement (line 269)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'log(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'log' in the type store
        # Getting the type of 'stypy_return_type' (line 268)
        stypy_return_type_25165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25165)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'log'
        return stypy_return_type_25165


    @norecursion
    def warn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'warn'
        module_type_store = module_type_store.open_function_context('warn', 271, 4, False)
        # Assigning a type to the variable 'self' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PackageLoader.warn.__dict__.__setitem__('stypy_localization', localization)
        PackageLoader.warn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PackageLoader.warn.__dict__.__setitem__('stypy_type_store', module_type_store)
        PackageLoader.warn.__dict__.__setitem__('stypy_function_name', 'PackageLoader.warn')
        PackageLoader.warn.__dict__.__setitem__('stypy_param_names_list', ['mess'])
        PackageLoader.warn.__dict__.__setitem__('stypy_varargs_param_name', None)
        PackageLoader.warn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PackageLoader.warn.__dict__.__setitem__('stypy_call_defaults', defaults)
        PackageLoader.warn.__dict__.__setitem__('stypy_call_varargs', varargs)
        PackageLoader.warn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PackageLoader.warn.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PackageLoader.warn', ['mess'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'warn', localization, ['mess'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'warn(...)' code ##################

        
        
        # Getting the type of 'self' (line 272)
        self_25166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 11), 'self')
        # Obtaining the member 'verbose' of a type (line 272)
        verbose_25167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 11), self_25166, 'verbose')
        int_25168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 25), 'int')
        # Applying the binary operator '>=' (line 272)
        result_ge_25169 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 11), '>=', verbose_25167, int_25168)
        
        # Testing the type of an if condition (line 272)
        if_condition_25170 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 8), result_ge_25169)
        # Assigning a type to the variable 'if_condition_25170' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'if_condition_25170', if_condition_25170)
        # SSA begins for if statement (line 272)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 273)
        # Processing the call arguments (line 273)
        
        # Call to str(...): (line 273)
        # Processing the call arguments (line 273)
        # Getting the type of 'mess' (line 273)
        mess_25173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 22), 'mess', False)
        # Processing the call keyword arguments (line 273)
        kwargs_25174 = {}
        # Getting the type of 'str' (line 273)
        str_25172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 18), 'str', False)
        # Calling str(args, kwargs) (line 273)
        str_call_result_25175 = invoke(stypy.reporting.localization.Localization(__file__, 273, 18), str_25172, *[mess_25173], **kwargs_25174)
        
        # Processing the call keyword arguments (line 273)
        # Getting the type of 'sys' (line 273)
        sys_25176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 34), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 273)
        stderr_25177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 34), sys_25176, 'stderr')
        keyword_25178 = stderr_25177
        kwargs_25179 = {'file': keyword_25178}
        # Getting the type of 'print' (line 273)
        print_25171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'print', False)
        # Calling print(args, kwargs) (line 273)
        print_call_result_25180 = invoke(stypy.reporting.localization.Localization(__file__, 273, 12), print_25171, *[str_call_result_25175], **kwargs_25179)
        
        # SSA join for if statement (line 272)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'warn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'warn' in the type store
        # Getting the type of 'stypy_return_type' (line 271)
        stypy_return_type_25181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25181)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'warn'
        return stypy_return_type_25181


    @norecursion
    def error(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'error'
        module_type_store = module_type_store.open_function_context('error', 274, 4, False)
        # Assigning a type to the variable 'self' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PackageLoader.error.__dict__.__setitem__('stypy_localization', localization)
        PackageLoader.error.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PackageLoader.error.__dict__.__setitem__('stypy_type_store', module_type_store)
        PackageLoader.error.__dict__.__setitem__('stypy_function_name', 'PackageLoader.error')
        PackageLoader.error.__dict__.__setitem__('stypy_param_names_list', ['mess'])
        PackageLoader.error.__dict__.__setitem__('stypy_varargs_param_name', None)
        PackageLoader.error.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PackageLoader.error.__dict__.__setitem__('stypy_call_defaults', defaults)
        PackageLoader.error.__dict__.__setitem__('stypy_call_varargs', varargs)
        PackageLoader.error.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PackageLoader.error.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PackageLoader.error', ['mess'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'error', localization, ['mess'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'error(...)' code ##################

        
        
        # Getting the type of 'self' (line 275)
        self_25182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 11), 'self')
        # Obtaining the member 'verbose' of a type (line 275)
        verbose_25183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 11), self_25182, 'verbose')
        int_25184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 25), 'int')
        # Applying the binary operator '!=' (line 275)
        result_ne_25185 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 11), '!=', verbose_25183, int_25184)
        
        # Testing the type of an if condition (line 275)
        if_condition_25186 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 8), result_ne_25185)
        # Assigning a type to the variable 'if_condition_25186' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'if_condition_25186', if_condition_25186)
        # SSA begins for if statement (line 275)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 276)
        # Processing the call arguments (line 276)
        
        # Call to str(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'mess' (line 276)
        mess_25189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 22), 'mess', False)
        # Processing the call keyword arguments (line 276)
        kwargs_25190 = {}
        # Getting the type of 'str' (line 276)
        str_25188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 18), 'str', False)
        # Calling str(args, kwargs) (line 276)
        str_call_result_25191 = invoke(stypy.reporting.localization.Localization(__file__, 276, 18), str_25188, *[mess_25189], **kwargs_25190)
        
        # Processing the call keyword arguments (line 276)
        # Getting the type of 'sys' (line 276)
        sys_25192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 34), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 276)
        stderr_25193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 34), sys_25192, 'stderr')
        keyword_25194 = stderr_25193
        kwargs_25195 = {'file': keyword_25194}
        # Getting the type of 'print' (line 276)
        print_25187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'print', False)
        # Calling print(args, kwargs) (line 276)
        print_call_result_25196 = invoke(stypy.reporting.localization.Localization(__file__, 276, 12), print_25187, *[str_call_result_25191], **kwargs_25195)
        
        # SSA join for if statement (line 275)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'error(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'error' in the type store
        # Getting the type of 'stypy_return_type' (line 274)
        stypy_return_type_25197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25197)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'error'
        return stypy_return_type_25197


    @norecursion
    def _get_doc_title(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_doc_title'
        module_type_store = module_type_store.open_function_context('_get_doc_title', 278, 4, False)
        # Assigning a type to the variable 'self' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PackageLoader._get_doc_title.__dict__.__setitem__('stypy_localization', localization)
        PackageLoader._get_doc_title.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PackageLoader._get_doc_title.__dict__.__setitem__('stypy_type_store', module_type_store)
        PackageLoader._get_doc_title.__dict__.__setitem__('stypy_function_name', 'PackageLoader._get_doc_title')
        PackageLoader._get_doc_title.__dict__.__setitem__('stypy_param_names_list', ['info_module'])
        PackageLoader._get_doc_title.__dict__.__setitem__('stypy_varargs_param_name', None)
        PackageLoader._get_doc_title.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PackageLoader._get_doc_title.__dict__.__setitem__('stypy_call_defaults', defaults)
        PackageLoader._get_doc_title.__dict__.__setitem__('stypy_call_varargs', varargs)
        PackageLoader._get_doc_title.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PackageLoader._get_doc_title.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PackageLoader._get_doc_title', ['info_module'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_doc_title', localization, ['info_module'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_doc_title(...)' code ##################

        str_25198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, (-1)), 'str', ' Get the title from a package info.py file.\n        ')
        
        # Assigning a Call to a Name (line 281):
        
        # Call to getattr(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'info_module' (line 281)
        info_module_25200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 24), 'info_module', False)
        str_25201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 37), 'str', '__doc_title__')
        # Getting the type of 'None' (line 281)
        None_25202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 54), 'None', False)
        # Processing the call keyword arguments (line 281)
        kwargs_25203 = {}
        # Getting the type of 'getattr' (line 281)
        getattr_25199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'getattr', False)
        # Calling getattr(args, kwargs) (line 281)
        getattr_call_result_25204 = invoke(stypy.reporting.localization.Localization(__file__, 281, 16), getattr_25199, *[info_module_25200, str_25201, None_25202], **kwargs_25203)
        
        # Assigning a type to the variable 'title' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'title', getattr_call_result_25204)
        
        # Type idiom detected: calculating its left and rigth part (line 282)
        # Getting the type of 'title' (line 282)
        title_25205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'title')
        # Getting the type of 'None' (line 282)
        None_25206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 24), 'None')
        
        (may_be_25207, more_types_in_union_25208) = may_not_be_none(title_25205, None_25206)

        if may_be_25207:

            if more_types_in_union_25208:
                # Runtime conditional SSA (line 282)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'title' (line 283)
            title_25209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 19), 'title')
            # Assigning a type to the variable 'stypy_return_type' (line 283)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'stypy_return_type', title_25209)

            if more_types_in_union_25208:
                # SSA join for if statement (line 282)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 284):
        
        # Call to getattr(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'info_module' (line 284)
        info_module_25211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), 'info_module', False)
        str_25212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 37), 'str', '__doc__')
        # Getting the type of 'None' (line 284)
        None_25213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 48), 'None', False)
        # Processing the call keyword arguments (line 284)
        kwargs_25214 = {}
        # Getting the type of 'getattr' (line 284)
        getattr_25210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'getattr', False)
        # Calling getattr(args, kwargs) (line 284)
        getattr_call_result_25215 = invoke(stypy.reporting.localization.Localization(__file__, 284, 16), getattr_25210, *[info_module_25211, str_25212, None_25213], **kwargs_25214)
        
        # Assigning a type to the variable 'title' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'title', getattr_call_result_25215)
        
        # Type idiom detected: calculating its left and rigth part (line 285)
        # Getting the type of 'title' (line 285)
        title_25216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'title')
        # Getting the type of 'None' (line 285)
        None_25217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 24), 'None')
        
        (may_be_25218, more_types_in_union_25219) = may_not_be_none(title_25216, None_25217)

        if may_be_25218:

            if more_types_in_union_25219:
                # Runtime conditional SSA (line 285)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 286):
            
            # Obtaining the type of the subscript
            int_25220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 50), 'int')
            
            # Call to split(...): (line 286)
            # Processing the call arguments (line 286)
            str_25226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 41), 'str', '\n')
            int_25227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 47), 'int')
            # Processing the call keyword arguments (line 286)
            kwargs_25228 = {}
            
            # Call to lstrip(...): (line 286)
            # Processing the call keyword arguments (line 286)
            kwargs_25223 = {}
            # Getting the type of 'title' (line 286)
            title_25221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 20), 'title', False)
            # Obtaining the member 'lstrip' of a type (line 286)
            lstrip_25222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 20), title_25221, 'lstrip')
            # Calling lstrip(args, kwargs) (line 286)
            lstrip_call_result_25224 = invoke(stypy.reporting.localization.Localization(__file__, 286, 20), lstrip_25222, *[], **kwargs_25223)
            
            # Obtaining the member 'split' of a type (line 286)
            split_25225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 20), lstrip_call_result_25224, 'split')
            # Calling split(args, kwargs) (line 286)
            split_call_result_25229 = invoke(stypy.reporting.localization.Localization(__file__, 286, 20), split_25225, *[str_25226, int_25227], **kwargs_25228)
            
            # Obtaining the member '__getitem__' of a type (line 286)
            getitem___25230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 20), split_call_result_25229, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 286)
            subscript_call_result_25231 = invoke(stypy.reporting.localization.Localization(__file__, 286, 20), getitem___25230, int_25220)
            
            # Assigning a type to the variable 'title' (line 286)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'title', subscript_call_result_25231)
            # Getting the type of 'title' (line 287)
            title_25232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 19), 'title')
            # Assigning a type to the variable 'stypy_return_type' (line 287)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'stypy_return_type', title_25232)

            if more_types_in_union_25219:
                # SSA join for if statement (line 285)
                module_type_store = module_type_store.join_ssa_context()


        
        str_25233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 15), 'str', '* Not Available *')
        # Assigning a type to the variable 'stypy_return_type' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'stypy_return_type', str_25233)
        
        # ################# End of '_get_doc_title(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_doc_title' in the type store
        # Getting the type of 'stypy_return_type' (line 278)
        stypy_return_type_25234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25234)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_doc_title'
        return stypy_return_type_25234


    @norecursion
    def _format_titles(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_25235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 42), 'str', '---')
        defaults = [str_25235]
        # Create a new context for function '_format_titles'
        module_type_store = module_type_store.open_function_context('_format_titles', 290, 4, False)
        # Assigning a type to the variable 'self' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PackageLoader._format_titles.__dict__.__setitem__('stypy_localization', localization)
        PackageLoader._format_titles.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PackageLoader._format_titles.__dict__.__setitem__('stypy_type_store', module_type_store)
        PackageLoader._format_titles.__dict__.__setitem__('stypy_function_name', 'PackageLoader._format_titles')
        PackageLoader._format_titles.__dict__.__setitem__('stypy_param_names_list', ['titles', 'colsep'])
        PackageLoader._format_titles.__dict__.__setitem__('stypy_varargs_param_name', None)
        PackageLoader._format_titles.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PackageLoader._format_titles.__dict__.__setitem__('stypy_call_defaults', defaults)
        PackageLoader._format_titles.__dict__.__setitem__('stypy_call_varargs', varargs)
        PackageLoader._format_titles.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PackageLoader._format_titles.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PackageLoader._format_titles', ['titles', 'colsep'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_format_titles', localization, ['titles', 'colsep'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_format_titles(...)' code ##################

        
        # Assigning a Num to a Name (line 291):
        int_25236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 31), 'int')
        # Assigning a type to the variable 'display_window_width' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'display_window_width', int_25236)
        
        # Assigning a BinOp to a Name (line 292):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'titles' (line 292)
        titles_25249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 67), 'titles')
        comprehension_25250 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 19), titles_25249)
        # Assigning a type to the variable 'name' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 19), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 19), comprehension_25250))
        # Assigning a type to the variable 'title' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 19), 'title', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 19), comprehension_25250))
        
        # Call to len(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'name' (line 292)
        name_25238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 23), 'name', False)
        # Processing the call keyword arguments (line 292)
        kwargs_25239 = {}
        # Getting the type of 'len' (line 292)
        len_25237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 19), 'len', False)
        # Calling len(args, kwargs) (line 292)
        len_call_result_25240 = invoke(stypy.reporting.localization.Localization(__file__, 292, 19), len_25237, *[name_25238], **kwargs_25239)
        
        
        # Call to find(...): (line 292)
        # Processing the call arguments (line 292)
        str_25243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 39), 'str', '.')
        # Processing the call keyword arguments (line 292)
        kwargs_25244 = {}
        # Getting the type of 'name' (line 292)
        name_25241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 29), 'name', False)
        # Obtaining the member 'find' of a type (line 292)
        find_25242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 29), name_25241, 'find')
        # Calling find(args, kwargs) (line 292)
        find_call_result_25245 = invoke(stypy.reporting.localization.Localization(__file__, 292, 29), find_25242, *[str_25243], **kwargs_25244)
        
        # Applying the binary operator '-' (line 292)
        result_sub_25246 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 19), '-', len_call_result_25240, find_call_result_25245)
        
        int_25247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 44), 'int')
        # Applying the binary operator '-' (line 292)
        result_sub_25248 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 43), '-', result_sub_25246, int_25247)
        
        list_25251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 19), list_25251, result_sub_25248)
        
        # Obtaining an instance of the builtin type 'list' (line 292)
        list_25252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 75), 'list')
        # Adding type elements to the builtin type 'list' instance (line 292)
        # Adding element type (line 292)
        int_25253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 76), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 75), list_25252, int_25253)
        
        # Applying the binary operator '+' (line 292)
        result_add_25254 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 18), '+', list_25251, list_25252)
        
        # Assigning a type to the variable 'lengths' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'lengths', result_add_25254)
        
        # Assigning a Call to a Name (line 293):
        
        # Call to max(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'lengths' (line 293)
        lengths_25256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 25), 'lengths', False)
        # Processing the call keyword arguments (line 293)
        kwargs_25257 = {}
        # Getting the type of 'max' (line 293)
        max_25255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 21), 'max', False)
        # Calling max(args, kwargs) (line 293)
        max_call_result_25258 = invoke(stypy.reporting.localization.Localization(__file__, 293, 21), max_25255, *[lengths_25256], **kwargs_25257)
        
        # Assigning a type to the variable 'max_length' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'max_length', max_call_result_25258)
        
        # Assigning a List to a Name (line 294):
        
        # Obtaining an instance of the builtin type 'list' (line 294)
        list_25259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 294)
        
        # Assigning a type to the variable 'lines' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'lines', list_25259)
        
        # Getting the type of 'titles' (line 295)
        titles_25260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 29), 'titles')
        # Testing the type of a for loop iterable (line 295)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 295, 8), titles_25260)
        # Getting the type of the for loop variable (line 295)
        for_loop_var_25261 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 295, 8), titles_25260)
        # Assigning a type to the variable 'name' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 8), for_loop_var_25261))
        # Assigning a type to the variable 'title' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'title', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 8), for_loop_var_25261))
        # SSA begins for a for statement (line 295)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 296):
        
        # Obtaining the type of the subscript
        
        # Call to find(...): (line 296)
        # Processing the call arguments (line 296)
        str_25264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 34), 'str', '.')
        # Processing the call keyword arguments (line 296)
        kwargs_25265 = {}
        # Getting the type of 'name' (line 296)
        name_25262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 24), 'name', False)
        # Obtaining the member 'find' of a type (line 296)
        find_25263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 24), name_25262, 'find')
        # Calling find(args, kwargs) (line 296)
        find_call_result_25266 = invoke(stypy.reporting.localization.Localization(__file__, 296, 24), find_25263, *[str_25264], **kwargs_25265)
        
        int_25267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 39), 'int')
        # Applying the binary operator '+' (line 296)
        result_add_25268 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 24), '+', find_call_result_25266, int_25267)
        
        slice_25269 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 296, 19), result_add_25268, None, None)
        # Getting the type of 'name' (line 296)
        name_25270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 19), 'name')
        # Obtaining the member '__getitem__' of a type (line 296)
        getitem___25271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 19), name_25270, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 296)
        subscript_call_result_25272 = invoke(stypy.reporting.localization.Localization(__file__, 296, 19), getitem___25271, slice_25269)
        
        # Assigning a type to the variable 'name' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'name', subscript_call_result_25272)
        
        # Assigning a BinOp to a Name (line 297):
        # Getting the type of 'max_length' (line 297)
        max_length_25273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 16), 'max_length')
        
        # Call to len(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'name' (line 297)
        name_25275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 33), 'name', False)
        # Processing the call keyword arguments (line 297)
        kwargs_25276 = {}
        # Getting the type of 'len' (line 297)
        len_25274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 29), 'len', False)
        # Calling len(args, kwargs) (line 297)
        len_call_result_25277 = invoke(stypy.reporting.localization.Localization(__file__, 297, 29), len_25274, *[name_25275], **kwargs_25276)
        
        # Applying the binary operator '-' (line 297)
        result_sub_25278 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 16), '-', max_length_25273, len_call_result_25277)
        
        # Assigning a type to the variable 'w' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'w', result_sub_25278)
        
        # Assigning a Call to a Name (line 298):
        
        # Call to split(...): (line 298)
        # Processing the call keyword arguments (line 298)
        kwargs_25281 = {}
        # Getting the type of 'title' (line 298)
        title_25279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 20), 'title', False)
        # Obtaining the member 'split' of a type (line 298)
        split_25280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 20), title_25279, 'split')
        # Calling split(args, kwargs) (line 298)
        split_call_result_25282 = invoke(stypy.reporting.localization.Localization(__file__, 298, 20), split_25280, *[], **kwargs_25281)
        
        # Assigning a type to the variable 'words' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'words', split_call_result_25282)
        
        # Assigning a BinOp to a Name (line 299):
        str_25283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 19), 'str', '%s%s %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 299)
        tuple_25284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 299)
        # Adding element type (line 299)
        # Getting the type of 'name' (line 299)
        name_25285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 32), 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 32), tuple_25284, name_25285)
        # Adding element type (line 299)
        # Getting the type of 'w' (line 299)
        w_25286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 38), 'w')
        str_25287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 40), 'str', ' ')
        # Applying the binary operator '*' (line 299)
        result_mul_25288 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 38), '*', w_25286, str_25287)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 32), tuple_25284, result_mul_25288)
        # Adding element type (line 299)
        # Getting the type of 'colsep' (line 299)
        colsep_25289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 45), 'colsep')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 32), tuple_25284, colsep_25289)
        
        # Applying the binary operator '%' (line 299)
        result_mod_25290 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 19), '%', str_25283, tuple_25284)
        
        # Assigning a type to the variable 'line' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'line', result_mod_25290)
        
        # Assigning a BinOp to a Name (line 300):
        
        # Call to len(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'line' (line 300)
        line_25292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 22), 'line', False)
        # Processing the call keyword arguments (line 300)
        kwargs_25293 = {}
        # Getting the type of 'len' (line 300)
        len_25291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 18), 'len', False)
        # Calling len(args, kwargs) (line 300)
        len_call_result_25294 = invoke(stypy.reporting.localization.Localization(__file__, 300, 18), len_25291, *[line_25292], **kwargs_25293)
        
        str_25295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 30), 'str', ' ')
        # Applying the binary operator '*' (line 300)
        result_mul_25296 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 18), '*', len_call_result_25294, str_25295)
        
        # Assigning a type to the variable 'tab' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'tab', result_mul_25296)
        
        # Getting the type of 'words' (line 301)
        words_25297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 18), 'words')
        # Testing the type of an if condition (line 301)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 301, 12), words_25297)
        # SSA begins for while statement (line 301)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 302):
        
        # Call to pop(...): (line 302)
        # Processing the call arguments (line 302)
        int_25300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 33), 'int')
        # Processing the call keyword arguments (line 302)
        kwargs_25301 = {}
        # Getting the type of 'words' (line 302)
        words_25298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 23), 'words', False)
        # Obtaining the member 'pop' of a type (line 302)
        pop_25299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 23), words_25298, 'pop')
        # Calling pop(args, kwargs) (line 302)
        pop_call_result_25302 = invoke(stypy.reporting.localization.Localization(__file__, 302, 23), pop_25299, *[int_25300], **kwargs_25301)
        
        # Assigning a type to the variable 'word' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'word', pop_call_result_25302)
        
        
        
        # Call to len(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'line' (line 303)
        line_25304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 23), 'line', False)
        # Processing the call keyword arguments (line 303)
        kwargs_25305 = {}
        # Getting the type of 'len' (line 303)
        len_25303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 19), 'len', False)
        # Calling len(args, kwargs) (line 303)
        len_call_result_25306 = invoke(stypy.reporting.localization.Localization(__file__, 303, 19), len_25303, *[line_25304], **kwargs_25305)
        
        
        # Call to len(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'word' (line 303)
        word_25308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 33), 'word', False)
        # Processing the call keyword arguments (line 303)
        kwargs_25309 = {}
        # Getting the type of 'len' (line 303)
        len_25307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 29), 'len', False)
        # Calling len(args, kwargs) (line 303)
        len_call_result_25310 = invoke(stypy.reporting.localization.Localization(__file__, 303, 29), len_25307, *[word_25308], **kwargs_25309)
        
        # Applying the binary operator '+' (line 303)
        result_add_25311 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 19), '+', len_call_result_25306, len_call_result_25310)
        
        # Getting the type of 'display_window_width' (line 303)
        display_window_width_25312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 39), 'display_window_width')
        # Applying the binary operator '>' (line 303)
        result_gt_25313 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 19), '>', result_add_25311, display_window_width_25312)
        
        # Testing the type of an if condition (line 303)
        if_condition_25314 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 303, 16), result_gt_25313)
        # Assigning a type to the variable 'if_condition_25314' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'if_condition_25314', if_condition_25314)
        # SSA begins for if statement (line 303)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'line' (line 304)
        line_25317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 33), 'line', False)
        # Processing the call keyword arguments (line 304)
        kwargs_25318 = {}
        # Getting the type of 'lines' (line 304)
        lines_25315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 20), 'lines', False)
        # Obtaining the member 'append' of a type (line 304)
        append_25316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 20), lines_25315, 'append')
        # Calling append(args, kwargs) (line 304)
        append_call_result_25319 = invoke(stypy.reporting.localization.Localization(__file__, 304, 20), append_25316, *[line_25317], **kwargs_25318)
        
        
        # Assigning a Name to a Name (line 305):
        # Getting the type of 'tab' (line 305)
        tab_25320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 27), 'tab')
        # Assigning a type to the variable 'line' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 20), 'line', tab_25320)
        # SSA join for if statement (line 303)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'line' (line 306)
        line_25321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'line')
        str_25322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 24), 'str', ' ')
        # Getting the type of 'word' (line 306)
        word_25323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 30), 'word')
        # Applying the binary operator '+' (line 306)
        result_add_25324 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 24), '+', str_25322, word_25323)
        
        # Applying the binary operator '+=' (line 306)
        result_iadd_25325 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 16), '+=', line_25321, result_add_25324)
        # Assigning a type to the variable 'line' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'line', result_iadd_25325)
        
        # SSA branch for the else part of a while statement (line 301)
        module_type_store.open_ssa_branch('while loop else')
        
        # Call to append(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'line' (line 308)
        line_25328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 29), 'line', False)
        # Processing the call keyword arguments (line 308)
        kwargs_25329 = {}
        # Getting the type of 'lines' (line 308)
        lines_25326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'lines', False)
        # Obtaining the member 'append' of a type (line 308)
        append_25327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 16), lines_25326, 'append')
        # Calling append(args, kwargs) (line 308)
        append_call_result_25330 = invoke(stypy.reporting.localization.Localization(__file__, 308, 16), append_25327, *[line_25328], **kwargs_25329)
        
        # SSA join for while statement (line 301)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to join(...): (line 309)
        # Processing the call arguments (line 309)
        # Getting the type of 'lines' (line 309)
        lines_25333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 25), 'lines', False)
        # Processing the call keyword arguments (line 309)
        kwargs_25334 = {}
        str_25331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 15), 'str', '\n')
        # Obtaining the member 'join' of a type (line 309)
        join_25332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 15), str_25331, 'join')
        # Calling join(args, kwargs) (line 309)
        join_call_result_25335 = invoke(stypy.reporting.localization.Localization(__file__, 309, 15), join_25332, *[lines_25333], **kwargs_25334)
        
        # Assigning a type to the variable 'stypy_return_type' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'stypy_return_type', join_call_result_25335)
        
        # ################# End of '_format_titles(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_format_titles' in the type store
        # Getting the type of 'stypy_return_type' (line 290)
        stypy_return_type_25336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25336)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_format_titles'
        return stypy_return_type_25336


    @norecursion
    def get_pkgdocs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_pkgdocs'
        module_type_store = module_type_store.open_function_context('get_pkgdocs', 311, 4, False)
        # Assigning a type to the variable 'self' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PackageLoader.get_pkgdocs.__dict__.__setitem__('stypy_localization', localization)
        PackageLoader.get_pkgdocs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PackageLoader.get_pkgdocs.__dict__.__setitem__('stypy_type_store', module_type_store)
        PackageLoader.get_pkgdocs.__dict__.__setitem__('stypy_function_name', 'PackageLoader.get_pkgdocs')
        PackageLoader.get_pkgdocs.__dict__.__setitem__('stypy_param_names_list', [])
        PackageLoader.get_pkgdocs.__dict__.__setitem__('stypy_varargs_param_name', None)
        PackageLoader.get_pkgdocs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PackageLoader.get_pkgdocs.__dict__.__setitem__('stypy_call_defaults', defaults)
        PackageLoader.get_pkgdocs.__dict__.__setitem__('stypy_call_varargs', varargs)
        PackageLoader.get_pkgdocs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PackageLoader.get_pkgdocs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PackageLoader.get_pkgdocs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_pkgdocs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_pkgdocs(...)' code ##################

        str_25337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, (-1)), 'str', ' Return documentation summary of subpackages.\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 314, 8))
        
        # 'import sys' statement (line 314)
        import sys

        import_module(stypy.reporting.localization.Localization(__file__, 314, 8), 'sys', sys, module_type_store)
        
        
        # Assigning a Dict to a Attribute (line 315):
        
        # Obtaining an instance of the builtin type 'dict' (line 315)
        dict_25338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 315)
        
        # Getting the type of 'self' (line 315)
        self_25339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'self')
        # Setting the type of the member 'info_modules' of a type (line 315)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 8), self_25339, 'info_modules', dict_25338)
        
        # Call to _init_info_modules(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'None' (line 316)
        None_25342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 32), 'None', False)
        # Processing the call keyword arguments (line 316)
        kwargs_25343 = {}
        # Getting the type of 'self' (line 316)
        self_25340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'self', False)
        # Obtaining the member '_init_info_modules' of a type (line 316)
        _init_info_modules_25341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), self_25340, '_init_info_modules')
        # Calling _init_info_modules(args, kwargs) (line 316)
        _init_info_modules_call_result_25344 = invoke(stypy.reporting.localization.Localization(__file__, 316, 8), _init_info_modules_25341, *[None_25342], **kwargs_25343)
        
        
        # Assigning a List to a Name (line 318):
        
        # Obtaining an instance of the builtin type 'list' (line 318)
        list_25345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 318)
        
        # Assigning a type to the variable 'titles' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'titles', list_25345)
        
        # Assigning a List to a Name (line 319):
        
        # Obtaining an instance of the builtin type 'list' (line 319)
        list_25346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 319)
        
        # Assigning a type to the variable 'symbols' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'symbols', list_25346)
        
        
        # Call to items(...): (line 320)
        # Processing the call keyword arguments (line 320)
        kwargs_25350 = {}
        # Getting the type of 'self' (line 320)
        self_25347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 41), 'self', False)
        # Obtaining the member 'info_modules' of a type (line 320)
        info_modules_25348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 41), self_25347, 'info_modules')
        # Obtaining the member 'items' of a type (line 320)
        items_25349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 41), info_modules_25348, 'items')
        # Calling items(args, kwargs) (line 320)
        items_call_result_25351 = invoke(stypy.reporting.localization.Localization(__file__, 320, 41), items_25349, *[], **kwargs_25350)
        
        # Testing the type of a for loop iterable (line 320)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 320, 8), items_call_result_25351)
        # Getting the type of the for loop variable (line 320)
        for_loop_var_25352 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 320, 8), items_call_result_25351)
        # Assigning a type to the variable 'package_name' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'package_name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 8), for_loop_var_25352))
        # Assigning a type to the variable 'info_module' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'info_module', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 8), for_loop_var_25352))
        # SSA begins for a for statement (line 320)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 321):
        
        # Call to getattr(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'info_module' (line 321)
        info_module_25354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 37), 'info_module', False)
        str_25355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 50), 'str', 'global_symbols')
        
        # Obtaining an instance of the builtin type 'list' (line 321)
        list_25356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 68), 'list')
        # Adding type elements to the builtin type 'list' instance (line 321)
        
        # Processing the call keyword arguments (line 321)
        kwargs_25357 = {}
        # Getting the type of 'getattr' (line 321)
        getattr_25353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 29), 'getattr', False)
        # Calling getattr(args, kwargs) (line 321)
        getattr_call_result_25358 = invoke(stypy.reporting.localization.Localization(__file__, 321, 29), getattr_25353, *[info_module_25354, str_25355, list_25356], **kwargs_25357)
        
        # Assigning a type to the variable 'global_symbols' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'global_symbols', getattr_call_result_25358)
        
        # Assigning a BinOp to a Name (line 322):
        # Getting the type of 'self' (line 322)
        self_25359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 23), 'self')
        # Obtaining the member 'parent_name' of a type (line 322)
        parent_name_25360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 23), self_25359, 'parent_name')
        str_25361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 41), 'str', '.')
        # Applying the binary operator '+' (line 322)
        result_add_25362 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 23), '+', parent_name_25360, str_25361)
        
        # Getting the type of 'package_name' (line 322)
        package_name_25363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 46), 'package_name')
        # Applying the binary operator '+' (line 322)
        result_add_25364 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 44), '+', result_add_25362, package_name_25363)
        
        # Assigning a type to the variable 'fullname' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'fullname', result_add_25364)
        
        # Assigning a Str to a Name (line 323):
        str_25365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 19), 'str', '')
        # Assigning a type to the variable 'note' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'note', str_25365)
        
        
        # Getting the type of 'fullname' (line 324)
        fullname_25366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 15), 'fullname')
        # Getting the type of 'sys' (line 324)
        sys_25367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 31), 'sys')
        # Obtaining the member 'modules' of a type (line 324)
        modules_25368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 31), sys_25367, 'modules')
        # Applying the binary operator 'notin' (line 324)
        result_contains_25369 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 15), 'notin', fullname_25366, modules_25368)
        
        # Testing the type of an if condition (line 324)
        if_condition_25370 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 324, 12), result_contains_25369)
        # Assigning a type to the variable 'if_condition_25370' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'if_condition_25370', if_condition_25370)
        # SSA begins for if statement (line 324)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 325):
        str_25371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 23), 'str', ' [*]')
        # Assigning a type to the variable 'note' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 16), 'note', str_25371)
        # SSA join for if statement (line 324)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 326)
        # Processing the call arguments (line 326)
        
        # Obtaining an instance of the builtin type 'tuple' (line 326)
        tuple_25374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 326)
        # Adding element type (line 326)
        # Getting the type of 'fullname' (line 326)
        fullname_25375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 27), 'fullname', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 27), tuple_25374, fullname_25375)
        # Adding element type (line 326)
        
        # Call to _get_doc_title(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'info_module' (line 326)
        info_module_25378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 57), 'info_module', False)
        # Processing the call keyword arguments (line 326)
        kwargs_25379 = {}
        # Getting the type of 'self' (line 326)
        self_25376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 37), 'self', False)
        # Obtaining the member '_get_doc_title' of a type (line 326)
        _get_doc_title_25377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 37), self_25376, '_get_doc_title')
        # Calling _get_doc_title(args, kwargs) (line 326)
        _get_doc_title_call_result_25380 = invoke(stypy.reporting.localization.Localization(__file__, 326, 37), _get_doc_title_25377, *[info_module_25378], **kwargs_25379)
        
        # Getting the type of 'note' (line 326)
        note_25381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 72), 'note', False)
        # Applying the binary operator '+' (line 326)
        result_add_25382 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 37), '+', _get_doc_title_call_result_25380, note_25381)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 27), tuple_25374, result_add_25382)
        
        # Processing the call keyword arguments (line 326)
        kwargs_25383 = {}
        # Getting the type of 'titles' (line 326)
        titles_25372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'titles', False)
        # Obtaining the member 'append' of a type (line 326)
        append_25373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 12), titles_25372, 'append')
        # Calling append(args, kwargs) (line 326)
        append_call_result_25384 = invoke(stypy.reporting.localization.Localization(__file__, 326, 12), append_25373, *[tuple_25374], **kwargs_25383)
        
        
        # Getting the type of 'global_symbols' (line 327)
        global_symbols_25385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 15), 'global_symbols')
        # Testing the type of an if condition (line 327)
        if_condition_25386 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 327, 12), global_symbols_25385)
        # Assigning a type to the variable 'if_condition_25386' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'if_condition_25386', if_condition_25386)
        # SSA begins for if statement (line 327)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 328)
        # Processing the call arguments (line 328)
        
        # Obtaining an instance of the builtin type 'tuple' (line 328)
        tuple_25389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 328)
        # Adding element type (line 328)
        # Getting the type of 'package_name' (line 328)
        package_name_25390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 32), 'package_name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 32), tuple_25389, package_name_25390)
        # Adding element type (line 328)
        
        # Call to join(...): (line 328)
        # Processing the call arguments (line 328)
        # Getting the type of 'global_symbols' (line 328)
        global_symbols_25393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 56), 'global_symbols', False)
        # Processing the call keyword arguments (line 328)
        kwargs_25394 = {}
        str_25391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 46), 'str', ', ')
        # Obtaining the member 'join' of a type (line 328)
        join_25392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 46), str_25391, 'join')
        # Calling join(args, kwargs) (line 328)
        join_call_result_25395 = invoke(stypy.reporting.localization.Localization(__file__, 328, 46), join_25392, *[global_symbols_25393], **kwargs_25394)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 32), tuple_25389, join_call_result_25395)
        
        # Processing the call keyword arguments (line 328)
        kwargs_25396 = {}
        # Getting the type of 'symbols' (line 328)
        symbols_25387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 16), 'symbols', False)
        # Obtaining the member 'append' of a type (line 328)
        append_25388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 16), symbols_25387, 'append')
        # Calling append(args, kwargs) (line 328)
        append_call_result_25397 = invoke(stypy.reporting.localization.Localization(__file__, 328, 16), append_25388, *[tuple_25389], **kwargs_25396)
        
        # SSA join for if statement (line 327)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 330):
        
        # Call to _format_titles(...): (line 330)
        # Processing the call arguments (line 330)
        # Getting the type of 'titles' (line 330)
        titles_25400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 37), 'titles', False)
        # Processing the call keyword arguments (line 330)
        kwargs_25401 = {}
        # Getting the type of 'self' (line 330)
        self_25398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 17), 'self', False)
        # Obtaining the member '_format_titles' of a type (line 330)
        _format_titles_25399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 17), self_25398, '_format_titles')
        # Calling _format_titles(args, kwargs) (line 330)
        _format_titles_call_result_25402 = invoke(stypy.reporting.localization.Localization(__file__, 330, 17), _format_titles_25399, *[titles_25400], **kwargs_25401)
        
        str_25403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 15), 'str', '\n  [*] - using a package requires explicit import (see pkgload)')
        # Applying the binary operator '+' (line 330)
        result_add_25404 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 17), '+', _format_titles_call_result_25402, str_25403)
        
        # Assigning a type to the variable 'retstr' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'retstr', result_add_25404)
        
        # Getting the type of 'symbols' (line 334)
        symbols_25405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 11), 'symbols')
        # Testing the type of an if condition (line 334)
        if_condition_25406 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 8), symbols_25405)
        # Assigning a type to the variable 'if_condition_25406' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'if_condition_25406', if_condition_25406)
        # SSA begins for if statement (line 334)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'retstr' (line 335)
        retstr_25407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'retstr')
        str_25408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 22), 'str', '\n\nGlobal symbols from subpackages\n-------------------------------\n')
        
        # Call to _format_titles(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'symbols' (line 337)
        symbols_25411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 42), 'symbols', False)
        str_25412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 51), 'str', '-->')
        # Processing the call keyword arguments (line 337)
        kwargs_25413 = {}
        # Getting the type of 'self' (line 337)
        self_25409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 22), 'self', False)
        # Obtaining the member '_format_titles' of a type (line 337)
        _format_titles_25410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 22), self_25409, '_format_titles')
        # Calling _format_titles(args, kwargs) (line 337)
        _format_titles_call_result_25414 = invoke(stypy.reporting.localization.Localization(__file__, 337, 22), _format_titles_25410, *[symbols_25411, str_25412], **kwargs_25413)
        
        # Applying the binary operator '+' (line 335)
        result_add_25415 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 22), '+', str_25408, _format_titles_call_result_25414)
        
        # Applying the binary operator '+=' (line 335)
        result_iadd_25416 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 12), '+=', retstr_25407, result_add_25415)
        # Assigning a type to the variable 'retstr' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'retstr', result_iadd_25416)
        
        # SSA join for if statement (line 334)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'retstr' (line 339)
        retstr_25417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 15), 'retstr')
        # Assigning a type to the variable 'stypy_return_type' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'stypy_return_type', retstr_25417)
        
        # ################# End of 'get_pkgdocs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_pkgdocs' in the type store
        # Getting the type of 'stypy_return_type' (line 311)
        stypy_return_type_25418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25418)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_pkgdocs'
        return stypy_return_type_25418


# Assigning a type to the variable 'PackageLoader' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'PackageLoader', PackageLoader)
# Declaration of the 'PackageLoaderDebug' class
# Getting the type of 'PackageLoader' (line 341)
PackageLoader_25419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 25), 'PackageLoader')

class PackageLoaderDebug(PackageLoader_25419, ):

    @norecursion
    def _execcmd(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_execcmd'
        module_type_store = module_type_store.open_function_context('_execcmd', 342, 4, False)
        # Assigning a type to the variable 'self' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PackageLoaderDebug._execcmd.__dict__.__setitem__('stypy_localization', localization)
        PackageLoaderDebug._execcmd.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PackageLoaderDebug._execcmd.__dict__.__setitem__('stypy_type_store', module_type_store)
        PackageLoaderDebug._execcmd.__dict__.__setitem__('stypy_function_name', 'PackageLoaderDebug._execcmd')
        PackageLoaderDebug._execcmd.__dict__.__setitem__('stypy_param_names_list', ['cmdstr'])
        PackageLoaderDebug._execcmd.__dict__.__setitem__('stypy_varargs_param_name', None)
        PackageLoaderDebug._execcmd.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PackageLoaderDebug._execcmd.__dict__.__setitem__('stypy_call_defaults', defaults)
        PackageLoaderDebug._execcmd.__dict__.__setitem__('stypy_call_varargs', varargs)
        PackageLoaderDebug._execcmd.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PackageLoaderDebug._execcmd.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PackageLoaderDebug._execcmd', ['cmdstr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_execcmd', localization, ['cmdstr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_execcmd(...)' code ##################

        str_25420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 8), 'str', ' Execute command in parent_frame.')
        
        # Assigning a Attribute to a Name (line 344):
        # Getting the type of 'self' (line 344)
        self_25421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 16), 'self')
        # Obtaining the member 'parent_frame' of a type (line 344)
        parent_frame_25422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 16), self_25421, 'parent_frame')
        # Assigning a type to the variable 'frame' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'frame', parent_frame_25422)
        
        # Call to print(...): (line 345)
        # Processing the call arguments (line 345)
        str_25424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 14), 'str', 'Executing')
        
        # Call to repr(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'cmdstr' (line 345)
        cmdstr_25426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 32), 'cmdstr', False)
        # Processing the call keyword arguments (line 345)
        kwargs_25427 = {}
        # Getting the type of 'repr' (line 345)
        repr_25425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 27), 'repr', False)
        # Calling repr(args, kwargs) (line 345)
        repr_call_result_25428 = invoke(stypy.reporting.localization.Localization(__file__, 345, 27), repr_25425, *[cmdstr_25426], **kwargs_25427)
        
        str_25429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 41), 'str', '...')
        # Processing the call keyword arguments (line 345)
        str_25430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 52), 'str', ' ')
        keyword_25431 = str_25430
        kwargs_25432 = {'end': keyword_25431}
        # Getting the type of 'print' (line 345)
        print_25423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'print', False)
        # Calling print(args, kwargs) (line 345)
        print_call_result_25433 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), print_25423, *[str_25424, repr_call_result_25428, str_25429], **kwargs_25432)
        
        
        # Call to flush(...): (line 346)
        # Processing the call keyword arguments (line 346)
        kwargs_25437 = {}
        # Getting the type of 'sys' (line 346)
        sys_25434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'sys', False)
        # Obtaining the member 'stdout' of a type (line 346)
        stdout_25435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 8), sys_25434, 'stdout')
        # Obtaining the member 'flush' of a type (line 346)
        flush_25436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 8), stdout_25435, 'flush')
        # Calling flush(args, kwargs) (line 346)
        flush_call_result_25438 = invoke(stypy.reporting.localization.Localization(__file__, 346, 8), flush_25436, *[], **kwargs_25437)
        
        # Dynamic code evaluation using an exec statement
        # Getting the type of 'cmdstr' (line 347)
        cmdstr_25439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 14), 'cmdstr')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 347, 8), cmdstr_25439, 'exec parameter', 'StringType', 'FileType', 'CodeType')
        enable_usage_of_dynamic_types_warning(stypy.reporting.localization.Localization(__file__, 347, 8))
        
        # Call to print(...): (line 348)
        # Processing the call arguments (line 348)
        str_25441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 14), 'str', 'ok')
        # Processing the call keyword arguments (line 348)
        kwargs_25442 = {}
        # Getting the type of 'print' (line 348)
        print_25440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'print', False)
        # Calling print(args, kwargs) (line 348)
        print_call_result_25443 = invoke(stypy.reporting.localization.Localization(__file__, 348, 8), print_25440, *[str_25441], **kwargs_25442)
        
        
        # Call to flush(...): (line 349)
        # Processing the call keyword arguments (line 349)
        kwargs_25447 = {}
        # Getting the type of 'sys' (line 349)
        sys_25444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'sys', False)
        # Obtaining the member 'stdout' of a type (line 349)
        stdout_25445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), sys_25444, 'stdout')
        # Obtaining the member 'flush' of a type (line 349)
        flush_25446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), stdout_25445, 'flush')
        # Calling flush(args, kwargs) (line 349)
        flush_call_result_25448 = invoke(stypy.reporting.localization.Localization(__file__, 349, 8), flush_25446, *[], **kwargs_25447)
        
        # Assigning a type to the variable 'stypy_return_type' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of '_execcmd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_execcmd' in the type store
        # Getting the type of 'stypy_return_type' (line 342)
        stypy_return_type_25449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25449)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_execcmd'
        return stypy_return_type_25449


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 341, 0, False)
        # Assigning a type to the variable 'self' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PackageLoaderDebug.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'PackageLoaderDebug' (line 341)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 0), 'PackageLoaderDebug', PackageLoaderDebug)


# Call to int(...): (line 352)
# Processing the call arguments (line 352)

# Call to get(...): (line 352)
# Processing the call arguments (line 352)
str_25454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 22), 'str', 'NUMPY_IMPORT_DEBUG')
str_25455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 44), 'str', '0')
# Processing the call keyword arguments (line 352)
kwargs_25456 = {}
# Getting the type of 'os' (line 352)
os_25451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 7), 'os', False)
# Obtaining the member 'environ' of a type (line 352)
environ_25452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 7), os_25451, 'environ')
# Obtaining the member 'get' of a type (line 352)
get_25453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 7), environ_25452, 'get')
# Calling get(args, kwargs) (line 352)
get_call_result_25457 = invoke(stypy.reporting.localization.Localization(__file__, 352, 7), get_25453, *[str_25454, str_25455], **kwargs_25456)

# Processing the call keyword arguments (line 352)
kwargs_25458 = {}
# Getting the type of 'int' (line 352)
int_25450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 3), 'int', False)
# Calling int(args, kwargs) (line 352)
int_call_result_25459 = invoke(stypy.reporting.localization.Localization(__file__, 352, 3), int_25450, *[get_call_result_25457], **kwargs_25458)

# Testing the type of an if condition (line 352)
if_condition_25460 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 352, 0), int_call_result_25459)
# Assigning a type to the variable 'if_condition_25460' (line 352)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 0), 'if_condition_25460', if_condition_25460)
# SSA begins for if statement (line 352)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 353):
# Getting the type of 'PackageLoaderDebug' (line 353)
PackageLoaderDebug_25461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 20), 'PackageLoaderDebug')
# Assigning a type to the variable 'PackageLoader' (line 353)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'PackageLoader', PackageLoaderDebug_25461)
# SSA join for if statement (line 352)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
