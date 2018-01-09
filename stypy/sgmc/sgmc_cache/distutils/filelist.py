
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.filelist
2: 
3: Provides the FileList class, used for poking about the filesystem
4: and building lists of files.
5: '''
6: 
7: __revision__ = "$Id$"
8: 
9: import os, re
10: import fnmatch
11: from distutils.util import convert_path
12: from distutils.errors import DistutilsTemplateError, DistutilsInternalError
13: from distutils import log
14: 
15: class FileList:
16:     '''A list of files built by on exploring the filesystem and filtered by
17:     applying various patterns to what we find there.
18: 
19:     Instance attributes:
20:       dir
21:         directory from which files will be taken -- only used if
22:         'allfiles' not supplied to constructor
23:       files
24:         list of filenames currently being built/filtered/manipulated
25:       allfiles
26:         complete list of files under consideration (ie. without any
27:         filtering applied)
28:     '''
29: 
30:     def __init__(self, warn=None, debug_print=None):
31:         # ignore argument to FileList, but keep them for backwards
32:         # compatibility
33:         self.allfiles = None
34:         self.files = []
35: 
36:     def set_allfiles(self, allfiles):
37:         self.allfiles = allfiles
38: 
39:     def findall(self, dir=os.curdir):
40:         self.allfiles = findall(dir)
41: 
42:     def debug_print(self, msg):
43:         '''Print 'msg' to stdout if the global DEBUG (taken from the
44:         DISTUTILS_DEBUG environment variable) flag is true.
45:         '''
46:         from distutils.debug import DEBUG
47:         if DEBUG:
48:             print msg
49: 
50:     # -- List-like methods ---------------------------------------------
51: 
52:     def append(self, item):
53:         self.files.append(item)
54: 
55:     def extend(self, items):
56:         self.files.extend(items)
57: 
58:     def sort(self):
59:         # Not a strict lexical sort!
60:         sortable_files = map(os.path.split, self.files)
61:         sortable_files.sort()
62:         self.files = []
63:         for sort_tuple in sortable_files:
64:             self.files.append(os.path.join(*sort_tuple))
65: 
66: 
67:     # -- Other miscellaneous utility methods ---------------------------
68: 
69:     def remove_duplicates(self):
70:         # Assumes list has been sorted!
71:         for i in range(len(self.files) - 1, 0, -1):
72:             if self.files[i] == self.files[i - 1]:
73:                 del self.files[i]
74: 
75: 
76:     # -- "File template" methods ---------------------------------------
77: 
78:     def _parse_template_line(self, line):
79:         words = line.split()
80:         action = words[0]
81: 
82:         patterns = dir = dir_pattern = None
83: 
84:         if action in ('include', 'exclude',
85:                       'global-include', 'global-exclude'):
86:             if len(words) < 2:
87:                 raise DistutilsTemplateError, \
88:                       "'%s' expects <pattern1> <pattern2> ..." % action
89: 
90:             patterns = map(convert_path, words[1:])
91: 
92:         elif action in ('recursive-include', 'recursive-exclude'):
93:             if len(words) < 3:
94:                 raise DistutilsTemplateError, \
95:                       "'%s' expects <dir> <pattern1> <pattern2> ..." % action
96: 
97:             dir = convert_path(words[1])
98:             patterns = map(convert_path, words[2:])
99: 
100:         elif action in ('graft', 'prune'):
101:             if len(words) != 2:
102:                 raise DistutilsTemplateError, \
103:                      "'%s' expects a single <dir_pattern>" % action
104: 
105:             dir_pattern = convert_path(words[1])
106: 
107:         else:
108:             raise DistutilsTemplateError, "unknown action '%s'" % action
109: 
110:         return (action, patterns, dir, dir_pattern)
111: 
112:     def process_template_line(self, line):
113:         # Parse the line: split it up, make sure the right number of words
114:         # is there, and return the relevant words.  'action' is always
115:         # defined: it's the first word of the line.  Which of the other
116:         # three are defined depends on the action; it'll be either
117:         # patterns, (dir and patterns), or (dir_pattern).
118:         action, patterns, dir, dir_pattern = self._parse_template_line(line)
119: 
120:         # OK, now we know that the action is valid and we have the
121:         # right number of words on the line for that action -- so we
122:         # can proceed with minimal error-checking.
123:         if action == 'include':
124:             self.debug_print("include " + ' '.join(patterns))
125:             for pattern in patterns:
126:                 if not self.include_pattern(pattern, anchor=1):
127:                     log.warn("warning: no files found matching '%s'",
128:                              pattern)
129: 
130:         elif action == 'exclude':
131:             self.debug_print("exclude " + ' '.join(patterns))
132:             for pattern in patterns:
133:                 if not self.exclude_pattern(pattern, anchor=1):
134:                     log.warn(("warning: no previously-included files "
135:                               "found matching '%s'"), pattern)
136: 
137:         elif action == 'global-include':
138:             self.debug_print("global-include " + ' '.join(patterns))
139:             for pattern in patterns:
140:                 if not self.include_pattern(pattern, anchor=0):
141:                     log.warn(("warning: no files found matching '%s' " +
142:                               "anywhere in distribution"), pattern)
143: 
144:         elif action == 'global-exclude':
145:             self.debug_print("global-exclude " + ' '.join(patterns))
146:             for pattern in patterns:
147:                 if not self.exclude_pattern(pattern, anchor=0):
148:                     log.warn(("warning: no previously-included files matching "
149:                               "'%s' found anywhere in distribution"),
150:                              pattern)
151: 
152:         elif action == 'recursive-include':
153:             self.debug_print("recursive-include %s %s" %
154:                              (dir, ' '.join(patterns)))
155:             for pattern in patterns:
156:                 if not self.include_pattern(pattern, prefix=dir):
157:                     log.warn(("warning: no files found matching '%s' " +
158:                                 "under directory '%s'"),
159:                              pattern, dir)
160: 
161:         elif action == 'recursive-exclude':
162:             self.debug_print("recursive-exclude %s %s" %
163:                              (dir, ' '.join(patterns)))
164:             for pattern in patterns:
165:                 if not self.exclude_pattern(pattern, prefix=dir):
166:                     log.warn(("warning: no previously-included files matching "
167:                               "'%s' found under directory '%s'"),
168:                              pattern, dir)
169: 
170:         elif action == 'graft':
171:             self.debug_print("graft " + dir_pattern)
172:             if not self.include_pattern(None, prefix=dir_pattern):
173:                 log.warn("warning: no directories found matching '%s'",
174:                          dir_pattern)
175: 
176:         elif action == 'prune':
177:             self.debug_print("prune " + dir_pattern)
178:             if not self.exclude_pattern(None, prefix=dir_pattern):
179:                 log.warn(("no previously-included directories found " +
180:                           "matching '%s'"), dir_pattern)
181:         else:
182:             raise DistutilsInternalError, \
183:                   "this cannot happen: invalid action '%s'" % action
184: 
185:     # -- Filtering/selection methods -----------------------------------
186: 
187:     def include_pattern(self, pattern, anchor=1, prefix=None, is_regex=0):
188:         '''Select strings (presumably filenames) from 'self.files' that
189:         match 'pattern', a Unix-style wildcard (glob) pattern.
190: 
191:         Patterns are not quite the same as implemented by the 'fnmatch'
192:         module: '*' and '?'  match non-special characters, where "special"
193:         is platform-dependent: slash on Unix; colon, slash, and backslash on
194:         DOS/Windows; and colon on Mac OS.
195: 
196:         If 'anchor' is true (the default), then the pattern match is more
197:         stringent: "*.py" will match "foo.py" but not "foo/bar.py".  If
198:         'anchor' is false, both of these will match.
199: 
200:         If 'prefix' is supplied, then only filenames starting with 'prefix'
201:         (itself a pattern) and ending with 'pattern', with anything in between
202:         them, will match.  'anchor' is ignored in this case.
203: 
204:         If 'is_regex' is true, 'anchor' and 'prefix' are ignored, and
205:         'pattern' is assumed to be either a string containing a regex or a
206:         regex object -- no translation is done, the regex is just compiled
207:         and used as-is.
208: 
209:         Selected strings will be added to self.files.
210: 
211:         Return 1 if files are found.
212:         '''
213:         # XXX docstring lying about what the special chars are?
214:         files_found = 0
215:         pattern_re = translate_pattern(pattern, anchor, prefix, is_regex)
216:         self.debug_print("include_pattern: applying regex r'%s'" %
217:                          pattern_re.pattern)
218: 
219:         # delayed loading of allfiles list
220:         if self.allfiles is None:
221:             self.findall()
222: 
223:         for name in self.allfiles:
224:             if pattern_re.search(name):
225:                 self.debug_print(" adding " + name)
226:                 self.files.append(name)
227:                 files_found = 1
228: 
229:         return files_found
230: 
231: 
232:     def exclude_pattern(self, pattern, anchor=1, prefix=None, is_regex=0):
233:         '''Remove strings (presumably filenames) from 'files' that match
234:         'pattern'.
235: 
236:         Other parameters are the same as for 'include_pattern()', above.
237:         The list 'self.files' is modified in place. Return 1 if files are
238:         found.
239:         '''
240:         files_found = 0
241:         pattern_re = translate_pattern(pattern, anchor, prefix, is_regex)
242:         self.debug_print("exclude_pattern: applying regex r'%s'" %
243:                          pattern_re.pattern)
244:         for i in range(len(self.files)-1, -1, -1):
245:             if pattern_re.search(self.files[i]):
246:                 self.debug_print(" removing " + self.files[i])
247:                 del self.files[i]
248:                 files_found = 1
249: 
250:         return files_found
251: 
252: 
253: # ----------------------------------------------------------------------
254: # Utility functions
255: 
256: def findall(dir = os.curdir):
257:     '''Find all files under 'dir' and return the list of full filenames
258:     (relative to 'dir').
259:     '''
260:     from stat import ST_MODE, S_ISREG, S_ISDIR, S_ISLNK
261: 
262:     list = []
263:     stack = [dir]
264:     pop = stack.pop
265:     push = stack.append
266: 
267:     while stack:
268:         dir = pop()
269:         names = os.listdir(dir)
270: 
271:         for name in names:
272:             if dir != os.curdir:        # avoid the dreaded "./" syndrome
273:                 fullname = os.path.join(dir, name)
274:             else:
275:                 fullname = name
276: 
277:             # Avoid excess stat calls -- just one will do, thank you!
278:             stat = os.stat(fullname)
279:             mode = stat[ST_MODE]
280:             if S_ISREG(mode):
281:                 list.append(fullname)
282:             elif S_ISDIR(mode) and not S_ISLNK(mode):
283:                 push(fullname)
284: 
285:     return list
286: 
287: 
288: def glob_to_re(pattern):
289:     '''Translate a shell-like glob pattern to a regular expression.
290: 
291:     Return a string containing the regex.  Differs from
292:     'fnmatch.translate()' in that '*' does not match "special characters"
293:     (which are platform-specific).
294:     '''
295:     pattern_re = fnmatch.translate(pattern)
296: 
297:     # '?' and '*' in the glob pattern become '.' and '.*' in the RE, which
298:     # IMHO is wrong -- '?' and '*' aren't supposed to match slash in Unix,
299:     # and by extension they shouldn't match such "special characters" under
300:     # any OS.  So change all non-escaped dots in the RE to match any
301:     # character except the special characters (currently: just os.sep).
302:     sep = os.sep
303:     if os.sep == '\\':
304:         # we're using a regex to manipulate a regex, so we need
305:         # to escape the backslash twice
306:         sep = r'\\\\'
307:     escaped = r'\1[^%s]' % sep
308:     pattern_re = re.sub(r'((?<!\\)(\\\\)*)\.', escaped, pattern_re)
309:     return pattern_re
310: 
311: 
312: def translate_pattern(pattern, anchor=1, prefix=None, is_regex=0):
313:     '''Translate a shell-like wildcard pattern to a compiled regular
314:     expression.
315: 
316:     Return the compiled regex.  If 'is_regex' true,
317:     then 'pattern' is directly compiled to a regex (if it's a string)
318:     or just returned as-is (assumes it's a regex object).
319:     '''
320:     if is_regex:
321:         if isinstance(pattern, str):
322:             return re.compile(pattern)
323:         else:
324:             return pattern
325: 
326:     if pattern:
327:         pattern_re = glob_to_re(pattern)
328:     else:
329:         pattern_re = ''
330: 
331:     if prefix is not None:
332:         # ditch end of pattern character
333:         empty_pattern = glob_to_re('')
334:         prefix_re = glob_to_re(prefix)[:-len(empty_pattern)]
335:         sep = os.sep
336:         if os.sep == '\\':
337:             sep = r'\\'
338:         pattern_re = "^" + sep.join((prefix_re, ".*" + pattern_re))
339:     else:                               # no prefix -- respect anchor flag
340:         if anchor:
341:             pattern_re = "^" + pattern_re
342: 
343:     return re.compile(pattern_re)
344: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_1431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', 'distutils.filelist\n\nProvides the FileList class, used for poking about the filesystem\nand building lists of files.\n')

# Assigning a Str to a Name (line 7):

# Assigning a Str to a Name (line 7):
str_1432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__revision__', str_1432)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# Multiple import statement. import os (1/2) (line 9)
import os

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os', os, module_type_store)
# Multiple import statement. import re (2/2) (line 9)
import re

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import fnmatch' statement (line 10)
import fnmatch

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'fnmatch', fnmatch, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.util import convert_path' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_1433 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.util')

if (type(import_1433) is not StypyTypeError):

    if (import_1433 != 'pyd_module'):
        __import__(import_1433)
        sys_modules_1434 = sys.modules[import_1433]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.util', sys_modules_1434.module_type_store, module_type_store, ['convert_path'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_1434, sys_modules_1434.module_type_store, module_type_store)
    else:
        from distutils.util import convert_path

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.util', None, module_type_store, ['convert_path'], [convert_path])

else:
    # Assigning a type to the variable 'distutils.util' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.util', import_1433)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.errors import DistutilsTemplateError, DistutilsInternalError' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_1435 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors')

if (type(import_1435) is not StypyTypeError):

    if (import_1435 != 'pyd_module'):
        __import__(import_1435)
        sys_modules_1436 = sys.modules[import_1435]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', sys_modules_1436.module_type_store, module_type_store, ['DistutilsTemplateError', 'DistutilsInternalError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_1436, sys_modules_1436.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsTemplateError, DistutilsInternalError

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', None, module_type_store, ['DistutilsTemplateError', 'DistutilsInternalError'], [DistutilsTemplateError, DistutilsInternalError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', import_1435)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from distutils import log' statement (line 13)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils', None, module_type_store, ['log'], [log])

# Declaration of the 'FileList' class

class FileList:
    str_1437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, (-1)), 'str', "A list of files built by on exploring the filesystem and filtered by\n    applying various patterns to what we find there.\n\n    Instance attributes:\n      dir\n        directory from which files will be taken -- only used if\n        'allfiles' not supplied to constructor\n      files\n        list of filenames currently being built/filtered/manipulated\n      allfiles\n        complete list of files under consideration (ie. without any\n        filtering applied)\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 30)
        None_1438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 28), 'None')
        # Getting the type of 'None' (line 30)
        None_1439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 46), 'None')
        defaults = [None_1438, None_1439]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileList.__init__', ['warn', 'debug_print'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['warn', 'debug_print'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 33):
        
        # Assigning a Name to a Attribute (line 33):
        # Getting the type of 'None' (line 33)
        None_1440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 24), 'None')
        # Getting the type of 'self' (line 33)
        self_1441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Setting the type of the member 'allfiles' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_1441, 'allfiles', None_1440)
        
        # Assigning a List to a Attribute (line 34):
        
        # Assigning a List to a Attribute (line 34):
        
        # Obtaining an instance of the builtin type 'list' (line 34)
        list_1442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 34)
        
        # Getting the type of 'self' (line 34)
        self_1443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'files' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_1443, 'files', list_1442)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_allfiles(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_allfiles'
        module_type_store = module_type_store.open_function_context('set_allfiles', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileList.set_allfiles.__dict__.__setitem__('stypy_localization', localization)
        FileList.set_allfiles.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileList.set_allfiles.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileList.set_allfiles.__dict__.__setitem__('stypy_function_name', 'FileList.set_allfiles')
        FileList.set_allfiles.__dict__.__setitem__('stypy_param_names_list', ['allfiles'])
        FileList.set_allfiles.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileList.set_allfiles.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileList.set_allfiles.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileList.set_allfiles.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileList.set_allfiles.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileList.set_allfiles.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileList.set_allfiles', ['allfiles'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_allfiles', localization, ['allfiles'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_allfiles(...)' code ##################

        
        # Assigning a Name to a Attribute (line 37):
        
        # Assigning a Name to a Attribute (line 37):
        # Getting the type of 'allfiles' (line 37)
        allfiles_1444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 24), 'allfiles')
        # Getting the type of 'self' (line 37)
        self_1445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self')
        # Setting the type of the member 'allfiles' of a type (line 37)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_1445, 'allfiles', allfiles_1444)
        
        # ################# End of 'set_allfiles(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_allfiles' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_1446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1446)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_allfiles'
        return stypy_return_type_1446


    @norecursion
    def findall(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'os' (line 39)
        os_1447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 26), 'os')
        # Obtaining the member 'curdir' of a type (line 39)
        curdir_1448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 26), os_1447, 'curdir')
        defaults = [curdir_1448]
        # Create a new context for function 'findall'
        module_type_store = module_type_store.open_function_context('findall', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileList.findall.__dict__.__setitem__('stypy_localization', localization)
        FileList.findall.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileList.findall.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileList.findall.__dict__.__setitem__('stypy_function_name', 'FileList.findall')
        FileList.findall.__dict__.__setitem__('stypy_param_names_list', ['dir'])
        FileList.findall.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileList.findall.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileList.findall.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileList.findall.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileList.findall.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileList.findall.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileList.findall', ['dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'findall', localization, ['dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'findall(...)' code ##################

        
        # Assigning a Call to a Attribute (line 40):
        
        # Assigning a Call to a Attribute (line 40):
        
        # Call to findall(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'dir' (line 40)
        dir_1450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 32), 'dir', False)
        # Processing the call keyword arguments (line 40)
        kwargs_1451 = {}
        # Getting the type of 'findall' (line 40)
        findall_1449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'findall', False)
        # Calling findall(args, kwargs) (line 40)
        findall_call_result_1452 = invoke(stypy.reporting.localization.Localization(__file__, 40, 24), findall_1449, *[dir_1450], **kwargs_1451)
        
        # Getting the type of 'self' (line 40)
        self_1453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self')
        # Setting the type of the member 'allfiles' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_1453, 'allfiles', findall_call_result_1452)
        
        # ################# End of 'findall(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'findall' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_1454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1454)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'findall'
        return stypy_return_type_1454


    @norecursion
    def debug_print(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'debug_print'
        module_type_store = module_type_store.open_function_context('debug_print', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileList.debug_print.__dict__.__setitem__('stypy_localization', localization)
        FileList.debug_print.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileList.debug_print.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileList.debug_print.__dict__.__setitem__('stypy_function_name', 'FileList.debug_print')
        FileList.debug_print.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        FileList.debug_print.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileList.debug_print.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileList.debug_print.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileList.debug_print.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileList.debug_print.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileList.debug_print.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileList.debug_print', ['msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'debug_print', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'debug_print(...)' code ##################

        str_1455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, (-1)), 'str', "Print 'msg' to stdout if the global DEBUG (taken from the\n        DISTUTILS_DEBUG environment variable) flag is true.\n        ")
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 46, 8))
        
        # 'from distutils.debug import DEBUG' statement (line 46)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/')
        import_1456 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 46, 8), 'distutils.debug')

        if (type(import_1456) is not StypyTypeError):

            if (import_1456 != 'pyd_module'):
                __import__(import_1456)
                sys_modules_1457 = sys.modules[import_1456]
                import_from_module(stypy.reporting.localization.Localization(__file__, 46, 8), 'distutils.debug', sys_modules_1457.module_type_store, module_type_store, ['DEBUG'])
                nest_module(stypy.reporting.localization.Localization(__file__, 46, 8), __file__, sys_modules_1457, sys_modules_1457.module_type_store, module_type_store)
            else:
                from distutils.debug import DEBUG

                import_from_module(stypy.reporting.localization.Localization(__file__, 46, 8), 'distutils.debug', None, module_type_store, ['DEBUG'], [DEBUG])

        else:
            # Assigning a type to the variable 'distutils.debug' (line 46)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'distutils.debug', import_1456)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
        
        
        # Getting the type of 'DEBUG' (line 47)
        DEBUG_1458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'DEBUG')
        # Testing the type of an if condition (line 47)
        if_condition_1459 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 8), DEBUG_1458)
        # Assigning a type to the variable 'if_condition_1459' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'if_condition_1459', if_condition_1459)
        # SSA begins for if statement (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'msg' (line 48)
        msg_1460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 18), 'msg')
        # SSA join for if statement (line 47)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'debug_print(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'debug_print' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_1461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1461)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'debug_print'
        return stypy_return_type_1461


    @norecursion
    def append(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'append'
        module_type_store = module_type_store.open_function_context('append', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileList.append.__dict__.__setitem__('stypy_localization', localization)
        FileList.append.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileList.append.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileList.append.__dict__.__setitem__('stypy_function_name', 'FileList.append')
        FileList.append.__dict__.__setitem__('stypy_param_names_list', ['item'])
        FileList.append.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileList.append.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileList.append.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileList.append.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileList.append.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileList.append.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileList.append', ['item'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'append', localization, ['item'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'append(...)' code ##################

        
        # Call to append(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'item' (line 53)
        item_1465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 26), 'item', False)
        # Processing the call keyword arguments (line 53)
        kwargs_1466 = {}
        # Getting the type of 'self' (line 53)
        self_1462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'self', False)
        # Obtaining the member 'files' of a type (line 53)
        files_1463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), self_1462, 'files')
        # Obtaining the member 'append' of a type (line 53)
        append_1464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), files_1463, 'append')
        # Calling append(args, kwargs) (line 53)
        append_call_result_1467 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), append_1464, *[item_1465], **kwargs_1466)
        
        
        # ################# End of 'append(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'append' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_1468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1468)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'append'
        return stypy_return_type_1468


    @norecursion
    def extend(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'extend'
        module_type_store = module_type_store.open_function_context('extend', 55, 4, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileList.extend.__dict__.__setitem__('stypy_localization', localization)
        FileList.extend.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileList.extend.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileList.extend.__dict__.__setitem__('stypy_function_name', 'FileList.extend')
        FileList.extend.__dict__.__setitem__('stypy_param_names_list', ['items'])
        FileList.extend.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileList.extend.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileList.extend.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileList.extend.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileList.extend.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileList.extend.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileList.extend', ['items'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'extend', localization, ['items'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'extend(...)' code ##################

        
        # Call to extend(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'items' (line 56)
        items_1472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 26), 'items', False)
        # Processing the call keyword arguments (line 56)
        kwargs_1473 = {}
        # Getting the type of 'self' (line 56)
        self_1469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self', False)
        # Obtaining the member 'files' of a type (line 56)
        files_1470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_1469, 'files')
        # Obtaining the member 'extend' of a type (line 56)
        extend_1471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), files_1470, 'extend')
        # Calling extend(args, kwargs) (line 56)
        extend_call_result_1474 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), extend_1471, *[items_1472], **kwargs_1473)
        
        
        # ################# End of 'extend(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'extend' in the type store
        # Getting the type of 'stypy_return_type' (line 55)
        stypy_return_type_1475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1475)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'extend'
        return stypy_return_type_1475


    @norecursion
    def sort(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sort'
        module_type_store = module_type_store.open_function_context('sort', 58, 4, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileList.sort.__dict__.__setitem__('stypy_localization', localization)
        FileList.sort.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileList.sort.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileList.sort.__dict__.__setitem__('stypy_function_name', 'FileList.sort')
        FileList.sort.__dict__.__setitem__('stypy_param_names_list', [])
        FileList.sort.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileList.sort.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileList.sort.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileList.sort.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileList.sort.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileList.sort.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileList.sort', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sort', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sort(...)' code ##################

        
        # Assigning a Call to a Name (line 60):
        
        # Assigning a Call to a Name (line 60):
        
        # Call to map(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'os' (line 60)
        os_1477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 29), 'os', False)
        # Obtaining the member 'path' of a type (line 60)
        path_1478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 29), os_1477, 'path')
        # Obtaining the member 'split' of a type (line 60)
        split_1479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 29), path_1478, 'split')
        # Getting the type of 'self' (line 60)
        self_1480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 44), 'self', False)
        # Obtaining the member 'files' of a type (line 60)
        files_1481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 44), self_1480, 'files')
        # Processing the call keyword arguments (line 60)
        kwargs_1482 = {}
        # Getting the type of 'map' (line 60)
        map_1476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'map', False)
        # Calling map(args, kwargs) (line 60)
        map_call_result_1483 = invoke(stypy.reporting.localization.Localization(__file__, 60, 25), map_1476, *[split_1479, files_1481], **kwargs_1482)
        
        # Assigning a type to the variable 'sortable_files' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'sortable_files', map_call_result_1483)
        
        # Call to sort(...): (line 61)
        # Processing the call keyword arguments (line 61)
        kwargs_1486 = {}
        # Getting the type of 'sortable_files' (line 61)
        sortable_files_1484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'sortable_files', False)
        # Obtaining the member 'sort' of a type (line 61)
        sort_1485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), sortable_files_1484, 'sort')
        # Calling sort(args, kwargs) (line 61)
        sort_call_result_1487 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), sort_1485, *[], **kwargs_1486)
        
        
        # Assigning a List to a Attribute (line 62):
        
        # Assigning a List to a Attribute (line 62):
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_1488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        
        # Getting the type of 'self' (line 62)
        self_1489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self')
        # Setting the type of the member 'files' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_1489, 'files', list_1488)
        
        # Getting the type of 'sortable_files' (line 63)
        sortable_files_1490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 26), 'sortable_files')
        # Testing the type of a for loop iterable (line 63)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 63, 8), sortable_files_1490)
        # Getting the type of the for loop variable (line 63)
        for_loop_var_1491 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 63, 8), sortable_files_1490)
        # Assigning a type to the variable 'sort_tuple' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'sort_tuple', for_loop_var_1491)
        # SSA begins for a for statement (line 63)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to join(...): (line 64)
        # Getting the type of 'sort_tuple' (line 64)
        sort_tuple_1498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 44), 'sort_tuple', False)
        # Processing the call keyword arguments (line 64)
        kwargs_1499 = {}
        # Getting the type of 'os' (line 64)
        os_1495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 64)
        path_1496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 30), os_1495, 'path')
        # Obtaining the member 'join' of a type (line 64)
        join_1497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 30), path_1496, 'join')
        # Calling join(args, kwargs) (line 64)
        join_call_result_1500 = invoke(stypy.reporting.localization.Localization(__file__, 64, 30), join_1497, *[sort_tuple_1498], **kwargs_1499)
        
        # Processing the call keyword arguments (line 64)
        kwargs_1501 = {}
        # Getting the type of 'self' (line 64)
        self_1492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'self', False)
        # Obtaining the member 'files' of a type (line 64)
        files_1493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), self_1492, 'files')
        # Obtaining the member 'append' of a type (line 64)
        append_1494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), files_1493, 'append')
        # Calling append(args, kwargs) (line 64)
        append_call_result_1502 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), append_1494, *[join_call_result_1500], **kwargs_1501)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'sort(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sort' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_1503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1503)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sort'
        return stypy_return_type_1503


    @norecursion
    def remove_duplicates(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'remove_duplicates'
        module_type_store = module_type_store.open_function_context('remove_duplicates', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileList.remove_duplicates.__dict__.__setitem__('stypy_localization', localization)
        FileList.remove_duplicates.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileList.remove_duplicates.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileList.remove_duplicates.__dict__.__setitem__('stypy_function_name', 'FileList.remove_duplicates')
        FileList.remove_duplicates.__dict__.__setitem__('stypy_param_names_list', [])
        FileList.remove_duplicates.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileList.remove_duplicates.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileList.remove_duplicates.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileList.remove_duplicates.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileList.remove_duplicates.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileList.remove_duplicates.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileList.remove_duplicates', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'remove_duplicates', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'remove_duplicates(...)' code ##################

        
        
        # Call to range(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Call to len(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'self' (line 71)
        self_1506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), 'self', False)
        # Obtaining the member 'files' of a type (line 71)
        files_1507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 27), self_1506, 'files')
        # Processing the call keyword arguments (line 71)
        kwargs_1508 = {}
        # Getting the type of 'len' (line 71)
        len_1505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 23), 'len', False)
        # Calling len(args, kwargs) (line 71)
        len_call_result_1509 = invoke(stypy.reporting.localization.Localization(__file__, 71, 23), len_1505, *[files_1507], **kwargs_1508)
        
        int_1510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 41), 'int')
        # Applying the binary operator '-' (line 71)
        result_sub_1511 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 23), '-', len_call_result_1509, int_1510)
        
        int_1512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 44), 'int')
        int_1513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 47), 'int')
        # Processing the call keyword arguments (line 71)
        kwargs_1514 = {}
        # Getting the type of 'range' (line 71)
        range_1504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 17), 'range', False)
        # Calling range(args, kwargs) (line 71)
        range_call_result_1515 = invoke(stypy.reporting.localization.Localization(__file__, 71, 17), range_1504, *[result_sub_1511, int_1512, int_1513], **kwargs_1514)
        
        # Testing the type of a for loop iterable (line 71)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 71, 8), range_call_result_1515)
        # Getting the type of the for loop variable (line 71)
        for_loop_var_1516 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 71, 8), range_call_result_1515)
        # Assigning a type to the variable 'i' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'i', for_loop_var_1516)
        # SSA begins for a for statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 72)
        i_1517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 26), 'i')
        # Getting the type of 'self' (line 72)
        self_1518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'self')
        # Obtaining the member 'files' of a type (line 72)
        files_1519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 15), self_1518, 'files')
        # Obtaining the member '__getitem__' of a type (line 72)
        getitem___1520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 15), files_1519, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 72)
        subscript_call_result_1521 = invoke(stypy.reporting.localization.Localization(__file__, 72, 15), getitem___1520, i_1517)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 72)
        i_1522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 43), 'i')
        int_1523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 47), 'int')
        # Applying the binary operator '-' (line 72)
        result_sub_1524 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 43), '-', i_1522, int_1523)
        
        # Getting the type of 'self' (line 72)
        self_1525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 32), 'self')
        # Obtaining the member 'files' of a type (line 72)
        files_1526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 32), self_1525, 'files')
        # Obtaining the member '__getitem__' of a type (line 72)
        getitem___1527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 32), files_1526, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 72)
        subscript_call_result_1528 = invoke(stypy.reporting.localization.Localization(__file__, 72, 32), getitem___1527, result_sub_1524)
        
        # Applying the binary operator '==' (line 72)
        result_eq_1529 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 15), '==', subscript_call_result_1521, subscript_call_result_1528)
        
        # Testing the type of an if condition (line 72)
        if_condition_1530 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 12), result_eq_1529)
        # Assigning a type to the variable 'if_condition_1530' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'if_condition_1530', if_condition_1530)
        # SSA begins for if statement (line 72)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Deleting a member
        # Getting the type of 'self' (line 73)
        self_1531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'self')
        # Obtaining the member 'files' of a type (line 73)
        files_1532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 20), self_1531, 'files')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 73)
        i_1533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 31), 'i')
        # Getting the type of 'self' (line 73)
        self_1534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'self')
        # Obtaining the member 'files' of a type (line 73)
        files_1535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 20), self_1534, 'files')
        # Obtaining the member '__getitem__' of a type (line 73)
        getitem___1536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 20), files_1535, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
        subscript_call_result_1537 = invoke(stypy.reporting.localization.Localization(__file__, 73, 20), getitem___1536, i_1533)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 16), files_1532, subscript_call_result_1537)
        # SSA join for if statement (line 72)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'remove_duplicates(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'remove_duplicates' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_1538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1538)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'remove_duplicates'
        return stypy_return_type_1538


    @norecursion
    def _parse_template_line(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_parse_template_line'
        module_type_store = module_type_store.open_function_context('_parse_template_line', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileList._parse_template_line.__dict__.__setitem__('stypy_localization', localization)
        FileList._parse_template_line.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileList._parse_template_line.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileList._parse_template_line.__dict__.__setitem__('stypy_function_name', 'FileList._parse_template_line')
        FileList._parse_template_line.__dict__.__setitem__('stypy_param_names_list', ['line'])
        FileList._parse_template_line.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileList._parse_template_line.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileList._parse_template_line.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileList._parse_template_line.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileList._parse_template_line.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileList._parse_template_line.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileList._parse_template_line', ['line'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_parse_template_line', localization, ['line'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_parse_template_line(...)' code ##################

        
        # Assigning a Call to a Name (line 79):
        
        # Assigning a Call to a Name (line 79):
        
        # Call to split(...): (line 79)
        # Processing the call keyword arguments (line 79)
        kwargs_1541 = {}
        # Getting the type of 'line' (line 79)
        line_1539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 16), 'line', False)
        # Obtaining the member 'split' of a type (line 79)
        split_1540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 16), line_1539, 'split')
        # Calling split(args, kwargs) (line 79)
        split_call_result_1542 = invoke(stypy.reporting.localization.Localization(__file__, 79, 16), split_1540, *[], **kwargs_1541)
        
        # Assigning a type to the variable 'words' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'words', split_call_result_1542)
        
        # Assigning a Subscript to a Name (line 80):
        
        # Assigning a Subscript to a Name (line 80):
        
        # Obtaining the type of the subscript
        int_1543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 23), 'int')
        # Getting the type of 'words' (line 80)
        words_1544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'words')
        # Obtaining the member '__getitem__' of a type (line 80)
        getitem___1545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 17), words_1544, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 80)
        subscript_call_result_1546 = invoke(stypy.reporting.localization.Localization(__file__, 80, 17), getitem___1545, int_1543)
        
        # Assigning a type to the variable 'action' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'action', subscript_call_result_1546)
        
        # Multiple assignment of 3 elements.
        
        # Assigning a Name to a Name (line 82):
        # Getting the type of 'None' (line 82)
        None_1547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 39), 'None')
        # Assigning a type to the variable 'dir_pattern' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 25), 'dir_pattern', None_1547)
        
        # Assigning a Name to a Name (line 82):
        # Getting the type of 'dir_pattern' (line 82)
        dir_pattern_1548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 25), 'dir_pattern')
        # Assigning a type to the variable 'dir' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'dir', dir_pattern_1548)
        
        # Assigning a Name to a Name (line 82):
        # Getting the type of 'dir' (line 82)
        dir_1549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'dir')
        # Assigning a type to the variable 'patterns' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'patterns', dir_1549)
        
        
        # Getting the type of 'action' (line 84)
        action_1550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 11), 'action')
        
        # Obtaining an instance of the builtin type 'tuple' (line 84)
        tuple_1551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 84)
        # Adding element type (line 84)
        str_1552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 22), 'str', 'include')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 22), tuple_1551, str_1552)
        # Adding element type (line 84)
        str_1553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 33), 'str', 'exclude')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 22), tuple_1551, str_1553)
        # Adding element type (line 84)
        str_1554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 22), 'str', 'global-include')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 22), tuple_1551, str_1554)
        # Adding element type (line 84)
        str_1555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 40), 'str', 'global-exclude')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 22), tuple_1551, str_1555)
        
        # Applying the binary operator 'in' (line 84)
        result_contains_1556 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 11), 'in', action_1550, tuple_1551)
        
        # Testing the type of an if condition (line 84)
        if_condition_1557 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 8), result_contains_1556)
        # Assigning a type to the variable 'if_condition_1557' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'if_condition_1557', if_condition_1557)
        # SSA begins for if statement (line 84)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to len(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'words' (line 86)
        words_1559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'words', False)
        # Processing the call keyword arguments (line 86)
        kwargs_1560 = {}
        # Getting the type of 'len' (line 86)
        len_1558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'len', False)
        # Calling len(args, kwargs) (line 86)
        len_call_result_1561 = invoke(stypy.reporting.localization.Localization(__file__, 86, 15), len_1558, *[words_1559], **kwargs_1560)
        
        int_1562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 28), 'int')
        # Applying the binary operator '<' (line 86)
        result_lt_1563 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 15), '<', len_call_result_1561, int_1562)
        
        # Testing the type of an if condition (line 86)
        if_condition_1564 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 12), result_lt_1563)
        # Assigning a type to the variable 'if_condition_1564' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'if_condition_1564', if_condition_1564)
        # SSA begins for if statement (line 86)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsTemplateError' (line 87)
        DistutilsTemplateError_1565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 22), 'DistutilsTemplateError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 87, 16), DistutilsTemplateError_1565, 'raise parameter', BaseException)
        # SSA join for if statement (line 86)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 90):
        
        # Assigning a Call to a Name (line 90):
        
        # Call to map(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'convert_path' (line 90)
        convert_path_1567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 27), 'convert_path', False)
        
        # Obtaining the type of the subscript
        int_1568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 47), 'int')
        slice_1569 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 90, 41), int_1568, None, None)
        # Getting the type of 'words' (line 90)
        words_1570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 41), 'words', False)
        # Obtaining the member '__getitem__' of a type (line 90)
        getitem___1571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 41), words_1570, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 90)
        subscript_call_result_1572 = invoke(stypy.reporting.localization.Localization(__file__, 90, 41), getitem___1571, slice_1569)
        
        # Processing the call keyword arguments (line 90)
        kwargs_1573 = {}
        # Getting the type of 'map' (line 90)
        map_1566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 23), 'map', False)
        # Calling map(args, kwargs) (line 90)
        map_call_result_1574 = invoke(stypy.reporting.localization.Localization(__file__, 90, 23), map_1566, *[convert_path_1567, subscript_call_result_1572], **kwargs_1573)
        
        # Assigning a type to the variable 'patterns' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'patterns', map_call_result_1574)
        # SSA branch for the else part of an if statement (line 84)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'action' (line 92)
        action_1575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 13), 'action')
        
        # Obtaining an instance of the builtin type 'tuple' (line 92)
        tuple_1576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 92)
        # Adding element type (line 92)
        str_1577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 24), 'str', 'recursive-include')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 24), tuple_1576, str_1577)
        # Adding element type (line 92)
        str_1578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 45), 'str', 'recursive-exclude')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 24), tuple_1576, str_1578)
        
        # Applying the binary operator 'in' (line 92)
        result_contains_1579 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 13), 'in', action_1575, tuple_1576)
        
        # Testing the type of an if condition (line 92)
        if_condition_1580 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 13), result_contains_1579)
        # Assigning a type to the variable 'if_condition_1580' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 13), 'if_condition_1580', if_condition_1580)
        # SSA begins for if statement (line 92)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to len(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'words' (line 93)
        words_1582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 19), 'words', False)
        # Processing the call keyword arguments (line 93)
        kwargs_1583 = {}
        # Getting the type of 'len' (line 93)
        len_1581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 15), 'len', False)
        # Calling len(args, kwargs) (line 93)
        len_call_result_1584 = invoke(stypy.reporting.localization.Localization(__file__, 93, 15), len_1581, *[words_1582], **kwargs_1583)
        
        int_1585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 28), 'int')
        # Applying the binary operator '<' (line 93)
        result_lt_1586 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 15), '<', len_call_result_1584, int_1585)
        
        # Testing the type of an if condition (line 93)
        if_condition_1587 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 12), result_lt_1586)
        # Assigning a type to the variable 'if_condition_1587' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'if_condition_1587', if_condition_1587)
        # SSA begins for if statement (line 93)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsTemplateError' (line 94)
        DistutilsTemplateError_1588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 22), 'DistutilsTemplateError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 94, 16), DistutilsTemplateError_1588, 'raise parameter', BaseException)
        # SSA join for if statement (line 93)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 97):
        
        # Assigning a Call to a Name (line 97):
        
        # Call to convert_path(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Obtaining the type of the subscript
        int_1590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 37), 'int')
        # Getting the type of 'words' (line 97)
        words_1591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 31), 'words', False)
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___1592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 31), words_1591, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_1593 = invoke(stypy.reporting.localization.Localization(__file__, 97, 31), getitem___1592, int_1590)
        
        # Processing the call keyword arguments (line 97)
        kwargs_1594 = {}
        # Getting the type of 'convert_path' (line 97)
        convert_path_1589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 18), 'convert_path', False)
        # Calling convert_path(args, kwargs) (line 97)
        convert_path_call_result_1595 = invoke(stypy.reporting.localization.Localization(__file__, 97, 18), convert_path_1589, *[subscript_call_result_1593], **kwargs_1594)
        
        # Assigning a type to the variable 'dir' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'dir', convert_path_call_result_1595)
        
        # Assigning a Call to a Name (line 98):
        
        # Assigning a Call to a Name (line 98):
        
        # Call to map(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'convert_path' (line 98)
        convert_path_1597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), 'convert_path', False)
        
        # Obtaining the type of the subscript
        int_1598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 47), 'int')
        slice_1599 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 98, 41), int_1598, None, None)
        # Getting the type of 'words' (line 98)
        words_1600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 41), 'words', False)
        # Obtaining the member '__getitem__' of a type (line 98)
        getitem___1601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 41), words_1600, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 98)
        subscript_call_result_1602 = invoke(stypy.reporting.localization.Localization(__file__, 98, 41), getitem___1601, slice_1599)
        
        # Processing the call keyword arguments (line 98)
        kwargs_1603 = {}
        # Getting the type of 'map' (line 98)
        map_1596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 23), 'map', False)
        # Calling map(args, kwargs) (line 98)
        map_call_result_1604 = invoke(stypy.reporting.localization.Localization(__file__, 98, 23), map_1596, *[convert_path_1597, subscript_call_result_1602], **kwargs_1603)
        
        # Assigning a type to the variable 'patterns' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'patterns', map_call_result_1604)
        # SSA branch for the else part of an if statement (line 92)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'action' (line 100)
        action_1605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 13), 'action')
        
        # Obtaining an instance of the builtin type 'tuple' (line 100)
        tuple_1606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 100)
        # Adding element type (line 100)
        str_1607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 24), 'str', 'graft')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 24), tuple_1606, str_1607)
        # Adding element type (line 100)
        str_1608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 33), 'str', 'prune')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 24), tuple_1606, str_1608)
        
        # Applying the binary operator 'in' (line 100)
        result_contains_1609 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 13), 'in', action_1605, tuple_1606)
        
        # Testing the type of an if condition (line 100)
        if_condition_1610 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 13), result_contains_1609)
        # Assigning a type to the variable 'if_condition_1610' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 13), 'if_condition_1610', if_condition_1610)
        # SSA begins for if statement (line 100)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to len(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'words' (line 101)
        words_1612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'words', False)
        # Processing the call keyword arguments (line 101)
        kwargs_1613 = {}
        # Getting the type of 'len' (line 101)
        len_1611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'len', False)
        # Calling len(args, kwargs) (line 101)
        len_call_result_1614 = invoke(stypy.reporting.localization.Localization(__file__, 101, 15), len_1611, *[words_1612], **kwargs_1613)
        
        int_1615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 29), 'int')
        # Applying the binary operator '!=' (line 101)
        result_ne_1616 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 15), '!=', len_call_result_1614, int_1615)
        
        # Testing the type of an if condition (line 101)
        if_condition_1617 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 12), result_ne_1616)
        # Assigning a type to the variable 'if_condition_1617' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'if_condition_1617', if_condition_1617)
        # SSA begins for if statement (line 101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsTemplateError' (line 102)
        DistutilsTemplateError_1618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 22), 'DistutilsTemplateError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 102, 16), DistutilsTemplateError_1618, 'raise parameter', BaseException)
        # SSA join for if statement (line 101)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 105):
        
        # Assigning a Call to a Name (line 105):
        
        # Call to convert_path(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Obtaining the type of the subscript
        int_1620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 45), 'int')
        # Getting the type of 'words' (line 105)
        words_1621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 39), 'words', False)
        # Obtaining the member '__getitem__' of a type (line 105)
        getitem___1622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 39), words_1621, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 105)
        subscript_call_result_1623 = invoke(stypy.reporting.localization.Localization(__file__, 105, 39), getitem___1622, int_1620)
        
        # Processing the call keyword arguments (line 105)
        kwargs_1624 = {}
        # Getting the type of 'convert_path' (line 105)
        convert_path_1619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 26), 'convert_path', False)
        # Calling convert_path(args, kwargs) (line 105)
        convert_path_call_result_1625 = invoke(stypy.reporting.localization.Localization(__file__, 105, 26), convert_path_1619, *[subscript_call_result_1623], **kwargs_1624)
        
        # Assigning a type to the variable 'dir_pattern' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'dir_pattern', convert_path_call_result_1625)
        # SSA branch for the else part of an if statement (line 100)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'DistutilsTemplateError' (line 108)
        DistutilsTemplateError_1626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 18), 'DistutilsTemplateError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 108, 12), DistutilsTemplateError_1626, 'raise parameter', BaseException)
        # SSA join for if statement (line 100)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 92)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 84)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 110)
        tuple_1627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 110)
        # Adding element type (line 110)
        # Getting the type of 'action' (line 110)
        action_1628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'action')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 16), tuple_1627, action_1628)
        # Adding element type (line 110)
        # Getting the type of 'patterns' (line 110)
        patterns_1629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 24), 'patterns')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 16), tuple_1627, patterns_1629)
        # Adding element type (line 110)
        # Getting the type of 'dir' (line 110)
        dir_1630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 34), 'dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 16), tuple_1627, dir_1630)
        # Adding element type (line 110)
        # Getting the type of 'dir_pattern' (line 110)
        dir_pattern_1631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 39), 'dir_pattern')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 16), tuple_1627, dir_pattern_1631)
        
        # Assigning a type to the variable 'stypy_return_type' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'stypy_return_type', tuple_1627)
        
        # ################# End of '_parse_template_line(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_parse_template_line' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_1632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1632)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_parse_template_line'
        return stypy_return_type_1632


    @norecursion
    def process_template_line(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'process_template_line'
        module_type_store = module_type_store.open_function_context('process_template_line', 112, 4, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileList.process_template_line.__dict__.__setitem__('stypy_localization', localization)
        FileList.process_template_line.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileList.process_template_line.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileList.process_template_line.__dict__.__setitem__('stypy_function_name', 'FileList.process_template_line')
        FileList.process_template_line.__dict__.__setitem__('stypy_param_names_list', ['line'])
        FileList.process_template_line.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileList.process_template_line.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileList.process_template_line.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileList.process_template_line.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileList.process_template_line.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileList.process_template_line.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileList.process_template_line', ['line'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'process_template_line', localization, ['line'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'process_template_line(...)' code ##################

        
        # Assigning a Call to a Tuple (line 118):
        
        # Assigning a Subscript to a Name (line 118):
        
        # Obtaining the type of the subscript
        int_1633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 8), 'int')
        
        # Call to _parse_template_line(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'line' (line 118)
        line_1636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 71), 'line', False)
        # Processing the call keyword arguments (line 118)
        kwargs_1637 = {}
        # Getting the type of 'self' (line 118)
        self_1634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 45), 'self', False)
        # Obtaining the member '_parse_template_line' of a type (line 118)
        _parse_template_line_1635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 45), self_1634, '_parse_template_line')
        # Calling _parse_template_line(args, kwargs) (line 118)
        _parse_template_line_call_result_1638 = invoke(stypy.reporting.localization.Localization(__file__, 118, 45), _parse_template_line_1635, *[line_1636], **kwargs_1637)
        
        # Obtaining the member '__getitem__' of a type (line 118)
        getitem___1639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), _parse_template_line_call_result_1638, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 118)
        subscript_call_result_1640 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), getitem___1639, int_1633)
        
        # Assigning a type to the variable 'tuple_var_assignment_1427' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'tuple_var_assignment_1427', subscript_call_result_1640)
        
        # Assigning a Subscript to a Name (line 118):
        
        # Obtaining the type of the subscript
        int_1641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 8), 'int')
        
        # Call to _parse_template_line(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'line' (line 118)
        line_1644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 71), 'line', False)
        # Processing the call keyword arguments (line 118)
        kwargs_1645 = {}
        # Getting the type of 'self' (line 118)
        self_1642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 45), 'self', False)
        # Obtaining the member '_parse_template_line' of a type (line 118)
        _parse_template_line_1643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 45), self_1642, '_parse_template_line')
        # Calling _parse_template_line(args, kwargs) (line 118)
        _parse_template_line_call_result_1646 = invoke(stypy.reporting.localization.Localization(__file__, 118, 45), _parse_template_line_1643, *[line_1644], **kwargs_1645)
        
        # Obtaining the member '__getitem__' of a type (line 118)
        getitem___1647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), _parse_template_line_call_result_1646, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 118)
        subscript_call_result_1648 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), getitem___1647, int_1641)
        
        # Assigning a type to the variable 'tuple_var_assignment_1428' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'tuple_var_assignment_1428', subscript_call_result_1648)
        
        # Assigning a Subscript to a Name (line 118):
        
        # Obtaining the type of the subscript
        int_1649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 8), 'int')
        
        # Call to _parse_template_line(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'line' (line 118)
        line_1652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 71), 'line', False)
        # Processing the call keyword arguments (line 118)
        kwargs_1653 = {}
        # Getting the type of 'self' (line 118)
        self_1650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 45), 'self', False)
        # Obtaining the member '_parse_template_line' of a type (line 118)
        _parse_template_line_1651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 45), self_1650, '_parse_template_line')
        # Calling _parse_template_line(args, kwargs) (line 118)
        _parse_template_line_call_result_1654 = invoke(stypy.reporting.localization.Localization(__file__, 118, 45), _parse_template_line_1651, *[line_1652], **kwargs_1653)
        
        # Obtaining the member '__getitem__' of a type (line 118)
        getitem___1655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), _parse_template_line_call_result_1654, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 118)
        subscript_call_result_1656 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), getitem___1655, int_1649)
        
        # Assigning a type to the variable 'tuple_var_assignment_1429' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'tuple_var_assignment_1429', subscript_call_result_1656)
        
        # Assigning a Subscript to a Name (line 118):
        
        # Obtaining the type of the subscript
        int_1657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 8), 'int')
        
        # Call to _parse_template_line(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'line' (line 118)
        line_1660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 71), 'line', False)
        # Processing the call keyword arguments (line 118)
        kwargs_1661 = {}
        # Getting the type of 'self' (line 118)
        self_1658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 45), 'self', False)
        # Obtaining the member '_parse_template_line' of a type (line 118)
        _parse_template_line_1659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 45), self_1658, '_parse_template_line')
        # Calling _parse_template_line(args, kwargs) (line 118)
        _parse_template_line_call_result_1662 = invoke(stypy.reporting.localization.Localization(__file__, 118, 45), _parse_template_line_1659, *[line_1660], **kwargs_1661)
        
        # Obtaining the member '__getitem__' of a type (line 118)
        getitem___1663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), _parse_template_line_call_result_1662, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 118)
        subscript_call_result_1664 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), getitem___1663, int_1657)
        
        # Assigning a type to the variable 'tuple_var_assignment_1430' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'tuple_var_assignment_1430', subscript_call_result_1664)
        
        # Assigning a Name to a Name (line 118):
        # Getting the type of 'tuple_var_assignment_1427' (line 118)
        tuple_var_assignment_1427_1665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'tuple_var_assignment_1427')
        # Assigning a type to the variable 'action' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'action', tuple_var_assignment_1427_1665)
        
        # Assigning a Name to a Name (line 118):
        # Getting the type of 'tuple_var_assignment_1428' (line 118)
        tuple_var_assignment_1428_1666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'tuple_var_assignment_1428')
        # Assigning a type to the variable 'patterns' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'patterns', tuple_var_assignment_1428_1666)
        
        # Assigning a Name to a Name (line 118):
        # Getting the type of 'tuple_var_assignment_1429' (line 118)
        tuple_var_assignment_1429_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'tuple_var_assignment_1429')
        # Assigning a type to the variable 'dir' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 26), 'dir', tuple_var_assignment_1429_1667)
        
        # Assigning a Name to a Name (line 118):
        # Getting the type of 'tuple_var_assignment_1430' (line 118)
        tuple_var_assignment_1430_1668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'tuple_var_assignment_1430')
        # Assigning a type to the variable 'dir_pattern' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 31), 'dir_pattern', tuple_var_assignment_1430_1668)
        
        
        # Getting the type of 'action' (line 123)
        action_1669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'action')
        str_1670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 21), 'str', 'include')
        # Applying the binary operator '==' (line 123)
        result_eq_1671 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 11), '==', action_1669, str_1670)
        
        # Testing the type of an if condition (line 123)
        if_condition_1672 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 8), result_eq_1671)
        # Assigning a type to the variable 'if_condition_1672' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'if_condition_1672', if_condition_1672)
        # SSA begins for if statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to debug_print(...): (line 124)
        # Processing the call arguments (line 124)
        str_1675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 29), 'str', 'include ')
        
        # Call to join(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'patterns' (line 124)
        patterns_1678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 51), 'patterns', False)
        # Processing the call keyword arguments (line 124)
        kwargs_1679 = {}
        str_1676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 42), 'str', ' ')
        # Obtaining the member 'join' of a type (line 124)
        join_1677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 42), str_1676, 'join')
        # Calling join(args, kwargs) (line 124)
        join_call_result_1680 = invoke(stypy.reporting.localization.Localization(__file__, 124, 42), join_1677, *[patterns_1678], **kwargs_1679)
        
        # Applying the binary operator '+' (line 124)
        result_add_1681 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 29), '+', str_1675, join_call_result_1680)
        
        # Processing the call keyword arguments (line 124)
        kwargs_1682 = {}
        # Getting the type of 'self' (line 124)
        self_1673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'self', False)
        # Obtaining the member 'debug_print' of a type (line 124)
        debug_print_1674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 12), self_1673, 'debug_print')
        # Calling debug_print(args, kwargs) (line 124)
        debug_print_call_result_1683 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), debug_print_1674, *[result_add_1681], **kwargs_1682)
        
        
        # Getting the type of 'patterns' (line 125)
        patterns_1684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 27), 'patterns')
        # Testing the type of a for loop iterable (line 125)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 125, 12), patterns_1684)
        # Getting the type of the for loop variable (line 125)
        for_loop_var_1685 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 125, 12), patterns_1684)
        # Assigning a type to the variable 'pattern' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'pattern', for_loop_var_1685)
        # SSA begins for a for statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to include_pattern(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'pattern' (line 126)
        pattern_1688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 44), 'pattern', False)
        # Processing the call keyword arguments (line 126)
        int_1689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 60), 'int')
        keyword_1690 = int_1689
        kwargs_1691 = {'anchor': keyword_1690}
        # Getting the type of 'self' (line 126)
        self_1686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'self', False)
        # Obtaining the member 'include_pattern' of a type (line 126)
        include_pattern_1687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 23), self_1686, 'include_pattern')
        # Calling include_pattern(args, kwargs) (line 126)
        include_pattern_call_result_1692 = invoke(stypy.reporting.localization.Localization(__file__, 126, 23), include_pattern_1687, *[pattern_1688], **kwargs_1691)
        
        # Applying the 'not' unary operator (line 126)
        result_not__1693 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 19), 'not', include_pattern_call_result_1692)
        
        # Testing the type of an if condition (line 126)
        if_condition_1694 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 16), result_not__1693)
        # Assigning a type to the variable 'if_condition_1694' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'if_condition_1694', if_condition_1694)
        # SSA begins for if statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 127)
        # Processing the call arguments (line 127)
        str_1697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 29), 'str', "warning: no files found matching '%s'")
        # Getting the type of 'pattern' (line 128)
        pattern_1698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 29), 'pattern', False)
        # Processing the call keyword arguments (line 127)
        kwargs_1699 = {}
        # Getting the type of 'log' (line 127)
        log_1695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 20), 'log', False)
        # Obtaining the member 'warn' of a type (line 127)
        warn_1696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 20), log_1695, 'warn')
        # Calling warn(args, kwargs) (line 127)
        warn_call_result_1700 = invoke(stypy.reporting.localization.Localization(__file__, 127, 20), warn_1696, *[str_1697, pattern_1698], **kwargs_1699)
        
        # SSA join for if statement (line 126)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 123)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'action' (line 130)
        action_1701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 13), 'action')
        str_1702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 23), 'str', 'exclude')
        # Applying the binary operator '==' (line 130)
        result_eq_1703 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 13), '==', action_1701, str_1702)
        
        # Testing the type of an if condition (line 130)
        if_condition_1704 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 13), result_eq_1703)
        # Assigning a type to the variable 'if_condition_1704' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 13), 'if_condition_1704', if_condition_1704)
        # SSA begins for if statement (line 130)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to debug_print(...): (line 131)
        # Processing the call arguments (line 131)
        str_1707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 29), 'str', 'exclude ')
        
        # Call to join(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'patterns' (line 131)
        patterns_1710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 51), 'patterns', False)
        # Processing the call keyword arguments (line 131)
        kwargs_1711 = {}
        str_1708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 42), 'str', ' ')
        # Obtaining the member 'join' of a type (line 131)
        join_1709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 42), str_1708, 'join')
        # Calling join(args, kwargs) (line 131)
        join_call_result_1712 = invoke(stypy.reporting.localization.Localization(__file__, 131, 42), join_1709, *[patterns_1710], **kwargs_1711)
        
        # Applying the binary operator '+' (line 131)
        result_add_1713 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 29), '+', str_1707, join_call_result_1712)
        
        # Processing the call keyword arguments (line 131)
        kwargs_1714 = {}
        # Getting the type of 'self' (line 131)
        self_1705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'self', False)
        # Obtaining the member 'debug_print' of a type (line 131)
        debug_print_1706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), self_1705, 'debug_print')
        # Calling debug_print(args, kwargs) (line 131)
        debug_print_call_result_1715 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), debug_print_1706, *[result_add_1713], **kwargs_1714)
        
        
        # Getting the type of 'patterns' (line 132)
        patterns_1716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 27), 'patterns')
        # Testing the type of a for loop iterable (line 132)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 132, 12), patterns_1716)
        # Getting the type of the for loop variable (line 132)
        for_loop_var_1717 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 132, 12), patterns_1716)
        # Assigning a type to the variable 'pattern' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'pattern', for_loop_var_1717)
        # SSA begins for a for statement (line 132)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to exclude_pattern(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'pattern' (line 133)
        pattern_1720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 44), 'pattern', False)
        # Processing the call keyword arguments (line 133)
        int_1721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 60), 'int')
        keyword_1722 = int_1721
        kwargs_1723 = {'anchor': keyword_1722}
        # Getting the type of 'self' (line 133)
        self_1718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'self', False)
        # Obtaining the member 'exclude_pattern' of a type (line 133)
        exclude_pattern_1719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 23), self_1718, 'exclude_pattern')
        # Calling exclude_pattern(args, kwargs) (line 133)
        exclude_pattern_call_result_1724 = invoke(stypy.reporting.localization.Localization(__file__, 133, 23), exclude_pattern_1719, *[pattern_1720], **kwargs_1723)
        
        # Applying the 'not' unary operator (line 133)
        result_not__1725 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 19), 'not', exclude_pattern_call_result_1724)
        
        # Testing the type of an if condition (line 133)
        if_condition_1726 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 16), result_not__1725)
        # Assigning a type to the variable 'if_condition_1726' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'if_condition_1726', if_condition_1726)
        # SSA begins for if statement (line 133)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 134)
        # Processing the call arguments (line 134)
        str_1729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 30), 'str', "warning: no previously-included files found matching '%s'")
        # Getting the type of 'pattern' (line 135)
        pattern_1730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 54), 'pattern', False)
        # Processing the call keyword arguments (line 134)
        kwargs_1731 = {}
        # Getting the type of 'log' (line 134)
        log_1727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 20), 'log', False)
        # Obtaining the member 'warn' of a type (line 134)
        warn_1728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 20), log_1727, 'warn')
        # Calling warn(args, kwargs) (line 134)
        warn_call_result_1732 = invoke(stypy.reporting.localization.Localization(__file__, 134, 20), warn_1728, *[str_1729, pattern_1730], **kwargs_1731)
        
        # SSA join for if statement (line 133)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 130)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'action' (line 137)
        action_1733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 13), 'action')
        str_1734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 23), 'str', 'global-include')
        # Applying the binary operator '==' (line 137)
        result_eq_1735 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 13), '==', action_1733, str_1734)
        
        # Testing the type of an if condition (line 137)
        if_condition_1736 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 13), result_eq_1735)
        # Assigning a type to the variable 'if_condition_1736' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 13), 'if_condition_1736', if_condition_1736)
        # SSA begins for if statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to debug_print(...): (line 138)
        # Processing the call arguments (line 138)
        str_1739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 29), 'str', 'global-include ')
        
        # Call to join(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'patterns' (line 138)
        patterns_1742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 58), 'patterns', False)
        # Processing the call keyword arguments (line 138)
        kwargs_1743 = {}
        str_1740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 49), 'str', ' ')
        # Obtaining the member 'join' of a type (line 138)
        join_1741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 49), str_1740, 'join')
        # Calling join(args, kwargs) (line 138)
        join_call_result_1744 = invoke(stypy.reporting.localization.Localization(__file__, 138, 49), join_1741, *[patterns_1742], **kwargs_1743)
        
        # Applying the binary operator '+' (line 138)
        result_add_1745 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 29), '+', str_1739, join_call_result_1744)
        
        # Processing the call keyword arguments (line 138)
        kwargs_1746 = {}
        # Getting the type of 'self' (line 138)
        self_1737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'self', False)
        # Obtaining the member 'debug_print' of a type (line 138)
        debug_print_1738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), self_1737, 'debug_print')
        # Calling debug_print(args, kwargs) (line 138)
        debug_print_call_result_1747 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), debug_print_1738, *[result_add_1745], **kwargs_1746)
        
        
        # Getting the type of 'patterns' (line 139)
        patterns_1748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 27), 'patterns')
        # Testing the type of a for loop iterable (line 139)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 139, 12), patterns_1748)
        # Getting the type of the for loop variable (line 139)
        for_loop_var_1749 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 139, 12), patterns_1748)
        # Assigning a type to the variable 'pattern' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'pattern', for_loop_var_1749)
        # SSA begins for a for statement (line 139)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to include_pattern(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'pattern' (line 140)
        pattern_1752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 44), 'pattern', False)
        # Processing the call keyword arguments (line 140)
        int_1753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 60), 'int')
        keyword_1754 = int_1753
        kwargs_1755 = {'anchor': keyword_1754}
        # Getting the type of 'self' (line 140)
        self_1750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 23), 'self', False)
        # Obtaining the member 'include_pattern' of a type (line 140)
        include_pattern_1751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 23), self_1750, 'include_pattern')
        # Calling include_pattern(args, kwargs) (line 140)
        include_pattern_call_result_1756 = invoke(stypy.reporting.localization.Localization(__file__, 140, 23), include_pattern_1751, *[pattern_1752], **kwargs_1755)
        
        # Applying the 'not' unary operator (line 140)
        result_not__1757 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 19), 'not', include_pattern_call_result_1756)
        
        # Testing the type of an if condition (line 140)
        if_condition_1758 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 16), result_not__1757)
        # Assigning a type to the variable 'if_condition_1758' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'if_condition_1758', if_condition_1758)
        # SSA begins for if statement (line 140)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 141)
        # Processing the call arguments (line 141)
        str_1761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 30), 'str', "warning: no files found matching '%s' ")
        str_1762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 30), 'str', 'anywhere in distribution')
        # Applying the binary operator '+' (line 141)
        result_add_1763 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 30), '+', str_1761, str_1762)
        
        # Getting the type of 'pattern' (line 142)
        pattern_1764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 59), 'pattern', False)
        # Processing the call keyword arguments (line 141)
        kwargs_1765 = {}
        # Getting the type of 'log' (line 141)
        log_1759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'log', False)
        # Obtaining the member 'warn' of a type (line 141)
        warn_1760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 20), log_1759, 'warn')
        # Calling warn(args, kwargs) (line 141)
        warn_call_result_1766 = invoke(stypy.reporting.localization.Localization(__file__, 141, 20), warn_1760, *[result_add_1763, pattern_1764], **kwargs_1765)
        
        # SSA join for if statement (line 140)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 137)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'action' (line 144)
        action_1767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 13), 'action')
        str_1768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 23), 'str', 'global-exclude')
        # Applying the binary operator '==' (line 144)
        result_eq_1769 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 13), '==', action_1767, str_1768)
        
        # Testing the type of an if condition (line 144)
        if_condition_1770 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 13), result_eq_1769)
        # Assigning a type to the variable 'if_condition_1770' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 13), 'if_condition_1770', if_condition_1770)
        # SSA begins for if statement (line 144)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to debug_print(...): (line 145)
        # Processing the call arguments (line 145)
        str_1773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 29), 'str', 'global-exclude ')
        
        # Call to join(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'patterns' (line 145)
        patterns_1776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 58), 'patterns', False)
        # Processing the call keyword arguments (line 145)
        kwargs_1777 = {}
        str_1774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 49), 'str', ' ')
        # Obtaining the member 'join' of a type (line 145)
        join_1775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 49), str_1774, 'join')
        # Calling join(args, kwargs) (line 145)
        join_call_result_1778 = invoke(stypy.reporting.localization.Localization(__file__, 145, 49), join_1775, *[patterns_1776], **kwargs_1777)
        
        # Applying the binary operator '+' (line 145)
        result_add_1779 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 29), '+', str_1773, join_call_result_1778)
        
        # Processing the call keyword arguments (line 145)
        kwargs_1780 = {}
        # Getting the type of 'self' (line 145)
        self_1771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'self', False)
        # Obtaining the member 'debug_print' of a type (line 145)
        debug_print_1772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 12), self_1771, 'debug_print')
        # Calling debug_print(args, kwargs) (line 145)
        debug_print_call_result_1781 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), debug_print_1772, *[result_add_1779], **kwargs_1780)
        
        
        # Getting the type of 'patterns' (line 146)
        patterns_1782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 27), 'patterns')
        # Testing the type of a for loop iterable (line 146)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 146, 12), patterns_1782)
        # Getting the type of the for loop variable (line 146)
        for_loop_var_1783 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 146, 12), patterns_1782)
        # Assigning a type to the variable 'pattern' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'pattern', for_loop_var_1783)
        # SSA begins for a for statement (line 146)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to exclude_pattern(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'pattern' (line 147)
        pattern_1786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 44), 'pattern', False)
        # Processing the call keyword arguments (line 147)
        int_1787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 60), 'int')
        keyword_1788 = int_1787
        kwargs_1789 = {'anchor': keyword_1788}
        # Getting the type of 'self' (line 147)
        self_1784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 23), 'self', False)
        # Obtaining the member 'exclude_pattern' of a type (line 147)
        exclude_pattern_1785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 23), self_1784, 'exclude_pattern')
        # Calling exclude_pattern(args, kwargs) (line 147)
        exclude_pattern_call_result_1790 = invoke(stypy.reporting.localization.Localization(__file__, 147, 23), exclude_pattern_1785, *[pattern_1786], **kwargs_1789)
        
        # Applying the 'not' unary operator (line 147)
        result_not__1791 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 19), 'not', exclude_pattern_call_result_1790)
        
        # Testing the type of an if condition (line 147)
        if_condition_1792 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 16), result_not__1791)
        # Assigning a type to the variable 'if_condition_1792' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'if_condition_1792', if_condition_1792)
        # SSA begins for if statement (line 147)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 148)
        # Processing the call arguments (line 148)
        str_1795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 30), 'str', "warning: no previously-included files matching '%s' found anywhere in distribution")
        # Getting the type of 'pattern' (line 150)
        pattern_1796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 29), 'pattern', False)
        # Processing the call keyword arguments (line 148)
        kwargs_1797 = {}
        # Getting the type of 'log' (line 148)
        log_1793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'log', False)
        # Obtaining the member 'warn' of a type (line 148)
        warn_1794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 20), log_1793, 'warn')
        # Calling warn(args, kwargs) (line 148)
        warn_call_result_1798 = invoke(stypy.reporting.localization.Localization(__file__, 148, 20), warn_1794, *[str_1795, pattern_1796], **kwargs_1797)
        
        # SSA join for if statement (line 147)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 144)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'action' (line 152)
        action_1799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 13), 'action')
        str_1800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 23), 'str', 'recursive-include')
        # Applying the binary operator '==' (line 152)
        result_eq_1801 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 13), '==', action_1799, str_1800)
        
        # Testing the type of an if condition (line 152)
        if_condition_1802 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 13), result_eq_1801)
        # Assigning a type to the variable 'if_condition_1802' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 13), 'if_condition_1802', if_condition_1802)
        # SSA begins for if statement (line 152)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to debug_print(...): (line 153)
        # Processing the call arguments (line 153)
        str_1805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 29), 'str', 'recursive-include %s %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 154)
        tuple_1806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 154)
        # Adding element type (line 154)
        # Getting the type of 'dir' (line 154)
        dir_1807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 30), 'dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 30), tuple_1806, dir_1807)
        # Adding element type (line 154)
        
        # Call to join(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'patterns' (line 154)
        patterns_1810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 44), 'patterns', False)
        # Processing the call keyword arguments (line 154)
        kwargs_1811 = {}
        str_1808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 35), 'str', ' ')
        # Obtaining the member 'join' of a type (line 154)
        join_1809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 35), str_1808, 'join')
        # Calling join(args, kwargs) (line 154)
        join_call_result_1812 = invoke(stypy.reporting.localization.Localization(__file__, 154, 35), join_1809, *[patterns_1810], **kwargs_1811)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 30), tuple_1806, join_call_result_1812)
        
        # Applying the binary operator '%' (line 153)
        result_mod_1813 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 29), '%', str_1805, tuple_1806)
        
        # Processing the call keyword arguments (line 153)
        kwargs_1814 = {}
        # Getting the type of 'self' (line 153)
        self_1803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'self', False)
        # Obtaining the member 'debug_print' of a type (line 153)
        debug_print_1804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 12), self_1803, 'debug_print')
        # Calling debug_print(args, kwargs) (line 153)
        debug_print_call_result_1815 = invoke(stypy.reporting.localization.Localization(__file__, 153, 12), debug_print_1804, *[result_mod_1813], **kwargs_1814)
        
        
        # Getting the type of 'patterns' (line 155)
        patterns_1816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 27), 'patterns')
        # Testing the type of a for loop iterable (line 155)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 155, 12), patterns_1816)
        # Getting the type of the for loop variable (line 155)
        for_loop_var_1817 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 155, 12), patterns_1816)
        # Assigning a type to the variable 'pattern' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'pattern', for_loop_var_1817)
        # SSA begins for a for statement (line 155)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to include_pattern(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'pattern' (line 156)
        pattern_1820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 44), 'pattern', False)
        # Processing the call keyword arguments (line 156)
        # Getting the type of 'dir' (line 156)
        dir_1821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 60), 'dir', False)
        keyword_1822 = dir_1821
        kwargs_1823 = {'prefix': keyword_1822}
        # Getting the type of 'self' (line 156)
        self_1818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 23), 'self', False)
        # Obtaining the member 'include_pattern' of a type (line 156)
        include_pattern_1819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 23), self_1818, 'include_pattern')
        # Calling include_pattern(args, kwargs) (line 156)
        include_pattern_call_result_1824 = invoke(stypy.reporting.localization.Localization(__file__, 156, 23), include_pattern_1819, *[pattern_1820], **kwargs_1823)
        
        # Applying the 'not' unary operator (line 156)
        result_not__1825 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 19), 'not', include_pattern_call_result_1824)
        
        # Testing the type of an if condition (line 156)
        if_condition_1826 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 16), result_not__1825)
        # Assigning a type to the variable 'if_condition_1826' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'if_condition_1826', if_condition_1826)
        # SSA begins for if statement (line 156)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 157)
        # Processing the call arguments (line 157)
        str_1829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 30), 'str', "warning: no files found matching '%s' ")
        str_1830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 32), 'str', "under directory '%s'")
        # Applying the binary operator '+' (line 157)
        result_add_1831 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 30), '+', str_1829, str_1830)
        
        # Getting the type of 'pattern' (line 159)
        pattern_1832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 29), 'pattern', False)
        # Getting the type of 'dir' (line 159)
        dir_1833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 38), 'dir', False)
        # Processing the call keyword arguments (line 157)
        kwargs_1834 = {}
        # Getting the type of 'log' (line 157)
        log_1827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'log', False)
        # Obtaining the member 'warn' of a type (line 157)
        warn_1828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 20), log_1827, 'warn')
        # Calling warn(args, kwargs) (line 157)
        warn_call_result_1835 = invoke(stypy.reporting.localization.Localization(__file__, 157, 20), warn_1828, *[result_add_1831, pattern_1832, dir_1833], **kwargs_1834)
        
        # SSA join for if statement (line 156)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 152)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'action' (line 161)
        action_1836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 13), 'action')
        str_1837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 23), 'str', 'recursive-exclude')
        # Applying the binary operator '==' (line 161)
        result_eq_1838 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 13), '==', action_1836, str_1837)
        
        # Testing the type of an if condition (line 161)
        if_condition_1839 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 13), result_eq_1838)
        # Assigning a type to the variable 'if_condition_1839' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 13), 'if_condition_1839', if_condition_1839)
        # SSA begins for if statement (line 161)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to debug_print(...): (line 162)
        # Processing the call arguments (line 162)
        str_1842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 29), 'str', 'recursive-exclude %s %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 163)
        tuple_1843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 163)
        # Adding element type (line 163)
        # Getting the type of 'dir' (line 163)
        dir_1844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 30), 'dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 30), tuple_1843, dir_1844)
        # Adding element type (line 163)
        
        # Call to join(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'patterns' (line 163)
        patterns_1847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 44), 'patterns', False)
        # Processing the call keyword arguments (line 163)
        kwargs_1848 = {}
        str_1845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 35), 'str', ' ')
        # Obtaining the member 'join' of a type (line 163)
        join_1846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 35), str_1845, 'join')
        # Calling join(args, kwargs) (line 163)
        join_call_result_1849 = invoke(stypy.reporting.localization.Localization(__file__, 163, 35), join_1846, *[patterns_1847], **kwargs_1848)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 30), tuple_1843, join_call_result_1849)
        
        # Applying the binary operator '%' (line 162)
        result_mod_1850 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 29), '%', str_1842, tuple_1843)
        
        # Processing the call keyword arguments (line 162)
        kwargs_1851 = {}
        # Getting the type of 'self' (line 162)
        self_1840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'self', False)
        # Obtaining the member 'debug_print' of a type (line 162)
        debug_print_1841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), self_1840, 'debug_print')
        # Calling debug_print(args, kwargs) (line 162)
        debug_print_call_result_1852 = invoke(stypy.reporting.localization.Localization(__file__, 162, 12), debug_print_1841, *[result_mod_1850], **kwargs_1851)
        
        
        # Getting the type of 'patterns' (line 164)
        patterns_1853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 27), 'patterns')
        # Testing the type of a for loop iterable (line 164)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 164, 12), patterns_1853)
        # Getting the type of the for loop variable (line 164)
        for_loop_var_1854 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 164, 12), patterns_1853)
        # Assigning a type to the variable 'pattern' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'pattern', for_loop_var_1854)
        # SSA begins for a for statement (line 164)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to exclude_pattern(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'pattern' (line 165)
        pattern_1857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 44), 'pattern', False)
        # Processing the call keyword arguments (line 165)
        # Getting the type of 'dir' (line 165)
        dir_1858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 60), 'dir', False)
        keyword_1859 = dir_1858
        kwargs_1860 = {'prefix': keyword_1859}
        # Getting the type of 'self' (line 165)
        self_1855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 23), 'self', False)
        # Obtaining the member 'exclude_pattern' of a type (line 165)
        exclude_pattern_1856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 23), self_1855, 'exclude_pattern')
        # Calling exclude_pattern(args, kwargs) (line 165)
        exclude_pattern_call_result_1861 = invoke(stypy.reporting.localization.Localization(__file__, 165, 23), exclude_pattern_1856, *[pattern_1857], **kwargs_1860)
        
        # Applying the 'not' unary operator (line 165)
        result_not__1862 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 19), 'not', exclude_pattern_call_result_1861)
        
        # Testing the type of an if condition (line 165)
        if_condition_1863 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 16), result_not__1862)
        # Assigning a type to the variable 'if_condition_1863' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'if_condition_1863', if_condition_1863)
        # SSA begins for if statement (line 165)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 166)
        # Processing the call arguments (line 166)
        str_1866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 30), 'str', "warning: no previously-included files matching '%s' found under directory '%s'")
        # Getting the type of 'pattern' (line 168)
        pattern_1867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 29), 'pattern', False)
        # Getting the type of 'dir' (line 168)
        dir_1868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 38), 'dir', False)
        # Processing the call keyword arguments (line 166)
        kwargs_1869 = {}
        # Getting the type of 'log' (line 166)
        log_1864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 20), 'log', False)
        # Obtaining the member 'warn' of a type (line 166)
        warn_1865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 20), log_1864, 'warn')
        # Calling warn(args, kwargs) (line 166)
        warn_call_result_1870 = invoke(stypy.reporting.localization.Localization(__file__, 166, 20), warn_1865, *[str_1866, pattern_1867, dir_1868], **kwargs_1869)
        
        # SSA join for if statement (line 165)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 161)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'action' (line 170)
        action_1871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 13), 'action')
        str_1872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 23), 'str', 'graft')
        # Applying the binary operator '==' (line 170)
        result_eq_1873 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 13), '==', action_1871, str_1872)
        
        # Testing the type of an if condition (line 170)
        if_condition_1874 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 13), result_eq_1873)
        # Assigning a type to the variable 'if_condition_1874' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 13), 'if_condition_1874', if_condition_1874)
        # SSA begins for if statement (line 170)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to debug_print(...): (line 171)
        # Processing the call arguments (line 171)
        str_1877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 29), 'str', 'graft ')
        # Getting the type of 'dir_pattern' (line 171)
        dir_pattern_1878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 40), 'dir_pattern', False)
        # Applying the binary operator '+' (line 171)
        result_add_1879 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 29), '+', str_1877, dir_pattern_1878)
        
        # Processing the call keyword arguments (line 171)
        kwargs_1880 = {}
        # Getting the type of 'self' (line 171)
        self_1875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'self', False)
        # Obtaining the member 'debug_print' of a type (line 171)
        debug_print_1876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 12), self_1875, 'debug_print')
        # Calling debug_print(args, kwargs) (line 171)
        debug_print_call_result_1881 = invoke(stypy.reporting.localization.Localization(__file__, 171, 12), debug_print_1876, *[result_add_1879], **kwargs_1880)
        
        
        
        
        # Call to include_pattern(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'None' (line 172)
        None_1884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 40), 'None', False)
        # Processing the call keyword arguments (line 172)
        # Getting the type of 'dir_pattern' (line 172)
        dir_pattern_1885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 53), 'dir_pattern', False)
        keyword_1886 = dir_pattern_1885
        kwargs_1887 = {'prefix': keyword_1886}
        # Getting the type of 'self' (line 172)
        self_1882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 19), 'self', False)
        # Obtaining the member 'include_pattern' of a type (line 172)
        include_pattern_1883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 19), self_1882, 'include_pattern')
        # Calling include_pattern(args, kwargs) (line 172)
        include_pattern_call_result_1888 = invoke(stypy.reporting.localization.Localization(__file__, 172, 19), include_pattern_1883, *[None_1884], **kwargs_1887)
        
        # Applying the 'not' unary operator (line 172)
        result_not__1889 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 15), 'not', include_pattern_call_result_1888)
        
        # Testing the type of an if condition (line 172)
        if_condition_1890 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 12), result_not__1889)
        # Assigning a type to the variable 'if_condition_1890' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'if_condition_1890', if_condition_1890)
        # SSA begins for if statement (line 172)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 173)
        # Processing the call arguments (line 173)
        str_1893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 25), 'str', "warning: no directories found matching '%s'")
        # Getting the type of 'dir_pattern' (line 174)
        dir_pattern_1894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 25), 'dir_pattern', False)
        # Processing the call keyword arguments (line 173)
        kwargs_1895 = {}
        # Getting the type of 'log' (line 173)
        log_1891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'log', False)
        # Obtaining the member 'warn' of a type (line 173)
        warn_1892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 16), log_1891, 'warn')
        # Calling warn(args, kwargs) (line 173)
        warn_call_result_1896 = invoke(stypy.reporting.localization.Localization(__file__, 173, 16), warn_1892, *[str_1893, dir_pattern_1894], **kwargs_1895)
        
        # SSA join for if statement (line 172)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 170)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'action' (line 176)
        action_1897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 13), 'action')
        str_1898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 23), 'str', 'prune')
        # Applying the binary operator '==' (line 176)
        result_eq_1899 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 13), '==', action_1897, str_1898)
        
        # Testing the type of an if condition (line 176)
        if_condition_1900 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 13), result_eq_1899)
        # Assigning a type to the variable 'if_condition_1900' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 13), 'if_condition_1900', if_condition_1900)
        # SSA begins for if statement (line 176)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to debug_print(...): (line 177)
        # Processing the call arguments (line 177)
        str_1903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 29), 'str', 'prune ')
        # Getting the type of 'dir_pattern' (line 177)
        dir_pattern_1904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 40), 'dir_pattern', False)
        # Applying the binary operator '+' (line 177)
        result_add_1905 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 29), '+', str_1903, dir_pattern_1904)
        
        # Processing the call keyword arguments (line 177)
        kwargs_1906 = {}
        # Getting the type of 'self' (line 177)
        self_1901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'self', False)
        # Obtaining the member 'debug_print' of a type (line 177)
        debug_print_1902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 12), self_1901, 'debug_print')
        # Calling debug_print(args, kwargs) (line 177)
        debug_print_call_result_1907 = invoke(stypy.reporting.localization.Localization(__file__, 177, 12), debug_print_1902, *[result_add_1905], **kwargs_1906)
        
        
        
        
        # Call to exclude_pattern(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'None' (line 178)
        None_1910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 40), 'None', False)
        # Processing the call keyword arguments (line 178)
        # Getting the type of 'dir_pattern' (line 178)
        dir_pattern_1911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 53), 'dir_pattern', False)
        keyword_1912 = dir_pattern_1911
        kwargs_1913 = {'prefix': keyword_1912}
        # Getting the type of 'self' (line 178)
        self_1908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 19), 'self', False)
        # Obtaining the member 'exclude_pattern' of a type (line 178)
        exclude_pattern_1909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 19), self_1908, 'exclude_pattern')
        # Calling exclude_pattern(args, kwargs) (line 178)
        exclude_pattern_call_result_1914 = invoke(stypy.reporting.localization.Localization(__file__, 178, 19), exclude_pattern_1909, *[None_1910], **kwargs_1913)
        
        # Applying the 'not' unary operator (line 178)
        result_not__1915 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 15), 'not', exclude_pattern_call_result_1914)
        
        # Testing the type of an if condition (line 178)
        if_condition_1916 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 12), result_not__1915)
        # Assigning a type to the variable 'if_condition_1916' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'if_condition_1916', if_condition_1916)
        # SSA begins for if statement (line 178)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 179)
        # Processing the call arguments (line 179)
        str_1919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 26), 'str', 'no previously-included directories found ')
        str_1920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 26), 'str', "matching '%s'")
        # Applying the binary operator '+' (line 179)
        result_add_1921 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 26), '+', str_1919, str_1920)
        
        # Getting the type of 'dir_pattern' (line 180)
        dir_pattern_1922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 44), 'dir_pattern', False)
        # Processing the call keyword arguments (line 179)
        kwargs_1923 = {}
        # Getting the type of 'log' (line 179)
        log_1917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'log', False)
        # Obtaining the member 'warn' of a type (line 179)
        warn_1918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 16), log_1917, 'warn')
        # Calling warn(args, kwargs) (line 179)
        warn_call_result_1924 = invoke(stypy.reporting.localization.Localization(__file__, 179, 16), warn_1918, *[result_add_1921, dir_pattern_1922], **kwargs_1923)
        
        # SSA join for if statement (line 178)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 176)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'DistutilsInternalError' (line 182)
        DistutilsInternalError_1925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 18), 'DistutilsInternalError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 182, 12), DistutilsInternalError_1925, 'raise parameter', BaseException)
        # SSA join for if statement (line 176)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 170)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 161)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 152)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 144)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 137)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 130)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 123)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'process_template_line(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'process_template_line' in the type store
        # Getting the type of 'stypy_return_type' (line 112)
        stypy_return_type_1926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1926)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'process_template_line'
        return stypy_return_type_1926


    @norecursion
    def include_pattern(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_1927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 46), 'int')
        # Getting the type of 'None' (line 187)
        None_1928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 56), 'None')
        int_1929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 71), 'int')
        defaults = [int_1927, None_1928, int_1929]
        # Create a new context for function 'include_pattern'
        module_type_store = module_type_store.open_function_context('include_pattern', 187, 4, False)
        # Assigning a type to the variable 'self' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileList.include_pattern.__dict__.__setitem__('stypy_localization', localization)
        FileList.include_pattern.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileList.include_pattern.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileList.include_pattern.__dict__.__setitem__('stypy_function_name', 'FileList.include_pattern')
        FileList.include_pattern.__dict__.__setitem__('stypy_param_names_list', ['pattern', 'anchor', 'prefix', 'is_regex'])
        FileList.include_pattern.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileList.include_pattern.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileList.include_pattern.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileList.include_pattern.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileList.include_pattern.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileList.include_pattern.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileList.include_pattern', ['pattern', 'anchor', 'prefix', 'is_regex'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'include_pattern', localization, ['pattern', 'anchor', 'prefix', 'is_regex'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'include_pattern(...)' code ##################

        str_1930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, (-1)), 'str', 'Select strings (presumably filenames) from \'self.files\' that\n        match \'pattern\', a Unix-style wildcard (glob) pattern.\n\n        Patterns are not quite the same as implemented by the \'fnmatch\'\n        module: \'*\' and \'?\'  match non-special characters, where "special"\n        is platform-dependent: slash on Unix; colon, slash, and backslash on\n        DOS/Windows; and colon on Mac OS.\n\n        If \'anchor\' is true (the default), then the pattern match is more\n        stringent: "*.py" will match "foo.py" but not "foo/bar.py".  If\n        \'anchor\' is false, both of these will match.\n\n        If \'prefix\' is supplied, then only filenames starting with \'prefix\'\n        (itself a pattern) and ending with \'pattern\', with anything in between\n        them, will match.  \'anchor\' is ignored in this case.\n\n        If \'is_regex\' is true, \'anchor\' and \'prefix\' are ignored, and\n        \'pattern\' is assumed to be either a string containing a regex or a\n        regex object -- no translation is done, the regex is just compiled\n        and used as-is.\n\n        Selected strings will be added to self.files.\n\n        Return 1 if files are found.\n        ')
        
        # Assigning a Num to a Name (line 214):
        
        # Assigning a Num to a Name (line 214):
        int_1931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 22), 'int')
        # Assigning a type to the variable 'files_found' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'files_found', int_1931)
        
        # Assigning a Call to a Name (line 215):
        
        # Assigning a Call to a Name (line 215):
        
        # Call to translate_pattern(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'pattern' (line 215)
        pattern_1933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 39), 'pattern', False)
        # Getting the type of 'anchor' (line 215)
        anchor_1934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 48), 'anchor', False)
        # Getting the type of 'prefix' (line 215)
        prefix_1935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 56), 'prefix', False)
        # Getting the type of 'is_regex' (line 215)
        is_regex_1936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 64), 'is_regex', False)
        # Processing the call keyword arguments (line 215)
        kwargs_1937 = {}
        # Getting the type of 'translate_pattern' (line 215)
        translate_pattern_1932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 21), 'translate_pattern', False)
        # Calling translate_pattern(args, kwargs) (line 215)
        translate_pattern_call_result_1938 = invoke(stypy.reporting.localization.Localization(__file__, 215, 21), translate_pattern_1932, *[pattern_1933, anchor_1934, prefix_1935, is_regex_1936], **kwargs_1937)
        
        # Assigning a type to the variable 'pattern_re' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'pattern_re', translate_pattern_call_result_1938)
        
        # Call to debug_print(...): (line 216)
        # Processing the call arguments (line 216)
        str_1941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 25), 'str', "include_pattern: applying regex r'%s'")
        # Getting the type of 'pattern_re' (line 217)
        pattern_re_1942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 25), 'pattern_re', False)
        # Obtaining the member 'pattern' of a type (line 217)
        pattern_1943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 25), pattern_re_1942, 'pattern')
        # Applying the binary operator '%' (line 216)
        result_mod_1944 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 25), '%', str_1941, pattern_1943)
        
        # Processing the call keyword arguments (line 216)
        kwargs_1945 = {}
        # Getting the type of 'self' (line 216)
        self_1939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'self', False)
        # Obtaining the member 'debug_print' of a type (line 216)
        debug_print_1940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), self_1939, 'debug_print')
        # Calling debug_print(args, kwargs) (line 216)
        debug_print_call_result_1946 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), debug_print_1940, *[result_mod_1944], **kwargs_1945)
        
        
        # Type idiom detected: calculating its left and rigth part (line 220)
        # Getting the type of 'self' (line 220)
        self_1947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 11), 'self')
        # Obtaining the member 'allfiles' of a type (line 220)
        allfiles_1948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 11), self_1947, 'allfiles')
        # Getting the type of 'None' (line 220)
        None_1949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 28), 'None')
        
        (may_be_1950, more_types_in_union_1951) = may_be_none(allfiles_1948, None_1949)

        if may_be_1950:

            if more_types_in_union_1951:
                # Runtime conditional SSA (line 220)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to findall(...): (line 221)
            # Processing the call keyword arguments (line 221)
            kwargs_1954 = {}
            # Getting the type of 'self' (line 221)
            self_1952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'self', False)
            # Obtaining the member 'findall' of a type (line 221)
            findall_1953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 12), self_1952, 'findall')
            # Calling findall(args, kwargs) (line 221)
            findall_call_result_1955 = invoke(stypy.reporting.localization.Localization(__file__, 221, 12), findall_1953, *[], **kwargs_1954)
            

            if more_types_in_union_1951:
                # SSA join for if statement (line 220)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'self' (line 223)
        self_1956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 20), 'self')
        # Obtaining the member 'allfiles' of a type (line 223)
        allfiles_1957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 20), self_1956, 'allfiles')
        # Testing the type of a for loop iterable (line 223)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 223, 8), allfiles_1957)
        # Getting the type of the for loop variable (line 223)
        for_loop_var_1958 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 223, 8), allfiles_1957)
        # Assigning a type to the variable 'name' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'name', for_loop_var_1958)
        # SSA begins for a for statement (line 223)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to search(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'name' (line 224)
        name_1961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 33), 'name', False)
        # Processing the call keyword arguments (line 224)
        kwargs_1962 = {}
        # Getting the type of 'pattern_re' (line 224)
        pattern_re_1959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 15), 'pattern_re', False)
        # Obtaining the member 'search' of a type (line 224)
        search_1960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 15), pattern_re_1959, 'search')
        # Calling search(args, kwargs) (line 224)
        search_call_result_1963 = invoke(stypy.reporting.localization.Localization(__file__, 224, 15), search_1960, *[name_1961], **kwargs_1962)
        
        # Testing the type of an if condition (line 224)
        if_condition_1964 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 224, 12), search_call_result_1963)
        # Assigning a type to the variable 'if_condition_1964' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'if_condition_1964', if_condition_1964)
        # SSA begins for if statement (line 224)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to debug_print(...): (line 225)
        # Processing the call arguments (line 225)
        str_1967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 33), 'str', ' adding ')
        # Getting the type of 'name' (line 225)
        name_1968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 46), 'name', False)
        # Applying the binary operator '+' (line 225)
        result_add_1969 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 33), '+', str_1967, name_1968)
        
        # Processing the call keyword arguments (line 225)
        kwargs_1970 = {}
        # Getting the type of 'self' (line 225)
        self_1965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'self', False)
        # Obtaining the member 'debug_print' of a type (line 225)
        debug_print_1966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 16), self_1965, 'debug_print')
        # Calling debug_print(args, kwargs) (line 225)
        debug_print_call_result_1971 = invoke(stypy.reporting.localization.Localization(__file__, 225, 16), debug_print_1966, *[result_add_1969], **kwargs_1970)
        
        
        # Call to append(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'name' (line 226)
        name_1975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 34), 'name', False)
        # Processing the call keyword arguments (line 226)
        kwargs_1976 = {}
        # Getting the type of 'self' (line 226)
        self_1972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), 'self', False)
        # Obtaining the member 'files' of a type (line 226)
        files_1973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 16), self_1972, 'files')
        # Obtaining the member 'append' of a type (line 226)
        append_1974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 16), files_1973, 'append')
        # Calling append(args, kwargs) (line 226)
        append_call_result_1977 = invoke(stypy.reporting.localization.Localization(__file__, 226, 16), append_1974, *[name_1975], **kwargs_1976)
        
        
        # Assigning a Num to a Name (line 227):
        
        # Assigning a Num to a Name (line 227):
        int_1978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 30), 'int')
        # Assigning a type to the variable 'files_found' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'files_found', int_1978)
        # SSA join for if statement (line 224)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'files_found' (line 229)
        files_found_1979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'files_found')
        # Assigning a type to the variable 'stypy_return_type' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'stypy_return_type', files_found_1979)
        
        # ################# End of 'include_pattern(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'include_pattern' in the type store
        # Getting the type of 'stypy_return_type' (line 187)
        stypy_return_type_1980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1980)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'include_pattern'
        return stypy_return_type_1980


    @norecursion
    def exclude_pattern(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_1981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 46), 'int')
        # Getting the type of 'None' (line 232)
        None_1982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 56), 'None')
        int_1983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 71), 'int')
        defaults = [int_1981, None_1982, int_1983]
        # Create a new context for function 'exclude_pattern'
        module_type_store = module_type_store.open_function_context('exclude_pattern', 232, 4, False)
        # Assigning a type to the variable 'self' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileList.exclude_pattern.__dict__.__setitem__('stypy_localization', localization)
        FileList.exclude_pattern.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileList.exclude_pattern.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileList.exclude_pattern.__dict__.__setitem__('stypy_function_name', 'FileList.exclude_pattern')
        FileList.exclude_pattern.__dict__.__setitem__('stypy_param_names_list', ['pattern', 'anchor', 'prefix', 'is_regex'])
        FileList.exclude_pattern.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileList.exclude_pattern.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileList.exclude_pattern.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileList.exclude_pattern.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileList.exclude_pattern.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileList.exclude_pattern.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileList.exclude_pattern', ['pattern', 'anchor', 'prefix', 'is_regex'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'exclude_pattern', localization, ['pattern', 'anchor', 'prefix', 'is_regex'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'exclude_pattern(...)' code ##################

        str_1984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, (-1)), 'str', "Remove strings (presumably filenames) from 'files' that match\n        'pattern'.\n\n        Other parameters are the same as for 'include_pattern()', above.\n        The list 'self.files' is modified in place. Return 1 if files are\n        found.\n        ")
        
        # Assigning a Num to a Name (line 240):
        
        # Assigning a Num to a Name (line 240):
        int_1985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 22), 'int')
        # Assigning a type to the variable 'files_found' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'files_found', int_1985)
        
        # Assigning a Call to a Name (line 241):
        
        # Assigning a Call to a Name (line 241):
        
        # Call to translate_pattern(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'pattern' (line 241)
        pattern_1987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 39), 'pattern', False)
        # Getting the type of 'anchor' (line 241)
        anchor_1988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 48), 'anchor', False)
        # Getting the type of 'prefix' (line 241)
        prefix_1989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 56), 'prefix', False)
        # Getting the type of 'is_regex' (line 241)
        is_regex_1990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 64), 'is_regex', False)
        # Processing the call keyword arguments (line 241)
        kwargs_1991 = {}
        # Getting the type of 'translate_pattern' (line 241)
        translate_pattern_1986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 21), 'translate_pattern', False)
        # Calling translate_pattern(args, kwargs) (line 241)
        translate_pattern_call_result_1992 = invoke(stypy.reporting.localization.Localization(__file__, 241, 21), translate_pattern_1986, *[pattern_1987, anchor_1988, prefix_1989, is_regex_1990], **kwargs_1991)
        
        # Assigning a type to the variable 'pattern_re' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'pattern_re', translate_pattern_call_result_1992)
        
        # Call to debug_print(...): (line 242)
        # Processing the call arguments (line 242)
        str_1995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 25), 'str', "exclude_pattern: applying regex r'%s'")
        # Getting the type of 'pattern_re' (line 243)
        pattern_re_1996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 25), 'pattern_re', False)
        # Obtaining the member 'pattern' of a type (line 243)
        pattern_1997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 25), pattern_re_1996, 'pattern')
        # Applying the binary operator '%' (line 242)
        result_mod_1998 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 25), '%', str_1995, pattern_1997)
        
        # Processing the call keyword arguments (line 242)
        kwargs_1999 = {}
        # Getting the type of 'self' (line 242)
        self_1993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'self', False)
        # Obtaining the member 'debug_print' of a type (line 242)
        debug_print_1994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), self_1993, 'debug_print')
        # Calling debug_print(args, kwargs) (line 242)
        debug_print_call_result_2000 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), debug_print_1994, *[result_mod_1998], **kwargs_1999)
        
        
        
        # Call to range(...): (line 244)
        # Processing the call arguments (line 244)
        
        # Call to len(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'self' (line 244)
        self_2003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 27), 'self', False)
        # Obtaining the member 'files' of a type (line 244)
        files_2004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 27), self_2003, 'files')
        # Processing the call keyword arguments (line 244)
        kwargs_2005 = {}
        # Getting the type of 'len' (line 244)
        len_2002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 23), 'len', False)
        # Calling len(args, kwargs) (line 244)
        len_call_result_2006 = invoke(stypy.reporting.localization.Localization(__file__, 244, 23), len_2002, *[files_2004], **kwargs_2005)
        
        int_2007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 39), 'int')
        # Applying the binary operator '-' (line 244)
        result_sub_2008 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 23), '-', len_call_result_2006, int_2007)
        
        int_2009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 42), 'int')
        int_2010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 46), 'int')
        # Processing the call keyword arguments (line 244)
        kwargs_2011 = {}
        # Getting the type of 'range' (line 244)
        range_2001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 17), 'range', False)
        # Calling range(args, kwargs) (line 244)
        range_call_result_2012 = invoke(stypy.reporting.localization.Localization(__file__, 244, 17), range_2001, *[result_sub_2008, int_2009, int_2010], **kwargs_2011)
        
        # Testing the type of a for loop iterable (line 244)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 244, 8), range_call_result_2012)
        # Getting the type of the for loop variable (line 244)
        for_loop_var_2013 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 244, 8), range_call_result_2012)
        # Assigning a type to the variable 'i' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'i', for_loop_var_2013)
        # SSA begins for a for statement (line 244)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to search(...): (line 245)
        # Processing the call arguments (line 245)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 245)
        i_2016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 44), 'i', False)
        # Getting the type of 'self' (line 245)
        self_2017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 33), 'self', False)
        # Obtaining the member 'files' of a type (line 245)
        files_2018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 33), self_2017, 'files')
        # Obtaining the member '__getitem__' of a type (line 245)
        getitem___2019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 33), files_2018, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 245)
        subscript_call_result_2020 = invoke(stypy.reporting.localization.Localization(__file__, 245, 33), getitem___2019, i_2016)
        
        # Processing the call keyword arguments (line 245)
        kwargs_2021 = {}
        # Getting the type of 'pattern_re' (line 245)
        pattern_re_2014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 15), 'pattern_re', False)
        # Obtaining the member 'search' of a type (line 245)
        search_2015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 15), pattern_re_2014, 'search')
        # Calling search(args, kwargs) (line 245)
        search_call_result_2022 = invoke(stypy.reporting.localization.Localization(__file__, 245, 15), search_2015, *[subscript_call_result_2020], **kwargs_2021)
        
        # Testing the type of an if condition (line 245)
        if_condition_2023 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 245, 12), search_call_result_2022)
        # Assigning a type to the variable 'if_condition_2023' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'if_condition_2023', if_condition_2023)
        # SSA begins for if statement (line 245)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to debug_print(...): (line 246)
        # Processing the call arguments (line 246)
        str_2026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 33), 'str', ' removing ')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 246)
        i_2027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 59), 'i', False)
        # Getting the type of 'self' (line 246)
        self_2028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 48), 'self', False)
        # Obtaining the member 'files' of a type (line 246)
        files_2029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 48), self_2028, 'files')
        # Obtaining the member '__getitem__' of a type (line 246)
        getitem___2030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 48), files_2029, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 246)
        subscript_call_result_2031 = invoke(stypy.reporting.localization.Localization(__file__, 246, 48), getitem___2030, i_2027)
        
        # Applying the binary operator '+' (line 246)
        result_add_2032 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 33), '+', str_2026, subscript_call_result_2031)
        
        # Processing the call keyword arguments (line 246)
        kwargs_2033 = {}
        # Getting the type of 'self' (line 246)
        self_2024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'self', False)
        # Obtaining the member 'debug_print' of a type (line 246)
        debug_print_2025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 16), self_2024, 'debug_print')
        # Calling debug_print(args, kwargs) (line 246)
        debug_print_call_result_2034 = invoke(stypy.reporting.localization.Localization(__file__, 246, 16), debug_print_2025, *[result_add_2032], **kwargs_2033)
        
        # Deleting a member
        # Getting the type of 'self' (line 247)
        self_2035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 20), 'self')
        # Obtaining the member 'files' of a type (line 247)
        files_2036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 20), self_2035, 'files')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 247)
        i_2037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 31), 'i')
        # Getting the type of 'self' (line 247)
        self_2038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 20), 'self')
        # Obtaining the member 'files' of a type (line 247)
        files_2039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 20), self_2038, 'files')
        # Obtaining the member '__getitem__' of a type (line 247)
        getitem___2040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 20), files_2039, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 247)
        subscript_call_result_2041 = invoke(stypy.reporting.localization.Localization(__file__, 247, 20), getitem___2040, i_2037)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 16), files_2036, subscript_call_result_2041)
        
        # Assigning a Num to a Name (line 248):
        
        # Assigning a Num to a Name (line 248):
        int_2042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 30), 'int')
        # Assigning a type to the variable 'files_found' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'files_found', int_2042)
        # SSA join for if statement (line 245)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'files_found' (line 250)
        files_found_2043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 15), 'files_found')
        # Assigning a type to the variable 'stypy_return_type' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'stypy_return_type', files_found_2043)
        
        # ################# End of 'exclude_pattern(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'exclude_pattern' in the type store
        # Getting the type of 'stypy_return_type' (line 232)
        stypy_return_type_2044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2044)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'exclude_pattern'
        return stypy_return_type_2044


# Assigning a type to the variable 'FileList' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'FileList', FileList)

@norecursion
def findall(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'os' (line 256)
    os_2045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 18), 'os')
    # Obtaining the member 'curdir' of a type (line 256)
    curdir_2046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 18), os_2045, 'curdir')
    defaults = [curdir_2046]
    # Create a new context for function 'findall'
    module_type_store = module_type_store.open_function_context('findall', 256, 0, False)
    
    # Passed parameters checking function
    findall.stypy_localization = localization
    findall.stypy_type_of_self = None
    findall.stypy_type_store = module_type_store
    findall.stypy_function_name = 'findall'
    findall.stypy_param_names_list = ['dir']
    findall.stypy_varargs_param_name = None
    findall.stypy_kwargs_param_name = None
    findall.stypy_call_defaults = defaults
    findall.stypy_call_varargs = varargs
    findall.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'findall', ['dir'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'findall', localization, ['dir'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'findall(...)' code ##################

    str_2047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, (-1)), 'str', "Find all files under 'dir' and return the list of full filenames\n    (relative to 'dir').\n    ")
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 260, 4))
    
    # 'from stat import ST_MODE, S_ISREG, S_ISDIR, S_ISLNK' statement (line 260)
    try:
        from stat import ST_MODE, S_ISREG, S_ISDIR, S_ISLNK

    except:
        ST_MODE = UndefinedType
        S_ISREG = UndefinedType
        S_ISDIR = UndefinedType
        S_ISLNK = UndefinedType
    import_from_module(stypy.reporting.localization.Localization(__file__, 260, 4), 'stat', None, module_type_store, ['ST_MODE', 'S_ISREG', 'S_ISDIR', 'S_ISLNK'], [ST_MODE, S_ISREG, S_ISDIR, S_ISLNK])
    
    
    # Assigning a List to a Name (line 262):
    
    # Assigning a List to a Name (line 262):
    
    # Obtaining an instance of the builtin type 'list' (line 262)
    list_2048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 262)
    
    # Assigning a type to the variable 'list' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'list', list_2048)
    
    # Assigning a List to a Name (line 263):
    
    # Assigning a List to a Name (line 263):
    
    # Obtaining an instance of the builtin type 'list' (line 263)
    list_2049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 263)
    # Adding element type (line 263)
    # Getting the type of 'dir' (line 263)
    dir_2050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 13), 'dir')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 12), list_2049, dir_2050)
    
    # Assigning a type to the variable 'stack' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'stack', list_2049)
    
    # Assigning a Attribute to a Name (line 264):
    
    # Assigning a Attribute to a Name (line 264):
    # Getting the type of 'stack' (line 264)
    stack_2051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 10), 'stack')
    # Obtaining the member 'pop' of a type (line 264)
    pop_2052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 10), stack_2051, 'pop')
    # Assigning a type to the variable 'pop' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'pop', pop_2052)
    
    # Assigning a Attribute to a Name (line 265):
    
    # Assigning a Attribute to a Name (line 265):
    # Getting the type of 'stack' (line 265)
    stack_2053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 11), 'stack')
    # Obtaining the member 'append' of a type (line 265)
    append_2054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 11), stack_2053, 'append')
    # Assigning a type to the variable 'push' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'push', append_2054)
    
    # Getting the type of 'stack' (line 267)
    stack_2055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 10), 'stack')
    # Testing the type of an if condition (line 267)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 267, 4), stack_2055)
    # SSA begins for while statement (line 267)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 268):
    
    # Assigning a Call to a Name (line 268):
    
    # Call to pop(...): (line 268)
    # Processing the call keyword arguments (line 268)
    kwargs_2057 = {}
    # Getting the type of 'pop' (line 268)
    pop_2056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 14), 'pop', False)
    # Calling pop(args, kwargs) (line 268)
    pop_call_result_2058 = invoke(stypy.reporting.localization.Localization(__file__, 268, 14), pop_2056, *[], **kwargs_2057)
    
    # Assigning a type to the variable 'dir' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'dir', pop_call_result_2058)
    
    # Assigning a Call to a Name (line 269):
    
    # Assigning a Call to a Name (line 269):
    
    # Call to listdir(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'dir' (line 269)
    dir_2061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 27), 'dir', False)
    # Processing the call keyword arguments (line 269)
    kwargs_2062 = {}
    # Getting the type of 'os' (line 269)
    os_2059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 16), 'os', False)
    # Obtaining the member 'listdir' of a type (line 269)
    listdir_2060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 16), os_2059, 'listdir')
    # Calling listdir(args, kwargs) (line 269)
    listdir_call_result_2063 = invoke(stypy.reporting.localization.Localization(__file__, 269, 16), listdir_2060, *[dir_2061], **kwargs_2062)
    
    # Assigning a type to the variable 'names' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'names', listdir_call_result_2063)
    
    # Getting the type of 'names' (line 271)
    names_2064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 20), 'names')
    # Testing the type of a for loop iterable (line 271)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 271, 8), names_2064)
    # Getting the type of the for loop variable (line 271)
    for_loop_var_2065 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 271, 8), names_2064)
    # Assigning a type to the variable 'name' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'name', for_loop_var_2065)
    # SSA begins for a for statement (line 271)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'dir' (line 272)
    dir_2066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 15), 'dir')
    # Getting the type of 'os' (line 272)
    os_2067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 22), 'os')
    # Obtaining the member 'curdir' of a type (line 272)
    curdir_2068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 22), os_2067, 'curdir')
    # Applying the binary operator '!=' (line 272)
    result_ne_2069 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 15), '!=', dir_2066, curdir_2068)
    
    # Testing the type of an if condition (line 272)
    if_condition_2070 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 12), result_ne_2069)
    # Assigning a type to the variable 'if_condition_2070' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'if_condition_2070', if_condition_2070)
    # SSA begins for if statement (line 272)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 273):
    
    # Assigning a Call to a Name (line 273):
    
    # Call to join(...): (line 273)
    # Processing the call arguments (line 273)
    # Getting the type of 'dir' (line 273)
    dir_2074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 40), 'dir', False)
    # Getting the type of 'name' (line 273)
    name_2075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 45), 'name', False)
    # Processing the call keyword arguments (line 273)
    kwargs_2076 = {}
    # Getting the type of 'os' (line 273)
    os_2071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 27), 'os', False)
    # Obtaining the member 'path' of a type (line 273)
    path_2072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 27), os_2071, 'path')
    # Obtaining the member 'join' of a type (line 273)
    join_2073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 27), path_2072, 'join')
    # Calling join(args, kwargs) (line 273)
    join_call_result_2077 = invoke(stypy.reporting.localization.Localization(__file__, 273, 27), join_2073, *[dir_2074, name_2075], **kwargs_2076)
    
    # Assigning a type to the variable 'fullname' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'fullname', join_call_result_2077)
    # SSA branch for the else part of an if statement (line 272)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 275):
    
    # Assigning a Name to a Name (line 275):
    # Getting the type of 'name' (line 275)
    name_2078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 27), 'name')
    # Assigning a type to the variable 'fullname' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'fullname', name_2078)
    # SSA join for if statement (line 272)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 278):
    
    # Assigning a Call to a Name (line 278):
    
    # Call to stat(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'fullname' (line 278)
    fullname_2081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 27), 'fullname', False)
    # Processing the call keyword arguments (line 278)
    kwargs_2082 = {}
    # Getting the type of 'os' (line 278)
    os_2079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 19), 'os', False)
    # Obtaining the member 'stat' of a type (line 278)
    stat_2080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 19), os_2079, 'stat')
    # Calling stat(args, kwargs) (line 278)
    stat_call_result_2083 = invoke(stypy.reporting.localization.Localization(__file__, 278, 19), stat_2080, *[fullname_2081], **kwargs_2082)
    
    # Assigning a type to the variable 'stat' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'stat', stat_call_result_2083)
    
    # Assigning a Subscript to a Name (line 279):
    
    # Assigning a Subscript to a Name (line 279):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ST_MODE' (line 279)
    ST_MODE_2084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 24), 'ST_MODE')
    # Getting the type of 'stat' (line 279)
    stat_2085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 19), 'stat')
    # Obtaining the member '__getitem__' of a type (line 279)
    getitem___2086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 19), stat_2085, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 279)
    subscript_call_result_2087 = invoke(stypy.reporting.localization.Localization(__file__, 279, 19), getitem___2086, ST_MODE_2084)
    
    # Assigning a type to the variable 'mode' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'mode', subscript_call_result_2087)
    
    
    # Call to S_ISREG(...): (line 280)
    # Processing the call arguments (line 280)
    # Getting the type of 'mode' (line 280)
    mode_2089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 23), 'mode', False)
    # Processing the call keyword arguments (line 280)
    kwargs_2090 = {}
    # Getting the type of 'S_ISREG' (line 280)
    S_ISREG_2088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), 'S_ISREG', False)
    # Calling S_ISREG(args, kwargs) (line 280)
    S_ISREG_call_result_2091 = invoke(stypy.reporting.localization.Localization(__file__, 280, 15), S_ISREG_2088, *[mode_2089], **kwargs_2090)
    
    # Testing the type of an if condition (line 280)
    if_condition_2092 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 12), S_ISREG_call_result_2091)
    # Assigning a type to the variable 'if_condition_2092' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'if_condition_2092', if_condition_2092)
    # SSA begins for if statement (line 280)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 281)
    # Processing the call arguments (line 281)
    # Getting the type of 'fullname' (line 281)
    fullname_2095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 28), 'fullname', False)
    # Processing the call keyword arguments (line 281)
    kwargs_2096 = {}
    # Getting the type of 'list' (line 281)
    list_2093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'list', False)
    # Obtaining the member 'append' of a type (line 281)
    append_2094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 16), list_2093, 'append')
    # Calling append(args, kwargs) (line 281)
    append_call_result_2097 = invoke(stypy.reporting.localization.Localization(__file__, 281, 16), append_2094, *[fullname_2095], **kwargs_2096)
    
    # SSA branch for the else part of an if statement (line 280)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Call to S_ISDIR(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'mode' (line 282)
    mode_2099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 25), 'mode', False)
    # Processing the call keyword arguments (line 282)
    kwargs_2100 = {}
    # Getting the type of 'S_ISDIR' (line 282)
    S_ISDIR_2098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 17), 'S_ISDIR', False)
    # Calling S_ISDIR(args, kwargs) (line 282)
    S_ISDIR_call_result_2101 = invoke(stypy.reporting.localization.Localization(__file__, 282, 17), S_ISDIR_2098, *[mode_2099], **kwargs_2100)
    
    
    
    # Call to S_ISLNK(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'mode' (line 282)
    mode_2103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 47), 'mode', False)
    # Processing the call keyword arguments (line 282)
    kwargs_2104 = {}
    # Getting the type of 'S_ISLNK' (line 282)
    S_ISLNK_2102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 39), 'S_ISLNK', False)
    # Calling S_ISLNK(args, kwargs) (line 282)
    S_ISLNK_call_result_2105 = invoke(stypy.reporting.localization.Localization(__file__, 282, 39), S_ISLNK_2102, *[mode_2103], **kwargs_2104)
    
    # Applying the 'not' unary operator (line 282)
    result_not__2106 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 35), 'not', S_ISLNK_call_result_2105)
    
    # Applying the binary operator 'and' (line 282)
    result_and_keyword_2107 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 17), 'and', S_ISDIR_call_result_2101, result_not__2106)
    
    # Testing the type of an if condition (line 282)
    if_condition_2108 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 282, 17), result_and_keyword_2107)
    # Assigning a type to the variable 'if_condition_2108' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 17), 'if_condition_2108', if_condition_2108)
    # SSA begins for if statement (line 282)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to push(...): (line 283)
    # Processing the call arguments (line 283)
    # Getting the type of 'fullname' (line 283)
    fullname_2110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 21), 'fullname', False)
    # Processing the call keyword arguments (line 283)
    kwargs_2111 = {}
    # Getting the type of 'push' (line 283)
    push_2109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'push', False)
    # Calling push(args, kwargs) (line 283)
    push_call_result_2112 = invoke(stypy.reporting.localization.Localization(__file__, 283, 16), push_2109, *[fullname_2110], **kwargs_2111)
    
    # SSA join for if statement (line 282)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 280)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 267)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'list' (line 285)
    list_2113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 11), 'list')
    # Assigning a type to the variable 'stypy_return_type' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'stypy_return_type', list_2113)
    
    # ################# End of 'findall(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'findall' in the type store
    # Getting the type of 'stypy_return_type' (line 256)
    stypy_return_type_2114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2114)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'findall'
    return stypy_return_type_2114

# Assigning a type to the variable 'findall' (line 256)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 0), 'findall', findall)

@norecursion
def glob_to_re(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'glob_to_re'
    module_type_store = module_type_store.open_function_context('glob_to_re', 288, 0, False)
    
    # Passed parameters checking function
    glob_to_re.stypy_localization = localization
    glob_to_re.stypy_type_of_self = None
    glob_to_re.stypy_type_store = module_type_store
    glob_to_re.stypy_function_name = 'glob_to_re'
    glob_to_re.stypy_param_names_list = ['pattern']
    glob_to_re.stypy_varargs_param_name = None
    glob_to_re.stypy_kwargs_param_name = None
    glob_to_re.stypy_call_defaults = defaults
    glob_to_re.stypy_call_varargs = varargs
    glob_to_re.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'glob_to_re', ['pattern'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'glob_to_re', localization, ['pattern'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'glob_to_re(...)' code ##################

    str_2115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, (-1)), 'str', 'Translate a shell-like glob pattern to a regular expression.\n\n    Return a string containing the regex.  Differs from\n    \'fnmatch.translate()\' in that \'*\' does not match "special characters"\n    (which are platform-specific).\n    ')
    
    # Assigning a Call to a Name (line 295):
    
    # Assigning a Call to a Name (line 295):
    
    # Call to translate(...): (line 295)
    # Processing the call arguments (line 295)
    # Getting the type of 'pattern' (line 295)
    pattern_2118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 35), 'pattern', False)
    # Processing the call keyword arguments (line 295)
    kwargs_2119 = {}
    # Getting the type of 'fnmatch' (line 295)
    fnmatch_2116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 17), 'fnmatch', False)
    # Obtaining the member 'translate' of a type (line 295)
    translate_2117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 17), fnmatch_2116, 'translate')
    # Calling translate(args, kwargs) (line 295)
    translate_call_result_2120 = invoke(stypy.reporting.localization.Localization(__file__, 295, 17), translate_2117, *[pattern_2118], **kwargs_2119)
    
    # Assigning a type to the variable 'pattern_re' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'pattern_re', translate_call_result_2120)
    
    # Assigning a Attribute to a Name (line 302):
    
    # Assigning a Attribute to a Name (line 302):
    # Getting the type of 'os' (line 302)
    os_2121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 10), 'os')
    # Obtaining the member 'sep' of a type (line 302)
    sep_2122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 10), os_2121, 'sep')
    # Assigning a type to the variable 'sep' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'sep', sep_2122)
    
    
    # Getting the type of 'os' (line 303)
    os_2123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 7), 'os')
    # Obtaining the member 'sep' of a type (line 303)
    sep_2124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 7), os_2123, 'sep')
    str_2125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 17), 'str', '\\')
    # Applying the binary operator '==' (line 303)
    result_eq_2126 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 7), '==', sep_2124, str_2125)
    
    # Testing the type of an if condition (line 303)
    if_condition_2127 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 303, 4), result_eq_2126)
    # Assigning a type to the variable 'if_condition_2127' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'if_condition_2127', if_condition_2127)
    # SSA begins for if statement (line 303)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 306):
    
    # Assigning a Str to a Name (line 306):
    str_2128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 14), 'str', '\\\\\\\\')
    # Assigning a type to the variable 'sep' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'sep', str_2128)
    # SSA join for if statement (line 303)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 307):
    
    # Assigning a BinOp to a Name (line 307):
    str_2129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 14), 'str', '\\1[^%s]')
    # Getting the type of 'sep' (line 307)
    sep_2130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 27), 'sep')
    # Applying the binary operator '%' (line 307)
    result_mod_2131 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 14), '%', str_2129, sep_2130)
    
    # Assigning a type to the variable 'escaped' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'escaped', result_mod_2131)
    
    # Assigning a Call to a Name (line 308):
    
    # Assigning a Call to a Name (line 308):
    
    # Call to sub(...): (line 308)
    # Processing the call arguments (line 308)
    str_2134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 24), 'str', '((?<!\\\\)(\\\\\\\\)*)\\.')
    # Getting the type of 'escaped' (line 308)
    escaped_2135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 47), 'escaped', False)
    # Getting the type of 'pattern_re' (line 308)
    pattern_re_2136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 56), 'pattern_re', False)
    # Processing the call keyword arguments (line 308)
    kwargs_2137 = {}
    # Getting the type of 're' (line 308)
    re_2132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 17), 're', False)
    # Obtaining the member 'sub' of a type (line 308)
    sub_2133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 17), re_2132, 'sub')
    # Calling sub(args, kwargs) (line 308)
    sub_call_result_2138 = invoke(stypy.reporting.localization.Localization(__file__, 308, 17), sub_2133, *[str_2134, escaped_2135, pattern_re_2136], **kwargs_2137)
    
    # Assigning a type to the variable 'pattern_re' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'pattern_re', sub_call_result_2138)
    # Getting the type of 'pattern_re' (line 309)
    pattern_re_2139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 11), 'pattern_re')
    # Assigning a type to the variable 'stypy_return_type' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'stypy_return_type', pattern_re_2139)
    
    # ################# End of 'glob_to_re(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'glob_to_re' in the type store
    # Getting the type of 'stypy_return_type' (line 288)
    stypy_return_type_2140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2140)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'glob_to_re'
    return stypy_return_type_2140

# Assigning a type to the variable 'glob_to_re' (line 288)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 0), 'glob_to_re', glob_to_re)

@norecursion
def translate_pattern(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_2141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 38), 'int')
    # Getting the type of 'None' (line 312)
    None_2142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 48), 'None')
    int_2143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 63), 'int')
    defaults = [int_2141, None_2142, int_2143]
    # Create a new context for function 'translate_pattern'
    module_type_store = module_type_store.open_function_context('translate_pattern', 312, 0, False)
    
    # Passed parameters checking function
    translate_pattern.stypy_localization = localization
    translate_pattern.stypy_type_of_self = None
    translate_pattern.stypy_type_store = module_type_store
    translate_pattern.stypy_function_name = 'translate_pattern'
    translate_pattern.stypy_param_names_list = ['pattern', 'anchor', 'prefix', 'is_regex']
    translate_pattern.stypy_varargs_param_name = None
    translate_pattern.stypy_kwargs_param_name = None
    translate_pattern.stypy_call_defaults = defaults
    translate_pattern.stypy_call_varargs = varargs
    translate_pattern.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'translate_pattern', ['pattern', 'anchor', 'prefix', 'is_regex'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'translate_pattern', localization, ['pattern', 'anchor', 'prefix', 'is_regex'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'translate_pattern(...)' code ##################

    str_2144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, (-1)), 'str', "Translate a shell-like wildcard pattern to a compiled regular\n    expression.\n\n    Return the compiled regex.  If 'is_regex' true,\n    then 'pattern' is directly compiled to a regex (if it's a string)\n    or just returned as-is (assumes it's a regex object).\n    ")
    
    # Getting the type of 'is_regex' (line 320)
    is_regex_2145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 7), 'is_regex')
    # Testing the type of an if condition (line 320)
    if_condition_2146 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 320, 4), is_regex_2145)
    # Assigning a type to the variable 'if_condition_2146' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'if_condition_2146', if_condition_2146)
    # SSA begins for if statement (line 320)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 321)
    # Getting the type of 'str' (line 321)
    str_2147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 31), 'str')
    # Getting the type of 'pattern' (line 321)
    pattern_2148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 22), 'pattern')
    
    (may_be_2149, more_types_in_union_2150) = may_be_subtype(str_2147, pattern_2148)

    if may_be_2149:

        if more_types_in_union_2150:
            # Runtime conditional SSA (line 321)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'pattern' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'pattern', remove_not_subtype_from_union(pattern_2148, str))
        
        # Call to compile(...): (line 322)
        # Processing the call arguments (line 322)
        # Getting the type of 'pattern' (line 322)
        pattern_2153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 30), 'pattern', False)
        # Processing the call keyword arguments (line 322)
        kwargs_2154 = {}
        # Getting the type of 're' (line 322)
        re_2151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 19), 're', False)
        # Obtaining the member 'compile' of a type (line 322)
        compile_2152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 19), re_2151, 'compile')
        # Calling compile(args, kwargs) (line 322)
        compile_call_result_2155 = invoke(stypy.reporting.localization.Localization(__file__, 322, 19), compile_2152, *[pattern_2153], **kwargs_2154)
        
        # Assigning a type to the variable 'stypy_return_type' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'stypy_return_type', compile_call_result_2155)

        if more_types_in_union_2150:
            # Runtime conditional SSA for else branch (line 321)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_2149) or more_types_in_union_2150):
        # Assigning a type to the variable 'pattern' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'pattern', remove_subtype_from_union(pattern_2148, str))
        # Getting the type of 'pattern' (line 324)
        pattern_2156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 19), 'pattern')
        # Assigning a type to the variable 'stypy_return_type' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'stypy_return_type', pattern_2156)

        if (may_be_2149 and more_types_in_union_2150):
            # SSA join for if statement (line 321)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 320)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'pattern' (line 326)
    pattern_2157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 7), 'pattern')
    # Testing the type of an if condition (line 326)
    if_condition_2158 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 326, 4), pattern_2157)
    # Assigning a type to the variable 'if_condition_2158' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'if_condition_2158', if_condition_2158)
    # SSA begins for if statement (line 326)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 327):
    
    # Assigning a Call to a Name (line 327):
    
    # Call to glob_to_re(...): (line 327)
    # Processing the call arguments (line 327)
    # Getting the type of 'pattern' (line 327)
    pattern_2160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 32), 'pattern', False)
    # Processing the call keyword arguments (line 327)
    kwargs_2161 = {}
    # Getting the type of 'glob_to_re' (line 327)
    glob_to_re_2159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 21), 'glob_to_re', False)
    # Calling glob_to_re(args, kwargs) (line 327)
    glob_to_re_call_result_2162 = invoke(stypy.reporting.localization.Localization(__file__, 327, 21), glob_to_re_2159, *[pattern_2160], **kwargs_2161)
    
    # Assigning a type to the variable 'pattern_re' (line 327)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'pattern_re', glob_to_re_call_result_2162)
    # SSA branch for the else part of an if statement (line 326)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 329):
    
    # Assigning a Str to a Name (line 329):
    str_2163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 21), 'str', '')
    # Assigning a type to the variable 'pattern_re' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'pattern_re', str_2163)
    # SSA join for if statement (line 326)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 331)
    # Getting the type of 'prefix' (line 331)
    prefix_2164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'prefix')
    # Getting the type of 'None' (line 331)
    None_2165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 21), 'None')
    
    (may_be_2166, more_types_in_union_2167) = may_not_be_none(prefix_2164, None_2165)

    if may_be_2166:

        if more_types_in_union_2167:
            # Runtime conditional SSA (line 331)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 333):
        
        # Assigning a Call to a Name (line 333):
        
        # Call to glob_to_re(...): (line 333)
        # Processing the call arguments (line 333)
        str_2169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 35), 'str', '')
        # Processing the call keyword arguments (line 333)
        kwargs_2170 = {}
        # Getting the type of 'glob_to_re' (line 333)
        glob_to_re_2168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 24), 'glob_to_re', False)
        # Calling glob_to_re(args, kwargs) (line 333)
        glob_to_re_call_result_2171 = invoke(stypy.reporting.localization.Localization(__file__, 333, 24), glob_to_re_2168, *[str_2169], **kwargs_2170)
        
        # Assigning a type to the variable 'empty_pattern' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'empty_pattern', glob_to_re_call_result_2171)
        
        # Assigning a Subscript to a Name (line 334):
        
        # Assigning a Subscript to a Name (line 334):
        
        # Obtaining the type of the subscript
        
        
        # Call to len(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'empty_pattern' (line 334)
        empty_pattern_2173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 45), 'empty_pattern', False)
        # Processing the call keyword arguments (line 334)
        kwargs_2174 = {}
        # Getting the type of 'len' (line 334)
        len_2172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 41), 'len', False)
        # Calling len(args, kwargs) (line 334)
        len_call_result_2175 = invoke(stypy.reporting.localization.Localization(__file__, 334, 41), len_2172, *[empty_pattern_2173], **kwargs_2174)
        
        # Applying the 'usub' unary operator (line 334)
        result___neg___2176 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 40), 'usub', len_call_result_2175)
        
        slice_2177 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 334, 20), None, result___neg___2176, None)
        
        # Call to glob_to_re(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'prefix' (line 334)
        prefix_2179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 31), 'prefix', False)
        # Processing the call keyword arguments (line 334)
        kwargs_2180 = {}
        # Getting the type of 'glob_to_re' (line 334)
        glob_to_re_2178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 20), 'glob_to_re', False)
        # Calling glob_to_re(args, kwargs) (line 334)
        glob_to_re_call_result_2181 = invoke(stypy.reporting.localization.Localization(__file__, 334, 20), glob_to_re_2178, *[prefix_2179], **kwargs_2180)
        
        # Obtaining the member '__getitem__' of a type (line 334)
        getitem___2182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 20), glob_to_re_call_result_2181, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 334)
        subscript_call_result_2183 = invoke(stypy.reporting.localization.Localization(__file__, 334, 20), getitem___2182, slice_2177)
        
        # Assigning a type to the variable 'prefix_re' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'prefix_re', subscript_call_result_2183)
        
        # Assigning a Attribute to a Name (line 335):
        
        # Assigning a Attribute to a Name (line 335):
        # Getting the type of 'os' (line 335)
        os_2184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 14), 'os')
        # Obtaining the member 'sep' of a type (line 335)
        sep_2185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 14), os_2184, 'sep')
        # Assigning a type to the variable 'sep' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'sep', sep_2185)
        
        
        # Getting the type of 'os' (line 336)
        os_2186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 11), 'os')
        # Obtaining the member 'sep' of a type (line 336)
        sep_2187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 11), os_2186, 'sep')
        str_2188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 21), 'str', '\\')
        # Applying the binary operator '==' (line 336)
        result_eq_2189 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 11), '==', sep_2187, str_2188)
        
        # Testing the type of an if condition (line 336)
        if_condition_2190 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 336, 8), result_eq_2189)
        # Assigning a type to the variable 'if_condition_2190' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'if_condition_2190', if_condition_2190)
        # SSA begins for if statement (line 336)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 337):
        
        # Assigning a Str to a Name (line 337):
        str_2191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 18), 'str', '\\\\')
        # Assigning a type to the variable 'sep' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'sep', str_2191)
        # SSA join for if statement (line 336)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 338):
        
        # Assigning a BinOp to a Name (line 338):
        str_2192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 21), 'str', '^')
        
        # Call to join(...): (line 338)
        # Processing the call arguments (line 338)
        
        # Obtaining an instance of the builtin type 'tuple' (line 338)
        tuple_2195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 338)
        # Adding element type (line 338)
        # Getting the type of 'prefix_re' (line 338)
        prefix_re_2196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 37), 'prefix_re', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 37), tuple_2195, prefix_re_2196)
        # Adding element type (line 338)
        str_2197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 48), 'str', '.*')
        # Getting the type of 'pattern_re' (line 338)
        pattern_re_2198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 55), 'pattern_re', False)
        # Applying the binary operator '+' (line 338)
        result_add_2199 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 48), '+', str_2197, pattern_re_2198)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 37), tuple_2195, result_add_2199)
        
        # Processing the call keyword arguments (line 338)
        kwargs_2200 = {}
        # Getting the type of 'sep' (line 338)
        sep_2193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 27), 'sep', False)
        # Obtaining the member 'join' of a type (line 338)
        join_2194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 27), sep_2193, 'join')
        # Calling join(args, kwargs) (line 338)
        join_call_result_2201 = invoke(stypy.reporting.localization.Localization(__file__, 338, 27), join_2194, *[tuple_2195], **kwargs_2200)
        
        # Applying the binary operator '+' (line 338)
        result_add_2202 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 21), '+', str_2192, join_call_result_2201)
        
        # Assigning a type to the variable 'pattern_re' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'pattern_re', result_add_2202)

        if more_types_in_union_2167:
            # Runtime conditional SSA for else branch (line 331)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_2166) or more_types_in_union_2167):
        
        # Getting the type of 'anchor' (line 340)
        anchor_2203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 11), 'anchor')
        # Testing the type of an if condition (line 340)
        if_condition_2204 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 8), anchor_2203)
        # Assigning a type to the variable 'if_condition_2204' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'if_condition_2204', if_condition_2204)
        # SSA begins for if statement (line 340)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 341):
        
        # Assigning a BinOp to a Name (line 341):
        str_2205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 25), 'str', '^')
        # Getting the type of 'pattern_re' (line 341)
        pattern_re_2206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 31), 'pattern_re')
        # Applying the binary operator '+' (line 341)
        result_add_2207 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 25), '+', str_2205, pattern_re_2206)
        
        # Assigning a type to the variable 'pattern_re' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'pattern_re', result_add_2207)
        # SSA join for if statement (line 340)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_2166 and more_types_in_union_2167):
            # SSA join for if statement (line 331)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to compile(...): (line 343)
    # Processing the call arguments (line 343)
    # Getting the type of 'pattern_re' (line 343)
    pattern_re_2210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 22), 'pattern_re', False)
    # Processing the call keyword arguments (line 343)
    kwargs_2211 = {}
    # Getting the type of 're' (line 343)
    re_2208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 11), 're', False)
    # Obtaining the member 'compile' of a type (line 343)
    compile_2209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 11), re_2208, 'compile')
    # Calling compile(args, kwargs) (line 343)
    compile_call_result_2212 = invoke(stypy.reporting.localization.Localization(__file__, 343, 11), compile_2209, *[pattern_re_2210], **kwargs_2211)
    
    # Assigning a type to the variable 'stypy_return_type' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'stypy_return_type', compile_call_result_2212)
    
    # ################# End of 'translate_pattern(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'translate_pattern' in the type store
    # Getting the type of 'stypy_return_type' (line 312)
    stypy_return_type_2213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2213)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'translate_pattern'
    return stypy_return_type_2213

# Assigning a type to the variable 'translate_pattern' (line 312)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 0), 'translate_pattern', translate_pattern)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
