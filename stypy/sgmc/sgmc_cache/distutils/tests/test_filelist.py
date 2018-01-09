
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.filelist.'''
2: import os
3: import re
4: import unittest
5: from distutils import debug
6: from distutils.log import WARN
7: from distutils.errors import DistutilsTemplateError
8: from distutils.filelist import glob_to_re, translate_pattern, FileList
9: 
10: from test.test_support import captured_stdout, run_unittest
11: from distutils.tests import support
12: 
13: MANIFEST_IN = '''\
14: include ok
15: include xo
16: exclude xo
17: include foo.tmp
18: include buildout.cfg
19: global-include *.x
20: global-include *.txt
21: global-exclude *.tmp
22: recursive-include f *.oo
23: recursive-exclude global *.x
24: graft dir
25: prune dir3
26: '''
27: 
28: 
29: def make_local_path(s):
30:     '''Converts '/' in a string to os.sep'''
31:     return s.replace('/', os.sep)
32: 
33: 
34: class FileListTestCase(support.LoggingSilencer,
35:                        unittest.TestCase):
36: 
37:     def assertNoWarnings(self):
38:         self.assertEqual(self.get_logs(WARN), [])
39:         self.clear_logs()
40: 
41:     def assertWarnings(self):
42:         self.assertGreater(len(self.get_logs(WARN)), 0)
43:         self.clear_logs()
44: 
45:     def test_glob_to_re(self):
46:         sep = os.sep
47:         if os.sep == '\\':
48:             sep = re.escape(os.sep)
49: 
50:         for glob, regex in (
51:             # simple cases
52:             ('foo*', r'foo[^%(sep)s]*\Z(?ms)'),
53:             ('foo?', r'foo[^%(sep)s]\Z(?ms)'),
54:             ('foo??', r'foo[^%(sep)s][^%(sep)s]\Z(?ms)'),
55:             # special cases
56:             (r'foo\\*', r'foo\\\\[^%(sep)s]*\Z(?ms)'),
57:             (r'foo\\\*', r'foo\\\\\\[^%(sep)s]*\Z(?ms)'),
58:             ('foo????', r'foo[^%(sep)s][^%(sep)s][^%(sep)s][^%(sep)s]\Z(?ms)'),
59:             (r'foo\\??', r'foo\\\\[^%(sep)s][^%(sep)s]\Z(?ms)')):
60:             regex = regex % {'sep': sep}
61:             self.assertEqual(glob_to_re(glob), regex)
62: 
63:     def test_process_template_line(self):
64:         # testing  all MANIFEST.in template patterns
65:         file_list = FileList()
66:         l = make_local_path
67: 
68:         # simulated file list
69:         file_list.allfiles = ['foo.tmp', 'ok', 'xo', 'four.txt',
70:                               'buildout.cfg',
71:                               # filelist does not filter out VCS directories,
72:                               # it's sdist that does
73:                               l('.hg/last-message.txt'),
74:                               l('global/one.txt'),
75:                               l('global/two.txt'),
76:                               l('global/files.x'),
77:                               l('global/here.tmp'),
78:                               l('f/o/f.oo'),
79:                               l('dir/graft-one'),
80:                               l('dir/dir2/graft2'),
81:                               l('dir3/ok'),
82:                               l('dir3/sub/ok.txt'),
83:                              ]
84: 
85:         for line in MANIFEST_IN.split('\n'):
86:             if line.strip() == '':
87:                 continue
88:             file_list.process_template_line(line)
89: 
90:         wanted = ['ok',
91:                   'buildout.cfg',
92:                   'four.txt',
93:                   l('.hg/last-message.txt'),
94:                   l('global/one.txt'),
95:                   l('global/two.txt'),
96:                   l('f/o/f.oo'),
97:                   l('dir/graft-one'),
98:                   l('dir/dir2/graft2'),
99:                  ]
100: 
101:         self.assertEqual(file_list.files, wanted)
102: 
103:     def test_debug_print(self):
104:         file_list = FileList()
105:         with captured_stdout() as stdout:
106:             file_list.debug_print('xxx')
107:         self.assertEqual(stdout.getvalue(), '')
108: 
109:         debug.DEBUG = True
110:         try:
111:             with captured_stdout() as stdout:
112:                 file_list.debug_print('xxx')
113:             self.assertEqual(stdout.getvalue(), 'xxx\n')
114:         finally:
115:             debug.DEBUG = False
116: 
117:     def test_set_allfiles(self):
118:         file_list = FileList()
119:         files = ['a', 'b', 'c']
120:         file_list.set_allfiles(files)
121:         self.assertEqual(file_list.allfiles, files)
122: 
123:     def test_remove_duplicates(self):
124:         file_list = FileList()
125:         file_list.files = ['a', 'b', 'a', 'g', 'c', 'g']
126:         # files must be sorted beforehand (sdist does it)
127:         file_list.sort()
128:         file_list.remove_duplicates()
129:         self.assertEqual(file_list.files, ['a', 'b', 'c', 'g'])
130: 
131:     def test_translate_pattern(self):
132:         # not regex
133:         self.assertTrue(hasattr(
134:             translate_pattern('a', anchor=True, is_regex=False),
135:             'search'))
136: 
137:         # is a regex
138:         regex = re.compile('a')
139:         self.assertEqual(
140:             translate_pattern(regex, anchor=True, is_regex=True),
141:             regex)
142: 
143:         # plain string flagged as regex
144:         self.assertTrue(hasattr(
145:             translate_pattern('a', anchor=True, is_regex=True),
146:             'search'))
147: 
148:         # glob support
149:         self.assertTrue(translate_pattern(
150:             '*.py', anchor=True, is_regex=False).search('filelist.py'))
151: 
152:     def test_exclude_pattern(self):
153:         # return False if no match
154:         file_list = FileList()
155:         self.assertFalse(file_list.exclude_pattern('*.py'))
156: 
157:         # return True if files match
158:         file_list = FileList()
159:         file_list.files = ['a.py', 'b.py']
160:         self.assertTrue(file_list.exclude_pattern('*.py'))
161: 
162:         # test excludes
163:         file_list = FileList()
164:         file_list.files = ['a.py', 'a.txt']
165:         file_list.exclude_pattern('*.py')
166:         self.assertEqual(file_list.files, ['a.txt'])
167: 
168:     def test_include_pattern(self):
169:         # return False if no match
170:         file_list = FileList()
171:         file_list.set_allfiles([])
172:         self.assertFalse(file_list.include_pattern('*.py'))
173: 
174:         # return True if files match
175:         file_list = FileList()
176:         file_list.set_allfiles(['a.py', 'b.txt'])
177:         self.assertTrue(file_list.include_pattern('*.py'))
178: 
179:         # test * matches all files
180:         file_list = FileList()
181:         self.assertIsNone(file_list.allfiles)
182:         file_list.set_allfiles(['a.py', 'b.txt'])
183:         file_list.include_pattern('*')
184:         self.assertEqual(file_list.allfiles, ['a.py', 'b.txt'])
185: 
186:     def test_process_template(self):
187:         l = make_local_path
188:         # invalid lines
189:         file_list = FileList()
190:         for action in ('include', 'exclude', 'global-include',
191:                        'global-exclude', 'recursive-include',
192:                        'recursive-exclude', 'graft', 'prune', 'blarg'):
193:             self.assertRaises(DistutilsTemplateError,
194:                               file_list.process_template_line, action)
195: 
196:         # include
197:         file_list = FileList()
198:         file_list.set_allfiles(['a.py', 'b.txt', l('d/c.py')])
199: 
200:         file_list.process_template_line('include *.py')
201:         self.assertEqual(file_list.files, ['a.py'])
202:         self.assertNoWarnings()
203: 
204:         file_list.process_template_line('include *.rb')
205:         self.assertEqual(file_list.files, ['a.py'])
206:         self.assertWarnings()
207: 
208:         # exclude
209:         file_list = FileList()
210:         file_list.files = ['a.py', 'b.txt', l('d/c.py')]
211: 
212:         file_list.process_template_line('exclude *.py')
213:         self.assertEqual(file_list.files, ['b.txt', l('d/c.py')])
214:         self.assertNoWarnings()
215: 
216:         file_list.process_template_line('exclude *.rb')
217:         self.assertEqual(file_list.files, ['b.txt', l('d/c.py')])
218:         self.assertWarnings()
219: 
220:         # global-include
221:         file_list = FileList()
222:         file_list.set_allfiles(['a.py', 'b.txt', l('d/c.py')])
223: 
224:         file_list.process_template_line('global-include *.py')
225:         self.assertEqual(file_list.files, ['a.py', l('d/c.py')])
226:         self.assertNoWarnings()
227: 
228:         file_list.process_template_line('global-include *.rb')
229:         self.assertEqual(file_list.files, ['a.py', l('d/c.py')])
230:         self.assertWarnings()
231: 
232:         # global-exclude
233:         file_list = FileList()
234:         file_list.files = ['a.py', 'b.txt', l('d/c.py')]
235: 
236:         file_list.process_template_line('global-exclude *.py')
237:         self.assertEqual(file_list.files, ['b.txt'])
238:         self.assertNoWarnings()
239: 
240:         file_list.process_template_line('global-exclude *.rb')
241:         self.assertEqual(file_list.files, ['b.txt'])
242:         self.assertWarnings()
243: 
244:         # recursive-include
245:         file_list = FileList()
246:         file_list.set_allfiles(['a.py', l('d/b.py'), l('d/c.txt'),
247:                                 l('d/d/e.py')])
248: 
249:         file_list.process_template_line('recursive-include d *.py')
250:         self.assertEqual(file_list.files, [l('d/b.py'), l('d/d/e.py')])
251:         self.assertNoWarnings()
252: 
253:         file_list.process_template_line('recursive-include e *.py')
254:         self.assertEqual(file_list.files, [l('d/b.py'), l('d/d/e.py')])
255:         self.assertWarnings()
256: 
257:         # recursive-exclude
258:         file_list = FileList()
259:         file_list.files = ['a.py', l('d/b.py'), l('d/c.txt'), l('d/d/e.py')]
260: 
261:         file_list.process_template_line('recursive-exclude d *.py')
262:         self.assertEqual(file_list.files, ['a.py', l('d/c.txt')])
263:         self.assertNoWarnings()
264: 
265:         file_list.process_template_line('recursive-exclude e *.py')
266:         self.assertEqual(file_list.files, ['a.py', l('d/c.txt')])
267:         self.assertWarnings()
268: 
269:         # graft
270:         file_list = FileList()
271:         file_list.set_allfiles(['a.py', l('d/b.py'), l('d/d/e.py'),
272:                                 l('f/f.py')])
273: 
274:         file_list.process_template_line('graft d')
275:         self.assertEqual(file_list.files, [l('d/b.py'), l('d/d/e.py')])
276:         self.assertNoWarnings()
277: 
278:         file_list.process_template_line('graft e')
279:         self.assertEqual(file_list.files, [l('d/b.py'), l('d/d/e.py')])
280:         self.assertWarnings()
281: 
282:         # prune
283:         file_list = FileList()
284:         file_list.files = ['a.py', l('d/b.py'), l('d/d/e.py'), l('f/f.py')]
285: 
286:         file_list.process_template_line('prune d')
287:         self.assertEqual(file_list.files, ['a.py', l('f/f.py')])
288:         self.assertNoWarnings()
289: 
290:         file_list.process_template_line('prune e')
291:         self.assertEqual(file_list.files, ['a.py', l('f/f.py')])
292:         self.assertWarnings()
293: 
294: 
295: def test_suite():
296:     return unittest.makeSuite(FileListTestCase)
297: 
298: if __name__ == "__main__":
299:     run_unittest(test_suite())
300: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_38162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.filelist.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import os' statement (line 2)
import os

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import re' statement (line 3)
import re

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import unittest' statement (line 4)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from distutils import debug' statement (line 5)
try:
    from distutils import debug

except:
    debug = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils', None, module_type_store, ['debug'], [debug])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from distutils.log import WARN' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_38163 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.log')

if (type(import_38163) is not StypyTypeError):

    if (import_38163 != 'pyd_module'):
        __import__(import_38163)
        sys_modules_38164 = sys.modules[import_38163]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.log', sys_modules_38164.module_type_store, module_type_store, ['WARN'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_38164, sys_modules_38164.module_type_store, module_type_store)
    else:
        from distutils.log import WARN

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.log', None, module_type_store, ['WARN'], [WARN])

else:
    # Assigning a type to the variable 'distutils.log' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.log', import_38163)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.errors import DistutilsTemplateError' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_38165 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.errors')

if (type(import_38165) is not StypyTypeError):

    if (import_38165 != 'pyd_module'):
        __import__(import_38165)
        sys_modules_38166 = sys.modules[import_38165]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.errors', sys_modules_38166.module_type_store, module_type_store, ['DistutilsTemplateError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_38166, sys_modules_38166.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsTemplateError

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.errors', None, module_type_store, ['DistutilsTemplateError'], [DistutilsTemplateError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.errors', import_38165)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.filelist import glob_to_re, translate_pattern, FileList' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_38167 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.filelist')

if (type(import_38167) is not StypyTypeError):

    if (import_38167 != 'pyd_module'):
        __import__(import_38167)
        sys_modules_38168 = sys.modules[import_38167]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.filelist', sys_modules_38168.module_type_store, module_type_store, ['glob_to_re', 'translate_pattern', 'FileList'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_38168, sys_modules_38168.module_type_store, module_type_store)
    else:
        from distutils.filelist import glob_to_re, translate_pattern, FileList

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.filelist', None, module_type_store, ['glob_to_re', 'translate_pattern', 'FileList'], [glob_to_re, translate_pattern, FileList])

else:
    # Assigning a type to the variable 'distutils.filelist' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.filelist', import_38167)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from test.test_support import captured_stdout, run_unittest' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_38169 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'test.test_support')

if (type(import_38169) is not StypyTypeError):

    if (import_38169 != 'pyd_module'):
        __import__(import_38169)
        sys_modules_38170 = sys.modules[import_38169]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'test.test_support', sys_modules_38170.module_type_store, module_type_store, ['captured_stdout', 'run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_38170, sys_modules_38170.module_type_store, module_type_store)
    else:
        from test.test_support import captured_stdout, run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'test.test_support', None, module_type_store, ['captured_stdout', 'run_unittest'], [captured_stdout, run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'test.test_support', import_38169)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.tests import support' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_38171 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.tests')

if (type(import_38171) is not StypyTypeError):

    if (import_38171 != 'pyd_module'):
        __import__(import_38171)
        sys_modules_38172 = sys.modules[import_38171]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.tests', sys_modules_38172.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_38172, sys_modules_38172.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.tests', import_38171)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')


# Assigning a Str to a Name (line 13):
str_38173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, (-1)), 'str', 'include ok\ninclude xo\nexclude xo\ninclude foo.tmp\ninclude buildout.cfg\nglobal-include *.x\nglobal-include *.txt\nglobal-exclude *.tmp\nrecursive-include f *.oo\nrecursive-exclude global *.x\ngraft dir\nprune dir3\n')
# Assigning a type to the variable 'MANIFEST_IN' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'MANIFEST_IN', str_38173)

@norecursion
def make_local_path(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'make_local_path'
    module_type_store = module_type_store.open_function_context('make_local_path', 29, 0, False)
    
    # Passed parameters checking function
    make_local_path.stypy_localization = localization
    make_local_path.stypy_type_of_self = None
    make_local_path.stypy_type_store = module_type_store
    make_local_path.stypy_function_name = 'make_local_path'
    make_local_path.stypy_param_names_list = ['s']
    make_local_path.stypy_varargs_param_name = None
    make_local_path.stypy_kwargs_param_name = None
    make_local_path.stypy_call_defaults = defaults
    make_local_path.stypy_call_varargs = varargs
    make_local_path.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_local_path', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_local_path', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_local_path(...)' code ##################

    str_38174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 4), 'str', "Converts '/' in a string to os.sep")
    
    # Call to replace(...): (line 31)
    # Processing the call arguments (line 31)
    str_38177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 21), 'str', '/')
    # Getting the type of 'os' (line 31)
    os_38178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 26), 'os', False)
    # Obtaining the member 'sep' of a type (line 31)
    sep_38179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 26), os_38178, 'sep')
    # Processing the call keyword arguments (line 31)
    kwargs_38180 = {}
    # Getting the type of 's' (line 31)
    s_38175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 's', False)
    # Obtaining the member 'replace' of a type (line 31)
    replace_38176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 11), s_38175, 'replace')
    # Calling replace(args, kwargs) (line 31)
    replace_call_result_38181 = invoke(stypy.reporting.localization.Localization(__file__, 31, 11), replace_38176, *[str_38177, sep_38179], **kwargs_38180)
    
    # Assigning a type to the variable 'stypy_return_type' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type', replace_call_result_38181)
    
    # ################# End of 'make_local_path(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_local_path' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_38182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38182)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_local_path'
    return stypy_return_type_38182

# Assigning a type to the variable 'make_local_path' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'make_local_path', make_local_path)
# Declaration of the 'FileListTestCase' class
# Getting the type of 'support' (line 34)
support_38183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 23), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 34)
LoggingSilencer_38184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 23), support_38183, 'LoggingSilencer')
# Getting the type of 'unittest' (line 35)
unittest_38185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 23), 'unittest')
# Obtaining the member 'TestCase' of a type (line 35)
TestCase_38186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 23), unittest_38185, 'TestCase')

class FileListTestCase(LoggingSilencer_38184, TestCase_38186, ):

    @norecursion
    def assertNoWarnings(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'assertNoWarnings'
        module_type_store = module_type_store.open_function_context('assertNoWarnings', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileListTestCase.assertNoWarnings.__dict__.__setitem__('stypy_localization', localization)
        FileListTestCase.assertNoWarnings.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileListTestCase.assertNoWarnings.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileListTestCase.assertNoWarnings.__dict__.__setitem__('stypy_function_name', 'FileListTestCase.assertNoWarnings')
        FileListTestCase.assertNoWarnings.__dict__.__setitem__('stypy_param_names_list', [])
        FileListTestCase.assertNoWarnings.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileListTestCase.assertNoWarnings.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileListTestCase.assertNoWarnings.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileListTestCase.assertNoWarnings.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileListTestCase.assertNoWarnings.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileListTestCase.assertNoWarnings.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileListTestCase.assertNoWarnings', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertNoWarnings', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertNoWarnings(...)' code ##################

        
        # Call to assertEqual(...): (line 38)
        # Processing the call arguments (line 38)
        
        # Call to get_logs(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'WARN' (line 38)
        WARN_38191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 39), 'WARN', False)
        # Processing the call keyword arguments (line 38)
        kwargs_38192 = {}
        # Getting the type of 'self' (line 38)
        self_38189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'self', False)
        # Obtaining the member 'get_logs' of a type (line 38)
        get_logs_38190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 25), self_38189, 'get_logs')
        # Calling get_logs(args, kwargs) (line 38)
        get_logs_call_result_38193 = invoke(stypy.reporting.localization.Localization(__file__, 38, 25), get_logs_38190, *[WARN_38191], **kwargs_38192)
        
        
        # Obtaining an instance of the builtin type 'list' (line 38)
        list_38194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 38)
        
        # Processing the call keyword arguments (line 38)
        kwargs_38195 = {}
        # Getting the type of 'self' (line 38)
        self_38187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 38)
        assertEqual_38188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_38187, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 38)
        assertEqual_call_result_38196 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), assertEqual_38188, *[get_logs_call_result_38193, list_38194], **kwargs_38195)
        
        
        # Call to clear_logs(...): (line 39)
        # Processing the call keyword arguments (line 39)
        kwargs_38199 = {}
        # Getting the type of 'self' (line 39)
        self_38197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'self', False)
        # Obtaining the member 'clear_logs' of a type (line 39)
        clear_logs_38198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), self_38197, 'clear_logs')
        # Calling clear_logs(args, kwargs) (line 39)
        clear_logs_call_result_38200 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), clear_logs_38198, *[], **kwargs_38199)
        
        
        # ################# End of 'assertNoWarnings(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertNoWarnings' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_38201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_38201)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertNoWarnings'
        return stypy_return_type_38201


    @norecursion
    def assertWarnings(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'assertWarnings'
        module_type_store = module_type_store.open_function_context('assertWarnings', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileListTestCase.assertWarnings.__dict__.__setitem__('stypy_localization', localization)
        FileListTestCase.assertWarnings.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileListTestCase.assertWarnings.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileListTestCase.assertWarnings.__dict__.__setitem__('stypy_function_name', 'FileListTestCase.assertWarnings')
        FileListTestCase.assertWarnings.__dict__.__setitem__('stypy_param_names_list', [])
        FileListTestCase.assertWarnings.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileListTestCase.assertWarnings.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileListTestCase.assertWarnings.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileListTestCase.assertWarnings.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileListTestCase.assertWarnings.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileListTestCase.assertWarnings.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileListTestCase.assertWarnings', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertWarnings', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertWarnings(...)' code ##################

        
        # Call to assertGreater(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Call to len(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Call to get_logs(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'WARN' (line 42)
        WARN_38207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 45), 'WARN', False)
        # Processing the call keyword arguments (line 42)
        kwargs_38208 = {}
        # Getting the type of 'self' (line 42)
        self_38205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 31), 'self', False)
        # Obtaining the member 'get_logs' of a type (line 42)
        get_logs_38206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 31), self_38205, 'get_logs')
        # Calling get_logs(args, kwargs) (line 42)
        get_logs_call_result_38209 = invoke(stypy.reporting.localization.Localization(__file__, 42, 31), get_logs_38206, *[WARN_38207], **kwargs_38208)
        
        # Processing the call keyword arguments (line 42)
        kwargs_38210 = {}
        # Getting the type of 'len' (line 42)
        len_38204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 27), 'len', False)
        # Calling len(args, kwargs) (line 42)
        len_call_result_38211 = invoke(stypy.reporting.localization.Localization(__file__, 42, 27), len_38204, *[get_logs_call_result_38209], **kwargs_38210)
        
        int_38212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 53), 'int')
        # Processing the call keyword arguments (line 42)
        kwargs_38213 = {}
        # Getting the type of 'self' (line 42)
        self_38202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self', False)
        # Obtaining the member 'assertGreater' of a type (line 42)
        assertGreater_38203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_38202, 'assertGreater')
        # Calling assertGreater(args, kwargs) (line 42)
        assertGreater_call_result_38214 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), assertGreater_38203, *[len_call_result_38211, int_38212], **kwargs_38213)
        
        
        # Call to clear_logs(...): (line 43)
        # Processing the call keyword arguments (line 43)
        kwargs_38217 = {}
        # Getting the type of 'self' (line 43)
        self_38215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self', False)
        # Obtaining the member 'clear_logs' of a type (line 43)
        clear_logs_38216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_38215, 'clear_logs')
        # Calling clear_logs(args, kwargs) (line 43)
        clear_logs_call_result_38218 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), clear_logs_38216, *[], **kwargs_38217)
        
        
        # ################# End of 'assertWarnings(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertWarnings' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_38219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_38219)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertWarnings'
        return stypy_return_type_38219


    @norecursion
    def test_glob_to_re(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_glob_to_re'
        module_type_store = module_type_store.open_function_context('test_glob_to_re', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileListTestCase.test_glob_to_re.__dict__.__setitem__('stypy_localization', localization)
        FileListTestCase.test_glob_to_re.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileListTestCase.test_glob_to_re.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileListTestCase.test_glob_to_re.__dict__.__setitem__('stypy_function_name', 'FileListTestCase.test_glob_to_re')
        FileListTestCase.test_glob_to_re.__dict__.__setitem__('stypy_param_names_list', [])
        FileListTestCase.test_glob_to_re.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileListTestCase.test_glob_to_re.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileListTestCase.test_glob_to_re.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileListTestCase.test_glob_to_re.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileListTestCase.test_glob_to_re.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileListTestCase.test_glob_to_re.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileListTestCase.test_glob_to_re', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_glob_to_re', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_glob_to_re(...)' code ##################

        
        # Assigning a Attribute to a Name (line 46):
        # Getting the type of 'os' (line 46)
        os_38220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 14), 'os')
        # Obtaining the member 'sep' of a type (line 46)
        sep_38221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 14), os_38220, 'sep')
        # Assigning a type to the variable 'sep' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'sep', sep_38221)
        
        
        # Getting the type of 'os' (line 47)
        os_38222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'os')
        # Obtaining the member 'sep' of a type (line 47)
        sep_38223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 11), os_38222, 'sep')
        str_38224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 21), 'str', '\\')
        # Applying the binary operator '==' (line 47)
        result_eq_38225 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 11), '==', sep_38223, str_38224)
        
        # Testing the type of an if condition (line 47)
        if_condition_38226 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 8), result_eq_38225)
        # Assigning a type to the variable 'if_condition_38226' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'if_condition_38226', if_condition_38226)
        # SSA begins for if statement (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 48):
        
        # Call to escape(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'os' (line 48)
        os_38229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 28), 'os', False)
        # Obtaining the member 'sep' of a type (line 48)
        sep_38230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 28), os_38229, 'sep')
        # Processing the call keyword arguments (line 48)
        kwargs_38231 = {}
        # Getting the type of 're' (line 48)
        re_38227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 18), 're', False)
        # Obtaining the member 'escape' of a type (line 48)
        escape_38228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 18), re_38227, 'escape')
        # Calling escape(args, kwargs) (line 48)
        escape_call_result_38232 = invoke(stypy.reporting.localization.Localization(__file__, 48, 18), escape_38228, *[sep_38230], **kwargs_38231)
        
        # Assigning a type to the variable 'sep' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'sep', escape_call_result_38232)
        # SSA join for if statement (line 47)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 52)
        tuple_38233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 12), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 52)
        # Adding element type (line 52)
        
        # Obtaining an instance of the builtin type 'tuple' (line 52)
        tuple_38234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 52)
        # Adding element type (line 52)
        str_38235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 13), 'str', 'foo*')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 13), tuple_38234, str_38235)
        # Adding element type (line 52)
        str_38236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 21), 'str', 'foo[^%(sep)s]*\\Z(?ms)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 13), tuple_38234, str_38236)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 12), tuple_38233, tuple_38234)
        # Adding element type (line 52)
        
        # Obtaining an instance of the builtin type 'tuple' (line 53)
        tuple_38237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 53)
        # Adding element type (line 53)
        str_38238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 13), 'str', 'foo?')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 13), tuple_38237, str_38238)
        # Adding element type (line 53)
        str_38239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 21), 'str', 'foo[^%(sep)s]\\Z(?ms)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 13), tuple_38237, str_38239)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 12), tuple_38233, tuple_38237)
        # Adding element type (line 52)
        
        # Obtaining an instance of the builtin type 'tuple' (line 54)
        tuple_38240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 54)
        # Adding element type (line 54)
        str_38241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 13), 'str', 'foo??')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 13), tuple_38240, str_38241)
        # Adding element type (line 54)
        str_38242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 22), 'str', 'foo[^%(sep)s][^%(sep)s]\\Z(?ms)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 13), tuple_38240, str_38242)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 12), tuple_38233, tuple_38240)
        # Adding element type (line 52)
        
        # Obtaining an instance of the builtin type 'tuple' (line 56)
        tuple_38243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 56)
        # Adding element type (line 56)
        str_38244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 13), 'str', 'foo\\\\*')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 13), tuple_38243, str_38244)
        # Adding element type (line 56)
        str_38245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 24), 'str', 'foo\\\\\\\\[^%(sep)s]*\\Z(?ms)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 13), tuple_38243, str_38245)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 12), tuple_38233, tuple_38243)
        # Adding element type (line 52)
        
        # Obtaining an instance of the builtin type 'tuple' (line 57)
        tuple_38246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 57)
        # Adding element type (line 57)
        str_38247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 13), 'str', 'foo\\\\\\*')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 13), tuple_38246, str_38247)
        # Adding element type (line 57)
        str_38248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 25), 'str', 'foo\\\\\\\\\\\\[^%(sep)s]*\\Z(?ms)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 13), tuple_38246, str_38248)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 12), tuple_38233, tuple_38246)
        # Adding element type (line 52)
        
        # Obtaining an instance of the builtin type 'tuple' (line 58)
        tuple_38249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 58)
        # Adding element type (line 58)
        str_38250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 13), 'str', 'foo????')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 13), tuple_38249, str_38250)
        # Adding element type (line 58)
        str_38251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 24), 'str', 'foo[^%(sep)s][^%(sep)s][^%(sep)s][^%(sep)s]\\Z(?ms)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 13), tuple_38249, str_38251)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 12), tuple_38233, tuple_38249)
        # Adding element type (line 52)
        
        # Obtaining an instance of the builtin type 'tuple' (line 59)
        tuple_38252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 59)
        # Adding element type (line 59)
        str_38253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 13), 'str', 'foo\\\\??')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 13), tuple_38252, str_38253)
        # Adding element type (line 59)
        str_38254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 25), 'str', 'foo\\\\\\\\[^%(sep)s][^%(sep)s]\\Z(?ms)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 13), tuple_38252, str_38254)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 12), tuple_38233, tuple_38252)
        
        # Testing the type of a for loop iterable (line 50)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 50, 8), tuple_38233)
        # Getting the type of the for loop variable (line 50)
        for_loop_var_38255 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 50, 8), tuple_38233)
        # Assigning a type to the variable 'glob' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'glob', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 8), for_loop_var_38255))
        # Assigning a type to the variable 'regex' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'regex', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 8), for_loop_var_38255))
        # SSA begins for a for statement (line 50)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 60):
        # Getting the type of 'regex' (line 60)
        regex_38256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'regex')
        
        # Obtaining an instance of the builtin type 'dict' (line 60)
        dict_38257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 60)
        # Adding element type (key, value) (line 60)
        str_38258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 29), 'str', 'sep')
        # Getting the type of 'sep' (line 60)
        sep_38259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 36), 'sep')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 28), dict_38257, (str_38258, sep_38259))
        
        # Applying the binary operator '%' (line 60)
        result_mod_38260 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 20), '%', regex_38256, dict_38257)
        
        # Assigning a type to the variable 'regex' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'regex', result_mod_38260)
        
        # Call to assertEqual(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Call to glob_to_re(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'glob' (line 61)
        glob_38264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 40), 'glob', False)
        # Processing the call keyword arguments (line 61)
        kwargs_38265 = {}
        # Getting the type of 'glob_to_re' (line 61)
        glob_to_re_38263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 'glob_to_re', False)
        # Calling glob_to_re(args, kwargs) (line 61)
        glob_to_re_call_result_38266 = invoke(stypy.reporting.localization.Localization(__file__, 61, 29), glob_to_re_38263, *[glob_38264], **kwargs_38265)
        
        # Getting the type of 'regex' (line 61)
        regex_38267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 47), 'regex', False)
        # Processing the call keyword arguments (line 61)
        kwargs_38268 = {}
        # Getting the type of 'self' (line 61)
        self_38261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 61)
        assertEqual_38262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), self_38261, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 61)
        assertEqual_call_result_38269 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), assertEqual_38262, *[glob_to_re_call_result_38266, regex_38267], **kwargs_38268)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_glob_to_re(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_glob_to_re' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_38270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_38270)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_glob_to_re'
        return stypy_return_type_38270


    @norecursion
    def test_process_template_line(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_process_template_line'
        module_type_store = module_type_store.open_function_context('test_process_template_line', 63, 4, False)
        # Assigning a type to the variable 'self' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileListTestCase.test_process_template_line.__dict__.__setitem__('stypy_localization', localization)
        FileListTestCase.test_process_template_line.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileListTestCase.test_process_template_line.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileListTestCase.test_process_template_line.__dict__.__setitem__('stypy_function_name', 'FileListTestCase.test_process_template_line')
        FileListTestCase.test_process_template_line.__dict__.__setitem__('stypy_param_names_list', [])
        FileListTestCase.test_process_template_line.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileListTestCase.test_process_template_line.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileListTestCase.test_process_template_line.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileListTestCase.test_process_template_line.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileListTestCase.test_process_template_line.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileListTestCase.test_process_template_line.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileListTestCase.test_process_template_line', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_process_template_line', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_process_template_line(...)' code ##################

        
        # Assigning a Call to a Name (line 65):
        
        # Call to FileList(...): (line 65)
        # Processing the call keyword arguments (line 65)
        kwargs_38272 = {}
        # Getting the type of 'FileList' (line 65)
        FileList_38271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'FileList', False)
        # Calling FileList(args, kwargs) (line 65)
        FileList_call_result_38273 = invoke(stypy.reporting.localization.Localization(__file__, 65, 20), FileList_38271, *[], **kwargs_38272)
        
        # Assigning a type to the variable 'file_list' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'file_list', FileList_call_result_38273)
        
        # Assigning a Name to a Name (line 66):
        # Getting the type of 'make_local_path' (line 66)
        make_local_path_38274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'make_local_path')
        # Assigning a type to the variable 'l' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'l', make_local_path_38274)
        
        # Assigning a List to a Attribute (line 69):
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_38275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        str_38276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 30), 'str', 'foo.tmp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_38275, str_38276)
        # Adding element type (line 69)
        str_38277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 41), 'str', 'ok')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_38275, str_38277)
        # Adding element type (line 69)
        str_38278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 47), 'str', 'xo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_38275, str_38278)
        # Adding element type (line 69)
        str_38279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 53), 'str', 'four.txt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_38275, str_38279)
        # Adding element type (line 69)
        str_38280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 30), 'str', 'buildout.cfg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_38275, str_38280)
        # Adding element type (line 69)
        
        # Call to l(...): (line 73)
        # Processing the call arguments (line 73)
        str_38282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 32), 'str', '.hg/last-message.txt')
        # Processing the call keyword arguments (line 73)
        kwargs_38283 = {}
        # Getting the type of 'l' (line 73)
        l_38281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 30), 'l', False)
        # Calling l(args, kwargs) (line 73)
        l_call_result_38284 = invoke(stypy.reporting.localization.Localization(__file__, 73, 30), l_38281, *[str_38282], **kwargs_38283)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_38275, l_call_result_38284)
        # Adding element type (line 69)
        
        # Call to l(...): (line 74)
        # Processing the call arguments (line 74)
        str_38286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 32), 'str', 'global/one.txt')
        # Processing the call keyword arguments (line 74)
        kwargs_38287 = {}
        # Getting the type of 'l' (line 74)
        l_38285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 30), 'l', False)
        # Calling l(args, kwargs) (line 74)
        l_call_result_38288 = invoke(stypy.reporting.localization.Localization(__file__, 74, 30), l_38285, *[str_38286], **kwargs_38287)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_38275, l_call_result_38288)
        # Adding element type (line 69)
        
        # Call to l(...): (line 75)
        # Processing the call arguments (line 75)
        str_38290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 32), 'str', 'global/two.txt')
        # Processing the call keyword arguments (line 75)
        kwargs_38291 = {}
        # Getting the type of 'l' (line 75)
        l_38289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 30), 'l', False)
        # Calling l(args, kwargs) (line 75)
        l_call_result_38292 = invoke(stypy.reporting.localization.Localization(__file__, 75, 30), l_38289, *[str_38290], **kwargs_38291)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_38275, l_call_result_38292)
        # Adding element type (line 69)
        
        # Call to l(...): (line 76)
        # Processing the call arguments (line 76)
        str_38294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 32), 'str', 'global/files.x')
        # Processing the call keyword arguments (line 76)
        kwargs_38295 = {}
        # Getting the type of 'l' (line 76)
        l_38293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 30), 'l', False)
        # Calling l(args, kwargs) (line 76)
        l_call_result_38296 = invoke(stypy.reporting.localization.Localization(__file__, 76, 30), l_38293, *[str_38294], **kwargs_38295)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_38275, l_call_result_38296)
        # Adding element type (line 69)
        
        # Call to l(...): (line 77)
        # Processing the call arguments (line 77)
        str_38298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 32), 'str', 'global/here.tmp')
        # Processing the call keyword arguments (line 77)
        kwargs_38299 = {}
        # Getting the type of 'l' (line 77)
        l_38297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 30), 'l', False)
        # Calling l(args, kwargs) (line 77)
        l_call_result_38300 = invoke(stypy.reporting.localization.Localization(__file__, 77, 30), l_38297, *[str_38298], **kwargs_38299)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_38275, l_call_result_38300)
        # Adding element type (line 69)
        
        # Call to l(...): (line 78)
        # Processing the call arguments (line 78)
        str_38302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 32), 'str', 'f/o/f.oo')
        # Processing the call keyword arguments (line 78)
        kwargs_38303 = {}
        # Getting the type of 'l' (line 78)
        l_38301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), 'l', False)
        # Calling l(args, kwargs) (line 78)
        l_call_result_38304 = invoke(stypy.reporting.localization.Localization(__file__, 78, 30), l_38301, *[str_38302], **kwargs_38303)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_38275, l_call_result_38304)
        # Adding element type (line 69)
        
        # Call to l(...): (line 79)
        # Processing the call arguments (line 79)
        str_38306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 32), 'str', 'dir/graft-one')
        # Processing the call keyword arguments (line 79)
        kwargs_38307 = {}
        # Getting the type of 'l' (line 79)
        l_38305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 30), 'l', False)
        # Calling l(args, kwargs) (line 79)
        l_call_result_38308 = invoke(stypy.reporting.localization.Localization(__file__, 79, 30), l_38305, *[str_38306], **kwargs_38307)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_38275, l_call_result_38308)
        # Adding element type (line 69)
        
        # Call to l(...): (line 80)
        # Processing the call arguments (line 80)
        str_38310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 32), 'str', 'dir/dir2/graft2')
        # Processing the call keyword arguments (line 80)
        kwargs_38311 = {}
        # Getting the type of 'l' (line 80)
        l_38309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 30), 'l', False)
        # Calling l(args, kwargs) (line 80)
        l_call_result_38312 = invoke(stypy.reporting.localization.Localization(__file__, 80, 30), l_38309, *[str_38310], **kwargs_38311)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_38275, l_call_result_38312)
        # Adding element type (line 69)
        
        # Call to l(...): (line 81)
        # Processing the call arguments (line 81)
        str_38314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 32), 'str', 'dir3/ok')
        # Processing the call keyword arguments (line 81)
        kwargs_38315 = {}
        # Getting the type of 'l' (line 81)
        l_38313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 30), 'l', False)
        # Calling l(args, kwargs) (line 81)
        l_call_result_38316 = invoke(stypy.reporting.localization.Localization(__file__, 81, 30), l_38313, *[str_38314], **kwargs_38315)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_38275, l_call_result_38316)
        # Adding element type (line 69)
        
        # Call to l(...): (line 82)
        # Processing the call arguments (line 82)
        str_38318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 32), 'str', 'dir3/sub/ok.txt')
        # Processing the call keyword arguments (line 82)
        kwargs_38319 = {}
        # Getting the type of 'l' (line 82)
        l_38317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 30), 'l', False)
        # Calling l(args, kwargs) (line 82)
        l_call_result_38320 = invoke(stypy.reporting.localization.Localization(__file__, 82, 30), l_38317, *[str_38318], **kwargs_38319)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 29), list_38275, l_call_result_38320)
        
        # Getting the type of 'file_list' (line 69)
        file_list_38321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'file_list')
        # Setting the type of the member 'allfiles' of a type (line 69)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), file_list_38321, 'allfiles', list_38275)
        
        
        # Call to split(...): (line 85)
        # Processing the call arguments (line 85)
        str_38324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 38), 'str', '\n')
        # Processing the call keyword arguments (line 85)
        kwargs_38325 = {}
        # Getting the type of 'MANIFEST_IN' (line 85)
        MANIFEST_IN_38322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'MANIFEST_IN', False)
        # Obtaining the member 'split' of a type (line 85)
        split_38323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 20), MANIFEST_IN_38322, 'split')
        # Calling split(args, kwargs) (line 85)
        split_call_result_38326 = invoke(stypy.reporting.localization.Localization(__file__, 85, 20), split_38323, *[str_38324], **kwargs_38325)
        
        # Testing the type of a for loop iterable (line 85)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 85, 8), split_call_result_38326)
        # Getting the type of the for loop variable (line 85)
        for_loop_var_38327 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 85, 8), split_call_result_38326)
        # Assigning a type to the variable 'line' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'line', for_loop_var_38327)
        # SSA begins for a for statement (line 85)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to strip(...): (line 86)
        # Processing the call keyword arguments (line 86)
        kwargs_38330 = {}
        # Getting the type of 'line' (line 86)
        line_38328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'line', False)
        # Obtaining the member 'strip' of a type (line 86)
        strip_38329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 15), line_38328, 'strip')
        # Calling strip(args, kwargs) (line 86)
        strip_call_result_38331 = invoke(stypy.reporting.localization.Localization(__file__, 86, 15), strip_38329, *[], **kwargs_38330)
        
        str_38332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 31), 'str', '')
        # Applying the binary operator '==' (line 86)
        result_eq_38333 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 15), '==', strip_call_result_38331, str_38332)
        
        # Testing the type of an if condition (line 86)
        if_condition_38334 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 12), result_eq_38333)
        # Assigning a type to the variable 'if_condition_38334' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'if_condition_38334', if_condition_38334)
        # SSA begins for if statement (line 86)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 86)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to process_template_line(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'line' (line 88)
        line_38337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 44), 'line', False)
        # Processing the call keyword arguments (line 88)
        kwargs_38338 = {}
        # Getting the type of 'file_list' (line 88)
        file_list_38335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'file_list', False)
        # Obtaining the member 'process_template_line' of a type (line 88)
        process_template_line_38336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), file_list_38335, 'process_template_line')
        # Calling process_template_line(args, kwargs) (line 88)
        process_template_line_call_result_38339 = invoke(stypy.reporting.localization.Localization(__file__, 88, 12), process_template_line_38336, *[line_38337], **kwargs_38338)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 90):
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_38340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        str_38341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 18), 'str', 'ok')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 17), list_38340, str_38341)
        # Adding element type (line 90)
        str_38342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 18), 'str', 'buildout.cfg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 17), list_38340, str_38342)
        # Adding element type (line 90)
        str_38343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 18), 'str', 'four.txt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 17), list_38340, str_38343)
        # Adding element type (line 90)
        
        # Call to l(...): (line 93)
        # Processing the call arguments (line 93)
        str_38345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 20), 'str', '.hg/last-message.txt')
        # Processing the call keyword arguments (line 93)
        kwargs_38346 = {}
        # Getting the type of 'l' (line 93)
        l_38344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 18), 'l', False)
        # Calling l(args, kwargs) (line 93)
        l_call_result_38347 = invoke(stypy.reporting.localization.Localization(__file__, 93, 18), l_38344, *[str_38345], **kwargs_38346)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 17), list_38340, l_call_result_38347)
        # Adding element type (line 90)
        
        # Call to l(...): (line 94)
        # Processing the call arguments (line 94)
        str_38349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 20), 'str', 'global/one.txt')
        # Processing the call keyword arguments (line 94)
        kwargs_38350 = {}
        # Getting the type of 'l' (line 94)
        l_38348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 18), 'l', False)
        # Calling l(args, kwargs) (line 94)
        l_call_result_38351 = invoke(stypy.reporting.localization.Localization(__file__, 94, 18), l_38348, *[str_38349], **kwargs_38350)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 17), list_38340, l_call_result_38351)
        # Adding element type (line 90)
        
        # Call to l(...): (line 95)
        # Processing the call arguments (line 95)
        str_38353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 20), 'str', 'global/two.txt')
        # Processing the call keyword arguments (line 95)
        kwargs_38354 = {}
        # Getting the type of 'l' (line 95)
        l_38352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 18), 'l', False)
        # Calling l(args, kwargs) (line 95)
        l_call_result_38355 = invoke(stypy.reporting.localization.Localization(__file__, 95, 18), l_38352, *[str_38353], **kwargs_38354)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 17), list_38340, l_call_result_38355)
        # Adding element type (line 90)
        
        # Call to l(...): (line 96)
        # Processing the call arguments (line 96)
        str_38357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 20), 'str', 'f/o/f.oo')
        # Processing the call keyword arguments (line 96)
        kwargs_38358 = {}
        # Getting the type of 'l' (line 96)
        l_38356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 18), 'l', False)
        # Calling l(args, kwargs) (line 96)
        l_call_result_38359 = invoke(stypy.reporting.localization.Localization(__file__, 96, 18), l_38356, *[str_38357], **kwargs_38358)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 17), list_38340, l_call_result_38359)
        # Adding element type (line 90)
        
        # Call to l(...): (line 97)
        # Processing the call arguments (line 97)
        str_38361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 20), 'str', 'dir/graft-one')
        # Processing the call keyword arguments (line 97)
        kwargs_38362 = {}
        # Getting the type of 'l' (line 97)
        l_38360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 18), 'l', False)
        # Calling l(args, kwargs) (line 97)
        l_call_result_38363 = invoke(stypy.reporting.localization.Localization(__file__, 97, 18), l_38360, *[str_38361], **kwargs_38362)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 17), list_38340, l_call_result_38363)
        # Adding element type (line 90)
        
        # Call to l(...): (line 98)
        # Processing the call arguments (line 98)
        str_38365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 20), 'str', 'dir/dir2/graft2')
        # Processing the call keyword arguments (line 98)
        kwargs_38366 = {}
        # Getting the type of 'l' (line 98)
        l_38364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 18), 'l', False)
        # Calling l(args, kwargs) (line 98)
        l_call_result_38367 = invoke(stypy.reporting.localization.Localization(__file__, 98, 18), l_38364, *[str_38365], **kwargs_38366)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 17), list_38340, l_call_result_38367)
        
        # Assigning a type to the variable 'wanted' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'wanted', list_38340)
        
        # Call to assertEqual(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'file_list' (line 101)
        file_list_38370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 25), 'file_list', False)
        # Obtaining the member 'files' of a type (line 101)
        files_38371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 25), file_list_38370, 'files')
        # Getting the type of 'wanted' (line 101)
        wanted_38372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 42), 'wanted', False)
        # Processing the call keyword arguments (line 101)
        kwargs_38373 = {}
        # Getting the type of 'self' (line 101)
        self_38368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 101)
        assertEqual_38369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), self_38368, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 101)
        assertEqual_call_result_38374 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), assertEqual_38369, *[files_38371, wanted_38372], **kwargs_38373)
        
        
        # ################# End of 'test_process_template_line(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_process_template_line' in the type store
        # Getting the type of 'stypy_return_type' (line 63)
        stypy_return_type_38375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_38375)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_process_template_line'
        return stypy_return_type_38375


    @norecursion
    def test_debug_print(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_debug_print'
        module_type_store = module_type_store.open_function_context('test_debug_print', 103, 4, False)
        # Assigning a type to the variable 'self' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileListTestCase.test_debug_print.__dict__.__setitem__('stypy_localization', localization)
        FileListTestCase.test_debug_print.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileListTestCase.test_debug_print.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileListTestCase.test_debug_print.__dict__.__setitem__('stypy_function_name', 'FileListTestCase.test_debug_print')
        FileListTestCase.test_debug_print.__dict__.__setitem__('stypy_param_names_list', [])
        FileListTestCase.test_debug_print.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileListTestCase.test_debug_print.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileListTestCase.test_debug_print.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileListTestCase.test_debug_print.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileListTestCase.test_debug_print.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileListTestCase.test_debug_print.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileListTestCase.test_debug_print', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_debug_print', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_debug_print(...)' code ##################

        
        # Assigning a Call to a Name (line 104):
        
        # Call to FileList(...): (line 104)
        # Processing the call keyword arguments (line 104)
        kwargs_38377 = {}
        # Getting the type of 'FileList' (line 104)
        FileList_38376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), 'FileList', False)
        # Calling FileList(args, kwargs) (line 104)
        FileList_call_result_38378 = invoke(stypy.reporting.localization.Localization(__file__, 104, 20), FileList_38376, *[], **kwargs_38377)
        
        # Assigning a type to the variable 'file_list' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'file_list', FileList_call_result_38378)
        
        # Call to captured_stdout(...): (line 105)
        # Processing the call keyword arguments (line 105)
        kwargs_38380 = {}
        # Getting the type of 'captured_stdout' (line 105)
        captured_stdout_38379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 13), 'captured_stdout', False)
        # Calling captured_stdout(args, kwargs) (line 105)
        captured_stdout_call_result_38381 = invoke(stypy.reporting.localization.Localization(__file__, 105, 13), captured_stdout_38379, *[], **kwargs_38380)
        
        with_38382 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 105, 13), captured_stdout_call_result_38381, 'with parameter', '__enter__', '__exit__')

        if with_38382:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 105)
            enter___38383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 13), captured_stdout_call_result_38381, '__enter__')
            with_enter_38384 = invoke(stypy.reporting.localization.Localization(__file__, 105, 13), enter___38383)
            # Assigning a type to the variable 'stdout' (line 105)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 13), 'stdout', with_enter_38384)
            
            # Call to debug_print(...): (line 106)
            # Processing the call arguments (line 106)
            str_38387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 34), 'str', 'xxx')
            # Processing the call keyword arguments (line 106)
            kwargs_38388 = {}
            # Getting the type of 'file_list' (line 106)
            file_list_38385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'file_list', False)
            # Obtaining the member 'debug_print' of a type (line 106)
            debug_print_38386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), file_list_38385, 'debug_print')
            # Calling debug_print(args, kwargs) (line 106)
            debug_print_call_result_38389 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), debug_print_38386, *[str_38387], **kwargs_38388)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 105)
            exit___38390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 13), captured_stdout_call_result_38381, '__exit__')
            with_exit_38391 = invoke(stypy.reporting.localization.Localization(__file__, 105, 13), exit___38390, None, None, None)

        
        # Call to assertEqual(...): (line 107)
        # Processing the call arguments (line 107)
        
        # Call to getvalue(...): (line 107)
        # Processing the call keyword arguments (line 107)
        kwargs_38396 = {}
        # Getting the type of 'stdout' (line 107)
        stdout_38394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 25), 'stdout', False)
        # Obtaining the member 'getvalue' of a type (line 107)
        getvalue_38395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 25), stdout_38394, 'getvalue')
        # Calling getvalue(args, kwargs) (line 107)
        getvalue_call_result_38397 = invoke(stypy.reporting.localization.Localization(__file__, 107, 25), getvalue_38395, *[], **kwargs_38396)
        
        str_38398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 44), 'str', '')
        # Processing the call keyword arguments (line 107)
        kwargs_38399 = {}
        # Getting the type of 'self' (line 107)
        self_38392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 107)
        assertEqual_38393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 8), self_38392, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 107)
        assertEqual_call_result_38400 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), assertEqual_38393, *[getvalue_call_result_38397, str_38398], **kwargs_38399)
        
        
        # Assigning a Name to a Attribute (line 109):
        # Getting the type of 'True' (line 109)
        True_38401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 22), 'True')
        # Getting the type of 'debug' (line 109)
        debug_38402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'debug')
        # Setting the type of the member 'DEBUG' of a type (line 109)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), debug_38402, 'DEBUG', True_38401)
        
        # Try-finally block (line 110)
        
        # Call to captured_stdout(...): (line 111)
        # Processing the call keyword arguments (line 111)
        kwargs_38404 = {}
        # Getting the type of 'captured_stdout' (line 111)
        captured_stdout_38403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 17), 'captured_stdout', False)
        # Calling captured_stdout(args, kwargs) (line 111)
        captured_stdout_call_result_38405 = invoke(stypy.reporting.localization.Localization(__file__, 111, 17), captured_stdout_38403, *[], **kwargs_38404)
        
        with_38406 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 111, 17), captured_stdout_call_result_38405, 'with parameter', '__enter__', '__exit__')

        if with_38406:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 111)
            enter___38407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 17), captured_stdout_call_result_38405, '__enter__')
            with_enter_38408 = invoke(stypy.reporting.localization.Localization(__file__, 111, 17), enter___38407)
            # Assigning a type to the variable 'stdout' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 17), 'stdout', with_enter_38408)
            
            # Call to debug_print(...): (line 112)
            # Processing the call arguments (line 112)
            str_38411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 38), 'str', 'xxx')
            # Processing the call keyword arguments (line 112)
            kwargs_38412 = {}
            # Getting the type of 'file_list' (line 112)
            file_list_38409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'file_list', False)
            # Obtaining the member 'debug_print' of a type (line 112)
            debug_print_38410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 16), file_list_38409, 'debug_print')
            # Calling debug_print(args, kwargs) (line 112)
            debug_print_call_result_38413 = invoke(stypy.reporting.localization.Localization(__file__, 112, 16), debug_print_38410, *[str_38411], **kwargs_38412)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 111)
            exit___38414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 17), captured_stdout_call_result_38405, '__exit__')
            with_exit_38415 = invoke(stypy.reporting.localization.Localization(__file__, 111, 17), exit___38414, None, None, None)

        
        # Call to assertEqual(...): (line 113)
        # Processing the call arguments (line 113)
        
        # Call to getvalue(...): (line 113)
        # Processing the call keyword arguments (line 113)
        kwargs_38420 = {}
        # Getting the type of 'stdout' (line 113)
        stdout_38418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 29), 'stdout', False)
        # Obtaining the member 'getvalue' of a type (line 113)
        getvalue_38419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 29), stdout_38418, 'getvalue')
        # Calling getvalue(args, kwargs) (line 113)
        getvalue_call_result_38421 = invoke(stypy.reporting.localization.Localization(__file__, 113, 29), getvalue_38419, *[], **kwargs_38420)
        
        str_38422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 48), 'str', 'xxx\n')
        # Processing the call keyword arguments (line 113)
        kwargs_38423 = {}
        # Getting the type of 'self' (line 113)
        self_38416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 113)
        assertEqual_38417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 12), self_38416, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 113)
        assertEqual_call_result_38424 = invoke(stypy.reporting.localization.Localization(__file__, 113, 12), assertEqual_38417, *[getvalue_call_result_38421, str_38422], **kwargs_38423)
        
        
        # finally branch of the try-finally block (line 110)
        
        # Assigning a Name to a Attribute (line 115):
        # Getting the type of 'False' (line 115)
        False_38425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 26), 'False')
        # Getting the type of 'debug' (line 115)
        debug_38426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'debug')
        # Setting the type of the member 'DEBUG' of a type (line 115)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), debug_38426, 'DEBUG', False_38425)
        
        
        # ################# End of 'test_debug_print(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_debug_print' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_38427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_38427)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_debug_print'
        return stypy_return_type_38427


    @norecursion
    def test_set_allfiles(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_set_allfiles'
        module_type_store = module_type_store.open_function_context('test_set_allfiles', 117, 4, False)
        # Assigning a type to the variable 'self' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileListTestCase.test_set_allfiles.__dict__.__setitem__('stypy_localization', localization)
        FileListTestCase.test_set_allfiles.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileListTestCase.test_set_allfiles.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileListTestCase.test_set_allfiles.__dict__.__setitem__('stypy_function_name', 'FileListTestCase.test_set_allfiles')
        FileListTestCase.test_set_allfiles.__dict__.__setitem__('stypy_param_names_list', [])
        FileListTestCase.test_set_allfiles.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileListTestCase.test_set_allfiles.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileListTestCase.test_set_allfiles.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileListTestCase.test_set_allfiles.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileListTestCase.test_set_allfiles.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileListTestCase.test_set_allfiles.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileListTestCase.test_set_allfiles', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_set_allfiles', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_set_allfiles(...)' code ##################

        
        # Assigning a Call to a Name (line 118):
        
        # Call to FileList(...): (line 118)
        # Processing the call keyword arguments (line 118)
        kwargs_38429 = {}
        # Getting the type of 'FileList' (line 118)
        FileList_38428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 20), 'FileList', False)
        # Calling FileList(args, kwargs) (line 118)
        FileList_call_result_38430 = invoke(stypy.reporting.localization.Localization(__file__, 118, 20), FileList_38428, *[], **kwargs_38429)
        
        # Assigning a type to the variable 'file_list' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'file_list', FileList_call_result_38430)
        
        # Assigning a List to a Name (line 119):
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_38431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        str_38432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 17), 'str', 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 16), list_38431, str_38432)
        # Adding element type (line 119)
        str_38433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 22), 'str', 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 16), list_38431, str_38433)
        # Adding element type (line 119)
        str_38434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 27), 'str', 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 16), list_38431, str_38434)
        
        # Assigning a type to the variable 'files' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'files', list_38431)
        
        # Call to set_allfiles(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'files' (line 120)
        files_38437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 31), 'files', False)
        # Processing the call keyword arguments (line 120)
        kwargs_38438 = {}
        # Getting the type of 'file_list' (line 120)
        file_list_38435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'file_list', False)
        # Obtaining the member 'set_allfiles' of a type (line 120)
        set_allfiles_38436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), file_list_38435, 'set_allfiles')
        # Calling set_allfiles(args, kwargs) (line 120)
        set_allfiles_call_result_38439 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), set_allfiles_38436, *[files_38437], **kwargs_38438)
        
        
        # Call to assertEqual(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'file_list' (line 121)
        file_list_38442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 25), 'file_list', False)
        # Obtaining the member 'allfiles' of a type (line 121)
        allfiles_38443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 25), file_list_38442, 'allfiles')
        # Getting the type of 'files' (line 121)
        files_38444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 45), 'files', False)
        # Processing the call keyword arguments (line 121)
        kwargs_38445 = {}
        # Getting the type of 'self' (line 121)
        self_38440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 121)
        assertEqual_38441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), self_38440, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 121)
        assertEqual_call_result_38446 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), assertEqual_38441, *[allfiles_38443, files_38444], **kwargs_38445)
        
        
        # ################# End of 'test_set_allfiles(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_set_allfiles' in the type store
        # Getting the type of 'stypy_return_type' (line 117)
        stypy_return_type_38447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_38447)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_set_allfiles'
        return stypy_return_type_38447


    @norecursion
    def test_remove_duplicates(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_remove_duplicates'
        module_type_store = module_type_store.open_function_context('test_remove_duplicates', 123, 4, False)
        # Assigning a type to the variable 'self' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileListTestCase.test_remove_duplicates.__dict__.__setitem__('stypy_localization', localization)
        FileListTestCase.test_remove_duplicates.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileListTestCase.test_remove_duplicates.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileListTestCase.test_remove_duplicates.__dict__.__setitem__('stypy_function_name', 'FileListTestCase.test_remove_duplicates')
        FileListTestCase.test_remove_duplicates.__dict__.__setitem__('stypy_param_names_list', [])
        FileListTestCase.test_remove_duplicates.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileListTestCase.test_remove_duplicates.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileListTestCase.test_remove_duplicates.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileListTestCase.test_remove_duplicates.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileListTestCase.test_remove_duplicates.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileListTestCase.test_remove_duplicates.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileListTestCase.test_remove_duplicates', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_remove_duplicates', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_remove_duplicates(...)' code ##################

        
        # Assigning a Call to a Name (line 124):
        
        # Call to FileList(...): (line 124)
        # Processing the call keyword arguments (line 124)
        kwargs_38449 = {}
        # Getting the type of 'FileList' (line 124)
        FileList_38448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'FileList', False)
        # Calling FileList(args, kwargs) (line 124)
        FileList_call_result_38450 = invoke(stypy.reporting.localization.Localization(__file__, 124, 20), FileList_38448, *[], **kwargs_38449)
        
        # Assigning a type to the variable 'file_list' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'file_list', FileList_call_result_38450)
        
        # Assigning a List to a Attribute (line 125):
        
        # Obtaining an instance of the builtin type 'list' (line 125)
        list_38451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 125)
        # Adding element type (line 125)
        str_38452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 27), 'str', 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 26), list_38451, str_38452)
        # Adding element type (line 125)
        str_38453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 32), 'str', 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 26), list_38451, str_38453)
        # Adding element type (line 125)
        str_38454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 37), 'str', 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 26), list_38451, str_38454)
        # Adding element type (line 125)
        str_38455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 42), 'str', 'g')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 26), list_38451, str_38455)
        # Adding element type (line 125)
        str_38456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 47), 'str', 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 26), list_38451, str_38456)
        # Adding element type (line 125)
        str_38457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 52), 'str', 'g')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 26), list_38451, str_38457)
        
        # Getting the type of 'file_list' (line 125)
        file_list_38458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'file_list')
        # Setting the type of the member 'files' of a type (line 125)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), file_list_38458, 'files', list_38451)
        
        # Call to sort(...): (line 127)
        # Processing the call keyword arguments (line 127)
        kwargs_38461 = {}
        # Getting the type of 'file_list' (line 127)
        file_list_38459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'file_list', False)
        # Obtaining the member 'sort' of a type (line 127)
        sort_38460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), file_list_38459, 'sort')
        # Calling sort(args, kwargs) (line 127)
        sort_call_result_38462 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), sort_38460, *[], **kwargs_38461)
        
        
        # Call to remove_duplicates(...): (line 128)
        # Processing the call keyword arguments (line 128)
        kwargs_38465 = {}
        # Getting the type of 'file_list' (line 128)
        file_list_38463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'file_list', False)
        # Obtaining the member 'remove_duplicates' of a type (line 128)
        remove_duplicates_38464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), file_list_38463, 'remove_duplicates')
        # Calling remove_duplicates(args, kwargs) (line 128)
        remove_duplicates_call_result_38466 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), remove_duplicates_38464, *[], **kwargs_38465)
        
        
        # Call to assertEqual(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'file_list' (line 129)
        file_list_38469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 25), 'file_list', False)
        # Obtaining the member 'files' of a type (line 129)
        files_38470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 25), file_list_38469, 'files')
        
        # Obtaining an instance of the builtin type 'list' (line 129)
        list_38471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 129)
        # Adding element type (line 129)
        str_38472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 43), 'str', 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 42), list_38471, str_38472)
        # Adding element type (line 129)
        str_38473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 48), 'str', 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 42), list_38471, str_38473)
        # Adding element type (line 129)
        str_38474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 53), 'str', 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 42), list_38471, str_38474)
        # Adding element type (line 129)
        str_38475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 58), 'str', 'g')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 42), list_38471, str_38475)
        
        # Processing the call keyword arguments (line 129)
        kwargs_38476 = {}
        # Getting the type of 'self' (line 129)
        self_38467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 129)
        assertEqual_38468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_38467, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 129)
        assertEqual_call_result_38477 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), assertEqual_38468, *[files_38470, list_38471], **kwargs_38476)
        
        
        # ################# End of 'test_remove_duplicates(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_remove_duplicates' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_38478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_38478)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_remove_duplicates'
        return stypy_return_type_38478


    @norecursion
    def test_translate_pattern(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_translate_pattern'
        module_type_store = module_type_store.open_function_context('test_translate_pattern', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileListTestCase.test_translate_pattern.__dict__.__setitem__('stypy_localization', localization)
        FileListTestCase.test_translate_pattern.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileListTestCase.test_translate_pattern.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileListTestCase.test_translate_pattern.__dict__.__setitem__('stypy_function_name', 'FileListTestCase.test_translate_pattern')
        FileListTestCase.test_translate_pattern.__dict__.__setitem__('stypy_param_names_list', [])
        FileListTestCase.test_translate_pattern.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileListTestCase.test_translate_pattern.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileListTestCase.test_translate_pattern.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileListTestCase.test_translate_pattern.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileListTestCase.test_translate_pattern.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileListTestCase.test_translate_pattern.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileListTestCase.test_translate_pattern', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_translate_pattern', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_translate_pattern(...)' code ##################

        
        # Call to assertTrue(...): (line 133)
        # Processing the call arguments (line 133)
        
        # Call to hasattr(...): (line 133)
        # Processing the call arguments (line 133)
        
        # Call to translate_pattern(...): (line 134)
        # Processing the call arguments (line 134)
        str_38483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 30), 'str', 'a')
        # Processing the call keyword arguments (line 134)
        # Getting the type of 'True' (line 134)
        True_38484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 42), 'True', False)
        keyword_38485 = True_38484
        # Getting the type of 'False' (line 134)
        False_38486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 57), 'False', False)
        keyword_38487 = False_38486
        kwargs_38488 = {'is_regex': keyword_38487, 'anchor': keyword_38485}
        # Getting the type of 'translate_pattern' (line 134)
        translate_pattern_38482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'translate_pattern', False)
        # Calling translate_pattern(args, kwargs) (line 134)
        translate_pattern_call_result_38489 = invoke(stypy.reporting.localization.Localization(__file__, 134, 12), translate_pattern_38482, *[str_38483], **kwargs_38488)
        
        str_38490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 12), 'str', 'search')
        # Processing the call keyword arguments (line 133)
        kwargs_38491 = {}
        # Getting the type of 'hasattr' (line 133)
        hasattr_38481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 133)
        hasattr_call_result_38492 = invoke(stypy.reporting.localization.Localization(__file__, 133, 24), hasattr_38481, *[translate_pattern_call_result_38489, str_38490], **kwargs_38491)
        
        # Processing the call keyword arguments (line 133)
        kwargs_38493 = {}
        # Getting the type of 'self' (line 133)
        self_38479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 133)
        assertTrue_38480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), self_38479, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 133)
        assertTrue_call_result_38494 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), assertTrue_38480, *[hasattr_call_result_38492], **kwargs_38493)
        
        
        # Assigning a Call to a Name (line 138):
        
        # Call to compile(...): (line 138)
        # Processing the call arguments (line 138)
        str_38497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 27), 'str', 'a')
        # Processing the call keyword arguments (line 138)
        kwargs_38498 = {}
        # Getting the type of 're' (line 138)
        re_38495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 're', False)
        # Obtaining the member 'compile' of a type (line 138)
        compile_38496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 16), re_38495, 'compile')
        # Calling compile(args, kwargs) (line 138)
        compile_call_result_38499 = invoke(stypy.reporting.localization.Localization(__file__, 138, 16), compile_38496, *[str_38497], **kwargs_38498)
        
        # Assigning a type to the variable 'regex' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'regex', compile_call_result_38499)
        
        # Call to assertEqual(...): (line 139)
        # Processing the call arguments (line 139)
        
        # Call to translate_pattern(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'regex' (line 140)
        regex_38503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 30), 'regex', False)
        # Processing the call keyword arguments (line 140)
        # Getting the type of 'True' (line 140)
        True_38504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 44), 'True', False)
        keyword_38505 = True_38504
        # Getting the type of 'True' (line 140)
        True_38506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 59), 'True', False)
        keyword_38507 = True_38506
        kwargs_38508 = {'is_regex': keyword_38507, 'anchor': keyword_38505}
        # Getting the type of 'translate_pattern' (line 140)
        translate_pattern_38502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'translate_pattern', False)
        # Calling translate_pattern(args, kwargs) (line 140)
        translate_pattern_call_result_38509 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), translate_pattern_38502, *[regex_38503], **kwargs_38508)
        
        # Getting the type of 'regex' (line 141)
        regex_38510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'regex', False)
        # Processing the call keyword arguments (line 139)
        kwargs_38511 = {}
        # Getting the type of 'self' (line 139)
        self_38500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 139)
        assertEqual_38501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), self_38500, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 139)
        assertEqual_call_result_38512 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), assertEqual_38501, *[translate_pattern_call_result_38509, regex_38510], **kwargs_38511)
        
        
        # Call to assertTrue(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Call to hasattr(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Call to translate_pattern(...): (line 145)
        # Processing the call arguments (line 145)
        str_38517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 30), 'str', 'a')
        # Processing the call keyword arguments (line 145)
        # Getting the type of 'True' (line 145)
        True_38518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 42), 'True', False)
        keyword_38519 = True_38518
        # Getting the type of 'True' (line 145)
        True_38520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 57), 'True', False)
        keyword_38521 = True_38520
        kwargs_38522 = {'is_regex': keyword_38521, 'anchor': keyword_38519}
        # Getting the type of 'translate_pattern' (line 145)
        translate_pattern_38516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'translate_pattern', False)
        # Calling translate_pattern(args, kwargs) (line 145)
        translate_pattern_call_result_38523 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), translate_pattern_38516, *[str_38517], **kwargs_38522)
        
        str_38524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 12), 'str', 'search')
        # Processing the call keyword arguments (line 144)
        kwargs_38525 = {}
        # Getting the type of 'hasattr' (line 144)
        hasattr_38515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 24), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 144)
        hasattr_call_result_38526 = invoke(stypy.reporting.localization.Localization(__file__, 144, 24), hasattr_38515, *[translate_pattern_call_result_38523, str_38524], **kwargs_38525)
        
        # Processing the call keyword arguments (line 144)
        kwargs_38527 = {}
        # Getting the type of 'self' (line 144)
        self_38513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 144)
        assertTrue_38514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), self_38513, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 144)
        assertTrue_call_result_38528 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), assertTrue_38514, *[hasattr_call_result_38526], **kwargs_38527)
        
        
        # Call to assertTrue(...): (line 149)
        # Processing the call arguments (line 149)
        
        # Call to search(...): (line 149)
        # Processing the call arguments (line 149)
        str_38540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 56), 'str', 'filelist.py')
        # Processing the call keyword arguments (line 149)
        kwargs_38541 = {}
        
        # Call to translate_pattern(...): (line 149)
        # Processing the call arguments (line 149)
        str_38532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 12), 'str', '*.py')
        # Processing the call keyword arguments (line 149)
        # Getting the type of 'True' (line 150)
        True_38533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 27), 'True', False)
        keyword_38534 = True_38533
        # Getting the type of 'False' (line 150)
        False_38535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 42), 'False', False)
        keyword_38536 = False_38535
        kwargs_38537 = {'is_regex': keyword_38536, 'anchor': keyword_38534}
        # Getting the type of 'translate_pattern' (line 149)
        translate_pattern_38531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'translate_pattern', False)
        # Calling translate_pattern(args, kwargs) (line 149)
        translate_pattern_call_result_38538 = invoke(stypy.reporting.localization.Localization(__file__, 149, 24), translate_pattern_38531, *[str_38532], **kwargs_38537)
        
        # Obtaining the member 'search' of a type (line 149)
        search_38539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 24), translate_pattern_call_result_38538, 'search')
        # Calling search(args, kwargs) (line 149)
        search_call_result_38542 = invoke(stypy.reporting.localization.Localization(__file__, 149, 24), search_38539, *[str_38540], **kwargs_38541)
        
        # Processing the call keyword arguments (line 149)
        kwargs_38543 = {}
        # Getting the type of 'self' (line 149)
        self_38529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 149)
        assertTrue_38530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), self_38529, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 149)
        assertTrue_call_result_38544 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), assertTrue_38530, *[search_call_result_38542], **kwargs_38543)
        
        
        # ################# End of 'test_translate_pattern(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_translate_pattern' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_38545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_38545)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_translate_pattern'
        return stypy_return_type_38545


    @norecursion
    def test_exclude_pattern(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_exclude_pattern'
        module_type_store = module_type_store.open_function_context('test_exclude_pattern', 152, 4, False)
        # Assigning a type to the variable 'self' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileListTestCase.test_exclude_pattern.__dict__.__setitem__('stypy_localization', localization)
        FileListTestCase.test_exclude_pattern.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileListTestCase.test_exclude_pattern.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileListTestCase.test_exclude_pattern.__dict__.__setitem__('stypy_function_name', 'FileListTestCase.test_exclude_pattern')
        FileListTestCase.test_exclude_pattern.__dict__.__setitem__('stypy_param_names_list', [])
        FileListTestCase.test_exclude_pattern.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileListTestCase.test_exclude_pattern.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileListTestCase.test_exclude_pattern.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileListTestCase.test_exclude_pattern.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileListTestCase.test_exclude_pattern.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileListTestCase.test_exclude_pattern.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileListTestCase.test_exclude_pattern', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_exclude_pattern', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_exclude_pattern(...)' code ##################

        
        # Assigning a Call to a Name (line 154):
        
        # Call to FileList(...): (line 154)
        # Processing the call keyword arguments (line 154)
        kwargs_38547 = {}
        # Getting the type of 'FileList' (line 154)
        FileList_38546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'FileList', False)
        # Calling FileList(args, kwargs) (line 154)
        FileList_call_result_38548 = invoke(stypy.reporting.localization.Localization(__file__, 154, 20), FileList_38546, *[], **kwargs_38547)
        
        # Assigning a type to the variable 'file_list' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'file_list', FileList_call_result_38548)
        
        # Call to assertFalse(...): (line 155)
        # Processing the call arguments (line 155)
        
        # Call to exclude_pattern(...): (line 155)
        # Processing the call arguments (line 155)
        str_38553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 51), 'str', '*.py')
        # Processing the call keyword arguments (line 155)
        kwargs_38554 = {}
        # Getting the type of 'file_list' (line 155)
        file_list_38551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 25), 'file_list', False)
        # Obtaining the member 'exclude_pattern' of a type (line 155)
        exclude_pattern_38552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 25), file_list_38551, 'exclude_pattern')
        # Calling exclude_pattern(args, kwargs) (line 155)
        exclude_pattern_call_result_38555 = invoke(stypy.reporting.localization.Localization(__file__, 155, 25), exclude_pattern_38552, *[str_38553], **kwargs_38554)
        
        # Processing the call keyword arguments (line 155)
        kwargs_38556 = {}
        # Getting the type of 'self' (line 155)
        self_38549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 155)
        assertFalse_38550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), self_38549, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 155)
        assertFalse_call_result_38557 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), assertFalse_38550, *[exclude_pattern_call_result_38555], **kwargs_38556)
        
        
        # Assigning a Call to a Name (line 158):
        
        # Call to FileList(...): (line 158)
        # Processing the call keyword arguments (line 158)
        kwargs_38559 = {}
        # Getting the type of 'FileList' (line 158)
        FileList_38558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 20), 'FileList', False)
        # Calling FileList(args, kwargs) (line 158)
        FileList_call_result_38560 = invoke(stypy.reporting.localization.Localization(__file__, 158, 20), FileList_38558, *[], **kwargs_38559)
        
        # Assigning a type to the variable 'file_list' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'file_list', FileList_call_result_38560)
        
        # Assigning a List to a Attribute (line 159):
        
        # Obtaining an instance of the builtin type 'list' (line 159)
        list_38561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 159)
        # Adding element type (line 159)
        str_38562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 27), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 26), list_38561, str_38562)
        # Adding element type (line 159)
        str_38563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 35), 'str', 'b.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 26), list_38561, str_38563)
        
        # Getting the type of 'file_list' (line 159)
        file_list_38564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'file_list')
        # Setting the type of the member 'files' of a type (line 159)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), file_list_38564, 'files', list_38561)
        
        # Call to assertTrue(...): (line 160)
        # Processing the call arguments (line 160)
        
        # Call to exclude_pattern(...): (line 160)
        # Processing the call arguments (line 160)
        str_38569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 50), 'str', '*.py')
        # Processing the call keyword arguments (line 160)
        kwargs_38570 = {}
        # Getting the type of 'file_list' (line 160)
        file_list_38567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'file_list', False)
        # Obtaining the member 'exclude_pattern' of a type (line 160)
        exclude_pattern_38568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 24), file_list_38567, 'exclude_pattern')
        # Calling exclude_pattern(args, kwargs) (line 160)
        exclude_pattern_call_result_38571 = invoke(stypy.reporting.localization.Localization(__file__, 160, 24), exclude_pattern_38568, *[str_38569], **kwargs_38570)
        
        # Processing the call keyword arguments (line 160)
        kwargs_38572 = {}
        # Getting the type of 'self' (line 160)
        self_38565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 160)
        assertTrue_38566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), self_38565, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 160)
        assertTrue_call_result_38573 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), assertTrue_38566, *[exclude_pattern_call_result_38571], **kwargs_38572)
        
        
        # Assigning a Call to a Name (line 163):
        
        # Call to FileList(...): (line 163)
        # Processing the call keyword arguments (line 163)
        kwargs_38575 = {}
        # Getting the type of 'FileList' (line 163)
        FileList_38574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'FileList', False)
        # Calling FileList(args, kwargs) (line 163)
        FileList_call_result_38576 = invoke(stypy.reporting.localization.Localization(__file__, 163, 20), FileList_38574, *[], **kwargs_38575)
        
        # Assigning a type to the variable 'file_list' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'file_list', FileList_call_result_38576)
        
        # Assigning a List to a Attribute (line 164):
        
        # Obtaining an instance of the builtin type 'list' (line 164)
        list_38577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 164)
        # Adding element type (line 164)
        str_38578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 27), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 26), list_38577, str_38578)
        # Adding element type (line 164)
        str_38579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 35), 'str', 'a.txt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 26), list_38577, str_38579)
        
        # Getting the type of 'file_list' (line 164)
        file_list_38580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'file_list')
        # Setting the type of the member 'files' of a type (line 164)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), file_list_38580, 'files', list_38577)
        
        # Call to exclude_pattern(...): (line 165)
        # Processing the call arguments (line 165)
        str_38583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 34), 'str', '*.py')
        # Processing the call keyword arguments (line 165)
        kwargs_38584 = {}
        # Getting the type of 'file_list' (line 165)
        file_list_38581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'file_list', False)
        # Obtaining the member 'exclude_pattern' of a type (line 165)
        exclude_pattern_38582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), file_list_38581, 'exclude_pattern')
        # Calling exclude_pattern(args, kwargs) (line 165)
        exclude_pattern_call_result_38585 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), exclude_pattern_38582, *[str_38583], **kwargs_38584)
        
        
        # Call to assertEqual(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'file_list' (line 166)
        file_list_38588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 25), 'file_list', False)
        # Obtaining the member 'files' of a type (line 166)
        files_38589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 25), file_list_38588, 'files')
        
        # Obtaining an instance of the builtin type 'list' (line 166)
        list_38590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 166)
        # Adding element type (line 166)
        str_38591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 43), 'str', 'a.txt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 42), list_38590, str_38591)
        
        # Processing the call keyword arguments (line 166)
        kwargs_38592 = {}
        # Getting the type of 'self' (line 166)
        self_38586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 166)
        assertEqual_38587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), self_38586, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 166)
        assertEqual_call_result_38593 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), assertEqual_38587, *[files_38589, list_38590], **kwargs_38592)
        
        
        # ################# End of 'test_exclude_pattern(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_exclude_pattern' in the type store
        # Getting the type of 'stypy_return_type' (line 152)
        stypy_return_type_38594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_38594)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_exclude_pattern'
        return stypy_return_type_38594


    @norecursion
    def test_include_pattern(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_include_pattern'
        module_type_store = module_type_store.open_function_context('test_include_pattern', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileListTestCase.test_include_pattern.__dict__.__setitem__('stypy_localization', localization)
        FileListTestCase.test_include_pattern.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileListTestCase.test_include_pattern.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileListTestCase.test_include_pattern.__dict__.__setitem__('stypy_function_name', 'FileListTestCase.test_include_pattern')
        FileListTestCase.test_include_pattern.__dict__.__setitem__('stypy_param_names_list', [])
        FileListTestCase.test_include_pattern.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileListTestCase.test_include_pattern.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileListTestCase.test_include_pattern.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileListTestCase.test_include_pattern.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileListTestCase.test_include_pattern.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileListTestCase.test_include_pattern.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileListTestCase.test_include_pattern', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_include_pattern', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_include_pattern(...)' code ##################

        
        # Assigning a Call to a Name (line 170):
        
        # Call to FileList(...): (line 170)
        # Processing the call keyword arguments (line 170)
        kwargs_38596 = {}
        # Getting the type of 'FileList' (line 170)
        FileList_38595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 20), 'FileList', False)
        # Calling FileList(args, kwargs) (line 170)
        FileList_call_result_38597 = invoke(stypy.reporting.localization.Localization(__file__, 170, 20), FileList_38595, *[], **kwargs_38596)
        
        # Assigning a type to the variable 'file_list' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'file_list', FileList_call_result_38597)
        
        # Call to set_allfiles(...): (line 171)
        # Processing the call arguments (line 171)
        
        # Obtaining an instance of the builtin type 'list' (line 171)
        list_38600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 171)
        
        # Processing the call keyword arguments (line 171)
        kwargs_38601 = {}
        # Getting the type of 'file_list' (line 171)
        file_list_38598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'file_list', False)
        # Obtaining the member 'set_allfiles' of a type (line 171)
        set_allfiles_38599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), file_list_38598, 'set_allfiles')
        # Calling set_allfiles(args, kwargs) (line 171)
        set_allfiles_call_result_38602 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), set_allfiles_38599, *[list_38600], **kwargs_38601)
        
        
        # Call to assertFalse(...): (line 172)
        # Processing the call arguments (line 172)
        
        # Call to include_pattern(...): (line 172)
        # Processing the call arguments (line 172)
        str_38607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 51), 'str', '*.py')
        # Processing the call keyword arguments (line 172)
        kwargs_38608 = {}
        # Getting the type of 'file_list' (line 172)
        file_list_38605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 25), 'file_list', False)
        # Obtaining the member 'include_pattern' of a type (line 172)
        include_pattern_38606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 25), file_list_38605, 'include_pattern')
        # Calling include_pattern(args, kwargs) (line 172)
        include_pattern_call_result_38609 = invoke(stypy.reporting.localization.Localization(__file__, 172, 25), include_pattern_38606, *[str_38607], **kwargs_38608)
        
        # Processing the call keyword arguments (line 172)
        kwargs_38610 = {}
        # Getting the type of 'self' (line 172)
        self_38603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 172)
        assertFalse_38604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), self_38603, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 172)
        assertFalse_call_result_38611 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), assertFalse_38604, *[include_pattern_call_result_38609], **kwargs_38610)
        
        
        # Assigning a Call to a Name (line 175):
        
        # Call to FileList(...): (line 175)
        # Processing the call keyword arguments (line 175)
        kwargs_38613 = {}
        # Getting the type of 'FileList' (line 175)
        FileList_38612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 20), 'FileList', False)
        # Calling FileList(args, kwargs) (line 175)
        FileList_call_result_38614 = invoke(stypy.reporting.localization.Localization(__file__, 175, 20), FileList_38612, *[], **kwargs_38613)
        
        # Assigning a type to the variable 'file_list' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'file_list', FileList_call_result_38614)
        
        # Call to set_allfiles(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_38617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        # Adding element type (line 176)
        str_38618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 32), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 31), list_38617, str_38618)
        # Adding element type (line 176)
        str_38619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 40), 'str', 'b.txt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 31), list_38617, str_38619)
        
        # Processing the call keyword arguments (line 176)
        kwargs_38620 = {}
        # Getting the type of 'file_list' (line 176)
        file_list_38615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'file_list', False)
        # Obtaining the member 'set_allfiles' of a type (line 176)
        set_allfiles_38616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), file_list_38615, 'set_allfiles')
        # Calling set_allfiles(args, kwargs) (line 176)
        set_allfiles_call_result_38621 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), set_allfiles_38616, *[list_38617], **kwargs_38620)
        
        
        # Call to assertTrue(...): (line 177)
        # Processing the call arguments (line 177)
        
        # Call to include_pattern(...): (line 177)
        # Processing the call arguments (line 177)
        str_38626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 50), 'str', '*.py')
        # Processing the call keyword arguments (line 177)
        kwargs_38627 = {}
        # Getting the type of 'file_list' (line 177)
        file_list_38624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 24), 'file_list', False)
        # Obtaining the member 'include_pattern' of a type (line 177)
        include_pattern_38625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 24), file_list_38624, 'include_pattern')
        # Calling include_pattern(args, kwargs) (line 177)
        include_pattern_call_result_38628 = invoke(stypy.reporting.localization.Localization(__file__, 177, 24), include_pattern_38625, *[str_38626], **kwargs_38627)
        
        # Processing the call keyword arguments (line 177)
        kwargs_38629 = {}
        # Getting the type of 'self' (line 177)
        self_38622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 177)
        assertTrue_38623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), self_38622, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 177)
        assertTrue_call_result_38630 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), assertTrue_38623, *[include_pattern_call_result_38628], **kwargs_38629)
        
        
        # Assigning a Call to a Name (line 180):
        
        # Call to FileList(...): (line 180)
        # Processing the call keyword arguments (line 180)
        kwargs_38632 = {}
        # Getting the type of 'FileList' (line 180)
        FileList_38631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 20), 'FileList', False)
        # Calling FileList(args, kwargs) (line 180)
        FileList_call_result_38633 = invoke(stypy.reporting.localization.Localization(__file__, 180, 20), FileList_38631, *[], **kwargs_38632)
        
        # Assigning a type to the variable 'file_list' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'file_list', FileList_call_result_38633)
        
        # Call to assertIsNone(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'file_list' (line 181)
        file_list_38636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 26), 'file_list', False)
        # Obtaining the member 'allfiles' of a type (line 181)
        allfiles_38637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 26), file_list_38636, 'allfiles')
        # Processing the call keyword arguments (line 181)
        kwargs_38638 = {}
        # Getting the type of 'self' (line 181)
        self_38634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'self', False)
        # Obtaining the member 'assertIsNone' of a type (line 181)
        assertIsNone_38635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), self_38634, 'assertIsNone')
        # Calling assertIsNone(args, kwargs) (line 181)
        assertIsNone_call_result_38639 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), assertIsNone_38635, *[allfiles_38637], **kwargs_38638)
        
        
        # Call to set_allfiles(...): (line 182)
        # Processing the call arguments (line 182)
        
        # Obtaining an instance of the builtin type 'list' (line 182)
        list_38642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 182)
        # Adding element type (line 182)
        str_38643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 32), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 31), list_38642, str_38643)
        # Adding element type (line 182)
        str_38644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 40), 'str', 'b.txt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 31), list_38642, str_38644)
        
        # Processing the call keyword arguments (line 182)
        kwargs_38645 = {}
        # Getting the type of 'file_list' (line 182)
        file_list_38640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'file_list', False)
        # Obtaining the member 'set_allfiles' of a type (line 182)
        set_allfiles_38641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), file_list_38640, 'set_allfiles')
        # Calling set_allfiles(args, kwargs) (line 182)
        set_allfiles_call_result_38646 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), set_allfiles_38641, *[list_38642], **kwargs_38645)
        
        
        # Call to include_pattern(...): (line 183)
        # Processing the call arguments (line 183)
        str_38649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 34), 'str', '*')
        # Processing the call keyword arguments (line 183)
        kwargs_38650 = {}
        # Getting the type of 'file_list' (line 183)
        file_list_38647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'file_list', False)
        # Obtaining the member 'include_pattern' of a type (line 183)
        include_pattern_38648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), file_list_38647, 'include_pattern')
        # Calling include_pattern(args, kwargs) (line 183)
        include_pattern_call_result_38651 = invoke(stypy.reporting.localization.Localization(__file__, 183, 8), include_pattern_38648, *[str_38649], **kwargs_38650)
        
        
        # Call to assertEqual(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'file_list' (line 184)
        file_list_38654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 25), 'file_list', False)
        # Obtaining the member 'allfiles' of a type (line 184)
        allfiles_38655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 25), file_list_38654, 'allfiles')
        
        # Obtaining an instance of the builtin type 'list' (line 184)
        list_38656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 184)
        # Adding element type (line 184)
        str_38657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 46), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 45), list_38656, str_38657)
        # Adding element type (line 184)
        str_38658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 54), 'str', 'b.txt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 45), list_38656, str_38658)
        
        # Processing the call keyword arguments (line 184)
        kwargs_38659 = {}
        # Getting the type of 'self' (line 184)
        self_38652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 184)
        assertEqual_38653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), self_38652, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 184)
        assertEqual_call_result_38660 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), assertEqual_38653, *[allfiles_38655, list_38656], **kwargs_38659)
        
        
        # ################# End of 'test_include_pattern(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_include_pattern' in the type store
        # Getting the type of 'stypy_return_type' (line 168)
        stypy_return_type_38661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_38661)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_include_pattern'
        return stypy_return_type_38661


    @norecursion
    def test_process_template(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_process_template'
        module_type_store = module_type_store.open_function_context('test_process_template', 186, 4, False)
        # Assigning a type to the variable 'self' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileListTestCase.test_process_template.__dict__.__setitem__('stypy_localization', localization)
        FileListTestCase.test_process_template.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileListTestCase.test_process_template.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileListTestCase.test_process_template.__dict__.__setitem__('stypy_function_name', 'FileListTestCase.test_process_template')
        FileListTestCase.test_process_template.__dict__.__setitem__('stypy_param_names_list', [])
        FileListTestCase.test_process_template.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileListTestCase.test_process_template.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileListTestCase.test_process_template.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileListTestCase.test_process_template.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileListTestCase.test_process_template.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileListTestCase.test_process_template.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileListTestCase.test_process_template', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_process_template', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_process_template(...)' code ##################

        
        # Assigning a Name to a Name (line 187):
        # Getting the type of 'make_local_path' (line 187)
        make_local_path_38662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'make_local_path')
        # Assigning a type to the variable 'l' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'l', make_local_path_38662)
        
        # Assigning a Call to a Name (line 189):
        
        # Call to FileList(...): (line 189)
        # Processing the call keyword arguments (line 189)
        kwargs_38664 = {}
        # Getting the type of 'FileList' (line 189)
        FileList_38663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 20), 'FileList', False)
        # Calling FileList(args, kwargs) (line 189)
        FileList_call_result_38665 = invoke(stypy.reporting.localization.Localization(__file__, 189, 20), FileList_38663, *[], **kwargs_38664)
        
        # Assigning a type to the variable 'file_list' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'file_list', FileList_call_result_38665)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 190)
        tuple_38666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 190)
        # Adding element type (line 190)
        str_38667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 23), 'str', 'include')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 23), tuple_38666, str_38667)
        # Adding element type (line 190)
        str_38668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 34), 'str', 'exclude')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 23), tuple_38666, str_38668)
        # Adding element type (line 190)
        str_38669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 45), 'str', 'global-include')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 23), tuple_38666, str_38669)
        # Adding element type (line 190)
        str_38670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 23), 'str', 'global-exclude')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 23), tuple_38666, str_38670)
        # Adding element type (line 190)
        str_38671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 41), 'str', 'recursive-include')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 23), tuple_38666, str_38671)
        # Adding element type (line 190)
        str_38672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 23), 'str', 'recursive-exclude')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 23), tuple_38666, str_38672)
        # Adding element type (line 190)
        str_38673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 44), 'str', 'graft')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 23), tuple_38666, str_38673)
        # Adding element type (line 190)
        str_38674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 53), 'str', 'prune')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 23), tuple_38666, str_38674)
        # Adding element type (line 190)
        str_38675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 62), 'str', 'blarg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 23), tuple_38666, str_38675)
        
        # Testing the type of a for loop iterable (line 190)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 190, 8), tuple_38666)
        # Getting the type of the for loop variable (line 190)
        for_loop_var_38676 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 190, 8), tuple_38666)
        # Assigning a type to the variable 'action' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'action', for_loop_var_38676)
        # SSA begins for a for statement (line 190)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assertRaises(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'DistutilsTemplateError' (line 193)
        DistutilsTemplateError_38679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 30), 'DistutilsTemplateError', False)
        # Getting the type of 'file_list' (line 194)
        file_list_38680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 30), 'file_list', False)
        # Obtaining the member 'process_template_line' of a type (line 194)
        process_template_line_38681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 30), file_list_38680, 'process_template_line')
        # Getting the type of 'action' (line 194)
        action_38682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 63), 'action', False)
        # Processing the call keyword arguments (line 193)
        kwargs_38683 = {}
        # Getting the type of 'self' (line 193)
        self_38677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 193)
        assertRaises_38678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), self_38677, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 193)
        assertRaises_call_result_38684 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), assertRaises_38678, *[DistutilsTemplateError_38679, process_template_line_38681, action_38682], **kwargs_38683)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 197):
        
        # Call to FileList(...): (line 197)
        # Processing the call keyword arguments (line 197)
        kwargs_38686 = {}
        # Getting the type of 'FileList' (line 197)
        FileList_38685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 20), 'FileList', False)
        # Calling FileList(args, kwargs) (line 197)
        FileList_call_result_38687 = invoke(stypy.reporting.localization.Localization(__file__, 197, 20), FileList_38685, *[], **kwargs_38686)
        
        # Assigning a type to the variable 'file_list' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'file_list', FileList_call_result_38687)
        
        # Call to set_allfiles(...): (line 198)
        # Processing the call arguments (line 198)
        
        # Obtaining an instance of the builtin type 'list' (line 198)
        list_38690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 198)
        # Adding element type (line 198)
        str_38691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 32), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 31), list_38690, str_38691)
        # Adding element type (line 198)
        str_38692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 40), 'str', 'b.txt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 31), list_38690, str_38692)
        # Adding element type (line 198)
        
        # Call to l(...): (line 198)
        # Processing the call arguments (line 198)
        str_38694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 51), 'str', 'd/c.py')
        # Processing the call keyword arguments (line 198)
        kwargs_38695 = {}
        # Getting the type of 'l' (line 198)
        l_38693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 49), 'l', False)
        # Calling l(args, kwargs) (line 198)
        l_call_result_38696 = invoke(stypy.reporting.localization.Localization(__file__, 198, 49), l_38693, *[str_38694], **kwargs_38695)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 31), list_38690, l_call_result_38696)
        
        # Processing the call keyword arguments (line 198)
        kwargs_38697 = {}
        # Getting the type of 'file_list' (line 198)
        file_list_38688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'file_list', False)
        # Obtaining the member 'set_allfiles' of a type (line 198)
        set_allfiles_38689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), file_list_38688, 'set_allfiles')
        # Calling set_allfiles(args, kwargs) (line 198)
        set_allfiles_call_result_38698 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), set_allfiles_38689, *[list_38690], **kwargs_38697)
        
        
        # Call to process_template_line(...): (line 200)
        # Processing the call arguments (line 200)
        str_38701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 40), 'str', 'include *.py')
        # Processing the call keyword arguments (line 200)
        kwargs_38702 = {}
        # Getting the type of 'file_list' (line 200)
        file_list_38699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'file_list', False)
        # Obtaining the member 'process_template_line' of a type (line 200)
        process_template_line_38700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), file_list_38699, 'process_template_line')
        # Calling process_template_line(args, kwargs) (line 200)
        process_template_line_call_result_38703 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), process_template_line_38700, *[str_38701], **kwargs_38702)
        
        
        # Call to assertEqual(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'file_list' (line 201)
        file_list_38706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 25), 'file_list', False)
        # Obtaining the member 'files' of a type (line 201)
        files_38707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 25), file_list_38706, 'files')
        
        # Obtaining an instance of the builtin type 'list' (line 201)
        list_38708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 201)
        # Adding element type (line 201)
        str_38709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 43), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 42), list_38708, str_38709)
        
        # Processing the call keyword arguments (line 201)
        kwargs_38710 = {}
        # Getting the type of 'self' (line 201)
        self_38704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 201)
        assertEqual_38705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), self_38704, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 201)
        assertEqual_call_result_38711 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), assertEqual_38705, *[files_38707, list_38708], **kwargs_38710)
        
        
        # Call to assertNoWarnings(...): (line 202)
        # Processing the call keyword arguments (line 202)
        kwargs_38714 = {}
        # Getting the type of 'self' (line 202)
        self_38712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'self', False)
        # Obtaining the member 'assertNoWarnings' of a type (line 202)
        assertNoWarnings_38713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), self_38712, 'assertNoWarnings')
        # Calling assertNoWarnings(args, kwargs) (line 202)
        assertNoWarnings_call_result_38715 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), assertNoWarnings_38713, *[], **kwargs_38714)
        
        
        # Call to process_template_line(...): (line 204)
        # Processing the call arguments (line 204)
        str_38718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 40), 'str', 'include *.rb')
        # Processing the call keyword arguments (line 204)
        kwargs_38719 = {}
        # Getting the type of 'file_list' (line 204)
        file_list_38716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'file_list', False)
        # Obtaining the member 'process_template_line' of a type (line 204)
        process_template_line_38717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), file_list_38716, 'process_template_line')
        # Calling process_template_line(args, kwargs) (line 204)
        process_template_line_call_result_38720 = invoke(stypy.reporting.localization.Localization(__file__, 204, 8), process_template_line_38717, *[str_38718], **kwargs_38719)
        
        
        # Call to assertEqual(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'file_list' (line 205)
        file_list_38723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 25), 'file_list', False)
        # Obtaining the member 'files' of a type (line 205)
        files_38724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 25), file_list_38723, 'files')
        
        # Obtaining an instance of the builtin type 'list' (line 205)
        list_38725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 205)
        # Adding element type (line 205)
        str_38726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 43), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 42), list_38725, str_38726)
        
        # Processing the call keyword arguments (line 205)
        kwargs_38727 = {}
        # Getting the type of 'self' (line 205)
        self_38721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 205)
        assertEqual_38722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), self_38721, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 205)
        assertEqual_call_result_38728 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), assertEqual_38722, *[files_38724, list_38725], **kwargs_38727)
        
        
        # Call to assertWarnings(...): (line 206)
        # Processing the call keyword arguments (line 206)
        kwargs_38731 = {}
        # Getting the type of 'self' (line 206)
        self_38729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'self', False)
        # Obtaining the member 'assertWarnings' of a type (line 206)
        assertWarnings_38730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), self_38729, 'assertWarnings')
        # Calling assertWarnings(args, kwargs) (line 206)
        assertWarnings_call_result_38732 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), assertWarnings_38730, *[], **kwargs_38731)
        
        
        # Assigning a Call to a Name (line 209):
        
        # Call to FileList(...): (line 209)
        # Processing the call keyword arguments (line 209)
        kwargs_38734 = {}
        # Getting the type of 'FileList' (line 209)
        FileList_38733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 20), 'FileList', False)
        # Calling FileList(args, kwargs) (line 209)
        FileList_call_result_38735 = invoke(stypy.reporting.localization.Localization(__file__, 209, 20), FileList_38733, *[], **kwargs_38734)
        
        # Assigning a type to the variable 'file_list' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'file_list', FileList_call_result_38735)
        
        # Assigning a List to a Attribute (line 210):
        
        # Obtaining an instance of the builtin type 'list' (line 210)
        list_38736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 210)
        # Adding element type (line 210)
        str_38737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 27), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 26), list_38736, str_38737)
        # Adding element type (line 210)
        str_38738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 35), 'str', 'b.txt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 26), list_38736, str_38738)
        # Adding element type (line 210)
        
        # Call to l(...): (line 210)
        # Processing the call arguments (line 210)
        str_38740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 46), 'str', 'd/c.py')
        # Processing the call keyword arguments (line 210)
        kwargs_38741 = {}
        # Getting the type of 'l' (line 210)
        l_38739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 44), 'l', False)
        # Calling l(args, kwargs) (line 210)
        l_call_result_38742 = invoke(stypy.reporting.localization.Localization(__file__, 210, 44), l_38739, *[str_38740], **kwargs_38741)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 26), list_38736, l_call_result_38742)
        
        # Getting the type of 'file_list' (line 210)
        file_list_38743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'file_list')
        # Setting the type of the member 'files' of a type (line 210)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), file_list_38743, 'files', list_38736)
        
        # Call to process_template_line(...): (line 212)
        # Processing the call arguments (line 212)
        str_38746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 40), 'str', 'exclude *.py')
        # Processing the call keyword arguments (line 212)
        kwargs_38747 = {}
        # Getting the type of 'file_list' (line 212)
        file_list_38744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'file_list', False)
        # Obtaining the member 'process_template_line' of a type (line 212)
        process_template_line_38745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), file_list_38744, 'process_template_line')
        # Calling process_template_line(args, kwargs) (line 212)
        process_template_line_call_result_38748 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), process_template_line_38745, *[str_38746], **kwargs_38747)
        
        
        # Call to assertEqual(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'file_list' (line 213)
        file_list_38751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 25), 'file_list', False)
        # Obtaining the member 'files' of a type (line 213)
        files_38752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 25), file_list_38751, 'files')
        
        # Obtaining an instance of the builtin type 'list' (line 213)
        list_38753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 213)
        # Adding element type (line 213)
        str_38754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 43), 'str', 'b.txt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 42), list_38753, str_38754)
        # Adding element type (line 213)
        
        # Call to l(...): (line 213)
        # Processing the call arguments (line 213)
        str_38756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 54), 'str', 'd/c.py')
        # Processing the call keyword arguments (line 213)
        kwargs_38757 = {}
        # Getting the type of 'l' (line 213)
        l_38755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 52), 'l', False)
        # Calling l(args, kwargs) (line 213)
        l_call_result_38758 = invoke(stypy.reporting.localization.Localization(__file__, 213, 52), l_38755, *[str_38756], **kwargs_38757)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 42), list_38753, l_call_result_38758)
        
        # Processing the call keyword arguments (line 213)
        kwargs_38759 = {}
        # Getting the type of 'self' (line 213)
        self_38749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 213)
        assertEqual_38750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 8), self_38749, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 213)
        assertEqual_call_result_38760 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), assertEqual_38750, *[files_38752, list_38753], **kwargs_38759)
        
        
        # Call to assertNoWarnings(...): (line 214)
        # Processing the call keyword arguments (line 214)
        kwargs_38763 = {}
        # Getting the type of 'self' (line 214)
        self_38761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'self', False)
        # Obtaining the member 'assertNoWarnings' of a type (line 214)
        assertNoWarnings_38762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), self_38761, 'assertNoWarnings')
        # Calling assertNoWarnings(args, kwargs) (line 214)
        assertNoWarnings_call_result_38764 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), assertNoWarnings_38762, *[], **kwargs_38763)
        
        
        # Call to process_template_line(...): (line 216)
        # Processing the call arguments (line 216)
        str_38767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 40), 'str', 'exclude *.rb')
        # Processing the call keyword arguments (line 216)
        kwargs_38768 = {}
        # Getting the type of 'file_list' (line 216)
        file_list_38765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'file_list', False)
        # Obtaining the member 'process_template_line' of a type (line 216)
        process_template_line_38766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), file_list_38765, 'process_template_line')
        # Calling process_template_line(args, kwargs) (line 216)
        process_template_line_call_result_38769 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), process_template_line_38766, *[str_38767], **kwargs_38768)
        
        
        # Call to assertEqual(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 'file_list' (line 217)
        file_list_38772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 25), 'file_list', False)
        # Obtaining the member 'files' of a type (line 217)
        files_38773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 25), file_list_38772, 'files')
        
        # Obtaining an instance of the builtin type 'list' (line 217)
        list_38774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 217)
        # Adding element type (line 217)
        str_38775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 43), 'str', 'b.txt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 42), list_38774, str_38775)
        # Adding element type (line 217)
        
        # Call to l(...): (line 217)
        # Processing the call arguments (line 217)
        str_38777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 54), 'str', 'd/c.py')
        # Processing the call keyword arguments (line 217)
        kwargs_38778 = {}
        # Getting the type of 'l' (line 217)
        l_38776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 52), 'l', False)
        # Calling l(args, kwargs) (line 217)
        l_call_result_38779 = invoke(stypy.reporting.localization.Localization(__file__, 217, 52), l_38776, *[str_38777], **kwargs_38778)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 42), list_38774, l_call_result_38779)
        
        # Processing the call keyword arguments (line 217)
        kwargs_38780 = {}
        # Getting the type of 'self' (line 217)
        self_38770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 217)
        assertEqual_38771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 8), self_38770, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 217)
        assertEqual_call_result_38781 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), assertEqual_38771, *[files_38773, list_38774], **kwargs_38780)
        
        
        # Call to assertWarnings(...): (line 218)
        # Processing the call keyword arguments (line 218)
        kwargs_38784 = {}
        # Getting the type of 'self' (line 218)
        self_38782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'self', False)
        # Obtaining the member 'assertWarnings' of a type (line 218)
        assertWarnings_38783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), self_38782, 'assertWarnings')
        # Calling assertWarnings(args, kwargs) (line 218)
        assertWarnings_call_result_38785 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), assertWarnings_38783, *[], **kwargs_38784)
        
        
        # Assigning a Call to a Name (line 221):
        
        # Call to FileList(...): (line 221)
        # Processing the call keyword arguments (line 221)
        kwargs_38787 = {}
        # Getting the type of 'FileList' (line 221)
        FileList_38786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'FileList', False)
        # Calling FileList(args, kwargs) (line 221)
        FileList_call_result_38788 = invoke(stypy.reporting.localization.Localization(__file__, 221, 20), FileList_38786, *[], **kwargs_38787)
        
        # Assigning a type to the variable 'file_list' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'file_list', FileList_call_result_38788)
        
        # Call to set_allfiles(...): (line 222)
        # Processing the call arguments (line 222)
        
        # Obtaining an instance of the builtin type 'list' (line 222)
        list_38791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 222)
        # Adding element type (line 222)
        str_38792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 32), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 31), list_38791, str_38792)
        # Adding element type (line 222)
        str_38793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 40), 'str', 'b.txt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 31), list_38791, str_38793)
        # Adding element type (line 222)
        
        # Call to l(...): (line 222)
        # Processing the call arguments (line 222)
        str_38795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 51), 'str', 'd/c.py')
        # Processing the call keyword arguments (line 222)
        kwargs_38796 = {}
        # Getting the type of 'l' (line 222)
        l_38794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 49), 'l', False)
        # Calling l(args, kwargs) (line 222)
        l_call_result_38797 = invoke(stypy.reporting.localization.Localization(__file__, 222, 49), l_38794, *[str_38795], **kwargs_38796)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 31), list_38791, l_call_result_38797)
        
        # Processing the call keyword arguments (line 222)
        kwargs_38798 = {}
        # Getting the type of 'file_list' (line 222)
        file_list_38789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'file_list', False)
        # Obtaining the member 'set_allfiles' of a type (line 222)
        set_allfiles_38790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), file_list_38789, 'set_allfiles')
        # Calling set_allfiles(args, kwargs) (line 222)
        set_allfiles_call_result_38799 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), set_allfiles_38790, *[list_38791], **kwargs_38798)
        
        
        # Call to process_template_line(...): (line 224)
        # Processing the call arguments (line 224)
        str_38802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 40), 'str', 'global-include *.py')
        # Processing the call keyword arguments (line 224)
        kwargs_38803 = {}
        # Getting the type of 'file_list' (line 224)
        file_list_38800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'file_list', False)
        # Obtaining the member 'process_template_line' of a type (line 224)
        process_template_line_38801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), file_list_38800, 'process_template_line')
        # Calling process_template_line(args, kwargs) (line 224)
        process_template_line_call_result_38804 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), process_template_line_38801, *[str_38802], **kwargs_38803)
        
        
        # Call to assertEqual(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'file_list' (line 225)
        file_list_38807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 25), 'file_list', False)
        # Obtaining the member 'files' of a type (line 225)
        files_38808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 25), file_list_38807, 'files')
        
        # Obtaining an instance of the builtin type 'list' (line 225)
        list_38809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 225)
        # Adding element type (line 225)
        str_38810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 43), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 42), list_38809, str_38810)
        # Adding element type (line 225)
        
        # Call to l(...): (line 225)
        # Processing the call arguments (line 225)
        str_38812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 53), 'str', 'd/c.py')
        # Processing the call keyword arguments (line 225)
        kwargs_38813 = {}
        # Getting the type of 'l' (line 225)
        l_38811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 51), 'l', False)
        # Calling l(args, kwargs) (line 225)
        l_call_result_38814 = invoke(stypy.reporting.localization.Localization(__file__, 225, 51), l_38811, *[str_38812], **kwargs_38813)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 42), list_38809, l_call_result_38814)
        
        # Processing the call keyword arguments (line 225)
        kwargs_38815 = {}
        # Getting the type of 'self' (line 225)
        self_38805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 225)
        assertEqual_38806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), self_38805, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 225)
        assertEqual_call_result_38816 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), assertEqual_38806, *[files_38808, list_38809], **kwargs_38815)
        
        
        # Call to assertNoWarnings(...): (line 226)
        # Processing the call keyword arguments (line 226)
        kwargs_38819 = {}
        # Getting the type of 'self' (line 226)
        self_38817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'self', False)
        # Obtaining the member 'assertNoWarnings' of a type (line 226)
        assertNoWarnings_38818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), self_38817, 'assertNoWarnings')
        # Calling assertNoWarnings(args, kwargs) (line 226)
        assertNoWarnings_call_result_38820 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), assertNoWarnings_38818, *[], **kwargs_38819)
        
        
        # Call to process_template_line(...): (line 228)
        # Processing the call arguments (line 228)
        str_38823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 40), 'str', 'global-include *.rb')
        # Processing the call keyword arguments (line 228)
        kwargs_38824 = {}
        # Getting the type of 'file_list' (line 228)
        file_list_38821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'file_list', False)
        # Obtaining the member 'process_template_line' of a type (line 228)
        process_template_line_38822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), file_list_38821, 'process_template_line')
        # Calling process_template_line(args, kwargs) (line 228)
        process_template_line_call_result_38825 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), process_template_line_38822, *[str_38823], **kwargs_38824)
        
        
        # Call to assertEqual(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'file_list' (line 229)
        file_list_38828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 25), 'file_list', False)
        # Obtaining the member 'files' of a type (line 229)
        files_38829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 25), file_list_38828, 'files')
        
        # Obtaining an instance of the builtin type 'list' (line 229)
        list_38830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 229)
        # Adding element type (line 229)
        str_38831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 43), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 42), list_38830, str_38831)
        # Adding element type (line 229)
        
        # Call to l(...): (line 229)
        # Processing the call arguments (line 229)
        str_38833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 53), 'str', 'd/c.py')
        # Processing the call keyword arguments (line 229)
        kwargs_38834 = {}
        # Getting the type of 'l' (line 229)
        l_38832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 51), 'l', False)
        # Calling l(args, kwargs) (line 229)
        l_call_result_38835 = invoke(stypy.reporting.localization.Localization(__file__, 229, 51), l_38832, *[str_38833], **kwargs_38834)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 42), list_38830, l_call_result_38835)
        
        # Processing the call keyword arguments (line 229)
        kwargs_38836 = {}
        # Getting the type of 'self' (line 229)
        self_38826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 229)
        assertEqual_38827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), self_38826, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 229)
        assertEqual_call_result_38837 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), assertEqual_38827, *[files_38829, list_38830], **kwargs_38836)
        
        
        # Call to assertWarnings(...): (line 230)
        # Processing the call keyword arguments (line 230)
        kwargs_38840 = {}
        # Getting the type of 'self' (line 230)
        self_38838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'self', False)
        # Obtaining the member 'assertWarnings' of a type (line 230)
        assertWarnings_38839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), self_38838, 'assertWarnings')
        # Calling assertWarnings(args, kwargs) (line 230)
        assertWarnings_call_result_38841 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), assertWarnings_38839, *[], **kwargs_38840)
        
        
        # Assigning a Call to a Name (line 233):
        
        # Call to FileList(...): (line 233)
        # Processing the call keyword arguments (line 233)
        kwargs_38843 = {}
        # Getting the type of 'FileList' (line 233)
        FileList_38842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 20), 'FileList', False)
        # Calling FileList(args, kwargs) (line 233)
        FileList_call_result_38844 = invoke(stypy.reporting.localization.Localization(__file__, 233, 20), FileList_38842, *[], **kwargs_38843)
        
        # Assigning a type to the variable 'file_list' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'file_list', FileList_call_result_38844)
        
        # Assigning a List to a Attribute (line 234):
        
        # Obtaining an instance of the builtin type 'list' (line 234)
        list_38845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 234)
        # Adding element type (line 234)
        str_38846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 27), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 26), list_38845, str_38846)
        # Adding element type (line 234)
        str_38847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 35), 'str', 'b.txt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 26), list_38845, str_38847)
        # Adding element type (line 234)
        
        # Call to l(...): (line 234)
        # Processing the call arguments (line 234)
        str_38849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 46), 'str', 'd/c.py')
        # Processing the call keyword arguments (line 234)
        kwargs_38850 = {}
        # Getting the type of 'l' (line 234)
        l_38848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 44), 'l', False)
        # Calling l(args, kwargs) (line 234)
        l_call_result_38851 = invoke(stypy.reporting.localization.Localization(__file__, 234, 44), l_38848, *[str_38849], **kwargs_38850)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 26), list_38845, l_call_result_38851)
        
        # Getting the type of 'file_list' (line 234)
        file_list_38852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'file_list')
        # Setting the type of the member 'files' of a type (line 234)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), file_list_38852, 'files', list_38845)
        
        # Call to process_template_line(...): (line 236)
        # Processing the call arguments (line 236)
        str_38855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 40), 'str', 'global-exclude *.py')
        # Processing the call keyword arguments (line 236)
        kwargs_38856 = {}
        # Getting the type of 'file_list' (line 236)
        file_list_38853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'file_list', False)
        # Obtaining the member 'process_template_line' of a type (line 236)
        process_template_line_38854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), file_list_38853, 'process_template_line')
        # Calling process_template_line(args, kwargs) (line 236)
        process_template_line_call_result_38857 = invoke(stypy.reporting.localization.Localization(__file__, 236, 8), process_template_line_38854, *[str_38855], **kwargs_38856)
        
        
        # Call to assertEqual(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'file_list' (line 237)
        file_list_38860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 25), 'file_list', False)
        # Obtaining the member 'files' of a type (line 237)
        files_38861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 25), file_list_38860, 'files')
        
        # Obtaining an instance of the builtin type 'list' (line 237)
        list_38862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 237)
        # Adding element type (line 237)
        str_38863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 43), 'str', 'b.txt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 42), list_38862, str_38863)
        
        # Processing the call keyword arguments (line 237)
        kwargs_38864 = {}
        # Getting the type of 'self' (line 237)
        self_38858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 237)
        assertEqual_38859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), self_38858, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 237)
        assertEqual_call_result_38865 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), assertEqual_38859, *[files_38861, list_38862], **kwargs_38864)
        
        
        # Call to assertNoWarnings(...): (line 238)
        # Processing the call keyword arguments (line 238)
        kwargs_38868 = {}
        # Getting the type of 'self' (line 238)
        self_38866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'self', False)
        # Obtaining the member 'assertNoWarnings' of a type (line 238)
        assertNoWarnings_38867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), self_38866, 'assertNoWarnings')
        # Calling assertNoWarnings(args, kwargs) (line 238)
        assertNoWarnings_call_result_38869 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), assertNoWarnings_38867, *[], **kwargs_38868)
        
        
        # Call to process_template_line(...): (line 240)
        # Processing the call arguments (line 240)
        str_38872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 40), 'str', 'global-exclude *.rb')
        # Processing the call keyword arguments (line 240)
        kwargs_38873 = {}
        # Getting the type of 'file_list' (line 240)
        file_list_38870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'file_list', False)
        # Obtaining the member 'process_template_line' of a type (line 240)
        process_template_line_38871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), file_list_38870, 'process_template_line')
        # Calling process_template_line(args, kwargs) (line 240)
        process_template_line_call_result_38874 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), process_template_line_38871, *[str_38872], **kwargs_38873)
        
        
        # Call to assertEqual(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'file_list' (line 241)
        file_list_38877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 25), 'file_list', False)
        # Obtaining the member 'files' of a type (line 241)
        files_38878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 25), file_list_38877, 'files')
        
        # Obtaining an instance of the builtin type 'list' (line 241)
        list_38879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 241)
        # Adding element type (line 241)
        str_38880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 43), 'str', 'b.txt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 42), list_38879, str_38880)
        
        # Processing the call keyword arguments (line 241)
        kwargs_38881 = {}
        # Getting the type of 'self' (line 241)
        self_38875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 241)
        assertEqual_38876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 8), self_38875, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 241)
        assertEqual_call_result_38882 = invoke(stypy.reporting.localization.Localization(__file__, 241, 8), assertEqual_38876, *[files_38878, list_38879], **kwargs_38881)
        
        
        # Call to assertWarnings(...): (line 242)
        # Processing the call keyword arguments (line 242)
        kwargs_38885 = {}
        # Getting the type of 'self' (line 242)
        self_38883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'self', False)
        # Obtaining the member 'assertWarnings' of a type (line 242)
        assertWarnings_38884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), self_38883, 'assertWarnings')
        # Calling assertWarnings(args, kwargs) (line 242)
        assertWarnings_call_result_38886 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), assertWarnings_38884, *[], **kwargs_38885)
        
        
        # Assigning a Call to a Name (line 245):
        
        # Call to FileList(...): (line 245)
        # Processing the call keyword arguments (line 245)
        kwargs_38888 = {}
        # Getting the type of 'FileList' (line 245)
        FileList_38887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 20), 'FileList', False)
        # Calling FileList(args, kwargs) (line 245)
        FileList_call_result_38889 = invoke(stypy.reporting.localization.Localization(__file__, 245, 20), FileList_38887, *[], **kwargs_38888)
        
        # Assigning a type to the variable 'file_list' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'file_list', FileList_call_result_38889)
        
        # Call to set_allfiles(...): (line 246)
        # Processing the call arguments (line 246)
        
        # Obtaining an instance of the builtin type 'list' (line 246)
        list_38892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 246)
        # Adding element type (line 246)
        str_38893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 32), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 31), list_38892, str_38893)
        # Adding element type (line 246)
        
        # Call to l(...): (line 246)
        # Processing the call arguments (line 246)
        str_38895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 42), 'str', 'd/b.py')
        # Processing the call keyword arguments (line 246)
        kwargs_38896 = {}
        # Getting the type of 'l' (line 246)
        l_38894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 40), 'l', False)
        # Calling l(args, kwargs) (line 246)
        l_call_result_38897 = invoke(stypy.reporting.localization.Localization(__file__, 246, 40), l_38894, *[str_38895], **kwargs_38896)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 31), list_38892, l_call_result_38897)
        # Adding element type (line 246)
        
        # Call to l(...): (line 246)
        # Processing the call arguments (line 246)
        str_38899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 55), 'str', 'd/c.txt')
        # Processing the call keyword arguments (line 246)
        kwargs_38900 = {}
        # Getting the type of 'l' (line 246)
        l_38898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 53), 'l', False)
        # Calling l(args, kwargs) (line 246)
        l_call_result_38901 = invoke(stypy.reporting.localization.Localization(__file__, 246, 53), l_38898, *[str_38899], **kwargs_38900)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 31), list_38892, l_call_result_38901)
        # Adding element type (line 246)
        
        # Call to l(...): (line 247)
        # Processing the call arguments (line 247)
        str_38903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 34), 'str', 'd/d/e.py')
        # Processing the call keyword arguments (line 247)
        kwargs_38904 = {}
        # Getting the type of 'l' (line 247)
        l_38902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 32), 'l', False)
        # Calling l(args, kwargs) (line 247)
        l_call_result_38905 = invoke(stypy.reporting.localization.Localization(__file__, 247, 32), l_38902, *[str_38903], **kwargs_38904)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 31), list_38892, l_call_result_38905)
        
        # Processing the call keyword arguments (line 246)
        kwargs_38906 = {}
        # Getting the type of 'file_list' (line 246)
        file_list_38890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'file_list', False)
        # Obtaining the member 'set_allfiles' of a type (line 246)
        set_allfiles_38891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), file_list_38890, 'set_allfiles')
        # Calling set_allfiles(args, kwargs) (line 246)
        set_allfiles_call_result_38907 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), set_allfiles_38891, *[list_38892], **kwargs_38906)
        
        
        # Call to process_template_line(...): (line 249)
        # Processing the call arguments (line 249)
        str_38910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 40), 'str', 'recursive-include d *.py')
        # Processing the call keyword arguments (line 249)
        kwargs_38911 = {}
        # Getting the type of 'file_list' (line 249)
        file_list_38908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'file_list', False)
        # Obtaining the member 'process_template_line' of a type (line 249)
        process_template_line_38909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), file_list_38908, 'process_template_line')
        # Calling process_template_line(args, kwargs) (line 249)
        process_template_line_call_result_38912 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), process_template_line_38909, *[str_38910], **kwargs_38911)
        
        
        # Call to assertEqual(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'file_list' (line 250)
        file_list_38915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 25), 'file_list', False)
        # Obtaining the member 'files' of a type (line 250)
        files_38916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 25), file_list_38915, 'files')
        
        # Obtaining an instance of the builtin type 'list' (line 250)
        list_38917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 250)
        # Adding element type (line 250)
        
        # Call to l(...): (line 250)
        # Processing the call arguments (line 250)
        str_38919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 45), 'str', 'd/b.py')
        # Processing the call keyword arguments (line 250)
        kwargs_38920 = {}
        # Getting the type of 'l' (line 250)
        l_38918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 43), 'l', False)
        # Calling l(args, kwargs) (line 250)
        l_call_result_38921 = invoke(stypy.reporting.localization.Localization(__file__, 250, 43), l_38918, *[str_38919], **kwargs_38920)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 42), list_38917, l_call_result_38921)
        # Adding element type (line 250)
        
        # Call to l(...): (line 250)
        # Processing the call arguments (line 250)
        str_38923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 58), 'str', 'd/d/e.py')
        # Processing the call keyword arguments (line 250)
        kwargs_38924 = {}
        # Getting the type of 'l' (line 250)
        l_38922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 56), 'l', False)
        # Calling l(args, kwargs) (line 250)
        l_call_result_38925 = invoke(stypy.reporting.localization.Localization(__file__, 250, 56), l_38922, *[str_38923], **kwargs_38924)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 42), list_38917, l_call_result_38925)
        
        # Processing the call keyword arguments (line 250)
        kwargs_38926 = {}
        # Getting the type of 'self' (line 250)
        self_38913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 250)
        assertEqual_38914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), self_38913, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 250)
        assertEqual_call_result_38927 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), assertEqual_38914, *[files_38916, list_38917], **kwargs_38926)
        
        
        # Call to assertNoWarnings(...): (line 251)
        # Processing the call keyword arguments (line 251)
        kwargs_38930 = {}
        # Getting the type of 'self' (line 251)
        self_38928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'self', False)
        # Obtaining the member 'assertNoWarnings' of a type (line 251)
        assertNoWarnings_38929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), self_38928, 'assertNoWarnings')
        # Calling assertNoWarnings(args, kwargs) (line 251)
        assertNoWarnings_call_result_38931 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), assertNoWarnings_38929, *[], **kwargs_38930)
        
        
        # Call to process_template_line(...): (line 253)
        # Processing the call arguments (line 253)
        str_38934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 40), 'str', 'recursive-include e *.py')
        # Processing the call keyword arguments (line 253)
        kwargs_38935 = {}
        # Getting the type of 'file_list' (line 253)
        file_list_38932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'file_list', False)
        # Obtaining the member 'process_template_line' of a type (line 253)
        process_template_line_38933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), file_list_38932, 'process_template_line')
        # Calling process_template_line(args, kwargs) (line 253)
        process_template_line_call_result_38936 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), process_template_line_38933, *[str_38934], **kwargs_38935)
        
        
        # Call to assertEqual(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'file_list' (line 254)
        file_list_38939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 25), 'file_list', False)
        # Obtaining the member 'files' of a type (line 254)
        files_38940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 25), file_list_38939, 'files')
        
        # Obtaining an instance of the builtin type 'list' (line 254)
        list_38941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 254)
        # Adding element type (line 254)
        
        # Call to l(...): (line 254)
        # Processing the call arguments (line 254)
        str_38943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 45), 'str', 'd/b.py')
        # Processing the call keyword arguments (line 254)
        kwargs_38944 = {}
        # Getting the type of 'l' (line 254)
        l_38942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 43), 'l', False)
        # Calling l(args, kwargs) (line 254)
        l_call_result_38945 = invoke(stypy.reporting.localization.Localization(__file__, 254, 43), l_38942, *[str_38943], **kwargs_38944)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 42), list_38941, l_call_result_38945)
        # Adding element type (line 254)
        
        # Call to l(...): (line 254)
        # Processing the call arguments (line 254)
        str_38947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 58), 'str', 'd/d/e.py')
        # Processing the call keyword arguments (line 254)
        kwargs_38948 = {}
        # Getting the type of 'l' (line 254)
        l_38946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 56), 'l', False)
        # Calling l(args, kwargs) (line 254)
        l_call_result_38949 = invoke(stypy.reporting.localization.Localization(__file__, 254, 56), l_38946, *[str_38947], **kwargs_38948)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 42), list_38941, l_call_result_38949)
        
        # Processing the call keyword arguments (line 254)
        kwargs_38950 = {}
        # Getting the type of 'self' (line 254)
        self_38937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 254)
        assertEqual_38938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 8), self_38937, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 254)
        assertEqual_call_result_38951 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), assertEqual_38938, *[files_38940, list_38941], **kwargs_38950)
        
        
        # Call to assertWarnings(...): (line 255)
        # Processing the call keyword arguments (line 255)
        kwargs_38954 = {}
        # Getting the type of 'self' (line 255)
        self_38952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'self', False)
        # Obtaining the member 'assertWarnings' of a type (line 255)
        assertWarnings_38953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 8), self_38952, 'assertWarnings')
        # Calling assertWarnings(args, kwargs) (line 255)
        assertWarnings_call_result_38955 = invoke(stypy.reporting.localization.Localization(__file__, 255, 8), assertWarnings_38953, *[], **kwargs_38954)
        
        
        # Assigning a Call to a Name (line 258):
        
        # Call to FileList(...): (line 258)
        # Processing the call keyword arguments (line 258)
        kwargs_38957 = {}
        # Getting the type of 'FileList' (line 258)
        FileList_38956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 20), 'FileList', False)
        # Calling FileList(args, kwargs) (line 258)
        FileList_call_result_38958 = invoke(stypy.reporting.localization.Localization(__file__, 258, 20), FileList_38956, *[], **kwargs_38957)
        
        # Assigning a type to the variable 'file_list' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'file_list', FileList_call_result_38958)
        
        # Assigning a List to a Attribute (line 259):
        
        # Obtaining an instance of the builtin type 'list' (line 259)
        list_38959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 259)
        # Adding element type (line 259)
        str_38960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 27), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 26), list_38959, str_38960)
        # Adding element type (line 259)
        
        # Call to l(...): (line 259)
        # Processing the call arguments (line 259)
        str_38962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 37), 'str', 'd/b.py')
        # Processing the call keyword arguments (line 259)
        kwargs_38963 = {}
        # Getting the type of 'l' (line 259)
        l_38961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 35), 'l', False)
        # Calling l(args, kwargs) (line 259)
        l_call_result_38964 = invoke(stypy.reporting.localization.Localization(__file__, 259, 35), l_38961, *[str_38962], **kwargs_38963)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 26), list_38959, l_call_result_38964)
        # Adding element type (line 259)
        
        # Call to l(...): (line 259)
        # Processing the call arguments (line 259)
        str_38966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 50), 'str', 'd/c.txt')
        # Processing the call keyword arguments (line 259)
        kwargs_38967 = {}
        # Getting the type of 'l' (line 259)
        l_38965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 48), 'l', False)
        # Calling l(args, kwargs) (line 259)
        l_call_result_38968 = invoke(stypy.reporting.localization.Localization(__file__, 259, 48), l_38965, *[str_38966], **kwargs_38967)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 26), list_38959, l_call_result_38968)
        # Adding element type (line 259)
        
        # Call to l(...): (line 259)
        # Processing the call arguments (line 259)
        str_38970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 64), 'str', 'd/d/e.py')
        # Processing the call keyword arguments (line 259)
        kwargs_38971 = {}
        # Getting the type of 'l' (line 259)
        l_38969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 62), 'l', False)
        # Calling l(args, kwargs) (line 259)
        l_call_result_38972 = invoke(stypy.reporting.localization.Localization(__file__, 259, 62), l_38969, *[str_38970], **kwargs_38971)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 26), list_38959, l_call_result_38972)
        
        # Getting the type of 'file_list' (line 259)
        file_list_38973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'file_list')
        # Setting the type of the member 'files' of a type (line 259)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 8), file_list_38973, 'files', list_38959)
        
        # Call to process_template_line(...): (line 261)
        # Processing the call arguments (line 261)
        str_38976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 40), 'str', 'recursive-exclude d *.py')
        # Processing the call keyword arguments (line 261)
        kwargs_38977 = {}
        # Getting the type of 'file_list' (line 261)
        file_list_38974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'file_list', False)
        # Obtaining the member 'process_template_line' of a type (line 261)
        process_template_line_38975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), file_list_38974, 'process_template_line')
        # Calling process_template_line(args, kwargs) (line 261)
        process_template_line_call_result_38978 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), process_template_line_38975, *[str_38976], **kwargs_38977)
        
        
        # Call to assertEqual(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'file_list' (line 262)
        file_list_38981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 25), 'file_list', False)
        # Obtaining the member 'files' of a type (line 262)
        files_38982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 25), file_list_38981, 'files')
        
        # Obtaining an instance of the builtin type 'list' (line 262)
        list_38983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 262)
        # Adding element type (line 262)
        str_38984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 43), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 42), list_38983, str_38984)
        # Adding element type (line 262)
        
        # Call to l(...): (line 262)
        # Processing the call arguments (line 262)
        str_38986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 53), 'str', 'd/c.txt')
        # Processing the call keyword arguments (line 262)
        kwargs_38987 = {}
        # Getting the type of 'l' (line 262)
        l_38985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 51), 'l', False)
        # Calling l(args, kwargs) (line 262)
        l_call_result_38988 = invoke(stypy.reporting.localization.Localization(__file__, 262, 51), l_38985, *[str_38986], **kwargs_38987)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 42), list_38983, l_call_result_38988)
        
        # Processing the call keyword arguments (line 262)
        kwargs_38989 = {}
        # Getting the type of 'self' (line 262)
        self_38979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 262)
        assertEqual_38980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), self_38979, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 262)
        assertEqual_call_result_38990 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), assertEqual_38980, *[files_38982, list_38983], **kwargs_38989)
        
        
        # Call to assertNoWarnings(...): (line 263)
        # Processing the call keyword arguments (line 263)
        kwargs_38993 = {}
        # Getting the type of 'self' (line 263)
        self_38991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'self', False)
        # Obtaining the member 'assertNoWarnings' of a type (line 263)
        assertNoWarnings_38992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), self_38991, 'assertNoWarnings')
        # Calling assertNoWarnings(args, kwargs) (line 263)
        assertNoWarnings_call_result_38994 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), assertNoWarnings_38992, *[], **kwargs_38993)
        
        
        # Call to process_template_line(...): (line 265)
        # Processing the call arguments (line 265)
        str_38997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 40), 'str', 'recursive-exclude e *.py')
        # Processing the call keyword arguments (line 265)
        kwargs_38998 = {}
        # Getting the type of 'file_list' (line 265)
        file_list_38995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'file_list', False)
        # Obtaining the member 'process_template_line' of a type (line 265)
        process_template_line_38996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), file_list_38995, 'process_template_line')
        # Calling process_template_line(args, kwargs) (line 265)
        process_template_line_call_result_38999 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), process_template_line_38996, *[str_38997], **kwargs_38998)
        
        
        # Call to assertEqual(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'file_list' (line 266)
        file_list_39002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 25), 'file_list', False)
        # Obtaining the member 'files' of a type (line 266)
        files_39003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 25), file_list_39002, 'files')
        
        # Obtaining an instance of the builtin type 'list' (line 266)
        list_39004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 266)
        # Adding element type (line 266)
        str_39005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 43), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 42), list_39004, str_39005)
        # Adding element type (line 266)
        
        # Call to l(...): (line 266)
        # Processing the call arguments (line 266)
        str_39007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 53), 'str', 'd/c.txt')
        # Processing the call keyword arguments (line 266)
        kwargs_39008 = {}
        # Getting the type of 'l' (line 266)
        l_39006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 51), 'l', False)
        # Calling l(args, kwargs) (line 266)
        l_call_result_39009 = invoke(stypy.reporting.localization.Localization(__file__, 266, 51), l_39006, *[str_39007], **kwargs_39008)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 42), list_39004, l_call_result_39009)
        
        # Processing the call keyword arguments (line 266)
        kwargs_39010 = {}
        # Getting the type of 'self' (line 266)
        self_39000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 266)
        assertEqual_39001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), self_39000, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 266)
        assertEqual_call_result_39011 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), assertEqual_39001, *[files_39003, list_39004], **kwargs_39010)
        
        
        # Call to assertWarnings(...): (line 267)
        # Processing the call keyword arguments (line 267)
        kwargs_39014 = {}
        # Getting the type of 'self' (line 267)
        self_39012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'self', False)
        # Obtaining the member 'assertWarnings' of a type (line 267)
        assertWarnings_39013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 8), self_39012, 'assertWarnings')
        # Calling assertWarnings(args, kwargs) (line 267)
        assertWarnings_call_result_39015 = invoke(stypy.reporting.localization.Localization(__file__, 267, 8), assertWarnings_39013, *[], **kwargs_39014)
        
        
        # Assigning a Call to a Name (line 270):
        
        # Call to FileList(...): (line 270)
        # Processing the call keyword arguments (line 270)
        kwargs_39017 = {}
        # Getting the type of 'FileList' (line 270)
        FileList_39016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 20), 'FileList', False)
        # Calling FileList(args, kwargs) (line 270)
        FileList_call_result_39018 = invoke(stypy.reporting.localization.Localization(__file__, 270, 20), FileList_39016, *[], **kwargs_39017)
        
        # Assigning a type to the variable 'file_list' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'file_list', FileList_call_result_39018)
        
        # Call to set_allfiles(...): (line 271)
        # Processing the call arguments (line 271)
        
        # Obtaining an instance of the builtin type 'list' (line 271)
        list_39021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 271)
        # Adding element type (line 271)
        str_39022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 32), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 31), list_39021, str_39022)
        # Adding element type (line 271)
        
        # Call to l(...): (line 271)
        # Processing the call arguments (line 271)
        str_39024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 42), 'str', 'd/b.py')
        # Processing the call keyword arguments (line 271)
        kwargs_39025 = {}
        # Getting the type of 'l' (line 271)
        l_39023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 40), 'l', False)
        # Calling l(args, kwargs) (line 271)
        l_call_result_39026 = invoke(stypy.reporting.localization.Localization(__file__, 271, 40), l_39023, *[str_39024], **kwargs_39025)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 31), list_39021, l_call_result_39026)
        # Adding element type (line 271)
        
        # Call to l(...): (line 271)
        # Processing the call arguments (line 271)
        str_39028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 55), 'str', 'd/d/e.py')
        # Processing the call keyword arguments (line 271)
        kwargs_39029 = {}
        # Getting the type of 'l' (line 271)
        l_39027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 53), 'l', False)
        # Calling l(args, kwargs) (line 271)
        l_call_result_39030 = invoke(stypy.reporting.localization.Localization(__file__, 271, 53), l_39027, *[str_39028], **kwargs_39029)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 31), list_39021, l_call_result_39030)
        # Adding element type (line 271)
        
        # Call to l(...): (line 272)
        # Processing the call arguments (line 272)
        str_39032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 34), 'str', 'f/f.py')
        # Processing the call keyword arguments (line 272)
        kwargs_39033 = {}
        # Getting the type of 'l' (line 272)
        l_39031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 32), 'l', False)
        # Calling l(args, kwargs) (line 272)
        l_call_result_39034 = invoke(stypy.reporting.localization.Localization(__file__, 272, 32), l_39031, *[str_39032], **kwargs_39033)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 31), list_39021, l_call_result_39034)
        
        # Processing the call keyword arguments (line 271)
        kwargs_39035 = {}
        # Getting the type of 'file_list' (line 271)
        file_list_39019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'file_list', False)
        # Obtaining the member 'set_allfiles' of a type (line 271)
        set_allfiles_39020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 8), file_list_39019, 'set_allfiles')
        # Calling set_allfiles(args, kwargs) (line 271)
        set_allfiles_call_result_39036 = invoke(stypy.reporting.localization.Localization(__file__, 271, 8), set_allfiles_39020, *[list_39021], **kwargs_39035)
        
        
        # Call to process_template_line(...): (line 274)
        # Processing the call arguments (line 274)
        str_39039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 40), 'str', 'graft d')
        # Processing the call keyword arguments (line 274)
        kwargs_39040 = {}
        # Getting the type of 'file_list' (line 274)
        file_list_39037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'file_list', False)
        # Obtaining the member 'process_template_line' of a type (line 274)
        process_template_line_39038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), file_list_39037, 'process_template_line')
        # Calling process_template_line(args, kwargs) (line 274)
        process_template_line_call_result_39041 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), process_template_line_39038, *[str_39039], **kwargs_39040)
        
        
        # Call to assertEqual(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'file_list' (line 275)
        file_list_39044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 25), 'file_list', False)
        # Obtaining the member 'files' of a type (line 275)
        files_39045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 25), file_list_39044, 'files')
        
        # Obtaining an instance of the builtin type 'list' (line 275)
        list_39046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 275)
        # Adding element type (line 275)
        
        # Call to l(...): (line 275)
        # Processing the call arguments (line 275)
        str_39048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 45), 'str', 'd/b.py')
        # Processing the call keyword arguments (line 275)
        kwargs_39049 = {}
        # Getting the type of 'l' (line 275)
        l_39047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 43), 'l', False)
        # Calling l(args, kwargs) (line 275)
        l_call_result_39050 = invoke(stypy.reporting.localization.Localization(__file__, 275, 43), l_39047, *[str_39048], **kwargs_39049)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 42), list_39046, l_call_result_39050)
        # Adding element type (line 275)
        
        # Call to l(...): (line 275)
        # Processing the call arguments (line 275)
        str_39052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 58), 'str', 'd/d/e.py')
        # Processing the call keyword arguments (line 275)
        kwargs_39053 = {}
        # Getting the type of 'l' (line 275)
        l_39051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 56), 'l', False)
        # Calling l(args, kwargs) (line 275)
        l_call_result_39054 = invoke(stypy.reporting.localization.Localization(__file__, 275, 56), l_39051, *[str_39052], **kwargs_39053)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 42), list_39046, l_call_result_39054)
        
        # Processing the call keyword arguments (line 275)
        kwargs_39055 = {}
        # Getting the type of 'self' (line 275)
        self_39042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 275)
        assertEqual_39043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), self_39042, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 275)
        assertEqual_call_result_39056 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), assertEqual_39043, *[files_39045, list_39046], **kwargs_39055)
        
        
        # Call to assertNoWarnings(...): (line 276)
        # Processing the call keyword arguments (line 276)
        kwargs_39059 = {}
        # Getting the type of 'self' (line 276)
        self_39057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'self', False)
        # Obtaining the member 'assertNoWarnings' of a type (line 276)
        assertNoWarnings_39058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), self_39057, 'assertNoWarnings')
        # Calling assertNoWarnings(args, kwargs) (line 276)
        assertNoWarnings_call_result_39060 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), assertNoWarnings_39058, *[], **kwargs_39059)
        
        
        # Call to process_template_line(...): (line 278)
        # Processing the call arguments (line 278)
        str_39063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 40), 'str', 'graft e')
        # Processing the call keyword arguments (line 278)
        kwargs_39064 = {}
        # Getting the type of 'file_list' (line 278)
        file_list_39061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'file_list', False)
        # Obtaining the member 'process_template_line' of a type (line 278)
        process_template_line_39062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), file_list_39061, 'process_template_line')
        # Calling process_template_line(args, kwargs) (line 278)
        process_template_line_call_result_39065 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), process_template_line_39062, *[str_39063], **kwargs_39064)
        
        
        # Call to assertEqual(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'file_list' (line 279)
        file_list_39068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 25), 'file_list', False)
        # Obtaining the member 'files' of a type (line 279)
        files_39069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 25), file_list_39068, 'files')
        
        # Obtaining an instance of the builtin type 'list' (line 279)
        list_39070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 279)
        # Adding element type (line 279)
        
        # Call to l(...): (line 279)
        # Processing the call arguments (line 279)
        str_39072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 45), 'str', 'd/b.py')
        # Processing the call keyword arguments (line 279)
        kwargs_39073 = {}
        # Getting the type of 'l' (line 279)
        l_39071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 43), 'l', False)
        # Calling l(args, kwargs) (line 279)
        l_call_result_39074 = invoke(stypy.reporting.localization.Localization(__file__, 279, 43), l_39071, *[str_39072], **kwargs_39073)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 42), list_39070, l_call_result_39074)
        # Adding element type (line 279)
        
        # Call to l(...): (line 279)
        # Processing the call arguments (line 279)
        str_39076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 58), 'str', 'd/d/e.py')
        # Processing the call keyword arguments (line 279)
        kwargs_39077 = {}
        # Getting the type of 'l' (line 279)
        l_39075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 56), 'l', False)
        # Calling l(args, kwargs) (line 279)
        l_call_result_39078 = invoke(stypy.reporting.localization.Localization(__file__, 279, 56), l_39075, *[str_39076], **kwargs_39077)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 42), list_39070, l_call_result_39078)
        
        # Processing the call keyword arguments (line 279)
        kwargs_39079 = {}
        # Getting the type of 'self' (line 279)
        self_39066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 279)
        assertEqual_39067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 8), self_39066, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 279)
        assertEqual_call_result_39080 = invoke(stypy.reporting.localization.Localization(__file__, 279, 8), assertEqual_39067, *[files_39069, list_39070], **kwargs_39079)
        
        
        # Call to assertWarnings(...): (line 280)
        # Processing the call keyword arguments (line 280)
        kwargs_39083 = {}
        # Getting the type of 'self' (line 280)
        self_39081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'self', False)
        # Obtaining the member 'assertWarnings' of a type (line 280)
        assertWarnings_39082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 8), self_39081, 'assertWarnings')
        # Calling assertWarnings(args, kwargs) (line 280)
        assertWarnings_call_result_39084 = invoke(stypy.reporting.localization.Localization(__file__, 280, 8), assertWarnings_39082, *[], **kwargs_39083)
        
        
        # Assigning a Call to a Name (line 283):
        
        # Call to FileList(...): (line 283)
        # Processing the call keyword arguments (line 283)
        kwargs_39086 = {}
        # Getting the type of 'FileList' (line 283)
        FileList_39085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 20), 'FileList', False)
        # Calling FileList(args, kwargs) (line 283)
        FileList_call_result_39087 = invoke(stypy.reporting.localization.Localization(__file__, 283, 20), FileList_39085, *[], **kwargs_39086)
        
        # Assigning a type to the variable 'file_list' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'file_list', FileList_call_result_39087)
        
        # Assigning a List to a Attribute (line 284):
        
        # Obtaining an instance of the builtin type 'list' (line 284)
        list_39088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 284)
        # Adding element type (line 284)
        str_39089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 27), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 26), list_39088, str_39089)
        # Adding element type (line 284)
        
        # Call to l(...): (line 284)
        # Processing the call arguments (line 284)
        str_39091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 37), 'str', 'd/b.py')
        # Processing the call keyword arguments (line 284)
        kwargs_39092 = {}
        # Getting the type of 'l' (line 284)
        l_39090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 35), 'l', False)
        # Calling l(args, kwargs) (line 284)
        l_call_result_39093 = invoke(stypy.reporting.localization.Localization(__file__, 284, 35), l_39090, *[str_39091], **kwargs_39092)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 26), list_39088, l_call_result_39093)
        # Adding element type (line 284)
        
        # Call to l(...): (line 284)
        # Processing the call arguments (line 284)
        str_39095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 50), 'str', 'd/d/e.py')
        # Processing the call keyword arguments (line 284)
        kwargs_39096 = {}
        # Getting the type of 'l' (line 284)
        l_39094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 48), 'l', False)
        # Calling l(args, kwargs) (line 284)
        l_call_result_39097 = invoke(stypy.reporting.localization.Localization(__file__, 284, 48), l_39094, *[str_39095], **kwargs_39096)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 26), list_39088, l_call_result_39097)
        # Adding element type (line 284)
        
        # Call to l(...): (line 284)
        # Processing the call arguments (line 284)
        str_39099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 65), 'str', 'f/f.py')
        # Processing the call keyword arguments (line 284)
        kwargs_39100 = {}
        # Getting the type of 'l' (line 284)
        l_39098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 63), 'l', False)
        # Calling l(args, kwargs) (line 284)
        l_call_result_39101 = invoke(stypy.reporting.localization.Localization(__file__, 284, 63), l_39098, *[str_39099], **kwargs_39100)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 26), list_39088, l_call_result_39101)
        
        # Getting the type of 'file_list' (line 284)
        file_list_39102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'file_list')
        # Setting the type of the member 'files' of a type (line 284)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), file_list_39102, 'files', list_39088)
        
        # Call to process_template_line(...): (line 286)
        # Processing the call arguments (line 286)
        str_39105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 40), 'str', 'prune d')
        # Processing the call keyword arguments (line 286)
        kwargs_39106 = {}
        # Getting the type of 'file_list' (line 286)
        file_list_39103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'file_list', False)
        # Obtaining the member 'process_template_line' of a type (line 286)
        process_template_line_39104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), file_list_39103, 'process_template_line')
        # Calling process_template_line(args, kwargs) (line 286)
        process_template_line_call_result_39107 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), process_template_line_39104, *[str_39105], **kwargs_39106)
        
        
        # Call to assertEqual(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'file_list' (line 287)
        file_list_39110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 25), 'file_list', False)
        # Obtaining the member 'files' of a type (line 287)
        files_39111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 25), file_list_39110, 'files')
        
        # Obtaining an instance of the builtin type 'list' (line 287)
        list_39112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 287)
        # Adding element type (line 287)
        str_39113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 43), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 42), list_39112, str_39113)
        # Adding element type (line 287)
        
        # Call to l(...): (line 287)
        # Processing the call arguments (line 287)
        str_39115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 53), 'str', 'f/f.py')
        # Processing the call keyword arguments (line 287)
        kwargs_39116 = {}
        # Getting the type of 'l' (line 287)
        l_39114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 51), 'l', False)
        # Calling l(args, kwargs) (line 287)
        l_call_result_39117 = invoke(stypy.reporting.localization.Localization(__file__, 287, 51), l_39114, *[str_39115], **kwargs_39116)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 42), list_39112, l_call_result_39117)
        
        # Processing the call keyword arguments (line 287)
        kwargs_39118 = {}
        # Getting the type of 'self' (line 287)
        self_39108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 287)
        assertEqual_39109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), self_39108, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 287)
        assertEqual_call_result_39119 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), assertEqual_39109, *[files_39111, list_39112], **kwargs_39118)
        
        
        # Call to assertNoWarnings(...): (line 288)
        # Processing the call keyword arguments (line 288)
        kwargs_39122 = {}
        # Getting the type of 'self' (line 288)
        self_39120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'self', False)
        # Obtaining the member 'assertNoWarnings' of a type (line 288)
        assertNoWarnings_39121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), self_39120, 'assertNoWarnings')
        # Calling assertNoWarnings(args, kwargs) (line 288)
        assertNoWarnings_call_result_39123 = invoke(stypy.reporting.localization.Localization(__file__, 288, 8), assertNoWarnings_39121, *[], **kwargs_39122)
        
        
        # Call to process_template_line(...): (line 290)
        # Processing the call arguments (line 290)
        str_39126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 40), 'str', 'prune e')
        # Processing the call keyword arguments (line 290)
        kwargs_39127 = {}
        # Getting the type of 'file_list' (line 290)
        file_list_39124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'file_list', False)
        # Obtaining the member 'process_template_line' of a type (line 290)
        process_template_line_39125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), file_list_39124, 'process_template_line')
        # Calling process_template_line(args, kwargs) (line 290)
        process_template_line_call_result_39128 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), process_template_line_39125, *[str_39126], **kwargs_39127)
        
        
        # Call to assertEqual(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'file_list' (line 291)
        file_list_39131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 25), 'file_list', False)
        # Obtaining the member 'files' of a type (line 291)
        files_39132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 25), file_list_39131, 'files')
        
        # Obtaining an instance of the builtin type 'list' (line 291)
        list_39133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 291)
        # Adding element type (line 291)
        str_39134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 43), 'str', 'a.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 42), list_39133, str_39134)
        # Adding element type (line 291)
        
        # Call to l(...): (line 291)
        # Processing the call arguments (line 291)
        str_39136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 53), 'str', 'f/f.py')
        # Processing the call keyword arguments (line 291)
        kwargs_39137 = {}
        # Getting the type of 'l' (line 291)
        l_39135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 51), 'l', False)
        # Calling l(args, kwargs) (line 291)
        l_call_result_39138 = invoke(stypy.reporting.localization.Localization(__file__, 291, 51), l_39135, *[str_39136], **kwargs_39137)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 42), list_39133, l_call_result_39138)
        
        # Processing the call keyword arguments (line 291)
        kwargs_39139 = {}
        # Getting the type of 'self' (line 291)
        self_39129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 291)
        assertEqual_39130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), self_39129, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 291)
        assertEqual_call_result_39140 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), assertEqual_39130, *[files_39132, list_39133], **kwargs_39139)
        
        
        # Call to assertWarnings(...): (line 292)
        # Processing the call keyword arguments (line 292)
        kwargs_39143 = {}
        # Getting the type of 'self' (line 292)
        self_39141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'self', False)
        # Obtaining the member 'assertWarnings' of a type (line 292)
        assertWarnings_39142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), self_39141, 'assertWarnings')
        # Calling assertWarnings(args, kwargs) (line 292)
        assertWarnings_call_result_39144 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), assertWarnings_39142, *[], **kwargs_39143)
        
        
        # ################# End of 'test_process_template(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_process_template' in the type store
        # Getting the type of 'stypy_return_type' (line 186)
        stypy_return_type_39145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39145)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_process_template'
        return stypy_return_type_39145


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 34, 0, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileListTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FileListTestCase' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'FileListTestCase', FileListTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 295, 0, False)
    
    # Passed parameters checking function
    test_suite.stypy_localization = localization
    test_suite.stypy_type_of_self = None
    test_suite.stypy_type_store = module_type_store
    test_suite.stypy_function_name = 'test_suite'
    test_suite.stypy_param_names_list = []
    test_suite.stypy_varargs_param_name = None
    test_suite.stypy_kwargs_param_name = None
    test_suite.stypy_call_defaults = defaults
    test_suite.stypy_call_varargs = varargs
    test_suite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_suite', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_suite', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_suite(...)' code ##################

    
    # Call to makeSuite(...): (line 296)
    # Processing the call arguments (line 296)
    # Getting the type of 'FileListTestCase' (line 296)
    FileListTestCase_39148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 30), 'FileListTestCase', False)
    # Processing the call keyword arguments (line 296)
    kwargs_39149 = {}
    # Getting the type of 'unittest' (line 296)
    unittest_39146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 296)
    makeSuite_39147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 11), unittest_39146, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 296)
    makeSuite_call_result_39150 = invoke(stypy.reporting.localization.Localization(__file__, 296, 11), makeSuite_39147, *[FileListTestCase_39148], **kwargs_39149)
    
    # Assigning a type to the variable 'stypy_return_type' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'stypy_return_type', makeSuite_call_result_39150)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 295)
    stypy_return_type_39151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39151)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_39151

# Assigning a type to the variable 'test_suite' (line 295)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 299)
    # Processing the call arguments (line 299)
    
    # Call to test_suite(...): (line 299)
    # Processing the call keyword arguments (line 299)
    kwargs_39154 = {}
    # Getting the type of 'test_suite' (line 299)
    test_suite_39153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 299)
    test_suite_call_result_39155 = invoke(stypy.reporting.localization.Localization(__file__, 299, 17), test_suite_39153, *[], **kwargs_39154)
    
    # Processing the call keyword arguments (line 299)
    kwargs_39156 = {}
    # Getting the type of 'run_unittest' (line 299)
    run_unittest_39152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 299)
    run_unittest_call_result_39157 = invoke(stypy.reporting.localization.Localization(__file__, 299, 4), run_unittest_39152, *[test_suite_call_result_39155], **kwargs_39156)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
