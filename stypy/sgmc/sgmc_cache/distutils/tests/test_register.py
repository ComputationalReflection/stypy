
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # -*- encoding: utf8 -*-
2: '''Tests for distutils.command.register.'''
3: import os
4: import unittest
5: import getpass
6: import urllib2
7: import warnings
8: 
9: from test.test_support import check_warnings, run_unittest
10: 
11: from distutils.command import register as register_module
12: from distutils.command.register import register
13: from distutils.errors import DistutilsSetupError
14: 
15: from distutils.tests.test_config import PyPIRCCommandTestCase
16: 
17: try:
18:     import docutils
19: except ImportError:
20:     docutils = None
21: 
22: PYPIRC_NOPASSWORD = '''\
23: [distutils]
24: 
25: index-servers =
26:     server1
27: 
28: [server1]
29: username:me
30: '''
31: 
32: WANTED_PYPIRC = '''\
33: [distutils]
34: index-servers =
35:     pypi
36: 
37: [pypi]
38: username:tarek
39: password:password
40: '''
41: 
42: class RawInputs(object):
43:     '''Fakes user inputs.'''
44:     def __init__(self, *answers):
45:         self.answers = answers
46:         self.index = 0
47: 
48:     def __call__(self, prompt=''):
49:         try:
50:             return self.answers[self.index]
51:         finally:
52:             self.index += 1
53: 
54: class FakeOpener(object):
55:     '''Fakes a PyPI server'''
56:     def __init__(self):
57:         self.reqs = []
58: 
59:     def __call__(self, *args):
60:         return self
61: 
62:     def open(self, req):
63:         self.reqs.append(req)
64:         return self
65: 
66:     def read(self):
67:         return 'xxx'
68: 
69: class RegisterTestCase(PyPIRCCommandTestCase):
70: 
71:     def setUp(self):
72:         super(RegisterTestCase, self).setUp()
73:         # patching the password prompt
74:         self._old_getpass = getpass.getpass
75:         def _getpass(prompt):
76:             return 'password'
77:         getpass.getpass = _getpass
78:         self.old_opener = urllib2.build_opener
79:         self.conn = urllib2.build_opener = FakeOpener()
80: 
81:     def tearDown(self):
82:         getpass.getpass = self._old_getpass
83:         urllib2.build_opener = self.old_opener
84:         super(RegisterTestCase, self).tearDown()
85: 
86:     def _get_cmd(self, metadata=None):
87:         if metadata is None:
88:             metadata = {'url': 'xxx', 'author': 'xxx',
89:                         'author_email': 'xxx',
90:                         'name': 'xxx', 'version': 'xxx'}
91:         pkg_info, dist = self.create_dist(**metadata)
92:         return register(dist)
93: 
94:     def test_create_pypirc(self):
95:         # this test makes sure a .pypirc file
96:         # is created when requested.
97: 
98:         # let's create a register instance
99:         cmd = self._get_cmd()
100: 
101:         # we shouldn't have a .pypirc file yet
102:         self.assertFalse(os.path.exists(self.rc))
103: 
104:         # patching raw_input and getpass.getpass
105:         # so register gets happy
106:         #
107:         # Here's what we are faking :
108:         # use your existing login (choice 1.)
109:         # Username : 'tarek'
110:         # Password : 'password'
111:         # Save your login (y/N)? : 'y'
112:         inputs = RawInputs('1', 'tarek', 'y')
113:         register_module.raw_input = inputs.__call__
114:         # let's run the command
115:         try:
116:             cmd.run()
117:         finally:
118:             del register_module.raw_input
119: 
120:         # we should have a brand new .pypirc file
121:         self.assertTrue(os.path.exists(self.rc))
122: 
123:         # with the content similar to WANTED_PYPIRC
124:         f = open(self.rc)
125:         try:
126:             content = f.read()
127:             self.assertEqual(content, WANTED_PYPIRC)
128:         finally:
129:             f.close()
130: 
131:         # now let's make sure the .pypirc file generated
132:         # really works : we shouldn't be asked anything
133:         # if we run the command again
134:         def _no_way(prompt=''):
135:             raise AssertionError(prompt)
136:         register_module.raw_input = _no_way
137: 
138:         cmd.show_response = 1
139:         cmd.run()
140: 
141:         # let's see what the server received : we should
142:         # have 2 similar requests
143:         self.assertEqual(len(self.conn.reqs), 2)
144:         req1 = dict(self.conn.reqs[0].headers)
145:         req2 = dict(self.conn.reqs[1].headers)
146:         self.assertEqual(req2['Content-length'], req1['Content-length'])
147:         self.assertIn('xxx', self.conn.reqs[1].data)
148: 
149:     def test_password_not_in_file(self):
150: 
151:         self.write_file(self.rc, PYPIRC_NOPASSWORD)
152:         cmd = self._get_cmd()
153:         cmd._set_config()
154:         cmd.finalize_options()
155:         cmd.send_metadata()
156: 
157:         # dist.password should be set
158:         # therefore used afterwards by other commands
159:         self.assertEqual(cmd.distribution.password, 'password')
160: 
161:     def test_registering(self):
162:         # this test runs choice 2
163:         cmd = self._get_cmd()
164:         inputs = RawInputs('2', 'tarek', 'tarek@ziade.org')
165:         register_module.raw_input = inputs.__call__
166:         try:
167:             # let's run the command
168:             cmd.run()
169:         finally:
170:             del register_module.raw_input
171: 
172:         # we should have send a request
173:         self.assertEqual(len(self.conn.reqs), 1)
174:         req = self.conn.reqs[0]
175:         headers = dict(req.headers)
176:         self.assertEqual(headers['Content-length'], '608')
177:         self.assertIn('tarek', req.data)
178: 
179:     def test_password_reset(self):
180:         # this test runs choice 3
181:         cmd = self._get_cmd()
182:         inputs = RawInputs('3', 'tarek@ziade.org')
183:         register_module.raw_input = inputs.__call__
184:         try:
185:             # let's run the command
186:             cmd.run()
187:         finally:
188:             del register_module.raw_input
189: 
190:         # we should have send a request
191:         self.assertEqual(len(self.conn.reqs), 1)
192:         req = self.conn.reqs[0]
193:         headers = dict(req.headers)
194:         self.assertEqual(headers['Content-length'], '290')
195:         self.assertIn('tarek', req.data)
196: 
197:     @unittest.skipUnless(docutils is not None, 'needs docutils')
198:     def test_strict(self):
199:         # testing the script option
200:         # when on, the register command stops if
201:         # the metadata is incomplete or if
202:         # long_description is not reSt compliant
203: 
204:         # empty metadata
205:         cmd = self._get_cmd({})
206:         cmd.ensure_finalized()
207:         cmd.strict = 1
208:         self.assertRaises(DistutilsSetupError, cmd.run)
209: 
210:         # metadata are OK but long_description is broken
211:         metadata = {'url': 'xxx', 'author': 'xxx',
212:                     'author_email': u'éxéxé',
213:                     'name': 'xxx', 'version': 'xxx',
214:                     'long_description': 'title\n==\n\ntext'}
215: 
216:         cmd = self._get_cmd(metadata)
217:         cmd.ensure_finalized()
218:         cmd.strict = 1
219:         self.assertRaises(DistutilsSetupError, cmd.run)
220: 
221:         # now something that works
222:         metadata['long_description'] = 'title\n=====\n\ntext'
223:         cmd = self._get_cmd(metadata)
224:         cmd.ensure_finalized()
225:         cmd.strict = 1
226:         inputs = RawInputs('1', 'tarek', 'y')
227:         register_module.raw_input = inputs.__call__
228:         # let's run the command
229:         try:
230:             cmd.run()
231:         finally:
232:             del register_module.raw_input
233: 
234:         # strict is not by default
235:         cmd = self._get_cmd()
236:         cmd.ensure_finalized()
237:         inputs = RawInputs('1', 'tarek', 'y')
238:         register_module.raw_input = inputs.__call__
239:         # let's run the command
240:         try:
241:             cmd.run()
242:         finally:
243:             del register_module.raw_input
244: 
245:         # and finally a Unicode test (bug #12114)
246:         metadata = {'url': u'xxx', 'author': u'\u00c9ric',
247:                     'author_email': u'xxx', u'name': 'xxx',
248:                     'version': u'xxx',
249:                     'description': u'Something about esszet \u00df',
250:                     'long_description': u'More things about esszet \u00df'}
251: 
252:         cmd = self._get_cmd(metadata)
253:         cmd.ensure_finalized()
254:         cmd.strict = 1
255:         inputs = RawInputs('1', 'tarek', 'y')
256:         register_module.raw_input = inputs.__call__
257:         # let's run the command
258:         try:
259:             cmd.run()
260:         finally:
261:             del register_module.raw_input
262: 
263:     @unittest.skipUnless(docutils is not None, 'needs docutils')
264:     def test_register_invalid_long_description(self):
265:         description = ':funkie:`str`'  # mimic Sphinx-specific markup
266:         metadata = {'url': 'xxx', 'author': 'xxx',
267:                     'author_email': 'xxx',
268:                     'name': 'xxx', 'version': 'xxx',
269:                     'long_description': description}
270:         cmd = self._get_cmd(metadata)
271:         cmd.ensure_finalized()
272:         cmd.strict = True
273:         inputs = RawInputs('2', 'tarek', 'tarek@ziade.org')
274:         register_module.raw_input = inputs
275:         self.addCleanup(delattr, register_module, 'raw_input')
276:         self.assertRaises(DistutilsSetupError, cmd.run)
277: 
278:     def test_check_metadata_deprecated(self):
279:         # makes sure make_metadata is deprecated
280:         cmd = self._get_cmd()
281:         with check_warnings() as w:
282:             warnings.simplefilter("always")
283:             cmd.check_metadata()
284:             self.assertEqual(len(w.warnings), 1)
285: 
286: def test_suite():
287:     return unittest.makeSuite(RegisterTestCase)
288: 
289: if __name__ == "__main__":
290:     run_unittest(test_suite())
291: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_41769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 0), 'str', 'Tests for distutils.command.register.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import unittest' statement (line 4)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import getpass' statement (line 5)
import getpass

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'getpass', getpass, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import urllib2' statement (line 6)
import urllib2

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'urllib2', urllib2, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import warnings' statement (line 7)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from test.test_support import check_warnings, run_unittest' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_41770 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support')

if (type(import_41770) is not StypyTypeError):

    if (import_41770 != 'pyd_module'):
        __import__(import_41770)
        sys_modules_41771 = sys.modules[import_41770]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', sys_modules_41771.module_type_store, module_type_store, ['check_warnings', 'run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_41771, sys_modules_41771.module_type_store, module_type_store)
    else:
        from test.test_support import check_warnings, run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', None, module_type_store, ['check_warnings', 'run_unittest'], [check_warnings, run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', import_41770)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.command import register_module' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_41772 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.command')

if (type(import_41772) is not StypyTypeError):

    if (import_41772 != 'pyd_module'):
        __import__(import_41772)
        sys_modules_41773 = sys.modules[import_41772]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.command', sys_modules_41773.module_type_store, module_type_store, ['register'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_41773, sys_modules_41773.module_type_store, module_type_store)
    else:
        from distutils.command import register as register_module

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.command', None, module_type_store, ['register'], [register_module])

else:
    # Assigning a type to the variable 'distutils.command' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.command', import_41772)

# Adding an alias
module_type_store.add_alias('register_module', 'register')
remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.command.register import register' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_41774 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.command.register')

if (type(import_41774) is not StypyTypeError):

    if (import_41774 != 'pyd_module'):
        __import__(import_41774)
        sys_modules_41775 = sys.modules[import_41774]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.command.register', sys_modules_41775.module_type_store, module_type_store, ['register'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_41775, sys_modules_41775.module_type_store, module_type_store)
    else:
        from distutils.command.register import register

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.command.register', None, module_type_store, ['register'], [register])

else:
    # Assigning a type to the variable 'distutils.command.register' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.command.register', import_41774)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from distutils.errors import DistutilsSetupError' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_41776 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.errors')

if (type(import_41776) is not StypyTypeError):

    if (import_41776 != 'pyd_module'):
        __import__(import_41776)
        sys_modules_41777 = sys.modules[import_41776]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.errors', sys_modules_41777.module_type_store, module_type_store, ['DistutilsSetupError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_41777, sys_modules_41777.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsSetupError

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.errors', None, module_type_store, ['DistutilsSetupError'], [DistutilsSetupError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.errors', import_41776)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from distutils.tests.test_config import PyPIRCCommandTestCase' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_41778 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.tests.test_config')

if (type(import_41778) is not StypyTypeError):

    if (import_41778 != 'pyd_module'):
        __import__(import_41778)
        sys_modules_41779 = sys.modules[import_41778]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.tests.test_config', sys_modules_41779.module_type_store, module_type_store, ['PyPIRCCommandTestCase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_41779, sys_modules_41779.module_type_store, module_type_store)
    else:
        from distutils.tests.test_config import PyPIRCCommandTestCase

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.tests.test_config', None, module_type_store, ['PyPIRCCommandTestCase'], [PyPIRCCommandTestCase])

else:
    # Assigning a type to the variable 'distutils.tests.test_config' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.tests.test_config', import_41778)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')



# SSA begins for try-except statement (line 17)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 4))

# 'import docutils' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_41780 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'docutils')

if (type(import_41780) is not StypyTypeError):

    if (import_41780 != 'pyd_module'):
        __import__(import_41780)
        sys_modules_41781 = sys.modules[import_41780]
        import_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'docutils', sys_modules_41781.module_type_store, module_type_store)
    else:
        import docutils

        import_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'docutils', docutils, module_type_store)

else:
    # Assigning a type to the variable 'docutils' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'docutils', import_41780)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# SSA branch for the except part of a try statement (line 17)
# SSA branch for the except 'ImportError' branch of a try statement (line 17)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 20):

# Assigning a Name to a Name (line 20):
# Getting the type of 'None' (line 20)
None_41782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 15), 'None')
# Assigning a type to the variable 'docutils' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'docutils', None_41782)
# SSA join for try-except statement (line 17)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Str to a Name (line 22):

# Assigning a Str to a Name (line 22):
str_41783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, (-1)), 'str', '[distutils]\n\nindex-servers =\n    server1\n\n[server1]\nusername:me\n')
# Assigning a type to the variable 'PYPIRC_NOPASSWORD' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'PYPIRC_NOPASSWORD', str_41783)

# Assigning a Str to a Name (line 32):

# Assigning a Str to a Name (line 32):
str_41784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, (-1)), 'str', '[distutils]\nindex-servers =\n    pypi\n\n[pypi]\nusername:tarek\npassword:password\n')
# Assigning a type to the variable 'WANTED_PYPIRC' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'WANTED_PYPIRC', str_41784)
# Declaration of the 'RawInputs' class

class RawInputs(object, ):
    str_41785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 4), 'str', 'Fakes user inputs.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RawInputs.__init__', [], 'answers', None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 45):
        
        # Assigning a Name to a Attribute (line 45):
        # Getting the type of 'answers' (line 45)
        answers_41786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 23), 'answers')
        # Getting the type of 'self' (line 45)
        self_41787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self')
        # Setting the type of the member 'answers' of a type (line 45)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_41787, 'answers', answers_41786)
        
        # Assigning a Num to a Attribute (line 46):
        
        # Assigning a Num to a Attribute (line 46):
        int_41788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 21), 'int')
        # Getting the type of 'self' (line 46)
        self_41789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self')
        # Setting the type of the member 'index' of a type (line 46)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_41789, 'index', int_41788)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_41790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 30), 'str', '')
        defaults = [str_41790]
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 48, 4, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RawInputs.__call__.__dict__.__setitem__('stypy_localization', localization)
        RawInputs.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RawInputs.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        RawInputs.__call__.__dict__.__setitem__('stypy_function_name', 'RawInputs.__call__')
        RawInputs.__call__.__dict__.__setitem__('stypy_param_names_list', ['prompt'])
        RawInputs.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        RawInputs.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RawInputs.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        RawInputs.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        RawInputs.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RawInputs.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RawInputs.__call__', ['prompt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['prompt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Try-finally block (line 49)
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 50)
        self_41791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 32), 'self')
        # Obtaining the member 'index' of a type (line 50)
        index_41792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 32), self_41791, 'index')
        # Getting the type of 'self' (line 50)
        self_41793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 19), 'self')
        # Obtaining the member 'answers' of a type (line 50)
        answers_41794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 19), self_41793, 'answers')
        # Obtaining the member '__getitem__' of a type (line 50)
        getitem___41795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 19), answers_41794, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 50)
        subscript_call_result_41796 = invoke(stypy.reporting.localization.Localization(__file__, 50, 19), getitem___41795, index_41792)
        
        # Assigning a type to the variable 'stypy_return_type' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'stypy_return_type', subscript_call_result_41796)
        
        # finally branch of the try-finally block (line 49)
        
        # Getting the type of 'self' (line 52)
        self_41797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'self')
        # Obtaining the member 'index' of a type (line 52)
        index_41798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), self_41797, 'index')
        int_41799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 26), 'int')
        # Applying the binary operator '+=' (line 52)
        result_iadd_41800 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 12), '+=', index_41798, int_41799)
        # Getting the type of 'self' (line 52)
        self_41801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'self')
        # Setting the type of the member 'index' of a type (line 52)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), self_41801, 'index', result_iadd_41800)
        
        
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 48)
        stypy_return_type_41802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_41802)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_41802


# Assigning a type to the variable 'RawInputs' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'RawInputs', RawInputs)
# Declaration of the 'FakeOpener' class

class FakeOpener(object, ):
    str_41803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 4), 'str', 'Fakes a PyPI server')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FakeOpener.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Attribute (line 57):
        
        # Assigning a List to a Attribute (line 57):
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_41804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        
        # Getting the type of 'self' (line 57)
        self_41805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self')
        # Setting the type of the member 'reqs' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_41805, 'reqs', list_41804)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FakeOpener.__call__.__dict__.__setitem__('stypy_localization', localization)
        FakeOpener.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FakeOpener.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FakeOpener.__call__.__dict__.__setitem__('stypy_function_name', 'FakeOpener.__call__')
        FakeOpener.__call__.__dict__.__setitem__('stypy_param_names_list', [])
        FakeOpener.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FakeOpener.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FakeOpener.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FakeOpener.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FakeOpener.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FakeOpener.__call__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FakeOpener.__call__', [], 'args', None, defaults, varargs, kwargs)

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

        # Getting the type of 'self' (line 60)
        self_41806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'stypy_return_type', self_41806)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_41807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_41807)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_41807


    @norecursion
    def open(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'open'
        module_type_store = module_type_store.open_function_context('open', 62, 4, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FakeOpener.open.__dict__.__setitem__('stypy_localization', localization)
        FakeOpener.open.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FakeOpener.open.__dict__.__setitem__('stypy_type_store', module_type_store)
        FakeOpener.open.__dict__.__setitem__('stypy_function_name', 'FakeOpener.open')
        FakeOpener.open.__dict__.__setitem__('stypy_param_names_list', ['req'])
        FakeOpener.open.__dict__.__setitem__('stypy_varargs_param_name', None)
        FakeOpener.open.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FakeOpener.open.__dict__.__setitem__('stypy_call_defaults', defaults)
        FakeOpener.open.__dict__.__setitem__('stypy_call_varargs', varargs)
        FakeOpener.open.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FakeOpener.open.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FakeOpener.open', ['req'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'open', localization, ['req'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'open(...)' code ##################

        
        # Call to append(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'req' (line 63)
        req_41811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 25), 'req', False)
        # Processing the call keyword arguments (line 63)
        kwargs_41812 = {}
        # Getting the type of 'self' (line 63)
        self_41808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self', False)
        # Obtaining the member 'reqs' of a type (line 63)
        reqs_41809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_41808, 'reqs')
        # Obtaining the member 'append' of a type (line 63)
        append_41810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), reqs_41809, 'append')
        # Calling append(args, kwargs) (line 63)
        append_call_result_41813 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), append_41810, *[req_41811], **kwargs_41812)
        
        # Getting the type of 'self' (line 64)
        self_41814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'stypy_return_type', self_41814)
        
        # ################# End of 'open(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'open' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_41815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_41815)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'open'
        return stypy_return_type_41815


    @norecursion
    def read(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'read'
        module_type_store = module_type_store.open_function_context('read', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FakeOpener.read.__dict__.__setitem__('stypy_localization', localization)
        FakeOpener.read.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FakeOpener.read.__dict__.__setitem__('stypy_type_store', module_type_store)
        FakeOpener.read.__dict__.__setitem__('stypy_function_name', 'FakeOpener.read')
        FakeOpener.read.__dict__.__setitem__('stypy_param_names_list', [])
        FakeOpener.read.__dict__.__setitem__('stypy_varargs_param_name', None)
        FakeOpener.read.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FakeOpener.read.__dict__.__setitem__('stypy_call_defaults', defaults)
        FakeOpener.read.__dict__.__setitem__('stypy_call_varargs', varargs)
        FakeOpener.read.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FakeOpener.read.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FakeOpener.read', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'read', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'read(...)' code ##################

        str_41816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 15), 'str', 'xxx')
        # Assigning a type to the variable 'stypy_return_type' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'stypy_return_type', str_41816)
        
        # ################# End of 'read(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_41817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_41817)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read'
        return stypy_return_type_41817


# Assigning a type to the variable 'FakeOpener' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'FakeOpener', FakeOpener)
# Declaration of the 'RegisterTestCase' class
# Getting the type of 'PyPIRCCommandTestCase' (line 69)
PyPIRCCommandTestCase_41818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 23), 'PyPIRCCommandTestCase')

class RegisterTestCase(PyPIRCCommandTestCase_41818, ):

    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 71, 4, False)
        # Assigning a type to the variable 'self' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RegisterTestCase.setUp.__dict__.__setitem__('stypy_localization', localization)
        RegisterTestCase.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RegisterTestCase.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        RegisterTestCase.setUp.__dict__.__setitem__('stypy_function_name', 'RegisterTestCase.setUp')
        RegisterTestCase.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        RegisterTestCase.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        RegisterTestCase.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RegisterTestCase.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        RegisterTestCase.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        RegisterTestCase.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RegisterTestCase.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RegisterTestCase.setUp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setUp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setUp(...)' code ##################

        
        # Call to setUp(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_41825 = {}
        
        # Call to super(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'RegisterTestCase' (line 72)
        RegisterTestCase_41820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 14), 'RegisterTestCase', False)
        # Getting the type of 'self' (line 72)
        self_41821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 32), 'self', False)
        # Processing the call keyword arguments (line 72)
        kwargs_41822 = {}
        # Getting the type of 'super' (line 72)
        super_41819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'super', False)
        # Calling super(args, kwargs) (line 72)
        super_call_result_41823 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), super_41819, *[RegisterTestCase_41820, self_41821], **kwargs_41822)
        
        # Obtaining the member 'setUp' of a type (line 72)
        setUp_41824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), super_call_result_41823, 'setUp')
        # Calling setUp(args, kwargs) (line 72)
        setUp_call_result_41826 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), setUp_41824, *[], **kwargs_41825)
        
        
        # Assigning a Attribute to a Attribute (line 74):
        
        # Assigning a Attribute to a Attribute (line 74):
        # Getting the type of 'getpass' (line 74)
        getpass_41827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'getpass')
        # Obtaining the member 'getpass' of a type (line 74)
        getpass_41828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 28), getpass_41827, 'getpass')
        # Getting the type of 'self' (line 74)
        self_41829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self')
        # Setting the type of the member '_old_getpass' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_41829, '_old_getpass', getpass_41828)

        @norecursion
        def _getpass(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_getpass'
            module_type_store = module_type_store.open_function_context('_getpass', 75, 8, False)
            
            # Passed parameters checking function
            _getpass.stypy_localization = localization
            _getpass.stypy_type_of_self = None
            _getpass.stypy_type_store = module_type_store
            _getpass.stypy_function_name = '_getpass'
            _getpass.stypy_param_names_list = ['prompt']
            _getpass.stypy_varargs_param_name = None
            _getpass.stypy_kwargs_param_name = None
            _getpass.stypy_call_defaults = defaults
            _getpass.stypy_call_varargs = varargs
            _getpass.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_getpass', ['prompt'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_getpass', localization, ['prompt'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_getpass(...)' code ##################

            str_41830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 19), 'str', 'password')
            # Assigning a type to the variable 'stypy_return_type' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'stypy_return_type', str_41830)
            
            # ################# End of '_getpass(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_getpass' in the type store
            # Getting the type of 'stypy_return_type' (line 75)
            stypy_return_type_41831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_41831)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_getpass'
            return stypy_return_type_41831

        # Assigning a type to the variable '_getpass' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), '_getpass', _getpass)
        
        # Assigning a Name to a Attribute (line 77):
        
        # Assigning a Name to a Attribute (line 77):
        # Getting the type of '_getpass' (line 77)
        _getpass_41832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 26), '_getpass')
        # Getting the type of 'getpass' (line 77)
        getpass_41833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'getpass')
        # Setting the type of the member 'getpass' of a type (line 77)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), getpass_41833, 'getpass', _getpass_41832)
        
        # Assigning a Attribute to a Attribute (line 78):
        
        # Assigning a Attribute to a Attribute (line 78):
        # Getting the type of 'urllib2' (line 78)
        urllib2_41834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'urllib2')
        # Obtaining the member 'build_opener' of a type (line 78)
        build_opener_41835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 26), urllib2_41834, 'build_opener')
        # Getting the type of 'self' (line 78)
        self_41836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self')
        # Setting the type of the member 'old_opener' of a type (line 78)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_41836, 'old_opener', build_opener_41835)
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Call to a Attribute (line 79):
        
        # Call to FakeOpener(...): (line 79)
        # Processing the call keyword arguments (line 79)
        kwargs_41838 = {}
        # Getting the type of 'FakeOpener' (line 79)
        FakeOpener_41837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 43), 'FakeOpener', False)
        # Calling FakeOpener(args, kwargs) (line 79)
        FakeOpener_call_result_41839 = invoke(stypy.reporting.localization.Localization(__file__, 79, 43), FakeOpener_41837, *[], **kwargs_41838)
        
        # Getting the type of 'urllib2' (line 79)
        urllib2_41840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'urllib2')
        # Setting the type of the member 'build_opener' of a type (line 79)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 20), urllib2_41840, 'build_opener', FakeOpener_call_result_41839)
        
        # Assigning a Attribute to a Attribute (line 79):
        # Getting the type of 'urllib2' (line 79)
        urllib2_41841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'urllib2')
        # Obtaining the member 'build_opener' of a type (line 79)
        build_opener_41842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 20), urllib2_41841, 'build_opener')
        # Getting the type of 'self' (line 79)
        self_41843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self')
        # Setting the type of the member 'conn' of a type (line 79)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), self_41843, 'conn', build_opener_41842)
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 71)
        stypy_return_type_41844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_41844)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_41844


    @norecursion
    def tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDown'
        module_type_store = module_type_store.open_function_context('tearDown', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RegisterTestCase.tearDown.__dict__.__setitem__('stypy_localization', localization)
        RegisterTestCase.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RegisterTestCase.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        RegisterTestCase.tearDown.__dict__.__setitem__('stypy_function_name', 'RegisterTestCase.tearDown')
        RegisterTestCase.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        RegisterTestCase.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        RegisterTestCase.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RegisterTestCase.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        RegisterTestCase.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        RegisterTestCase.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RegisterTestCase.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RegisterTestCase.tearDown', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tearDown', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tearDown(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 82):
        
        # Assigning a Attribute to a Attribute (line 82):
        # Getting the type of 'self' (line 82)
        self_41845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'self')
        # Obtaining the member '_old_getpass' of a type (line 82)
        _old_getpass_41846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 26), self_41845, '_old_getpass')
        # Getting the type of 'getpass' (line 82)
        getpass_41847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'getpass')
        # Setting the type of the member 'getpass' of a type (line 82)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), getpass_41847, 'getpass', _old_getpass_41846)
        
        # Assigning a Attribute to a Attribute (line 83):
        
        # Assigning a Attribute to a Attribute (line 83):
        # Getting the type of 'self' (line 83)
        self_41848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 31), 'self')
        # Obtaining the member 'old_opener' of a type (line 83)
        old_opener_41849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 31), self_41848, 'old_opener')
        # Getting the type of 'urllib2' (line 83)
        urllib2_41850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'urllib2')
        # Setting the type of the member 'build_opener' of a type (line 83)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), urllib2_41850, 'build_opener', old_opener_41849)
        
        # Call to tearDown(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_41857 = {}
        
        # Call to super(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'RegisterTestCase' (line 84)
        RegisterTestCase_41852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 14), 'RegisterTestCase', False)
        # Getting the type of 'self' (line 84)
        self_41853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 32), 'self', False)
        # Processing the call keyword arguments (line 84)
        kwargs_41854 = {}
        # Getting the type of 'super' (line 84)
        super_41851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'super', False)
        # Calling super(args, kwargs) (line 84)
        super_call_result_41855 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), super_41851, *[RegisterTestCase_41852, self_41853], **kwargs_41854)
        
        # Obtaining the member 'tearDown' of a type (line 84)
        tearDown_41856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), super_call_result_41855, 'tearDown')
        # Calling tearDown(args, kwargs) (line 84)
        tearDown_call_result_41858 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), tearDown_41856, *[], **kwargs_41857)
        
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_41859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_41859)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_41859


    @norecursion
    def _get_cmd(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 86)
        None_41860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 32), 'None')
        defaults = [None_41860]
        # Create a new context for function '_get_cmd'
        module_type_store = module_type_store.open_function_context('_get_cmd', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RegisterTestCase._get_cmd.__dict__.__setitem__('stypy_localization', localization)
        RegisterTestCase._get_cmd.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RegisterTestCase._get_cmd.__dict__.__setitem__('stypy_type_store', module_type_store)
        RegisterTestCase._get_cmd.__dict__.__setitem__('stypy_function_name', 'RegisterTestCase._get_cmd')
        RegisterTestCase._get_cmd.__dict__.__setitem__('stypy_param_names_list', ['metadata'])
        RegisterTestCase._get_cmd.__dict__.__setitem__('stypy_varargs_param_name', None)
        RegisterTestCase._get_cmd.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RegisterTestCase._get_cmd.__dict__.__setitem__('stypy_call_defaults', defaults)
        RegisterTestCase._get_cmd.__dict__.__setitem__('stypy_call_varargs', varargs)
        RegisterTestCase._get_cmd.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RegisterTestCase._get_cmd.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RegisterTestCase._get_cmd', ['metadata'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_cmd', localization, ['metadata'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_cmd(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 87)
        # Getting the type of 'metadata' (line 87)
        metadata_41861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 11), 'metadata')
        # Getting the type of 'None' (line 87)
        None_41862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'None')
        
        (may_be_41863, more_types_in_union_41864) = may_be_none(metadata_41861, None_41862)

        if may_be_41863:

            if more_types_in_union_41864:
                # Runtime conditional SSA (line 87)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Dict to a Name (line 88):
            
            # Assigning a Dict to a Name (line 88):
            
            # Obtaining an instance of the builtin type 'dict' (line 88)
            dict_41865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 23), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 88)
            # Adding element type (key, value) (line 88)
            str_41866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 24), 'str', 'url')
            str_41867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 31), 'str', 'xxx')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 23), dict_41865, (str_41866, str_41867))
            # Adding element type (key, value) (line 88)
            str_41868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 38), 'str', 'author')
            str_41869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 48), 'str', 'xxx')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 23), dict_41865, (str_41868, str_41869))
            # Adding element type (key, value) (line 88)
            str_41870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 24), 'str', 'author_email')
            str_41871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 40), 'str', 'xxx')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 23), dict_41865, (str_41870, str_41871))
            # Adding element type (key, value) (line 88)
            str_41872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 24), 'str', 'name')
            str_41873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 32), 'str', 'xxx')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 23), dict_41865, (str_41872, str_41873))
            # Adding element type (key, value) (line 88)
            str_41874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 39), 'str', 'version')
            str_41875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 50), 'str', 'xxx')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 23), dict_41865, (str_41874, str_41875))
            
            # Assigning a type to the variable 'metadata' (line 88)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'metadata', dict_41865)

            if more_types_in_union_41864:
                # SSA join for if statement (line 87)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Tuple (line 91):
        
        # Assigning a Subscript to a Name (line 91):
        
        # Obtaining the type of the subscript
        int_41876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'int')
        
        # Call to create_dist(...): (line 91)
        # Processing the call keyword arguments (line 91)
        # Getting the type of 'metadata' (line 91)
        metadata_41879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 44), 'metadata', False)
        kwargs_41880 = {'metadata_41879': metadata_41879}
        # Getting the type of 'self' (line 91)
        self_41877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 25), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 91)
        create_dist_41878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 25), self_41877, 'create_dist')
        # Calling create_dist(args, kwargs) (line 91)
        create_dist_call_result_41881 = invoke(stypy.reporting.localization.Localization(__file__, 91, 25), create_dist_41878, *[], **kwargs_41880)
        
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___41882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), create_dist_call_result_41881, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 91)
        subscript_call_result_41883 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), getitem___41882, int_41876)
        
        # Assigning a type to the variable 'tuple_var_assignment_41767' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tuple_var_assignment_41767', subscript_call_result_41883)
        
        # Assigning a Subscript to a Name (line 91):
        
        # Obtaining the type of the subscript
        int_41884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'int')
        
        # Call to create_dist(...): (line 91)
        # Processing the call keyword arguments (line 91)
        # Getting the type of 'metadata' (line 91)
        metadata_41887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 44), 'metadata', False)
        kwargs_41888 = {'metadata_41887': metadata_41887}
        # Getting the type of 'self' (line 91)
        self_41885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 25), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 91)
        create_dist_41886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 25), self_41885, 'create_dist')
        # Calling create_dist(args, kwargs) (line 91)
        create_dist_call_result_41889 = invoke(stypy.reporting.localization.Localization(__file__, 91, 25), create_dist_41886, *[], **kwargs_41888)
        
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___41890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), create_dist_call_result_41889, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 91)
        subscript_call_result_41891 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), getitem___41890, int_41884)
        
        # Assigning a type to the variable 'tuple_var_assignment_41768' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tuple_var_assignment_41768', subscript_call_result_41891)
        
        # Assigning a Name to a Name (line 91):
        # Getting the type of 'tuple_var_assignment_41767' (line 91)
        tuple_var_assignment_41767_41892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tuple_var_assignment_41767')
        # Assigning a type to the variable 'pkg_info' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'pkg_info', tuple_var_assignment_41767_41892)
        
        # Assigning a Name to a Name (line 91):
        # Getting the type of 'tuple_var_assignment_41768' (line 91)
        tuple_var_assignment_41768_41893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tuple_var_assignment_41768')
        # Assigning a type to the variable 'dist' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 18), 'dist', tuple_var_assignment_41768_41893)
        
        # Call to register(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'dist' (line 92)
        dist_41895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'dist', False)
        # Processing the call keyword arguments (line 92)
        kwargs_41896 = {}
        # Getting the type of 'register' (line 92)
        register_41894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'register', False)
        # Calling register(args, kwargs) (line 92)
        register_call_result_41897 = invoke(stypy.reporting.localization.Localization(__file__, 92, 15), register_41894, *[dist_41895], **kwargs_41896)
        
        # Assigning a type to the variable 'stypy_return_type' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'stypy_return_type', register_call_result_41897)
        
        # ################# End of '_get_cmd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_cmd' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_41898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_41898)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_cmd'
        return stypy_return_type_41898


    @norecursion
    def test_create_pypirc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_create_pypirc'
        module_type_store = module_type_store.open_function_context('test_create_pypirc', 94, 4, False)
        # Assigning a type to the variable 'self' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RegisterTestCase.test_create_pypirc.__dict__.__setitem__('stypy_localization', localization)
        RegisterTestCase.test_create_pypirc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RegisterTestCase.test_create_pypirc.__dict__.__setitem__('stypy_type_store', module_type_store)
        RegisterTestCase.test_create_pypirc.__dict__.__setitem__('stypy_function_name', 'RegisterTestCase.test_create_pypirc')
        RegisterTestCase.test_create_pypirc.__dict__.__setitem__('stypy_param_names_list', [])
        RegisterTestCase.test_create_pypirc.__dict__.__setitem__('stypy_varargs_param_name', None)
        RegisterTestCase.test_create_pypirc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RegisterTestCase.test_create_pypirc.__dict__.__setitem__('stypy_call_defaults', defaults)
        RegisterTestCase.test_create_pypirc.__dict__.__setitem__('stypy_call_varargs', varargs)
        RegisterTestCase.test_create_pypirc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RegisterTestCase.test_create_pypirc.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RegisterTestCase.test_create_pypirc', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_create_pypirc', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_create_pypirc(...)' code ##################

        
        # Assigning a Call to a Name (line 99):
        
        # Assigning a Call to a Name (line 99):
        
        # Call to _get_cmd(...): (line 99)
        # Processing the call keyword arguments (line 99)
        kwargs_41901 = {}
        # Getting the type of 'self' (line 99)
        self_41899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 14), 'self', False)
        # Obtaining the member '_get_cmd' of a type (line 99)
        _get_cmd_41900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 14), self_41899, '_get_cmd')
        # Calling _get_cmd(args, kwargs) (line 99)
        _get_cmd_call_result_41902 = invoke(stypy.reporting.localization.Localization(__file__, 99, 14), _get_cmd_41900, *[], **kwargs_41901)
        
        # Assigning a type to the variable 'cmd' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'cmd', _get_cmd_call_result_41902)
        
        # Call to assertFalse(...): (line 102)
        # Processing the call arguments (line 102)
        
        # Call to exists(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'self' (line 102)
        self_41908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 40), 'self', False)
        # Obtaining the member 'rc' of a type (line 102)
        rc_41909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 40), self_41908, 'rc')
        # Processing the call keyword arguments (line 102)
        kwargs_41910 = {}
        # Getting the type of 'os' (line 102)
        os_41905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 102)
        path_41906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 25), os_41905, 'path')
        # Obtaining the member 'exists' of a type (line 102)
        exists_41907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 25), path_41906, 'exists')
        # Calling exists(args, kwargs) (line 102)
        exists_call_result_41911 = invoke(stypy.reporting.localization.Localization(__file__, 102, 25), exists_41907, *[rc_41909], **kwargs_41910)
        
        # Processing the call keyword arguments (line 102)
        kwargs_41912 = {}
        # Getting the type of 'self' (line 102)
        self_41903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 102)
        assertFalse_41904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), self_41903, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 102)
        assertFalse_call_result_41913 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), assertFalse_41904, *[exists_call_result_41911], **kwargs_41912)
        
        
        # Assigning a Call to a Name (line 112):
        
        # Assigning a Call to a Name (line 112):
        
        # Call to RawInputs(...): (line 112)
        # Processing the call arguments (line 112)
        str_41915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 27), 'str', '1')
        str_41916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 32), 'str', 'tarek')
        str_41917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 41), 'str', 'y')
        # Processing the call keyword arguments (line 112)
        kwargs_41918 = {}
        # Getting the type of 'RawInputs' (line 112)
        RawInputs_41914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 17), 'RawInputs', False)
        # Calling RawInputs(args, kwargs) (line 112)
        RawInputs_call_result_41919 = invoke(stypy.reporting.localization.Localization(__file__, 112, 17), RawInputs_41914, *[str_41915, str_41916, str_41917], **kwargs_41918)
        
        # Assigning a type to the variable 'inputs' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'inputs', RawInputs_call_result_41919)
        
        # Assigning a Attribute to a Attribute (line 113):
        
        # Assigning a Attribute to a Attribute (line 113):
        # Getting the type of 'inputs' (line 113)
        inputs_41920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 36), 'inputs')
        # Obtaining the member '__call__' of a type (line 113)
        call___41921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 36), inputs_41920, '__call__')
        # Getting the type of 'register_module' (line 113)
        register_module_41922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'register_module')
        # Setting the type of the member 'raw_input' of a type (line 113)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), register_module_41922, 'raw_input', call___41921)
        
        # Try-finally block (line 115)
        
        # Call to run(...): (line 116)
        # Processing the call keyword arguments (line 116)
        kwargs_41925 = {}
        # Getting the type of 'cmd' (line 116)
        cmd_41923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'cmd', False)
        # Obtaining the member 'run' of a type (line 116)
        run_41924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 12), cmd_41923, 'run')
        # Calling run(args, kwargs) (line 116)
        run_call_result_41926 = invoke(stypy.reporting.localization.Localization(__file__, 116, 12), run_41924, *[], **kwargs_41925)
        
        
        # finally branch of the try-finally block (line 115)
        # Deleting a member
        # Getting the type of 'register_module' (line 118)
        register_module_41927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'register_module')
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 118, 12), register_module_41927, 'raw_input')
        
        
        # Call to assertTrue(...): (line 121)
        # Processing the call arguments (line 121)
        
        # Call to exists(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'self' (line 121)
        self_41933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 39), 'self', False)
        # Obtaining the member 'rc' of a type (line 121)
        rc_41934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 39), self_41933, 'rc')
        # Processing the call keyword arguments (line 121)
        kwargs_41935 = {}
        # Getting the type of 'os' (line 121)
        os_41930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 121)
        path_41931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 24), os_41930, 'path')
        # Obtaining the member 'exists' of a type (line 121)
        exists_41932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 24), path_41931, 'exists')
        # Calling exists(args, kwargs) (line 121)
        exists_call_result_41936 = invoke(stypy.reporting.localization.Localization(__file__, 121, 24), exists_41932, *[rc_41934], **kwargs_41935)
        
        # Processing the call keyword arguments (line 121)
        kwargs_41937 = {}
        # Getting the type of 'self' (line 121)
        self_41928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 121)
        assertTrue_41929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), self_41928, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 121)
        assertTrue_call_result_41938 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), assertTrue_41929, *[exists_call_result_41936], **kwargs_41937)
        
        
        # Assigning a Call to a Name (line 124):
        
        # Assigning a Call to a Name (line 124):
        
        # Call to open(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'self' (line 124)
        self_41940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 17), 'self', False)
        # Obtaining the member 'rc' of a type (line 124)
        rc_41941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 17), self_41940, 'rc')
        # Processing the call keyword arguments (line 124)
        kwargs_41942 = {}
        # Getting the type of 'open' (line 124)
        open_41939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'open', False)
        # Calling open(args, kwargs) (line 124)
        open_call_result_41943 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), open_41939, *[rc_41941], **kwargs_41942)
        
        # Assigning a type to the variable 'f' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'f', open_call_result_41943)
        
        # Try-finally block (line 125)
        
        # Assigning a Call to a Name (line 126):
        
        # Assigning a Call to a Name (line 126):
        
        # Call to read(...): (line 126)
        # Processing the call keyword arguments (line 126)
        kwargs_41946 = {}
        # Getting the type of 'f' (line 126)
        f_41944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 22), 'f', False)
        # Obtaining the member 'read' of a type (line 126)
        read_41945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 22), f_41944, 'read')
        # Calling read(args, kwargs) (line 126)
        read_call_result_41947 = invoke(stypy.reporting.localization.Localization(__file__, 126, 22), read_41945, *[], **kwargs_41946)
        
        # Assigning a type to the variable 'content' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'content', read_call_result_41947)
        
        # Call to assertEqual(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'content' (line 127)
        content_41950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 29), 'content', False)
        # Getting the type of 'WANTED_PYPIRC' (line 127)
        WANTED_PYPIRC_41951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 38), 'WANTED_PYPIRC', False)
        # Processing the call keyword arguments (line 127)
        kwargs_41952 = {}
        # Getting the type of 'self' (line 127)
        self_41948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 127)
        assertEqual_41949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), self_41948, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 127)
        assertEqual_call_result_41953 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), assertEqual_41949, *[content_41950, WANTED_PYPIRC_41951], **kwargs_41952)
        
        
        # finally branch of the try-finally block (line 125)
        
        # Call to close(...): (line 129)
        # Processing the call keyword arguments (line 129)
        kwargs_41956 = {}
        # Getting the type of 'f' (line 129)
        f_41954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 129)
        close_41955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), f_41954, 'close')
        # Calling close(args, kwargs) (line 129)
        close_call_result_41957 = invoke(stypy.reporting.localization.Localization(__file__, 129, 12), close_41955, *[], **kwargs_41956)
        
        

        @norecursion
        def _no_way(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            str_41958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 27), 'str', '')
            defaults = [str_41958]
            # Create a new context for function '_no_way'
            module_type_store = module_type_store.open_function_context('_no_way', 134, 8, False)
            
            # Passed parameters checking function
            _no_way.stypy_localization = localization
            _no_way.stypy_type_of_self = None
            _no_way.stypy_type_store = module_type_store
            _no_way.stypy_function_name = '_no_way'
            _no_way.stypy_param_names_list = ['prompt']
            _no_way.stypy_varargs_param_name = None
            _no_way.stypy_kwargs_param_name = None
            _no_way.stypy_call_defaults = defaults
            _no_way.stypy_call_varargs = varargs
            _no_way.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_no_way', ['prompt'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_no_way', localization, ['prompt'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_no_way(...)' code ##################

            
            # Call to AssertionError(...): (line 135)
            # Processing the call arguments (line 135)
            # Getting the type of 'prompt' (line 135)
            prompt_41960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 33), 'prompt', False)
            # Processing the call keyword arguments (line 135)
            kwargs_41961 = {}
            # Getting the type of 'AssertionError' (line 135)
            AssertionError_41959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 18), 'AssertionError', False)
            # Calling AssertionError(args, kwargs) (line 135)
            AssertionError_call_result_41962 = invoke(stypy.reporting.localization.Localization(__file__, 135, 18), AssertionError_41959, *[prompt_41960], **kwargs_41961)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 135, 12), AssertionError_call_result_41962, 'raise parameter', BaseException)
            
            # ################# End of '_no_way(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_no_way' in the type store
            # Getting the type of 'stypy_return_type' (line 134)
            stypy_return_type_41963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_41963)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_no_way'
            return stypy_return_type_41963

        # Assigning a type to the variable '_no_way' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), '_no_way', _no_way)
        
        # Assigning a Name to a Attribute (line 136):
        
        # Assigning a Name to a Attribute (line 136):
        # Getting the type of '_no_way' (line 136)
        _no_way_41964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 36), '_no_way')
        # Getting the type of 'register_module' (line 136)
        register_module_41965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'register_module')
        # Setting the type of the member 'raw_input' of a type (line 136)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), register_module_41965, 'raw_input', _no_way_41964)
        
        # Assigning a Num to a Attribute (line 138):
        
        # Assigning a Num to a Attribute (line 138):
        int_41966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 28), 'int')
        # Getting the type of 'cmd' (line 138)
        cmd_41967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'cmd')
        # Setting the type of the member 'show_response' of a type (line 138)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), cmd_41967, 'show_response', int_41966)
        
        # Call to run(...): (line 139)
        # Processing the call keyword arguments (line 139)
        kwargs_41970 = {}
        # Getting the type of 'cmd' (line 139)
        cmd_41968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'cmd', False)
        # Obtaining the member 'run' of a type (line 139)
        run_41969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), cmd_41968, 'run')
        # Calling run(args, kwargs) (line 139)
        run_call_result_41971 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), run_41969, *[], **kwargs_41970)
        
        
        # Call to assertEqual(...): (line 143)
        # Processing the call arguments (line 143)
        
        # Call to len(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'self' (line 143)
        self_41975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 29), 'self', False)
        # Obtaining the member 'conn' of a type (line 143)
        conn_41976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 29), self_41975, 'conn')
        # Obtaining the member 'reqs' of a type (line 143)
        reqs_41977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 29), conn_41976, 'reqs')
        # Processing the call keyword arguments (line 143)
        kwargs_41978 = {}
        # Getting the type of 'len' (line 143)
        len_41974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 25), 'len', False)
        # Calling len(args, kwargs) (line 143)
        len_call_result_41979 = invoke(stypy.reporting.localization.Localization(__file__, 143, 25), len_41974, *[reqs_41977], **kwargs_41978)
        
        int_41980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 46), 'int')
        # Processing the call keyword arguments (line 143)
        kwargs_41981 = {}
        # Getting the type of 'self' (line 143)
        self_41972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 143)
        assertEqual_41973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), self_41972, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 143)
        assertEqual_call_result_41982 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), assertEqual_41973, *[len_call_result_41979, int_41980], **kwargs_41981)
        
        
        # Assigning a Call to a Name (line 144):
        
        # Assigning a Call to a Name (line 144):
        
        # Call to dict(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Obtaining the type of the subscript
        int_41984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 35), 'int')
        # Getting the type of 'self' (line 144)
        self_41985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'self', False)
        # Obtaining the member 'conn' of a type (line 144)
        conn_41986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 20), self_41985, 'conn')
        # Obtaining the member 'reqs' of a type (line 144)
        reqs_41987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 20), conn_41986, 'reqs')
        # Obtaining the member '__getitem__' of a type (line 144)
        getitem___41988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 20), reqs_41987, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 144)
        subscript_call_result_41989 = invoke(stypy.reporting.localization.Localization(__file__, 144, 20), getitem___41988, int_41984)
        
        # Obtaining the member 'headers' of a type (line 144)
        headers_41990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 20), subscript_call_result_41989, 'headers')
        # Processing the call keyword arguments (line 144)
        kwargs_41991 = {}
        # Getting the type of 'dict' (line 144)
        dict_41983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 15), 'dict', False)
        # Calling dict(args, kwargs) (line 144)
        dict_call_result_41992 = invoke(stypy.reporting.localization.Localization(__file__, 144, 15), dict_41983, *[headers_41990], **kwargs_41991)
        
        # Assigning a type to the variable 'req1' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'req1', dict_call_result_41992)
        
        # Assigning a Call to a Name (line 145):
        
        # Assigning a Call to a Name (line 145):
        
        # Call to dict(...): (line 145)
        # Processing the call arguments (line 145)
        
        # Obtaining the type of the subscript
        int_41994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 35), 'int')
        # Getting the type of 'self' (line 145)
        self_41995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 20), 'self', False)
        # Obtaining the member 'conn' of a type (line 145)
        conn_41996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 20), self_41995, 'conn')
        # Obtaining the member 'reqs' of a type (line 145)
        reqs_41997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 20), conn_41996, 'reqs')
        # Obtaining the member '__getitem__' of a type (line 145)
        getitem___41998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 20), reqs_41997, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 145)
        subscript_call_result_41999 = invoke(stypy.reporting.localization.Localization(__file__, 145, 20), getitem___41998, int_41994)
        
        # Obtaining the member 'headers' of a type (line 145)
        headers_42000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 20), subscript_call_result_41999, 'headers')
        # Processing the call keyword arguments (line 145)
        kwargs_42001 = {}
        # Getting the type of 'dict' (line 145)
        dict_41993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 15), 'dict', False)
        # Calling dict(args, kwargs) (line 145)
        dict_call_result_42002 = invoke(stypy.reporting.localization.Localization(__file__, 145, 15), dict_41993, *[headers_42000], **kwargs_42001)
        
        # Assigning a type to the variable 'req2' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'req2', dict_call_result_42002)
        
        # Call to assertEqual(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Obtaining the type of the subscript
        str_42005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 30), 'str', 'Content-length')
        # Getting the type of 'req2' (line 146)
        req2_42006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 25), 'req2', False)
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___42007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 25), req2_42006, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_42008 = invoke(stypy.reporting.localization.Localization(__file__, 146, 25), getitem___42007, str_42005)
        
        
        # Obtaining the type of the subscript
        str_42009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 54), 'str', 'Content-length')
        # Getting the type of 'req1' (line 146)
        req1_42010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 49), 'req1', False)
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___42011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 49), req1_42010, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_42012 = invoke(stypy.reporting.localization.Localization(__file__, 146, 49), getitem___42011, str_42009)
        
        # Processing the call keyword arguments (line 146)
        kwargs_42013 = {}
        # Getting the type of 'self' (line 146)
        self_42003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 146)
        assertEqual_42004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), self_42003, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 146)
        assertEqual_call_result_42014 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), assertEqual_42004, *[subscript_call_result_42008, subscript_call_result_42012], **kwargs_42013)
        
        
        # Call to assertIn(...): (line 147)
        # Processing the call arguments (line 147)
        str_42017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 22), 'str', 'xxx')
        
        # Obtaining the type of the subscript
        int_42018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 44), 'int')
        # Getting the type of 'self' (line 147)
        self_42019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 29), 'self', False)
        # Obtaining the member 'conn' of a type (line 147)
        conn_42020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 29), self_42019, 'conn')
        # Obtaining the member 'reqs' of a type (line 147)
        reqs_42021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 29), conn_42020, 'reqs')
        # Obtaining the member '__getitem__' of a type (line 147)
        getitem___42022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 29), reqs_42021, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 147)
        subscript_call_result_42023 = invoke(stypy.reporting.localization.Localization(__file__, 147, 29), getitem___42022, int_42018)
        
        # Obtaining the member 'data' of a type (line 147)
        data_42024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 29), subscript_call_result_42023, 'data')
        # Processing the call keyword arguments (line 147)
        kwargs_42025 = {}
        # Getting the type of 'self' (line 147)
        self_42015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 147)
        assertIn_42016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), self_42015, 'assertIn')
        # Calling assertIn(args, kwargs) (line 147)
        assertIn_call_result_42026 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), assertIn_42016, *[str_42017, data_42024], **kwargs_42025)
        
        
        # ################# End of 'test_create_pypirc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_create_pypirc' in the type store
        # Getting the type of 'stypy_return_type' (line 94)
        stypy_return_type_42027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42027)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_create_pypirc'
        return stypy_return_type_42027


    @norecursion
    def test_password_not_in_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_password_not_in_file'
        module_type_store = module_type_store.open_function_context('test_password_not_in_file', 149, 4, False)
        # Assigning a type to the variable 'self' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RegisterTestCase.test_password_not_in_file.__dict__.__setitem__('stypy_localization', localization)
        RegisterTestCase.test_password_not_in_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RegisterTestCase.test_password_not_in_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        RegisterTestCase.test_password_not_in_file.__dict__.__setitem__('stypy_function_name', 'RegisterTestCase.test_password_not_in_file')
        RegisterTestCase.test_password_not_in_file.__dict__.__setitem__('stypy_param_names_list', [])
        RegisterTestCase.test_password_not_in_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        RegisterTestCase.test_password_not_in_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RegisterTestCase.test_password_not_in_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        RegisterTestCase.test_password_not_in_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        RegisterTestCase.test_password_not_in_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RegisterTestCase.test_password_not_in_file.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RegisterTestCase.test_password_not_in_file', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_password_not_in_file', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_password_not_in_file(...)' code ##################

        
        # Call to write_file(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'self' (line 151)
        self_42030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 24), 'self', False)
        # Obtaining the member 'rc' of a type (line 151)
        rc_42031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 24), self_42030, 'rc')
        # Getting the type of 'PYPIRC_NOPASSWORD' (line 151)
        PYPIRC_NOPASSWORD_42032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 33), 'PYPIRC_NOPASSWORD', False)
        # Processing the call keyword arguments (line 151)
        kwargs_42033 = {}
        # Getting the type of 'self' (line 151)
        self_42028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 151)
        write_file_42029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), self_42028, 'write_file')
        # Calling write_file(args, kwargs) (line 151)
        write_file_call_result_42034 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), write_file_42029, *[rc_42031, PYPIRC_NOPASSWORD_42032], **kwargs_42033)
        
        
        # Assigning a Call to a Name (line 152):
        
        # Assigning a Call to a Name (line 152):
        
        # Call to _get_cmd(...): (line 152)
        # Processing the call keyword arguments (line 152)
        kwargs_42037 = {}
        # Getting the type of 'self' (line 152)
        self_42035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 14), 'self', False)
        # Obtaining the member '_get_cmd' of a type (line 152)
        _get_cmd_42036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 14), self_42035, '_get_cmd')
        # Calling _get_cmd(args, kwargs) (line 152)
        _get_cmd_call_result_42038 = invoke(stypy.reporting.localization.Localization(__file__, 152, 14), _get_cmd_42036, *[], **kwargs_42037)
        
        # Assigning a type to the variable 'cmd' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'cmd', _get_cmd_call_result_42038)
        
        # Call to _set_config(...): (line 153)
        # Processing the call keyword arguments (line 153)
        kwargs_42041 = {}
        # Getting the type of 'cmd' (line 153)
        cmd_42039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'cmd', False)
        # Obtaining the member '_set_config' of a type (line 153)
        _set_config_42040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), cmd_42039, '_set_config')
        # Calling _set_config(args, kwargs) (line 153)
        _set_config_call_result_42042 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), _set_config_42040, *[], **kwargs_42041)
        
        
        # Call to finalize_options(...): (line 154)
        # Processing the call keyword arguments (line 154)
        kwargs_42045 = {}
        # Getting the type of 'cmd' (line 154)
        cmd_42043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 154)
        finalize_options_42044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), cmd_42043, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 154)
        finalize_options_call_result_42046 = invoke(stypy.reporting.localization.Localization(__file__, 154, 8), finalize_options_42044, *[], **kwargs_42045)
        
        
        # Call to send_metadata(...): (line 155)
        # Processing the call keyword arguments (line 155)
        kwargs_42049 = {}
        # Getting the type of 'cmd' (line 155)
        cmd_42047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'cmd', False)
        # Obtaining the member 'send_metadata' of a type (line 155)
        send_metadata_42048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), cmd_42047, 'send_metadata')
        # Calling send_metadata(args, kwargs) (line 155)
        send_metadata_call_result_42050 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), send_metadata_42048, *[], **kwargs_42049)
        
        
        # Call to assertEqual(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'cmd' (line 159)
        cmd_42053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 25), 'cmd', False)
        # Obtaining the member 'distribution' of a type (line 159)
        distribution_42054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 25), cmd_42053, 'distribution')
        # Obtaining the member 'password' of a type (line 159)
        password_42055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 25), distribution_42054, 'password')
        str_42056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 52), 'str', 'password')
        # Processing the call keyword arguments (line 159)
        kwargs_42057 = {}
        # Getting the type of 'self' (line 159)
        self_42051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 159)
        assertEqual_42052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), self_42051, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 159)
        assertEqual_call_result_42058 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), assertEqual_42052, *[password_42055, str_42056], **kwargs_42057)
        
        
        # ################# End of 'test_password_not_in_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_password_not_in_file' in the type store
        # Getting the type of 'stypy_return_type' (line 149)
        stypy_return_type_42059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42059)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_password_not_in_file'
        return stypy_return_type_42059


    @norecursion
    def test_registering(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_registering'
        module_type_store = module_type_store.open_function_context('test_registering', 161, 4, False)
        # Assigning a type to the variable 'self' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RegisterTestCase.test_registering.__dict__.__setitem__('stypy_localization', localization)
        RegisterTestCase.test_registering.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RegisterTestCase.test_registering.__dict__.__setitem__('stypy_type_store', module_type_store)
        RegisterTestCase.test_registering.__dict__.__setitem__('stypy_function_name', 'RegisterTestCase.test_registering')
        RegisterTestCase.test_registering.__dict__.__setitem__('stypy_param_names_list', [])
        RegisterTestCase.test_registering.__dict__.__setitem__('stypy_varargs_param_name', None)
        RegisterTestCase.test_registering.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RegisterTestCase.test_registering.__dict__.__setitem__('stypy_call_defaults', defaults)
        RegisterTestCase.test_registering.__dict__.__setitem__('stypy_call_varargs', varargs)
        RegisterTestCase.test_registering.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RegisterTestCase.test_registering.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RegisterTestCase.test_registering', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_registering', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_registering(...)' code ##################

        
        # Assigning a Call to a Name (line 163):
        
        # Assigning a Call to a Name (line 163):
        
        # Call to _get_cmd(...): (line 163)
        # Processing the call keyword arguments (line 163)
        kwargs_42062 = {}
        # Getting the type of 'self' (line 163)
        self_42060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 14), 'self', False)
        # Obtaining the member '_get_cmd' of a type (line 163)
        _get_cmd_42061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 14), self_42060, '_get_cmd')
        # Calling _get_cmd(args, kwargs) (line 163)
        _get_cmd_call_result_42063 = invoke(stypy.reporting.localization.Localization(__file__, 163, 14), _get_cmd_42061, *[], **kwargs_42062)
        
        # Assigning a type to the variable 'cmd' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'cmd', _get_cmd_call_result_42063)
        
        # Assigning a Call to a Name (line 164):
        
        # Assigning a Call to a Name (line 164):
        
        # Call to RawInputs(...): (line 164)
        # Processing the call arguments (line 164)
        str_42065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 27), 'str', '2')
        str_42066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 32), 'str', 'tarek')
        str_42067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 41), 'str', 'tarek@ziade.org')
        # Processing the call keyword arguments (line 164)
        kwargs_42068 = {}
        # Getting the type of 'RawInputs' (line 164)
        RawInputs_42064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 17), 'RawInputs', False)
        # Calling RawInputs(args, kwargs) (line 164)
        RawInputs_call_result_42069 = invoke(stypy.reporting.localization.Localization(__file__, 164, 17), RawInputs_42064, *[str_42065, str_42066, str_42067], **kwargs_42068)
        
        # Assigning a type to the variable 'inputs' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'inputs', RawInputs_call_result_42069)
        
        # Assigning a Attribute to a Attribute (line 165):
        
        # Assigning a Attribute to a Attribute (line 165):
        # Getting the type of 'inputs' (line 165)
        inputs_42070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 36), 'inputs')
        # Obtaining the member '__call__' of a type (line 165)
        call___42071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 36), inputs_42070, '__call__')
        # Getting the type of 'register_module' (line 165)
        register_module_42072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'register_module')
        # Setting the type of the member 'raw_input' of a type (line 165)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), register_module_42072, 'raw_input', call___42071)
        
        # Try-finally block (line 166)
        
        # Call to run(...): (line 168)
        # Processing the call keyword arguments (line 168)
        kwargs_42075 = {}
        # Getting the type of 'cmd' (line 168)
        cmd_42073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'cmd', False)
        # Obtaining the member 'run' of a type (line 168)
        run_42074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 12), cmd_42073, 'run')
        # Calling run(args, kwargs) (line 168)
        run_call_result_42076 = invoke(stypy.reporting.localization.Localization(__file__, 168, 12), run_42074, *[], **kwargs_42075)
        
        
        # finally branch of the try-finally block (line 166)
        # Deleting a member
        # Getting the type of 'register_module' (line 170)
        register_module_42077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'register_module')
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 170, 12), register_module_42077, 'raw_input')
        
        
        # Call to assertEqual(...): (line 173)
        # Processing the call arguments (line 173)
        
        # Call to len(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'self' (line 173)
        self_42081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 29), 'self', False)
        # Obtaining the member 'conn' of a type (line 173)
        conn_42082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 29), self_42081, 'conn')
        # Obtaining the member 'reqs' of a type (line 173)
        reqs_42083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 29), conn_42082, 'reqs')
        # Processing the call keyword arguments (line 173)
        kwargs_42084 = {}
        # Getting the type of 'len' (line 173)
        len_42080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 25), 'len', False)
        # Calling len(args, kwargs) (line 173)
        len_call_result_42085 = invoke(stypy.reporting.localization.Localization(__file__, 173, 25), len_42080, *[reqs_42083], **kwargs_42084)
        
        int_42086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 46), 'int')
        # Processing the call keyword arguments (line 173)
        kwargs_42087 = {}
        # Getting the type of 'self' (line 173)
        self_42078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 173)
        assertEqual_42079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), self_42078, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 173)
        assertEqual_call_result_42088 = invoke(stypy.reporting.localization.Localization(__file__, 173, 8), assertEqual_42079, *[len_call_result_42085, int_42086], **kwargs_42087)
        
        
        # Assigning a Subscript to a Name (line 174):
        
        # Assigning a Subscript to a Name (line 174):
        
        # Obtaining the type of the subscript
        int_42089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 29), 'int')
        # Getting the type of 'self' (line 174)
        self_42090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 14), 'self')
        # Obtaining the member 'conn' of a type (line 174)
        conn_42091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 14), self_42090, 'conn')
        # Obtaining the member 'reqs' of a type (line 174)
        reqs_42092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 14), conn_42091, 'reqs')
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___42093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 14), reqs_42092, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 174)
        subscript_call_result_42094 = invoke(stypy.reporting.localization.Localization(__file__, 174, 14), getitem___42093, int_42089)
        
        # Assigning a type to the variable 'req' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'req', subscript_call_result_42094)
        
        # Assigning a Call to a Name (line 175):
        
        # Assigning a Call to a Name (line 175):
        
        # Call to dict(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'req' (line 175)
        req_42096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 23), 'req', False)
        # Obtaining the member 'headers' of a type (line 175)
        headers_42097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 23), req_42096, 'headers')
        # Processing the call keyword arguments (line 175)
        kwargs_42098 = {}
        # Getting the type of 'dict' (line 175)
        dict_42095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 18), 'dict', False)
        # Calling dict(args, kwargs) (line 175)
        dict_call_result_42099 = invoke(stypy.reporting.localization.Localization(__file__, 175, 18), dict_42095, *[headers_42097], **kwargs_42098)
        
        # Assigning a type to the variable 'headers' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'headers', dict_call_result_42099)
        
        # Call to assertEqual(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Obtaining the type of the subscript
        str_42102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 33), 'str', 'Content-length')
        # Getting the type of 'headers' (line 176)
        headers_42103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 25), 'headers', False)
        # Obtaining the member '__getitem__' of a type (line 176)
        getitem___42104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 25), headers_42103, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 176)
        subscript_call_result_42105 = invoke(stypy.reporting.localization.Localization(__file__, 176, 25), getitem___42104, str_42102)
        
        str_42106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 52), 'str', '608')
        # Processing the call keyword arguments (line 176)
        kwargs_42107 = {}
        # Getting the type of 'self' (line 176)
        self_42100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 176)
        assertEqual_42101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), self_42100, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 176)
        assertEqual_call_result_42108 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), assertEqual_42101, *[subscript_call_result_42105, str_42106], **kwargs_42107)
        
        
        # Call to assertIn(...): (line 177)
        # Processing the call arguments (line 177)
        str_42111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 22), 'str', 'tarek')
        # Getting the type of 'req' (line 177)
        req_42112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 31), 'req', False)
        # Obtaining the member 'data' of a type (line 177)
        data_42113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 31), req_42112, 'data')
        # Processing the call keyword arguments (line 177)
        kwargs_42114 = {}
        # Getting the type of 'self' (line 177)
        self_42109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 177)
        assertIn_42110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), self_42109, 'assertIn')
        # Calling assertIn(args, kwargs) (line 177)
        assertIn_call_result_42115 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), assertIn_42110, *[str_42111, data_42113], **kwargs_42114)
        
        
        # ################# End of 'test_registering(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_registering' in the type store
        # Getting the type of 'stypy_return_type' (line 161)
        stypy_return_type_42116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42116)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_registering'
        return stypy_return_type_42116


    @norecursion
    def test_password_reset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_password_reset'
        module_type_store = module_type_store.open_function_context('test_password_reset', 179, 4, False)
        # Assigning a type to the variable 'self' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RegisterTestCase.test_password_reset.__dict__.__setitem__('stypy_localization', localization)
        RegisterTestCase.test_password_reset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RegisterTestCase.test_password_reset.__dict__.__setitem__('stypy_type_store', module_type_store)
        RegisterTestCase.test_password_reset.__dict__.__setitem__('stypy_function_name', 'RegisterTestCase.test_password_reset')
        RegisterTestCase.test_password_reset.__dict__.__setitem__('stypy_param_names_list', [])
        RegisterTestCase.test_password_reset.__dict__.__setitem__('stypy_varargs_param_name', None)
        RegisterTestCase.test_password_reset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RegisterTestCase.test_password_reset.__dict__.__setitem__('stypy_call_defaults', defaults)
        RegisterTestCase.test_password_reset.__dict__.__setitem__('stypy_call_varargs', varargs)
        RegisterTestCase.test_password_reset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RegisterTestCase.test_password_reset.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RegisterTestCase.test_password_reset', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_password_reset', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_password_reset(...)' code ##################

        
        # Assigning a Call to a Name (line 181):
        
        # Assigning a Call to a Name (line 181):
        
        # Call to _get_cmd(...): (line 181)
        # Processing the call keyword arguments (line 181)
        kwargs_42119 = {}
        # Getting the type of 'self' (line 181)
        self_42117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 14), 'self', False)
        # Obtaining the member '_get_cmd' of a type (line 181)
        _get_cmd_42118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 14), self_42117, '_get_cmd')
        # Calling _get_cmd(args, kwargs) (line 181)
        _get_cmd_call_result_42120 = invoke(stypy.reporting.localization.Localization(__file__, 181, 14), _get_cmd_42118, *[], **kwargs_42119)
        
        # Assigning a type to the variable 'cmd' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'cmd', _get_cmd_call_result_42120)
        
        # Assigning a Call to a Name (line 182):
        
        # Assigning a Call to a Name (line 182):
        
        # Call to RawInputs(...): (line 182)
        # Processing the call arguments (line 182)
        str_42122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 27), 'str', '3')
        str_42123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 32), 'str', 'tarek@ziade.org')
        # Processing the call keyword arguments (line 182)
        kwargs_42124 = {}
        # Getting the type of 'RawInputs' (line 182)
        RawInputs_42121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 17), 'RawInputs', False)
        # Calling RawInputs(args, kwargs) (line 182)
        RawInputs_call_result_42125 = invoke(stypy.reporting.localization.Localization(__file__, 182, 17), RawInputs_42121, *[str_42122, str_42123], **kwargs_42124)
        
        # Assigning a type to the variable 'inputs' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'inputs', RawInputs_call_result_42125)
        
        # Assigning a Attribute to a Attribute (line 183):
        
        # Assigning a Attribute to a Attribute (line 183):
        # Getting the type of 'inputs' (line 183)
        inputs_42126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 36), 'inputs')
        # Obtaining the member '__call__' of a type (line 183)
        call___42127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 36), inputs_42126, '__call__')
        # Getting the type of 'register_module' (line 183)
        register_module_42128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'register_module')
        # Setting the type of the member 'raw_input' of a type (line 183)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), register_module_42128, 'raw_input', call___42127)
        
        # Try-finally block (line 184)
        
        # Call to run(...): (line 186)
        # Processing the call keyword arguments (line 186)
        kwargs_42131 = {}
        # Getting the type of 'cmd' (line 186)
        cmd_42129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'cmd', False)
        # Obtaining the member 'run' of a type (line 186)
        run_42130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 12), cmd_42129, 'run')
        # Calling run(args, kwargs) (line 186)
        run_call_result_42132 = invoke(stypy.reporting.localization.Localization(__file__, 186, 12), run_42130, *[], **kwargs_42131)
        
        
        # finally branch of the try-finally block (line 184)
        # Deleting a member
        # Getting the type of 'register_module' (line 188)
        register_module_42133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'register_module')
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 188, 12), register_module_42133, 'raw_input')
        
        
        # Call to assertEqual(...): (line 191)
        # Processing the call arguments (line 191)
        
        # Call to len(...): (line 191)
        # Processing the call arguments (line 191)
        # Getting the type of 'self' (line 191)
        self_42137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 29), 'self', False)
        # Obtaining the member 'conn' of a type (line 191)
        conn_42138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 29), self_42137, 'conn')
        # Obtaining the member 'reqs' of a type (line 191)
        reqs_42139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 29), conn_42138, 'reqs')
        # Processing the call keyword arguments (line 191)
        kwargs_42140 = {}
        # Getting the type of 'len' (line 191)
        len_42136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 25), 'len', False)
        # Calling len(args, kwargs) (line 191)
        len_call_result_42141 = invoke(stypy.reporting.localization.Localization(__file__, 191, 25), len_42136, *[reqs_42139], **kwargs_42140)
        
        int_42142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 46), 'int')
        # Processing the call keyword arguments (line 191)
        kwargs_42143 = {}
        # Getting the type of 'self' (line 191)
        self_42134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 191)
        assertEqual_42135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), self_42134, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 191)
        assertEqual_call_result_42144 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), assertEqual_42135, *[len_call_result_42141, int_42142], **kwargs_42143)
        
        
        # Assigning a Subscript to a Name (line 192):
        
        # Assigning a Subscript to a Name (line 192):
        
        # Obtaining the type of the subscript
        int_42145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 29), 'int')
        # Getting the type of 'self' (line 192)
        self_42146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 14), 'self')
        # Obtaining the member 'conn' of a type (line 192)
        conn_42147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 14), self_42146, 'conn')
        # Obtaining the member 'reqs' of a type (line 192)
        reqs_42148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 14), conn_42147, 'reqs')
        # Obtaining the member '__getitem__' of a type (line 192)
        getitem___42149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 14), reqs_42148, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 192)
        subscript_call_result_42150 = invoke(stypy.reporting.localization.Localization(__file__, 192, 14), getitem___42149, int_42145)
        
        # Assigning a type to the variable 'req' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'req', subscript_call_result_42150)
        
        # Assigning a Call to a Name (line 193):
        
        # Assigning a Call to a Name (line 193):
        
        # Call to dict(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'req' (line 193)
        req_42152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), 'req', False)
        # Obtaining the member 'headers' of a type (line 193)
        headers_42153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 23), req_42152, 'headers')
        # Processing the call keyword arguments (line 193)
        kwargs_42154 = {}
        # Getting the type of 'dict' (line 193)
        dict_42151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 18), 'dict', False)
        # Calling dict(args, kwargs) (line 193)
        dict_call_result_42155 = invoke(stypy.reporting.localization.Localization(__file__, 193, 18), dict_42151, *[headers_42153], **kwargs_42154)
        
        # Assigning a type to the variable 'headers' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'headers', dict_call_result_42155)
        
        # Call to assertEqual(...): (line 194)
        # Processing the call arguments (line 194)
        
        # Obtaining the type of the subscript
        str_42158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 33), 'str', 'Content-length')
        # Getting the type of 'headers' (line 194)
        headers_42159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 25), 'headers', False)
        # Obtaining the member '__getitem__' of a type (line 194)
        getitem___42160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 25), headers_42159, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 194)
        subscript_call_result_42161 = invoke(stypy.reporting.localization.Localization(__file__, 194, 25), getitem___42160, str_42158)
        
        str_42162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 52), 'str', '290')
        # Processing the call keyword arguments (line 194)
        kwargs_42163 = {}
        # Getting the type of 'self' (line 194)
        self_42156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 194)
        assertEqual_42157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 8), self_42156, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 194)
        assertEqual_call_result_42164 = invoke(stypy.reporting.localization.Localization(__file__, 194, 8), assertEqual_42157, *[subscript_call_result_42161, str_42162], **kwargs_42163)
        
        
        # Call to assertIn(...): (line 195)
        # Processing the call arguments (line 195)
        str_42167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 22), 'str', 'tarek')
        # Getting the type of 'req' (line 195)
        req_42168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 31), 'req', False)
        # Obtaining the member 'data' of a type (line 195)
        data_42169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 31), req_42168, 'data')
        # Processing the call keyword arguments (line 195)
        kwargs_42170 = {}
        # Getting the type of 'self' (line 195)
        self_42165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 195)
        assertIn_42166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 8), self_42165, 'assertIn')
        # Calling assertIn(args, kwargs) (line 195)
        assertIn_call_result_42171 = invoke(stypy.reporting.localization.Localization(__file__, 195, 8), assertIn_42166, *[str_42167, data_42169], **kwargs_42170)
        
        
        # ################# End of 'test_password_reset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_password_reset' in the type store
        # Getting the type of 'stypy_return_type' (line 179)
        stypy_return_type_42172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42172)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_password_reset'
        return stypy_return_type_42172


    @norecursion
    def test_strict(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_strict'
        module_type_store = module_type_store.open_function_context('test_strict', 197, 4, False)
        # Assigning a type to the variable 'self' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RegisterTestCase.test_strict.__dict__.__setitem__('stypy_localization', localization)
        RegisterTestCase.test_strict.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RegisterTestCase.test_strict.__dict__.__setitem__('stypy_type_store', module_type_store)
        RegisterTestCase.test_strict.__dict__.__setitem__('stypy_function_name', 'RegisterTestCase.test_strict')
        RegisterTestCase.test_strict.__dict__.__setitem__('stypy_param_names_list', [])
        RegisterTestCase.test_strict.__dict__.__setitem__('stypy_varargs_param_name', None)
        RegisterTestCase.test_strict.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RegisterTestCase.test_strict.__dict__.__setitem__('stypy_call_defaults', defaults)
        RegisterTestCase.test_strict.__dict__.__setitem__('stypy_call_varargs', varargs)
        RegisterTestCase.test_strict.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RegisterTestCase.test_strict.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RegisterTestCase.test_strict', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_strict', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_strict(...)' code ##################

        
        # Assigning a Call to a Name (line 205):
        
        # Assigning a Call to a Name (line 205):
        
        # Call to _get_cmd(...): (line 205)
        # Processing the call arguments (line 205)
        
        # Obtaining an instance of the builtin type 'dict' (line 205)
        dict_42175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 205)
        
        # Processing the call keyword arguments (line 205)
        kwargs_42176 = {}
        # Getting the type of 'self' (line 205)
        self_42173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 14), 'self', False)
        # Obtaining the member '_get_cmd' of a type (line 205)
        _get_cmd_42174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 14), self_42173, '_get_cmd')
        # Calling _get_cmd(args, kwargs) (line 205)
        _get_cmd_call_result_42177 = invoke(stypy.reporting.localization.Localization(__file__, 205, 14), _get_cmd_42174, *[dict_42175], **kwargs_42176)
        
        # Assigning a type to the variable 'cmd' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'cmd', _get_cmd_call_result_42177)
        
        # Call to ensure_finalized(...): (line 206)
        # Processing the call keyword arguments (line 206)
        kwargs_42180 = {}
        # Getting the type of 'cmd' (line 206)
        cmd_42178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 206)
        ensure_finalized_42179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), cmd_42178, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 206)
        ensure_finalized_call_result_42181 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), ensure_finalized_42179, *[], **kwargs_42180)
        
        
        # Assigning a Num to a Attribute (line 207):
        
        # Assigning a Num to a Attribute (line 207):
        int_42182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 21), 'int')
        # Getting the type of 'cmd' (line 207)
        cmd_42183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'cmd')
        # Setting the type of the member 'strict' of a type (line 207)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), cmd_42183, 'strict', int_42182)
        
        # Call to assertRaises(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'DistutilsSetupError' (line 208)
        DistutilsSetupError_42186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 26), 'DistutilsSetupError', False)
        # Getting the type of 'cmd' (line 208)
        cmd_42187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 47), 'cmd', False)
        # Obtaining the member 'run' of a type (line 208)
        run_42188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 47), cmd_42187, 'run')
        # Processing the call keyword arguments (line 208)
        kwargs_42189 = {}
        # Getting the type of 'self' (line 208)
        self_42184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 208)
        assertRaises_42185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), self_42184, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 208)
        assertRaises_call_result_42190 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), assertRaises_42185, *[DistutilsSetupError_42186, run_42188], **kwargs_42189)
        
        
        # Assigning a Dict to a Name (line 211):
        
        # Assigning a Dict to a Name (line 211):
        
        # Obtaining an instance of the builtin type 'dict' (line 211)
        dict_42191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 19), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 211)
        # Adding element type (key, value) (line 211)
        str_42192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 20), 'str', 'url')
        str_42193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 27), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 19), dict_42191, (str_42192, str_42193))
        # Adding element type (key, value) (line 211)
        str_42194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 34), 'str', 'author')
        str_42195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 44), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 19), dict_42191, (str_42194, str_42195))
        # Adding element type (key, value) (line 211)
        str_42196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 20), 'str', 'author_email')
        unicode_42197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 36), 'unicode', u'\xe9x\xe9x\xe9')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 19), dict_42191, (str_42196, unicode_42197))
        # Adding element type (key, value) (line 211)
        str_42198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 20), 'str', 'name')
        str_42199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 28), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 19), dict_42191, (str_42198, str_42199))
        # Adding element type (key, value) (line 211)
        str_42200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 35), 'str', 'version')
        str_42201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 46), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 19), dict_42191, (str_42200, str_42201))
        # Adding element type (key, value) (line 211)
        str_42202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 20), 'str', 'long_description')
        str_42203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 40), 'str', 'title\n==\n\ntext')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 19), dict_42191, (str_42202, str_42203))
        
        # Assigning a type to the variable 'metadata' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'metadata', dict_42191)
        
        # Assigning a Call to a Name (line 216):
        
        # Assigning a Call to a Name (line 216):
        
        # Call to _get_cmd(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'metadata' (line 216)
        metadata_42206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 28), 'metadata', False)
        # Processing the call keyword arguments (line 216)
        kwargs_42207 = {}
        # Getting the type of 'self' (line 216)
        self_42204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 14), 'self', False)
        # Obtaining the member '_get_cmd' of a type (line 216)
        _get_cmd_42205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 14), self_42204, '_get_cmd')
        # Calling _get_cmd(args, kwargs) (line 216)
        _get_cmd_call_result_42208 = invoke(stypy.reporting.localization.Localization(__file__, 216, 14), _get_cmd_42205, *[metadata_42206], **kwargs_42207)
        
        # Assigning a type to the variable 'cmd' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'cmd', _get_cmd_call_result_42208)
        
        # Call to ensure_finalized(...): (line 217)
        # Processing the call keyword arguments (line 217)
        kwargs_42211 = {}
        # Getting the type of 'cmd' (line 217)
        cmd_42209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 217)
        ensure_finalized_42210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 8), cmd_42209, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 217)
        ensure_finalized_call_result_42212 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), ensure_finalized_42210, *[], **kwargs_42211)
        
        
        # Assigning a Num to a Attribute (line 218):
        
        # Assigning a Num to a Attribute (line 218):
        int_42213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 21), 'int')
        # Getting the type of 'cmd' (line 218)
        cmd_42214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'cmd')
        # Setting the type of the member 'strict' of a type (line 218)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), cmd_42214, 'strict', int_42213)
        
        # Call to assertRaises(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'DistutilsSetupError' (line 219)
        DistutilsSetupError_42217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 26), 'DistutilsSetupError', False)
        # Getting the type of 'cmd' (line 219)
        cmd_42218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 47), 'cmd', False)
        # Obtaining the member 'run' of a type (line 219)
        run_42219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 47), cmd_42218, 'run')
        # Processing the call keyword arguments (line 219)
        kwargs_42220 = {}
        # Getting the type of 'self' (line 219)
        self_42215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 219)
        assertRaises_42216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 8), self_42215, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 219)
        assertRaises_call_result_42221 = invoke(stypy.reporting.localization.Localization(__file__, 219, 8), assertRaises_42216, *[DistutilsSetupError_42217, run_42219], **kwargs_42220)
        
        
        # Assigning a Str to a Subscript (line 222):
        
        # Assigning a Str to a Subscript (line 222):
        str_42222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 39), 'str', 'title\n=====\n\ntext')
        # Getting the type of 'metadata' (line 222)
        metadata_42223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'metadata')
        str_42224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 17), 'str', 'long_description')
        # Storing an element on a container (line 222)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 8), metadata_42223, (str_42224, str_42222))
        
        # Assigning a Call to a Name (line 223):
        
        # Assigning a Call to a Name (line 223):
        
        # Call to _get_cmd(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'metadata' (line 223)
        metadata_42227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 28), 'metadata', False)
        # Processing the call keyword arguments (line 223)
        kwargs_42228 = {}
        # Getting the type of 'self' (line 223)
        self_42225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 14), 'self', False)
        # Obtaining the member '_get_cmd' of a type (line 223)
        _get_cmd_42226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 14), self_42225, '_get_cmd')
        # Calling _get_cmd(args, kwargs) (line 223)
        _get_cmd_call_result_42229 = invoke(stypy.reporting.localization.Localization(__file__, 223, 14), _get_cmd_42226, *[metadata_42227], **kwargs_42228)
        
        # Assigning a type to the variable 'cmd' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'cmd', _get_cmd_call_result_42229)
        
        # Call to ensure_finalized(...): (line 224)
        # Processing the call keyword arguments (line 224)
        kwargs_42232 = {}
        # Getting the type of 'cmd' (line 224)
        cmd_42230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 224)
        ensure_finalized_42231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), cmd_42230, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 224)
        ensure_finalized_call_result_42233 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), ensure_finalized_42231, *[], **kwargs_42232)
        
        
        # Assigning a Num to a Attribute (line 225):
        
        # Assigning a Num to a Attribute (line 225):
        int_42234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 21), 'int')
        # Getting the type of 'cmd' (line 225)
        cmd_42235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'cmd')
        # Setting the type of the member 'strict' of a type (line 225)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), cmd_42235, 'strict', int_42234)
        
        # Assigning a Call to a Name (line 226):
        
        # Assigning a Call to a Name (line 226):
        
        # Call to RawInputs(...): (line 226)
        # Processing the call arguments (line 226)
        str_42237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 27), 'str', '1')
        str_42238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 32), 'str', 'tarek')
        str_42239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 41), 'str', 'y')
        # Processing the call keyword arguments (line 226)
        kwargs_42240 = {}
        # Getting the type of 'RawInputs' (line 226)
        RawInputs_42236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 17), 'RawInputs', False)
        # Calling RawInputs(args, kwargs) (line 226)
        RawInputs_call_result_42241 = invoke(stypy.reporting.localization.Localization(__file__, 226, 17), RawInputs_42236, *[str_42237, str_42238, str_42239], **kwargs_42240)
        
        # Assigning a type to the variable 'inputs' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'inputs', RawInputs_call_result_42241)
        
        # Assigning a Attribute to a Attribute (line 227):
        
        # Assigning a Attribute to a Attribute (line 227):
        # Getting the type of 'inputs' (line 227)
        inputs_42242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 36), 'inputs')
        # Obtaining the member '__call__' of a type (line 227)
        call___42243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 36), inputs_42242, '__call__')
        # Getting the type of 'register_module' (line 227)
        register_module_42244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'register_module')
        # Setting the type of the member 'raw_input' of a type (line 227)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), register_module_42244, 'raw_input', call___42243)
        
        # Try-finally block (line 229)
        
        # Call to run(...): (line 230)
        # Processing the call keyword arguments (line 230)
        kwargs_42247 = {}
        # Getting the type of 'cmd' (line 230)
        cmd_42245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'cmd', False)
        # Obtaining the member 'run' of a type (line 230)
        run_42246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), cmd_42245, 'run')
        # Calling run(args, kwargs) (line 230)
        run_call_result_42248 = invoke(stypy.reporting.localization.Localization(__file__, 230, 12), run_42246, *[], **kwargs_42247)
        
        
        # finally branch of the try-finally block (line 229)
        # Deleting a member
        # Getting the type of 'register_module' (line 232)
        register_module_42249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'register_module')
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 232, 12), register_module_42249, 'raw_input')
        
        
        # Assigning a Call to a Name (line 235):
        
        # Assigning a Call to a Name (line 235):
        
        # Call to _get_cmd(...): (line 235)
        # Processing the call keyword arguments (line 235)
        kwargs_42252 = {}
        # Getting the type of 'self' (line 235)
        self_42250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 14), 'self', False)
        # Obtaining the member '_get_cmd' of a type (line 235)
        _get_cmd_42251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 14), self_42250, '_get_cmd')
        # Calling _get_cmd(args, kwargs) (line 235)
        _get_cmd_call_result_42253 = invoke(stypy.reporting.localization.Localization(__file__, 235, 14), _get_cmd_42251, *[], **kwargs_42252)
        
        # Assigning a type to the variable 'cmd' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'cmd', _get_cmd_call_result_42253)
        
        # Call to ensure_finalized(...): (line 236)
        # Processing the call keyword arguments (line 236)
        kwargs_42256 = {}
        # Getting the type of 'cmd' (line 236)
        cmd_42254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 236)
        ensure_finalized_42255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), cmd_42254, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 236)
        ensure_finalized_call_result_42257 = invoke(stypy.reporting.localization.Localization(__file__, 236, 8), ensure_finalized_42255, *[], **kwargs_42256)
        
        
        # Assigning a Call to a Name (line 237):
        
        # Assigning a Call to a Name (line 237):
        
        # Call to RawInputs(...): (line 237)
        # Processing the call arguments (line 237)
        str_42259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 27), 'str', '1')
        str_42260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 32), 'str', 'tarek')
        str_42261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 41), 'str', 'y')
        # Processing the call keyword arguments (line 237)
        kwargs_42262 = {}
        # Getting the type of 'RawInputs' (line 237)
        RawInputs_42258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 17), 'RawInputs', False)
        # Calling RawInputs(args, kwargs) (line 237)
        RawInputs_call_result_42263 = invoke(stypy.reporting.localization.Localization(__file__, 237, 17), RawInputs_42258, *[str_42259, str_42260, str_42261], **kwargs_42262)
        
        # Assigning a type to the variable 'inputs' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'inputs', RawInputs_call_result_42263)
        
        # Assigning a Attribute to a Attribute (line 238):
        
        # Assigning a Attribute to a Attribute (line 238):
        # Getting the type of 'inputs' (line 238)
        inputs_42264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 36), 'inputs')
        # Obtaining the member '__call__' of a type (line 238)
        call___42265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 36), inputs_42264, '__call__')
        # Getting the type of 'register_module' (line 238)
        register_module_42266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'register_module')
        # Setting the type of the member 'raw_input' of a type (line 238)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), register_module_42266, 'raw_input', call___42265)
        
        # Try-finally block (line 240)
        
        # Call to run(...): (line 241)
        # Processing the call keyword arguments (line 241)
        kwargs_42269 = {}
        # Getting the type of 'cmd' (line 241)
        cmd_42267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'cmd', False)
        # Obtaining the member 'run' of a type (line 241)
        run_42268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), cmd_42267, 'run')
        # Calling run(args, kwargs) (line 241)
        run_call_result_42270 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), run_42268, *[], **kwargs_42269)
        
        
        # finally branch of the try-finally block (line 240)
        # Deleting a member
        # Getting the type of 'register_module' (line 243)
        register_module_42271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'register_module')
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 243, 12), register_module_42271, 'raw_input')
        
        
        # Assigning a Dict to a Name (line 246):
        
        # Assigning a Dict to a Name (line 246):
        
        # Obtaining an instance of the builtin type 'dict' (line 246)
        dict_42272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 19), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 246)
        # Adding element type (key, value) (line 246)
        str_42273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 20), 'str', 'url')
        unicode_42274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 27), 'unicode', u'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 19), dict_42272, (str_42273, unicode_42274))
        # Adding element type (key, value) (line 246)
        str_42275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 35), 'str', 'author')
        unicode_42276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 45), 'unicode', u'\xc9ric')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 19), dict_42272, (str_42275, unicode_42276))
        # Adding element type (key, value) (line 246)
        str_42277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 20), 'str', 'author_email')
        unicode_42278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 36), 'unicode', u'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 19), dict_42272, (str_42277, unicode_42278))
        # Adding element type (key, value) (line 246)
        unicode_42279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 44), 'unicode', u'name')
        str_42280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 53), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 19), dict_42272, (unicode_42279, str_42280))
        # Adding element type (key, value) (line 246)
        str_42281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 20), 'str', 'version')
        unicode_42282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 31), 'unicode', u'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 19), dict_42272, (str_42281, unicode_42282))
        # Adding element type (key, value) (line 246)
        str_42283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 20), 'str', 'description')
        unicode_42284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 35), 'unicode', u'Something about esszet \xdf')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 19), dict_42272, (str_42283, unicode_42284))
        # Adding element type (key, value) (line 246)
        str_42285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 20), 'str', 'long_description')
        unicode_42286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 40), 'unicode', u'More things about esszet \xdf')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 19), dict_42272, (str_42285, unicode_42286))
        
        # Assigning a type to the variable 'metadata' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'metadata', dict_42272)
        
        # Assigning a Call to a Name (line 252):
        
        # Assigning a Call to a Name (line 252):
        
        # Call to _get_cmd(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'metadata' (line 252)
        metadata_42289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 28), 'metadata', False)
        # Processing the call keyword arguments (line 252)
        kwargs_42290 = {}
        # Getting the type of 'self' (line 252)
        self_42287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 14), 'self', False)
        # Obtaining the member '_get_cmd' of a type (line 252)
        _get_cmd_42288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 14), self_42287, '_get_cmd')
        # Calling _get_cmd(args, kwargs) (line 252)
        _get_cmd_call_result_42291 = invoke(stypy.reporting.localization.Localization(__file__, 252, 14), _get_cmd_42288, *[metadata_42289], **kwargs_42290)
        
        # Assigning a type to the variable 'cmd' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'cmd', _get_cmd_call_result_42291)
        
        # Call to ensure_finalized(...): (line 253)
        # Processing the call keyword arguments (line 253)
        kwargs_42294 = {}
        # Getting the type of 'cmd' (line 253)
        cmd_42292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 253)
        ensure_finalized_42293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), cmd_42292, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 253)
        ensure_finalized_call_result_42295 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), ensure_finalized_42293, *[], **kwargs_42294)
        
        
        # Assigning a Num to a Attribute (line 254):
        
        # Assigning a Num to a Attribute (line 254):
        int_42296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 21), 'int')
        # Getting the type of 'cmd' (line 254)
        cmd_42297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'cmd')
        # Setting the type of the member 'strict' of a type (line 254)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 8), cmd_42297, 'strict', int_42296)
        
        # Assigning a Call to a Name (line 255):
        
        # Assigning a Call to a Name (line 255):
        
        # Call to RawInputs(...): (line 255)
        # Processing the call arguments (line 255)
        str_42299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 27), 'str', '1')
        str_42300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 32), 'str', 'tarek')
        str_42301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 41), 'str', 'y')
        # Processing the call keyword arguments (line 255)
        kwargs_42302 = {}
        # Getting the type of 'RawInputs' (line 255)
        RawInputs_42298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 17), 'RawInputs', False)
        # Calling RawInputs(args, kwargs) (line 255)
        RawInputs_call_result_42303 = invoke(stypy.reporting.localization.Localization(__file__, 255, 17), RawInputs_42298, *[str_42299, str_42300, str_42301], **kwargs_42302)
        
        # Assigning a type to the variable 'inputs' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'inputs', RawInputs_call_result_42303)
        
        # Assigning a Attribute to a Attribute (line 256):
        
        # Assigning a Attribute to a Attribute (line 256):
        # Getting the type of 'inputs' (line 256)
        inputs_42304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 36), 'inputs')
        # Obtaining the member '__call__' of a type (line 256)
        call___42305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 36), inputs_42304, '__call__')
        # Getting the type of 'register_module' (line 256)
        register_module_42306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'register_module')
        # Setting the type of the member 'raw_input' of a type (line 256)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 8), register_module_42306, 'raw_input', call___42305)
        
        # Try-finally block (line 258)
        
        # Call to run(...): (line 259)
        # Processing the call keyword arguments (line 259)
        kwargs_42309 = {}
        # Getting the type of 'cmd' (line 259)
        cmd_42307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'cmd', False)
        # Obtaining the member 'run' of a type (line 259)
        run_42308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 12), cmd_42307, 'run')
        # Calling run(args, kwargs) (line 259)
        run_call_result_42310 = invoke(stypy.reporting.localization.Localization(__file__, 259, 12), run_42308, *[], **kwargs_42309)
        
        
        # finally branch of the try-finally block (line 258)
        # Deleting a member
        # Getting the type of 'register_module' (line 261)
        register_module_42311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'register_module')
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 261, 12), register_module_42311, 'raw_input')
        
        
        # ################# End of 'test_strict(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_strict' in the type store
        # Getting the type of 'stypy_return_type' (line 197)
        stypy_return_type_42312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42312)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_strict'
        return stypy_return_type_42312


    @norecursion
    def test_register_invalid_long_description(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_register_invalid_long_description'
        module_type_store = module_type_store.open_function_context('test_register_invalid_long_description', 263, 4, False)
        # Assigning a type to the variable 'self' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RegisterTestCase.test_register_invalid_long_description.__dict__.__setitem__('stypy_localization', localization)
        RegisterTestCase.test_register_invalid_long_description.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RegisterTestCase.test_register_invalid_long_description.__dict__.__setitem__('stypy_type_store', module_type_store)
        RegisterTestCase.test_register_invalid_long_description.__dict__.__setitem__('stypy_function_name', 'RegisterTestCase.test_register_invalid_long_description')
        RegisterTestCase.test_register_invalid_long_description.__dict__.__setitem__('stypy_param_names_list', [])
        RegisterTestCase.test_register_invalid_long_description.__dict__.__setitem__('stypy_varargs_param_name', None)
        RegisterTestCase.test_register_invalid_long_description.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RegisterTestCase.test_register_invalid_long_description.__dict__.__setitem__('stypy_call_defaults', defaults)
        RegisterTestCase.test_register_invalid_long_description.__dict__.__setitem__('stypy_call_varargs', varargs)
        RegisterTestCase.test_register_invalid_long_description.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RegisterTestCase.test_register_invalid_long_description.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RegisterTestCase.test_register_invalid_long_description', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_register_invalid_long_description', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_register_invalid_long_description(...)' code ##################

        
        # Assigning a Str to a Name (line 265):
        
        # Assigning a Str to a Name (line 265):
        str_42313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 22), 'str', ':funkie:`str`')
        # Assigning a type to the variable 'description' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'description', str_42313)
        
        # Assigning a Dict to a Name (line 266):
        
        # Assigning a Dict to a Name (line 266):
        
        # Obtaining an instance of the builtin type 'dict' (line 266)
        dict_42314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 19), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 266)
        # Adding element type (key, value) (line 266)
        str_42315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 20), 'str', 'url')
        str_42316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 27), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 19), dict_42314, (str_42315, str_42316))
        # Adding element type (key, value) (line 266)
        str_42317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 34), 'str', 'author')
        str_42318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 44), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 19), dict_42314, (str_42317, str_42318))
        # Adding element type (key, value) (line 266)
        str_42319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 20), 'str', 'author_email')
        str_42320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 36), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 19), dict_42314, (str_42319, str_42320))
        # Adding element type (key, value) (line 266)
        str_42321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 20), 'str', 'name')
        str_42322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 28), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 19), dict_42314, (str_42321, str_42322))
        # Adding element type (key, value) (line 266)
        str_42323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 35), 'str', 'version')
        str_42324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 46), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 19), dict_42314, (str_42323, str_42324))
        # Adding element type (key, value) (line 266)
        str_42325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 20), 'str', 'long_description')
        # Getting the type of 'description' (line 269)
        description_42326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 40), 'description')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 19), dict_42314, (str_42325, description_42326))
        
        # Assigning a type to the variable 'metadata' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'metadata', dict_42314)
        
        # Assigning a Call to a Name (line 270):
        
        # Assigning a Call to a Name (line 270):
        
        # Call to _get_cmd(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'metadata' (line 270)
        metadata_42329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 28), 'metadata', False)
        # Processing the call keyword arguments (line 270)
        kwargs_42330 = {}
        # Getting the type of 'self' (line 270)
        self_42327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 14), 'self', False)
        # Obtaining the member '_get_cmd' of a type (line 270)
        _get_cmd_42328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 14), self_42327, '_get_cmd')
        # Calling _get_cmd(args, kwargs) (line 270)
        _get_cmd_call_result_42331 = invoke(stypy.reporting.localization.Localization(__file__, 270, 14), _get_cmd_42328, *[metadata_42329], **kwargs_42330)
        
        # Assigning a type to the variable 'cmd' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'cmd', _get_cmd_call_result_42331)
        
        # Call to ensure_finalized(...): (line 271)
        # Processing the call keyword arguments (line 271)
        kwargs_42334 = {}
        # Getting the type of 'cmd' (line 271)
        cmd_42332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 271)
        ensure_finalized_42333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 8), cmd_42332, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 271)
        ensure_finalized_call_result_42335 = invoke(stypy.reporting.localization.Localization(__file__, 271, 8), ensure_finalized_42333, *[], **kwargs_42334)
        
        
        # Assigning a Name to a Attribute (line 272):
        
        # Assigning a Name to a Attribute (line 272):
        # Getting the type of 'True' (line 272)
        True_42336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 21), 'True')
        # Getting the type of 'cmd' (line 272)
        cmd_42337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'cmd')
        # Setting the type of the member 'strict' of a type (line 272)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), cmd_42337, 'strict', True_42336)
        
        # Assigning a Call to a Name (line 273):
        
        # Assigning a Call to a Name (line 273):
        
        # Call to RawInputs(...): (line 273)
        # Processing the call arguments (line 273)
        str_42339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 27), 'str', '2')
        str_42340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 32), 'str', 'tarek')
        str_42341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 41), 'str', 'tarek@ziade.org')
        # Processing the call keyword arguments (line 273)
        kwargs_42342 = {}
        # Getting the type of 'RawInputs' (line 273)
        RawInputs_42338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 17), 'RawInputs', False)
        # Calling RawInputs(args, kwargs) (line 273)
        RawInputs_call_result_42343 = invoke(stypy.reporting.localization.Localization(__file__, 273, 17), RawInputs_42338, *[str_42339, str_42340, str_42341], **kwargs_42342)
        
        # Assigning a type to the variable 'inputs' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'inputs', RawInputs_call_result_42343)
        
        # Assigning a Name to a Attribute (line 274):
        
        # Assigning a Name to a Attribute (line 274):
        # Getting the type of 'inputs' (line 274)
        inputs_42344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 36), 'inputs')
        # Getting the type of 'register_module' (line 274)
        register_module_42345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'register_module')
        # Setting the type of the member 'raw_input' of a type (line 274)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), register_module_42345, 'raw_input', inputs_42344)
        
        # Call to addCleanup(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'delattr' (line 275)
        delattr_42348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 24), 'delattr', False)
        # Getting the type of 'register_module' (line 275)
        register_module_42349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 33), 'register_module', False)
        str_42350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 50), 'str', 'raw_input')
        # Processing the call keyword arguments (line 275)
        kwargs_42351 = {}
        # Getting the type of 'self' (line 275)
        self_42346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 275)
        addCleanup_42347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), self_42346, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 275)
        addCleanup_call_result_42352 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), addCleanup_42347, *[delattr_42348, register_module_42349, str_42350], **kwargs_42351)
        
        
        # Call to assertRaises(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'DistutilsSetupError' (line 276)
        DistutilsSetupError_42355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 26), 'DistutilsSetupError', False)
        # Getting the type of 'cmd' (line 276)
        cmd_42356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 47), 'cmd', False)
        # Obtaining the member 'run' of a type (line 276)
        run_42357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 47), cmd_42356, 'run')
        # Processing the call keyword arguments (line 276)
        kwargs_42358 = {}
        # Getting the type of 'self' (line 276)
        self_42353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 276)
        assertRaises_42354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), self_42353, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 276)
        assertRaises_call_result_42359 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), assertRaises_42354, *[DistutilsSetupError_42355, run_42357], **kwargs_42358)
        
        
        # ################# End of 'test_register_invalid_long_description(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_register_invalid_long_description' in the type store
        # Getting the type of 'stypy_return_type' (line 263)
        stypy_return_type_42360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42360)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_register_invalid_long_description'
        return stypy_return_type_42360


    @norecursion
    def test_check_metadata_deprecated(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_check_metadata_deprecated'
        module_type_store = module_type_store.open_function_context('test_check_metadata_deprecated', 278, 4, False)
        # Assigning a type to the variable 'self' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RegisterTestCase.test_check_metadata_deprecated.__dict__.__setitem__('stypy_localization', localization)
        RegisterTestCase.test_check_metadata_deprecated.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RegisterTestCase.test_check_metadata_deprecated.__dict__.__setitem__('stypy_type_store', module_type_store)
        RegisterTestCase.test_check_metadata_deprecated.__dict__.__setitem__('stypy_function_name', 'RegisterTestCase.test_check_metadata_deprecated')
        RegisterTestCase.test_check_metadata_deprecated.__dict__.__setitem__('stypy_param_names_list', [])
        RegisterTestCase.test_check_metadata_deprecated.__dict__.__setitem__('stypy_varargs_param_name', None)
        RegisterTestCase.test_check_metadata_deprecated.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RegisterTestCase.test_check_metadata_deprecated.__dict__.__setitem__('stypy_call_defaults', defaults)
        RegisterTestCase.test_check_metadata_deprecated.__dict__.__setitem__('stypy_call_varargs', varargs)
        RegisterTestCase.test_check_metadata_deprecated.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RegisterTestCase.test_check_metadata_deprecated.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RegisterTestCase.test_check_metadata_deprecated', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_check_metadata_deprecated', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_check_metadata_deprecated(...)' code ##################

        
        # Assigning a Call to a Name (line 280):
        
        # Assigning a Call to a Name (line 280):
        
        # Call to _get_cmd(...): (line 280)
        # Processing the call keyword arguments (line 280)
        kwargs_42363 = {}
        # Getting the type of 'self' (line 280)
        self_42361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 14), 'self', False)
        # Obtaining the member '_get_cmd' of a type (line 280)
        _get_cmd_42362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 14), self_42361, '_get_cmd')
        # Calling _get_cmd(args, kwargs) (line 280)
        _get_cmd_call_result_42364 = invoke(stypy.reporting.localization.Localization(__file__, 280, 14), _get_cmd_42362, *[], **kwargs_42363)
        
        # Assigning a type to the variable 'cmd' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'cmd', _get_cmd_call_result_42364)
        
        # Call to check_warnings(...): (line 281)
        # Processing the call keyword arguments (line 281)
        kwargs_42366 = {}
        # Getting the type of 'check_warnings' (line 281)
        check_warnings_42365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 13), 'check_warnings', False)
        # Calling check_warnings(args, kwargs) (line 281)
        check_warnings_call_result_42367 = invoke(stypy.reporting.localization.Localization(__file__, 281, 13), check_warnings_42365, *[], **kwargs_42366)
        
        with_42368 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 281, 13), check_warnings_call_result_42367, 'with parameter', '__enter__', '__exit__')

        if with_42368:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 281)
            enter___42369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 13), check_warnings_call_result_42367, '__enter__')
            with_enter_42370 = invoke(stypy.reporting.localization.Localization(__file__, 281, 13), enter___42369)
            # Assigning a type to the variable 'w' (line 281)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 13), 'w', with_enter_42370)
            
            # Call to simplefilter(...): (line 282)
            # Processing the call arguments (line 282)
            str_42373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 34), 'str', 'always')
            # Processing the call keyword arguments (line 282)
            kwargs_42374 = {}
            # Getting the type of 'warnings' (line 282)
            warnings_42371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'warnings', False)
            # Obtaining the member 'simplefilter' of a type (line 282)
            simplefilter_42372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 12), warnings_42371, 'simplefilter')
            # Calling simplefilter(args, kwargs) (line 282)
            simplefilter_call_result_42375 = invoke(stypy.reporting.localization.Localization(__file__, 282, 12), simplefilter_42372, *[str_42373], **kwargs_42374)
            
            
            # Call to check_metadata(...): (line 283)
            # Processing the call keyword arguments (line 283)
            kwargs_42378 = {}
            # Getting the type of 'cmd' (line 283)
            cmd_42376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'cmd', False)
            # Obtaining the member 'check_metadata' of a type (line 283)
            check_metadata_42377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 12), cmd_42376, 'check_metadata')
            # Calling check_metadata(args, kwargs) (line 283)
            check_metadata_call_result_42379 = invoke(stypy.reporting.localization.Localization(__file__, 283, 12), check_metadata_42377, *[], **kwargs_42378)
            
            
            # Call to assertEqual(...): (line 284)
            # Processing the call arguments (line 284)
            
            # Call to len(...): (line 284)
            # Processing the call arguments (line 284)
            # Getting the type of 'w' (line 284)
            w_42383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 33), 'w', False)
            # Obtaining the member 'warnings' of a type (line 284)
            warnings_42384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 33), w_42383, 'warnings')
            # Processing the call keyword arguments (line 284)
            kwargs_42385 = {}
            # Getting the type of 'len' (line 284)
            len_42382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 29), 'len', False)
            # Calling len(args, kwargs) (line 284)
            len_call_result_42386 = invoke(stypy.reporting.localization.Localization(__file__, 284, 29), len_42382, *[warnings_42384], **kwargs_42385)
            
            int_42387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 46), 'int')
            # Processing the call keyword arguments (line 284)
            kwargs_42388 = {}
            # Getting the type of 'self' (line 284)
            self_42380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'self', False)
            # Obtaining the member 'assertEqual' of a type (line 284)
            assertEqual_42381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 12), self_42380, 'assertEqual')
            # Calling assertEqual(args, kwargs) (line 284)
            assertEqual_call_result_42389 = invoke(stypy.reporting.localization.Localization(__file__, 284, 12), assertEqual_42381, *[len_call_result_42386, int_42387], **kwargs_42388)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 281)
            exit___42390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 13), check_warnings_call_result_42367, '__exit__')
            with_exit_42391 = invoke(stypy.reporting.localization.Localization(__file__, 281, 13), exit___42390, None, None, None)

        
        # ################# End of 'test_check_metadata_deprecated(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_check_metadata_deprecated' in the type store
        # Getting the type of 'stypy_return_type' (line 278)
        stypy_return_type_42392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_42392)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_check_metadata_deprecated'
        return stypy_return_type_42392


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 69, 0, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RegisterTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'RegisterTestCase' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'RegisterTestCase', RegisterTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 286, 0, False)
    
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

    
    # Call to makeSuite(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'RegisterTestCase' (line 287)
    RegisterTestCase_42395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 30), 'RegisterTestCase', False)
    # Processing the call keyword arguments (line 287)
    kwargs_42396 = {}
    # Getting the type of 'unittest' (line 287)
    unittest_42393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 287)
    makeSuite_42394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 11), unittest_42393, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 287)
    makeSuite_call_result_42397 = invoke(stypy.reporting.localization.Localization(__file__, 287, 11), makeSuite_42394, *[RegisterTestCase_42395], **kwargs_42396)
    
    # Assigning a type to the variable 'stypy_return_type' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'stypy_return_type', makeSuite_call_result_42397)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 286)
    stypy_return_type_42398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_42398)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_42398

# Assigning a type to the variable 'test_suite' (line 286)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 290)
    # Processing the call arguments (line 290)
    
    # Call to test_suite(...): (line 290)
    # Processing the call keyword arguments (line 290)
    kwargs_42401 = {}
    # Getting the type of 'test_suite' (line 290)
    test_suite_42400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 290)
    test_suite_call_result_42402 = invoke(stypy.reporting.localization.Localization(__file__, 290, 17), test_suite_42400, *[], **kwargs_42401)
    
    # Processing the call keyword arguments (line 290)
    kwargs_42403 = {}
    # Getting the type of 'run_unittest' (line 290)
    run_unittest_42399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 290)
    run_unittest_call_result_42404 = invoke(stypy.reporting.localization.Localization(__file__, 290, 4), run_unittest_42399, *[test_suite_call_result_42402], **kwargs_42403)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
