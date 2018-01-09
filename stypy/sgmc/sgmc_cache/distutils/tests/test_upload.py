
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # -*- encoding: utf8 -*-
2: '''Tests for distutils.command.upload.'''
3: import os
4: import unittest
5: from test.test_support import run_unittest
6: 
7: from distutils.command import upload as upload_mod
8: from distutils.command.upload import upload
9: from distutils.core import Distribution
10: from distutils.errors import DistutilsError
11: 
12: from distutils.tests.test_config import PYPIRC, PyPIRCCommandTestCase
13: 
14: PYPIRC_LONG_PASSWORD = '''\
15: [distutils]
16: 
17: index-servers =
18:     server1
19:     server2
20: 
21: [server1]
22: username:me
23: password:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
24: 
25: [server2]
26: username:meagain
27: password: secret
28: realm:acme
29: repository:http://another.pypi/
30: '''
31: 
32: 
33: PYPIRC_NOPASSWORD = '''\
34: [distutils]
35: 
36: index-servers =
37:     server1
38: 
39: [server1]
40: username:me
41: '''
42: 
43: class FakeOpen(object):
44: 
45:     def __init__(self, url, msg=None, code=None):
46:         self.url = url
47:         if not isinstance(url, str):
48:             self.req = url
49:         else:
50:             self.req = None
51:         self.msg = msg or 'OK'
52:         self.code = code or 200
53: 
54:     def getcode(self):
55:         return self.code
56: 
57: 
58: class uploadTestCase(PyPIRCCommandTestCase):
59: 
60:     def setUp(self):
61:         super(uploadTestCase, self).setUp()
62:         self.old_open = upload_mod.urlopen
63:         upload_mod.urlopen = self._urlopen
64:         self.last_open = None
65:         self.next_msg = None
66:         self.next_code = None
67: 
68:     def tearDown(self):
69:         upload_mod.urlopen = self.old_open
70:         super(uploadTestCase, self).tearDown()
71: 
72:     def _urlopen(self, url):
73:         self.last_open = FakeOpen(url, msg=self.next_msg, code=self.next_code)
74:         return self.last_open
75: 
76:     def test_finalize_options(self):
77: 
78:         # new format
79:         self.write_file(self.rc, PYPIRC)
80:         dist = Distribution()
81:         cmd = upload(dist)
82:         cmd.finalize_options()
83:         for attr, waited in (('username', 'me'), ('password', 'secret'),
84:                              ('realm', 'pypi'),
85:                              ('repository', 'https://upload.pypi.org/legacy/')):
86:             self.assertEqual(getattr(cmd, attr), waited)
87: 
88:     def test_saved_password(self):
89:         # file with no password
90:         self.write_file(self.rc, PYPIRC_NOPASSWORD)
91: 
92:         # make sure it passes
93:         dist = Distribution()
94:         cmd = upload(dist)
95:         cmd.finalize_options()
96:         self.assertEqual(cmd.password, None)
97: 
98:         # make sure we get it as well, if another command
99:         # initialized it at the dist level
100:         dist.password = 'xxx'
101:         cmd = upload(dist)
102:         cmd.finalize_options()
103:         self.assertEqual(cmd.password, 'xxx')
104: 
105:     def test_upload(self):
106:         tmp = self.mkdtemp()
107:         path = os.path.join(tmp, 'xxx')
108:         self.write_file(path)
109:         command, pyversion, filename = 'xxx', '2.6', path
110:         dist_files = [(command, pyversion, filename)]
111:         self.write_file(self.rc, PYPIRC_LONG_PASSWORD)
112: 
113:         # lets run it
114:         pkg_dir, dist = self.create_dist(dist_files=dist_files, author=u'dédé')
115:         cmd = upload(dist)
116:         cmd.ensure_finalized()
117:         cmd.run()
118: 
119:         # what did we send ?
120:         self.assertIn('dédé', self.last_open.req.data)
121:         headers = dict(self.last_open.req.headers)
122:         self.assertEqual(headers['Content-length'], '2159')
123:         self.assertTrue(headers['Content-type'].startswith('multipart/form-data'))
124:         self.assertEqual(self.last_open.req.get_method(), 'POST')
125:         self.assertEqual(self.last_open.req.get_full_url(),
126:                          'https://upload.pypi.org/legacy/')
127:         self.assertIn('xxx', self.last_open.req.data)
128:         auth = self.last_open.req.headers['Authorization']
129:         self.assertNotIn('\n', auth)
130: 
131:     def test_upload_fails(self):
132:         self.next_msg = "Not Found"
133:         self.next_code = 404
134:         self.assertRaises(DistutilsError, self.test_upload)
135: 
136: def test_suite():
137:     return unittest.makeSuite(uploadTestCase)
138: 
139: if __name__ == "__main__":
140:     run_unittest(test_suite())
141: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_45182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 0), 'str', 'Tests for distutils.command.upload.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import unittest' statement (line 4)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from test.test_support import run_unittest' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_45183 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support')

if (type(import_45183) is not StypyTypeError):

    if (import_45183 != 'pyd_module'):
        __import__(import_45183)
        sys_modules_45184 = sys.modules[import_45183]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', sys_modules_45184.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_45184, sys_modules_45184.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', import_45183)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.command import upload_mod' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_45185 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command')

if (type(import_45185) is not StypyTypeError):

    if (import_45185 != 'pyd_module'):
        __import__(import_45185)
        sys_modules_45186 = sys.modules[import_45185]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command', sys_modules_45186.module_type_store, module_type_store, ['upload'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_45186, sys_modules_45186.module_type_store, module_type_store)
    else:
        from distutils.command import upload as upload_mod

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command', None, module_type_store, ['upload'], [upload_mod])

else:
    # Assigning a type to the variable 'distutils.command' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command', import_45185)

# Adding an alias
module_type_store.add_alias('upload_mod', 'upload')
remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.command.upload import upload' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_45187 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.command.upload')

if (type(import_45187) is not StypyTypeError):

    if (import_45187 != 'pyd_module'):
        __import__(import_45187)
        sys_modules_45188 = sys.modules[import_45187]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.command.upload', sys_modules_45188.module_type_store, module_type_store, ['upload'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_45188, sys_modules_45188.module_type_store, module_type_store)
    else:
        from distutils.command.upload import upload

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.command.upload', None, module_type_store, ['upload'], [upload])

else:
    # Assigning a type to the variable 'distutils.command.upload' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.command.upload', import_45187)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils.core import Distribution' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_45189 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.core')

if (type(import_45189) is not StypyTypeError):

    if (import_45189 != 'pyd_module'):
        __import__(import_45189)
        sys_modules_45190 = sys.modules[import_45189]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.core', sys_modules_45190.module_type_store, module_type_store, ['Distribution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_45190, sys_modules_45190.module_type_store, module_type_store)
    else:
        from distutils.core import Distribution

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.core', None, module_type_store, ['Distribution'], [Distribution])

else:
    # Assigning a type to the variable 'distutils.core' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.core', import_45189)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils.errors import DistutilsError' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_45191 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors')

if (type(import_45191) is not StypyTypeError):

    if (import_45191 != 'pyd_module'):
        __import__(import_45191)
        sys_modules_45192 = sys.modules[import_45191]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', sys_modules_45192.module_type_store, module_type_store, ['DistutilsError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_45192, sys_modules_45192.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsError

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', None, module_type_store, ['DistutilsError'], [DistutilsError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', import_45191)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.tests.test_config import PYPIRC, PyPIRCCommandTestCase' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_45193 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.tests.test_config')

if (type(import_45193) is not StypyTypeError):

    if (import_45193 != 'pyd_module'):
        __import__(import_45193)
        sys_modules_45194 = sys.modules[import_45193]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.tests.test_config', sys_modules_45194.module_type_store, module_type_store, ['PYPIRC', 'PyPIRCCommandTestCase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_45194, sys_modules_45194.module_type_store, module_type_store)
    else:
        from distutils.tests.test_config import PYPIRC, PyPIRCCommandTestCase

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.tests.test_config', None, module_type_store, ['PYPIRC', 'PyPIRCCommandTestCase'], [PYPIRC, PyPIRCCommandTestCase])

else:
    # Assigning a type to the variable 'distutils.tests.test_config' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.tests.test_config', import_45193)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')


# Assigning a Str to a Name (line 14):

# Assigning a Str to a Name (line 14):
str_45195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, (-1)), 'str', '[distutils]\n\nindex-servers =\n    server1\n    server2\n\n[server1]\nusername:me\npassword:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n\n[server2]\nusername:meagain\npassword: secret\nrealm:acme\nrepository:http://another.pypi/\n')
# Assigning a type to the variable 'PYPIRC_LONG_PASSWORD' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'PYPIRC_LONG_PASSWORD', str_45195)

# Assigning a Str to a Name (line 33):

# Assigning a Str to a Name (line 33):
str_45196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, (-1)), 'str', '[distutils]\n\nindex-servers =\n    server1\n\n[server1]\nusername:me\n')
# Assigning a type to the variable 'PYPIRC_NOPASSWORD' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'PYPIRC_NOPASSWORD', str_45196)
# Declaration of the 'FakeOpen' class

class FakeOpen(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 45)
        None_45197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 32), 'None')
        # Getting the type of 'None' (line 45)
        None_45198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 43), 'None')
        defaults = [None_45197, None_45198]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FakeOpen.__init__', ['url', 'msg', 'code'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['url', 'msg', 'code'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 46):
        
        # Assigning a Name to a Attribute (line 46):
        # Getting the type of 'url' (line 46)
        url_45199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 19), 'url')
        # Getting the type of 'self' (line 46)
        self_45200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self')
        # Setting the type of the member 'url' of a type (line 46)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_45200, 'url', url_45199)
        
        # Type idiom detected: calculating its left and rigth part (line 47)
        # Getting the type of 'str' (line 47)
        str_45201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 31), 'str')
        # Getting the type of 'url' (line 47)
        url_45202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 26), 'url')
        
        (may_be_45203, more_types_in_union_45204) = may_not_be_subtype(str_45201, url_45202)

        if may_be_45203:

            if more_types_in_union_45204:
                # Runtime conditional SSA (line 47)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'url' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'url', remove_subtype_from_union(url_45202, str))
            
            # Assigning a Name to a Attribute (line 48):
            
            # Assigning a Name to a Attribute (line 48):
            # Getting the type of 'url' (line 48)
            url_45205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'url')
            # Getting the type of 'self' (line 48)
            self_45206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'self')
            # Setting the type of the member 'req' of a type (line 48)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), self_45206, 'req', url_45205)

            if more_types_in_union_45204:
                # Runtime conditional SSA for else branch (line 47)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_45203) or more_types_in_union_45204):
            # Assigning a type to the variable 'url' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'url', remove_not_subtype_from_union(url_45202, str))
            
            # Assigning a Name to a Attribute (line 50):
            
            # Assigning a Name to a Attribute (line 50):
            # Getting the type of 'None' (line 50)
            None_45207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 23), 'None')
            # Getting the type of 'self' (line 50)
            self_45208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'self')
            # Setting the type of the member 'req' of a type (line 50)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), self_45208, 'req', None_45207)

            if (may_be_45203 and more_types_in_union_45204):
                # SSA join for if statement (line 47)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BoolOp to a Attribute (line 51):
        
        # Assigning a BoolOp to a Attribute (line 51):
        
        # Evaluating a boolean operation
        # Getting the type of 'msg' (line 51)
        msg_45209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'msg')
        str_45210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 26), 'str', 'OK')
        # Applying the binary operator 'or' (line 51)
        result_or_keyword_45211 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 19), 'or', msg_45209, str_45210)
        
        # Getting the type of 'self' (line 51)
        self_45212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'self')
        # Setting the type of the member 'msg' of a type (line 51)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), self_45212, 'msg', result_or_keyword_45211)
        
        # Assigning a BoolOp to a Attribute (line 52):
        
        # Assigning a BoolOp to a Attribute (line 52):
        
        # Evaluating a boolean operation
        # Getting the type of 'code' (line 52)
        code_45213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'code')
        int_45214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 28), 'int')
        # Applying the binary operator 'or' (line 52)
        result_or_keyword_45215 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 20), 'or', code_45213, int_45214)
        
        # Getting the type of 'self' (line 52)
        self_45216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'self')
        # Setting the type of the member 'code' of a type (line 52)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), self_45216, 'code', result_or_keyword_45215)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def getcode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getcode'
        module_type_store = module_type_store.open_function_context('getcode', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FakeOpen.getcode.__dict__.__setitem__('stypy_localization', localization)
        FakeOpen.getcode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FakeOpen.getcode.__dict__.__setitem__('stypy_type_store', module_type_store)
        FakeOpen.getcode.__dict__.__setitem__('stypy_function_name', 'FakeOpen.getcode')
        FakeOpen.getcode.__dict__.__setitem__('stypy_param_names_list', [])
        FakeOpen.getcode.__dict__.__setitem__('stypy_varargs_param_name', None)
        FakeOpen.getcode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FakeOpen.getcode.__dict__.__setitem__('stypy_call_defaults', defaults)
        FakeOpen.getcode.__dict__.__setitem__('stypy_call_varargs', varargs)
        FakeOpen.getcode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FakeOpen.getcode.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FakeOpen.getcode', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getcode', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getcode(...)' code ##################

        # Getting the type of 'self' (line 55)
        self_45217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 15), 'self')
        # Obtaining the member 'code' of a type (line 55)
        code_45218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 15), self_45217, 'code')
        # Assigning a type to the variable 'stypy_return_type' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'stypy_return_type', code_45218)
        
        # ################# End of 'getcode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getcode' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_45219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_45219)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getcode'
        return stypy_return_type_45219


# Assigning a type to the variable 'FakeOpen' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'FakeOpen', FakeOpen)
# Declaration of the 'uploadTestCase' class
# Getting the type of 'PyPIRCCommandTestCase' (line 58)
PyPIRCCommandTestCase_45220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'PyPIRCCommandTestCase')

class uploadTestCase(PyPIRCCommandTestCase_45220, ):

    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 60, 4, False)
        # Assigning a type to the variable 'self' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        uploadTestCase.setUp.__dict__.__setitem__('stypy_localization', localization)
        uploadTestCase.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        uploadTestCase.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        uploadTestCase.setUp.__dict__.__setitem__('stypy_function_name', 'uploadTestCase.setUp')
        uploadTestCase.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        uploadTestCase.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        uploadTestCase.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        uploadTestCase.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        uploadTestCase.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        uploadTestCase.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        uploadTestCase.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'uploadTestCase.setUp', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to setUp(...): (line 61)
        # Processing the call keyword arguments (line 61)
        kwargs_45227 = {}
        
        # Call to super(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'uploadTestCase' (line 61)
        uploadTestCase_45222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 14), 'uploadTestCase', False)
        # Getting the type of 'self' (line 61)
        self_45223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'self', False)
        # Processing the call keyword arguments (line 61)
        kwargs_45224 = {}
        # Getting the type of 'super' (line 61)
        super_45221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'super', False)
        # Calling super(args, kwargs) (line 61)
        super_call_result_45225 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), super_45221, *[uploadTestCase_45222, self_45223], **kwargs_45224)
        
        # Obtaining the member 'setUp' of a type (line 61)
        setUp_45226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), super_call_result_45225, 'setUp')
        # Calling setUp(args, kwargs) (line 61)
        setUp_call_result_45228 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), setUp_45226, *[], **kwargs_45227)
        
        
        # Assigning a Attribute to a Attribute (line 62):
        
        # Assigning a Attribute to a Attribute (line 62):
        # Getting the type of 'upload_mod' (line 62)
        upload_mod_45229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'upload_mod')
        # Obtaining the member 'urlopen' of a type (line 62)
        urlopen_45230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 24), upload_mod_45229, 'urlopen')
        # Getting the type of 'self' (line 62)
        self_45231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self')
        # Setting the type of the member 'old_open' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_45231, 'old_open', urlopen_45230)
        
        # Assigning a Attribute to a Attribute (line 63):
        
        # Assigning a Attribute to a Attribute (line 63):
        # Getting the type of 'self' (line 63)
        self_45232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 29), 'self')
        # Obtaining the member '_urlopen' of a type (line 63)
        _urlopen_45233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 29), self_45232, '_urlopen')
        # Getting the type of 'upload_mod' (line 63)
        upload_mod_45234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'upload_mod')
        # Setting the type of the member 'urlopen' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), upload_mod_45234, 'urlopen', _urlopen_45233)
        
        # Assigning a Name to a Attribute (line 64):
        
        # Assigning a Name to a Attribute (line 64):
        # Getting the type of 'None' (line 64)
        None_45235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 25), 'None')
        # Getting the type of 'self' (line 64)
        self_45236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self')
        # Setting the type of the member 'last_open' of a type (line 64)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_45236, 'last_open', None_45235)
        
        # Assigning a Name to a Attribute (line 65):
        
        # Assigning a Name to a Attribute (line 65):
        # Getting the type of 'None' (line 65)
        None_45237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 24), 'None')
        # Getting the type of 'self' (line 65)
        self_45238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self')
        # Setting the type of the member 'next_msg' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_45238, 'next_msg', None_45237)
        
        # Assigning a Name to a Attribute (line 66):
        
        # Assigning a Name to a Attribute (line 66):
        # Getting the type of 'None' (line 66)
        None_45239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'None')
        # Getting the type of 'self' (line 66)
        self_45240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self')
        # Setting the type of the member 'next_code' of a type (line 66)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_45240, 'next_code', None_45239)
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 60)
        stypy_return_type_45241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_45241)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_45241


    @norecursion
    def tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDown'
        module_type_store = module_type_store.open_function_context('tearDown', 68, 4, False)
        # Assigning a type to the variable 'self' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        uploadTestCase.tearDown.__dict__.__setitem__('stypy_localization', localization)
        uploadTestCase.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        uploadTestCase.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        uploadTestCase.tearDown.__dict__.__setitem__('stypy_function_name', 'uploadTestCase.tearDown')
        uploadTestCase.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        uploadTestCase.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        uploadTestCase.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        uploadTestCase.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        uploadTestCase.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        uploadTestCase.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        uploadTestCase.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'uploadTestCase.tearDown', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Attribute (line 69):
        
        # Assigning a Attribute to a Attribute (line 69):
        # Getting the type of 'self' (line 69)
        self_45242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 29), 'self')
        # Obtaining the member 'old_open' of a type (line 69)
        old_open_45243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 29), self_45242, 'old_open')
        # Getting the type of 'upload_mod' (line 69)
        upload_mod_45244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'upload_mod')
        # Setting the type of the member 'urlopen' of a type (line 69)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), upload_mod_45244, 'urlopen', old_open_45243)
        
        # Call to tearDown(...): (line 70)
        # Processing the call keyword arguments (line 70)
        kwargs_45251 = {}
        
        # Call to super(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'uploadTestCase' (line 70)
        uploadTestCase_45246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 14), 'uploadTestCase', False)
        # Getting the type of 'self' (line 70)
        self_45247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 30), 'self', False)
        # Processing the call keyword arguments (line 70)
        kwargs_45248 = {}
        # Getting the type of 'super' (line 70)
        super_45245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'super', False)
        # Calling super(args, kwargs) (line 70)
        super_call_result_45249 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), super_45245, *[uploadTestCase_45246, self_45247], **kwargs_45248)
        
        # Obtaining the member 'tearDown' of a type (line 70)
        tearDown_45250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), super_call_result_45249, 'tearDown')
        # Calling tearDown(args, kwargs) (line 70)
        tearDown_call_result_45252 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), tearDown_45250, *[], **kwargs_45251)
        
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_45253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_45253)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_45253


    @norecursion
    def _urlopen(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_urlopen'
        module_type_store = module_type_store.open_function_context('_urlopen', 72, 4, False)
        # Assigning a type to the variable 'self' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        uploadTestCase._urlopen.__dict__.__setitem__('stypy_localization', localization)
        uploadTestCase._urlopen.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        uploadTestCase._urlopen.__dict__.__setitem__('stypy_type_store', module_type_store)
        uploadTestCase._urlopen.__dict__.__setitem__('stypy_function_name', 'uploadTestCase._urlopen')
        uploadTestCase._urlopen.__dict__.__setitem__('stypy_param_names_list', ['url'])
        uploadTestCase._urlopen.__dict__.__setitem__('stypy_varargs_param_name', None)
        uploadTestCase._urlopen.__dict__.__setitem__('stypy_kwargs_param_name', None)
        uploadTestCase._urlopen.__dict__.__setitem__('stypy_call_defaults', defaults)
        uploadTestCase._urlopen.__dict__.__setitem__('stypy_call_varargs', varargs)
        uploadTestCase._urlopen.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        uploadTestCase._urlopen.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'uploadTestCase._urlopen', ['url'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_urlopen', localization, ['url'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_urlopen(...)' code ##################

        
        # Assigning a Call to a Attribute (line 73):
        
        # Assigning a Call to a Attribute (line 73):
        
        # Call to FakeOpen(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'url' (line 73)
        url_45255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 34), 'url', False)
        # Processing the call keyword arguments (line 73)
        # Getting the type of 'self' (line 73)
        self_45256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 43), 'self', False)
        # Obtaining the member 'next_msg' of a type (line 73)
        next_msg_45257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 43), self_45256, 'next_msg')
        keyword_45258 = next_msg_45257
        # Getting the type of 'self' (line 73)
        self_45259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 63), 'self', False)
        # Obtaining the member 'next_code' of a type (line 73)
        next_code_45260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 63), self_45259, 'next_code')
        keyword_45261 = next_code_45260
        kwargs_45262 = {'msg': keyword_45258, 'code': keyword_45261}
        # Getting the type of 'FakeOpen' (line 73)
        FakeOpen_45254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 25), 'FakeOpen', False)
        # Calling FakeOpen(args, kwargs) (line 73)
        FakeOpen_call_result_45263 = invoke(stypy.reporting.localization.Localization(__file__, 73, 25), FakeOpen_45254, *[url_45255], **kwargs_45262)
        
        # Getting the type of 'self' (line 73)
        self_45264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self')
        # Setting the type of the member 'last_open' of a type (line 73)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_45264, 'last_open', FakeOpen_call_result_45263)
        # Getting the type of 'self' (line 74)
        self_45265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'self')
        # Obtaining the member 'last_open' of a type (line 74)
        last_open_45266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 15), self_45265, 'last_open')
        # Assigning a type to the variable 'stypy_return_type' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'stypy_return_type', last_open_45266)
        
        # ################# End of '_urlopen(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_urlopen' in the type store
        # Getting the type of 'stypy_return_type' (line 72)
        stypy_return_type_45267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_45267)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_urlopen'
        return stypy_return_type_45267


    @norecursion
    def test_finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_finalize_options'
        module_type_store = module_type_store.open_function_context('test_finalize_options', 76, 4, False)
        # Assigning a type to the variable 'self' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        uploadTestCase.test_finalize_options.__dict__.__setitem__('stypy_localization', localization)
        uploadTestCase.test_finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        uploadTestCase.test_finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        uploadTestCase.test_finalize_options.__dict__.__setitem__('stypy_function_name', 'uploadTestCase.test_finalize_options')
        uploadTestCase.test_finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        uploadTestCase.test_finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        uploadTestCase.test_finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        uploadTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        uploadTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        uploadTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        uploadTestCase.test_finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'uploadTestCase.test_finalize_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_finalize_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_finalize_options(...)' code ##################

        
        # Call to write_file(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'self' (line 79)
        self_45270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 24), 'self', False)
        # Obtaining the member 'rc' of a type (line 79)
        rc_45271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 24), self_45270, 'rc')
        # Getting the type of 'PYPIRC' (line 79)
        PYPIRC_45272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 33), 'PYPIRC', False)
        # Processing the call keyword arguments (line 79)
        kwargs_45273 = {}
        # Getting the type of 'self' (line 79)
        self_45268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 79)
        write_file_45269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), self_45268, 'write_file')
        # Calling write_file(args, kwargs) (line 79)
        write_file_call_result_45274 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), write_file_45269, *[rc_45271, PYPIRC_45272], **kwargs_45273)
        
        
        # Assigning a Call to a Name (line 80):
        
        # Assigning a Call to a Name (line 80):
        
        # Call to Distribution(...): (line 80)
        # Processing the call keyword arguments (line 80)
        kwargs_45276 = {}
        # Getting the type of 'Distribution' (line 80)
        Distribution_45275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 80)
        Distribution_call_result_45277 = invoke(stypy.reporting.localization.Localization(__file__, 80, 15), Distribution_45275, *[], **kwargs_45276)
        
        # Assigning a type to the variable 'dist' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'dist', Distribution_call_result_45277)
        
        # Assigning a Call to a Name (line 81):
        
        # Assigning a Call to a Name (line 81):
        
        # Call to upload(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'dist' (line 81)
        dist_45279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 21), 'dist', False)
        # Processing the call keyword arguments (line 81)
        kwargs_45280 = {}
        # Getting the type of 'upload' (line 81)
        upload_45278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 14), 'upload', False)
        # Calling upload(args, kwargs) (line 81)
        upload_call_result_45281 = invoke(stypy.reporting.localization.Localization(__file__, 81, 14), upload_45278, *[dist_45279], **kwargs_45280)
        
        # Assigning a type to the variable 'cmd' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'cmd', upload_call_result_45281)
        
        # Call to finalize_options(...): (line 82)
        # Processing the call keyword arguments (line 82)
        kwargs_45284 = {}
        # Getting the type of 'cmd' (line 82)
        cmd_45282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 82)
        finalize_options_45283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), cmd_45282, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 82)
        finalize_options_call_result_45285 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), finalize_options_45283, *[], **kwargs_45284)
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 83)
        tuple_45286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 83)
        # Adding element type (line 83)
        
        # Obtaining an instance of the builtin type 'tuple' (line 83)
        tuple_45287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 83)
        # Adding element type (line 83)
        str_45288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 30), 'str', 'username')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 30), tuple_45287, str_45288)
        # Adding element type (line 83)
        str_45289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 42), 'str', 'me')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 30), tuple_45287, str_45289)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 29), tuple_45286, tuple_45287)
        # Adding element type (line 83)
        
        # Obtaining an instance of the builtin type 'tuple' (line 83)
        tuple_45290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 83)
        # Adding element type (line 83)
        str_45291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 50), 'str', 'password')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 50), tuple_45290, str_45291)
        # Adding element type (line 83)
        str_45292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 62), 'str', 'secret')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 50), tuple_45290, str_45292)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 29), tuple_45286, tuple_45290)
        # Adding element type (line 83)
        
        # Obtaining an instance of the builtin type 'tuple' (line 84)
        tuple_45293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 84)
        # Adding element type (line 84)
        str_45294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 30), 'str', 'realm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 30), tuple_45293, str_45294)
        # Adding element type (line 84)
        str_45295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 39), 'str', 'pypi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 30), tuple_45293, str_45295)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 29), tuple_45286, tuple_45293)
        # Adding element type (line 83)
        
        # Obtaining an instance of the builtin type 'tuple' (line 85)
        tuple_45296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 85)
        # Adding element type (line 85)
        str_45297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 30), 'str', 'repository')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 30), tuple_45296, str_45297)
        # Adding element type (line 85)
        str_45298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 44), 'str', 'https://upload.pypi.org/legacy/')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 30), tuple_45296, str_45298)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 29), tuple_45286, tuple_45296)
        
        # Testing the type of a for loop iterable (line 83)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 83, 8), tuple_45286)
        # Getting the type of the for loop variable (line 83)
        for_loop_var_45299 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 83, 8), tuple_45286)
        # Assigning a type to the variable 'attr' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'attr', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 8), for_loop_var_45299))
        # Assigning a type to the variable 'waited' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'waited', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 8), for_loop_var_45299))
        # SSA begins for a for statement (line 83)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assertEqual(...): (line 86)
        # Processing the call arguments (line 86)
        
        # Call to getattr(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'cmd' (line 86)
        cmd_45303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 37), 'cmd', False)
        # Getting the type of 'attr' (line 86)
        attr_45304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 42), 'attr', False)
        # Processing the call keyword arguments (line 86)
        kwargs_45305 = {}
        # Getting the type of 'getattr' (line 86)
        getattr_45302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 29), 'getattr', False)
        # Calling getattr(args, kwargs) (line 86)
        getattr_call_result_45306 = invoke(stypy.reporting.localization.Localization(__file__, 86, 29), getattr_45302, *[cmd_45303, attr_45304], **kwargs_45305)
        
        # Getting the type of 'waited' (line 86)
        waited_45307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 49), 'waited', False)
        # Processing the call keyword arguments (line 86)
        kwargs_45308 = {}
        # Getting the type of 'self' (line 86)
        self_45300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 86)
        assertEqual_45301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), self_45300, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 86)
        assertEqual_call_result_45309 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), assertEqual_45301, *[getattr_call_result_45306, waited_45307], **kwargs_45308)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 76)
        stypy_return_type_45310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_45310)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_finalize_options'
        return stypy_return_type_45310


    @norecursion
    def test_saved_password(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_saved_password'
        module_type_store = module_type_store.open_function_context('test_saved_password', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        uploadTestCase.test_saved_password.__dict__.__setitem__('stypy_localization', localization)
        uploadTestCase.test_saved_password.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        uploadTestCase.test_saved_password.__dict__.__setitem__('stypy_type_store', module_type_store)
        uploadTestCase.test_saved_password.__dict__.__setitem__('stypy_function_name', 'uploadTestCase.test_saved_password')
        uploadTestCase.test_saved_password.__dict__.__setitem__('stypy_param_names_list', [])
        uploadTestCase.test_saved_password.__dict__.__setitem__('stypy_varargs_param_name', None)
        uploadTestCase.test_saved_password.__dict__.__setitem__('stypy_kwargs_param_name', None)
        uploadTestCase.test_saved_password.__dict__.__setitem__('stypy_call_defaults', defaults)
        uploadTestCase.test_saved_password.__dict__.__setitem__('stypy_call_varargs', varargs)
        uploadTestCase.test_saved_password.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        uploadTestCase.test_saved_password.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'uploadTestCase.test_saved_password', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_saved_password', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_saved_password(...)' code ##################

        
        # Call to write_file(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'self' (line 90)
        self_45313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 24), 'self', False)
        # Obtaining the member 'rc' of a type (line 90)
        rc_45314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 24), self_45313, 'rc')
        # Getting the type of 'PYPIRC_NOPASSWORD' (line 90)
        PYPIRC_NOPASSWORD_45315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 33), 'PYPIRC_NOPASSWORD', False)
        # Processing the call keyword arguments (line 90)
        kwargs_45316 = {}
        # Getting the type of 'self' (line 90)
        self_45311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 90)
        write_file_45312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_45311, 'write_file')
        # Calling write_file(args, kwargs) (line 90)
        write_file_call_result_45317 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), write_file_45312, *[rc_45314, PYPIRC_NOPASSWORD_45315], **kwargs_45316)
        
        
        # Assigning a Call to a Name (line 93):
        
        # Assigning a Call to a Name (line 93):
        
        # Call to Distribution(...): (line 93)
        # Processing the call keyword arguments (line 93)
        kwargs_45319 = {}
        # Getting the type of 'Distribution' (line 93)
        Distribution_45318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 93)
        Distribution_call_result_45320 = invoke(stypy.reporting.localization.Localization(__file__, 93, 15), Distribution_45318, *[], **kwargs_45319)
        
        # Assigning a type to the variable 'dist' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'dist', Distribution_call_result_45320)
        
        # Assigning a Call to a Name (line 94):
        
        # Assigning a Call to a Name (line 94):
        
        # Call to upload(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'dist' (line 94)
        dist_45322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 21), 'dist', False)
        # Processing the call keyword arguments (line 94)
        kwargs_45323 = {}
        # Getting the type of 'upload' (line 94)
        upload_45321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 14), 'upload', False)
        # Calling upload(args, kwargs) (line 94)
        upload_call_result_45324 = invoke(stypy.reporting.localization.Localization(__file__, 94, 14), upload_45321, *[dist_45322], **kwargs_45323)
        
        # Assigning a type to the variable 'cmd' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'cmd', upload_call_result_45324)
        
        # Call to finalize_options(...): (line 95)
        # Processing the call keyword arguments (line 95)
        kwargs_45327 = {}
        # Getting the type of 'cmd' (line 95)
        cmd_45325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 95)
        finalize_options_45326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), cmd_45325, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 95)
        finalize_options_call_result_45328 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), finalize_options_45326, *[], **kwargs_45327)
        
        
        # Call to assertEqual(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'cmd' (line 96)
        cmd_45331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 25), 'cmd', False)
        # Obtaining the member 'password' of a type (line 96)
        password_45332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 25), cmd_45331, 'password')
        # Getting the type of 'None' (line 96)
        None_45333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 39), 'None', False)
        # Processing the call keyword arguments (line 96)
        kwargs_45334 = {}
        # Getting the type of 'self' (line 96)
        self_45329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 96)
        assertEqual_45330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_45329, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 96)
        assertEqual_call_result_45335 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), assertEqual_45330, *[password_45332, None_45333], **kwargs_45334)
        
        
        # Assigning a Str to a Attribute (line 100):
        
        # Assigning a Str to a Attribute (line 100):
        str_45336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 24), 'str', 'xxx')
        # Getting the type of 'dist' (line 100)
        dist_45337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'dist')
        # Setting the type of the member 'password' of a type (line 100)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), dist_45337, 'password', str_45336)
        
        # Assigning a Call to a Name (line 101):
        
        # Assigning a Call to a Name (line 101):
        
        # Call to upload(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'dist' (line 101)
        dist_45339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'dist', False)
        # Processing the call keyword arguments (line 101)
        kwargs_45340 = {}
        # Getting the type of 'upload' (line 101)
        upload_45338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 14), 'upload', False)
        # Calling upload(args, kwargs) (line 101)
        upload_call_result_45341 = invoke(stypy.reporting.localization.Localization(__file__, 101, 14), upload_45338, *[dist_45339], **kwargs_45340)
        
        # Assigning a type to the variable 'cmd' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'cmd', upload_call_result_45341)
        
        # Call to finalize_options(...): (line 102)
        # Processing the call keyword arguments (line 102)
        kwargs_45344 = {}
        # Getting the type of 'cmd' (line 102)
        cmd_45342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 102)
        finalize_options_45343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), cmd_45342, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 102)
        finalize_options_call_result_45345 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), finalize_options_45343, *[], **kwargs_45344)
        
        
        # Call to assertEqual(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'cmd' (line 103)
        cmd_45348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 25), 'cmd', False)
        # Obtaining the member 'password' of a type (line 103)
        password_45349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 25), cmd_45348, 'password')
        str_45350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 39), 'str', 'xxx')
        # Processing the call keyword arguments (line 103)
        kwargs_45351 = {}
        # Getting the type of 'self' (line 103)
        self_45346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 103)
        assertEqual_45347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), self_45346, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 103)
        assertEqual_call_result_45352 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), assertEqual_45347, *[password_45349, str_45350], **kwargs_45351)
        
        
        # ################# End of 'test_saved_password(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_saved_password' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_45353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_45353)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_saved_password'
        return stypy_return_type_45353


    @norecursion
    def test_upload(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_upload'
        module_type_store = module_type_store.open_function_context('test_upload', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        uploadTestCase.test_upload.__dict__.__setitem__('stypy_localization', localization)
        uploadTestCase.test_upload.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        uploadTestCase.test_upload.__dict__.__setitem__('stypy_type_store', module_type_store)
        uploadTestCase.test_upload.__dict__.__setitem__('stypy_function_name', 'uploadTestCase.test_upload')
        uploadTestCase.test_upload.__dict__.__setitem__('stypy_param_names_list', [])
        uploadTestCase.test_upload.__dict__.__setitem__('stypy_varargs_param_name', None)
        uploadTestCase.test_upload.__dict__.__setitem__('stypy_kwargs_param_name', None)
        uploadTestCase.test_upload.__dict__.__setitem__('stypy_call_defaults', defaults)
        uploadTestCase.test_upload.__dict__.__setitem__('stypy_call_varargs', varargs)
        uploadTestCase.test_upload.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        uploadTestCase.test_upload.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'uploadTestCase.test_upload', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_upload', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_upload(...)' code ##################

        
        # Assigning a Call to a Name (line 106):
        
        # Assigning a Call to a Name (line 106):
        
        # Call to mkdtemp(...): (line 106)
        # Processing the call keyword arguments (line 106)
        kwargs_45356 = {}
        # Getting the type of 'self' (line 106)
        self_45354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 106)
        mkdtemp_45355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 14), self_45354, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 106)
        mkdtemp_call_result_45357 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), mkdtemp_45355, *[], **kwargs_45356)
        
        # Assigning a type to the variable 'tmp' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tmp', mkdtemp_call_result_45357)
        
        # Assigning a Call to a Name (line 107):
        
        # Assigning a Call to a Name (line 107):
        
        # Call to join(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'tmp' (line 107)
        tmp_45361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 28), 'tmp', False)
        str_45362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 33), 'str', 'xxx')
        # Processing the call keyword arguments (line 107)
        kwargs_45363 = {}
        # Getting the type of 'os' (line 107)
        os_45358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 107)
        path_45359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 15), os_45358, 'path')
        # Obtaining the member 'join' of a type (line 107)
        join_45360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 15), path_45359, 'join')
        # Calling join(args, kwargs) (line 107)
        join_call_result_45364 = invoke(stypy.reporting.localization.Localization(__file__, 107, 15), join_45360, *[tmp_45361, str_45362], **kwargs_45363)
        
        # Assigning a type to the variable 'path' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'path', join_call_result_45364)
        
        # Call to write_file(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'path' (line 108)
        path_45367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'path', False)
        # Processing the call keyword arguments (line 108)
        kwargs_45368 = {}
        # Getting the type of 'self' (line 108)
        self_45365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 108)
        write_file_45366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), self_45365, 'write_file')
        # Calling write_file(args, kwargs) (line 108)
        write_file_call_result_45369 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), write_file_45366, *[path_45367], **kwargs_45368)
        
        
        # Assigning a Tuple to a Tuple (line 109):
        
        # Assigning a Str to a Name (line 109):
        str_45370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 39), 'str', 'xxx')
        # Assigning a type to the variable 'tuple_assignment_45177' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tuple_assignment_45177', str_45370)
        
        # Assigning a Str to a Name (line 109):
        str_45371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 46), 'str', '2.6')
        # Assigning a type to the variable 'tuple_assignment_45178' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tuple_assignment_45178', str_45371)
        
        # Assigning a Name to a Name (line 109):
        # Getting the type of 'path' (line 109)
        path_45372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 53), 'path')
        # Assigning a type to the variable 'tuple_assignment_45179' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tuple_assignment_45179', path_45372)
        
        # Assigning a Name to a Name (line 109):
        # Getting the type of 'tuple_assignment_45177' (line 109)
        tuple_assignment_45177_45373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tuple_assignment_45177')
        # Assigning a type to the variable 'command' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'command', tuple_assignment_45177_45373)
        
        # Assigning a Name to a Name (line 109):
        # Getting the type of 'tuple_assignment_45178' (line 109)
        tuple_assignment_45178_45374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tuple_assignment_45178')
        # Assigning a type to the variable 'pyversion' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 17), 'pyversion', tuple_assignment_45178_45374)
        
        # Assigning a Name to a Name (line 109):
        # Getting the type of 'tuple_assignment_45179' (line 109)
        tuple_assignment_45179_45375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tuple_assignment_45179')
        # Assigning a type to the variable 'filename' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 28), 'filename', tuple_assignment_45179_45375)
        
        # Assigning a List to a Name (line 110):
        
        # Assigning a List to a Name (line 110):
        
        # Obtaining an instance of the builtin type 'list' (line 110)
        list_45376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 110)
        # Adding element type (line 110)
        
        # Obtaining an instance of the builtin type 'tuple' (line 110)
        tuple_45377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 110)
        # Adding element type (line 110)
        # Getting the type of 'command' (line 110)
        command_45378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'command')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 23), tuple_45377, command_45378)
        # Adding element type (line 110)
        # Getting the type of 'pyversion' (line 110)
        pyversion_45379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 32), 'pyversion')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 23), tuple_45377, pyversion_45379)
        # Adding element type (line 110)
        # Getting the type of 'filename' (line 110)
        filename_45380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 43), 'filename')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 23), tuple_45377, filename_45380)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 21), list_45376, tuple_45377)
        
        # Assigning a type to the variable 'dist_files' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'dist_files', list_45376)
        
        # Call to write_file(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'self' (line 111)
        self_45383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'self', False)
        # Obtaining the member 'rc' of a type (line 111)
        rc_45384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 24), self_45383, 'rc')
        # Getting the type of 'PYPIRC_LONG_PASSWORD' (line 111)
        PYPIRC_LONG_PASSWORD_45385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 33), 'PYPIRC_LONG_PASSWORD', False)
        # Processing the call keyword arguments (line 111)
        kwargs_45386 = {}
        # Getting the type of 'self' (line 111)
        self_45381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 111)
        write_file_45382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_45381, 'write_file')
        # Calling write_file(args, kwargs) (line 111)
        write_file_call_result_45387 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), write_file_45382, *[rc_45384, PYPIRC_LONG_PASSWORD_45385], **kwargs_45386)
        
        
        # Assigning a Call to a Tuple (line 114):
        
        # Assigning a Subscript to a Name (line 114):
        
        # Obtaining the type of the subscript
        int_45388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 8), 'int')
        
        # Call to create_dist(...): (line 114)
        # Processing the call keyword arguments (line 114)
        # Getting the type of 'dist_files' (line 114)
        dist_files_45391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 52), 'dist_files', False)
        keyword_45392 = dist_files_45391
        unicode_45393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 71), 'unicode', u'd\xe9d\xe9')
        keyword_45394 = unicode_45393
        kwargs_45395 = {'dist_files': keyword_45392, 'author': keyword_45394}
        # Getting the type of 'self' (line 114)
        self_45389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 114)
        create_dist_45390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 24), self_45389, 'create_dist')
        # Calling create_dist(args, kwargs) (line 114)
        create_dist_call_result_45396 = invoke(stypy.reporting.localization.Localization(__file__, 114, 24), create_dist_45390, *[], **kwargs_45395)
        
        # Obtaining the member '__getitem__' of a type (line 114)
        getitem___45397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), create_dist_call_result_45396, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 114)
        subscript_call_result_45398 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), getitem___45397, int_45388)
        
        # Assigning a type to the variable 'tuple_var_assignment_45180' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'tuple_var_assignment_45180', subscript_call_result_45398)
        
        # Assigning a Subscript to a Name (line 114):
        
        # Obtaining the type of the subscript
        int_45399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 8), 'int')
        
        # Call to create_dist(...): (line 114)
        # Processing the call keyword arguments (line 114)
        # Getting the type of 'dist_files' (line 114)
        dist_files_45402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 52), 'dist_files', False)
        keyword_45403 = dist_files_45402
        unicode_45404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 71), 'unicode', u'd\xe9d\xe9')
        keyword_45405 = unicode_45404
        kwargs_45406 = {'dist_files': keyword_45403, 'author': keyword_45405}
        # Getting the type of 'self' (line 114)
        self_45400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 114)
        create_dist_45401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 24), self_45400, 'create_dist')
        # Calling create_dist(args, kwargs) (line 114)
        create_dist_call_result_45407 = invoke(stypy.reporting.localization.Localization(__file__, 114, 24), create_dist_45401, *[], **kwargs_45406)
        
        # Obtaining the member '__getitem__' of a type (line 114)
        getitem___45408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), create_dist_call_result_45407, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 114)
        subscript_call_result_45409 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), getitem___45408, int_45399)
        
        # Assigning a type to the variable 'tuple_var_assignment_45181' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'tuple_var_assignment_45181', subscript_call_result_45409)
        
        # Assigning a Name to a Name (line 114):
        # Getting the type of 'tuple_var_assignment_45180' (line 114)
        tuple_var_assignment_45180_45410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'tuple_var_assignment_45180')
        # Assigning a type to the variable 'pkg_dir' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'pkg_dir', tuple_var_assignment_45180_45410)
        
        # Assigning a Name to a Name (line 114):
        # Getting the type of 'tuple_var_assignment_45181' (line 114)
        tuple_var_assignment_45181_45411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'tuple_var_assignment_45181')
        # Assigning a type to the variable 'dist' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 17), 'dist', tuple_var_assignment_45181_45411)
        
        # Assigning a Call to a Name (line 115):
        
        # Assigning a Call to a Name (line 115):
        
        # Call to upload(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'dist' (line 115)
        dist_45413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 21), 'dist', False)
        # Processing the call keyword arguments (line 115)
        kwargs_45414 = {}
        # Getting the type of 'upload' (line 115)
        upload_45412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 14), 'upload', False)
        # Calling upload(args, kwargs) (line 115)
        upload_call_result_45415 = invoke(stypy.reporting.localization.Localization(__file__, 115, 14), upload_45412, *[dist_45413], **kwargs_45414)
        
        # Assigning a type to the variable 'cmd' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'cmd', upload_call_result_45415)
        
        # Call to ensure_finalized(...): (line 116)
        # Processing the call keyword arguments (line 116)
        kwargs_45418 = {}
        # Getting the type of 'cmd' (line 116)
        cmd_45416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 116)
        ensure_finalized_45417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), cmd_45416, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 116)
        ensure_finalized_call_result_45419 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), ensure_finalized_45417, *[], **kwargs_45418)
        
        
        # Call to run(...): (line 117)
        # Processing the call keyword arguments (line 117)
        kwargs_45422 = {}
        # Getting the type of 'cmd' (line 117)
        cmd_45420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'cmd', False)
        # Obtaining the member 'run' of a type (line 117)
        run_45421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), cmd_45420, 'run')
        # Calling run(args, kwargs) (line 117)
        run_call_result_45423 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), run_45421, *[], **kwargs_45422)
        
        
        # Call to assertIn(...): (line 120)
        # Processing the call arguments (line 120)
        str_45426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 22), 'str', 'd\xc3\xa9d\xc3\xa9')
        # Getting the type of 'self' (line 120)
        self_45427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 32), 'self', False)
        # Obtaining the member 'last_open' of a type (line 120)
        last_open_45428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 32), self_45427, 'last_open')
        # Obtaining the member 'req' of a type (line 120)
        req_45429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 32), last_open_45428, 'req')
        # Obtaining the member 'data' of a type (line 120)
        data_45430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 32), req_45429, 'data')
        # Processing the call keyword arguments (line 120)
        kwargs_45431 = {}
        # Getting the type of 'self' (line 120)
        self_45424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 120)
        assertIn_45425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), self_45424, 'assertIn')
        # Calling assertIn(args, kwargs) (line 120)
        assertIn_call_result_45432 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), assertIn_45425, *[str_45426, data_45430], **kwargs_45431)
        
        
        # Assigning a Call to a Name (line 121):
        
        # Assigning a Call to a Name (line 121):
        
        # Call to dict(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'self' (line 121)
        self_45434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'self', False)
        # Obtaining the member 'last_open' of a type (line 121)
        last_open_45435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 23), self_45434, 'last_open')
        # Obtaining the member 'req' of a type (line 121)
        req_45436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 23), last_open_45435, 'req')
        # Obtaining the member 'headers' of a type (line 121)
        headers_45437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 23), req_45436, 'headers')
        # Processing the call keyword arguments (line 121)
        kwargs_45438 = {}
        # Getting the type of 'dict' (line 121)
        dict_45433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 18), 'dict', False)
        # Calling dict(args, kwargs) (line 121)
        dict_call_result_45439 = invoke(stypy.reporting.localization.Localization(__file__, 121, 18), dict_45433, *[headers_45437], **kwargs_45438)
        
        # Assigning a type to the variable 'headers' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'headers', dict_call_result_45439)
        
        # Call to assertEqual(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Obtaining the type of the subscript
        str_45442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 33), 'str', 'Content-length')
        # Getting the type of 'headers' (line 122)
        headers_45443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 25), 'headers', False)
        # Obtaining the member '__getitem__' of a type (line 122)
        getitem___45444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 25), headers_45443, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
        subscript_call_result_45445 = invoke(stypy.reporting.localization.Localization(__file__, 122, 25), getitem___45444, str_45442)
        
        str_45446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 52), 'str', '2159')
        # Processing the call keyword arguments (line 122)
        kwargs_45447 = {}
        # Getting the type of 'self' (line 122)
        self_45440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 122)
        assertEqual_45441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), self_45440, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 122)
        assertEqual_call_result_45448 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), assertEqual_45441, *[subscript_call_result_45445, str_45446], **kwargs_45447)
        
        
        # Call to assertTrue(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Call to startswith(...): (line 123)
        # Processing the call arguments (line 123)
        str_45456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 59), 'str', 'multipart/form-data')
        # Processing the call keyword arguments (line 123)
        kwargs_45457 = {}
        
        # Obtaining the type of the subscript
        str_45451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 32), 'str', 'Content-type')
        # Getting the type of 'headers' (line 123)
        headers_45452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 24), 'headers', False)
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___45453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 24), headers_45452, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_45454 = invoke(stypy.reporting.localization.Localization(__file__, 123, 24), getitem___45453, str_45451)
        
        # Obtaining the member 'startswith' of a type (line 123)
        startswith_45455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 24), subscript_call_result_45454, 'startswith')
        # Calling startswith(args, kwargs) (line 123)
        startswith_call_result_45458 = invoke(stypy.reporting.localization.Localization(__file__, 123, 24), startswith_45455, *[str_45456], **kwargs_45457)
        
        # Processing the call keyword arguments (line 123)
        kwargs_45459 = {}
        # Getting the type of 'self' (line 123)
        self_45449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 123)
        assertTrue_45450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), self_45449, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 123)
        assertTrue_call_result_45460 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), assertTrue_45450, *[startswith_call_result_45458], **kwargs_45459)
        
        
        # Call to assertEqual(...): (line 124)
        # Processing the call arguments (line 124)
        
        # Call to get_method(...): (line 124)
        # Processing the call keyword arguments (line 124)
        kwargs_45467 = {}
        # Getting the type of 'self' (line 124)
        self_45463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 25), 'self', False)
        # Obtaining the member 'last_open' of a type (line 124)
        last_open_45464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 25), self_45463, 'last_open')
        # Obtaining the member 'req' of a type (line 124)
        req_45465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 25), last_open_45464, 'req')
        # Obtaining the member 'get_method' of a type (line 124)
        get_method_45466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 25), req_45465, 'get_method')
        # Calling get_method(args, kwargs) (line 124)
        get_method_call_result_45468 = invoke(stypy.reporting.localization.Localization(__file__, 124, 25), get_method_45466, *[], **kwargs_45467)
        
        str_45469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 58), 'str', 'POST')
        # Processing the call keyword arguments (line 124)
        kwargs_45470 = {}
        # Getting the type of 'self' (line 124)
        self_45461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 124)
        assertEqual_45462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), self_45461, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 124)
        assertEqual_call_result_45471 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), assertEqual_45462, *[get_method_call_result_45468, str_45469], **kwargs_45470)
        
        
        # Call to assertEqual(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Call to get_full_url(...): (line 125)
        # Processing the call keyword arguments (line 125)
        kwargs_45478 = {}
        # Getting the type of 'self' (line 125)
        self_45474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 25), 'self', False)
        # Obtaining the member 'last_open' of a type (line 125)
        last_open_45475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 25), self_45474, 'last_open')
        # Obtaining the member 'req' of a type (line 125)
        req_45476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 25), last_open_45475, 'req')
        # Obtaining the member 'get_full_url' of a type (line 125)
        get_full_url_45477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 25), req_45476, 'get_full_url')
        # Calling get_full_url(args, kwargs) (line 125)
        get_full_url_call_result_45479 = invoke(stypy.reporting.localization.Localization(__file__, 125, 25), get_full_url_45477, *[], **kwargs_45478)
        
        str_45480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 25), 'str', 'https://upload.pypi.org/legacy/')
        # Processing the call keyword arguments (line 125)
        kwargs_45481 = {}
        # Getting the type of 'self' (line 125)
        self_45472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 125)
        assertEqual_45473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), self_45472, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 125)
        assertEqual_call_result_45482 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), assertEqual_45473, *[get_full_url_call_result_45479, str_45480], **kwargs_45481)
        
        
        # Call to assertIn(...): (line 127)
        # Processing the call arguments (line 127)
        str_45485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 22), 'str', 'xxx')
        # Getting the type of 'self' (line 127)
        self_45486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 29), 'self', False)
        # Obtaining the member 'last_open' of a type (line 127)
        last_open_45487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 29), self_45486, 'last_open')
        # Obtaining the member 'req' of a type (line 127)
        req_45488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 29), last_open_45487, 'req')
        # Obtaining the member 'data' of a type (line 127)
        data_45489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 29), req_45488, 'data')
        # Processing the call keyword arguments (line 127)
        kwargs_45490 = {}
        # Getting the type of 'self' (line 127)
        self_45483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 127)
        assertIn_45484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), self_45483, 'assertIn')
        # Calling assertIn(args, kwargs) (line 127)
        assertIn_call_result_45491 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), assertIn_45484, *[str_45485, data_45489], **kwargs_45490)
        
        
        # Assigning a Subscript to a Name (line 128):
        
        # Assigning a Subscript to a Name (line 128):
        
        # Obtaining the type of the subscript
        str_45492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 42), 'str', 'Authorization')
        # Getting the type of 'self' (line 128)
        self_45493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 'self')
        # Obtaining the member 'last_open' of a type (line 128)
        last_open_45494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 15), self_45493, 'last_open')
        # Obtaining the member 'req' of a type (line 128)
        req_45495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 15), last_open_45494, 'req')
        # Obtaining the member 'headers' of a type (line 128)
        headers_45496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 15), req_45495, 'headers')
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___45497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 15), headers_45496, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_45498 = invoke(stypy.reporting.localization.Localization(__file__, 128, 15), getitem___45497, str_45492)
        
        # Assigning a type to the variable 'auth' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'auth', subscript_call_result_45498)
        
        # Call to assertNotIn(...): (line 129)
        # Processing the call arguments (line 129)
        str_45501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 25), 'str', '\n')
        # Getting the type of 'auth' (line 129)
        auth_45502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 31), 'auth', False)
        # Processing the call keyword arguments (line 129)
        kwargs_45503 = {}
        # Getting the type of 'self' (line 129)
        self_45499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self', False)
        # Obtaining the member 'assertNotIn' of a type (line 129)
        assertNotIn_45500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_45499, 'assertNotIn')
        # Calling assertNotIn(args, kwargs) (line 129)
        assertNotIn_call_result_45504 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), assertNotIn_45500, *[str_45501, auth_45502], **kwargs_45503)
        
        
        # ################# End of 'test_upload(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_upload' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_45505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_45505)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_upload'
        return stypy_return_type_45505


    @norecursion
    def test_upload_fails(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_upload_fails'
        module_type_store = module_type_store.open_function_context('test_upload_fails', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        uploadTestCase.test_upload_fails.__dict__.__setitem__('stypy_localization', localization)
        uploadTestCase.test_upload_fails.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        uploadTestCase.test_upload_fails.__dict__.__setitem__('stypy_type_store', module_type_store)
        uploadTestCase.test_upload_fails.__dict__.__setitem__('stypy_function_name', 'uploadTestCase.test_upload_fails')
        uploadTestCase.test_upload_fails.__dict__.__setitem__('stypy_param_names_list', [])
        uploadTestCase.test_upload_fails.__dict__.__setitem__('stypy_varargs_param_name', None)
        uploadTestCase.test_upload_fails.__dict__.__setitem__('stypy_kwargs_param_name', None)
        uploadTestCase.test_upload_fails.__dict__.__setitem__('stypy_call_defaults', defaults)
        uploadTestCase.test_upload_fails.__dict__.__setitem__('stypy_call_varargs', varargs)
        uploadTestCase.test_upload_fails.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        uploadTestCase.test_upload_fails.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'uploadTestCase.test_upload_fails', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_upload_fails', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_upload_fails(...)' code ##################

        
        # Assigning a Str to a Attribute (line 132):
        
        # Assigning a Str to a Attribute (line 132):
        str_45506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 24), 'str', 'Not Found')
        # Getting the type of 'self' (line 132)
        self_45507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'self')
        # Setting the type of the member 'next_msg' of a type (line 132)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), self_45507, 'next_msg', str_45506)
        
        # Assigning a Num to a Attribute (line 133):
        
        # Assigning a Num to a Attribute (line 133):
        int_45508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 25), 'int')
        # Getting the type of 'self' (line 133)
        self_45509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'self')
        # Setting the type of the member 'next_code' of a type (line 133)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), self_45509, 'next_code', int_45508)
        
        # Call to assertRaises(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'DistutilsError' (line 134)
        DistutilsError_45512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 26), 'DistutilsError', False)
        # Getting the type of 'self' (line 134)
        self_45513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 42), 'self', False)
        # Obtaining the member 'test_upload' of a type (line 134)
        test_upload_45514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 42), self_45513, 'test_upload')
        # Processing the call keyword arguments (line 134)
        kwargs_45515 = {}
        # Getting the type of 'self' (line 134)
        self_45510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 134)
        assertRaises_45511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), self_45510, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 134)
        assertRaises_call_result_45516 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), assertRaises_45511, *[DistutilsError_45512, test_upload_45514], **kwargs_45515)
        
        
        # ################# End of 'test_upload_fails(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_upload_fails' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_45517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_45517)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_upload_fails'
        return stypy_return_type_45517


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 58, 0, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'uploadTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'uploadTestCase' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'uploadTestCase', uploadTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 136, 0, False)
    
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

    
    # Call to makeSuite(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'uploadTestCase' (line 137)
    uploadTestCase_45520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 30), 'uploadTestCase', False)
    # Processing the call keyword arguments (line 137)
    kwargs_45521 = {}
    # Getting the type of 'unittest' (line 137)
    unittest_45518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 137)
    makeSuite_45519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 11), unittest_45518, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 137)
    makeSuite_call_result_45522 = invoke(stypy.reporting.localization.Localization(__file__, 137, 11), makeSuite_45519, *[uploadTestCase_45520], **kwargs_45521)
    
    # Assigning a type to the variable 'stypy_return_type' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type', makeSuite_call_result_45522)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 136)
    stypy_return_type_45523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_45523)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_45523

# Assigning a type to the variable 'test_suite' (line 136)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 140)
    # Processing the call arguments (line 140)
    
    # Call to test_suite(...): (line 140)
    # Processing the call keyword arguments (line 140)
    kwargs_45526 = {}
    # Getting the type of 'test_suite' (line 140)
    test_suite_45525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 140)
    test_suite_call_result_45527 = invoke(stypy.reporting.localization.Localization(__file__, 140, 17), test_suite_45525, *[], **kwargs_45526)
    
    # Processing the call keyword arguments (line 140)
    kwargs_45528 = {}
    # Getting the type of 'run_unittest' (line 140)
    run_unittest_45524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 140)
    run_unittest_call_result_45529 = invoke(stypy.reporting.localization.Localization(__file__, 140, 4), run_unittest_45524, *[test_suite_call_result_45527], **kwargs_45528)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
