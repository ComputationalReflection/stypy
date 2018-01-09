
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.pypirc.pypirc.'''
2: import sys
3: import os
4: import unittest
5: import tempfile
6: import shutil
7: 
8: from distutils.core import PyPIRCCommand
9: from distutils.core import Distribution
10: from distutils.log import set_threshold
11: from distutils.log import WARN
12: 
13: from distutils.tests import support
14: from test.test_support import run_unittest
15: 
16: PYPIRC = '''\
17: [distutils]
18: 
19: index-servers =
20:     server1
21:     server2
22: 
23: [server1]
24: username:me
25: password:secret
26: 
27: [server2]
28: username:meagain
29: password: secret
30: realm:acme
31: repository:http://another.pypi/
32: '''
33: 
34: PYPIRC_OLD = '''\
35: [server-login]
36: username:tarek
37: password:secret
38: '''
39: 
40: WANTED = '''\
41: [distutils]
42: index-servers =
43:     pypi
44: 
45: [pypi]
46: username:tarek
47: password:xxx
48: '''
49: 
50: 
51: class PyPIRCCommandTestCase(support.TempdirManager,
52:                             support.LoggingSilencer,
53:                             support.EnvironGuard,
54:                             unittest.TestCase):
55: 
56:     def setUp(self):
57:         '''Patches the environment.'''
58:         super(PyPIRCCommandTestCase, self).setUp()
59:         self.tmp_dir = self.mkdtemp()
60:         os.environ['HOME'] = self.tmp_dir
61:         self.rc = os.path.join(self.tmp_dir, '.pypirc')
62:         self.dist = Distribution()
63: 
64:         class command(PyPIRCCommand):
65:             def __init__(self, dist):
66:                 PyPIRCCommand.__init__(self, dist)
67:             def initialize_options(self):
68:                 pass
69:             finalize_options = initialize_options
70: 
71:         self._cmd = command
72:         self.old_threshold = set_threshold(WARN)
73: 
74:     def tearDown(self):
75:         '''Removes the patch.'''
76:         set_threshold(self.old_threshold)
77:         super(PyPIRCCommandTestCase, self).tearDown()
78: 
79:     def test_server_registration(self):
80:         # This test makes sure PyPIRCCommand knows how to:
81:         # 1. handle several sections in .pypirc
82:         # 2. handle the old format
83: 
84:         # new format
85:         self.write_file(self.rc, PYPIRC)
86:         cmd = self._cmd(self.dist)
87:         config = cmd._read_pypirc()
88: 
89:         config = config.items()
90:         config.sort()
91:         waited = [('password', 'secret'), ('realm', 'pypi'),
92:                   ('repository', 'https://upload.pypi.org/legacy/'),
93:                   ('server', 'server1'), ('username', 'me')]
94:         self.assertEqual(config, waited)
95: 
96:         # old format
97:         self.write_file(self.rc, PYPIRC_OLD)
98:         config = cmd._read_pypirc()
99:         config = config.items()
100:         config.sort()
101:         waited = [('password', 'secret'), ('realm', 'pypi'),
102:                   ('repository', 'https://upload.pypi.org/legacy/'),
103:                   ('server', 'server-login'), ('username', 'tarek')]
104:         self.assertEqual(config, waited)
105: 
106:     def test_server_empty_registration(self):
107:         cmd = self._cmd(self.dist)
108:         rc = cmd._get_rc_file()
109:         self.assertFalse(os.path.exists(rc))
110:         cmd._store_pypirc('tarek', 'xxx')
111:         self.assertTrue(os.path.exists(rc))
112:         f = open(rc)
113:         try:
114:             content = f.read()
115:             self.assertEqual(content, WANTED)
116:         finally:
117:             f.close()
118: 
119: def test_suite():
120:     return unittest.makeSuite(PyPIRCCommandTestCase)
121: 
122: if __name__ == "__main__":
123:     run_unittest(test_suite())
124: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_35086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.pypirc.pypirc.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import sys' statement (line 2)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import unittest' statement (line 4)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import tempfile' statement (line 5)
import tempfile

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'tempfile', tempfile, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import shutil' statement (line 6)
import shutil

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'shutil', shutil, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.core import PyPIRCCommand' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_35087 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.core')

if (type(import_35087) is not StypyTypeError):

    if (import_35087 != 'pyd_module'):
        __import__(import_35087)
        sys_modules_35088 = sys.modules[import_35087]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.core', sys_modules_35088.module_type_store, module_type_store, ['PyPIRCCommand'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_35088, sys_modules_35088.module_type_store, module_type_store)
    else:
        from distutils.core import PyPIRCCommand

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.core', None, module_type_store, ['PyPIRCCommand'], [PyPIRCCommand])

else:
    # Assigning a type to the variable 'distutils.core' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.core', import_35087)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils.core import Distribution' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_35089 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.core')

if (type(import_35089) is not StypyTypeError):

    if (import_35089 != 'pyd_module'):
        __import__(import_35089)
        sys_modules_35090 = sys.modules[import_35089]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.core', sys_modules_35090.module_type_store, module_type_store, ['Distribution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_35090, sys_modules_35090.module_type_store, module_type_store)
    else:
        from distutils.core import Distribution

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.core', None, module_type_store, ['Distribution'], [Distribution])

else:
    # Assigning a type to the variable 'distutils.core' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.core', import_35089)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils.log import set_threshold' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_35091 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.log')

if (type(import_35091) is not StypyTypeError):

    if (import_35091 != 'pyd_module'):
        __import__(import_35091)
        sys_modules_35092 = sys.modules[import_35091]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.log', sys_modules_35092.module_type_store, module_type_store, ['set_threshold'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_35092, sys_modules_35092.module_type_store, module_type_store)
    else:
        from distutils.log import set_threshold

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.log', None, module_type_store, ['set_threshold'], [set_threshold])

else:
    # Assigning a type to the variable 'distutils.log' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.log', import_35091)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.log import WARN' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_35093 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.log')

if (type(import_35093) is not StypyTypeError):

    if (import_35093 != 'pyd_module'):
        __import__(import_35093)
        sys_modules_35094 = sys.modules[import_35093]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.log', sys_modules_35094.module_type_store, module_type_store, ['WARN'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_35094, sys_modules_35094.module_type_store, module_type_store)
    else:
        from distutils.log import WARN

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.log', None, module_type_store, ['WARN'], [WARN])

else:
    # Assigning a type to the variable 'distutils.log' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.log', import_35093)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from distutils.tests import support' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_35095 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.tests')

if (type(import_35095) is not StypyTypeError):

    if (import_35095 != 'pyd_module'):
        __import__(import_35095)
        sys_modules_35096 = sys.modules[import_35095]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.tests', sys_modules_35096.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_35096, sys_modules_35096.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.tests', import_35095)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from test.test_support import run_unittest' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_35097 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'test.test_support')

if (type(import_35097) is not StypyTypeError):

    if (import_35097 != 'pyd_module'):
        __import__(import_35097)
        sys_modules_35098 = sys.modules[import_35097]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'test.test_support', sys_modules_35098.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_35098, sys_modules_35098.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'test.test_support', import_35097)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')


# Assigning a Str to a Name (line 16):
str_35099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, (-1)), 'str', '[distutils]\n\nindex-servers =\n    server1\n    server2\n\n[server1]\nusername:me\npassword:secret\n\n[server2]\nusername:meagain\npassword: secret\nrealm:acme\nrepository:http://another.pypi/\n')
# Assigning a type to the variable 'PYPIRC' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'PYPIRC', str_35099)

# Assigning a Str to a Name (line 34):
str_35100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, (-1)), 'str', '[server-login]\nusername:tarek\npassword:secret\n')
# Assigning a type to the variable 'PYPIRC_OLD' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'PYPIRC_OLD', str_35100)

# Assigning a Str to a Name (line 40):
str_35101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, (-1)), 'str', '[distutils]\nindex-servers =\n    pypi\n\n[pypi]\nusername:tarek\npassword:xxx\n')
# Assigning a type to the variable 'WANTED' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'WANTED', str_35101)
# Declaration of the 'PyPIRCCommandTestCase' class
# Getting the type of 'support' (line 51)
support_35102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'support')
# Obtaining the member 'TempdirManager' of a type (line 51)
TempdirManager_35103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 28), support_35102, 'TempdirManager')
# Getting the type of 'support' (line 52)
support_35104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 28), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 52)
LoggingSilencer_35105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 28), support_35104, 'LoggingSilencer')
# Getting the type of 'support' (line 53)
support_35106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 28), 'support')
# Obtaining the member 'EnvironGuard' of a type (line 53)
EnvironGuard_35107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 28), support_35106, 'EnvironGuard')
# Getting the type of 'unittest' (line 54)
unittest_35108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'unittest')
# Obtaining the member 'TestCase' of a type (line 54)
TestCase_35109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 28), unittest_35108, 'TestCase')

class PyPIRCCommandTestCase(TempdirManager_35103, LoggingSilencer_35105, EnvironGuard_35107, TestCase_35109, ):

    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PyPIRCCommandTestCase.setUp.__dict__.__setitem__('stypy_localization', localization)
        PyPIRCCommandTestCase.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PyPIRCCommandTestCase.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        PyPIRCCommandTestCase.setUp.__dict__.__setitem__('stypy_function_name', 'PyPIRCCommandTestCase.setUp')
        PyPIRCCommandTestCase.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        PyPIRCCommandTestCase.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        PyPIRCCommandTestCase.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PyPIRCCommandTestCase.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        PyPIRCCommandTestCase.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        PyPIRCCommandTestCase.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PyPIRCCommandTestCase.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PyPIRCCommandTestCase.setUp', [], None, None, defaults, varargs, kwargs)

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

        str_35110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 8), 'str', 'Patches the environment.')
        
        # Call to setUp(...): (line 58)
        # Processing the call keyword arguments (line 58)
        kwargs_35117 = {}
        
        # Call to super(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'PyPIRCCommandTestCase' (line 58)
        PyPIRCCommandTestCase_35112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 14), 'PyPIRCCommandTestCase', False)
        # Getting the type of 'self' (line 58)
        self_35113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 37), 'self', False)
        # Processing the call keyword arguments (line 58)
        kwargs_35114 = {}
        # Getting the type of 'super' (line 58)
        super_35111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'super', False)
        # Calling super(args, kwargs) (line 58)
        super_call_result_35115 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), super_35111, *[PyPIRCCommandTestCase_35112, self_35113], **kwargs_35114)
        
        # Obtaining the member 'setUp' of a type (line 58)
        setUp_35116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), super_call_result_35115, 'setUp')
        # Calling setUp(args, kwargs) (line 58)
        setUp_call_result_35118 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), setUp_35116, *[], **kwargs_35117)
        
        
        # Assigning a Call to a Attribute (line 59):
        
        # Call to mkdtemp(...): (line 59)
        # Processing the call keyword arguments (line 59)
        kwargs_35121 = {}
        # Getting the type of 'self' (line 59)
        self_35119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 23), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 59)
        mkdtemp_35120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 23), self_35119, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 59)
        mkdtemp_call_result_35122 = invoke(stypy.reporting.localization.Localization(__file__, 59, 23), mkdtemp_35120, *[], **kwargs_35121)
        
        # Getting the type of 'self' (line 59)
        self_35123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self')
        # Setting the type of the member 'tmp_dir' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_35123, 'tmp_dir', mkdtemp_call_result_35122)
        
        # Assigning a Attribute to a Subscript (line 60):
        # Getting the type of 'self' (line 60)
        self_35124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 29), 'self')
        # Obtaining the member 'tmp_dir' of a type (line 60)
        tmp_dir_35125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 29), self_35124, 'tmp_dir')
        # Getting the type of 'os' (line 60)
        os_35126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'os')
        # Obtaining the member 'environ' of a type (line 60)
        environ_35127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), os_35126, 'environ')
        str_35128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 19), 'str', 'HOME')
        # Storing an element on a container (line 60)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 8), environ_35127, (str_35128, tmp_dir_35125))
        
        # Assigning a Call to a Attribute (line 61):
        
        # Call to join(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'self' (line 61)
        self_35132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 31), 'self', False)
        # Obtaining the member 'tmp_dir' of a type (line 61)
        tmp_dir_35133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 31), self_35132, 'tmp_dir')
        str_35134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 45), 'str', '.pypirc')
        # Processing the call keyword arguments (line 61)
        kwargs_35135 = {}
        # Getting the type of 'os' (line 61)
        os_35129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 61)
        path_35130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 18), os_35129, 'path')
        # Obtaining the member 'join' of a type (line 61)
        join_35131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 18), path_35130, 'join')
        # Calling join(args, kwargs) (line 61)
        join_call_result_35136 = invoke(stypy.reporting.localization.Localization(__file__, 61, 18), join_35131, *[tmp_dir_35133, str_35134], **kwargs_35135)
        
        # Getting the type of 'self' (line 61)
        self_35137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self')
        # Setting the type of the member 'rc' of a type (line 61)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_35137, 'rc', join_call_result_35136)
        
        # Assigning a Call to a Attribute (line 62):
        
        # Call to Distribution(...): (line 62)
        # Processing the call keyword arguments (line 62)
        kwargs_35139 = {}
        # Getting the type of 'Distribution' (line 62)
        Distribution_35138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 62)
        Distribution_call_result_35140 = invoke(stypy.reporting.localization.Localization(__file__, 62, 20), Distribution_35138, *[], **kwargs_35139)
        
        # Getting the type of 'self' (line 62)
        self_35141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self')
        # Setting the type of the member 'dist' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_35141, 'dist', Distribution_call_result_35140)
        # Declaration of the 'command' class
        # Getting the type of 'PyPIRCCommand' (line 64)
        PyPIRCCommand_35142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 22), 'PyPIRCCommand')

        class command(PyPIRCCommand_35142, ):

            @norecursion
            def __init__(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '__init__'
                module_type_store = module_type_store.open_function_context('__init__', 65, 12, False)
                # Assigning a type to the variable 'self' (line 66)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'command.__init__', ['dist'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return

                # Initialize method data
                init_call_information(module_type_store, '__init__', localization, ['dist'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of '__init__(...)' code ##################

                
                # Call to __init__(...): (line 66)
                # Processing the call arguments (line 66)
                # Getting the type of 'self' (line 66)
                self_35145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 39), 'self', False)
                # Getting the type of 'dist' (line 66)
                dist_35146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 45), 'dist', False)
                # Processing the call keyword arguments (line 66)
                kwargs_35147 = {}
                # Getting the type of 'PyPIRCCommand' (line 66)
                PyPIRCCommand_35143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'PyPIRCCommand', False)
                # Obtaining the member '__init__' of a type (line 66)
                init___35144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 16), PyPIRCCommand_35143, '__init__')
                # Calling __init__(args, kwargs) (line 66)
                init___call_result_35148 = invoke(stypy.reporting.localization.Localization(__file__, 66, 16), init___35144, *[self_35145, dist_35146], **kwargs_35147)
                
                
                # ################# End of '__init__(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()


            @norecursion
            def initialize_options(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'initialize_options'
                module_type_store = module_type_store.open_function_context('initialize_options', 67, 12, False)
                # Assigning a type to the variable 'self' (line 68)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                command.initialize_options.__dict__.__setitem__('stypy_localization', localization)
                command.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                command.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
                command.initialize_options.__dict__.__setitem__('stypy_function_name', 'command.initialize_options')
                command.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
                command.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
                command.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
                command.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
                command.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
                command.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                command.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'command.initialize_options', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'initialize_options', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'initialize_options(...)' code ##################

                pass
                
                # ################# End of 'initialize_options(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'initialize_options' in the type store
                # Getting the type of 'stypy_return_type' (line 67)
                stypy_return_type_35149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_35149)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'initialize_options'
                return stypy_return_type_35149

            
            # Assigning a Name to a Name (line 69):
            # Getting the type of 'initialize_options' (line 69)
            initialize_options_35150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 31), 'initialize_options')
            # Assigning a type to the variable 'finalize_options' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'finalize_options', initialize_options_35150)
        
        # Assigning a type to the variable 'command' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'command', command)
        
        # Assigning a Name to a Attribute (line 71):
        # Getting the type of 'command' (line 71)
        command_35151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 20), 'command')
        # Getting the type of 'self' (line 71)
        self_35152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'self')
        # Setting the type of the member '_cmd' of a type (line 71)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), self_35152, '_cmd', command_35151)
        
        # Assigning a Call to a Attribute (line 72):
        
        # Call to set_threshold(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'WARN' (line 72)
        WARN_35154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 43), 'WARN', False)
        # Processing the call keyword arguments (line 72)
        kwargs_35155 = {}
        # Getting the type of 'set_threshold' (line 72)
        set_threshold_35153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 29), 'set_threshold', False)
        # Calling set_threshold(args, kwargs) (line 72)
        set_threshold_call_result_35156 = invoke(stypy.reporting.localization.Localization(__file__, 72, 29), set_threshold_35153, *[WARN_35154], **kwargs_35155)
        
        # Getting the type of 'self' (line 72)
        self_35157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self')
        # Setting the type of the member 'old_threshold' of a type (line 72)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_35157, 'old_threshold', set_threshold_call_result_35156)
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_35158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35158)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_35158


    @norecursion
    def tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDown'
        module_type_store = module_type_store.open_function_context('tearDown', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PyPIRCCommandTestCase.tearDown.__dict__.__setitem__('stypy_localization', localization)
        PyPIRCCommandTestCase.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PyPIRCCommandTestCase.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        PyPIRCCommandTestCase.tearDown.__dict__.__setitem__('stypy_function_name', 'PyPIRCCommandTestCase.tearDown')
        PyPIRCCommandTestCase.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        PyPIRCCommandTestCase.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        PyPIRCCommandTestCase.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PyPIRCCommandTestCase.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        PyPIRCCommandTestCase.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        PyPIRCCommandTestCase.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PyPIRCCommandTestCase.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PyPIRCCommandTestCase.tearDown', [], None, None, defaults, varargs, kwargs)

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

        str_35159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 8), 'str', 'Removes the patch.')
        
        # Call to set_threshold(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'self' (line 76)
        self_35161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 22), 'self', False)
        # Obtaining the member 'old_threshold' of a type (line 76)
        old_threshold_35162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 22), self_35161, 'old_threshold')
        # Processing the call keyword arguments (line 76)
        kwargs_35163 = {}
        # Getting the type of 'set_threshold' (line 76)
        set_threshold_35160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'set_threshold', False)
        # Calling set_threshold(args, kwargs) (line 76)
        set_threshold_call_result_35164 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), set_threshold_35160, *[old_threshold_35162], **kwargs_35163)
        
        
        # Call to tearDown(...): (line 77)
        # Processing the call keyword arguments (line 77)
        kwargs_35171 = {}
        
        # Call to super(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'PyPIRCCommandTestCase' (line 77)
        PyPIRCCommandTestCase_35166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 14), 'PyPIRCCommandTestCase', False)
        # Getting the type of 'self' (line 77)
        self_35167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 37), 'self', False)
        # Processing the call keyword arguments (line 77)
        kwargs_35168 = {}
        # Getting the type of 'super' (line 77)
        super_35165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'super', False)
        # Calling super(args, kwargs) (line 77)
        super_call_result_35169 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), super_35165, *[PyPIRCCommandTestCase_35166, self_35167], **kwargs_35168)
        
        # Obtaining the member 'tearDown' of a type (line 77)
        tearDown_35170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), super_call_result_35169, 'tearDown')
        # Calling tearDown(args, kwargs) (line 77)
        tearDown_call_result_35172 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), tearDown_35170, *[], **kwargs_35171)
        
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_35173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35173)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_35173


    @norecursion
    def test_server_registration(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_server_registration'
        module_type_store = module_type_store.open_function_context('test_server_registration', 79, 4, False)
        # Assigning a type to the variable 'self' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PyPIRCCommandTestCase.test_server_registration.__dict__.__setitem__('stypy_localization', localization)
        PyPIRCCommandTestCase.test_server_registration.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PyPIRCCommandTestCase.test_server_registration.__dict__.__setitem__('stypy_type_store', module_type_store)
        PyPIRCCommandTestCase.test_server_registration.__dict__.__setitem__('stypy_function_name', 'PyPIRCCommandTestCase.test_server_registration')
        PyPIRCCommandTestCase.test_server_registration.__dict__.__setitem__('stypy_param_names_list', [])
        PyPIRCCommandTestCase.test_server_registration.__dict__.__setitem__('stypy_varargs_param_name', None)
        PyPIRCCommandTestCase.test_server_registration.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PyPIRCCommandTestCase.test_server_registration.__dict__.__setitem__('stypy_call_defaults', defaults)
        PyPIRCCommandTestCase.test_server_registration.__dict__.__setitem__('stypy_call_varargs', varargs)
        PyPIRCCommandTestCase.test_server_registration.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PyPIRCCommandTestCase.test_server_registration.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PyPIRCCommandTestCase.test_server_registration', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_server_registration', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_server_registration(...)' code ##################

        
        # Call to write_file(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'self' (line 85)
        self_35176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 24), 'self', False)
        # Obtaining the member 'rc' of a type (line 85)
        rc_35177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 24), self_35176, 'rc')
        # Getting the type of 'PYPIRC' (line 85)
        PYPIRC_35178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 33), 'PYPIRC', False)
        # Processing the call keyword arguments (line 85)
        kwargs_35179 = {}
        # Getting the type of 'self' (line 85)
        self_35174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 85)
        write_file_35175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), self_35174, 'write_file')
        # Calling write_file(args, kwargs) (line 85)
        write_file_call_result_35180 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), write_file_35175, *[rc_35177, PYPIRC_35178], **kwargs_35179)
        
        
        # Assigning a Call to a Name (line 86):
        
        # Call to _cmd(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'self' (line 86)
        self_35183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 24), 'self', False)
        # Obtaining the member 'dist' of a type (line 86)
        dist_35184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 24), self_35183, 'dist')
        # Processing the call keyword arguments (line 86)
        kwargs_35185 = {}
        # Getting the type of 'self' (line 86)
        self_35181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 14), 'self', False)
        # Obtaining the member '_cmd' of a type (line 86)
        _cmd_35182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 14), self_35181, '_cmd')
        # Calling _cmd(args, kwargs) (line 86)
        _cmd_call_result_35186 = invoke(stypy.reporting.localization.Localization(__file__, 86, 14), _cmd_35182, *[dist_35184], **kwargs_35185)
        
        # Assigning a type to the variable 'cmd' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'cmd', _cmd_call_result_35186)
        
        # Assigning a Call to a Name (line 87):
        
        # Call to _read_pypirc(...): (line 87)
        # Processing the call keyword arguments (line 87)
        kwargs_35189 = {}
        # Getting the type of 'cmd' (line 87)
        cmd_35187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 17), 'cmd', False)
        # Obtaining the member '_read_pypirc' of a type (line 87)
        _read_pypirc_35188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 17), cmd_35187, '_read_pypirc')
        # Calling _read_pypirc(args, kwargs) (line 87)
        _read_pypirc_call_result_35190 = invoke(stypy.reporting.localization.Localization(__file__, 87, 17), _read_pypirc_35188, *[], **kwargs_35189)
        
        # Assigning a type to the variable 'config' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'config', _read_pypirc_call_result_35190)
        
        # Assigning a Call to a Name (line 89):
        
        # Call to items(...): (line 89)
        # Processing the call keyword arguments (line 89)
        kwargs_35193 = {}
        # Getting the type of 'config' (line 89)
        config_35191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 17), 'config', False)
        # Obtaining the member 'items' of a type (line 89)
        items_35192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 17), config_35191, 'items')
        # Calling items(args, kwargs) (line 89)
        items_call_result_35194 = invoke(stypy.reporting.localization.Localization(__file__, 89, 17), items_35192, *[], **kwargs_35193)
        
        # Assigning a type to the variable 'config' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'config', items_call_result_35194)
        
        # Call to sort(...): (line 90)
        # Processing the call keyword arguments (line 90)
        kwargs_35197 = {}
        # Getting the type of 'config' (line 90)
        config_35195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'config', False)
        # Obtaining the member 'sort' of a type (line 90)
        sort_35196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), config_35195, 'sort')
        # Calling sort(args, kwargs) (line 90)
        sort_call_result_35198 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), sort_35196, *[], **kwargs_35197)
        
        
        # Assigning a List to a Name (line 91):
        
        # Obtaining an instance of the builtin type 'list' (line 91)
        list_35199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 91)
        # Adding element type (line 91)
        
        # Obtaining an instance of the builtin type 'tuple' (line 91)
        tuple_35200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 91)
        # Adding element type (line 91)
        str_35201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 19), 'str', 'password')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 19), tuple_35200, str_35201)
        # Adding element type (line 91)
        str_35202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 31), 'str', 'secret')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 19), tuple_35200, str_35202)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 17), list_35199, tuple_35200)
        # Adding element type (line 91)
        
        # Obtaining an instance of the builtin type 'tuple' (line 91)
        tuple_35203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 91)
        # Adding element type (line 91)
        str_35204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 43), 'str', 'realm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 43), tuple_35203, str_35204)
        # Adding element type (line 91)
        str_35205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 52), 'str', 'pypi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 43), tuple_35203, str_35205)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 17), list_35199, tuple_35203)
        # Adding element type (line 91)
        
        # Obtaining an instance of the builtin type 'tuple' (line 92)
        tuple_35206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 92)
        # Adding element type (line 92)
        str_35207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 19), 'str', 'repository')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 19), tuple_35206, str_35207)
        # Adding element type (line 92)
        str_35208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 33), 'str', 'https://upload.pypi.org/legacy/')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 19), tuple_35206, str_35208)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 17), list_35199, tuple_35206)
        # Adding element type (line 91)
        
        # Obtaining an instance of the builtin type 'tuple' (line 93)
        tuple_35209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 93)
        # Adding element type (line 93)
        str_35210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 19), 'str', 'server')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 19), tuple_35209, str_35210)
        # Adding element type (line 93)
        str_35211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 29), 'str', 'server1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 19), tuple_35209, str_35211)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 17), list_35199, tuple_35209)
        # Adding element type (line 91)
        
        # Obtaining an instance of the builtin type 'tuple' (line 93)
        tuple_35212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 93)
        # Adding element type (line 93)
        str_35213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 42), 'str', 'username')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 42), tuple_35212, str_35213)
        # Adding element type (line 93)
        str_35214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 54), 'str', 'me')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 42), tuple_35212, str_35214)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 17), list_35199, tuple_35212)
        
        # Assigning a type to the variable 'waited' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'waited', list_35199)
        
        # Call to assertEqual(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'config' (line 94)
        config_35217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 25), 'config', False)
        # Getting the type of 'waited' (line 94)
        waited_35218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 33), 'waited', False)
        # Processing the call keyword arguments (line 94)
        kwargs_35219 = {}
        # Getting the type of 'self' (line 94)
        self_35215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 94)
        assertEqual_35216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_35215, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 94)
        assertEqual_call_result_35220 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), assertEqual_35216, *[config_35217, waited_35218], **kwargs_35219)
        
        
        # Call to write_file(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'self' (line 97)
        self_35223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'self', False)
        # Obtaining the member 'rc' of a type (line 97)
        rc_35224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 24), self_35223, 'rc')
        # Getting the type of 'PYPIRC_OLD' (line 97)
        PYPIRC_OLD_35225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 33), 'PYPIRC_OLD', False)
        # Processing the call keyword arguments (line 97)
        kwargs_35226 = {}
        # Getting the type of 'self' (line 97)
        self_35221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 97)
        write_file_35222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), self_35221, 'write_file')
        # Calling write_file(args, kwargs) (line 97)
        write_file_call_result_35227 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), write_file_35222, *[rc_35224, PYPIRC_OLD_35225], **kwargs_35226)
        
        
        # Assigning a Call to a Name (line 98):
        
        # Call to _read_pypirc(...): (line 98)
        # Processing the call keyword arguments (line 98)
        kwargs_35230 = {}
        # Getting the type of 'cmd' (line 98)
        cmd_35228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 'cmd', False)
        # Obtaining the member '_read_pypirc' of a type (line 98)
        _read_pypirc_35229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 17), cmd_35228, '_read_pypirc')
        # Calling _read_pypirc(args, kwargs) (line 98)
        _read_pypirc_call_result_35231 = invoke(stypy.reporting.localization.Localization(__file__, 98, 17), _read_pypirc_35229, *[], **kwargs_35230)
        
        # Assigning a type to the variable 'config' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'config', _read_pypirc_call_result_35231)
        
        # Assigning a Call to a Name (line 99):
        
        # Call to items(...): (line 99)
        # Processing the call keyword arguments (line 99)
        kwargs_35234 = {}
        # Getting the type of 'config' (line 99)
        config_35232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 'config', False)
        # Obtaining the member 'items' of a type (line 99)
        items_35233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 17), config_35232, 'items')
        # Calling items(args, kwargs) (line 99)
        items_call_result_35235 = invoke(stypy.reporting.localization.Localization(__file__, 99, 17), items_35233, *[], **kwargs_35234)
        
        # Assigning a type to the variable 'config' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'config', items_call_result_35235)
        
        # Call to sort(...): (line 100)
        # Processing the call keyword arguments (line 100)
        kwargs_35238 = {}
        # Getting the type of 'config' (line 100)
        config_35236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'config', False)
        # Obtaining the member 'sort' of a type (line 100)
        sort_35237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), config_35236, 'sort')
        # Calling sort(args, kwargs) (line 100)
        sort_call_result_35239 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), sort_35237, *[], **kwargs_35238)
        
        
        # Assigning a List to a Name (line 101):
        
        # Obtaining an instance of the builtin type 'list' (line 101)
        list_35240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 101)
        # Adding element type (line 101)
        
        # Obtaining an instance of the builtin type 'tuple' (line 101)
        tuple_35241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 101)
        # Adding element type (line 101)
        str_35242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 19), 'str', 'password')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 19), tuple_35241, str_35242)
        # Adding element type (line 101)
        str_35243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 31), 'str', 'secret')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 19), tuple_35241, str_35243)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 17), list_35240, tuple_35241)
        # Adding element type (line 101)
        
        # Obtaining an instance of the builtin type 'tuple' (line 101)
        tuple_35244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 101)
        # Adding element type (line 101)
        str_35245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 43), 'str', 'realm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 43), tuple_35244, str_35245)
        # Adding element type (line 101)
        str_35246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 52), 'str', 'pypi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 43), tuple_35244, str_35246)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 17), list_35240, tuple_35244)
        # Adding element type (line 101)
        
        # Obtaining an instance of the builtin type 'tuple' (line 102)
        tuple_35247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 102)
        # Adding element type (line 102)
        str_35248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 19), 'str', 'repository')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 19), tuple_35247, str_35248)
        # Adding element type (line 102)
        str_35249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 33), 'str', 'https://upload.pypi.org/legacy/')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 19), tuple_35247, str_35249)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 17), list_35240, tuple_35247)
        # Adding element type (line 101)
        
        # Obtaining an instance of the builtin type 'tuple' (line 103)
        tuple_35250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 103)
        # Adding element type (line 103)
        str_35251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 19), 'str', 'server')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 19), tuple_35250, str_35251)
        # Adding element type (line 103)
        str_35252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 29), 'str', 'server-login')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 19), tuple_35250, str_35252)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 17), list_35240, tuple_35250)
        # Adding element type (line 101)
        
        # Obtaining an instance of the builtin type 'tuple' (line 103)
        tuple_35253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 103)
        # Adding element type (line 103)
        str_35254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 47), 'str', 'username')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 47), tuple_35253, str_35254)
        # Adding element type (line 103)
        str_35255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 59), 'str', 'tarek')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 47), tuple_35253, str_35255)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 17), list_35240, tuple_35253)
        
        # Assigning a type to the variable 'waited' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'waited', list_35240)
        
        # Call to assertEqual(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'config' (line 104)
        config_35258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 25), 'config', False)
        # Getting the type of 'waited' (line 104)
        waited_35259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 33), 'waited', False)
        # Processing the call keyword arguments (line 104)
        kwargs_35260 = {}
        # Getting the type of 'self' (line 104)
        self_35256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 104)
        assertEqual_35257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), self_35256, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 104)
        assertEqual_call_result_35261 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), assertEqual_35257, *[config_35258, waited_35259], **kwargs_35260)
        
        
        # ################# End of 'test_server_registration(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_server_registration' in the type store
        # Getting the type of 'stypy_return_type' (line 79)
        stypy_return_type_35262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35262)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_server_registration'
        return stypy_return_type_35262


    @norecursion
    def test_server_empty_registration(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_server_empty_registration'
        module_type_store = module_type_store.open_function_context('test_server_empty_registration', 106, 4, False)
        # Assigning a type to the variable 'self' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PyPIRCCommandTestCase.test_server_empty_registration.__dict__.__setitem__('stypy_localization', localization)
        PyPIRCCommandTestCase.test_server_empty_registration.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PyPIRCCommandTestCase.test_server_empty_registration.__dict__.__setitem__('stypy_type_store', module_type_store)
        PyPIRCCommandTestCase.test_server_empty_registration.__dict__.__setitem__('stypy_function_name', 'PyPIRCCommandTestCase.test_server_empty_registration')
        PyPIRCCommandTestCase.test_server_empty_registration.__dict__.__setitem__('stypy_param_names_list', [])
        PyPIRCCommandTestCase.test_server_empty_registration.__dict__.__setitem__('stypy_varargs_param_name', None)
        PyPIRCCommandTestCase.test_server_empty_registration.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PyPIRCCommandTestCase.test_server_empty_registration.__dict__.__setitem__('stypy_call_defaults', defaults)
        PyPIRCCommandTestCase.test_server_empty_registration.__dict__.__setitem__('stypy_call_varargs', varargs)
        PyPIRCCommandTestCase.test_server_empty_registration.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PyPIRCCommandTestCase.test_server_empty_registration.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PyPIRCCommandTestCase.test_server_empty_registration', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_server_empty_registration', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_server_empty_registration(...)' code ##################

        
        # Assigning a Call to a Name (line 107):
        
        # Call to _cmd(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'self' (line 107)
        self_35265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 24), 'self', False)
        # Obtaining the member 'dist' of a type (line 107)
        dist_35266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 24), self_35265, 'dist')
        # Processing the call keyword arguments (line 107)
        kwargs_35267 = {}
        # Getting the type of 'self' (line 107)
        self_35263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 14), 'self', False)
        # Obtaining the member '_cmd' of a type (line 107)
        _cmd_35264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 14), self_35263, '_cmd')
        # Calling _cmd(args, kwargs) (line 107)
        _cmd_call_result_35268 = invoke(stypy.reporting.localization.Localization(__file__, 107, 14), _cmd_35264, *[dist_35266], **kwargs_35267)
        
        # Assigning a type to the variable 'cmd' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'cmd', _cmd_call_result_35268)
        
        # Assigning a Call to a Name (line 108):
        
        # Call to _get_rc_file(...): (line 108)
        # Processing the call keyword arguments (line 108)
        kwargs_35271 = {}
        # Getting the type of 'cmd' (line 108)
        cmd_35269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'cmd', False)
        # Obtaining the member '_get_rc_file' of a type (line 108)
        _get_rc_file_35270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 13), cmd_35269, '_get_rc_file')
        # Calling _get_rc_file(args, kwargs) (line 108)
        _get_rc_file_call_result_35272 = invoke(stypy.reporting.localization.Localization(__file__, 108, 13), _get_rc_file_35270, *[], **kwargs_35271)
        
        # Assigning a type to the variable 'rc' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'rc', _get_rc_file_call_result_35272)
        
        # Call to assertFalse(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Call to exists(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'rc' (line 109)
        rc_35278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 40), 'rc', False)
        # Processing the call keyword arguments (line 109)
        kwargs_35279 = {}
        # Getting the type of 'os' (line 109)
        os_35275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 109)
        path_35276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 25), os_35275, 'path')
        # Obtaining the member 'exists' of a type (line 109)
        exists_35277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 25), path_35276, 'exists')
        # Calling exists(args, kwargs) (line 109)
        exists_call_result_35280 = invoke(stypy.reporting.localization.Localization(__file__, 109, 25), exists_35277, *[rc_35278], **kwargs_35279)
        
        # Processing the call keyword arguments (line 109)
        kwargs_35281 = {}
        # Getting the type of 'self' (line 109)
        self_35273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 109)
        assertFalse_35274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), self_35273, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 109)
        assertFalse_call_result_35282 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), assertFalse_35274, *[exists_call_result_35280], **kwargs_35281)
        
        
        # Call to _store_pypirc(...): (line 110)
        # Processing the call arguments (line 110)
        str_35285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 26), 'str', 'tarek')
        str_35286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 35), 'str', 'xxx')
        # Processing the call keyword arguments (line 110)
        kwargs_35287 = {}
        # Getting the type of 'cmd' (line 110)
        cmd_35283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'cmd', False)
        # Obtaining the member '_store_pypirc' of a type (line 110)
        _store_pypirc_35284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), cmd_35283, '_store_pypirc')
        # Calling _store_pypirc(args, kwargs) (line 110)
        _store_pypirc_call_result_35288 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), _store_pypirc_35284, *[str_35285, str_35286], **kwargs_35287)
        
        
        # Call to assertTrue(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Call to exists(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'rc' (line 111)
        rc_35294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 39), 'rc', False)
        # Processing the call keyword arguments (line 111)
        kwargs_35295 = {}
        # Getting the type of 'os' (line 111)
        os_35291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 111)
        path_35292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 24), os_35291, 'path')
        # Obtaining the member 'exists' of a type (line 111)
        exists_35293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 24), path_35292, 'exists')
        # Calling exists(args, kwargs) (line 111)
        exists_call_result_35296 = invoke(stypy.reporting.localization.Localization(__file__, 111, 24), exists_35293, *[rc_35294], **kwargs_35295)
        
        # Processing the call keyword arguments (line 111)
        kwargs_35297 = {}
        # Getting the type of 'self' (line 111)
        self_35289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 111)
        assertTrue_35290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_35289, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 111)
        assertTrue_call_result_35298 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), assertTrue_35290, *[exists_call_result_35296], **kwargs_35297)
        
        
        # Assigning a Call to a Name (line 112):
        
        # Call to open(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'rc' (line 112)
        rc_35300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 17), 'rc', False)
        # Processing the call keyword arguments (line 112)
        kwargs_35301 = {}
        # Getting the type of 'open' (line 112)
        open_35299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'open', False)
        # Calling open(args, kwargs) (line 112)
        open_call_result_35302 = invoke(stypy.reporting.localization.Localization(__file__, 112, 12), open_35299, *[rc_35300], **kwargs_35301)
        
        # Assigning a type to the variable 'f' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'f', open_call_result_35302)
        
        # Try-finally block (line 113)
        
        # Assigning a Call to a Name (line 114):
        
        # Call to read(...): (line 114)
        # Processing the call keyword arguments (line 114)
        kwargs_35305 = {}
        # Getting the type of 'f' (line 114)
        f_35303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'f', False)
        # Obtaining the member 'read' of a type (line 114)
        read_35304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 22), f_35303, 'read')
        # Calling read(args, kwargs) (line 114)
        read_call_result_35306 = invoke(stypy.reporting.localization.Localization(__file__, 114, 22), read_35304, *[], **kwargs_35305)
        
        # Assigning a type to the variable 'content' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'content', read_call_result_35306)
        
        # Call to assertEqual(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'content' (line 115)
        content_35309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 29), 'content', False)
        # Getting the type of 'WANTED' (line 115)
        WANTED_35310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 38), 'WANTED', False)
        # Processing the call keyword arguments (line 115)
        kwargs_35311 = {}
        # Getting the type of 'self' (line 115)
        self_35307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 115)
        assertEqual_35308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), self_35307, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 115)
        assertEqual_call_result_35312 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), assertEqual_35308, *[content_35309, WANTED_35310], **kwargs_35311)
        
        
        # finally branch of the try-finally block (line 113)
        
        # Call to close(...): (line 117)
        # Processing the call keyword arguments (line 117)
        kwargs_35315 = {}
        # Getting the type of 'f' (line 117)
        f_35313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 117)
        close_35314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), f_35313, 'close')
        # Calling close(args, kwargs) (line 117)
        close_call_result_35316 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), close_35314, *[], **kwargs_35315)
        
        
        
        # ################# End of 'test_server_empty_registration(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_server_empty_registration' in the type store
        # Getting the type of 'stypy_return_type' (line 106)
        stypy_return_type_35317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35317)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_server_empty_registration'
        return stypy_return_type_35317


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 51, 0, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PyPIRCCommandTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'PyPIRCCommandTestCase' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'PyPIRCCommandTestCase', PyPIRCCommandTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 119, 0, False)
    
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

    
    # Call to makeSuite(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'PyPIRCCommandTestCase' (line 120)
    PyPIRCCommandTestCase_35320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 30), 'PyPIRCCommandTestCase', False)
    # Processing the call keyword arguments (line 120)
    kwargs_35321 = {}
    # Getting the type of 'unittest' (line 120)
    unittest_35318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 120)
    makeSuite_35319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 11), unittest_35318, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 120)
    makeSuite_call_result_35322 = invoke(stypy.reporting.localization.Localization(__file__, 120, 11), makeSuite_35319, *[PyPIRCCommandTestCase_35320], **kwargs_35321)
    
    # Assigning a type to the variable 'stypy_return_type' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type', makeSuite_call_result_35322)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 119)
    stypy_return_type_35323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35323)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_35323

# Assigning a type to the variable 'test_suite' (line 119)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 123)
    # Processing the call arguments (line 123)
    
    # Call to test_suite(...): (line 123)
    # Processing the call keyword arguments (line 123)
    kwargs_35326 = {}
    # Getting the type of 'test_suite' (line 123)
    test_suite_35325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 123)
    test_suite_call_result_35327 = invoke(stypy.reporting.localization.Localization(__file__, 123, 17), test_suite_35325, *[], **kwargs_35326)
    
    # Processing the call keyword arguments (line 123)
    kwargs_35328 = {}
    # Getting the type of 'run_unittest' (line 123)
    run_unittest_35324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 123)
    run_unittest_call_result_35329 = invoke(stypy.reporting.localization.Localization(__file__, 123, 4), run_unittest_35324, *[test_suite_call_result_35327], **kwargs_35328)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
