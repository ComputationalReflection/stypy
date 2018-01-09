
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.core.'''
2: 
3: import StringIO
4: import distutils.core
5: import os
6: import shutil
7: import sys
8: import test.test_support
9: from test.test_support import captured_stdout, run_unittest
10: import unittest
11: from distutils.tests import support
12: from distutils import log
13: 
14: # setup script that uses __file__
15: setup_using___file__ = '''\
16: 
17: __file__
18: 
19: from distutils.core import setup
20: setup()
21: '''
22: 
23: setup_prints_cwd = '''\
24: 
25: import os
26: print os.getcwd()
27: 
28: from distutils.core import setup
29: setup()
30: '''
31: 
32: 
33: class CoreTestCase(support.EnvironGuard, unittest.TestCase):
34: 
35:     def setUp(self):
36:         super(CoreTestCase, self).setUp()
37:         self.old_stdout = sys.stdout
38:         self.cleanup_testfn()
39:         self.old_argv = sys.argv, sys.argv[:]
40:         self.addCleanup(log.set_threshold, log._global_log.threshold)
41: 
42:     def tearDown(self):
43:         sys.stdout = self.old_stdout
44:         self.cleanup_testfn()
45:         sys.argv = self.old_argv[0]
46:         sys.argv[:] = self.old_argv[1]
47:         super(CoreTestCase, self).tearDown()
48: 
49:     def cleanup_testfn(self):
50:         path = test.test_support.TESTFN
51:         if os.path.isfile(path):
52:             os.remove(path)
53:         elif os.path.isdir(path):
54:             shutil.rmtree(path)
55: 
56:     def write_setup(self, text, path=test.test_support.TESTFN):
57:         f = open(path, "w")
58:         try:
59:             f.write(text)
60:         finally:
61:             f.close()
62:         return path
63: 
64:     def test_run_setup_provides_file(self):
65:         # Make sure the script can use __file__; if that's missing, the test
66:         # setup.py script will raise NameError.
67:         distutils.core.run_setup(
68:             self.write_setup(setup_using___file__))
69: 
70:     def test_run_setup_uses_current_dir(self):
71:         # This tests that the setup script is run with the current directory
72:         # as its own current directory; this was temporarily broken by a
73:         # previous patch when TESTFN did not use the current directory.
74:         sys.stdout = StringIO.StringIO()
75:         cwd = os.getcwd()
76: 
77:         # Create a directory and write the setup.py file there:
78:         os.mkdir(test.test_support.TESTFN)
79:         setup_py = os.path.join(test.test_support.TESTFN, "setup.py")
80:         distutils.core.run_setup(
81:             self.write_setup(setup_prints_cwd, path=setup_py))
82: 
83:         output = sys.stdout.getvalue()
84:         if output.endswith("\n"):
85:             output = output[:-1]
86:         self.assertEqual(cwd, output)
87: 
88:     def test_debug_mode(self):
89:         # this covers the code called when DEBUG is set
90:         sys.argv = ['setup.py', '--name']
91:         with captured_stdout() as stdout:
92:             distutils.core.setup(name='bar')
93:         stdout.seek(0)
94:         self.assertEqual(stdout.read(), 'bar\n')
95: 
96:         distutils.core.DEBUG = True
97:         try:
98:             with captured_stdout() as stdout:
99:                 distutils.core.setup(name='bar')
100:         finally:
101:             distutils.core.DEBUG = False
102:         stdout.seek(0)
103:         wanted = "options (after parsing config files):\n"
104:         self.assertEqual(stdout.readlines()[0], wanted)
105: 
106: def test_suite():
107:     return unittest.makeSuite(CoreTestCase)
108: 
109: if __name__ == "__main__":
110:     run_unittest(test_suite())
111: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_35643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.core.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import StringIO' statement (line 3)
import StringIO

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'StringIO', StringIO, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import distutils.core' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_35644 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.core')

if (type(import_35644) is not StypyTypeError):

    if (import_35644 != 'pyd_module'):
        __import__(import_35644)
        sys_modules_35645 = sys.modules[import_35644]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.core', sys_modules_35645.module_type_store, module_type_store)
    else:
        import distutils.core

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.core', distutils.core, module_type_store)

else:
    # Assigning a type to the variable 'distutils.core' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.core', import_35644)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import os' statement (line 5)
import os

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import shutil' statement (line 6)
import shutil

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'shutil', shutil, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import sys' statement (line 7)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import test.test_support' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_35646 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'test.test_support')

if (type(import_35646) is not StypyTypeError):

    if (import_35646 != 'pyd_module'):
        __import__(import_35646)
        sys_modules_35647 = sys.modules[import_35646]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'test.test_support', sys_modules_35647.module_type_store, module_type_store)
    else:
        import test.test_support

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'test.test_support', test.test_support, module_type_store)

else:
    # Assigning a type to the variable 'test.test_support' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'test.test_support', import_35646)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from test.test_support import captured_stdout, run_unittest' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_35648 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support')

if (type(import_35648) is not StypyTypeError):

    if (import_35648 != 'pyd_module'):
        __import__(import_35648)
        sys_modules_35649 = sys.modules[import_35648]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', sys_modules_35649.module_type_store, module_type_store, ['captured_stdout', 'run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_35649, sys_modules_35649.module_type_store, module_type_store)
    else:
        from test.test_support import captured_stdout, run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', None, module_type_store, ['captured_stdout', 'run_unittest'], [captured_stdout, run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', import_35648)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import unittest' statement (line 10)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.tests import support' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_35650 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.tests')

if (type(import_35650) is not StypyTypeError):

    if (import_35650 != 'pyd_module'):
        __import__(import_35650)
        sys_modules_35651 = sys.modules[import_35650]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.tests', sys_modules_35651.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_35651, sys_modules_35651.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.tests', import_35650)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils import log' statement (line 12)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils', None, module_type_store, ['log'], [log])


# Assigning a Str to a Name (line 15):
str_35652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, (-1)), 'str', '\n__file__\n\nfrom distutils.core import setup\nsetup()\n')
# Assigning a type to the variable 'setup_using___file__' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'setup_using___file__', str_35652)

# Assigning a Str to a Name (line 23):
str_35653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, (-1)), 'str', '\nimport os\nprint os.getcwd()\n\nfrom distutils.core import setup\nsetup()\n')
# Assigning a type to the variable 'setup_prints_cwd' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'setup_prints_cwd', str_35653)
# Declaration of the 'CoreTestCase' class
# Getting the type of 'support' (line 33)
support_35654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'support')
# Obtaining the member 'EnvironGuard' of a type (line 33)
EnvironGuard_35655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 19), support_35654, 'EnvironGuard')
# Getting the type of 'unittest' (line 33)
unittest_35656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 41), 'unittest')
# Obtaining the member 'TestCase' of a type (line 33)
TestCase_35657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 41), unittest_35656, 'TestCase')

class CoreTestCase(EnvironGuard_35655, TestCase_35657, ):

    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CoreTestCase.setUp.__dict__.__setitem__('stypy_localization', localization)
        CoreTestCase.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CoreTestCase.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        CoreTestCase.setUp.__dict__.__setitem__('stypy_function_name', 'CoreTestCase.setUp')
        CoreTestCase.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        CoreTestCase.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        CoreTestCase.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CoreTestCase.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        CoreTestCase.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        CoreTestCase.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CoreTestCase.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CoreTestCase.setUp', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to setUp(...): (line 36)
        # Processing the call keyword arguments (line 36)
        kwargs_35664 = {}
        
        # Call to super(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'CoreTestCase' (line 36)
        CoreTestCase_35659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 14), 'CoreTestCase', False)
        # Getting the type of 'self' (line 36)
        self_35660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 28), 'self', False)
        # Processing the call keyword arguments (line 36)
        kwargs_35661 = {}
        # Getting the type of 'super' (line 36)
        super_35658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'super', False)
        # Calling super(args, kwargs) (line 36)
        super_call_result_35662 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), super_35658, *[CoreTestCase_35659, self_35660], **kwargs_35661)
        
        # Obtaining the member 'setUp' of a type (line 36)
        setUp_35663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), super_call_result_35662, 'setUp')
        # Calling setUp(args, kwargs) (line 36)
        setUp_call_result_35665 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), setUp_35663, *[], **kwargs_35664)
        
        
        # Assigning a Attribute to a Attribute (line 37):
        # Getting the type of 'sys' (line 37)
        sys_35666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 26), 'sys')
        # Obtaining the member 'stdout' of a type (line 37)
        stdout_35667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 26), sys_35666, 'stdout')
        # Getting the type of 'self' (line 37)
        self_35668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self')
        # Setting the type of the member 'old_stdout' of a type (line 37)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_35668, 'old_stdout', stdout_35667)
        
        # Call to cleanup_testfn(...): (line 38)
        # Processing the call keyword arguments (line 38)
        kwargs_35671 = {}
        # Getting the type of 'self' (line 38)
        self_35669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self', False)
        # Obtaining the member 'cleanup_testfn' of a type (line 38)
        cleanup_testfn_35670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_35669, 'cleanup_testfn')
        # Calling cleanup_testfn(args, kwargs) (line 38)
        cleanup_testfn_call_result_35672 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), cleanup_testfn_35670, *[], **kwargs_35671)
        
        
        # Assigning a Tuple to a Attribute (line 39):
        
        # Obtaining an instance of the builtin type 'tuple' (line 39)
        tuple_35673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 39)
        # Adding element type (line 39)
        # Getting the type of 'sys' (line 39)
        sys_35674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 24), 'sys')
        # Obtaining the member 'argv' of a type (line 39)
        argv_35675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 24), sys_35674, 'argv')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 24), tuple_35673, argv_35675)
        # Adding element type (line 39)
        
        # Obtaining the type of the subscript
        slice_35676 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 39, 34), None, None, None)
        # Getting the type of 'sys' (line 39)
        sys_35677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 34), 'sys')
        # Obtaining the member 'argv' of a type (line 39)
        argv_35678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 34), sys_35677, 'argv')
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___35679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 34), argv_35678, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_35680 = invoke(stypy.reporting.localization.Localization(__file__, 39, 34), getitem___35679, slice_35676)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 24), tuple_35673, subscript_call_result_35680)
        
        # Getting the type of 'self' (line 39)
        self_35681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'self')
        # Setting the type of the member 'old_argv' of a type (line 39)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), self_35681, 'old_argv', tuple_35673)
        
        # Call to addCleanup(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'log' (line 40)
        log_35684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'log', False)
        # Obtaining the member 'set_threshold' of a type (line 40)
        set_threshold_35685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 24), log_35684, 'set_threshold')
        # Getting the type of 'log' (line 40)
        log_35686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 43), 'log', False)
        # Obtaining the member '_global_log' of a type (line 40)
        _global_log_35687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 43), log_35686, '_global_log')
        # Obtaining the member 'threshold' of a type (line 40)
        threshold_35688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 43), _global_log_35687, 'threshold')
        # Processing the call keyword arguments (line 40)
        kwargs_35689 = {}
        # Getting the type of 'self' (line 40)
        self_35682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 40)
        addCleanup_35683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_35682, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 40)
        addCleanup_call_result_35690 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), addCleanup_35683, *[set_threshold_35685, threshold_35688], **kwargs_35689)
        
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_35691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35691)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_35691


    @norecursion
    def tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDown'
        module_type_store = module_type_store.open_function_context('tearDown', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CoreTestCase.tearDown.__dict__.__setitem__('stypy_localization', localization)
        CoreTestCase.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CoreTestCase.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        CoreTestCase.tearDown.__dict__.__setitem__('stypy_function_name', 'CoreTestCase.tearDown')
        CoreTestCase.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        CoreTestCase.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        CoreTestCase.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CoreTestCase.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        CoreTestCase.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        CoreTestCase.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CoreTestCase.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CoreTestCase.tearDown', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Attribute (line 43):
        # Getting the type of 'self' (line 43)
        self_35692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 21), 'self')
        # Obtaining the member 'old_stdout' of a type (line 43)
        old_stdout_35693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 21), self_35692, 'old_stdout')
        # Getting the type of 'sys' (line 43)
        sys_35694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'sys')
        # Setting the type of the member 'stdout' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), sys_35694, 'stdout', old_stdout_35693)
        
        # Call to cleanup_testfn(...): (line 44)
        # Processing the call keyword arguments (line 44)
        kwargs_35697 = {}
        # Getting the type of 'self' (line 44)
        self_35695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'self', False)
        # Obtaining the member 'cleanup_testfn' of a type (line 44)
        cleanup_testfn_35696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), self_35695, 'cleanup_testfn')
        # Calling cleanup_testfn(args, kwargs) (line 44)
        cleanup_testfn_call_result_35698 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), cleanup_testfn_35696, *[], **kwargs_35697)
        
        
        # Assigning a Subscript to a Attribute (line 45):
        
        # Obtaining the type of the subscript
        int_35699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 33), 'int')
        # Getting the type of 'self' (line 45)
        self_35700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'self')
        # Obtaining the member 'old_argv' of a type (line 45)
        old_argv_35701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 19), self_35700, 'old_argv')
        # Obtaining the member '__getitem__' of a type (line 45)
        getitem___35702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 19), old_argv_35701, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 45)
        subscript_call_result_35703 = invoke(stypy.reporting.localization.Localization(__file__, 45, 19), getitem___35702, int_35699)
        
        # Getting the type of 'sys' (line 45)
        sys_35704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'sys')
        # Setting the type of the member 'argv' of a type (line 45)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), sys_35704, 'argv', subscript_call_result_35703)
        
        # Assigning a Subscript to a Subscript (line 46):
        
        # Obtaining the type of the subscript
        int_35705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 36), 'int')
        # Getting the type of 'self' (line 46)
        self_35706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'self')
        # Obtaining the member 'old_argv' of a type (line 46)
        old_argv_35707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 22), self_35706, 'old_argv')
        # Obtaining the member '__getitem__' of a type (line 46)
        getitem___35708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 22), old_argv_35707, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 46)
        subscript_call_result_35709 = invoke(stypy.reporting.localization.Localization(__file__, 46, 22), getitem___35708, int_35705)
        
        # Getting the type of 'sys' (line 46)
        sys_35710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'sys')
        # Obtaining the member 'argv' of a type (line 46)
        argv_35711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), sys_35710, 'argv')
        slice_35712 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 46, 8), None, None, None)
        # Storing an element on a container (line 46)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 8), argv_35711, (slice_35712, subscript_call_result_35709))
        
        # Call to tearDown(...): (line 47)
        # Processing the call keyword arguments (line 47)
        kwargs_35719 = {}
        
        # Call to super(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'CoreTestCase' (line 47)
        CoreTestCase_35714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 14), 'CoreTestCase', False)
        # Getting the type of 'self' (line 47)
        self_35715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 28), 'self', False)
        # Processing the call keyword arguments (line 47)
        kwargs_35716 = {}
        # Getting the type of 'super' (line 47)
        super_35713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'super', False)
        # Calling super(args, kwargs) (line 47)
        super_call_result_35717 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), super_35713, *[CoreTestCase_35714, self_35715], **kwargs_35716)
        
        # Obtaining the member 'tearDown' of a type (line 47)
        tearDown_35718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), super_call_result_35717, 'tearDown')
        # Calling tearDown(args, kwargs) (line 47)
        tearDown_call_result_35720 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), tearDown_35718, *[], **kwargs_35719)
        
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_35721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35721)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_35721


    @norecursion
    def cleanup_testfn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cleanup_testfn'
        module_type_store = module_type_store.open_function_context('cleanup_testfn', 49, 4, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CoreTestCase.cleanup_testfn.__dict__.__setitem__('stypy_localization', localization)
        CoreTestCase.cleanup_testfn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CoreTestCase.cleanup_testfn.__dict__.__setitem__('stypy_type_store', module_type_store)
        CoreTestCase.cleanup_testfn.__dict__.__setitem__('stypy_function_name', 'CoreTestCase.cleanup_testfn')
        CoreTestCase.cleanup_testfn.__dict__.__setitem__('stypy_param_names_list', [])
        CoreTestCase.cleanup_testfn.__dict__.__setitem__('stypy_varargs_param_name', None)
        CoreTestCase.cleanup_testfn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CoreTestCase.cleanup_testfn.__dict__.__setitem__('stypy_call_defaults', defaults)
        CoreTestCase.cleanup_testfn.__dict__.__setitem__('stypy_call_varargs', varargs)
        CoreTestCase.cleanup_testfn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CoreTestCase.cleanup_testfn.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CoreTestCase.cleanup_testfn', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cleanup_testfn', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cleanup_testfn(...)' code ##################

        
        # Assigning a Attribute to a Name (line 50):
        # Getting the type of 'test' (line 50)
        test_35722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'test')
        # Obtaining the member 'test_support' of a type (line 50)
        test_support_35723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 15), test_35722, 'test_support')
        # Obtaining the member 'TESTFN' of a type (line 50)
        TESTFN_35724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 15), test_support_35723, 'TESTFN')
        # Assigning a type to the variable 'path' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'path', TESTFN_35724)
        
        
        # Call to isfile(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'path' (line 51)
        path_35728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 26), 'path', False)
        # Processing the call keyword arguments (line 51)
        kwargs_35729 = {}
        # Getting the type of 'os' (line 51)
        os_35725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 11), 'os', False)
        # Obtaining the member 'path' of a type (line 51)
        path_35726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 11), os_35725, 'path')
        # Obtaining the member 'isfile' of a type (line 51)
        isfile_35727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 11), path_35726, 'isfile')
        # Calling isfile(args, kwargs) (line 51)
        isfile_call_result_35730 = invoke(stypy.reporting.localization.Localization(__file__, 51, 11), isfile_35727, *[path_35728], **kwargs_35729)
        
        # Testing the type of an if condition (line 51)
        if_condition_35731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 51, 8), isfile_call_result_35730)
        # Assigning a type to the variable 'if_condition_35731' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'if_condition_35731', if_condition_35731)
        # SSA begins for if statement (line 51)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to remove(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'path' (line 52)
        path_35734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 22), 'path', False)
        # Processing the call keyword arguments (line 52)
        kwargs_35735 = {}
        # Getting the type of 'os' (line 52)
        os_35732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'os', False)
        # Obtaining the member 'remove' of a type (line 52)
        remove_35733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), os_35732, 'remove')
        # Calling remove(args, kwargs) (line 52)
        remove_call_result_35736 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), remove_35733, *[path_35734], **kwargs_35735)
        
        # SSA branch for the else part of an if statement (line 51)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isdir(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'path' (line 53)
        path_35740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 27), 'path', False)
        # Processing the call keyword arguments (line 53)
        kwargs_35741 = {}
        # Getting the type of 'os' (line 53)
        os_35737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 13), 'os', False)
        # Obtaining the member 'path' of a type (line 53)
        path_35738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 13), os_35737, 'path')
        # Obtaining the member 'isdir' of a type (line 53)
        isdir_35739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 13), path_35738, 'isdir')
        # Calling isdir(args, kwargs) (line 53)
        isdir_call_result_35742 = invoke(stypy.reporting.localization.Localization(__file__, 53, 13), isdir_35739, *[path_35740], **kwargs_35741)
        
        # Testing the type of an if condition (line 53)
        if_condition_35743 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 13), isdir_call_result_35742)
        # Assigning a type to the variable 'if_condition_35743' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 13), 'if_condition_35743', if_condition_35743)
        # SSA begins for if statement (line 53)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to rmtree(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'path' (line 54)
        path_35746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 26), 'path', False)
        # Processing the call keyword arguments (line 54)
        kwargs_35747 = {}
        # Getting the type of 'shutil' (line 54)
        shutil_35744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'shutil', False)
        # Obtaining the member 'rmtree' of a type (line 54)
        rmtree_35745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), shutil_35744, 'rmtree')
        # Calling rmtree(args, kwargs) (line 54)
        rmtree_call_result_35748 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), rmtree_35745, *[path_35746], **kwargs_35747)
        
        # SSA join for if statement (line 53)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 51)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'cleanup_testfn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cleanup_testfn' in the type store
        # Getting the type of 'stypy_return_type' (line 49)
        stypy_return_type_35749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35749)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cleanup_testfn'
        return stypy_return_type_35749


    @norecursion
    def write_setup(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'test' (line 56)
        test_35750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 37), 'test')
        # Obtaining the member 'test_support' of a type (line 56)
        test_support_35751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 37), test_35750, 'test_support')
        # Obtaining the member 'TESTFN' of a type (line 56)
        TESTFN_35752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 37), test_support_35751, 'TESTFN')
        defaults = [TESTFN_35752]
        # Create a new context for function 'write_setup'
        module_type_store = module_type_store.open_function_context('write_setup', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CoreTestCase.write_setup.__dict__.__setitem__('stypy_localization', localization)
        CoreTestCase.write_setup.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CoreTestCase.write_setup.__dict__.__setitem__('stypy_type_store', module_type_store)
        CoreTestCase.write_setup.__dict__.__setitem__('stypy_function_name', 'CoreTestCase.write_setup')
        CoreTestCase.write_setup.__dict__.__setitem__('stypy_param_names_list', ['text', 'path'])
        CoreTestCase.write_setup.__dict__.__setitem__('stypy_varargs_param_name', None)
        CoreTestCase.write_setup.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CoreTestCase.write_setup.__dict__.__setitem__('stypy_call_defaults', defaults)
        CoreTestCase.write_setup.__dict__.__setitem__('stypy_call_varargs', varargs)
        CoreTestCase.write_setup.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CoreTestCase.write_setup.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CoreTestCase.write_setup', ['text', 'path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_setup', localization, ['text', 'path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_setup(...)' code ##################

        
        # Assigning a Call to a Name (line 57):
        
        # Call to open(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'path' (line 57)
        path_35754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 17), 'path', False)
        str_35755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 23), 'str', 'w')
        # Processing the call keyword arguments (line 57)
        kwargs_35756 = {}
        # Getting the type of 'open' (line 57)
        open_35753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'open', False)
        # Calling open(args, kwargs) (line 57)
        open_call_result_35757 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), open_35753, *[path_35754, str_35755], **kwargs_35756)
        
        # Assigning a type to the variable 'f' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'f', open_call_result_35757)
        
        # Try-finally block (line 58)
        
        # Call to write(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'text' (line 59)
        text_35760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'text', False)
        # Processing the call keyword arguments (line 59)
        kwargs_35761 = {}
        # Getting the type of 'f' (line 59)
        f_35758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 59)
        write_35759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 12), f_35758, 'write')
        # Calling write(args, kwargs) (line 59)
        write_call_result_35762 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), write_35759, *[text_35760], **kwargs_35761)
        
        
        # finally branch of the try-finally block (line 58)
        
        # Call to close(...): (line 61)
        # Processing the call keyword arguments (line 61)
        kwargs_35765 = {}
        # Getting the type of 'f' (line 61)
        f_35763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 61)
        close_35764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), f_35763, 'close')
        # Calling close(args, kwargs) (line 61)
        close_call_result_35766 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), close_35764, *[], **kwargs_35765)
        
        
        # Getting the type of 'path' (line 62)
        path_35767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'path')
        # Assigning a type to the variable 'stypy_return_type' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'stypy_return_type', path_35767)
        
        # ################# End of 'write_setup(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_setup' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_35768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35768)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_setup'
        return stypy_return_type_35768


    @norecursion
    def test_run_setup_provides_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_run_setup_provides_file'
        module_type_store = module_type_store.open_function_context('test_run_setup_provides_file', 64, 4, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CoreTestCase.test_run_setup_provides_file.__dict__.__setitem__('stypy_localization', localization)
        CoreTestCase.test_run_setup_provides_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CoreTestCase.test_run_setup_provides_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        CoreTestCase.test_run_setup_provides_file.__dict__.__setitem__('stypy_function_name', 'CoreTestCase.test_run_setup_provides_file')
        CoreTestCase.test_run_setup_provides_file.__dict__.__setitem__('stypy_param_names_list', [])
        CoreTestCase.test_run_setup_provides_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        CoreTestCase.test_run_setup_provides_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CoreTestCase.test_run_setup_provides_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        CoreTestCase.test_run_setup_provides_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        CoreTestCase.test_run_setup_provides_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CoreTestCase.test_run_setup_provides_file.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CoreTestCase.test_run_setup_provides_file', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_run_setup_provides_file', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_run_setup_provides_file(...)' code ##################

        
        # Call to run_setup(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Call to write_setup(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'setup_using___file__' (line 68)
        setup_using___file___35774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 29), 'setup_using___file__', False)
        # Processing the call keyword arguments (line 68)
        kwargs_35775 = {}
        # Getting the type of 'self' (line 68)
        self_35772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'self', False)
        # Obtaining the member 'write_setup' of a type (line 68)
        write_setup_35773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), self_35772, 'write_setup')
        # Calling write_setup(args, kwargs) (line 68)
        write_setup_call_result_35776 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), write_setup_35773, *[setup_using___file___35774], **kwargs_35775)
        
        # Processing the call keyword arguments (line 67)
        kwargs_35777 = {}
        # Getting the type of 'distutils' (line 67)
        distutils_35769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'distutils', False)
        # Obtaining the member 'core' of a type (line 67)
        core_35770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), distutils_35769, 'core')
        # Obtaining the member 'run_setup' of a type (line 67)
        run_setup_35771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), core_35770, 'run_setup')
        # Calling run_setup(args, kwargs) (line 67)
        run_setup_call_result_35778 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), run_setup_35771, *[write_setup_call_result_35776], **kwargs_35777)
        
        
        # ################# End of 'test_run_setup_provides_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_run_setup_provides_file' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_35779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35779)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_run_setup_provides_file'
        return stypy_return_type_35779


    @norecursion
    def test_run_setup_uses_current_dir(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_run_setup_uses_current_dir'
        module_type_store = module_type_store.open_function_context('test_run_setup_uses_current_dir', 70, 4, False)
        # Assigning a type to the variable 'self' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CoreTestCase.test_run_setup_uses_current_dir.__dict__.__setitem__('stypy_localization', localization)
        CoreTestCase.test_run_setup_uses_current_dir.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CoreTestCase.test_run_setup_uses_current_dir.__dict__.__setitem__('stypy_type_store', module_type_store)
        CoreTestCase.test_run_setup_uses_current_dir.__dict__.__setitem__('stypy_function_name', 'CoreTestCase.test_run_setup_uses_current_dir')
        CoreTestCase.test_run_setup_uses_current_dir.__dict__.__setitem__('stypy_param_names_list', [])
        CoreTestCase.test_run_setup_uses_current_dir.__dict__.__setitem__('stypy_varargs_param_name', None)
        CoreTestCase.test_run_setup_uses_current_dir.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CoreTestCase.test_run_setup_uses_current_dir.__dict__.__setitem__('stypy_call_defaults', defaults)
        CoreTestCase.test_run_setup_uses_current_dir.__dict__.__setitem__('stypy_call_varargs', varargs)
        CoreTestCase.test_run_setup_uses_current_dir.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CoreTestCase.test_run_setup_uses_current_dir.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CoreTestCase.test_run_setup_uses_current_dir', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_run_setup_uses_current_dir', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_run_setup_uses_current_dir(...)' code ##################

        
        # Assigning a Call to a Attribute (line 74):
        
        # Call to StringIO(...): (line 74)
        # Processing the call keyword arguments (line 74)
        kwargs_35782 = {}
        # Getting the type of 'StringIO' (line 74)
        StringIO_35780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 21), 'StringIO', False)
        # Obtaining the member 'StringIO' of a type (line 74)
        StringIO_35781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 21), StringIO_35780, 'StringIO')
        # Calling StringIO(args, kwargs) (line 74)
        StringIO_call_result_35783 = invoke(stypy.reporting.localization.Localization(__file__, 74, 21), StringIO_35781, *[], **kwargs_35782)
        
        # Getting the type of 'sys' (line 74)
        sys_35784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'sys')
        # Setting the type of the member 'stdout' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), sys_35784, 'stdout', StringIO_call_result_35783)
        
        # Assigning a Call to a Name (line 75):
        
        # Call to getcwd(...): (line 75)
        # Processing the call keyword arguments (line 75)
        kwargs_35787 = {}
        # Getting the type of 'os' (line 75)
        os_35785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 14), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 75)
        getcwd_35786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 14), os_35785, 'getcwd')
        # Calling getcwd(args, kwargs) (line 75)
        getcwd_call_result_35788 = invoke(stypy.reporting.localization.Localization(__file__, 75, 14), getcwd_35786, *[], **kwargs_35787)
        
        # Assigning a type to the variable 'cwd' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'cwd', getcwd_call_result_35788)
        
        # Call to mkdir(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'test' (line 78)
        test_35791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 17), 'test', False)
        # Obtaining the member 'test_support' of a type (line 78)
        test_support_35792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 17), test_35791, 'test_support')
        # Obtaining the member 'TESTFN' of a type (line 78)
        TESTFN_35793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 17), test_support_35792, 'TESTFN')
        # Processing the call keyword arguments (line 78)
        kwargs_35794 = {}
        # Getting the type of 'os' (line 78)
        os_35789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 78)
        mkdir_35790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), os_35789, 'mkdir')
        # Calling mkdir(args, kwargs) (line 78)
        mkdir_call_result_35795 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), mkdir_35790, *[TESTFN_35793], **kwargs_35794)
        
        
        # Assigning a Call to a Name (line 79):
        
        # Call to join(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'test' (line 79)
        test_35799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 32), 'test', False)
        # Obtaining the member 'test_support' of a type (line 79)
        test_support_35800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 32), test_35799, 'test_support')
        # Obtaining the member 'TESTFN' of a type (line 79)
        TESTFN_35801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 32), test_support_35800, 'TESTFN')
        str_35802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 58), 'str', 'setup.py')
        # Processing the call keyword arguments (line 79)
        kwargs_35803 = {}
        # Getting the type of 'os' (line 79)
        os_35796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 79)
        path_35797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 19), os_35796, 'path')
        # Obtaining the member 'join' of a type (line 79)
        join_35798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 19), path_35797, 'join')
        # Calling join(args, kwargs) (line 79)
        join_call_result_35804 = invoke(stypy.reporting.localization.Localization(__file__, 79, 19), join_35798, *[TESTFN_35801, str_35802], **kwargs_35803)
        
        # Assigning a type to the variable 'setup_py' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'setup_py', join_call_result_35804)
        
        # Call to run_setup(...): (line 80)
        # Processing the call arguments (line 80)
        
        # Call to write_setup(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'setup_prints_cwd' (line 81)
        setup_prints_cwd_35810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 29), 'setup_prints_cwd', False)
        # Processing the call keyword arguments (line 81)
        # Getting the type of 'setup_py' (line 81)
        setup_py_35811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 52), 'setup_py', False)
        keyword_35812 = setup_py_35811
        kwargs_35813 = {'path': keyword_35812}
        # Getting the type of 'self' (line 81)
        self_35808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'self', False)
        # Obtaining the member 'write_setup' of a type (line 81)
        write_setup_35809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), self_35808, 'write_setup')
        # Calling write_setup(args, kwargs) (line 81)
        write_setup_call_result_35814 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), write_setup_35809, *[setup_prints_cwd_35810], **kwargs_35813)
        
        # Processing the call keyword arguments (line 80)
        kwargs_35815 = {}
        # Getting the type of 'distutils' (line 80)
        distutils_35805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'distutils', False)
        # Obtaining the member 'core' of a type (line 80)
        core_35806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), distutils_35805, 'core')
        # Obtaining the member 'run_setup' of a type (line 80)
        run_setup_35807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), core_35806, 'run_setup')
        # Calling run_setup(args, kwargs) (line 80)
        run_setup_call_result_35816 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), run_setup_35807, *[write_setup_call_result_35814], **kwargs_35815)
        
        
        # Assigning a Call to a Name (line 83):
        
        # Call to getvalue(...): (line 83)
        # Processing the call keyword arguments (line 83)
        kwargs_35820 = {}
        # Getting the type of 'sys' (line 83)
        sys_35817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 17), 'sys', False)
        # Obtaining the member 'stdout' of a type (line 83)
        stdout_35818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 17), sys_35817, 'stdout')
        # Obtaining the member 'getvalue' of a type (line 83)
        getvalue_35819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 17), stdout_35818, 'getvalue')
        # Calling getvalue(args, kwargs) (line 83)
        getvalue_call_result_35821 = invoke(stypy.reporting.localization.Localization(__file__, 83, 17), getvalue_35819, *[], **kwargs_35820)
        
        # Assigning a type to the variable 'output' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'output', getvalue_call_result_35821)
        
        
        # Call to endswith(...): (line 84)
        # Processing the call arguments (line 84)
        str_35824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 27), 'str', '\n')
        # Processing the call keyword arguments (line 84)
        kwargs_35825 = {}
        # Getting the type of 'output' (line 84)
        output_35822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 11), 'output', False)
        # Obtaining the member 'endswith' of a type (line 84)
        endswith_35823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 11), output_35822, 'endswith')
        # Calling endswith(args, kwargs) (line 84)
        endswith_call_result_35826 = invoke(stypy.reporting.localization.Localization(__file__, 84, 11), endswith_35823, *[str_35824], **kwargs_35825)
        
        # Testing the type of an if condition (line 84)
        if_condition_35827 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 8), endswith_call_result_35826)
        # Assigning a type to the variable 'if_condition_35827' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'if_condition_35827', if_condition_35827)
        # SSA begins for if statement (line 84)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 85):
        
        # Obtaining the type of the subscript
        int_35828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 29), 'int')
        slice_35829 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 85, 21), None, int_35828, None)
        # Getting the type of 'output' (line 85)
        output_35830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'output')
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___35831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 21), output_35830, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 85)
        subscript_call_result_35832 = invoke(stypy.reporting.localization.Localization(__file__, 85, 21), getitem___35831, slice_35829)
        
        # Assigning a type to the variable 'output' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'output', subscript_call_result_35832)
        # SSA join for if statement (line 84)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assertEqual(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'cwd' (line 86)
        cwd_35835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 25), 'cwd', False)
        # Getting the type of 'output' (line 86)
        output_35836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 30), 'output', False)
        # Processing the call keyword arguments (line 86)
        kwargs_35837 = {}
        # Getting the type of 'self' (line 86)
        self_35833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 86)
        assertEqual_35834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), self_35833, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 86)
        assertEqual_call_result_35838 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), assertEqual_35834, *[cwd_35835, output_35836], **kwargs_35837)
        
        
        # ################# End of 'test_run_setup_uses_current_dir(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_run_setup_uses_current_dir' in the type store
        # Getting the type of 'stypy_return_type' (line 70)
        stypy_return_type_35839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35839)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_run_setup_uses_current_dir'
        return stypy_return_type_35839


    @norecursion
    def test_debug_mode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_debug_mode'
        module_type_store = module_type_store.open_function_context('test_debug_mode', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CoreTestCase.test_debug_mode.__dict__.__setitem__('stypy_localization', localization)
        CoreTestCase.test_debug_mode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CoreTestCase.test_debug_mode.__dict__.__setitem__('stypy_type_store', module_type_store)
        CoreTestCase.test_debug_mode.__dict__.__setitem__('stypy_function_name', 'CoreTestCase.test_debug_mode')
        CoreTestCase.test_debug_mode.__dict__.__setitem__('stypy_param_names_list', [])
        CoreTestCase.test_debug_mode.__dict__.__setitem__('stypy_varargs_param_name', None)
        CoreTestCase.test_debug_mode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CoreTestCase.test_debug_mode.__dict__.__setitem__('stypy_call_defaults', defaults)
        CoreTestCase.test_debug_mode.__dict__.__setitem__('stypy_call_varargs', varargs)
        CoreTestCase.test_debug_mode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CoreTestCase.test_debug_mode.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CoreTestCase.test_debug_mode', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_debug_mode', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_debug_mode(...)' code ##################

        
        # Assigning a List to a Attribute (line 90):
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_35840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        str_35841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 20), 'str', 'setup.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 19), list_35840, str_35841)
        # Adding element type (line 90)
        str_35842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 32), 'str', '--name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 19), list_35840, str_35842)
        
        # Getting the type of 'sys' (line 90)
        sys_35843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'sys')
        # Setting the type of the member 'argv' of a type (line 90)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), sys_35843, 'argv', list_35840)
        
        # Call to captured_stdout(...): (line 91)
        # Processing the call keyword arguments (line 91)
        kwargs_35845 = {}
        # Getting the type of 'captured_stdout' (line 91)
        captured_stdout_35844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 13), 'captured_stdout', False)
        # Calling captured_stdout(args, kwargs) (line 91)
        captured_stdout_call_result_35846 = invoke(stypy.reporting.localization.Localization(__file__, 91, 13), captured_stdout_35844, *[], **kwargs_35845)
        
        with_35847 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 91, 13), captured_stdout_call_result_35846, 'with parameter', '__enter__', '__exit__')

        if with_35847:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 91)
            enter___35848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 13), captured_stdout_call_result_35846, '__enter__')
            with_enter_35849 = invoke(stypy.reporting.localization.Localization(__file__, 91, 13), enter___35848)
            # Assigning a type to the variable 'stdout' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 13), 'stdout', with_enter_35849)
            
            # Call to setup(...): (line 92)
            # Processing the call keyword arguments (line 92)
            str_35853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 38), 'str', 'bar')
            keyword_35854 = str_35853
            kwargs_35855 = {'name': keyword_35854}
            # Getting the type of 'distutils' (line 92)
            distutils_35850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'distutils', False)
            # Obtaining the member 'core' of a type (line 92)
            core_35851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), distutils_35850, 'core')
            # Obtaining the member 'setup' of a type (line 92)
            setup_35852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), core_35851, 'setup')
            # Calling setup(args, kwargs) (line 92)
            setup_call_result_35856 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), setup_35852, *[], **kwargs_35855)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 91)
            exit___35857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 13), captured_stdout_call_result_35846, '__exit__')
            with_exit_35858 = invoke(stypy.reporting.localization.Localization(__file__, 91, 13), exit___35857, None, None, None)

        
        # Call to seek(...): (line 93)
        # Processing the call arguments (line 93)
        int_35861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 20), 'int')
        # Processing the call keyword arguments (line 93)
        kwargs_35862 = {}
        # Getting the type of 'stdout' (line 93)
        stdout_35859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'stdout', False)
        # Obtaining the member 'seek' of a type (line 93)
        seek_35860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), stdout_35859, 'seek')
        # Calling seek(args, kwargs) (line 93)
        seek_call_result_35863 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), seek_35860, *[int_35861], **kwargs_35862)
        
        
        # Call to assertEqual(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Call to read(...): (line 94)
        # Processing the call keyword arguments (line 94)
        kwargs_35868 = {}
        # Getting the type of 'stdout' (line 94)
        stdout_35866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 25), 'stdout', False)
        # Obtaining the member 'read' of a type (line 94)
        read_35867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 25), stdout_35866, 'read')
        # Calling read(args, kwargs) (line 94)
        read_call_result_35869 = invoke(stypy.reporting.localization.Localization(__file__, 94, 25), read_35867, *[], **kwargs_35868)
        
        str_35870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 40), 'str', 'bar\n')
        # Processing the call keyword arguments (line 94)
        kwargs_35871 = {}
        # Getting the type of 'self' (line 94)
        self_35864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 94)
        assertEqual_35865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_35864, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 94)
        assertEqual_call_result_35872 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), assertEqual_35865, *[read_call_result_35869, str_35870], **kwargs_35871)
        
        
        # Assigning a Name to a Attribute (line 96):
        # Getting the type of 'True' (line 96)
        True_35873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 31), 'True')
        # Getting the type of 'distutils' (line 96)
        distutils_35874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'distutils')
        # Obtaining the member 'core' of a type (line 96)
        core_35875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), distutils_35874, 'core')
        # Setting the type of the member 'DEBUG' of a type (line 96)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), core_35875, 'DEBUG', True_35873)
        
        # Try-finally block (line 97)
        
        # Call to captured_stdout(...): (line 98)
        # Processing the call keyword arguments (line 98)
        kwargs_35877 = {}
        # Getting the type of 'captured_stdout' (line 98)
        captured_stdout_35876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 'captured_stdout', False)
        # Calling captured_stdout(args, kwargs) (line 98)
        captured_stdout_call_result_35878 = invoke(stypy.reporting.localization.Localization(__file__, 98, 17), captured_stdout_35876, *[], **kwargs_35877)
        
        with_35879 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 98, 17), captured_stdout_call_result_35878, 'with parameter', '__enter__', '__exit__')

        if with_35879:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 98)
            enter___35880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 17), captured_stdout_call_result_35878, '__enter__')
            with_enter_35881 = invoke(stypy.reporting.localization.Localization(__file__, 98, 17), enter___35880)
            # Assigning a type to the variable 'stdout' (line 98)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 'stdout', with_enter_35881)
            
            # Call to setup(...): (line 99)
            # Processing the call keyword arguments (line 99)
            str_35885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 42), 'str', 'bar')
            keyword_35886 = str_35885
            kwargs_35887 = {'name': keyword_35886}
            # Getting the type of 'distutils' (line 99)
            distutils_35882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'distutils', False)
            # Obtaining the member 'core' of a type (line 99)
            core_35883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), distutils_35882, 'core')
            # Obtaining the member 'setup' of a type (line 99)
            setup_35884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), core_35883, 'setup')
            # Calling setup(args, kwargs) (line 99)
            setup_call_result_35888 = invoke(stypy.reporting.localization.Localization(__file__, 99, 16), setup_35884, *[], **kwargs_35887)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 98)
            exit___35889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 17), captured_stdout_call_result_35878, '__exit__')
            with_exit_35890 = invoke(stypy.reporting.localization.Localization(__file__, 98, 17), exit___35889, None, None, None)

        
        # finally branch of the try-finally block (line 97)
        
        # Assigning a Name to a Attribute (line 101):
        # Getting the type of 'False' (line 101)
        False_35891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 35), 'False')
        # Getting the type of 'distutils' (line 101)
        distutils_35892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'distutils')
        # Obtaining the member 'core' of a type (line 101)
        core_35893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), distutils_35892, 'core')
        # Setting the type of the member 'DEBUG' of a type (line 101)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), core_35893, 'DEBUG', False_35891)
        
        
        # Call to seek(...): (line 102)
        # Processing the call arguments (line 102)
        int_35896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 20), 'int')
        # Processing the call keyword arguments (line 102)
        kwargs_35897 = {}
        # Getting the type of 'stdout' (line 102)
        stdout_35894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'stdout', False)
        # Obtaining the member 'seek' of a type (line 102)
        seek_35895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), stdout_35894, 'seek')
        # Calling seek(args, kwargs) (line 102)
        seek_call_result_35898 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), seek_35895, *[int_35896], **kwargs_35897)
        
        
        # Assigning a Str to a Name (line 103):
        str_35899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 17), 'str', 'options (after parsing config files):\n')
        # Assigning a type to the variable 'wanted' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'wanted', str_35899)
        
        # Call to assertEqual(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Obtaining the type of the subscript
        int_35902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 44), 'int')
        
        # Call to readlines(...): (line 104)
        # Processing the call keyword arguments (line 104)
        kwargs_35905 = {}
        # Getting the type of 'stdout' (line 104)
        stdout_35903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 25), 'stdout', False)
        # Obtaining the member 'readlines' of a type (line 104)
        readlines_35904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 25), stdout_35903, 'readlines')
        # Calling readlines(args, kwargs) (line 104)
        readlines_call_result_35906 = invoke(stypy.reporting.localization.Localization(__file__, 104, 25), readlines_35904, *[], **kwargs_35905)
        
        # Obtaining the member '__getitem__' of a type (line 104)
        getitem___35907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 25), readlines_call_result_35906, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 104)
        subscript_call_result_35908 = invoke(stypy.reporting.localization.Localization(__file__, 104, 25), getitem___35907, int_35902)
        
        # Getting the type of 'wanted' (line 104)
        wanted_35909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 48), 'wanted', False)
        # Processing the call keyword arguments (line 104)
        kwargs_35910 = {}
        # Getting the type of 'self' (line 104)
        self_35900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 104)
        assertEqual_35901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), self_35900, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 104)
        assertEqual_call_result_35911 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), assertEqual_35901, *[subscript_call_result_35908, wanted_35909], **kwargs_35910)
        
        
        # ################# End of 'test_debug_mode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_debug_mode' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_35912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35912)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_debug_mode'
        return stypy_return_type_35912


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 33, 0, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CoreTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'CoreTestCase' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'CoreTestCase', CoreTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 106, 0, False)
    
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

    
    # Call to makeSuite(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'CoreTestCase' (line 107)
    CoreTestCase_35915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 30), 'CoreTestCase', False)
    # Processing the call keyword arguments (line 107)
    kwargs_35916 = {}
    # Getting the type of 'unittest' (line 107)
    unittest_35913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 107)
    makeSuite_35914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 11), unittest_35913, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 107)
    makeSuite_call_result_35917 = invoke(stypy.reporting.localization.Localization(__file__, 107, 11), makeSuite_35914, *[CoreTestCase_35915], **kwargs_35916)
    
    # Assigning a type to the variable 'stypy_return_type' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type', makeSuite_call_result_35917)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 106)
    stypy_return_type_35918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35918)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_35918

# Assigning a type to the variable 'test_suite' (line 106)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Call to test_suite(...): (line 110)
    # Processing the call keyword arguments (line 110)
    kwargs_35921 = {}
    # Getting the type of 'test_suite' (line 110)
    test_suite_35920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 110)
    test_suite_call_result_35922 = invoke(stypy.reporting.localization.Localization(__file__, 110, 17), test_suite_35920, *[], **kwargs_35921)
    
    # Processing the call keyword arguments (line 110)
    kwargs_35923 = {}
    # Getting the type of 'run_unittest' (line 110)
    run_unittest_35919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 110)
    run_unittest_call_result_35924 = invoke(stypy.reporting.localization.Localization(__file__, 110, 4), run_unittest_35919, *[test_suite_call_result_35922], **kwargs_35923)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
