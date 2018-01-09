
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.command.clean.'''
2: import sys
3: import os
4: import unittest
5: import getpass
6: 
7: from distutils.command.clean import clean
8: from distutils.tests import support
9: from test.test_support import run_unittest
10: 
11: class cleanTestCase(support.TempdirManager,
12:                     support.LoggingSilencer,
13:                     unittest.TestCase):
14: 
15:     def test_simple_run(self):
16:         pkg_dir, dist = self.create_dist()
17:         cmd = clean(dist)
18: 
19:         # let's add some elements clean should remove
20:         dirs = [(d, os.path.join(pkg_dir, d))
21:                 for d in ('build_temp', 'build_lib', 'bdist_base',
22:                 'build_scripts', 'build_base')]
23: 
24:         for name, path in dirs:
25:             os.mkdir(path)
26:             setattr(cmd, name, path)
27:             if name == 'build_base':
28:                 continue
29:             for f in ('one', 'two', 'three'):
30:                 self.write_file(os.path.join(path, f))
31: 
32:         # let's run the command
33:         cmd.all = 1
34:         cmd.ensure_finalized()
35:         cmd.run()
36: 
37:         # make sure the files where removed
38:         for name, path in dirs:
39:             self.assertFalse(os.path.exists(path),
40:                          '%s was not removed' % path)
41: 
42:         # let's run the command again (should spit warnings but succeed)
43:         cmd.all = 1
44:         cmd.ensure_finalized()
45:         cmd.run()
46: 
47: def test_suite():
48:     return unittest.makeSuite(cleanTestCase)
49: 
50: if __name__ == "__main__":
51:     run_unittest(test_suite())
52: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_34612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.command.clean.')
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

# 'import getpass' statement (line 5)
import getpass

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'getpass', getpass, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.command.clean import clean' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_34613 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.clean')

if (type(import_34613) is not StypyTypeError):

    if (import_34613 != 'pyd_module'):
        __import__(import_34613)
        sys_modules_34614 = sys.modules[import_34613]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.clean', sys_modules_34614.module_type_store, module_type_store, ['clean'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_34614, sys_modules_34614.module_type_store, module_type_store)
    else:
        from distutils.command.clean import clean

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.clean', None, module_type_store, ['clean'], [clean])

else:
    # Assigning a type to the variable 'distutils.command.clean' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.clean', import_34613)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.tests import support' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_34615 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests')

if (type(import_34615) is not StypyTypeError):

    if (import_34615 != 'pyd_module'):
        __import__(import_34615)
        sys_modules_34616 = sys.modules[import_34615]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', sys_modules_34616.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_34616, sys_modules_34616.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.tests', import_34615)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from test.test_support import run_unittest' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_34617 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support')

if (type(import_34617) is not StypyTypeError):

    if (import_34617 != 'pyd_module'):
        __import__(import_34617)
        sys_modules_34618 = sys.modules[import_34617]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', sys_modules_34618.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_34618, sys_modules_34618.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'test.test_support', import_34617)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# Declaration of the 'cleanTestCase' class
# Getting the type of 'support' (line 11)
support_34619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 20), 'support')
# Obtaining the member 'TempdirManager' of a type (line 11)
TempdirManager_34620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 20), support_34619, 'TempdirManager')
# Getting the type of 'support' (line 12)
support_34621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 20), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 12)
LoggingSilencer_34622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 20), support_34621, 'LoggingSilencer')
# Getting the type of 'unittest' (line 13)
unittest_34623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 20), 'unittest')
# Obtaining the member 'TestCase' of a type (line 13)
TestCase_34624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 20), unittest_34623, 'TestCase')

class cleanTestCase(TempdirManager_34620, LoggingSilencer_34622, TestCase_34624, ):

    @norecursion
    def test_simple_run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_run'
        module_type_store = module_type_store.open_function_context('test_simple_run', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        cleanTestCase.test_simple_run.__dict__.__setitem__('stypy_localization', localization)
        cleanTestCase.test_simple_run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        cleanTestCase.test_simple_run.__dict__.__setitem__('stypy_type_store', module_type_store)
        cleanTestCase.test_simple_run.__dict__.__setitem__('stypy_function_name', 'cleanTestCase.test_simple_run')
        cleanTestCase.test_simple_run.__dict__.__setitem__('stypy_param_names_list', [])
        cleanTestCase.test_simple_run.__dict__.__setitem__('stypy_varargs_param_name', None)
        cleanTestCase.test_simple_run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        cleanTestCase.test_simple_run.__dict__.__setitem__('stypy_call_defaults', defaults)
        cleanTestCase.test_simple_run.__dict__.__setitem__('stypy_call_varargs', varargs)
        cleanTestCase.test_simple_run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        cleanTestCase.test_simple_run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'cleanTestCase.test_simple_run', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_run', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_run(...)' code ##################

        
        # Assigning a Call to a Tuple (line 16):
        
        # Assigning a Subscript to a Name (line 16):
        
        # Obtaining the type of the subscript
        int_34625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 8), 'int')
        
        # Call to create_dist(...): (line 16)
        # Processing the call keyword arguments (line 16)
        kwargs_34628 = {}
        # Getting the type of 'self' (line 16)
        self_34626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 16)
        create_dist_34627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 24), self_34626, 'create_dist')
        # Calling create_dist(args, kwargs) (line 16)
        create_dist_call_result_34629 = invoke(stypy.reporting.localization.Localization(__file__, 16, 24), create_dist_34627, *[], **kwargs_34628)
        
        # Obtaining the member '__getitem__' of a type (line 16)
        getitem___34630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), create_dist_call_result_34629, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 16)
        subscript_call_result_34631 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), getitem___34630, int_34625)
        
        # Assigning a type to the variable 'tuple_var_assignment_34610' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'tuple_var_assignment_34610', subscript_call_result_34631)
        
        # Assigning a Subscript to a Name (line 16):
        
        # Obtaining the type of the subscript
        int_34632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 8), 'int')
        
        # Call to create_dist(...): (line 16)
        # Processing the call keyword arguments (line 16)
        kwargs_34635 = {}
        # Getting the type of 'self' (line 16)
        self_34633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 24), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 16)
        create_dist_34634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 24), self_34633, 'create_dist')
        # Calling create_dist(args, kwargs) (line 16)
        create_dist_call_result_34636 = invoke(stypy.reporting.localization.Localization(__file__, 16, 24), create_dist_34634, *[], **kwargs_34635)
        
        # Obtaining the member '__getitem__' of a type (line 16)
        getitem___34637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), create_dist_call_result_34636, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 16)
        subscript_call_result_34638 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), getitem___34637, int_34632)
        
        # Assigning a type to the variable 'tuple_var_assignment_34611' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'tuple_var_assignment_34611', subscript_call_result_34638)
        
        # Assigning a Name to a Name (line 16):
        # Getting the type of 'tuple_var_assignment_34610' (line 16)
        tuple_var_assignment_34610_34639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'tuple_var_assignment_34610')
        # Assigning a type to the variable 'pkg_dir' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'pkg_dir', tuple_var_assignment_34610_34639)
        
        # Assigning a Name to a Name (line 16):
        # Getting the type of 'tuple_var_assignment_34611' (line 16)
        tuple_var_assignment_34611_34640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'tuple_var_assignment_34611')
        # Assigning a type to the variable 'dist' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 17), 'dist', tuple_var_assignment_34611_34640)
        
        # Assigning a Call to a Name (line 17):
        
        # Assigning a Call to a Name (line 17):
        
        # Call to clean(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'dist' (line 17)
        dist_34642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), 'dist', False)
        # Processing the call keyword arguments (line 17)
        kwargs_34643 = {}
        # Getting the type of 'clean' (line 17)
        clean_34641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 14), 'clean', False)
        # Calling clean(args, kwargs) (line 17)
        clean_call_result_34644 = invoke(stypy.reporting.localization.Localization(__file__, 17, 14), clean_34641, *[dist_34642], **kwargs_34643)
        
        # Assigning a type to the variable 'cmd' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'cmd', clean_call_result_34644)
        
        # Assigning a ListComp to a Name (line 20):
        
        # Assigning a ListComp to a Name (line 20):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'tuple' (line 21)
        tuple_34654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 21)
        # Adding element type (line 21)
        str_34655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 26), 'str', 'build_temp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 26), tuple_34654, str_34655)
        # Adding element type (line 21)
        str_34656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 40), 'str', 'build_lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 26), tuple_34654, str_34656)
        # Adding element type (line 21)
        str_34657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 53), 'str', 'bdist_base')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 26), tuple_34654, str_34657)
        # Adding element type (line 21)
        str_34658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'str', 'build_scripts')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 26), tuple_34654, str_34658)
        # Adding element type (line 21)
        str_34659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 33), 'str', 'build_base')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 26), tuple_34654, str_34659)
        
        comprehension_34660 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 16), tuple_34654)
        # Assigning a type to the variable 'd' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'd', comprehension_34660)
        
        # Obtaining an instance of the builtin type 'tuple' (line 20)
        tuple_34645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 20)
        # Adding element type (line 20)
        # Getting the type of 'd' (line 20)
        d_34646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 17), 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 17), tuple_34645, d_34646)
        # Adding element type (line 20)
        
        # Call to join(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'pkg_dir' (line 20)
        pkg_dir_34650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 33), 'pkg_dir', False)
        # Getting the type of 'd' (line 20)
        d_34651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 42), 'd', False)
        # Processing the call keyword arguments (line 20)
        kwargs_34652 = {}
        # Getting the type of 'os' (line 20)
        os_34647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 20)
        path_34648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 20), os_34647, 'path')
        # Obtaining the member 'join' of a type (line 20)
        join_34649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 20), path_34648, 'join')
        # Calling join(args, kwargs) (line 20)
        join_call_result_34653 = invoke(stypy.reporting.localization.Localization(__file__, 20, 20), join_34649, *[pkg_dir_34650, d_34651], **kwargs_34652)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 17), tuple_34645, join_call_result_34653)
        
        list_34661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 16), list_34661, tuple_34645)
        # Assigning a type to the variable 'dirs' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'dirs', list_34661)
        
        # Getting the type of 'dirs' (line 24)
        dirs_34662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 26), 'dirs')
        # Testing the type of a for loop iterable (line 24)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 24, 8), dirs_34662)
        # Getting the type of the for loop variable (line 24)
        for_loop_var_34663 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 24, 8), dirs_34662)
        # Assigning a type to the variable 'name' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 8), for_loop_var_34663))
        # Assigning a type to the variable 'path' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'path', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 8), for_loop_var_34663))
        # SSA begins for a for statement (line 24)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to mkdir(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'path' (line 25)
        path_34666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 21), 'path', False)
        # Processing the call keyword arguments (line 25)
        kwargs_34667 = {}
        # Getting the type of 'os' (line 25)
        os_34664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 25)
        mkdir_34665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 12), os_34664, 'mkdir')
        # Calling mkdir(args, kwargs) (line 25)
        mkdir_call_result_34668 = invoke(stypy.reporting.localization.Localization(__file__, 25, 12), mkdir_34665, *[path_34666], **kwargs_34667)
        
        
        # Call to setattr(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'cmd' (line 26)
        cmd_34670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 20), 'cmd', False)
        # Getting the type of 'name' (line 26)
        name_34671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 25), 'name', False)
        # Getting the type of 'path' (line 26)
        path_34672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 31), 'path', False)
        # Processing the call keyword arguments (line 26)
        kwargs_34673 = {}
        # Getting the type of 'setattr' (line 26)
        setattr_34669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'setattr', False)
        # Calling setattr(args, kwargs) (line 26)
        setattr_call_result_34674 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), setattr_34669, *[cmd_34670, name_34671, path_34672], **kwargs_34673)
        
        
        
        # Getting the type of 'name' (line 27)
        name_34675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'name')
        str_34676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 23), 'str', 'build_base')
        # Applying the binary operator '==' (line 27)
        result_eq_34677 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 15), '==', name_34675, str_34676)
        
        # Testing the type of an if condition (line 27)
        if_condition_34678 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 12), result_eq_34677)
        # Assigning a type to the variable 'if_condition_34678' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'if_condition_34678', if_condition_34678)
        # SSA begins for if statement (line 27)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 27)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 29)
        tuple_34679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 29)
        # Adding element type (line 29)
        str_34680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 22), 'str', 'one')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 22), tuple_34679, str_34680)
        # Adding element type (line 29)
        str_34681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 29), 'str', 'two')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 22), tuple_34679, str_34681)
        # Adding element type (line 29)
        str_34682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 36), 'str', 'three')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 22), tuple_34679, str_34682)
        
        # Testing the type of a for loop iterable (line 29)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 29, 12), tuple_34679)
        # Getting the type of the for loop variable (line 29)
        for_loop_var_34683 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 29, 12), tuple_34679)
        # Assigning a type to the variable 'f' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'f', for_loop_var_34683)
        # SSA begins for a for statement (line 29)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to write_file(...): (line 30)
        # Processing the call arguments (line 30)
        
        # Call to join(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'path' (line 30)
        path_34689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 45), 'path', False)
        # Getting the type of 'f' (line 30)
        f_34690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 51), 'f', False)
        # Processing the call keyword arguments (line 30)
        kwargs_34691 = {}
        # Getting the type of 'os' (line 30)
        os_34686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 32), 'os', False)
        # Obtaining the member 'path' of a type (line 30)
        path_34687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 32), os_34686, 'path')
        # Obtaining the member 'join' of a type (line 30)
        join_34688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 32), path_34687, 'join')
        # Calling join(args, kwargs) (line 30)
        join_call_result_34692 = invoke(stypy.reporting.localization.Localization(__file__, 30, 32), join_34688, *[path_34689, f_34690], **kwargs_34691)
        
        # Processing the call keyword arguments (line 30)
        kwargs_34693 = {}
        # Getting the type of 'self' (line 30)
        self_34684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'self', False)
        # Obtaining the member 'write_file' of a type (line 30)
        write_file_34685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 16), self_34684, 'write_file')
        # Calling write_file(args, kwargs) (line 30)
        write_file_call_result_34694 = invoke(stypy.reporting.localization.Localization(__file__, 30, 16), write_file_34685, *[join_call_result_34692], **kwargs_34693)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Attribute (line 33):
        
        # Assigning a Num to a Attribute (line 33):
        int_34695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 18), 'int')
        # Getting the type of 'cmd' (line 33)
        cmd_34696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'cmd')
        # Setting the type of the member 'all' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), cmd_34696, 'all', int_34695)
        
        # Call to ensure_finalized(...): (line 34)
        # Processing the call keyword arguments (line 34)
        kwargs_34699 = {}
        # Getting the type of 'cmd' (line 34)
        cmd_34697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 34)
        ensure_finalized_34698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), cmd_34697, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 34)
        ensure_finalized_call_result_34700 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), ensure_finalized_34698, *[], **kwargs_34699)
        
        
        # Call to run(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_34703 = {}
        # Getting the type of 'cmd' (line 35)
        cmd_34701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'cmd', False)
        # Obtaining the member 'run' of a type (line 35)
        run_34702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), cmd_34701, 'run')
        # Calling run(args, kwargs) (line 35)
        run_call_result_34704 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), run_34702, *[], **kwargs_34703)
        
        
        # Getting the type of 'dirs' (line 38)
        dirs_34705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 26), 'dirs')
        # Testing the type of a for loop iterable (line 38)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 38, 8), dirs_34705)
        # Getting the type of the for loop variable (line 38)
        for_loop_var_34706 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 38, 8), dirs_34705)
        # Assigning a type to the variable 'name' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 8), for_loop_var_34706))
        # Assigning a type to the variable 'path' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'path', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 8), for_loop_var_34706))
        # SSA begins for a for statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assertFalse(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Call to exists(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'path' (line 39)
        path_34712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 44), 'path', False)
        # Processing the call keyword arguments (line 39)
        kwargs_34713 = {}
        # Getting the type of 'os' (line 39)
        os_34709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'os', False)
        # Obtaining the member 'path' of a type (line 39)
        path_34710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 29), os_34709, 'path')
        # Obtaining the member 'exists' of a type (line 39)
        exists_34711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 29), path_34710, 'exists')
        # Calling exists(args, kwargs) (line 39)
        exists_call_result_34714 = invoke(stypy.reporting.localization.Localization(__file__, 39, 29), exists_34711, *[path_34712], **kwargs_34713)
        
        str_34715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 25), 'str', '%s was not removed')
        # Getting the type of 'path' (line 40)
        path_34716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 48), 'path', False)
        # Applying the binary operator '%' (line 40)
        result_mod_34717 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 25), '%', str_34715, path_34716)
        
        # Processing the call keyword arguments (line 39)
        kwargs_34718 = {}
        # Getting the type of 'self' (line 39)
        self_34707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 39)
        assertFalse_34708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 12), self_34707, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 39)
        assertFalse_call_result_34719 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), assertFalse_34708, *[exists_call_result_34714, result_mod_34717], **kwargs_34718)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Attribute (line 43):
        
        # Assigning a Num to a Attribute (line 43):
        int_34720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 18), 'int')
        # Getting the type of 'cmd' (line 43)
        cmd_34721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'cmd')
        # Setting the type of the member 'all' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), cmd_34721, 'all', int_34720)
        
        # Call to ensure_finalized(...): (line 44)
        # Processing the call keyword arguments (line 44)
        kwargs_34724 = {}
        # Getting the type of 'cmd' (line 44)
        cmd_34722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 44)
        ensure_finalized_34723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), cmd_34722, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 44)
        ensure_finalized_call_result_34725 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), ensure_finalized_34723, *[], **kwargs_34724)
        
        
        # Call to run(...): (line 45)
        # Processing the call keyword arguments (line 45)
        kwargs_34728 = {}
        # Getting the type of 'cmd' (line 45)
        cmd_34726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'cmd', False)
        # Obtaining the member 'run' of a type (line 45)
        run_34727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), cmd_34726, 'run')
        # Calling run(args, kwargs) (line 45)
        run_call_result_34729 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), run_34727, *[], **kwargs_34728)
        
        
        # ################# End of 'test_simple_run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_run' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_34730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_34730)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_run'
        return stypy_return_type_34730


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 11, 0, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'cleanTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'cleanTestCase' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'cleanTestCase', cleanTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 47, 0, False)
    
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

    
    # Call to makeSuite(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'cleanTestCase' (line 48)
    cleanTestCase_34733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 30), 'cleanTestCase', False)
    # Processing the call keyword arguments (line 48)
    kwargs_34734 = {}
    # Getting the type of 'unittest' (line 48)
    unittest_34731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 48)
    makeSuite_34732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 11), unittest_34731, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 48)
    makeSuite_call_result_34735 = invoke(stypy.reporting.localization.Localization(__file__, 48, 11), makeSuite_34732, *[cleanTestCase_34733], **kwargs_34734)
    
    # Assigning a type to the variable 'stypy_return_type' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type', makeSuite_call_result_34735)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_34736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34736)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_34736

# Assigning a type to the variable 'test_suite' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 51)
    # Processing the call arguments (line 51)
    
    # Call to test_suite(...): (line 51)
    # Processing the call keyword arguments (line 51)
    kwargs_34739 = {}
    # Getting the type of 'test_suite' (line 51)
    test_suite_34738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 51)
    test_suite_call_result_34740 = invoke(stypy.reporting.localization.Localization(__file__, 51, 17), test_suite_34738, *[], **kwargs_34739)
    
    # Processing the call keyword arguments (line 51)
    kwargs_34741 = {}
    # Getting the type of 'run_unittest' (line 51)
    run_unittest_34737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 51)
    run_unittest_call_result_34742 = invoke(stypy.reporting.localization.Localization(__file__, 51, 4), run_unittest_34737, *[test_suite_call_result_34740], **kwargs_34741)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
