
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.version.'''
2: import unittest
3: from distutils.version import LooseVersion
4: from distutils.version import StrictVersion
5: from test.test_support import run_unittest
6: 
7: class VersionTestCase(unittest.TestCase):
8: 
9:     def test_prerelease(self):
10:         version = StrictVersion('1.2.3a1')
11:         self.assertEqual(version.version, (1, 2, 3))
12:         self.assertEqual(version.prerelease, ('a', 1))
13:         self.assertEqual(str(version), '1.2.3a1')
14: 
15:         version = StrictVersion('1.2.0')
16:         self.assertEqual(str(version), '1.2')
17: 
18:     def test_cmp_strict(self):
19:         versions = (('1.5.1', '1.5.2b2', -1),
20:                     ('161', '3.10a', ValueError),
21:                     ('8.02', '8.02', 0),
22:                     ('3.4j', '1996.07.12', ValueError),
23:                     ('3.2.pl0', '3.1.1.6', ValueError),
24:                     ('2g6', '11g', ValueError),
25:                     ('0.9', '2.2', -1),
26:                     ('1.2.1', '1.2', 1),
27:                     ('1.1', '1.2.2', -1),
28:                     ('1.2', '1.1', 1),
29:                     ('1.2.1', '1.2.2', -1),
30:                     ('1.2.2', '1.2', 1),
31:                     ('1.2', '1.2.2', -1),
32:                     ('0.4.0', '0.4', 0),
33:                     ('1.13++', '5.5.kw', ValueError))
34: 
35:         for v1, v2, wanted in versions:
36:             try:
37:                 res = StrictVersion(v1).__cmp__(StrictVersion(v2))
38:             except ValueError:
39:                 if wanted is ValueError:
40:                     continue
41:                 else:
42:                     raise AssertionError(("cmp(%s, %s) "
43:                                           "shouldn't raise ValueError")
44:                                             % (v1, v2))
45:             self.assertEqual(res, wanted,
46:                              'cmp(%s, %s) should be %s, got %s' %
47:                              (v1, v2, wanted, res))
48: 
49: 
50:     def test_cmp(self):
51:         versions = (('1.5.1', '1.5.2b2', -1),
52:                     ('161', '3.10a', 1),
53:                     ('8.02', '8.02', 0),
54:                     ('3.4j', '1996.07.12', -1),
55:                     ('3.2.pl0', '3.1.1.6', 1),
56:                     ('2g6', '11g', -1),
57:                     ('0.960923', '2.2beta29', -1),
58:                     ('1.13++', '5.5.kw', -1))
59: 
60: 
61:         for v1, v2, wanted in versions:
62:             res = LooseVersion(v1).__cmp__(LooseVersion(v2))
63:             self.assertEqual(res, wanted,
64:                              'cmp(%s, %s) should be %s, got %s' %
65:                              (v1, v2, wanted, res))
66: 
67: def test_suite():
68:     return unittest.makeSuite(VersionTestCase)
69: 
70: if __name__ == "__main__":
71:     run_unittest(test_suite())
72: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_45580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.version.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import unittest' statement (line 2)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from distutils.version import LooseVersion' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_45581 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'distutils.version')

if (type(import_45581) is not StypyTypeError):

    if (import_45581 != 'pyd_module'):
        __import__(import_45581)
        sys_modules_45582 = sys.modules[import_45581]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'distutils.version', sys_modules_45582.module_type_store, module_type_store, ['LooseVersion'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_45582, sys_modules_45582.module_type_store, module_type_store)
    else:
        from distutils.version import LooseVersion

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'distutils.version', None, module_type_store, ['LooseVersion'], [LooseVersion])

else:
    # Assigning a type to the variable 'distutils.version' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'distutils.version', import_45581)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from distutils.version import StrictVersion' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_45583 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.version')

if (type(import_45583) is not StypyTypeError):

    if (import_45583 != 'pyd_module'):
        __import__(import_45583)
        sys_modules_45584 = sys.modules[import_45583]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.version', sys_modules_45584.module_type_store, module_type_store, ['StrictVersion'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_45584, sys_modules_45584.module_type_store, module_type_store)
    else:
        from distutils.version import StrictVersion

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.version', None, module_type_store, ['StrictVersion'], [StrictVersion])

else:
    # Assigning a type to the variable 'distutils.version' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.version', import_45583)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from test.test_support import run_unittest' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_45585 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support')

if (type(import_45585) is not StypyTypeError):

    if (import_45585 != 'pyd_module'):
        __import__(import_45585)
        sys_modules_45586 = sys.modules[import_45585]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', sys_modules_45586.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_45586, sys_modules_45586.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'test.test_support', import_45585)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# Declaration of the 'VersionTestCase' class
# Getting the type of 'unittest' (line 7)
unittest_45587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 22), 'unittest')
# Obtaining the member 'TestCase' of a type (line 7)
TestCase_45588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 22), unittest_45587, 'TestCase')

class VersionTestCase(TestCase_45588, ):

    @norecursion
    def test_prerelease(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_prerelease'
        module_type_store = module_type_store.open_function_context('test_prerelease', 9, 4, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VersionTestCase.test_prerelease.__dict__.__setitem__('stypy_localization', localization)
        VersionTestCase.test_prerelease.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VersionTestCase.test_prerelease.__dict__.__setitem__('stypy_type_store', module_type_store)
        VersionTestCase.test_prerelease.__dict__.__setitem__('stypy_function_name', 'VersionTestCase.test_prerelease')
        VersionTestCase.test_prerelease.__dict__.__setitem__('stypy_param_names_list', [])
        VersionTestCase.test_prerelease.__dict__.__setitem__('stypy_varargs_param_name', None)
        VersionTestCase.test_prerelease.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VersionTestCase.test_prerelease.__dict__.__setitem__('stypy_call_defaults', defaults)
        VersionTestCase.test_prerelease.__dict__.__setitem__('stypy_call_varargs', varargs)
        VersionTestCase.test_prerelease.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VersionTestCase.test_prerelease.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VersionTestCase.test_prerelease', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_prerelease', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_prerelease(...)' code ##################

        
        # Assigning a Call to a Name (line 10):
        
        # Call to StrictVersion(...): (line 10)
        # Processing the call arguments (line 10)
        str_45590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 32), 'str', '1.2.3a1')
        # Processing the call keyword arguments (line 10)
        kwargs_45591 = {}
        # Getting the type of 'StrictVersion' (line 10)
        StrictVersion_45589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 18), 'StrictVersion', False)
        # Calling StrictVersion(args, kwargs) (line 10)
        StrictVersion_call_result_45592 = invoke(stypy.reporting.localization.Localization(__file__, 10, 18), StrictVersion_45589, *[str_45590], **kwargs_45591)
        
        # Assigning a type to the variable 'version' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'version', StrictVersion_call_result_45592)
        
        # Call to assertEqual(...): (line 11)
        # Processing the call arguments (line 11)
        # Getting the type of 'version' (line 11)
        version_45595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 25), 'version', False)
        # Obtaining the member 'version' of a type (line 11)
        version_45596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 25), version_45595, 'version')
        
        # Obtaining an instance of the builtin type 'tuple' (line 11)
        tuple_45597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 11)
        # Adding element type (line 11)
        int_45598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 43), tuple_45597, int_45598)
        # Adding element type (line 11)
        int_45599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 43), tuple_45597, int_45599)
        # Adding element type (line 11)
        int_45600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 43), tuple_45597, int_45600)
        
        # Processing the call keyword arguments (line 11)
        kwargs_45601 = {}
        # Getting the type of 'self' (line 11)
        self_45593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 11)
        assertEqual_45594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 8), self_45593, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 11)
        assertEqual_call_result_45602 = invoke(stypy.reporting.localization.Localization(__file__, 11, 8), assertEqual_45594, *[version_45596, tuple_45597], **kwargs_45601)
        
        
        # Call to assertEqual(...): (line 12)
        # Processing the call arguments (line 12)
        # Getting the type of 'version' (line 12)
        version_45605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 25), 'version', False)
        # Obtaining the member 'prerelease' of a type (line 12)
        prerelease_45606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 25), version_45605, 'prerelease')
        
        # Obtaining an instance of the builtin type 'tuple' (line 12)
        tuple_45607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 12)
        # Adding element type (line 12)
        str_45608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 46), 'str', 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 46), tuple_45607, str_45608)
        # Adding element type (line 12)
        int_45609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 46), tuple_45607, int_45609)
        
        # Processing the call keyword arguments (line 12)
        kwargs_45610 = {}
        # Getting the type of 'self' (line 12)
        self_45603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 12)
        assertEqual_45604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), self_45603, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 12)
        assertEqual_call_result_45611 = invoke(stypy.reporting.localization.Localization(__file__, 12, 8), assertEqual_45604, *[prerelease_45606, tuple_45607], **kwargs_45610)
        
        
        # Call to assertEqual(...): (line 13)
        # Processing the call arguments (line 13)
        
        # Call to str(...): (line 13)
        # Processing the call arguments (line 13)
        # Getting the type of 'version' (line 13)
        version_45615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 29), 'version', False)
        # Processing the call keyword arguments (line 13)
        kwargs_45616 = {}
        # Getting the type of 'str' (line 13)
        str_45614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 25), 'str', False)
        # Calling str(args, kwargs) (line 13)
        str_call_result_45617 = invoke(stypy.reporting.localization.Localization(__file__, 13, 25), str_45614, *[version_45615], **kwargs_45616)
        
        str_45618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 39), 'str', '1.2.3a1')
        # Processing the call keyword arguments (line 13)
        kwargs_45619 = {}
        # Getting the type of 'self' (line 13)
        self_45612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 13)
        assertEqual_45613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), self_45612, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 13)
        assertEqual_call_result_45620 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), assertEqual_45613, *[str_call_result_45617, str_45618], **kwargs_45619)
        
        
        # Assigning a Call to a Name (line 15):
        
        # Call to StrictVersion(...): (line 15)
        # Processing the call arguments (line 15)
        str_45622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 32), 'str', '1.2.0')
        # Processing the call keyword arguments (line 15)
        kwargs_45623 = {}
        # Getting the type of 'StrictVersion' (line 15)
        StrictVersion_45621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 18), 'StrictVersion', False)
        # Calling StrictVersion(args, kwargs) (line 15)
        StrictVersion_call_result_45624 = invoke(stypy.reporting.localization.Localization(__file__, 15, 18), StrictVersion_45621, *[str_45622], **kwargs_45623)
        
        # Assigning a type to the variable 'version' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'version', StrictVersion_call_result_45624)
        
        # Call to assertEqual(...): (line 16)
        # Processing the call arguments (line 16)
        
        # Call to str(...): (line 16)
        # Processing the call arguments (line 16)
        # Getting the type of 'version' (line 16)
        version_45628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 29), 'version', False)
        # Processing the call keyword arguments (line 16)
        kwargs_45629 = {}
        # Getting the type of 'str' (line 16)
        str_45627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 25), 'str', False)
        # Calling str(args, kwargs) (line 16)
        str_call_result_45630 = invoke(stypy.reporting.localization.Localization(__file__, 16, 25), str_45627, *[version_45628], **kwargs_45629)
        
        str_45631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 39), 'str', '1.2')
        # Processing the call keyword arguments (line 16)
        kwargs_45632 = {}
        # Getting the type of 'self' (line 16)
        self_45625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 16)
        assertEqual_45626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), self_45625, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 16)
        assertEqual_call_result_45633 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), assertEqual_45626, *[str_call_result_45630, str_45631], **kwargs_45632)
        
        
        # ################# End of 'test_prerelease(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_prerelease' in the type store
        # Getting the type of 'stypy_return_type' (line 9)
        stypy_return_type_45634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_45634)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_prerelease'
        return stypy_return_type_45634


    @norecursion
    def test_cmp_strict(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cmp_strict'
        module_type_store = module_type_store.open_function_context('test_cmp_strict', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VersionTestCase.test_cmp_strict.__dict__.__setitem__('stypy_localization', localization)
        VersionTestCase.test_cmp_strict.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VersionTestCase.test_cmp_strict.__dict__.__setitem__('stypy_type_store', module_type_store)
        VersionTestCase.test_cmp_strict.__dict__.__setitem__('stypy_function_name', 'VersionTestCase.test_cmp_strict')
        VersionTestCase.test_cmp_strict.__dict__.__setitem__('stypy_param_names_list', [])
        VersionTestCase.test_cmp_strict.__dict__.__setitem__('stypy_varargs_param_name', None)
        VersionTestCase.test_cmp_strict.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VersionTestCase.test_cmp_strict.__dict__.__setitem__('stypy_call_defaults', defaults)
        VersionTestCase.test_cmp_strict.__dict__.__setitem__('stypy_call_varargs', varargs)
        VersionTestCase.test_cmp_strict.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VersionTestCase.test_cmp_strict.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VersionTestCase.test_cmp_strict', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cmp_strict', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cmp_strict(...)' code ##################

        
        # Assigning a Tuple to a Name (line 19):
        
        # Obtaining an instance of the builtin type 'tuple' (line 19)
        tuple_45635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 19)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 19)
        tuple_45636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 19)
        # Adding element type (line 19)
        str_45637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 21), 'str', '1.5.1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 21), tuple_45636, str_45637)
        # Adding element type (line 19)
        str_45638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 30), 'str', '1.5.2b2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 21), tuple_45636, str_45638)
        # Adding element type (line 19)
        int_45639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 21), tuple_45636, int_45639)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_45635, tuple_45636)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 20)
        tuple_45640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 20)
        # Adding element type (line 20)
        str_45641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 21), 'str', '161')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 21), tuple_45640, str_45641)
        # Adding element type (line 20)
        str_45642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 28), 'str', '3.10a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 21), tuple_45640, str_45642)
        # Adding element type (line 20)
        # Getting the type of 'ValueError' (line 20)
        ValueError_45643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 37), 'ValueError')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 21), tuple_45640, ValueError_45643)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_45635, tuple_45640)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 21)
        tuple_45644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 21)
        # Adding element type (line 21)
        str_45645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 21), 'str', '8.02')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 21), tuple_45644, str_45645)
        # Adding element type (line 21)
        str_45646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 29), 'str', '8.02')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 21), tuple_45644, str_45646)
        # Adding element type (line 21)
        int_45647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 21), tuple_45644, int_45647)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_45635, tuple_45644)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 22)
        tuple_45648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 22)
        # Adding element type (line 22)
        str_45649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'str', '3.4j')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 21), tuple_45648, str_45649)
        # Adding element type (line 22)
        str_45650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 29), 'str', '1996.07.12')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 21), tuple_45648, str_45650)
        # Adding element type (line 22)
        # Getting the type of 'ValueError' (line 22)
        ValueError_45651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 43), 'ValueError')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 21), tuple_45648, ValueError_45651)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_45635, tuple_45648)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 23)
        tuple_45652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 23)
        # Adding element type (line 23)
        str_45653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'str', '3.2.pl0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 21), tuple_45652, str_45653)
        # Adding element type (line 23)
        str_45654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 32), 'str', '3.1.1.6')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 21), tuple_45652, str_45654)
        # Adding element type (line 23)
        # Getting the type of 'ValueError' (line 23)
        ValueError_45655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 43), 'ValueError')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 21), tuple_45652, ValueError_45655)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_45635, tuple_45652)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 24)
        tuple_45656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 24)
        # Adding element type (line 24)
        str_45657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 21), 'str', '2g6')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 21), tuple_45656, str_45657)
        # Adding element type (line 24)
        str_45658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 28), 'str', '11g')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 21), tuple_45656, str_45658)
        # Adding element type (line 24)
        # Getting the type of 'ValueError' (line 24)
        ValueError_45659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 35), 'ValueError')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 21), tuple_45656, ValueError_45659)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_45635, tuple_45656)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 25)
        tuple_45660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 25)
        # Adding element type (line 25)
        str_45661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 21), 'str', '0.9')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 21), tuple_45660, str_45661)
        # Adding element type (line 25)
        str_45662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 28), 'str', '2.2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 21), tuple_45660, str_45662)
        # Adding element type (line 25)
        int_45663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 21), tuple_45660, int_45663)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_45635, tuple_45660)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 26)
        tuple_45664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 26)
        # Adding element type (line 26)
        str_45665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 21), 'str', '1.2.1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), tuple_45664, str_45665)
        # Adding element type (line 26)
        str_45666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 30), 'str', '1.2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), tuple_45664, str_45666)
        # Adding element type (line 26)
        int_45667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), tuple_45664, int_45667)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_45635, tuple_45664)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 27)
        tuple_45668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 27)
        # Adding element type (line 27)
        str_45669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 21), 'str', '1.1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 21), tuple_45668, str_45669)
        # Adding element type (line 27)
        str_45670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 28), 'str', '1.2.2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 21), tuple_45668, str_45670)
        # Adding element type (line 27)
        int_45671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 21), tuple_45668, int_45671)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_45635, tuple_45668)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 28)
        tuple_45672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 28)
        # Adding element type (line 28)
        str_45673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 21), 'str', '1.2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 21), tuple_45672, str_45673)
        # Adding element type (line 28)
        str_45674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 28), 'str', '1.1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 21), tuple_45672, str_45674)
        # Adding element type (line 28)
        int_45675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 21), tuple_45672, int_45675)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_45635, tuple_45672)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 29)
        tuple_45676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 29)
        # Adding element type (line 29)
        str_45677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 21), 'str', '1.2.1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 21), tuple_45676, str_45677)
        # Adding element type (line 29)
        str_45678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 30), 'str', '1.2.2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 21), tuple_45676, str_45678)
        # Adding element type (line 29)
        int_45679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 21), tuple_45676, int_45679)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_45635, tuple_45676)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 30)
        tuple_45680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 30)
        # Adding element type (line 30)
        str_45681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 21), 'str', '1.2.2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 21), tuple_45680, str_45681)
        # Adding element type (line 30)
        str_45682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 30), 'str', '1.2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 21), tuple_45680, str_45682)
        # Adding element type (line 30)
        int_45683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 21), tuple_45680, int_45683)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_45635, tuple_45680)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 31)
        tuple_45684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 31)
        # Adding element type (line 31)
        str_45685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 21), 'str', '1.2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 21), tuple_45684, str_45685)
        # Adding element type (line 31)
        str_45686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 28), 'str', '1.2.2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 21), tuple_45684, str_45686)
        # Adding element type (line 31)
        int_45687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 21), tuple_45684, int_45687)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_45635, tuple_45684)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 32)
        tuple_45688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 32)
        # Adding element type (line 32)
        str_45689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 21), 'str', '0.4.0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 21), tuple_45688, str_45689)
        # Adding element type (line 32)
        str_45690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 30), 'str', '0.4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 21), tuple_45688, str_45690)
        # Adding element type (line 32)
        int_45691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 21), tuple_45688, int_45691)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_45635, tuple_45688)
        # Adding element type (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 33)
        tuple_45692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 33)
        # Adding element type (line 33)
        str_45693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 21), 'str', '1.13++')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), tuple_45692, str_45693)
        # Adding element type (line 33)
        str_45694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 31), 'str', '5.5.kw')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), tuple_45692, str_45694)
        # Adding element type (line 33)
        # Getting the type of 'ValueError' (line 33)
        ValueError_45695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 41), 'ValueError')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), tuple_45692, ValueError_45695)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_45635, tuple_45692)
        
        # Assigning a type to the variable 'versions' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'versions', tuple_45635)
        
        # Getting the type of 'versions' (line 35)
        versions_45696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 30), 'versions')
        # Testing the type of a for loop iterable (line 35)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 35, 8), versions_45696)
        # Getting the type of the for loop variable (line 35)
        for_loop_var_45697 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 35, 8), versions_45696)
        # Assigning a type to the variable 'v1' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'v1', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 8), for_loop_var_45697))
        # Assigning a type to the variable 'v2' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'v2', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 8), for_loop_var_45697))
        # Assigning a type to the variable 'wanted' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'wanted', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 8), for_loop_var_45697))
        # SSA begins for a for statement (line 35)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # SSA begins for try-except statement (line 36)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 37):
        
        # Call to __cmp__(...): (line 37)
        # Processing the call arguments (line 37)
        
        # Call to StrictVersion(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'v2' (line 37)
        v2_45704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 62), 'v2', False)
        # Processing the call keyword arguments (line 37)
        kwargs_45705 = {}
        # Getting the type of 'StrictVersion' (line 37)
        StrictVersion_45703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 48), 'StrictVersion', False)
        # Calling StrictVersion(args, kwargs) (line 37)
        StrictVersion_call_result_45706 = invoke(stypy.reporting.localization.Localization(__file__, 37, 48), StrictVersion_45703, *[v2_45704], **kwargs_45705)
        
        # Processing the call keyword arguments (line 37)
        kwargs_45707 = {}
        
        # Call to StrictVersion(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'v1' (line 37)
        v1_45699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 36), 'v1', False)
        # Processing the call keyword arguments (line 37)
        kwargs_45700 = {}
        # Getting the type of 'StrictVersion' (line 37)
        StrictVersion_45698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 22), 'StrictVersion', False)
        # Calling StrictVersion(args, kwargs) (line 37)
        StrictVersion_call_result_45701 = invoke(stypy.reporting.localization.Localization(__file__, 37, 22), StrictVersion_45698, *[v1_45699], **kwargs_45700)
        
        # Obtaining the member '__cmp__' of a type (line 37)
        cmp___45702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 22), StrictVersion_call_result_45701, '__cmp__')
        # Calling __cmp__(args, kwargs) (line 37)
        cmp___call_result_45708 = invoke(stypy.reporting.localization.Localization(__file__, 37, 22), cmp___45702, *[StrictVersion_call_result_45706], **kwargs_45707)
        
        # Assigning a type to the variable 'res' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'res', cmp___call_result_45708)
        # SSA branch for the except part of a try statement (line 36)
        # SSA branch for the except 'ValueError' branch of a try statement (line 36)
        module_type_store.open_ssa_branch('except')
        
        
        # Getting the type of 'wanted' (line 39)
        wanted_45709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 19), 'wanted')
        # Getting the type of 'ValueError' (line 39)
        ValueError_45710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'ValueError')
        # Applying the binary operator 'is' (line 39)
        result_is__45711 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 19), 'is', wanted_45709, ValueError_45710)
        
        # Testing the type of an if condition (line 39)
        if_condition_45712 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 16), result_is__45711)
        # Assigning a type to the variable 'if_condition_45712' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'if_condition_45712', if_condition_45712)
        # SSA begins for if statement (line 39)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA branch for the else part of an if statement (line 39)
        module_type_store.open_ssa_branch('else')
        
        # Call to AssertionError(...): (line 42)
        # Processing the call arguments (line 42)
        str_45714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 42), 'str', "cmp(%s, %s) shouldn't raise ValueError")
        
        # Obtaining an instance of the builtin type 'tuple' (line 44)
        tuple_45715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 44)
        # Adding element type (line 44)
        # Getting the type of 'v1' (line 44)
        v1_45716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 47), 'v1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 47), tuple_45715, v1_45716)
        # Adding element type (line 44)
        # Getting the type of 'v2' (line 44)
        v2_45717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 51), 'v2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 47), tuple_45715, v2_45717)
        
        # Applying the binary operator '%' (line 42)
        result_mod_45718 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 41), '%', str_45714, tuple_45715)
        
        # Processing the call keyword arguments (line 42)
        kwargs_45719 = {}
        # Getting the type of 'AssertionError' (line 42)
        AssertionError_45713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 26), 'AssertionError', False)
        # Calling AssertionError(args, kwargs) (line 42)
        AssertionError_call_result_45720 = invoke(stypy.reporting.localization.Localization(__file__, 42, 26), AssertionError_45713, *[result_mod_45718], **kwargs_45719)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 42, 20), AssertionError_call_result_45720, 'raise parameter', BaseException)
        # SSA join for if statement (line 39)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for try-except statement (line 36)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assertEqual(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'res' (line 45)
        res_45723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 29), 'res', False)
        # Getting the type of 'wanted' (line 45)
        wanted_45724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 34), 'wanted', False)
        str_45725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 29), 'str', 'cmp(%s, %s) should be %s, got %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 47)
        tuple_45726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 47)
        # Adding element type (line 47)
        # Getting the type of 'v1' (line 47)
        v1_45727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 30), 'v1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 30), tuple_45726, v1_45727)
        # Adding element type (line 47)
        # Getting the type of 'v2' (line 47)
        v2_45728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 34), 'v2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 30), tuple_45726, v2_45728)
        # Adding element type (line 47)
        # Getting the type of 'wanted' (line 47)
        wanted_45729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 38), 'wanted', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 30), tuple_45726, wanted_45729)
        # Adding element type (line 47)
        # Getting the type of 'res' (line 47)
        res_45730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 46), 'res', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 30), tuple_45726, res_45730)
        
        # Applying the binary operator '%' (line 46)
        result_mod_45731 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 29), '%', str_45725, tuple_45726)
        
        # Processing the call keyword arguments (line 45)
        kwargs_45732 = {}
        # Getting the type of 'self' (line 45)
        self_45721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 45)
        assertEqual_45722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), self_45721, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 45)
        assertEqual_call_result_45733 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), assertEqual_45722, *[res_45723, wanted_45724, result_mod_45731], **kwargs_45732)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_cmp_strict(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cmp_strict' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_45734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_45734)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cmp_strict'
        return stypy_return_type_45734


    @norecursion
    def test_cmp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cmp'
        module_type_store = module_type_store.open_function_context('test_cmp', 50, 4, False)
        # Assigning a type to the variable 'self' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VersionTestCase.test_cmp.__dict__.__setitem__('stypy_localization', localization)
        VersionTestCase.test_cmp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VersionTestCase.test_cmp.__dict__.__setitem__('stypy_type_store', module_type_store)
        VersionTestCase.test_cmp.__dict__.__setitem__('stypy_function_name', 'VersionTestCase.test_cmp')
        VersionTestCase.test_cmp.__dict__.__setitem__('stypy_param_names_list', [])
        VersionTestCase.test_cmp.__dict__.__setitem__('stypy_varargs_param_name', None)
        VersionTestCase.test_cmp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VersionTestCase.test_cmp.__dict__.__setitem__('stypy_call_defaults', defaults)
        VersionTestCase.test_cmp.__dict__.__setitem__('stypy_call_varargs', varargs)
        VersionTestCase.test_cmp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VersionTestCase.test_cmp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VersionTestCase.test_cmp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cmp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cmp(...)' code ##################

        
        # Assigning a Tuple to a Name (line 51):
        
        # Obtaining an instance of the builtin type 'tuple' (line 51)
        tuple_45735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 51)
        # Adding element type (line 51)
        
        # Obtaining an instance of the builtin type 'tuple' (line 51)
        tuple_45736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 51)
        # Adding element type (line 51)
        str_45737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 21), 'str', '1.5.1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 21), tuple_45736, str_45737)
        # Adding element type (line 51)
        str_45738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 30), 'str', '1.5.2b2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 21), tuple_45736, str_45738)
        # Adding element type (line 51)
        int_45739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 21), tuple_45736, int_45739)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 20), tuple_45735, tuple_45736)
        # Adding element type (line 51)
        
        # Obtaining an instance of the builtin type 'tuple' (line 52)
        tuple_45740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 52)
        # Adding element type (line 52)
        str_45741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 21), 'str', '161')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 21), tuple_45740, str_45741)
        # Adding element type (line 52)
        str_45742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 28), 'str', '3.10a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 21), tuple_45740, str_45742)
        # Adding element type (line 52)
        int_45743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 21), tuple_45740, int_45743)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 20), tuple_45735, tuple_45740)
        # Adding element type (line 51)
        
        # Obtaining an instance of the builtin type 'tuple' (line 53)
        tuple_45744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 53)
        # Adding element type (line 53)
        str_45745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 21), 'str', '8.02')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 21), tuple_45744, str_45745)
        # Adding element type (line 53)
        str_45746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 29), 'str', '8.02')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 21), tuple_45744, str_45746)
        # Adding element type (line 53)
        int_45747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 21), tuple_45744, int_45747)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 20), tuple_45735, tuple_45744)
        # Adding element type (line 51)
        
        # Obtaining an instance of the builtin type 'tuple' (line 54)
        tuple_45748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 54)
        # Adding element type (line 54)
        str_45749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 21), 'str', '3.4j')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 21), tuple_45748, str_45749)
        # Adding element type (line 54)
        str_45750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 29), 'str', '1996.07.12')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 21), tuple_45748, str_45750)
        # Adding element type (line 54)
        int_45751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 21), tuple_45748, int_45751)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 20), tuple_45735, tuple_45748)
        # Adding element type (line 51)
        
        # Obtaining an instance of the builtin type 'tuple' (line 55)
        tuple_45752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 55)
        # Adding element type (line 55)
        str_45753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 21), 'str', '3.2.pl0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 21), tuple_45752, str_45753)
        # Adding element type (line 55)
        str_45754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 32), 'str', '3.1.1.6')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 21), tuple_45752, str_45754)
        # Adding element type (line 55)
        int_45755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 21), tuple_45752, int_45755)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 20), tuple_45735, tuple_45752)
        # Adding element type (line 51)
        
        # Obtaining an instance of the builtin type 'tuple' (line 56)
        tuple_45756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 56)
        # Adding element type (line 56)
        str_45757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 21), 'str', '2g6')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 21), tuple_45756, str_45757)
        # Adding element type (line 56)
        str_45758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 28), 'str', '11g')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 21), tuple_45756, str_45758)
        # Adding element type (line 56)
        int_45759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 21), tuple_45756, int_45759)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 20), tuple_45735, tuple_45756)
        # Adding element type (line 51)
        
        # Obtaining an instance of the builtin type 'tuple' (line 57)
        tuple_45760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 57)
        # Adding element type (line 57)
        str_45761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 21), 'str', '0.960923')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 21), tuple_45760, str_45761)
        # Adding element type (line 57)
        str_45762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 33), 'str', '2.2beta29')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 21), tuple_45760, str_45762)
        # Adding element type (line 57)
        int_45763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 21), tuple_45760, int_45763)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 20), tuple_45735, tuple_45760)
        # Adding element type (line 51)
        
        # Obtaining an instance of the builtin type 'tuple' (line 58)
        tuple_45764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 58)
        # Adding element type (line 58)
        str_45765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 21), 'str', '1.13++')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 21), tuple_45764, str_45765)
        # Adding element type (line 58)
        str_45766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 31), 'str', '5.5.kw')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 21), tuple_45764, str_45766)
        # Adding element type (line 58)
        int_45767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 21), tuple_45764, int_45767)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 20), tuple_45735, tuple_45764)
        
        # Assigning a type to the variable 'versions' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'versions', tuple_45735)
        
        # Getting the type of 'versions' (line 61)
        versions_45768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'versions')
        # Testing the type of a for loop iterable (line 61)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 61, 8), versions_45768)
        # Getting the type of the for loop variable (line 61)
        for_loop_var_45769 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 61, 8), versions_45768)
        # Assigning a type to the variable 'v1' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'v1', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 8), for_loop_var_45769))
        # Assigning a type to the variable 'v2' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'v2', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 8), for_loop_var_45769))
        # Assigning a type to the variable 'wanted' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'wanted', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 8), for_loop_var_45769))
        # SSA begins for a for statement (line 61)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 62):
        
        # Call to __cmp__(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Call to LooseVersion(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'v2' (line 62)
        v2_45776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 56), 'v2', False)
        # Processing the call keyword arguments (line 62)
        kwargs_45777 = {}
        # Getting the type of 'LooseVersion' (line 62)
        LooseVersion_45775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 43), 'LooseVersion', False)
        # Calling LooseVersion(args, kwargs) (line 62)
        LooseVersion_call_result_45778 = invoke(stypy.reporting.localization.Localization(__file__, 62, 43), LooseVersion_45775, *[v2_45776], **kwargs_45777)
        
        # Processing the call keyword arguments (line 62)
        kwargs_45779 = {}
        
        # Call to LooseVersion(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'v1' (line 62)
        v1_45771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 31), 'v1', False)
        # Processing the call keyword arguments (line 62)
        kwargs_45772 = {}
        # Getting the type of 'LooseVersion' (line 62)
        LooseVersion_45770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 18), 'LooseVersion', False)
        # Calling LooseVersion(args, kwargs) (line 62)
        LooseVersion_call_result_45773 = invoke(stypy.reporting.localization.Localization(__file__, 62, 18), LooseVersion_45770, *[v1_45771], **kwargs_45772)
        
        # Obtaining the member '__cmp__' of a type (line 62)
        cmp___45774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 18), LooseVersion_call_result_45773, '__cmp__')
        # Calling __cmp__(args, kwargs) (line 62)
        cmp___call_result_45780 = invoke(stypy.reporting.localization.Localization(__file__, 62, 18), cmp___45774, *[LooseVersion_call_result_45778], **kwargs_45779)
        
        # Assigning a type to the variable 'res' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'res', cmp___call_result_45780)
        
        # Call to assertEqual(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'res' (line 63)
        res_45783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 29), 'res', False)
        # Getting the type of 'wanted' (line 63)
        wanted_45784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 34), 'wanted', False)
        str_45785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 29), 'str', 'cmp(%s, %s) should be %s, got %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 65)
        tuple_45786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 65)
        # Adding element type (line 65)
        # Getting the type of 'v1' (line 65)
        v1_45787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'v1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 30), tuple_45786, v1_45787)
        # Adding element type (line 65)
        # Getting the type of 'v2' (line 65)
        v2_45788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 34), 'v2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 30), tuple_45786, v2_45788)
        # Adding element type (line 65)
        # Getting the type of 'wanted' (line 65)
        wanted_45789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 38), 'wanted', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 30), tuple_45786, wanted_45789)
        # Adding element type (line 65)
        # Getting the type of 'res' (line 65)
        res_45790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 46), 'res', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 30), tuple_45786, res_45790)
        
        # Applying the binary operator '%' (line 64)
        result_mod_45791 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 29), '%', str_45785, tuple_45786)
        
        # Processing the call keyword arguments (line 63)
        kwargs_45792 = {}
        # Getting the type of 'self' (line 63)
        self_45781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 63)
        assertEqual_45782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), self_45781, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 63)
        assertEqual_call_result_45793 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), assertEqual_45782, *[res_45783, wanted_45784, result_mod_45791], **kwargs_45792)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_cmp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cmp' in the type store
        # Getting the type of 'stypy_return_type' (line 50)
        stypy_return_type_45794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_45794)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cmp'
        return stypy_return_type_45794


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 7, 0, False)
        # Assigning a type to the variable 'self' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VersionTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'VersionTestCase' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'VersionTestCase', VersionTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 67, 0, False)
    
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

    
    # Call to makeSuite(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'VersionTestCase' (line 68)
    VersionTestCase_45797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 30), 'VersionTestCase', False)
    # Processing the call keyword arguments (line 68)
    kwargs_45798 = {}
    # Getting the type of 'unittest' (line 68)
    unittest_45795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 68)
    makeSuite_45796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 11), unittest_45795, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 68)
    makeSuite_call_result_45799 = invoke(stypy.reporting.localization.Localization(__file__, 68, 11), makeSuite_45796, *[VersionTestCase_45797], **kwargs_45798)
    
    # Assigning a type to the variable 'stypy_return_type' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type', makeSuite_call_result_45799)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 67)
    stypy_return_type_45800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_45800)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_45800

# Assigning a type to the variable 'test_suite' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 71)
    # Processing the call arguments (line 71)
    
    # Call to test_suite(...): (line 71)
    # Processing the call keyword arguments (line 71)
    kwargs_45803 = {}
    # Getting the type of 'test_suite' (line 71)
    test_suite_45802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 71)
    test_suite_call_result_45804 = invoke(stypy.reporting.localization.Localization(__file__, 71, 17), test_suite_45802, *[], **kwargs_45803)
    
    # Processing the call keyword arguments (line 71)
    kwargs_45805 = {}
    # Getting the type of 'run_unittest' (line 71)
    run_unittest_45801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 71)
    run_unittest_call_result_45806 = invoke(stypy.reporting.localization.Localization(__file__, 71, 4), run_unittest_45801, *[test_suite_call_result_45804], **kwargs_45805)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
