
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.util.'''
2: import sys
3: import unittest
4: from test.test_support import run_unittest
5: 
6: from distutils.errors import DistutilsByteCompileError
7: from distutils.util import byte_compile, grok_environment_error
8: 
9: 
10: class UtilTestCase(unittest.TestCase):
11: 
12:     def test_dont_write_bytecode(self):
13:         # makes sure byte_compile raise a DistutilsError
14:         # if sys.dont_write_bytecode is True
15:         old_dont_write_bytecode = sys.dont_write_bytecode
16:         sys.dont_write_bytecode = True
17:         try:
18:             self.assertRaises(DistutilsByteCompileError, byte_compile, [])
19:         finally:
20:             sys.dont_write_bytecode = old_dont_write_bytecode
21: 
22:     def test_grok_environment_error(self):
23:         # test obsolete function to ensure backward compat (#4931)
24:         exc = IOError("Unable to find batch file")
25:         msg = grok_environment_error(exc)
26:         self.assertEqual(msg, "error: Unable to find batch file")
27: 
28: 
29: def test_suite():
30:     return unittest.makeSuite(UtilTestCase)
31: 
32: if __name__ == "__main__":
33:     run_unittest(test_suite())
34: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_45530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.util.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import sys' statement (line 2)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import unittest' statement (line 3)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from test.test_support import run_unittest' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_45531 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'test.test_support')

if (type(import_45531) is not StypyTypeError):

    if (import_45531 != 'pyd_module'):
        __import__(import_45531)
        sys_modules_45532 = sys.modules[import_45531]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'test.test_support', sys_modules_45532.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_45532, sys_modules_45532.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'test.test_support', import_45531)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from distutils.errors import DistutilsByteCompileError' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_45533 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.errors')

if (type(import_45533) is not StypyTypeError):

    if (import_45533 != 'pyd_module'):
        __import__(import_45533)
        sys_modules_45534 = sys.modules[import_45533]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.errors', sys_modules_45534.module_type_store, module_type_store, ['DistutilsByteCompileError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_45534, sys_modules_45534.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsByteCompileError

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.errors', None, module_type_store, ['DistutilsByteCompileError'], [DistutilsByteCompileError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.errors', import_45533)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.util import byte_compile, grok_environment_error' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_45535 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.util')

if (type(import_45535) is not StypyTypeError):

    if (import_45535 != 'pyd_module'):
        __import__(import_45535)
        sys_modules_45536 = sys.modules[import_45535]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.util', sys_modules_45536.module_type_store, module_type_store, ['byte_compile', 'grok_environment_error'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_45536, sys_modules_45536.module_type_store, module_type_store)
    else:
        from distutils.util import byte_compile, grok_environment_error

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.util', None, module_type_store, ['byte_compile', 'grok_environment_error'], [byte_compile, grok_environment_error])

else:
    # Assigning a type to the variable 'distutils.util' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.util', import_45535)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# Declaration of the 'UtilTestCase' class
# Getting the type of 'unittest' (line 10)
unittest_45537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 19), 'unittest')
# Obtaining the member 'TestCase' of a type (line 10)
TestCase_45538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 19), unittest_45537, 'TestCase')

class UtilTestCase(TestCase_45538, ):

    @norecursion
    def test_dont_write_bytecode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dont_write_bytecode'
        module_type_store = module_type_store.open_function_context('test_dont_write_bytecode', 12, 4, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UtilTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_localization', localization)
        UtilTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UtilTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_type_store', module_type_store)
        UtilTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_function_name', 'UtilTestCase.test_dont_write_bytecode')
        UtilTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_param_names_list', [])
        UtilTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_varargs_param_name', None)
        UtilTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UtilTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_call_defaults', defaults)
        UtilTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_call_varargs', varargs)
        UtilTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UtilTestCase.test_dont_write_bytecode.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UtilTestCase.test_dont_write_bytecode', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dont_write_bytecode', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dont_write_bytecode(...)' code ##################

        
        # Assigning a Attribute to a Name (line 15):
        # Getting the type of 'sys' (line 15)
        sys_45539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 34), 'sys')
        # Obtaining the member 'dont_write_bytecode' of a type (line 15)
        dont_write_bytecode_45540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 34), sys_45539, 'dont_write_bytecode')
        # Assigning a type to the variable 'old_dont_write_bytecode' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'old_dont_write_bytecode', dont_write_bytecode_45540)
        
        # Assigning a Name to a Attribute (line 16):
        # Getting the type of 'True' (line 16)
        True_45541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 34), 'True')
        # Getting the type of 'sys' (line 16)
        sys_45542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'sys')
        # Setting the type of the member 'dont_write_bytecode' of a type (line 16)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), sys_45542, 'dont_write_bytecode', True_45541)
        
        # Try-finally block (line 17)
        
        # Call to assertRaises(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'DistutilsByteCompileError' (line 18)
        DistutilsByteCompileError_45545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 30), 'DistutilsByteCompileError', False)
        # Getting the type of 'byte_compile' (line 18)
        byte_compile_45546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 57), 'byte_compile', False)
        
        # Obtaining an instance of the builtin type 'list' (line 18)
        list_45547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 71), 'list')
        # Adding type elements to the builtin type 'list' instance (line 18)
        
        # Processing the call keyword arguments (line 18)
        kwargs_45548 = {}
        # Getting the type of 'self' (line 18)
        self_45543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 18)
        assertRaises_45544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 12), self_45543, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 18)
        assertRaises_call_result_45549 = invoke(stypy.reporting.localization.Localization(__file__, 18, 12), assertRaises_45544, *[DistutilsByteCompileError_45545, byte_compile_45546, list_45547], **kwargs_45548)
        
        
        # finally branch of the try-finally block (line 17)
        
        # Assigning a Name to a Attribute (line 20):
        # Getting the type of 'old_dont_write_bytecode' (line 20)
        old_dont_write_bytecode_45550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 38), 'old_dont_write_bytecode')
        # Getting the type of 'sys' (line 20)
        sys_45551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'sys')
        # Setting the type of the member 'dont_write_bytecode' of a type (line 20)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 12), sys_45551, 'dont_write_bytecode', old_dont_write_bytecode_45550)
        
        
        # ################# End of 'test_dont_write_bytecode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dont_write_bytecode' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_45552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_45552)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dont_write_bytecode'
        return stypy_return_type_45552


    @norecursion
    def test_grok_environment_error(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_grok_environment_error'
        module_type_store = module_type_store.open_function_context('test_grok_environment_error', 22, 4, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UtilTestCase.test_grok_environment_error.__dict__.__setitem__('stypy_localization', localization)
        UtilTestCase.test_grok_environment_error.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UtilTestCase.test_grok_environment_error.__dict__.__setitem__('stypy_type_store', module_type_store)
        UtilTestCase.test_grok_environment_error.__dict__.__setitem__('stypy_function_name', 'UtilTestCase.test_grok_environment_error')
        UtilTestCase.test_grok_environment_error.__dict__.__setitem__('stypy_param_names_list', [])
        UtilTestCase.test_grok_environment_error.__dict__.__setitem__('stypy_varargs_param_name', None)
        UtilTestCase.test_grok_environment_error.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UtilTestCase.test_grok_environment_error.__dict__.__setitem__('stypy_call_defaults', defaults)
        UtilTestCase.test_grok_environment_error.__dict__.__setitem__('stypy_call_varargs', varargs)
        UtilTestCase.test_grok_environment_error.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UtilTestCase.test_grok_environment_error.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UtilTestCase.test_grok_environment_error', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_grok_environment_error', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_grok_environment_error(...)' code ##################

        
        # Assigning a Call to a Name (line 24):
        
        # Call to IOError(...): (line 24)
        # Processing the call arguments (line 24)
        str_45554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 22), 'str', 'Unable to find batch file')
        # Processing the call keyword arguments (line 24)
        kwargs_45555 = {}
        # Getting the type of 'IOError' (line 24)
        IOError_45553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 14), 'IOError', False)
        # Calling IOError(args, kwargs) (line 24)
        IOError_call_result_45556 = invoke(stypy.reporting.localization.Localization(__file__, 24, 14), IOError_45553, *[str_45554], **kwargs_45555)
        
        # Assigning a type to the variable 'exc' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'exc', IOError_call_result_45556)
        
        # Assigning a Call to a Name (line 25):
        
        # Call to grok_environment_error(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'exc' (line 25)
        exc_45558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 37), 'exc', False)
        # Processing the call keyword arguments (line 25)
        kwargs_45559 = {}
        # Getting the type of 'grok_environment_error' (line 25)
        grok_environment_error_45557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 14), 'grok_environment_error', False)
        # Calling grok_environment_error(args, kwargs) (line 25)
        grok_environment_error_call_result_45560 = invoke(stypy.reporting.localization.Localization(__file__, 25, 14), grok_environment_error_45557, *[exc_45558], **kwargs_45559)
        
        # Assigning a type to the variable 'msg' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'msg', grok_environment_error_call_result_45560)
        
        # Call to assertEqual(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'msg' (line 26)
        msg_45563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 25), 'msg', False)
        str_45564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 30), 'str', 'error: Unable to find batch file')
        # Processing the call keyword arguments (line 26)
        kwargs_45565 = {}
        # Getting the type of 'self' (line 26)
        self_45561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 26)
        assertEqual_45562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_45561, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 26)
        assertEqual_call_result_45566 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), assertEqual_45562, *[msg_45563, str_45564], **kwargs_45565)
        
        
        # ################# End of 'test_grok_environment_error(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_grok_environment_error' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_45567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_45567)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_grok_environment_error'
        return stypy_return_type_45567


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 10, 0, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UtilTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'UtilTestCase' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'UtilTestCase', UtilTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 29, 0, False)
    
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

    
    # Call to makeSuite(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'UtilTestCase' (line 30)
    UtilTestCase_45570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 30), 'UtilTestCase', False)
    # Processing the call keyword arguments (line 30)
    kwargs_45571 = {}
    # Getting the type of 'unittest' (line 30)
    unittest_45568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 30)
    makeSuite_45569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 11), unittest_45568, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 30)
    makeSuite_call_result_45572 = invoke(stypy.reporting.localization.Localization(__file__, 30, 11), makeSuite_45569, *[UtilTestCase_45570], **kwargs_45571)
    
    # Assigning a type to the variable 'stypy_return_type' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type', makeSuite_call_result_45572)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_45573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_45573)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_45573

# Assigning a type to the variable 'test_suite' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 33)
    # Processing the call arguments (line 33)
    
    # Call to test_suite(...): (line 33)
    # Processing the call keyword arguments (line 33)
    kwargs_45576 = {}
    # Getting the type of 'test_suite' (line 33)
    test_suite_45575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 33)
    test_suite_call_result_45577 = invoke(stypy.reporting.localization.Localization(__file__, 33, 17), test_suite_45575, *[], **kwargs_45576)
    
    # Processing the call keyword arguments (line 33)
    kwargs_45578 = {}
    # Getting the type of 'run_unittest' (line 33)
    run_unittest_45574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 33)
    run_unittest_call_result_45579 = invoke(stypy.reporting.localization.Localization(__file__, 33, 4), run_unittest_45574, *[test_suite_call_result_45577], **kwargs_45578)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
