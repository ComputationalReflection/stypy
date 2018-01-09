
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests harness for distutils.versionpredicate.
2: 
3: '''
4: 
5: import distutils.versionpredicate
6: import doctest
7: from test.test_support import run_unittest
8: 
9: def test_suite():
10:     return doctest.DocTestSuite(distutils.versionpredicate)
11: 
12: if __name__ == '__main__':
13:     run_unittest(test_suite())
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_45807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', 'Tests harness for distutils.versionpredicate.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import distutils.versionpredicate' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_45808 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.versionpredicate')

if (type(import_45808) is not StypyTypeError):

    if (import_45808 != 'pyd_module'):
        __import__(import_45808)
        sys_modules_45809 = sys.modules[import_45808]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.versionpredicate', sys_modules_45809.module_type_store, module_type_store)
    else:
        import distutils.versionpredicate

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.versionpredicate', distutils.versionpredicate, module_type_store)

else:
    # Assigning a type to the variable 'distutils.versionpredicate' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.versionpredicate', import_45808)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import doctest' statement (line 6)
import doctest

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'doctest', doctest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from test.test_support import run_unittest' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_45810 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'test.test_support')

if (type(import_45810) is not StypyTypeError):

    if (import_45810 != 'pyd_module'):
        __import__(import_45810)
        sys_modules_45811 = sys.modules[import_45810]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'test.test_support', sys_modules_45811.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_45811, sys_modules_45811.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'test.test_support', import_45810)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')


@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 9, 0, False)
    
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

    
    # Call to DocTestSuite(...): (line 10)
    # Processing the call arguments (line 10)
    # Getting the type of 'distutils' (line 10)
    distutils_45814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 32), 'distutils', False)
    # Obtaining the member 'versionpredicate' of a type (line 10)
    versionpredicate_45815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 32), distutils_45814, 'versionpredicate')
    # Processing the call keyword arguments (line 10)
    kwargs_45816 = {}
    # Getting the type of 'doctest' (line 10)
    doctest_45812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 11), 'doctest', False)
    # Obtaining the member 'DocTestSuite' of a type (line 10)
    DocTestSuite_45813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 11), doctest_45812, 'DocTestSuite')
    # Calling DocTestSuite(args, kwargs) (line 10)
    DocTestSuite_call_result_45817 = invoke(stypy.reporting.localization.Localization(__file__, 10, 11), DocTestSuite_45813, *[versionpredicate_45815], **kwargs_45816)
    
    # Assigning a type to the variable 'stypy_return_type' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'stypy_return_type', DocTestSuite_call_result_45817)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_45818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_45818)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_45818

# Assigning a type to the variable 'test_suite' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 13)
    # Processing the call arguments (line 13)
    
    # Call to test_suite(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_45821 = {}
    # Getting the type of 'test_suite' (line 13)
    test_suite_45820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 13)
    test_suite_call_result_45822 = invoke(stypy.reporting.localization.Localization(__file__, 13, 17), test_suite_45820, *[], **kwargs_45821)
    
    # Processing the call keyword arguments (line 13)
    kwargs_45823 = {}
    # Getting the type of 'run_unittest' (line 13)
    run_unittest_45819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 13)
    run_unittest_call_result_45824 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), run_unittest_45819, *[test_suite_call_result_45822], **kwargs_45823)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
