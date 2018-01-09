
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Test suite for distutils.
2: 
3: This test suite consists of a collection of test modules in the
4: distutils.tests package.  Each test module has a name starting with
5: 'test' and contains a function test_suite().  The function is expected
6: to return an initialized unittest.TestSuite instance.
7: 
8: Tests for the command classes in the distutils.command package are
9: included in distutils.tests as well, instead of using a separate
10: distutils.command.tests package, since command identification is done
11: by import rather than matching pre-defined names.
12: 
13: '''
14: 
15: import os
16: import sys
17: import unittest
18: from test.test_support import run_unittest
19: 
20: 
21: here = os.path.dirname(__file__) or os.curdir
22: 
23: 
24: def test_suite():
25:     suite = unittest.TestSuite()
26:     for fn in os.listdir(here):
27:         if fn.startswith("test") and fn.endswith(".py"):
28:             modname = "distutils.tests." + fn[:-3]
29:             __import__(modname)
30:             module = sys.modules[modname]
31:             suite.addTest(module.test_suite())
32:     return suite
33: 
34: 
35: if __name__ == "__main__":
36:     run_unittest(test_suite())
37: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_45825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, (-1)), 'str', "Test suite for distutils.\n\nThis test suite consists of a collection of test modules in the\ndistutils.tests package.  Each test module has a name starting with\n'test' and contains a function test_suite().  The function is expected\nto return an initialized unittest.TestSuite instance.\n\nTests for the command classes in the distutils.command package are\nincluded in distutils.tests as well, instead of using a separate\ndistutils.command.tests package, since command identification is done\nby import rather than matching pre-defined names.\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import os' statement (line 15)
import os

import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import sys' statement (line 16)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import unittest' statement (line 17)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from test.test_support import run_unittest' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_45826 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'test.test_support')

if (type(import_45826) is not StypyTypeError):

    if (import_45826 != 'pyd_module'):
        __import__(import_45826)
        sys_modules_45827 = sys.modules[import_45826]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'test.test_support', sys_modules_45827.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_45827, sys_modules_45827.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'test.test_support', import_45826)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')


# Assigning a BoolOp to a Name (line 21):

# Evaluating a boolean operation

# Call to dirname(...): (line 21)
# Processing the call arguments (line 21)
# Getting the type of '__file__' (line 21)
file___45831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 23), '__file__', False)
# Processing the call keyword arguments (line 21)
kwargs_45832 = {}
# Getting the type of 'os' (line 21)
os_45828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 7), 'os', False)
# Obtaining the member 'path' of a type (line 21)
path_45829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 7), os_45828, 'path')
# Obtaining the member 'dirname' of a type (line 21)
dirname_45830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 7), path_45829, 'dirname')
# Calling dirname(args, kwargs) (line 21)
dirname_call_result_45833 = invoke(stypy.reporting.localization.Localization(__file__, 21, 7), dirname_45830, *[file___45831], **kwargs_45832)

# Getting the type of 'os' (line 21)
os_45834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 36), 'os')
# Obtaining the member 'curdir' of a type (line 21)
curdir_45835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 36), os_45834, 'curdir')
# Applying the binary operator 'or' (line 21)
result_or_keyword_45836 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 7), 'or', dirname_call_result_45833, curdir_45835)

# Assigning a type to the variable 'here' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'here', result_or_keyword_45836)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 24, 0, False)
    
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

    
    # Assigning a Call to a Name (line 25):
    
    # Call to TestSuite(...): (line 25)
    # Processing the call keyword arguments (line 25)
    kwargs_45839 = {}
    # Getting the type of 'unittest' (line 25)
    unittest_45837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'unittest', False)
    # Obtaining the member 'TestSuite' of a type (line 25)
    TestSuite_45838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 12), unittest_45837, 'TestSuite')
    # Calling TestSuite(args, kwargs) (line 25)
    TestSuite_call_result_45840 = invoke(stypy.reporting.localization.Localization(__file__, 25, 12), TestSuite_45838, *[], **kwargs_45839)
    
    # Assigning a type to the variable 'suite' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'suite', TestSuite_call_result_45840)
    
    
    # Call to listdir(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'here' (line 26)
    here_45843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 25), 'here', False)
    # Processing the call keyword arguments (line 26)
    kwargs_45844 = {}
    # Getting the type of 'os' (line 26)
    os_45841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'os', False)
    # Obtaining the member 'listdir' of a type (line 26)
    listdir_45842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 14), os_45841, 'listdir')
    # Calling listdir(args, kwargs) (line 26)
    listdir_call_result_45845 = invoke(stypy.reporting.localization.Localization(__file__, 26, 14), listdir_45842, *[here_45843], **kwargs_45844)
    
    # Testing the type of a for loop iterable (line 26)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 26, 4), listdir_call_result_45845)
    # Getting the type of the for loop variable (line 26)
    for_loop_var_45846 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 26, 4), listdir_call_result_45845)
    # Assigning a type to the variable 'fn' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'fn', for_loop_var_45846)
    # SSA begins for a for statement (line 26)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Call to startswith(...): (line 27)
    # Processing the call arguments (line 27)
    str_45849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 25), 'str', 'test')
    # Processing the call keyword arguments (line 27)
    kwargs_45850 = {}
    # Getting the type of 'fn' (line 27)
    fn_45847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'fn', False)
    # Obtaining the member 'startswith' of a type (line 27)
    startswith_45848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 11), fn_45847, 'startswith')
    # Calling startswith(args, kwargs) (line 27)
    startswith_call_result_45851 = invoke(stypy.reporting.localization.Localization(__file__, 27, 11), startswith_45848, *[str_45849], **kwargs_45850)
    
    
    # Call to endswith(...): (line 27)
    # Processing the call arguments (line 27)
    str_45854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 49), 'str', '.py')
    # Processing the call keyword arguments (line 27)
    kwargs_45855 = {}
    # Getting the type of 'fn' (line 27)
    fn_45852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 37), 'fn', False)
    # Obtaining the member 'endswith' of a type (line 27)
    endswith_45853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 37), fn_45852, 'endswith')
    # Calling endswith(args, kwargs) (line 27)
    endswith_call_result_45856 = invoke(stypy.reporting.localization.Localization(__file__, 27, 37), endswith_45853, *[str_45854], **kwargs_45855)
    
    # Applying the binary operator 'and' (line 27)
    result_and_keyword_45857 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 11), 'and', startswith_call_result_45851, endswith_call_result_45856)
    
    # Testing the type of an if condition (line 27)
    if_condition_45858 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 8), result_and_keyword_45857)
    # Assigning a type to the variable 'if_condition_45858' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'if_condition_45858', if_condition_45858)
    # SSA begins for if statement (line 27)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 28):
    str_45859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 22), 'str', 'distutils.tests.')
    
    # Obtaining the type of the subscript
    int_45860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 47), 'int')
    slice_45861 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 28, 43), None, int_45860, None)
    # Getting the type of 'fn' (line 28)
    fn_45862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 43), 'fn')
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___45863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 43), fn_45862, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_45864 = invoke(stypy.reporting.localization.Localization(__file__, 28, 43), getitem___45863, slice_45861)
    
    # Applying the binary operator '+' (line 28)
    result_add_45865 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 22), '+', str_45859, subscript_call_result_45864)
    
    # Assigning a type to the variable 'modname' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'modname', result_add_45865)
    
    # Call to __import__(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'modname' (line 29)
    modname_45867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'modname', False)
    # Processing the call keyword arguments (line 29)
    kwargs_45868 = {}
    # Getting the type of '__import__' (line 29)
    import___45866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), '__import__', False)
    # Calling __import__(args, kwargs) (line 29)
    import___call_result_45869 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), import___45866, *[modname_45867], **kwargs_45868)
    
    
    # Assigning a Subscript to a Name (line 30):
    
    # Obtaining the type of the subscript
    # Getting the type of 'modname' (line 30)
    modname_45870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 33), 'modname')
    # Getting the type of 'sys' (line 30)
    sys_45871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'sys')
    # Obtaining the member 'modules' of a type (line 30)
    modules_45872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 21), sys_45871, 'modules')
    # Obtaining the member '__getitem__' of a type (line 30)
    getitem___45873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 21), modules_45872, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 30)
    subscript_call_result_45874 = invoke(stypy.reporting.localization.Localization(__file__, 30, 21), getitem___45873, modname_45870)
    
    # Assigning a type to the variable 'module' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'module', subscript_call_result_45874)
    
    # Call to addTest(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Call to test_suite(...): (line 31)
    # Processing the call keyword arguments (line 31)
    kwargs_45879 = {}
    # Getting the type of 'module' (line 31)
    module_45877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 26), 'module', False)
    # Obtaining the member 'test_suite' of a type (line 31)
    test_suite_45878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 26), module_45877, 'test_suite')
    # Calling test_suite(args, kwargs) (line 31)
    test_suite_call_result_45880 = invoke(stypy.reporting.localization.Localization(__file__, 31, 26), test_suite_45878, *[], **kwargs_45879)
    
    # Processing the call keyword arguments (line 31)
    kwargs_45881 = {}
    # Getting the type of 'suite' (line 31)
    suite_45875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'suite', False)
    # Obtaining the member 'addTest' of a type (line 31)
    addTest_45876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), suite_45875, 'addTest')
    # Calling addTest(args, kwargs) (line 31)
    addTest_call_result_45882 = invoke(stypy.reporting.localization.Localization(__file__, 31, 12), addTest_45876, *[test_suite_call_result_45880], **kwargs_45881)
    
    # SSA join for if statement (line 27)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'suite' (line 32)
    suite_45883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'suite')
    # Assigning a type to the variable 'stypy_return_type' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type', suite_45883)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_45884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_45884)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_45884

# Assigning a type to the variable 'test_suite' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Call to test_suite(...): (line 36)
    # Processing the call keyword arguments (line 36)
    kwargs_45887 = {}
    # Getting the type of 'test_suite' (line 36)
    test_suite_45886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 36)
    test_suite_call_result_45888 = invoke(stypy.reporting.localization.Localization(__file__, 36, 17), test_suite_45886, *[], **kwargs_45887)
    
    # Processing the call keyword arguments (line 36)
    kwargs_45889 = {}
    # Getting the type of 'run_unittest' (line 36)
    run_unittest_45885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 36)
    run_unittest_call_result_45890 = invoke(stypy.reporting.localization.Localization(__file__, 36, 4), run_unittest_45885, *[test_suite_call_result_45888], **kwargs_45889)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
