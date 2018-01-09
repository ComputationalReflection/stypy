
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command
2: 
3: Package containing implementation of all the standard Distutils
4: commands.
5: 
6: '''
7: from __future__ import division, absolute_import, print_function
8: 
9: def test_na_writable_attributes_deletion():
10:     a = np.NA(2)
11:     attr =  ['payload', 'dtype']
12:     for s in attr:
13:         assert_raises(AttributeError, delattr, a, s)
14: 
15: 
16: __revision__ = "$Id: __init__.py,v 1.3 2005/05/16 11:08:49 pearu Exp $"
17: 
18: distutils_all = [  #'build_py',
19:                    'clean',
20:                    'install_clib',
21:                    'install_scripts',
22:                    'bdist',
23:                    'bdist_dumb',
24:                    'bdist_wininst',
25:                 ]
26: 
27: __import__('distutils.command', globals(), locals(), distutils_all)
28: 
29: __all__ = ['build',
30:            'config_compiler',
31:            'config',
32:            'build_src',
33:            'build_py',
34:            'build_ext',
35:            'build_clib',
36:            'build_scripts',
37:            'install',
38:            'install_data',
39:            'install_headers',
40:            'install_lib',
41:            'bdist_rpm',
42:            'sdist',
43:           ] + distutils_all
44: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_59791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, (-1)), 'str', 'distutils.command\n\nPackage containing implementation of all the standard Distutils\ncommands.\n\n')

@norecursion
def test_na_writable_attributes_deletion(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_na_writable_attributes_deletion'
    module_type_store = module_type_store.open_function_context('test_na_writable_attributes_deletion', 9, 0, False)
    
    # Passed parameters checking function
    test_na_writable_attributes_deletion.stypy_localization = localization
    test_na_writable_attributes_deletion.stypy_type_of_self = None
    test_na_writable_attributes_deletion.stypy_type_store = module_type_store
    test_na_writable_attributes_deletion.stypy_function_name = 'test_na_writable_attributes_deletion'
    test_na_writable_attributes_deletion.stypy_param_names_list = []
    test_na_writable_attributes_deletion.stypy_varargs_param_name = None
    test_na_writable_attributes_deletion.stypy_kwargs_param_name = None
    test_na_writable_attributes_deletion.stypy_call_defaults = defaults
    test_na_writable_attributes_deletion.stypy_call_varargs = varargs
    test_na_writable_attributes_deletion.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_na_writable_attributes_deletion', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_na_writable_attributes_deletion', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_na_writable_attributes_deletion(...)' code ##################

    
    # Assigning a Call to a Name (line 10):
    
    # Call to NA(...): (line 10)
    # Processing the call arguments (line 10)
    int_59794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'int')
    # Processing the call keyword arguments (line 10)
    kwargs_59795 = {}
    # Getting the type of 'np' (line 10)
    np_59792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'np', False)
    # Obtaining the member 'NA' of a type (line 10)
    NA_59793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 8), np_59792, 'NA')
    # Calling NA(args, kwargs) (line 10)
    NA_call_result_59796 = invoke(stypy.reporting.localization.Localization(__file__, 10, 8), NA_59793, *[int_59794], **kwargs_59795)
    
    # Assigning a type to the variable 'a' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'a', NA_call_result_59796)
    
    # Assigning a List to a Name (line 11):
    
    # Obtaining an instance of the builtin type 'list' (line 11)
    list_59797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 11)
    # Adding element type (line 11)
    str_59798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 13), 'str', 'payload')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 12), list_59797, str_59798)
    # Adding element type (line 11)
    str_59799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 24), 'str', 'dtype')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 12), list_59797, str_59799)
    
    # Assigning a type to the variable 'attr' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'attr', list_59797)
    
    # Getting the type of 'attr' (line 12)
    attr_59800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 13), 'attr')
    # Testing the type of a for loop iterable (line 12)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 12, 4), attr_59800)
    # Getting the type of the for loop variable (line 12)
    for_loop_var_59801 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 12, 4), attr_59800)
    # Assigning a type to the variable 's' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 's', for_loop_var_59801)
    # SSA begins for a for statement (line 12)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_raises(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'AttributeError' (line 13)
    AttributeError_59803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 22), 'AttributeError', False)
    # Getting the type of 'delattr' (line 13)
    delattr_59804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 38), 'delattr', False)
    # Getting the type of 'a' (line 13)
    a_59805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 47), 'a', False)
    # Getting the type of 's' (line 13)
    s_59806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 50), 's', False)
    # Processing the call keyword arguments (line 13)
    kwargs_59807 = {}
    # Getting the type of 'assert_raises' (line 13)
    assert_raises_59802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 13)
    assert_raises_call_result_59808 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), assert_raises_59802, *[AttributeError_59803, delattr_59804, a_59805, s_59806], **kwargs_59807)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_na_writable_attributes_deletion(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_na_writable_attributes_deletion' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_59809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_59809)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_na_writable_attributes_deletion'
    return stypy_return_type_59809

# Assigning a type to the variable 'test_na_writable_attributes_deletion' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'test_na_writable_attributes_deletion', test_na_writable_attributes_deletion)

# Assigning a Str to a Name (line 16):
str_59810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 15), 'str', '$Id: __init__.py,v 1.3 2005/05/16 11:08:49 pearu Exp $')
# Assigning a type to the variable '__revision__' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), '__revision__', str_59810)

# Assigning a List to a Name (line 18):

# Obtaining an instance of the builtin type 'list' (line 18)
list_59811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
str_59812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 19), 'str', 'clean')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 16), list_59811, str_59812)
# Adding element type (line 18)
str_59813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 19), 'str', 'install_clib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 16), list_59811, str_59813)
# Adding element type (line 18)
str_59814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 19), 'str', 'install_scripts')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 16), list_59811, str_59814)
# Adding element type (line 18)
str_59815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 19), 'str', 'bdist')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 16), list_59811, str_59815)
# Adding element type (line 18)
str_59816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 19), 'str', 'bdist_dumb')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 16), list_59811, str_59816)
# Adding element type (line 18)
str_59817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'str', 'bdist_wininst')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 16), list_59811, str_59817)

# Assigning a type to the variable 'distutils_all' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils_all', list_59811)

# Call to __import__(...): (line 27)
# Processing the call arguments (line 27)
str_59819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 11), 'str', 'distutils.command')

# Call to globals(...): (line 27)
# Processing the call keyword arguments (line 27)
kwargs_59821 = {}
# Getting the type of 'globals' (line 27)
globals_59820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 32), 'globals', False)
# Calling globals(args, kwargs) (line 27)
globals_call_result_59822 = invoke(stypy.reporting.localization.Localization(__file__, 27, 32), globals_59820, *[], **kwargs_59821)


# Call to locals(...): (line 27)
# Processing the call keyword arguments (line 27)
kwargs_59824 = {}
# Getting the type of 'locals' (line 27)
locals_59823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 43), 'locals', False)
# Calling locals(args, kwargs) (line 27)
locals_call_result_59825 = invoke(stypy.reporting.localization.Localization(__file__, 27, 43), locals_59823, *[], **kwargs_59824)

# Getting the type of 'distutils_all' (line 27)
distutils_all_59826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 53), 'distutils_all', False)
# Processing the call keyword arguments (line 27)
kwargs_59827 = {}
# Getting the type of '__import__' (line 27)
import___59818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), '__import__', False)
# Calling __import__(args, kwargs) (line 27)
import___call_result_59828 = invoke(stypy.reporting.localization.Localization(__file__, 27, 0), import___59818, *[str_59819, globals_call_result_59822, locals_call_result_59825, distutils_all_59826], **kwargs_59827)


# Assigning a BinOp to a Name (line 29):

# Obtaining an instance of the builtin type 'list' (line 29)
list_59829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 29)
# Adding element type (line 29)
str_59830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 11), 'str', 'build')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_59829, str_59830)
# Adding element type (line 29)
str_59831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 11), 'str', 'config_compiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_59829, str_59831)
# Adding element type (line 29)
str_59832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 11), 'str', 'config')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_59829, str_59832)
# Adding element type (line 29)
str_59833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 11), 'str', 'build_src')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_59829, str_59833)
# Adding element type (line 29)
str_59834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 11), 'str', 'build_py')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_59829, str_59834)
# Adding element type (line 29)
str_59835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 11), 'str', 'build_ext')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_59829, str_59835)
# Adding element type (line 29)
str_59836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 11), 'str', 'build_clib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_59829, str_59836)
# Adding element type (line 29)
str_59837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 11), 'str', 'build_scripts')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_59829, str_59837)
# Adding element type (line 29)
str_59838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 11), 'str', 'install')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_59829, str_59838)
# Adding element type (line 29)
str_59839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 11), 'str', 'install_data')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_59829, str_59839)
# Adding element type (line 29)
str_59840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 11), 'str', 'install_headers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_59829, str_59840)
# Adding element type (line 29)
str_59841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 11), 'str', 'install_lib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_59829, str_59841)
# Adding element type (line 29)
str_59842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 11), 'str', 'bdist_rpm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_59829, str_59842)
# Adding element type (line 29)
str_59843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 11), 'str', 'sdist')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_59829, str_59843)

# Getting the type of 'distutils_all' (line 43)
distutils_all_59844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 14), 'distutils_all')
# Applying the binary operator '+' (line 29)
result_add_59845 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 10), '+', list_59829, distutils_all_59844)

# Assigning a type to the variable '__all__' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), '__all__', result_add_59845)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
