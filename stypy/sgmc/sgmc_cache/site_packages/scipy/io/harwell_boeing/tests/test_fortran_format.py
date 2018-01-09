
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: 
5: from numpy.testing import assert_equal
6: from pytest import raises as assert_raises
7: 
8: from scipy.io.harwell_boeing._fortran_format_parser import (
9:         FortranFormatParser, IntFormat, ExpFormat, BadFortranFormat,
10:         number_digits)
11: 
12: 
13: class TestFortranFormatParser(object):
14:     def setup_method(self):
15:         self.parser = FortranFormatParser()
16: 
17:     def _test_equal(self, format, ref):
18:         ret = self.parser.parse(format)
19:         assert_equal(ret.__dict__, ref.__dict__)
20: 
21:     def test_simple_int(self):
22:         self._test_equal("(I4)", IntFormat(4))
23: 
24:     def test_simple_repeated_int(self):
25:         self._test_equal("(3I4)", IntFormat(4, repeat=3))
26: 
27:     def test_simple_exp(self):
28:         self._test_equal("(E4.3)", ExpFormat(4, 3))
29: 
30:     def test_exp_exp(self):
31:         self._test_equal("(E8.3E3)", ExpFormat(8, 3, 3))
32: 
33:     def test_repeat_exp(self):
34:         self._test_equal("(2E4.3)", ExpFormat(4, 3, repeat=2))
35: 
36:     def test_repeat_exp_exp(self):
37:         self._test_equal("(2E8.3E3)", ExpFormat(8, 3, 3, repeat=2))
38: 
39:     def test_wrong_formats(self):
40:         def _test_invalid(bad_format):
41:             assert_raises(BadFortranFormat, lambda: self.parser.parse(bad_format))
42:         _test_invalid("I4")
43:         _test_invalid("(E4)")
44:         _test_invalid("(E4.)")
45:         _test_invalid("(E4.E3)")
46: 
47: 
48: class TestIntFormat(object):
49:     def test_to_fortran(self):
50:         f = [IntFormat(10), IntFormat(12, 10), IntFormat(12, 10, 3)]
51:         res = ["(I10)", "(I12.10)", "(3I12.10)"]
52: 
53:         for i, j in zip(f, res):
54:             assert_equal(i.fortran_format, j)
55: 
56:     def test_from_number(self):
57:         f = [10, -12, 123456789]
58:         r_f = [IntFormat(3, repeat=26), IntFormat(4, repeat=20),
59:                IntFormat(10, repeat=8)]
60:         for i, j in zip(f, r_f):
61:             assert_equal(IntFormat.from_number(i).__dict__, j.__dict__)
62: 
63: 
64: class TestExpFormat(object):
65:     def test_to_fortran(self):
66:         f = [ExpFormat(10, 5), ExpFormat(12, 10), ExpFormat(12, 10, min=3),
67:              ExpFormat(10, 5, repeat=3)]
68:         res = ["(E10.5)", "(E12.10)", "(E12.10E3)", "(3E10.5)"]
69: 
70:         for i, j in zip(f, res):
71:             assert_equal(i.fortran_format, j)
72: 
73:     def test_from_number(self):
74:         f = np.array([1.0, -1.2])
75:         r_f = [ExpFormat(24, 16, repeat=3), ExpFormat(25, 16, repeat=3)]
76:         for i, j in zip(f, r_f):
77:             assert_equal(ExpFormat.from_number(i).__dict__, j.__dict__)
78: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/tests/')
import_132908 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_132908) is not StypyTypeError):

    if (import_132908 != 'pyd_module'):
        __import__(import_132908)
        sys_modules_132909 = sys.modules[import_132908]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_132909.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_132908)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.testing import assert_equal' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/tests/')
import_132910 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing')

if (type(import_132910) is not StypyTypeError):

    if (import_132910 != 'pyd_module'):
        __import__(import_132910)
        sys_modules_132911 = sys.modules[import_132910]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', sys_modules_132911.module_type_store, module_type_store, ['assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_132911, sys_modules_132911.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', None, module_type_store, ['assert_equal'], [assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', import_132910)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from pytest import assert_raises' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/tests/')
import_132912 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest')

if (type(import_132912) is not StypyTypeError):

    if (import_132912 != 'pyd_module'):
        __import__(import_132912)
        sys_modules_132913 = sys.modules[import_132912]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', sys_modules_132913.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_132913, sys_modules_132913.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', import_132912)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.io.harwell_boeing._fortran_format_parser import FortranFormatParser, IntFormat, ExpFormat, BadFortranFormat, number_digits' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/tests/')
import_132914 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.io.harwell_boeing._fortran_format_parser')

if (type(import_132914) is not StypyTypeError):

    if (import_132914 != 'pyd_module'):
        __import__(import_132914)
        sys_modules_132915 = sys.modules[import_132914]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.io.harwell_boeing._fortran_format_parser', sys_modules_132915.module_type_store, module_type_store, ['FortranFormatParser', 'IntFormat', 'ExpFormat', 'BadFortranFormat', 'number_digits'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_132915, sys_modules_132915.module_type_store, module_type_store)
    else:
        from scipy.io.harwell_boeing._fortran_format_parser import FortranFormatParser, IntFormat, ExpFormat, BadFortranFormat, number_digits

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.io.harwell_boeing._fortran_format_parser', None, module_type_store, ['FortranFormatParser', 'IntFormat', 'ExpFormat', 'BadFortranFormat', 'number_digits'], [FortranFormatParser, IntFormat, ExpFormat, BadFortranFormat, number_digits])

else:
    # Assigning a type to the variable 'scipy.io.harwell_boeing._fortran_format_parser' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.io.harwell_boeing._fortran_format_parser', import_132914)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/tests/')

# Declaration of the 'TestFortranFormatParser' class

class TestFortranFormatParser(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 14, 4, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFortranFormatParser.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestFortranFormatParser.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFortranFormatParser.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFortranFormatParser.setup_method.__dict__.__setitem__('stypy_function_name', 'TestFortranFormatParser.setup_method')
        TestFortranFormatParser.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestFortranFormatParser.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFortranFormatParser.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFortranFormatParser.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFortranFormatParser.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFortranFormatParser.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFortranFormatParser.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFortranFormatParser.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Call to a Attribute (line 15):
        
        # Call to FortranFormatParser(...): (line 15)
        # Processing the call keyword arguments (line 15)
        kwargs_132917 = {}
        # Getting the type of 'FortranFormatParser' (line 15)
        FortranFormatParser_132916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 22), 'FortranFormatParser', False)
        # Calling FortranFormatParser(args, kwargs) (line 15)
        FortranFormatParser_call_result_132918 = invoke(stypy.reporting.localization.Localization(__file__, 15, 22), FortranFormatParser_132916, *[], **kwargs_132917)
        
        # Getting the type of 'self' (line 15)
        self_132919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self')
        # Setting the type of the member 'parser' of a type (line 15)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), self_132919, 'parser', FortranFormatParser_call_result_132918)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 14)
        stypy_return_type_132920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132920)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_132920


    @norecursion
    def _test_equal(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_test_equal'
        module_type_store = module_type_store.open_function_context('_test_equal', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFortranFormatParser._test_equal.__dict__.__setitem__('stypy_localization', localization)
        TestFortranFormatParser._test_equal.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFortranFormatParser._test_equal.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFortranFormatParser._test_equal.__dict__.__setitem__('stypy_function_name', 'TestFortranFormatParser._test_equal')
        TestFortranFormatParser._test_equal.__dict__.__setitem__('stypy_param_names_list', ['format', 'ref'])
        TestFortranFormatParser._test_equal.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFortranFormatParser._test_equal.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFortranFormatParser._test_equal.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFortranFormatParser._test_equal.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFortranFormatParser._test_equal.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFortranFormatParser._test_equal.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFortranFormatParser._test_equal', ['format', 'ref'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_test_equal', localization, ['format', 'ref'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_test_equal(...)' code ##################

        
        # Assigning a Call to a Name (line 18):
        
        # Call to parse(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'format' (line 18)
        format_132924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 32), 'format', False)
        # Processing the call keyword arguments (line 18)
        kwargs_132925 = {}
        # Getting the type of 'self' (line 18)
        self_132921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), 'self', False)
        # Obtaining the member 'parser' of a type (line 18)
        parser_132922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 14), self_132921, 'parser')
        # Obtaining the member 'parse' of a type (line 18)
        parse_132923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 14), parser_132922, 'parse')
        # Calling parse(args, kwargs) (line 18)
        parse_call_result_132926 = invoke(stypy.reporting.localization.Localization(__file__, 18, 14), parse_132923, *[format_132924], **kwargs_132925)
        
        # Assigning a type to the variable 'ret' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'ret', parse_call_result_132926)
        
        # Call to assert_equal(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'ret' (line 19)
        ret_132928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 21), 'ret', False)
        # Obtaining the member '__dict__' of a type (line 19)
        dict___132929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 21), ret_132928, '__dict__')
        # Getting the type of 'ref' (line 19)
        ref_132930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 35), 'ref', False)
        # Obtaining the member '__dict__' of a type (line 19)
        dict___132931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 35), ref_132930, '__dict__')
        # Processing the call keyword arguments (line 19)
        kwargs_132932 = {}
        # Getting the type of 'assert_equal' (line 19)
        assert_equal_132927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 19)
        assert_equal_call_result_132933 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), assert_equal_132927, *[dict___132929, dict___132931], **kwargs_132932)
        
        
        # ################# End of '_test_equal(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_test_equal' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_132934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132934)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_test_equal'
        return stypy_return_type_132934


    @norecursion
    def test_simple_int(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_int'
        module_type_store = module_type_store.open_function_context('test_simple_int', 21, 4, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFortranFormatParser.test_simple_int.__dict__.__setitem__('stypy_localization', localization)
        TestFortranFormatParser.test_simple_int.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFortranFormatParser.test_simple_int.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFortranFormatParser.test_simple_int.__dict__.__setitem__('stypy_function_name', 'TestFortranFormatParser.test_simple_int')
        TestFortranFormatParser.test_simple_int.__dict__.__setitem__('stypy_param_names_list', [])
        TestFortranFormatParser.test_simple_int.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFortranFormatParser.test_simple_int.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFortranFormatParser.test_simple_int.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFortranFormatParser.test_simple_int.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFortranFormatParser.test_simple_int.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFortranFormatParser.test_simple_int.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFortranFormatParser.test_simple_int', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_int', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_int(...)' code ##################

        
        # Call to _test_equal(...): (line 22)
        # Processing the call arguments (line 22)
        str_132937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'str', '(I4)')
        
        # Call to IntFormat(...): (line 22)
        # Processing the call arguments (line 22)
        int_132939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 43), 'int')
        # Processing the call keyword arguments (line 22)
        kwargs_132940 = {}
        # Getting the type of 'IntFormat' (line 22)
        IntFormat_132938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 33), 'IntFormat', False)
        # Calling IntFormat(args, kwargs) (line 22)
        IntFormat_call_result_132941 = invoke(stypy.reporting.localization.Localization(__file__, 22, 33), IntFormat_132938, *[int_132939], **kwargs_132940)
        
        # Processing the call keyword arguments (line 22)
        kwargs_132942 = {}
        # Getting the type of 'self' (line 22)
        self_132935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self', False)
        # Obtaining the member '_test_equal' of a type (line 22)
        _test_equal_132936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_132935, '_test_equal')
        # Calling _test_equal(args, kwargs) (line 22)
        _test_equal_call_result_132943 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), _test_equal_132936, *[str_132937, IntFormat_call_result_132941], **kwargs_132942)
        
        
        # ################# End of 'test_simple_int(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_int' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_132944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132944)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_int'
        return stypy_return_type_132944


    @norecursion
    def test_simple_repeated_int(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_repeated_int'
        module_type_store = module_type_store.open_function_context('test_simple_repeated_int', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFortranFormatParser.test_simple_repeated_int.__dict__.__setitem__('stypy_localization', localization)
        TestFortranFormatParser.test_simple_repeated_int.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFortranFormatParser.test_simple_repeated_int.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFortranFormatParser.test_simple_repeated_int.__dict__.__setitem__('stypy_function_name', 'TestFortranFormatParser.test_simple_repeated_int')
        TestFortranFormatParser.test_simple_repeated_int.__dict__.__setitem__('stypy_param_names_list', [])
        TestFortranFormatParser.test_simple_repeated_int.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFortranFormatParser.test_simple_repeated_int.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFortranFormatParser.test_simple_repeated_int.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFortranFormatParser.test_simple_repeated_int.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFortranFormatParser.test_simple_repeated_int.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFortranFormatParser.test_simple_repeated_int.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFortranFormatParser.test_simple_repeated_int', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_repeated_int', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_repeated_int(...)' code ##################

        
        # Call to _test_equal(...): (line 25)
        # Processing the call arguments (line 25)
        str_132947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'str', '(3I4)')
        
        # Call to IntFormat(...): (line 25)
        # Processing the call arguments (line 25)
        int_132949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 44), 'int')
        # Processing the call keyword arguments (line 25)
        int_132950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 54), 'int')
        keyword_132951 = int_132950
        kwargs_132952 = {'repeat': keyword_132951}
        # Getting the type of 'IntFormat' (line 25)
        IntFormat_132948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 34), 'IntFormat', False)
        # Calling IntFormat(args, kwargs) (line 25)
        IntFormat_call_result_132953 = invoke(stypy.reporting.localization.Localization(__file__, 25, 34), IntFormat_132948, *[int_132949], **kwargs_132952)
        
        # Processing the call keyword arguments (line 25)
        kwargs_132954 = {}
        # Getting the type of 'self' (line 25)
        self_132945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self', False)
        # Obtaining the member '_test_equal' of a type (line 25)
        _test_equal_132946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_132945, '_test_equal')
        # Calling _test_equal(args, kwargs) (line 25)
        _test_equal_call_result_132955 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), _test_equal_132946, *[str_132947, IntFormat_call_result_132953], **kwargs_132954)
        
        
        # ################# End of 'test_simple_repeated_int(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_repeated_int' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_132956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132956)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_repeated_int'
        return stypy_return_type_132956


    @norecursion
    def test_simple_exp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_exp'
        module_type_store = module_type_store.open_function_context('test_simple_exp', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFortranFormatParser.test_simple_exp.__dict__.__setitem__('stypy_localization', localization)
        TestFortranFormatParser.test_simple_exp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFortranFormatParser.test_simple_exp.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFortranFormatParser.test_simple_exp.__dict__.__setitem__('stypy_function_name', 'TestFortranFormatParser.test_simple_exp')
        TestFortranFormatParser.test_simple_exp.__dict__.__setitem__('stypy_param_names_list', [])
        TestFortranFormatParser.test_simple_exp.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFortranFormatParser.test_simple_exp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFortranFormatParser.test_simple_exp.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFortranFormatParser.test_simple_exp.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFortranFormatParser.test_simple_exp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFortranFormatParser.test_simple_exp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFortranFormatParser.test_simple_exp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_exp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_exp(...)' code ##################

        
        # Call to _test_equal(...): (line 28)
        # Processing the call arguments (line 28)
        str_132959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'str', '(E4.3)')
        
        # Call to ExpFormat(...): (line 28)
        # Processing the call arguments (line 28)
        int_132961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 45), 'int')
        int_132962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 48), 'int')
        # Processing the call keyword arguments (line 28)
        kwargs_132963 = {}
        # Getting the type of 'ExpFormat' (line 28)
        ExpFormat_132960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 35), 'ExpFormat', False)
        # Calling ExpFormat(args, kwargs) (line 28)
        ExpFormat_call_result_132964 = invoke(stypy.reporting.localization.Localization(__file__, 28, 35), ExpFormat_132960, *[int_132961, int_132962], **kwargs_132963)
        
        # Processing the call keyword arguments (line 28)
        kwargs_132965 = {}
        # Getting the type of 'self' (line 28)
        self_132957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self', False)
        # Obtaining the member '_test_equal' of a type (line 28)
        _test_equal_132958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_132957, '_test_equal')
        # Calling _test_equal(args, kwargs) (line 28)
        _test_equal_call_result_132966 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), _test_equal_132958, *[str_132959, ExpFormat_call_result_132964], **kwargs_132965)
        
        
        # ################# End of 'test_simple_exp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_exp' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_132967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132967)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_exp'
        return stypy_return_type_132967


    @norecursion
    def test_exp_exp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_exp_exp'
        module_type_store = module_type_store.open_function_context('test_exp_exp', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFortranFormatParser.test_exp_exp.__dict__.__setitem__('stypy_localization', localization)
        TestFortranFormatParser.test_exp_exp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFortranFormatParser.test_exp_exp.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFortranFormatParser.test_exp_exp.__dict__.__setitem__('stypy_function_name', 'TestFortranFormatParser.test_exp_exp')
        TestFortranFormatParser.test_exp_exp.__dict__.__setitem__('stypy_param_names_list', [])
        TestFortranFormatParser.test_exp_exp.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFortranFormatParser.test_exp_exp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFortranFormatParser.test_exp_exp.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFortranFormatParser.test_exp_exp.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFortranFormatParser.test_exp_exp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFortranFormatParser.test_exp_exp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFortranFormatParser.test_exp_exp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_exp_exp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_exp_exp(...)' code ##################

        
        # Call to _test_equal(...): (line 31)
        # Processing the call arguments (line 31)
        str_132970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 25), 'str', '(E8.3E3)')
        
        # Call to ExpFormat(...): (line 31)
        # Processing the call arguments (line 31)
        int_132972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 47), 'int')
        int_132973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 50), 'int')
        int_132974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 53), 'int')
        # Processing the call keyword arguments (line 31)
        kwargs_132975 = {}
        # Getting the type of 'ExpFormat' (line 31)
        ExpFormat_132971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 37), 'ExpFormat', False)
        # Calling ExpFormat(args, kwargs) (line 31)
        ExpFormat_call_result_132976 = invoke(stypy.reporting.localization.Localization(__file__, 31, 37), ExpFormat_132971, *[int_132972, int_132973, int_132974], **kwargs_132975)
        
        # Processing the call keyword arguments (line 31)
        kwargs_132977 = {}
        # Getting the type of 'self' (line 31)
        self_132968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self', False)
        # Obtaining the member '_test_equal' of a type (line 31)
        _test_equal_132969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_132968, '_test_equal')
        # Calling _test_equal(args, kwargs) (line 31)
        _test_equal_call_result_132978 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), _test_equal_132969, *[str_132970, ExpFormat_call_result_132976], **kwargs_132977)
        
        
        # ################# End of 'test_exp_exp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_exp_exp' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_132979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132979)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_exp_exp'
        return stypy_return_type_132979


    @norecursion
    def test_repeat_exp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_repeat_exp'
        module_type_store = module_type_store.open_function_context('test_repeat_exp', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFortranFormatParser.test_repeat_exp.__dict__.__setitem__('stypy_localization', localization)
        TestFortranFormatParser.test_repeat_exp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFortranFormatParser.test_repeat_exp.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFortranFormatParser.test_repeat_exp.__dict__.__setitem__('stypy_function_name', 'TestFortranFormatParser.test_repeat_exp')
        TestFortranFormatParser.test_repeat_exp.__dict__.__setitem__('stypy_param_names_list', [])
        TestFortranFormatParser.test_repeat_exp.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFortranFormatParser.test_repeat_exp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFortranFormatParser.test_repeat_exp.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFortranFormatParser.test_repeat_exp.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFortranFormatParser.test_repeat_exp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFortranFormatParser.test_repeat_exp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFortranFormatParser.test_repeat_exp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_repeat_exp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_repeat_exp(...)' code ##################

        
        # Call to _test_equal(...): (line 34)
        # Processing the call arguments (line 34)
        str_132982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 25), 'str', '(2E4.3)')
        
        # Call to ExpFormat(...): (line 34)
        # Processing the call arguments (line 34)
        int_132984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 46), 'int')
        int_132985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 49), 'int')
        # Processing the call keyword arguments (line 34)
        int_132986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 59), 'int')
        keyword_132987 = int_132986
        kwargs_132988 = {'repeat': keyword_132987}
        # Getting the type of 'ExpFormat' (line 34)
        ExpFormat_132983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 36), 'ExpFormat', False)
        # Calling ExpFormat(args, kwargs) (line 34)
        ExpFormat_call_result_132989 = invoke(stypy.reporting.localization.Localization(__file__, 34, 36), ExpFormat_132983, *[int_132984, int_132985], **kwargs_132988)
        
        # Processing the call keyword arguments (line 34)
        kwargs_132990 = {}
        # Getting the type of 'self' (line 34)
        self_132980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self', False)
        # Obtaining the member '_test_equal' of a type (line 34)
        _test_equal_132981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_132980, '_test_equal')
        # Calling _test_equal(args, kwargs) (line 34)
        _test_equal_call_result_132991 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), _test_equal_132981, *[str_132982, ExpFormat_call_result_132989], **kwargs_132990)
        
        
        # ################# End of 'test_repeat_exp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_repeat_exp' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_132992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_132992)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_repeat_exp'
        return stypy_return_type_132992


    @norecursion
    def test_repeat_exp_exp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_repeat_exp_exp'
        module_type_store = module_type_store.open_function_context('test_repeat_exp_exp', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFortranFormatParser.test_repeat_exp_exp.__dict__.__setitem__('stypy_localization', localization)
        TestFortranFormatParser.test_repeat_exp_exp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFortranFormatParser.test_repeat_exp_exp.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFortranFormatParser.test_repeat_exp_exp.__dict__.__setitem__('stypy_function_name', 'TestFortranFormatParser.test_repeat_exp_exp')
        TestFortranFormatParser.test_repeat_exp_exp.__dict__.__setitem__('stypy_param_names_list', [])
        TestFortranFormatParser.test_repeat_exp_exp.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFortranFormatParser.test_repeat_exp_exp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFortranFormatParser.test_repeat_exp_exp.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFortranFormatParser.test_repeat_exp_exp.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFortranFormatParser.test_repeat_exp_exp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFortranFormatParser.test_repeat_exp_exp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFortranFormatParser.test_repeat_exp_exp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_repeat_exp_exp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_repeat_exp_exp(...)' code ##################

        
        # Call to _test_equal(...): (line 37)
        # Processing the call arguments (line 37)
        str_132995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 25), 'str', '(2E8.3E3)')
        
        # Call to ExpFormat(...): (line 37)
        # Processing the call arguments (line 37)
        int_132997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 48), 'int')
        int_132998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 51), 'int')
        int_132999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 54), 'int')
        # Processing the call keyword arguments (line 37)
        int_133000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 64), 'int')
        keyword_133001 = int_133000
        kwargs_133002 = {'repeat': keyword_133001}
        # Getting the type of 'ExpFormat' (line 37)
        ExpFormat_132996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 38), 'ExpFormat', False)
        # Calling ExpFormat(args, kwargs) (line 37)
        ExpFormat_call_result_133003 = invoke(stypy.reporting.localization.Localization(__file__, 37, 38), ExpFormat_132996, *[int_132997, int_132998, int_132999], **kwargs_133002)
        
        # Processing the call keyword arguments (line 37)
        kwargs_133004 = {}
        # Getting the type of 'self' (line 37)
        self_132993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self', False)
        # Obtaining the member '_test_equal' of a type (line 37)
        _test_equal_132994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_132993, '_test_equal')
        # Calling _test_equal(args, kwargs) (line 37)
        _test_equal_call_result_133005 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), _test_equal_132994, *[str_132995, ExpFormat_call_result_133003], **kwargs_133004)
        
        
        # ################# End of 'test_repeat_exp_exp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_repeat_exp_exp' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_133006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_133006)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_repeat_exp_exp'
        return stypy_return_type_133006


    @norecursion
    def test_wrong_formats(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_wrong_formats'
        module_type_store = module_type_store.open_function_context('test_wrong_formats', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFortranFormatParser.test_wrong_formats.__dict__.__setitem__('stypy_localization', localization)
        TestFortranFormatParser.test_wrong_formats.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFortranFormatParser.test_wrong_formats.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFortranFormatParser.test_wrong_formats.__dict__.__setitem__('stypy_function_name', 'TestFortranFormatParser.test_wrong_formats')
        TestFortranFormatParser.test_wrong_formats.__dict__.__setitem__('stypy_param_names_list', [])
        TestFortranFormatParser.test_wrong_formats.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFortranFormatParser.test_wrong_formats.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFortranFormatParser.test_wrong_formats.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFortranFormatParser.test_wrong_formats.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFortranFormatParser.test_wrong_formats.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFortranFormatParser.test_wrong_formats.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFortranFormatParser.test_wrong_formats', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_wrong_formats', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_wrong_formats(...)' code ##################


        @norecursion
        def _test_invalid(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_test_invalid'
            module_type_store = module_type_store.open_function_context('_test_invalid', 40, 8, False)
            
            # Passed parameters checking function
            _test_invalid.stypy_localization = localization
            _test_invalid.stypy_type_of_self = None
            _test_invalid.stypy_type_store = module_type_store
            _test_invalid.stypy_function_name = '_test_invalid'
            _test_invalid.stypy_param_names_list = ['bad_format']
            _test_invalid.stypy_varargs_param_name = None
            _test_invalid.stypy_kwargs_param_name = None
            _test_invalid.stypy_call_defaults = defaults
            _test_invalid.stypy_call_varargs = varargs
            _test_invalid.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_test_invalid', ['bad_format'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_test_invalid', localization, ['bad_format'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_test_invalid(...)' code ##################

            
            # Call to assert_raises(...): (line 41)
            # Processing the call arguments (line 41)
            # Getting the type of 'BadFortranFormat' (line 41)
            BadFortranFormat_133008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 26), 'BadFortranFormat', False)

            @norecursion
            def _stypy_temp_lambda_87(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_87'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_87', 41, 44, True)
                # Passed parameters checking function
                _stypy_temp_lambda_87.stypy_localization = localization
                _stypy_temp_lambda_87.stypy_type_of_self = None
                _stypy_temp_lambda_87.stypy_type_store = module_type_store
                _stypy_temp_lambda_87.stypy_function_name = '_stypy_temp_lambda_87'
                _stypy_temp_lambda_87.stypy_param_names_list = []
                _stypy_temp_lambda_87.stypy_varargs_param_name = None
                _stypy_temp_lambda_87.stypy_kwargs_param_name = None
                _stypy_temp_lambda_87.stypy_call_defaults = defaults
                _stypy_temp_lambda_87.stypy_call_varargs = varargs
                _stypy_temp_lambda_87.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_87', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_87', [], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Call to parse(...): (line 41)
                # Processing the call arguments (line 41)
                # Getting the type of 'bad_format' (line 41)
                bad_format_133012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 70), 'bad_format', False)
                # Processing the call keyword arguments (line 41)
                kwargs_133013 = {}
                # Getting the type of 'self' (line 41)
                self_133009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 52), 'self', False)
                # Obtaining the member 'parser' of a type (line 41)
                parser_133010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 52), self_133009, 'parser')
                # Obtaining the member 'parse' of a type (line 41)
                parse_133011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 52), parser_133010, 'parse')
                # Calling parse(args, kwargs) (line 41)
                parse_call_result_133014 = invoke(stypy.reporting.localization.Localization(__file__, 41, 52), parse_133011, *[bad_format_133012], **kwargs_133013)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 41)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 44), 'stypy_return_type', parse_call_result_133014)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_87' in the type store
                # Getting the type of 'stypy_return_type' (line 41)
                stypy_return_type_133015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 44), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_133015)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_87'
                return stypy_return_type_133015

            # Assigning a type to the variable '_stypy_temp_lambda_87' (line 41)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 44), '_stypy_temp_lambda_87', _stypy_temp_lambda_87)
            # Getting the type of '_stypy_temp_lambda_87' (line 41)
            _stypy_temp_lambda_87_133016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 44), '_stypy_temp_lambda_87')
            # Processing the call keyword arguments (line 41)
            kwargs_133017 = {}
            # Getting the type of 'assert_raises' (line 41)
            assert_raises_133007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'assert_raises', False)
            # Calling assert_raises(args, kwargs) (line 41)
            assert_raises_call_result_133018 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), assert_raises_133007, *[BadFortranFormat_133008, _stypy_temp_lambda_87_133016], **kwargs_133017)
            
            
            # ################# End of '_test_invalid(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_test_invalid' in the type store
            # Getting the type of 'stypy_return_type' (line 40)
            stypy_return_type_133019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_133019)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_test_invalid'
            return stypy_return_type_133019

        # Assigning a type to the variable '_test_invalid' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), '_test_invalid', _test_invalid)
        
        # Call to _test_invalid(...): (line 42)
        # Processing the call arguments (line 42)
        str_133021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 22), 'str', 'I4')
        # Processing the call keyword arguments (line 42)
        kwargs_133022 = {}
        # Getting the type of '_test_invalid' (line 42)
        _test_invalid_133020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), '_test_invalid', False)
        # Calling _test_invalid(args, kwargs) (line 42)
        _test_invalid_call_result_133023 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), _test_invalid_133020, *[str_133021], **kwargs_133022)
        
        
        # Call to _test_invalid(...): (line 43)
        # Processing the call arguments (line 43)
        str_133025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 22), 'str', '(E4)')
        # Processing the call keyword arguments (line 43)
        kwargs_133026 = {}
        # Getting the type of '_test_invalid' (line 43)
        _test_invalid_133024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), '_test_invalid', False)
        # Calling _test_invalid(args, kwargs) (line 43)
        _test_invalid_call_result_133027 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), _test_invalid_133024, *[str_133025], **kwargs_133026)
        
        
        # Call to _test_invalid(...): (line 44)
        # Processing the call arguments (line 44)
        str_133029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 22), 'str', '(E4.)')
        # Processing the call keyword arguments (line 44)
        kwargs_133030 = {}
        # Getting the type of '_test_invalid' (line 44)
        _test_invalid_133028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), '_test_invalid', False)
        # Calling _test_invalid(args, kwargs) (line 44)
        _test_invalid_call_result_133031 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), _test_invalid_133028, *[str_133029], **kwargs_133030)
        
        
        # Call to _test_invalid(...): (line 45)
        # Processing the call arguments (line 45)
        str_133033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 22), 'str', '(E4.E3)')
        # Processing the call keyword arguments (line 45)
        kwargs_133034 = {}
        # Getting the type of '_test_invalid' (line 45)
        _test_invalid_133032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), '_test_invalid', False)
        # Calling _test_invalid(args, kwargs) (line 45)
        _test_invalid_call_result_133035 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), _test_invalid_133032, *[str_133033], **kwargs_133034)
        
        
        # ################# End of 'test_wrong_formats(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_wrong_formats' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_133036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_133036)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_wrong_formats'
        return stypy_return_type_133036


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 13, 0, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFortranFormatParser.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestFortranFormatParser' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'TestFortranFormatParser', TestFortranFormatParser)
# Declaration of the 'TestIntFormat' class

class TestIntFormat(object, ):

    @norecursion
    def test_to_fortran(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_to_fortran'
        module_type_store = module_type_store.open_function_context('test_to_fortran', 49, 4, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIntFormat.test_to_fortran.__dict__.__setitem__('stypy_localization', localization)
        TestIntFormat.test_to_fortran.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIntFormat.test_to_fortran.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIntFormat.test_to_fortran.__dict__.__setitem__('stypy_function_name', 'TestIntFormat.test_to_fortran')
        TestIntFormat.test_to_fortran.__dict__.__setitem__('stypy_param_names_list', [])
        TestIntFormat.test_to_fortran.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIntFormat.test_to_fortran.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIntFormat.test_to_fortran.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIntFormat.test_to_fortran.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIntFormat.test_to_fortran.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIntFormat.test_to_fortran.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIntFormat.test_to_fortran', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_to_fortran', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_to_fortran(...)' code ##################

        
        # Assigning a List to a Name (line 50):
        
        # Obtaining an instance of the builtin type 'list' (line 50)
        list_133037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 50)
        # Adding element type (line 50)
        
        # Call to IntFormat(...): (line 50)
        # Processing the call arguments (line 50)
        int_133039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 23), 'int')
        # Processing the call keyword arguments (line 50)
        kwargs_133040 = {}
        # Getting the type of 'IntFormat' (line 50)
        IntFormat_133038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 13), 'IntFormat', False)
        # Calling IntFormat(args, kwargs) (line 50)
        IntFormat_call_result_133041 = invoke(stypy.reporting.localization.Localization(__file__, 50, 13), IntFormat_133038, *[int_133039], **kwargs_133040)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), list_133037, IntFormat_call_result_133041)
        # Adding element type (line 50)
        
        # Call to IntFormat(...): (line 50)
        # Processing the call arguments (line 50)
        int_133043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 38), 'int')
        int_133044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 42), 'int')
        # Processing the call keyword arguments (line 50)
        kwargs_133045 = {}
        # Getting the type of 'IntFormat' (line 50)
        IntFormat_133042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 28), 'IntFormat', False)
        # Calling IntFormat(args, kwargs) (line 50)
        IntFormat_call_result_133046 = invoke(stypy.reporting.localization.Localization(__file__, 50, 28), IntFormat_133042, *[int_133043, int_133044], **kwargs_133045)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), list_133037, IntFormat_call_result_133046)
        # Adding element type (line 50)
        
        # Call to IntFormat(...): (line 50)
        # Processing the call arguments (line 50)
        int_133048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 57), 'int')
        int_133049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 61), 'int')
        int_133050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 65), 'int')
        # Processing the call keyword arguments (line 50)
        kwargs_133051 = {}
        # Getting the type of 'IntFormat' (line 50)
        IntFormat_133047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 47), 'IntFormat', False)
        # Calling IntFormat(args, kwargs) (line 50)
        IntFormat_call_result_133052 = invoke(stypy.reporting.localization.Localization(__file__, 50, 47), IntFormat_133047, *[int_133048, int_133049, int_133050], **kwargs_133051)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), list_133037, IntFormat_call_result_133052)
        
        # Assigning a type to the variable 'f' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'f', list_133037)
        
        # Assigning a List to a Name (line 51):
        
        # Obtaining an instance of the builtin type 'list' (line 51)
        list_133053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 51)
        # Adding element type (line 51)
        str_133054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 15), 'str', '(I10)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 14), list_133053, str_133054)
        # Adding element type (line 51)
        str_133055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 24), 'str', '(I12.10)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 14), list_133053, str_133055)
        # Adding element type (line 51)
        str_133056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 36), 'str', '(3I12.10)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 14), list_133053, str_133056)
        
        # Assigning a type to the variable 'res' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'res', list_133053)
        
        
        # Call to zip(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'f' (line 53)
        f_133058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 24), 'f', False)
        # Getting the type of 'res' (line 53)
        res_133059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 27), 'res', False)
        # Processing the call keyword arguments (line 53)
        kwargs_133060 = {}
        # Getting the type of 'zip' (line 53)
        zip_133057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'zip', False)
        # Calling zip(args, kwargs) (line 53)
        zip_call_result_133061 = invoke(stypy.reporting.localization.Localization(__file__, 53, 20), zip_133057, *[f_133058, res_133059], **kwargs_133060)
        
        # Testing the type of a for loop iterable (line 53)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 53, 8), zip_call_result_133061)
        # Getting the type of the for loop variable (line 53)
        for_loop_var_133062 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 53, 8), zip_call_result_133061)
        # Assigning a type to the variable 'i' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 8), for_loop_var_133062))
        # Assigning a type to the variable 'j' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 8), for_loop_var_133062))
        # SSA begins for a for statement (line 53)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_equal(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'i' (line 54)
        i_133064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'i', False)
        # Obtaining the member 'fortran_format' of a type (line 54)
        fortran_format_133065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 25), i_133064, 'fortran_format')
        # Getting the type of 'j' (line 54)
        j_133066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 43), 'j', False)
        # Processing the call keyword arguments (line 54)
        kwargs_133067 = {}
        # Getting the type of 'assert_equal' (line 54)
        assert_equal_133063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 54)
        assert_equal_call_result_133068 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), assert_equal_133063, *[fortran_format_133065, j_133066], **kwargs_133067)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_to_fortran(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_to_fortran' in the type store
        # Getting the type of 'stypy_return_type' (line 49)
        stypy_return_type_133069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_133069)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_to_fortran'
        return stypy_return_type_133069


    @norecursion
    def test_from_number(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_from_number'
        module_type_store = module_type_store.open_function_context('test_from_number', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIntFormat.test_from_number.__dict__.__setitem__('stypy_localization', localization)
        TestIntFormat.test_from_number.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIntFormat.test_from_number.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIntFormat.test_from_number.__dict__.__setitem__('stypy_function_name', 'TestIntFormat.test_from_number')
        TestIntFormat.test_from_number.__dict__.__setitem__('stypy_param_names_list', [])
        TestIntFormat.test_from_number.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIntFormat.test_from_number.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIntFormat.test_from_number.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIntFormat.test_from_number.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIntFormat.test_from_number.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIntFormat.test_from_number.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIntFormat.test_from_number', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_from_number', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_from_number(...)' code ##################

        
        # Assigning a List to a Name (line 57):
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_133070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        # Adding element type (line 57)
        int_133071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 12), list_133070, int_133071)
        # Adding element type (line 57)
        int_133072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 12), list_133070, int_133072)
        # Adding element type (line 57)
        int_133073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 12), list_133070, int_133073)
        
        # Assigning a type to the variable 'f' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'f', list_133070)
        
        # Assigning a List to a Name (line 58):
        
        # Obtaining an instance of the builtin type 'list' (line 58)
        list_133074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 58)
        # Adding element type (line 58)
        
        # Call to IntFormat(...): (line 58)
        # Processing the call arguments (line 58)
        int_133076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 25), 'int')
        # Processing the call keyword arguments (line 58)
        int_133077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 35), 'int')
        keyword_133078 = int_133077
        kwargs_133079 = {'repeat': keyword_133078}
        # Getting the type of 'IntFormat' (line 58)
        IntFormat_133075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'IntFormat', False)
        # Calling IntFormat(args, kwargs) (line 58)
        IntFormat_call_result_133080 = invoke(stypy.reporting.localization.Localization(__file__, 58, 15), IntFormat_133075, *[int_133076], **kwargs_133079)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 14), list_133074, IntFormat_call_result_133080)
        # Adding element type (line 58)
        
        # Call to IntFormat(...): (line 58)
        # Processing the call arguments (line 58)
        int_133082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 50), 'int')
        # Processing the call keyword arguments (line 58)
        int_133083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 60), 'int')
        keyword_133084 = int_133083
        kwargs_133085 = {'repeat': keyword_133084}
        # Getting the type of 'IntFormat' (line 58)
        IntFormat_133081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 40), 'IntFormat', False)
        # Calling IntFormat(args, kwargs) (line 58)
        IntFormat_call_result_133086 = invoke(stypy.reporting.localization.Localization(__file__, 58, 40), IntFormat_133081, *[int_133082], **kwargs_133085)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 14), list_133074, IntFormat_call_result_133086)
        # Adding element type (line 58)
        
        # Call to IntFormat(...): (line 59)
        # Processing the call arguments (line 59)
        int_133088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 25), 'int')
        # Processing the call keyword arguments (line 59)
        int_133089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 36), 'int')
        keyword_133090 = int_133089
        kwargs_133091 = {'repeat': keyword_133090}
        # Getting the type of 'IntFormat' (line 59)
        IntFormat_133087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'IntFormat', False)
        # Calling IntFormat(args, kwargs) (line 59)
        IntFormat_call_result_133092 = invoke(stypy.reporting.localization.Localization(__file__, 59, 15), IntFormat_133087, *[int_133088], **kwargs_133091)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 14), list_133074, IntFormat_call_result_133092)
        
        # Assigning a type to the variable 'r_f' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'r_f', list_133074)
        
        
        # Call to zip(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'f' (line 60)
        f_133094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 24), 'f', False)
        # Getting the type of 'r_f' (line 60)
        r_f_133095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 27), 'r_f', False)
        # Processing the call keyword arguments (line 60)
        kwargs_133096 = {}
        # Getting the type of 'zip' (line 60)
        zip_133093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'zip', False)
        # Calling zip(args, kwargs) (line 60)
        zip_call_result_133097 = invoke(stypy.reporting.localization.Localization(__file__, 60, 20), zip_133093, *[f_133094, r_f_133095], **kwargs_133096)
        
        # Testing the type of a for loop iterable (line 60)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 60, 8), zip_call_result_133097)
        # Getting the type of the for loop variable (line 60)
        for_loop_var_133098 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 60, 8), zip_call_result_133097)
        # Assigning a type to the variable 'i' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 8), for_loop_var_133098))
        # Assigning a type to the variable 'j' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 8), for_loop_var_133098))
        # SSA begins for a for statement (line 60)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_equal(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Call to from_number(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'i' (line 61)
        i_133102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 47), 'i', False)
        # Processing the call keyword arguments (line 61)
        kwargs_133103 = {}
        # Getting the type of 'IntFormat' (line 61)
        IntFormat_133100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'IntFormat', False)
        # Obtaining the member 'from_number' of a type (line 61)
        from_number_133101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 25), IntFormat_133100, 'from_number')
        # Calling from_number(args, kwargs) (line 61)
        from_number_call_result_133104 = invoke(stypy.reporting.localization.Localization(__file__, 61, 25), from_number_133101, *[i_133102], **kwargs_133103)
        
        # Obtaining the member '__dict__' of a type (line 61)
        dict___133105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 25), from_number_call_result_133104, '__dict__')
        # Getting the type of 'j' (line 61)
        j_133106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 60), 'j', False)
        # Obtaining the member '__dict__' of a type (line 61)
        dict___133107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 60), j_133106, '__dict__')
        # Processing the call keyword arguments (line 61)
        kwargs_133108 = {}
        # Getting the type of 'assert_equal' (line 61)
        assert_equal_133099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 61)
        assert_equal_call_result_133109 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), assert_equal_133099, *[dict___133105, dict___133107], **kwargs_133108)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_from_number(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_from_number' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_133110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_133110)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_from_number'
        return stypy_return_type_133110


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 48, 0, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIntFormat.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIntFormat' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'TestIntFormat', TestIntFormat)
# Declaration of the 'TestExpFormat' class

class TestExpFormat(object, ):

    @norecursion
    def test_to_fortran(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_to_fortran'
        module_type_store = module_type_store.open_function_context('test_to_fortran', 65, 4, False)
        # Assigning a type to the variable 'self' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpFormat.test_to_fortran.__dict__.__setitem__('stypy_localization', localization)
        TestExpFormat.test_to_fortran.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpFormat.test_to_fortran.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpFormat.test_to_fortran.__dict__.__setitem__('stypy_function_name', 'TestExpFormat.test_to_fortran')
        TestExpFormat.test_to_fortran.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpFormat.test_to_fortran.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpFormat.test_to_fortran.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpFormat.test_to_fortran.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpFormat.test_to_fortran.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpFormat.test_to_fortran.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpFormat.test_to_fortran.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpFormat.test_to_fortran', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_to_fortran', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_to_fortran(...)' code ##################

        
        # Assigning a List to a Name (line 66):
        
        # Obtaining an instance of the builtin type 'list' (line 66)
        list_133111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 66)
        # Adding element type (line 66)
        
        # Call to ExpFormat(...): (line 66)
        # Processing the call arguments (line 66)
        int_133113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 23), 'int')
        int_133114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 27), 'int')
        # Processing the call keyword arguments (line 66)
        kwargs_133115 = {}
        # Getting the type of 'ExpFormat' (line 66)
        ExpFormat_133112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 13), 'ExpFormat', False)
        # Calling ExpFormat(args, kwargs) (line 66)
        ExpFormat_call_result_133116 = invoke(stypy.reporting.localization.Localization(__file__, 66, 13), ExpFormat_133112, *[int_133113, int_133114], **kwargs_133115)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 12), list_133111, ExpFormat_call_result_133116)
        # Adding element type (line 66)
        
        # Call to ExpFormat(...): (line 66)
        # Processing the call arguments (line 66)
        int_133118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 41), 'int')
        int_133119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 45), 'int')
        # Processing the call keyword arguments (line 66)
        kwargs_133120 = {}
        # Getting the type of 'ExpFormat' (line 66)
        ExpFormat_133117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 31), 'ExpFormat', False)
        # Calling ExpFormat(args, kwargs) (line 66)
        ExpFormat_call_result_133121 = invoke(stypy.reporting.localization.Localization(__file__, 66, 31), ExpFormat_133117, *[int_133118, int_133119], **kwargs_133120)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 12), list_133111, ExpFormat_call_result_133121)
        # Adding element type (line 66)
        
        # Call to ExpFormat(...): (line 66)
        # Processing the call arguments (line 66)
        int_133123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 60), 'int')
        int_133124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 64), 'int')
        # Processing the call keyword arguments (line 66)
        int_133125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 72), 'int')
        keyword_133126 = int_133125
        kwargs_133127 = {'min': keyword_133126}
        # Getting the type of 'ExpFormat' (line 66)
        ExpFormat_133122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 50), 'ExpFormat', False)
        # Calling ExpFormat(args, kwargs) (line 66)
        ExpFormat_call_result_133128 = invoke(stypy.reporting.localization.Localization(__file__, 66, 50), ExpFormat_133122, *[int_133123, int_133124], **kwargs_133127)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 12), list_133111, ExpFormat_call_result_133128)
        # Adding element type (line 66)
        
        # Call to ExpFormat(...): (line 67)
        # Processing the call arguments (line 67)
        int_133130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 23), 'int')
        int_133131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 27), 'int')
        # Processing the call keyword arguments (line 67)
        int_133132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 37), 'int')
        keyword_133133 = int_133132
        kwargs_133134 = {'repeat': keyword_133133}
        # Getting the type of 'ExpFormat' (line 67)
        ExpFormat_133129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 13), 'ExpFormat', False)
        # Calling ExpFormat(args, kwargs) (line 67)
        ExpFormat_call_result_133135 = invoke(stypy.reporting.localization.Localization(__file__, 67, 13), ExpFormat_133129, *[int_133130, int_133131], **kwargs_133134)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 12), list_133111, ExpFormat_call_result_133135)
        
        # Assigning a type to the variable 'f' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'f', list_133111)
        
        # Assigning a List to a Name (line 68):
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_133136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        str_133137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 15), 'str', '(E10.5)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 14), list_133136, str_133137)
        # Adding element type (line 68)
        str_133138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 26), 'str', '(E12.10)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 14), list_133136, str_133138)
        # Adding element type (line 68)
        str_133139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 38), 'str', '(E12.10E3)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 14), list_133136, str_133139)
        # Adding element type (line 68)
        str_133140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 52), 'str', '(3E10.5)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 14), list_133136, str_133140)
        
        # Assigning a type to the variable 'res' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'res', list_133136)
        
        
        # Call to zip(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'f' (line 70)
        f_133142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 24), 'f', False)
        # Getting the type of 'res' (line 70)
        res_133143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 27), 'res', False)
        # Processing the call keyword arguments (line 70)
        kwargs_133144 = {}
        # Getting the type of 'zip' (line 70)
        zip_133141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 20), 'zip', False)
        # Calling zip(args, kwargs) (line 70)
        zip_call_result_133145 = invoke(stypy.reporting.localization.Localization(__file__, 70, 20), zip_133141, *[f_133142, res_133143], **kwargs_133144)
        
        # Testing the type of a for loop iterable (line 70)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 70, 8), zip_call_result_133145)
        # Getting the type of the for loop variable (line 70)
        for_loop_var_133146 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 70, 8), zip_call_result_133145)
        # Assigning a type to the variable 'i' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 8), for_loop_var_133146))
        # Assigning a type to the variable 'j' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 8), for_loop_var_133146))
        # SSA begins for a for statement (line 70)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_equal(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'i' (line 71)
        i_133148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 25), 'i', False)
        # Obtaining the member 'fortran_format' of a type (line 71)
        fortran_format_133149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 25), i_133148, 'fortran_format')
        # Getting the type of 'j' (line 71)
        j_133150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 43), 'j', False)
        # Processing the call keyword arguments (line 71)
        kwargs_133151 = {}
        # Getting the type of 'assert_equal' (line 71)
        assert_equal_133147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 71)
        assert_equal_call_result_133152 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), assert_equal_133147, *[fortran_format_133149, j_133150], **kwargs_133151)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_to_fortran(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_to_fortran' in the type store
        # Getting the type of 'stypy_return_type' (line 65)
        stypy_return_type_133153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_133153)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_to_fortran'
        return stypy_return_type_133153


    @norecursion
    def test_from_number(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_from_number'
        module_type_store = module_type_store.open_function_context('test_from_number', 73, 4, False)
        # Assigning a type to the variable 'self' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestExpFormat.test_from_number.__dict__.__setitem__('stypy_localization', localization)
        TestExpFormat.test_from_number.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestExpFormat.test_from_number.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestExpFormat.test_from_number.__dict__.__setitem__('stypy_function_name', 'TestExpFormat.test_from_number')
        TestExpFormat.test_from_number.__dict__.__setitem__('stypy_param_names_list', [])
        TestExpFormat.test_from_number.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestExpFormat.test_from_number.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestExpFormat.test_from_number.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestExpFormat.test_from_number.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestExpFormat.test_from_number.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestExpFormat.test_from_number.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpFormat.test_from_number', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_from_number', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_from_number(...)' code ##################

        
        # Assigning a Call to a Name (line 74):
        
        # Call to array(...): (line 74)
        # Processing the call arguments (line 74)
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_133156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        # Adding element type (line 74)
        float_133157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 21), list_133156, float_133157)
        # Adding element type (line 74)
        float_133158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 21), list_133156, float_133158)
        
        # Processing the call keyword arguments (line 74)
        kwargs_133159 = {}
        # Getting the type of 'np' (line 74)
        np_133154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 74)
        array_133155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 12), np_133154, 'array')
        # Calling array(args, kwargs) (line 74)
        array_call_result_133160 = invoke(stypy.reporting.localization.Localization(__file__, 74, 12), array_133155, *[list_133156], **kwargs_133159)
        
        # Assigning a type to the variable 'f' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'f', array_call_result_133160)
        
        # Assigning a List to a Name (line 75):
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_133161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        
        # Call to ExpFormat(...): (line 75)
        # Processing the call arguments (line 75)
        int_133163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'int')
        int_133164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 29), 'int')
        # Processing the call keyword arguments (line 75)
        int_133165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 40), 'int')
        keyword_133166 = int_133165
        kwargs_133167 = {'repeat': keyword_133166}
        # Getting the type of 'ExpFormat' (line 75)
        ExpFormat_133162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'ExpFormat', False)
        # Calling ExpFormat(args, kwargs) (line 75)
        ExpFormat_call_result_133168 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), ExpFormat_133162, *[int_133163, int_133164], **kwargs_133167)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 14), list_133161, ExpFormat_call_result_133168)
        # Adding element type (line 75)
        
        # Call to ExpFormat(...): (line 75)
        # Processing the call arguments (line 75)
        int_133170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 54), 'int')
        int_133171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 58), 'int')
        # Processing the call keyword arguments (line 75)
        int_133172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 69), 'int')
        keyword_133173 = int_133172
        kwargs_133174 = {'repeat': keyword_133173}
        # Getting the type of 'ExpFormat' (line 75)
        ExpFormat_133169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 44), 'ExpFormat', False)
        # Calling ExpFormat(args, kwargs) (line 75)
        ExpFormat_call_result_133175 = invoke(stypy.reporting.localization.Localization(__file__, 75, 44), ExpFormat_133169, *[int_133170, int_133171], **kwargs_133174)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 14), list_133161, ExpFormat_call_result_133175)
        
        # Assigning a type to the variable 'r_f' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'r_f', list_133161)
        
        
        # Call to zip(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'f' (line 76)
        f_133177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'f', False)
        # Getting the type of 'r_f' (line 76)
        r_f_133178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'r_f', False)
        # Processing the call keyword arguments (line 76)
        kwargs_133179 = {}
        # Getting the type of 'zip' (line 76)
        zip_133176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), 'zip', False)
        # Calling zip(args, kwargs) (line 76)
        zip_call_result_133180 = invoke(stypy.reporting.localization.Localization(__file__, 76, 20), zip_133176, *[f_133177, r_f_133178], **kwargs_133179)
        
        # Testing the type of a for loop iterable (line 76)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 76, 8), zip_call_result_133180)
        # Getting the type of the for loop variable (line 76)
        for_loop_var_133181 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 76, 8), zip_call_result_133180)
        # Assigning a type to the variable 'i' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 8), for_loop_var_133181))
        # Assigning a type to the variable 'j' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 8), for_loop_var_133181))
        # SSA begins for a for statement (line 76)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_equal(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Call to from_number(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'i' (line 77)
        i_133185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 47), 'i', False)
        # Processing the call keyword arguments (line 77)
        kwargs_133186 = {}
        # Getting the type of 'ExpFormat' (line 77)
        ExpFormat_133183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), 'ExpFormat', False)
        # Obtaining the member 'from_number' of a type (line 77)
        from_number_133184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 25), ExpFormat_133183, 'from_number')
        # Calling from_number(args, kwargs) (line 77)
        from_number_call_result_133187 = invoke(stypy.reporting.localization.Localization(__file__, 77, 25), from_number_133184, *[i_133185], **kwargs_133186)
        
        # Obtaining the member '__dict__' of a type (line 77)
        dict___133188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 25), from_number_call_result_133187, '__dict__')
        # Getting the type of 'j' (line 77)
        j_133189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 60), 'j', False)
        # Obtaining the member '__dict__' of a type (line 77)
        dict___133190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 60), j_133189, '__dict__')
        # Processing the call keyword arguments (line 77)
        kwargs_133191 = {}
        # Getting the type of 'assert_equal' (line 77)
        assert_equal_133182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 77)
        assert_equal_call_result_133192 = invoke(stypy.reporting.localization.Localization(__file__, 77, 12), assert_equal_133182, *[dict___133188, dict___133190], **kwargs_133191)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_from_number(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_from_number' in the type store
        # Getting the type of 'stypy_return_type' (line 73)
        stypy_return_type_133193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_133193)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_from_number'
        return stypy_return_type_133193


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 64, 0, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestExpFormat.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestExpFormat' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'TestExpFormat', TestExpFormat)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
