
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.text_file.'''
2: import os
3: import unittest
4: from distutils.text_file import TextFile
5: from distutils.tests import support
6: from test.test_support import run_unittest
7: 
8: TEST_DATA = '''# test file
9: 
10: line 3 \\
11: # intervening comment
12:   continues on next line
13: '''
14: 
15: class TextFileTestCase(support.TempdirManager, unittest.TestCase):
16: 
17:     def test_class(self):
18:         # old tests moved from text_file.__main__
19:         # so they are really called by the buildbots
20: 
21:         # result 1: no fancy options
22:         result1 = ['# test file\n', '\n', 'line 3 \\\n',
23:                    '# intervening comment\n',
24:                    '  continues on next line\n']
25: 
26:         # result 2: just strip comments
27:         result2 = ["\n",
28:                    "line 3 \\\n",
29:                    "  continues on next line\n"]
30: 
31:         # result 3: just strip blank lines
32:         result3 = ["# test file\n",
33:                    "line 3 \\\n",
34:                    "# intervening comment\n",
35:                    "  continues on next line\n"]
36: 
37:         # result 4: default, strip comments, blank lines,
38:         # and trailing whitespace
39:         result4 = ["line 3 \\",
40:                    "  continues on next line"]
41: 
42:         # result 5: strip comments and blanks, plus join lines (but don't
43:         # "collapse" joined lines
44:         result5 = ["line 3   continues on next line"]
45: 
46:         # result 6: strip comments and blanks, plus join lines (and
47:         # "collapse" joined lines
48:         result6 = ["line 3 continues on next line"]
49: 
50:         def test_input(count, description, file, expected_result):
51:             result = file.readlines()
52:             self.assertEqual(result, expected_result)
53: 
54:         tmpdir = self.mkdtemp()
55:         filename = os.path.join(tmpdir, "test.txt")
56:         out_file = open(filename, "w")
57:         try:
58:             out_file.write(TEST_DATA)
59:         finally:
60:             out_file.close()
61: 
62:         in_file = TextFile(filename, strip_comments=0, skip_blanks=0,
63:                            lstrip_ws=0, rstrip_ws=0)
64:         try:
65:             test_input(1, "no processing", in_file, result1)
66:         finally:
67:             in_file.close()
68: 
69:         in_file = TextFile(filename, strip_comments=1, skip_blanks=0,
70:                            lstrip_ws=0, rstrip_ws=0)
71:         try:
72:             test_input(2, "strip comments", in_file, result2)
73:         finally:
74:             in_file.close()
75: 
76:         in_file = TextFile(filename, strip_comments=0, skip_blanks=1,
77:                            lstrip_ws=0, rstrip_ws=0)
78:         try:
79:             test_input(3, "strip blanks", in_file, result3)
80:         finally:
81:             in_file.close()
82: 
83:         in_file = TextFile(filename)
84:         try:
85:             test_input(4, "default processing", in_file, result4)
86:         finally:
87:             in_file.close()
88: 
89:         in_file = TextFile(filename, strip_comments=1, skip_blanks=1,
90:                            join_lines=1, rstrip_ws=1)
91:         try:
92:             test_input(5, "join lines without collapsing", in_file, result5)
93:         finally:
94:             in_file.close()
95: 
96:         in_file = TextFile(filename, strip_comments=1, skip_blanks=1,
97:                            join_lines=1, rstrip_ws=1, collapse_join=1)
98:         try:
99:             test_input(6, "join lines with collapsing", in_file, result6)
100:         finally:
101:             in_file.close()
102: 
103: def test_suite():
104:     return unittest.makeSuite(TextFileTestCase)
105: 
106: if __name__ == "__main__":
107:     run_unittest(test_suite())
108: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_44598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.text_file.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import os' statement (line 2)
import os

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import unittest' statement (line 3)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from distutils.text_file import TextFile' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_44599 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.text_file')

if (type(import_44599) is not StypyTypeError):

    if (import_44599 != 'pyd_module'):
        __import__(import_44599)
        sys_modules_44600 = sys.modules[import_44599]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.text_file', sys_modules_44600.module_type_store, module_type_store, ['TextFile'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_44600, sys_modules_44600.module_type_store, module_type_store)
    else:
        from distutils.text_file import TextFile

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.text_file', None, module_type_store, ['TextFile'], [TextFile])

else:
    # Assigning a type to the variable 'distutils.text_file' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.text_file', import_44599)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from distutils.tests import support' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_44601 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.tests')

if (type(import_44601) is not StypyTypeError):

    if (import_44601 != 'pyd_module'):
        __import__(import_44601)
        sys_modules_44602 = sys.modules[import_44601]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.tests', sys_modules_44602.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_44602, sys_modules_44602.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.tests', import_44601)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from test.test_support import run_unittest' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_44603 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'test.test_support')

if (type(import_44603) is not StypyTypeError):

    if (import_44603 != 'pyd_module'):
        __import__(import_44603)
        sys_modules_44604 = sys.modules[import_44603]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'test.test_support', sys_modules_44604.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_44604, sys_modules_44604.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'test.test_support', import_44603)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')


# Assigning a Str to a Name (line 8):
str_44605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, (-1)), 'str', '# test file\n\nline 3 \\\n# intervening comment\n  continues on next line\n')
# Assigning a type to the variable 'TEST_DATA' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'TEST_DATA', str_44605)
# Declaration of the 'TextFileTestCase' class
# Getting the type of 'support' (line 15)
support_44606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 23), 'support')
# Obtaining the member 'TempdirManager' of a type (line 15)
TempdirManager_44607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 23), support_44606, 'TempdirManager')
# Getting the type of 'unittest' (line 15)
unittest_44608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 47), 'unittest')
# Obtaining the member 'TestCase' of a type (line 15)
TestCase_44609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 47), unittest_44608, 'TestCase')

class TextFileTestCase(TempdirManager_44607, TestCase_44609, ):

    @norecursion
    def test_class(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_class'
        module_type_store = module_type_store.open_function_context('test_class', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TextFileTestCase.test_class.__dict__.__setitem__('stypy_localization', localization)
        TextFileTestCase.test_class.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TextFileTestCase.test_class.__dict__.__setitem__('stypy_type_store', module_type_store)
        TextFileTestCase.test_class.__dict__.__setitem__('stypy_function_name', 'TextFileTestCase.test_class')
        TextFileTestCase.test_class.__dict__.__setitem__('stypy_param_names_list', [])
        TextFileTestCase.test_class.__dict__.__setitem__('stypy_varargs_param_name', None)
        TextFileTestCase.test_class.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TextFileTestCase.test_class.__dict__.__setitem__('stypy_call_defaults', defaults)
        TextFileTestCase.test_class.__dict__.__setitem__('stypy_call_varargs', varargs)
        TextFileTestCase.test_class.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TextFileTestCase.test_class.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextFileTestCase.test_class', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_class', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_class(...)' code ##################

        
        # Assigning a List to a Name (line 22):
        
        # Obtaining an instance of the builtin type 'list' (line 22)
        list_44610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 22)
        # Adding element type (line 22)
        str_44611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 19), 'str', '# test file\n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 18), list_44610, str_44611)
        # Adding element type (line 22)
        str_44612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 36), 'str', '\n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 18), list_44610, str_44612)
        # Adding element type (line 22)
        str_44613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 42), 'str', 'line 3 \\\n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 18), list_44610, str_44613)
        # Adding element type (line 22)
        str_44614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 19), 'str', '# intervening comment\n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 18), list_44610, str_44614)
        # Adding element type (line 22)
        str_44615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'str', '  continues on next line\n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 18), list_44610, str_44615)
        
        # Assigning a type to the variable 'result1' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'result1', list_44610)
        
        # Assigning a List to a Name (line 27):
        
        # Obtaining an instance of the builtin type 'list' (line 27)
        list_44616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 27)
        # Adding element type (line 27)
        str_44617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 19), 'str', '\n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 18), list_44616, str_44617)
        # Adding element type (line 27)
        str_44618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 19), 'str', 'line 3 \\\n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 18), list_44616, str_44618)
        # Adding element type (line 27)
        str_44619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 19), 'str', '  continues on next line\n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 18), list_44616, str_44619)
        
        # Assigning a type to the variable 'result2' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'result2', list_44616)
        
        # Assigning a List to a Name (line 32):
        
        # Obtaining an instance of the builtin type 'list' (line 32)
        list_44620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 32)
        # Adding element type (line 32)
        str_44621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 19), 'str', '# test file\n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 18), list_44620, str_44621)
        # Adding element type (line 32)
        str_44622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 19), 'str', 'line 3 \\\n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 18), list_44620, str_44622)
        # Adding element type (line 32)
        str_44623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 19), 'str', '# intervening comment\n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 18), list_44620, str_44623)
        # Adding element type (line 32)
        str_44624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 19), 'str', '  continues on next line\n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 18), list_44620, str_44624)
        
        # Assigning a type to the variable 'result3' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'result3', list_44620)
        
        # Assigning a List to a Name (line 39):
        
        # Obtaining an instance of the builtin type 'list' (line 39)
        list_44625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 39)
        # Adding element type (line 39)
        str_44626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 19), 'str', 'line 3 \\')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 18), list_44625, str_44626)
        # Adding element type (line 39)
        str_44627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 19), 'str', '  continues on next line')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 18), list_44625, str_44627)
        
        # Assigning a type to the variable 'result4' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'result4', list_44625)
        
        # Assigning a List to a Name (line 44):
        
        # Obtaining an instance of the builtin type 'list' (line 44)
        list_44628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 44)
        # Adding element type (line 44)
        str_44629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'str', 'line 3   continues on next line')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 18), list_44628, str_44629)
        
        # Assigning a type to the variable 'result5' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'result5', list_44628)
        
        # Assigning a List to a Name (line 48):
        
        # Obtaining an instance of the builtin type 'list' (line 48)
        list_44630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 48)
        # Adding element type (line 48)
        str_44631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 19), 'str', 'line 3 continues on next line')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 18), list_44630, str_44631)
        
        # Assigning a type to the variable 'result6' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'result6', list_44630)

        @norecursion
        def test_input(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'test_input'
            module_type_store = module_type_store.open_function_context('test_input', 50, 8, False)
            
            # Passed parameters checking function
            test_input.stypy_localization = localization
            test_input.stypy_type_of_self = None
            test_input.stypy_type_store = module_type_store
            test_input.stypy_function_name = 'test_input'
            test_input.stypy_param_names_list = ['count', 'description', 'file', 'expected_result']
            test_input.stypy_varargs_param_name = None
            test_input.stypy_kwargs_param_name = None
            test_input.stypy_call_defaults = defaults
            test_input.stypy_call_varargs = varargs
            test_input.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'test_input', ['count', 'description', 'file', 'expected_result'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'test_input', localization, ['count', 'description', 'file', 'expected_result'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'test_input(...)' code ##################

            
            # Assigning a Call to a Name (line 51):
            
            # Call to readlines(...): (line 51)
            # Processing the call keyword arguments (line 51)
            kwargs_44634 = {}
            # Getting the type of 'file' (line 51)
            file_44632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 21), 'file', False)
            # Obtaining the member 'readlines' of a type (line 51)
            readlines_44633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 21), file_44632, 'readlines')
            # Calling readlines(args, kwargs) (line 51)
            readlines_call_result_44635 = invoke(stypy.reporting.localization.Localization(__file__, 51, 21), readlines_44633, *[], **kwargs_44634)
            
            # Assigning a type to the variable 'result' (line 51)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'result', readlines_call_result_44635)
            
            # Call to assertEqual(...): (line 52)
            # Processing the call arguments (line 52)
            # Getting the type of 'result' (line 52)
            result_44638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 29), 'result', False)
            # Getting the type of 'expected_result' (line 52)
            expected_result_44639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 37), 'expected_result', False)
            # Processing the call keyword arguments (line 52)
            kwargs_44640 = {}
            # Getting the type of 'self' (line 52)
            self_44636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'self', False)
            # Obtaining the member 'assertEqual' of a type (line 52)
            assertEqual_44637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), self_44636, 'assertEqual')
            # Calling assertEqual(args, kwargs) (line 52)
            assertEqual_call_result_44641 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), assertEqual_44637, *[result_44638, expected_result_44639], **kwargs_44640)
            
            
            # ################# End of 'test_input(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'test_input' in the type store
            # Getting the type of 'stypy_return_type' (line 50)
            stypy_return_type_44642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_44642)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'test_input'
            return stypy_return_type_44642

        # Assigning a type to the variable 'test_input' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'test_input', test_input)
        
        # Assigning a Call to a Name (line 54):
        
        # Call to mkdtemp(...): (line 54)
        # Processing the call keyword arguments (line 54)
        kwargs_44645 = {}
        # Getting the type of 'self' (line 54)
        self_44643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 17), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 54)
        mkdtemp_44644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 17), self_44643, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 54)
        mkdtemp_call_result_44646 = invoke(stypy.reporting.localization.Localization(__file__, 54, 17), mkdtemp_44644, *[], **kwargs_44645)
        
        # Assigning a type to the variable 'tmpdir' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tmpdir', mkdtemp_call_result_44646)
        
        # Assigning a Call to a Name (line 55):
        
        # Call to join(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'tmpdir' (line 55)
        tmpdir_44650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 32), 'tmpdir', False)
        str_44651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 40), 'str', 'test.txt')
        # Processing the call keyword arguments (line 55)
        kwargs_44652 = {}
        # Getting the type of 'os' (line 55)
        os_44647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 55)
        path_44648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 19), os_44647, 'path')
        # Obtaining the member 'join' of a type (line 55)
        join_44649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 19), path_44648, 'join')
        # Calling join(args, kwargs) (line 55)
        join_call_result_44653 = invoke(stypy.reporting.localization.Localization(__file__, 55, 19), join_44649, *[tmpdir_44650, str_44651], **kwargs_44652)
        
        # Assigning a type to the variable 'filename' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'filename', join_call_result_44653)
        
        # Assigning a Call to a Name (line 56):
        
        # Call to open(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'filename' (line 56)
        filename_44655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 24), 'filename', False)
        str_44656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 34), 'str', 'w')
        # Processing the call keyword arguments (line 56)
        kwargs_44657 = {}
        # Getting the type of 'open' (line 56)
        open_44654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 19), 'open', False)
        # Calling open(args, kwargs) (line 56)
        open_call_result_44658 = invoke(stypy.reporting.localization.Localization(__file__, 56, 19), open_44654, *[filename_44655, str_44656], **kwargs_44657)
        
        # Assigning a type to the variable 'out_file' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'out_file', open_call_result_44658)
        
        # Try-finally block (line 57)
        
        # Call to write(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'TEST_DATA' (line 58)
        TEST_DATA_44661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 27), 'TEST_DATA', False)
        # Processing the call keyword arguments (line 58)
        kwargs_44662 = {}
        # Getting the type of 'out_file' (line 58)
        out_file_44659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'out_file', False)
        # Obtaining the member 'write' of a type (line 58)
        write_44660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 12), out_file_44659, 'write')
        # Calling write(args, kwargs) (line 58)
        write_call_result_44663 = invoke(stypy.reporting.localization.Localization(__file__, 58, 12), write_44660, *[TEST_DATA_44661], **kwargs_44662)
        
        
        # finally branch of the try-finally block (line 57)
        
        # Call to close(...): (line 60)
        # Processing the call keyword arguments (line 60)
        kwargs_44666 = {}
        # Getting the type of 'out_file' (line 60)
        out_file_44664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'out_file', False)
        # Obtaining the member 'close' of a type (line 60)
        close_44665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), out_file_44664, 'close')
        # Calling close(args, kwargs) (line 60)
        close_call_result_44667 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), close_44665, *[], **kwargs_44666)
        
        
        
        # Assigning a Call to a Name (line 62):
        
        # Call to TextFile(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'filename' (line 62)
        filename_44669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 27), 'filename', False)
        # Processing the call keyword arguments (line 62)
        int_44670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 52), 'int')
        keyword_44671 = int_44670
        int_44672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 67), 'int')
        keyword_44673 = int_44672
        int_44674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 37), 'int')
        keyword_44675 = int_44674
        int_44676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 50), 'int')
        keyword_44677 = int_44676
        kwargs_44678 = {'strip_comments': keyword_44671, 'lstrip_ws': keyword_44675, 'rstrip_ws': keyword_44677, 'skip_blanks': keyword_44673}
        # Getting the type of 'TextFile' (line 62)
        TextFile_44668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 18), 'TextFile', False)
        # Calling TextFile(args, kwargs) (line 62)
        TextFile_call_result_44679 = invoke(stypy.reporting.localization.Localization(__file__, 62, 18), TextFile_44668, *[filename_44669], **kwargs_44678)
        
        # Assigning a type to the variable 'in_file' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'in_file', TextFile_call_result_44679)
        
        # Try-finally block (line 64)
        
        # Call to test_input(...): (line 65)
        # Processing the call arguments (line 65)
        int_44681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 23), 'int')
        str_44682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 26), 'str', 'no processing')
        # Getting the type of 'in_file' (line 65)
        in_file_44683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 43), 'in_file', False)
        # Getting the type of 'result1' (line 65)
        result1_44684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 52), 'result1', False)
        # Processing the call keyword arguments (line 65)
        kwargs_44685 = {}
        # Getting the type of 'test_input' (line 65)
        test_input_44680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'test_input', False)
        # Calling test_input(args, kwargs) (line 65)
        test_input_call_result_44686 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), test_input_44680, *[int_44681, str_44682, in_file_44683, result1_44684], **kwargs_44685)
        
        
        # finally branch of the try-finally block (line 64)
        
        # Call to close(...): (line 67)
        # Processing the call keyword arguments (line 67)
        kwargs_44689 = {}
        # Getting the type of 'in_file' (line 67)
        in_file_44687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'in_file', False)
        # Obtaining the member 'close' of a type (line 67)
        close_44688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 12), in_file_44687, 'close')
        # Calling close(args, kwargs) (line 67)
        close_call_result_44690 = invoke(stypy.reporting.localization.Localization(__file__, 67, 12), close_44688, *[], **kwargs_44689)
        
        
        
        # Assigning a Call to a Name (line 69):
        
        # Call to TextFile(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'filename' (line 69)
        filename_44692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 27), 'filename', False)
        # Processing the call keyword arguments (line 69)
        int_44693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 52), 'int')
        keyword_44694 = int_44693
        int_44695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 67), 'int')
        keyword_44696 = int_44695
        int_44697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 37), 'int')
        keyword_44698 = int_44697
        int_44699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 50), 'int')
        keyword_44700 = int_44699
        kwargs_44701 = {'strip_comments': keyword_44694, 'lstrip_ws': keyword_44698, 'rstrip_ws': keyword_44700, 'skip_blanks': keyword_44696}
        # Getting the type of 'TextFile' (line 69)
        TextFile_44691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 18), 'TextFile', False)
        # Calling TextFile(args, kwargs) (line 69)
        TextFile_call_result_44702 = invoke(stypy.reporting.localization.Localization(__file__, 69, 18), TextFile_44691, *[filename_44692], **kwargs_44701)
        
        # Assigning a type to the variable 'in_file' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'in_file', TextFile_call_result_44702)
        
        # Try-finally block (line 71)
        
        # Call to test_input(...): (line 72)
        # Processing the call arguments (line 72)
        int_44704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 23), 'int')
        str_44705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 26), 'str', 'strip comments')
        # Getting the type of 'in_file' (line 72)
        in_file_44706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 44), 'in_file', False)
        # Getting the type of 'result2' (line 72)
        result2_44707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 53), 'result2', False)
        # Processing the call keyword arguments (line 72)
        kwargs_44708 = {}
        # Getting the type of 'test_input' (line 72)
        test_input_44703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'test_input', False)
        # Calling test_input(args, kwargs) (line 72)
        test_input_call_result_44709 = invoke(stypy.reporting.localization.Localization(__file__, 72, 12), test_input_44703, *[int_44704, str_44705, in_file_44706, result2_44707], **kwargs_44708)
        
        
        # finally branch of the try-finally block (line 71)
        
        # Call to close(...): (line 74)
        # Processing the call keyword arguments (line 74)
        kwargs_44712 = {}
        # Getting the type of 'in_file' (line 74)
        in_file_44710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'in_file', False)
        # Obtaining the member 'close' of a type (line 74)
        close_44711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 12), in_file_44710, 'close')
        # Calling close(args, kwargs) (line 74)
        close_call_result_44713 = invoke(stypy.reporting.localization.Localization(__file__, 74, 12), close_44711, *[], **kwargs_44712)
        
        
        
        # Assigning a Call to a Name (line 76):
        
        # Call to TextFile(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'filename' (line 76)
        filename_44715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'filename', False)
        # Processing the call keyword arguments (line 76)
        int_44716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 52), 'int')
        keyword_44717 = int_44716
        int_44718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 67), 'int')
        keyword_44719 = int_44718
        int_44720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 37), 'int')
        keyword_44721 = int_44720
        int_44722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 50), 'int')
        keyword_44723 = int_44722
        kwargs_44724 = {'strip_comments': keyword_44717, 'lstrip_ws': keyword_44721, 'rstrip_ws': keyword_44723, 'skip_blanks': keyword_44719}
        # Getting the type of 'TextFile' (line 76)
        TextFile_44714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 18), 'TextFile', False)
        # Calling TextFile(args, kwargs) (line 76)
        TextFile_call_result_44725 = invoke(stypy.reporting.localization.Localization(__file__, 76, 18), TextFile_44714, *[filename_44715], **kwargs_44724)
        
        # Assigning a type to the variable 'in_file' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'in_file', TextFile_call_result_44725)
        
        # Try-finally block (line 78)
        
        # Call to test_input(...): (line 79)
        # Processing the call arguments (line 79)
        int_44727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 23), 'int')
        str_44728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 26), 'str', 'strip blanks')
        # Getting the type of 'in_file' (line 79)
        in_file_44729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 42), 'in_file', False)
        # Getting the type of 'result3' (line 79)
        result3_44730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 51), 'result3', False)
        # Processing the call keyword arguments (line 79)
        kwargs_44731 = {}
        # Getting the type of 'test_input' (line 79)
        test_input_44726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'test_input', False)
        # Calling test_input(args, kwargs) (line 79)
        test_input_call_result_44732 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), test_input_44726, *[int_44727, str_44728, in_file_44729, result3_44730], **kwargs_44731)
        
        
        # finally branch of the try-finally block (line 78)
        
        # Call to close(...): (line 81)
        # Processing the call keyword arguments (line 81)
        kwargs_44735 = {}
        # Getting the type of 'in_file' (line 81)
        in_file_44733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'in_file', False)
        # Obtaining the member 'close' of a type (line 81)
        close_44734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), in_file_44733, 'close')
        # Calling close(args, kwargs) (line 81)
        close_call_result_44736 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), close_44734, *[], **kwargs_44735)
        
        
        
        # Assigning a Call to a Name (line 83):
        
        # Call to TextFile(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'filename' (line 83)
        filename_44738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 27), 'filename', False)
        # Processing the call keyword arguments (line 83)
        kwargs_44739 = {}
        # Getting the type of 'TextFile' (line 83)
        TextFile_44737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 18), 'TextFile', False)
        # Calling TextFile(args, kwargs) (line 83)
        TextFile_call_result_44740 = invoke(stypy.reporting.localization.Localization(__file__, 83, 18), TextFile_44737, *[filename_44738], **kwargs_44739)
        
        # Assigning a type to the variable 'in_file' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'in_file', TextFile_call_result_44740)
        
        # Try-finally block (line 84)
        
        # Call to test_input(...): (line 85)
        # Processing the call arguments (line 85)
        int_44742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 23), 'int')
        str_44743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 26), 'str', 'default processing')
        # Getting the type of 'in_file' (line 85)
        in_file_44744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 48), 'in_file', False)
        # Getting the type of 'result4' (line 85)
        result4_44745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 57), 'result4', False)
        # Processing the call keyword arguments (line 85)
        kwargs_44746 = {}
        # Getting the type of 'test_input' (line 85)
        test_input_44741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'test_input', False)
        # Calling test_input(args, kwargs) (line 85)
        test_input_call_result_44747 = invoke(stypy.reporting.localization.Localization(__file__, 85, 12), test_input_44741, *[int_44742, str_44743, in_file_44744, result4_44745], **kwargs_44746)
        
        
        # finally branch of the try-finally block (line 84)
        
        # Call to close(...): (line 87)
        # Processing the call keyword arguments (line 87)
        kwargs_44750 = {}
        # Getting the type of 'in_file' (line 87)
        in_file_44748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'in_file', False)
        # Obtaining the member 'close' of a type (line 87)
        close_44749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 12), in_file_44748, 'close')
        # Calling close(args, kwargs) (line 87)
        close_call_result_44751 = invoke(stypy.reporting.localization.Localization(__file__, 87, 12), close_44749, *[], **kwargs_44750)
        
        
        
        # Assigning a Call to a Name (line 89):
        
        # Call to TextFile(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'filename' (line 89)
        filename_44753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 27), 'filename', False)
        # Processing the call keyword arguments (line 89)
        int_44754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 52), 'int')
        keyword_44755 = int_44754
        int_44756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 67), 'int')
        keyword_44757 = int_44756
        int_44758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 38), 'int')
        keyword_44759 = int_44758
        int_44760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 51), 'int')
        keyword_44761 = int_44760
        kwargs_44762 = {'strip_comments': keyword_44755, 'rstrip_ws': keyword_44761, 'join_lines': keyword_44759, 'skip_blanks': keyword_44757}
        # Getting the type of 'TextFile' (line 89)
        TextFile_44752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 18), 'TextFile', False)
        # Calling TextFile(args, kwargs) (line 89)
        TextFile_call_result_44763 = invoke(stypy.reporting.localization.Localization(__file__, 89, 18), TextFile_44752, *[filename_44753], **kwargs_44762)
        
        # Assigning a type to the variable 'in_file' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'in_file', TextFile_call_result_44763)
        
        # Try-finally block (line 91)
        
        # Call to test_input(...): (line 92)
        # Processing the call arguments (line 92)
        int_44765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 23), 'int')
        str_44766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 26), 'str', 'join lines without collapsing')
        # Getting the type of 'in_file' (line 92)
        in_file_44767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 59), 'in_file', False)
        # Getting the type of 'result5' (line 92)
        result5_44768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 68), 'result5', False)
        # Processing the call keyword arguments (line 92)
        kwargs_44769 = {}
        # Getting the type of 'test_input' (line 92)
        test_input_44764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'test_input', False)
        # Calling test_input(args, kwargs) (line 92)
        test_input_call_result_44770 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), test_input_44764, *[int_44765, str_44766, in_file_44767, result5_44768], **kwargs_44769)
        
        
        # finally branch of the try-finally block (line 91)
        
        # Call to close(...): (line 94)
        # Processing the call keyword arguments (line 94)
        kwargs_44773 = {}
        # Getting the type of 'in_file' (line 94)
        in_file_44771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'in_file', False)
        # Obtaining the member 'close' of a type (line 94)
        close_44772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), in_file_44771, 'close')
        # Calling close(args, kwargs) (line 94)
        close_call_result_44774 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), close_44772, *[], **kwargs_44773)
        
        
        
        # Assigning a Call to a Name (line 96):
        
        # Call to TextFile(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'filename' (line 96)
        filename_44776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 27), 'filename', False)
        # Processing the call keyword arguments (line 96)
        int_44777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 52), 'int')
        keyword_44778 = int_44777
        int_44779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 67), 'int')
        keyword_44780 = int_44779
        int_44781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 38), 'int')
        keyword_44782 = int_44781
        int_44783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 51), 'int')
        keyword_44784 = int_44783
        int_44785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 68), 'int')
        keyword_44786 = int_44785
        kwargs_44787 = {'strip_comments': keyword_44778, 'rstrip_ws': keyword_44784, 'join_lines': keyword_44782, 'skip_blanks': keyword_44780, 'collapse_join': keyword_44786}
        # Getting the type of 'TextFile' (line 96)
        TextFile_44775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 18), 'TextFile', False)
        # Calling TextFile(args, kwargs) (line 96)
        TextFile_call_result_44788 = invoke(stypy.reporting.localization.Localization(__file__, 96, 18), TextFile_44775, *[filename_44776], **kwargs_44787)
        
        # Assigning a type to the variable 'in_file' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'in_file', TextFile_call_result_44788)
        
        # Try-finally block (line 98)
        
        # Call to test_input(...): (line 99)
        # Processing the call arguments (line 99)
        int_44790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 23), 'int')
        str_44791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 26), 'str', 'join lines with collapsing')
        # Getting the type of 'in_file' (line 99)
        in_file_44792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 56), 'in_file', False)
        # Getting the type of 'result6' (line 99)
        result6_44793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 65), 'result6', False)
        # Processing the call keyword arguments (line 99)
        kwargs_44794 = {}
        # Getting the type of 'test_input' (line 99)
        test_input_44789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'test_input', False)
        # Calling test_input(args, kwargs) (line 99)
        test_input_call_result_44795 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), test_input_44789, *[int_44790, str_44791, in_file_44792, result6_44793], **kwargs_44794)
        
        
        # finally branch of the try-finally block (line 98)
        
        # Call to close(...): (line 101)
        # Processing the call keyword arguments (line 101)
        kwargs_44798 = {}
        # Getting the type of 'in_file' (line 101)
        in_file_44796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'in_file', False)
        # Obtaining the member 'close' of a type (line 101)
        close_44797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), in_file_44796, 'close')
        # Calling close(args, kwargs) (line 101)
        close_call_result_44799 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), close_44797, *[], **kwargs_44798)
        
        
        
        # ################# End of 'test_class(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_class' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_44800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44800)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_class'
        return stypy_return_type_44800


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 15, 0, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TextFileTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TextFileTestCase' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'TextFileTestCase', TextFileTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 103, 0, False)
    
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

    
    # Call to makeSuite(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'TextFileTestCase' (line 104)
    TextFileTestCase_44803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 30), 'TextFileTestCase', False)
    # Processing the call keyword arguments (line 104)
    kwargs_44804 = {}
    # Getting the type of 'unittest' (line 104)
    unittest_44801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 104)
    makeSuite_44802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 11), unittest_44801, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 104)
    makeSuite_call_result_44805 = invoke(stypy.reporting.localization.Localization(__file__, 104, 11), makeSuite_44802, *[TextFileTestCase_44803], **kwargs_44804)
    
    # Assigning a type to the variable 'stypy_return_type' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type', makeSuite_call_result_44805)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 103)
    stypy_return_type_44806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_44806)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_44806

# Assigning a type to the variable 'test_suite' (line 103)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 107)
    # Processing the call arguments (line 107)
    
    # Call to test_suite(...): (line 107)
    # Processing the call keyword arguments (line 107)
    kwargs_44809 = {}
    # Getting the type of 'test_suite' (line 107)
    test_suite_44808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 107)
    test_suite_call_result_44810 = invoke(stypy.reporting.localization.Localization(__file__, 107, 17), test_suite_44808, *[], **kwargs_44809)
    
    # Processing the call keyword arguments (line 107)
    kwargs_44811 = {}
    # Getting the type of 'run_unittest' (line 107)
    run_unittest_44807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 107)
    run_unittest_call_result_44812 = invoke(stypy.reporting.localization.Localization(__file__, 107, 4), run_unittest_44807, *[test_suite_call_result_44810], **kwargs_44811)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
