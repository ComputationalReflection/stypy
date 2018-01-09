
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Python unit testing framework, based on Erich Gamma's JUnit and Kent Beck's
3: Smalltalk testing framework.
4: 
5: This module contains the core framework classes that form the basis of
6: specific test cases and suites (TestCase, TestSuite etc.), and also a
7: text-based utility class for running the tests and reporting the results
8:  (TextTestRunner).
9: 
10: Simple usage:
11: 
12:     import unittest
13: 
14:     class IntegerArithmeticTestCase(unittest.TestCase):
15:         def testAdd(self):  ## test method names begin 'test*'
16:             self.assertEqual((1 + 2), 3)
17:             self.assertEqual(0 + 1, 1)
18:         def testMultiply(self):
19:             self.assertEqual((0 * 10), 0)
20:             self.assertEqual((5 * 8), 40)
21: 
22:     if __name__ == '__main__':
23:         unittest.main()
24: 
25: Further information is available in the bundled documentation, and from
26: 
27:   http://docs.python.org/library/unittest.html
28: 
29: Copyright (c) 1999-2003 Steve Purcell
30: Copyright (c) 2003-2010 Python Software Foundation
31: This module is free software, and you may redistribute it and/or modify
32: it under the same terms as Python itself, so long as this copyright message
33: and disclaimer are retained in their original form.
34: 
35: IN NO EVENT SHALL THE AUTHOR BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
36: SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF
37: THIS CODE, EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
38: DAMAGE.
39: 
40: THE AUTHOR SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
41: LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
42: PARTICULAR PURPOSE.  THE CODE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS,
43: AND THERE IS NO OBLIGATION WHATSOEVER TO PROVIDE MAINTENANCE,
44: SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
45: '''
46: 
47: __all__ = ['TestResult', 'TestCase', 'TestSuite',
48:            'TextTestRunner', 'TestLoader', 'FunctionTestCase', 'main',
49:            'defaultTestLoader', 'SkipTest', 'skip', 'skipIf', 'skipUnless',
50:            'expectedFailure', 'TextTestResult', 'installHandler',
51:            'registerResult', 'removeResult', 'removeHandler']
52: 
53: # Expose obsolete functions for backwards compatibility
54: __all__.extend(['getTestCaseNames', 'makeSuite', 'findTestCases'])
55: 
56: __unittest = True
57: 
58: from .result import TestResult
59: from .case import (TestCase, FunctionTestCase, SkipTest, skip, skipIf,
60:                    skipUnless, expectedFailure)
61: from .suite import BaseTestSuite, TestSuite
62: from .loader import (TestLoader, defaultTestLoader, makeSuite, getTestCaseNames,
63:                      findTestCases)
64: from .main import TestProgram, main
65: from .runner import TextTestRunner, TextTestResult
66: from .signals import installHandler, registerResult, removeResult, removeHandler
67: 
68: # deprecated
69: _TextTestResult = TextTestResult
70: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_193046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, (-1)), 'str', '\nPython unit testing framework, based on Erich Gamma\'s JUnit and Kent Beck\'s\nSmalltalk testing framework.\n\nThis module contains the core framework classes that form the basis of\nspecific test cases and suites (TestCase, TestSuite etc.), and also a\ntext-based utility class for running the tests and reporting the results\n (TextTestRunner).\n\nSimple usage:\n\n    import unittest\n\n    class IntegerArithmeticTestCase(unittest.TestCase):\n        def testAdd(self):  ## test method names begin \'test*\'\n            self.assertEqual((1 + 2), 3)\n            self.assertEqual(0 + 1, 1)\n        def testMultiply(self):\n            self.assertEqual((0 * 10), 0)\n            self.assertEqual((5 * 8), 40)\n\n    if __name__ == \'__main__\':\n        unittest.main()\n\nFurther information is available in the bundled documentation, and from\n\n  http://docs.python.org/library/unittest.html\n\nCopyright (c) 1999-2003 Steve Purcell\nCopyright (c) 2003-2010 Python Software Foundation\nThis module is free software, and you may redistribute it and/or modify\nit under the same terms as Python itself, so long as this copyright message\nand disclaimer are retained in their original form.\n\nIN NO EVENT SHALL THE AUTHOR BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,\nSPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF\nTHIS CODE, EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH\nDAMAGE.\n\nTHE AUTHOR SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT\nLIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A\nPARTICULAR PURPOSE.  THE CODE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS,\nAND THERE IS NO OBLIGATION WHATSOEVER TO PROVIDE MAINTENANCE,\nSUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.\n')

# Assigning a List to a Name (line 47):
__all__ = ['TestResult', 'TestCase', 'TestSuite', 'TextTestRunner', 'TestLoader', 'FunctionTestCase', 'main', 'defaultTestLoader', 'SkipTest', 'skip', 'skipIf', 'skipUnless', 'expectedFailure', 'TextTestResult', 'installHandler', 'registerResult', 'removeResult', 'removeHandler']
module_type_store.set_exportable_members(['TestResult', 'TestCase', 'TestSuite', 'TextTestRunner', 'TestLoader', 'FunctionTestCase', 'main', 'defaultTestLoader', 'SkipTest', 'skip', 'skipIf', 'skipUnless', 'expectedFailure', 'TextTestResult', 'installHandler', 'registerResult', 'removeResult', 'removeHandler'])

# Obtaining an instance of the builtin type 'list' (line 47)
list_193047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 47)
# Adding element type (line 47)
str_193048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 11), 'str', 'TestResult')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_193047, str_193048)
# Adding element type (line 47)
str_193049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 25), 'str', 'TestCase')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_193047, str_193049)
# Adding element type (line 47)
str_193050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 37), 'str', 'TestSuite')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_193047, str_193050)
# Adding element type (line 47)
str_193051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 11), 'str', 'TextTestRunner')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_193047, str_193051)
# Adding element type (line 47)
str_193052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 29), 'str', 'TestLoader')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_193047, str_193052)
# Adding element type (line 47)
str_193053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 43), 'str', 'FunctionTestCase')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_193047, str_193053)
# Adding element type (line 47)
str_193054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 63), 'str', 'main')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_193047, str_193054)
# Adding element type (line 47)
str_193055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 11), 'str', 'defaultTestLoader')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_193047, str_193055)
# Adding element type (line 47)
str_193056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 32), 'str', 'SkipTest')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_193047, str_193056)
# Adding element type (line 47)
str_193057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 44), 'str', 'skip')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_193047, str_193057)
# Adding element type (line 47)
str_193058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 52), 'str', 'skipIf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_193047, str_193058)
# Adding element type (line 47)
str_193059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 62), 'str', 'skipUnless')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_193047, str_193059)
# Adding element type (line 47)
str_193060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 11), 'str', 'expectedFailure')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_193047, str_193060)
# Adding element type (line 47)
str_193061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 30), 'str', 'TextTestResult')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_193047, str_193061)
# Adding element type (line 47)
str_193062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 48), 'str', 'installHandler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_193047, str_193062)
# Adding element type (line 47)
str_193063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 11), 'str', 'registerResult')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_193047, str_193063)
# Adding element type (line 47)
str_193064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 29), 'str', 'removeResult')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_193047, str_193064)
# Adding element type (line 47)
str_193065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 45), 'str', 'removeHandler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_193047, str_193065)

# Assigning a type to the variable '__all__' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), '__all__', list_193047)

# Call to extend(...): (line 54)
# Processing the call arguments (line 54)

# Obtaining an instance of the builtin type 'list' (line 54)
list_193068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 54)
# Adding element type (line 54)
str_193069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 16), 'str', 'getTestCaseNames')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 15), list_193068, str_193069)
# Adding element type (line 54)
str_193070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 36), 'str', 'makeSuite')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 15), list_193068, str_193070)
# Adding element type (line 54)
str_193071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 49), 'str', 'findTestCases')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 15), list_193068, str_193071)

# Processing the call keyword arguments (line 54)
kwargs_193072 = {}
# Getting the type of '__all__' (line 54)
all___193066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), '__all__', False)
# Obtaining the member 'extend' of a type (line 54)
extend_193067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 0), all___193066, 'extend')
# Calling extend(args, kwargs) (line 54)
extend_call_result_193073 = invoke(stypy.reporting.localization.Localization(__file__, 54, 0), extend_193067, *[list_193068], **kwargs_193072)


# Assigning a Name to a Name (line 56):
# Getting the type of 'True' (line 56)
True_193074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 13), 'True')
# Assigning a type to the variable '__unittest' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), '__unittest', True_193074)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 58, 0))

# 'from unittest.result import TestResult' statement (line 58)
update_path_to_current_file_folder('C:/Python27/lib/unittest/')
import_193075 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 58, 0), 'unittest.result')

if (type(import_193075) is not StypyTypeError):

    if (import_193075 != 'pyd_module'):
        __import__(import_193075)
        sys_modules_193076 = sys.modules[import_193075]
        import_from_module(stypy.reporting.localization.Localization(__file__, 58, 0), 'unittest.result', sys_modules_193076.module_type_store, module_type_store, ['TestResult'])
        nest_module(stypy.reporting.localization.Localization(__file__, 58, 0), __file__, sys_modules_193076, sys_modules_193076.module_type_store, module_type_store)
    else:
        from unittest.result import TestResult

        import_from_module(stypy.reporting.localization.Localization(__file__, 58, 0), 'unittest.result', None, module_type_store, ['TestResult'], [TestResult])

else:
    # Assigning a type to the variable 'unittest.result' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'unittest.result', import_193075)

remove_current_file_folder_from_path('C:/Python27/lib/unittest/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 59, 0))

# 'from unittest.case import TestCase, FunctionTestCase, SkipTest, skip, skipIf, skipUnless, expectedFailure' statement (line 59)
update_path_to_current_file_folder('C:/Python27/lib/unittest/')
import_193077 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 59, 0), 'unittest.case')

if (type(import_193077) is not StypyTypeError):

    if (import_193077 != 'pyd_module'):
        __import__(import_193077)
        sys_modules_193078 = sys.modules[import_193077]
        import_from_module(stypy.reporting.localization.Localization(__file__, 59, 0), 'unittest.case', sys_modules_193078.module_type_store, module_type_store, ['TestCase', 'FunctionTestCase', 'SkipTest', 'skip', 'skipIf', 'skipUnless', 'expectedFailure'])
        nest_module(stypy.reporting.localization.Localization(__file__, 59, 0), __file__, sys_modules_193078, sys_modules_193078.module_type_store, module_type_store)
    else:
        from unittest.case import TestCase, FunctionTestCase, SkipTest, skip, skipIf, skipUnless, expectedFailure

        import_from_module(stypy.reporting.localization.Localization(__file__, 59, 0), 'unittest.case', None, module_type_store, ['TestCase', 'FunctionTestCase', 'SkipTest', 'skip', 'skipIf', 'skipUnless', 'expectedFailure'], [TestCase, FunctionTestCase, SkipTest, skip, skipIf, skipUnless, expectedFailure])

else:
    # Assigning a type to the variable 'unittest.case' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'unittest.case', import_193077)

remove_current_file_folder_from_path('C:/Python27/lib/unittest/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 61, 0))

# 'from unittest.suite import BaseTestSuite, TestSuite' statement (line 61)
update_path_to_current_file_folder('C:/Python27/lib/unittest/')
import_193079 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 61, 0), 'unittest.suite')

if (type(import_193079) is not StypyTypeError):

    if (import_193079 != 'pyd_module'):
        __import__(import_193079)
        sys_modules_193080 = sys.modules[import_193079]
        import_from_module(stypy.reporting.localization.Localization(__file__, 61, 0), 'unittest.suite', sys_modules_193080.module_type_store, module_type_store, ['BaseTestSuite', 'TestSuite'])
        nest_module(stypy.reporting.localization.Localization(__file__, 61, 0), __file__, sys_modules_193080, sys_modules_193080.module_type_store, module_type_store)
    else:
        from unittest.suite import BaseTestSuite, TestSuite

        import_from_module(stypy.reporting.localization.Localization(__file__, 61, 0), 'unittest.suite', None, module_type_store, ['BaseTestSuite', 'TestSuite'], [BaseTestSuite, TestSuite])

else:
    # Assigning a type to the variable 'unittest.suite' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'unittest.suite', import_193079)

remove_current_file_folder_from_path('C:/Python27/lib/unittest/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 62, 0))

# 'from unittest.loader import TestLoader, defaultTestLoader, makeSuite, getTestCaseNames, findTestCases' statement (line 62)
update_path_to_current_file_folder('C:/Python27/lib/unittest/')
import_193081 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 62, 0), 'unittest.loader')

if (type(import_193081) is not StypyTypeError):

    if (import_193081 != 'pyd_module'):
        __import__(import_193081)
        sys_modules_193082 = sys.modules[import_193081]
        import_from_module(stypy.reporting.localization.Localization(__file__, 62, 0), 'unittest.loader', sys_modules_193082.module_type_store, module_type_store, ['TestLoader', 'defaultTestLoader', 'makeSuite', 'getTestCaseNames', 'findTestCases'])
        nest_module(stypy.reporting.localization.Localization(__file__, 62, 0), __file__, sys_modules_193082, sys_modules_193082.module_type_store, module_type_store)
    else:
        from unittest.loader import TestLoader, defaultTestLoader, makeSuite, getTestCaseNames, findTestCases

        import_from_module(stypy.reporting.localization.Localization(__file__, 62, 0), 'unittest.loader', None, module_type_store, ['TestLoader', 'defaultTestLoader', 'makeSuite', 'getTestCaseNames', 'findTestCases'], [TestLoader, defaultTestLoader, makeSuite, getTestCaseNames, findTestCases])

else:
    # Assigning a type to the variable 'unittest.loader' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'unittest.loader', import_193081)

remove_current_file_folder_from_path('C:/Python27/lib/unittest/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 64, 0))

# 'from unittest.main import TestProgram, main' statement (line 64)
update_path_to_current_file_folder('C:/Python27/lib/unittest/')
import_193083 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 64, 0), 'unittest.main')

if (type(import_193083) is not StypyTypeError):

    if (import_193083 != 'pyd_module'):
        __import__(import_193083)
        sys_modules_193084 = sys.modules[import_193083]
        import_from_module(stypy.reporting.localization.Localization(__file__, 64, 0), 'unittest.main', sys_modules_193084.module_type_store, module_type_store, ['TestProgram', 'main'])
        nest_module(stypy.reporting.localization.Localization(__file__, 64, 0), __file__, sys_modules_193084, sys_modules_193084.module_type_store, module_type_store)
    else:
        from unittest.main import TestProgram, main

        import_from_module(stypy.reporting.localization.Localization(__file__, 64, 0), 'unittest.main', None, module_type_store, ['TestProgram', 'main'], [TestProgram, main])

else:
    # Assigning a type to the variable 'unittest.main' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'unittest.main', import_193083)

remove_current_file_folder_from_path('C:/Python27/lib/unittest/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 65, 0))

# 'from unittest.runner import TextTestRunner, TextTestResult' statement (line 65)
update_path_to_current_file_folder('C:/Python27/lib/unittest/')
import_193085 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 65, 0), 'unittest.runner')

if (type(import_193085) is not StypyTypeError):

    if (import_193085 != 'pyd_module'):
        __import__(import_193085)
        sys_modules_193086 = sys.modules[import_193085]
        import_from_module(stypy.reporting.localization.Localization(__file__, 65, 0), 'unittest.runner', sys_modules_193086.module_type_store, module_type_store, ['TextTestRunner', 'TextTestResult'])
        nest_module(stypy.reporting.localization.Localization(__file__, 65, 0), __file__, sys_modules_193086, sys_modules_193086.module_type_store, module_type_store)
    else:
        from unittest.runner import TextTestRunner, TextTestResult

        import_from_module(stypy.reporting.localization.Localization(__file__, 65, 0), 'unittest.runner', None, module_type_store, ['TextTestRunner', 'TextTestResult'], [TextTestRunner, TextTestResult])

else:
    # Assigning a type to the variable 'unittest.runner' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'unittest.runner', import_193085)

remove_current_file_folder_from_path('C:/Python27/lib/unittest/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 66, 0))

# 'from unittest.signals import installHandler, registerResult, removeResult, removeHandler' statement (line 66)
update_path_to_current_file_folder('C:/Python27/lib/unittest/')
import_193087 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'unittest.signals')

if (type(import_193087) is not StypyTypeError):

    if (import_193087 != 'pyd_module'):
        __import__(import_193087)
        sys_modules_193088 = sys.modules[import_193087]
        import_from_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'unittest.signals', sys_modules_193088.module_type_store, module_type_store, ['installHandler', 'registerResult', 'removeResult', 'removeHandler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 66, 0), __file__, sys_modules_193088, sys_modules_193088.module_type_store, module_type_store)
    else:
        from unittest.signals import installHandler, registerResult, removeResult, removeHandler

        import_from_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'unittest.signals', None, module_type_store, ['installHandler', 'registerResult', 'removeResult', 'removeHandler'], [installHandler, registerResult, removeResult, removeHandler])

else:
    # Assigning a type to the variable 'unittest.signals' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'unittest.signals', import_193087)

remove_current_file_folder_from_path('C:/Python27/lib/unittest/')


# Assigning a Name to a Name (line 69):
# Getting the type of 'TextTestResult' (line 69)
TextTestResult_193089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 18), 'TextTestResult')
# Assigning a type to the variable '_TextTestResult' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), '_TextTestResult', TextTestResult_193089)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
