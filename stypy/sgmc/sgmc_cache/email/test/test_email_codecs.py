
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2002-2006 Python Software Foundation
2: # Contact: email-sig@python.org
3: # email package unit tests for (optional) Asian codecs
4: 
5: import unittest
6: from test.test_support import run_unittest
7: 
8: from email.test.test_email import TestEmailBase
9: from email.charset import Charset
10: from email.header import Header, decode_header
11: from email.message import Message
12: 
13: # We're compatible with Python 2.3, but it doesn't have the built-in Asian
14: # codecs, so we have to skip all these tests.
15: try:
16:     unicode('foo', 'euc-jp')
17: except LookupError:
18:     raise unittest.SkipTest
19: 
20: 
21: 
22: class TestEmailAsianCodecs(TestEmailBase):
23:     def test_japanese_codecs(self):
24:         eq = self.ndiffAssertEqual
25:         j = Charset("euc-jp")
26:         g = Charset("iso-8859-1")
27:         h = Header("Hello World!")
28:         jhello = '\xa5\xcf\xa5\xed\xa1\xbc\xa5\xef\xa1\xbc\xa5\xeb\xa5\xc9\xa1\xaa'
29:         ghello = 'Gr\xfc\xdf Gott!'
30:         h.append(jhello, j)
31:         h.append(ghello, g)
32:         # BAW: This used to -- and maybe should -- fold the two iso-8859-1
33:         # chunks into a single encoded word.  However it doesn't violate the
34:         # standard to have them as two encoded chunks and maybe it's
35:         # reasonable <wink> for each .append() call to result in a separate
36:         # encoded word.
37:         eq(h.encode(), '''\
38: Hello World! =?iso-2022-jp?b?GyRCJU8lbSE8JW8hPCVrJUkhKhsoQg==?=
39:  =?iso-8859-1?q?Gr=FC=DF?= =?iso-8859-1?q?_Gott!?=''')
40:         eq(decode_header(h.encode()),
41:            [('Hello World!', None),
42:             ('\x1b$B%O%m!<%o!<%k%I!*\x1b(B', 'iso-2022-jp'),
43:             ('Gr\xfc\xdf Gott!', 'iso-8859-1')])
44:         long = 'test-ja \xa4\xd8\xc5\xea\xb9\xc6\xa4\xb5\xa4\xec\xa4\xbf\xa5\xe1\xa1\xbc\xa5\xeb\xa4\xcf\xbb\xca\xb2\xf1\xbc\xd4\xa4\xce\xbe\xb5\xc7\xa7\xa4\xf2\xc2\xd4\xa4\xc3\xa4\xc6\xa4\xa4\xa4\xde\xa4\xb9'
45:         h = Header(long, j, header_name="Subject")
46:         # test a very long header
47:         enc = h.encode()
48:         # TK: splitting point may differ by codec design and/or Header encoding
49:         eq(enc , '''\
50: =?iso-2022-jp?b?dGVzdC1qYSAbJEIkWEVqOUYkNSRsJD8lYSE8JWskTztKGyhC?=
51:  =?iso-2022-jp?b?GyRCMnE8VCROPjVHJyRyQlQkQyRGJCQkXiQ5GyhC?=''')
52:         # TK: full decode comparison
53:         eq(h.__unicode__().encode('euc-jp'), long)
54: 
55:     def test_payload_encoding(self):
56:         jhello = '\xa5\xcf\xa5\xed\xa1\xbc\xa5\xef\xa1\xbc\xa5\xeb\xa5\xc9\xa1\xaa'
57:         jcode  = 'euc-jp'
58:         msg = Message()
59:         msg.set_payload(jhello, jcode)
60:         ustr = unicode(msg.get_payload(), msg.get_content_charset())
61:         self.assertEqual(jhello, ustr.encode(jcode))
62: 
63: 
64: 
65: def suite():
66:     suite = unittest.TestSuite()
67:     suite.addTest(unittest.makeSuite(TestEmailAsianCodecs))
68:     return suite
69: 
70: 
71: def test_main():
72:     run_unittest(TestEmailAsianCodecs)
73: 
74: 
75: 
76: if __name__ == '__main__':
77:     unittest.main(defaultTest='suite')
78: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import unittest' statement (line 5)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from test.test_support import run_unittest' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/email/test/')
import_23131 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'test.test_support')

if (type(import_23131) is not StypyTypeError):

    if (import_23131 != 'pyd_module'):
        __import__(import_23131)
        sys_modules_23132 = sys.modules[import_23131]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'test.test_support', sys_modules_23132.module_type_store, module_type_store, ['run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_23132, sys_modules_23132.module_type_store, module_type_store)
    else:
        from test.test_support import run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'test.test_support', None, module_type_store, ['run_unittest'], [run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'test.test_support', import_23131)

remove_current_file_folder_from_path('C:/Python27/lib/email/test/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from email.test.test_email import TestEmailBase' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/email/test/')
import_23133 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'email.test.test_email')

if (type(import_23133) is not StypyTypeError):

    if (import_23133 != 'pyd_module'):
        __import__(import_23133)
        sys_modules_23134 = sys.modules[import_23133]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'email.test.test_email', sys_modules_23134.module_type_store, module_type_store, ['TestEmailBase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_23134, sys_modules_23134.module_type_store, module_type_store)
    else:
        from email.test.test_email import TestEmailBase

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'email.test.test_email', None, module_type_store, ['TestEmailBase'], [TestEmailBase])

else:
    # Assigning a type to the variable 'email.test.test_email' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'email.test.test_email', import_23133)

remove_current_file_folder_from_path('C:/Python27/lib/email/test/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from email.charset import Charset' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/email/test/')
import_23135 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'email.charset')

if (type(import_23135) is not StypyTypeError):

    if (import_23135 != 'pyd_module'):
        __import__(import_23135)
        sys_modules_23136 = sys.modules[import_23135]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'email.charset', sys_modules_23136.module_type_store, module_type_store, ['Charset'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_23136, sys_modules_23136.module_type_store, module_type_store)
    else:
        from email.charset import Charset

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'email.charset', None, module_type_store, ['Charset'], [Charset])

else:
    # Assigning a type to the variable 'email.charset' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'email.charset', import_23135)

remove_current_file_folder_from_path('C:/Python27/lib/email/test/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from email.header import Header, decode_header' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/email/test/')
import_23137 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'email.header')

if (type(import_23137) is not StypyTypeError):

    if (import_23137 != 'pyd_module'):
        __import__(import_23137)
        sys_modules_23138 = sys.modules[import_23137]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'email.header', sys_modules_23138.module_type_store, module_type_store, ['Header', 'decode_header'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_23138, sys_modules_23138.module_type_store, module_type_store)
    else:
        from email.header import Header, decode_header

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'email.header', None, module_type_store, ['Header', 'decode_header'], [Header, decode_header])

else:
    # Assigning a type to the variable 'email.header' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'email.header', import_23137)

remove_current_file_folder_from_path('C:/Python27/lib/email/test/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from email.message import Message' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/email/test/')
import_23139 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'email.message')

if (type(import_23139) is not StypyTypeError):

    if (import_23139 != 'pyd_module'):
        __import__(import_23139)
        sys_modules_23140 = sys.modules[import_23139]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'email.message', sys_modules_23140.module_type_store, module_type_store, ['Message'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_23140, sys_modules_23140.module_type_store, module_type_store)
    else:
        from email.message import Message

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'email.message', None, module_type_store, ['Message'], [Message])

else:
    # Assigning a type to the variable 'email.message' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'email.message', import_23139)

remove_current_file_folder_from_path('C:/Python27/lib/email/test/')



# SSA begins for try-except statement (line 15)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Call to unicode(...): (line 16)
# Processing the call arguments (line 16)
str_23142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 12), 'str', 'foo')
str_23143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 19), 'str', 'euc-jp')
# Processing the call keyword arguments (line 16)
kwargs_23144 = {}
# Getting the type of 'unicode' (line 16)
unicode_23141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'unicode', False)
# Calling unicode(args, kwargs) (line 16)
unicode_call_result_23145 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), unicode_23141, *[str_23142, str_23143], **kwargs_23144)

# SSA branch for the except part of a try statement (line 15)
# SSA branch for the except 'LookupError' branch of a try statement (line 15)
module_type_store.open_ssa_branch('except')
# Getting the type of 'unittest' (line 18)
unittest_23146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'unittest')
# Obtaining the member 'SkipTest' of a type (line 18)
SkipTest_23147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 10), unittest_23146, 'SkipTest')
ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 18, 4), SkipTest_23147, 'raise parameter', BaseException)
# SSA join for try-except statement (line 15)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'TestEmailAsianCodecs' class
# Getting the type of 'TestEmailBase' (line 22)
TestEmailBase_23148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 27), 'TestEmailBase')

class TestEmailAsianCodecs(TestEmailBase_23148, ):

    @norecursion
    def test_japanese_codecs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_japanese_codecs'
        module_type_store = module_type_store.open_function_context('test_japanese_codecs', 23, 4, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestEmailAsianCodecs.test_japanese_codecs.__dict__.__setitem__('stypy_localization', localization)
        TestEmailAsianCodecs.test_japanese_codecs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestEmailAsianCodecs.test_japanese_codecs.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestEmailAsianCodecs.test_japanese_codecs.__dict__.__setitem__('stypy_function_name', 'TestEmailAsianCodecs.test_japanese_codecs')
        TestEmailAsianCodecs.test_japanese_codecs.__dict__.__setitem__('stypy_param_names_list', [])
        TestEmailAsianCodecs.test_japanese_codecs.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestEmailAsianCodecs.test_japanese_codecs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestEmailAsianCodecs.test_japanese_codecs.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestEmailAsianCodecs.test_japanese_codecs.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestEmailAsianCodecs.test_japanese_codecs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestEmailAsianCodecs.test_japanese_codecs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestEmailAsianCodecs.test_japanese_codecs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_japanese_codecs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_japanese_codecs(...)' code ##################

        
        # Assigning a Attribute to a Name (line 24):
        # Getting the type of 'self' (line 24)
        self_23149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 13), 'self')
        # Obtaining the member 'ndiffAssertEqual' of a type (line 24)
        ndiffAssertEqual_23150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 13), self_23149, 'ndiffAssertEqual')
        # Assigning a type to the variable 'eq' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'eq', ndiffAssertEqual_23150)
        
        # Assigning a Call to a Name (line 25):
        
        # Call to Charset(...): (line 25)
        # Processing the call arguments (line 25)
        str_23152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 20), 'str', 'euc-jp')
        # Processing the call keyword arguments (line 25)
        kwargs_23153 = {}
        # Getting the type of 'Charset' (line 25)
        Charset_23151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'Charset', False)
        # Calling Charset(args, kwargs) (line 25)
        Charset_call_result_23154 = invoke(stypy.reporting.localization.Localization(__file__, 25, 12), Charset_23151, *[str_23152], **kwargs_23153)
        
        # Assigning a type to the variable 'j' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'j', Charset_call_result_23154)
        
        # Assigning a Call to a Name (line 26):
        
        # Call to Charset(...): (line 26)
        # Processing the call arguments (line 26)
        str_23156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 20), 'str', 'iso-8859-1')
        # Processing the call keyword arguments (line 26)
        kwargs_23157 = {}
        # Getting the type of 'Charset' (line 26)
        Charset_23155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'Charset', False)
        # Calling Charset(args, kwargs) (line 26)
        Charset_call_result_23158 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), Charset_23155, *[str_23156], **kwargs_23157)
        
        # Assigning a type to the variable 'g' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'g', Charset_call_result_23158)
        
        # Assigning a Call to a Name (line 27):
        
        # Call to Header(...): (line 27)
        # Processing the call arguments (line 27)
        str_23160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 19), 'str', 'Hello World!')
        # Processing the call keyword arguments (line 27)
        kwargs_23161 = {}
        # Getting the type of 'Header' (line 27)
        Header_23159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'Header', False)
        # Calling Header(args, kwargs) (line 27)
        Header_call_result_23162 = invoke(stypy.reporting.localization.Localization(__file__, 27, 12), Header_23159, *[str_23160], **kwargs_23161)
        
        # Assigning a type to the variable 'h' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'h', Header_call_result_23162)
        
        # Assigning a Str to a Name (line 28):
        str_23163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 17), 'str', '\xa5\xcf\xa5\xed\xa1\xbc\xa5\xef\xa1\xbc\xa5\xeb\xa5\xc9\xa1\xaa')
        # Assigning a type to the variable 'jhello' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'jhello', str_23163)
        
        # Assigning a Str to a Name (line 29):
        str_23164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 17), 'str', 'Gr\xfc\xdf Gott!')
        # Assigning a type to the variable 'ghello' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'ghello', str_23164)
        
        # Call to append(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'jhello' (line 30)
        jhello_23167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 17), 'jhello', False)
        # Getting the type of 'j' (line 30)
        j_23168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 25), 'j', False)
        # Processing the call keyword arguments (line 30)
        kwargs_23169 = {}
        # Getting the type of 'h' (line 30)
        h_23165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'h', False)
        # Obtaining the member 'append' of a type (line 30)
        append_23166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), h_23165, 'append')
        # Calling append(args, kwargs) (line 30)
        append_call_result_23170 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), append_23166, *[jhello_23167, j_23168], **kwargs_23169)
        
        
        # Call to append(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'ghello' (line 31)
        ghello_23173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'ghello', False)
        # Getting the type of 'g' (line 31)
        g_23174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 25), 'g', False)
        # Processing the call keyword arguments (line 31)
        kwargs_23175 = {}
        # Getting the type of 'h' (line 31)
        h_23171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'h', False)
        # Obtaining the member 'append' of a type (line 31)
        append_23172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), h_23171, 'append')
        # Calling append(args, kwargs) (line 31)
        append_call_result_23176 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), append_23172, *[ghello_23173, g_23174], **kwargs_23175)
        
        
        # Call to eq(...): (line 37)
        # Processing the call arguments (line 37)
        
        # Call to encode(...): (line 37)
        # Processing the call keyword arguments (line 37)
        kwargs_23180 = {}
        # Getting the type of 'h' (line 37)
        h_23178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'h', False)
        # Obtaining the member 'encode' of a type (line 37)
        encode_23179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 11), h_23178, 'encode')
        # Calling encode(args, kwargs) (line 37)
        encode_call_result_23181 = invoke(stypy.reporting.localization.Localization(__file__, 37, 11), encode_23179, *[], **kwargs_23180)
        
        str_23182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, (-1)), 'str', 'Hello World! =?iso-2022-jp?b?GyRCJU8lbSE8JW8hPCVrJUkhKhsoQg==?=\n =?iso-8859-1?q?Gr=FC=DF?= =?iso-8859-1?q?_Gott!?=')
        # Processing the call keyword arguments (line 37)
        kwargs_23183 = {}
        # Getting the type of 'eq' (line 37)
        eq_23177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'eq', False)
        # Calling eq(args, kwargs) (line 37)
        eq_call_result_23184 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), eq_23177, *[encode_call_result_23181, str_23182], **kwargs_23183)
        
        
        # Call to eq(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Call to decode_header(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Call to encode(...): (line 40)
        # Processing the call keyword arguments (line 40)
        kwargs_23189 = {}
        # Getting the type of 'h' (line 40)
        h_23187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 25), 'h', False)
        # Obtaining the member 'encode' of a type (line 40)
        encode_23188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 25), h_23187, 'encode')
        # Calling encode(args, kwargs) (line 40)
        encode_call_result_23190 = invoke(stypy.reporting.localization.Localization(__file__, 40, 25), encode_23188, *[], **kwargs_23189)
        
        # Processing the call keyword arguments (line 40)
        kwargs_23191 = {}
        # Getting the type of 'decode_header' (line 40)
        decode_header_23186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'decode_header', False)
        # Calling decode_header(args, kwargs) (line 40)
        decode_header_call_result_23192 = invoke(stypy.reporting.localization.Localization(__file__, 40, 11), decode_header_23186, *[encode_call_result_23190], **kwargs_23191)
        
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_23193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 11), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        # Adding element type (line 41)
        
        # Obtaining an instance of the builtin type 'tuple' (line 41)
        tuple_23194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 41)
        # Adding element type (line 41)
        str_23195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 13), 'str', 'Hello World!')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 13), tuple_23194, str_23195)
        # Adding element type (line 41)
        # Getting the type of 'None' (line 41)
        None_23196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 29), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 13), tuple_23194, None_23196)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 11), list_23193, tuple_23194)
        # Adding element type (line 41)
        
        # Obtaining an instance of the builtin type 'tuple' (line 42)
        tuple_23197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 42)
        # Adding element type (line 42)
        str_23198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 13), 'str', '\x1b$B%O%m!<%o!<%k%I!*\x1b(B')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 13), tuple_23197, str_23198)
        # Adding element type (line 42)
        str_23199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 45), 'str', 'iso-2022-jp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 13), tuple_23197, str_23199)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 11), list_23193, tuple_23197)
        # Adding element type (line 41)
        
        # Obtaining an instance of the builtin type 'tuple' (line 43)
        tuple_23200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 43)
        # Adding element type (line 43)
        str_23201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 13), 'str', 'Gr\xfc\xdf Gott!')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 13), tuple_23200, str_23201)
        # Adding element type (line 43)
        str_23202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 33), 'str', 'iso-8859-1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 13), tuple_23200, str_23202)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 11), list_23193, tuple_23200)
        
        # Processing the call keyword arguments (line 40)
        kwargs_23203 = {}
        # Getting the type of 'eq' (line 40)
        eq_23185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'eq', False)
        # Calling eq(args, kwargs) (line 40)
        eq_call_result_23204 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), eq_23185, *[decode_header_call_result_23192, list_23193], **kwargs_23203)
        
        
        # Assigning a Str to a Name (line 44):
        str_23205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 15), 'str', 'test-ja \xa4\xd8\xc5\xea\xb9\xc6\xa4\xb5\xa4\xec\xa4\xbf\xa5\xe1\xa1\xbc\xa5\xeb\xa4\xcf\xbb\xca\xb2\xf1\xbc\xd4\xa4\xce\xbe\xb5\xc7\xa7\xa4\xf2\xc2\xd4\xa4\xc3\xa4\xc6\xa4\xa4\xa4\xde\xa4\xb9')
        # Assigning a type to the variable 'long' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'long', str_23205)
        
        # Assigning a Call to a Name (line 45):
        
        # Call to Header(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'long' (line 45)
        long_23207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'long', False)
        # Getting the type of 'j' (line 45)
        j_23208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), 'j', False)
        # Processing the call keyword arguments (line 45)
        str_23209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 40), 'str', 'Subject')
        keyword_23210 = str_23209
        kwargs_23211 = {'header_name': keyword_23210}
        # Getting the type of 'Header' (line 45)
        Header_23206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'Header', False)
        # Calling Header(args, kwargs) (line 45)
        Header_call_result_23212 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), Header_23206, *[long_23207, j_23208], **kwargs_23211)
        
        # Assigning a type to the variable 'h' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'h', Header_call_result_23212)
        
        # Assigning a Call to a Name (line 47):
        
        # Call to encode(...): (line 47)
        # Processing the call keyword arguments (line 47)
        kwargs_23215 = {}
        # Getting the type of 'h' (line 47)
        h_23213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 14), 'h', False)
        # Obtaining the member 'encode' of a type (line 47)
        encode_23214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 14), h_23213, 'encode')
        # Calling encode(args, kwargs) (line 47)
        encode_call_result_23216 = invoke(stypy.reporting.localization.Localization(__file__, 47, 14), encode_23214, *[], **kwargs_23215)
        
        # Assigning a type to the variable 'enc' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'enc', encode_call_result_23216)
        
        # Call to eq(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'enc' (line 49)
        enc_23218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'enc', False)
        str_23219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, (-1)), 'str', '=?iso-2022-jp?b?dGVzdC1qYSAbJEIkWEVqOUYkNSRsJD8lYSE8JWskTztKGyhC?=\n =?iso-2022-jp?b?GyRCMnE8VCROPjVHJyRyQlQkQyRGJCQkXiQ5GyhC?=')
        # Processing the call keyword arguments (line 49)
        kwargs_23220 = {}
        # Getting the type of 'eq' (line 49)
        eq_23217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'eq', False)
        # Calling eq(args, kwargs) (line 49)
        eq_call_result_23221 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), eq_23217, *[enc_23218, str_23219], **kwargs_23220)
        
        
        # Call to eq(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Call to encode(...): (line 53)
        # Processing the call arguments (line 53)
        str_23228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 34), 'str', 'euc-jp')
        # Processing the call keyword arguments (line 53)
        kwargs_23229 = {}
        
        # Call to __unicode__(...): (line 53)
        # Processing the call keyword arguments (line 53)
        kwargs_23225 = {}
        # Getting the type of 'h' (line 53)
        h_23223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'h', False)
        # Obtaining the member '__unicode__' of a type (line 53)
        unicode___23224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 11), h_23223, '__unicode__')
        # Calling __unicode__(args, kwargs) (line 53)
        unicode___call_result_23226 = invoke(stypy.reporting.localization.Localization(__file__, 53, 11), unicode___23224, *[], **kwargs_23225)
        
        # Obtaining the member 'encode' of a type (line 53)
        encode_23227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 11), unicode___call_result_23226, 'encode')
        # Calling encode(args, kwargs) (line 53)
        encode_call_result_23230 = invoke(stypy.reporting.localization.Localization(__file__, 53, 11), encode_23227, *[str_23228], **kwargs_23229)
        
        # Getting the type of 'long' (line 53)
        long_23231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 45), 'long', False)
        # Processing the call keyword arguments (line 53)
        kwargs_23232 = {}
        # Getting the type of 'eq' (line 53)
        eq_23222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'eq', False)
        # Calling eq(args, kwargs) (line 53)
        eq_call_result_23233 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), eq_23222, *[encode_call_result_23230, long_23231], **kwargs_23232)
        
        
        # ################# End of 'test_japanese_codecs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_japanese_codecs' in the type store
        # Getting the type of 'stypy_return_type' (line 23)
        stypy_return_type_23234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23234)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_japanese_codecs'
        return stypy_return_type_23234


    @norecursion
    def test_payload_encoding(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_payload_encoding'
        module_type_store = module_type_store.open_function_context('test_payload_encoding', 55, 4, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestEmailAsianCodecs.test_payload_encoding.__dict__.__setitem__('stypy_localization', localization)
        TestEmailAsianCodecs.test_payload_encoding.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestEmailAsianCodecs.test_payload_encoding.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestEmailAsianCodecs.test_payload_encoding.__dict__.__setitem__('stypy_function_name', 'TestEmailAsianCodecs.test_payload_encoding')
        TestEmailAsianCodecs.test_payload_encoding.__dict__.__setitem__('stypy_param_names_list', [])
        TestEmailAsianCodecs.test_payload_encoding.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestEmailAsianCodecs.test_payload_encoding.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestEmailAsianCodecs.test_payload_encoding.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestEmailAsianCodecs.test_payload_encoding.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestEmailAsianCodecs.test_payload_encoding.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestEmailAsianCodecs.test_payload_encoding.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestEmailAsianCodecs.test_payload_encoding', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_payload_encoding', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_payload_encoding(...)' code ##################

        
        # Assigning a Str to a Name (line 56):
        str_23235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 17), 'str', '\xa5\xcf\xa5\xed\xa1\xbc\xa5\xef\xa1\xbc\xa5\xeb\xa5\xc9\xa1\xaa')
        # Assigning a type to the variable 'jhello' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'jhello', str_23235)
        
        # Assigning a Str to a Name (line 57):
        str_23236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 17), 'str', 'euc-jp')
        # Assigning a type to the variable 'jcode' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'jcode', str_23236)
        
        # Assigning a Call to a Name (line 58):
        
        # Call to Message(...): (line 58)
        # Processing the call keyword arguments (line 58)
        kwargs_23238 = {}
        # Getting the type of 'Message' (line 58)
        Message_23237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 14), 'Message', False)
        # Calling Message(args, kwargs) (line 58)
        Message_call_result_23239 = invoke(stypy.reporting.localization.Localization(__file__, 58, 14), Message_23237, *[], **kwargs_23238)
        
        # Assigning a type to the variable 'msg' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'msg', Message_call_result_23239)
        
        # Call to set_payload(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'jhello' (line 59)
        jhello_23242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'jhello', False)
        # Getting the type of 'jcode' (line 59)
        jcode_23243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 32), 'jcode', False)
        # Processing the call keyword arguments (line 59)
        kwargs_23244 = {}
        # Getting the type of 'msg' (line 59)
        msg_23240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'msg', False)
        # Obtaining the member 'set_payload' of a type (line 59)
        set_payload_23241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), msg_23240, 'set_payload')
        # Calling set_payload(args, kwargs) (line 59)
        set_payload_call_result_23245 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), set_payload_23241, *[jhello_23242, jcode_23243], **kwargs_23244)
        
        
        # Assigning a Call to a Name (line 60):
        
        # Call to unicode(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Call to get_payload(...): (line 60)
        # Processing the call keyword arguments (line 60)
        kwargs_23249 = {}
        # Getting the type of 'msg' (line 60)
        msg_23247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'msg', False)
        # Obtaining the member 'get_payload' of a type (line 60)
        get_payload_23248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 23), msg_23247, 'get_payload')
        # Calling get_payload(args, kwargs) (line 60)
        get_payload_call_result_23250 = invoke(stypy.reporting.localization.Localization(__file__, 60, 23), get_payload_23248, *[], **kwargs_23249)
        
        
        # Call to get_content_charset(...): (line 60)
        # Processing the call keyword arguments (line 60)
        kwargs_23253 = {}
        # Getting the type of 'msg' (line 60)
        msg_23251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 42), 'msg', False)
        # Obtaining the member 'get_content_charset' of a type (line 60)
        get_content_charset_23252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 42), msg_23251, 'get_content_charset')
        # Calling get_content_charset(args, kwargs) (line 60)
        get_content_charset_call_result_23254 = invoke(stypy.reporting.localization.Localization(__file__, 60, 42), get_content_charset_23252, *[], **kwargs_23253)
        
        # Processing the call keyword arguments (line 60)
        kwargs_23255 = {}
        # Getting the type of 'unicode' (line 60)
        unicode_23246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'unicode', False)
        # Calling unicode(args, kwargs) (line 60)
        unicode_call_result_23256 = invoke(stypy.reporting.localization.Localization(__file__, 60, 15), unicode_23246, *[get_payload_call_result_23250, get_content_charset_call_result_23254], **kwargs_23255)
        
        # Assigning a type to the variable 'ustr' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'ustr', unicode_call_result_23256)
        
        # Call to assertEqual(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'jhello' (line 61)
        jhello_23259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'jhello', False)
        
        # Call to encode(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'jcode' (line 61)
        jcode_23262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 45), 'jcode', False)
        # Processing the call keyword arguments (line 61)
        kwargs_23263 = {}
        # Getting the type of 'ustr' (line 61)
        ustr_23260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 33), 'ustr', False)
        # Obtaining the member 'encode' of a type (line 61)
        encode_23261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 33), ustr_23260, 'encode')
        # Calling encode(args, kwargs) (line 61)
        encode_call_result_23264 = invoke(stypy.reporting.localization.Localization(__file__, 61, 33), encode_23261, *[jcode_23262], **kwargs_23263)
        
        # Processing the call keyword arguments (line 61)
        kwargs_23265 = {}
        # Getting the type of 'self' (line 61)
        self_23257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 61)
        assertEqual_23258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_23257, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 61)
        assertEqual_call_result_23266 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), assertEqual_23258, *[jhello_23259, encode_call_result_23264], **kwargs_23265)
        
        
        # ################# End of 'test_payload_encoding(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_payload_encoding' in the type store
        # Getting the type of 'stypy_return_type' (line 55)
        stypy_return_type_23267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23267)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_payload_encoding'
        return stypy_return_type_23267


# Assigning a type to the variable 'TestEmailAsianCodecs' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'TestEmailAsianCodecs', TestEmailAsianCodecs)

@norecursion
def suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'suite'
    module_type_store = module_type_store.open_function_context('suite', 65, 0, False)
    
    # Passed parameters checking function
    suite.stypy_localization = localization
    suite.stypy_type_of_self = None
    suite.stypy_type_store = module_type_store
    suite.stypy_function_name = 'suite'
    suite.stypy_param_names_list = []
    suite.stypy_varargs_param_name = None
    suite.stypy_kwargs_param_name = None
    suite.stypy_call_defaults = defaults
    suite.stypy_call_varargs = varargs
    suite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'suite', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'suite', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'suite(...)' code ##################

    
    # Assigning a Call to a Name (line 66):
    
    # Call to TestSuite(...): (line 66)
    # Processing the call keyword arguments (line 66)
    kwargs_23270 = {}
    # Getting the type of 'unittest' (line 66)
    unittest_23268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'unittest', False)
    # Obtaining the member 'TestSuite' of a type (line 66)
    TestSuite_23269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), unittest_23268, 'TestSuite')
    # Calling TestSuite(args, kwargs) (line 66)
    TestSuite_call_result_23271 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), TestSuite_23269, *[], **kwargs_23270)
    
    # Assigning a type to the variable 'suite' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'suite', TestSuite_call_result_23271)
    
    # Call to addTest(...): (line 67)
    # Processing the call arguments (line 67)
    
    # Call to makeSuite(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'TestEmailAsianCodecs' (line 67)
    TestEmailAsianCodecs_23276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 37), 'TestEmailAsianCodecs', False)
    # Processing the call keyword arguments (line 67)
    kwargs_23277 = {}
    # Getting the type of 'unittest' (line 67)
    unittest_23274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 67)
    makeSuite_23275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 18), unittest_23274, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 67)
    makeSuite_call_result_23278 = invoke(stypy.reporting.localization.Localization(__file__, 67, 18), makeSuite_23275, *[TestEmailAsianCodecs_23276], **kwargs_23277)
    
    # Processing the call keyword arguments (line 67)
    kwargs_23279 = {}
    # Getting the type of 'suite' (line 67)
    suite_23272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'suite', False)
    # Obtaining the member 'addTest' of a type (line 67)
    addTest_23273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 4), suite_23272, 'addTest')
    # Calling addTest(args, kwargs) (line 67)
    addTest_call_result_23280 = invoke(stypy.reporting.localization.Localization(__file__, 67, 4), addTest_23273, *[makeSuite_call_result_23278], **kwargs_23279)
    
    # Getting the type of 'suite' (line 68)
    suite_23281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'suite')
    # Assigning a type to the variable 'stypy_return_type' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type', suite_23281)
    
    # ################# End of 'suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'suite' in the type store
    # Getting the type of 'stypy_return_type' (line 65)
    stypy_return_type_23282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23282)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'suite'
    return stypy_return_type_23282

# Assigning a type to the variable 'suite' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'suite', suite)

@norecursion
def test_main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_main'
    module_type_store = module_type_store.open_function_context('test_main', 71, 0, False)
    
    # Passed parameters checking function
    test_main.stypy_localization = localization
    test_main.stypy_type_of_self = None
    test_main.stypy_type_store = module_type_store
    test_main.stypy_function_name = 'test_main'
    test_main.stypy_param_names_list = []
    test_main.stypy_varargs_param_name = None
    test_main.stypy_kwargs_param_name = None
    test_main.stypy_call_defaults = defaults
    test_main.stypy_call_varargs = varargs
    test_main.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_main', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_main', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_main(...)' code ##################

    
    # Call to run_unittest(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'TestEmailAsianCodecs' (line 72)
    TestEmailAsianCodecs_23284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'TestEmailAsianCodecs', False)
    # Processing the call keyword arguments (line 72)
    kwargs_23285 = {}
    # Getting the type of 'run_unittest' (line 72)
    run_unittest_23283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 72)
    run_unittest_call_result_23286 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), run_unittest_23283, *[TestEmailAsianCodecs_23284], **kwargs_23285)
    
    
    # ################# End of 'test_main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_main' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_23287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23287)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_main'
    return stypy_return_type_23287

# Assigning a type to the variable 'test_main' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'test_main', test_main)

if (__name__ == '__main__'):
    
    # Call to main(...): (line 77)
    # Processing the call keyword arguments (line 77)
    str_23290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 30), 'str', 'suite')
    keyword_23291 = str_23290
    kwargs_23292 = {'defaultTest': keyword_23291}
    # Getting the type of 'unittest' (line 77)
    unittest_23288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'unittest', False)
    # Obtaining the member 'main' of a type (line 77)
    main_23289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 4), unittest_23288, 'main')
    # Calling main(args, kwargs) (line 77)
    main_call_result_23293 = invoke(stypy.reporting.localization.Localization(__file__, 77, 4), main_23289, *[], **kwargs_23292)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
