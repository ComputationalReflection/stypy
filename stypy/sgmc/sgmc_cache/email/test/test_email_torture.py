
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2002-2004 Python Software Foundation
2: #
3: # A torture test of the email package.  This should not be run as part of the
4: # standard Python test suite since it requires several meg of email messages
5: # collected in the wild.  These source messages are not checked into the
6: # Python distro, but are available as part of the standalone email package at
7: # http://sf.net/projects/mimelib
8: 
9: import sys
10: import os
11: import unittest
12: from cStringIO import StringIO
13: from types import ListType
14: 
15: from email.test.test_email import TestEmailBase
16: from test.test_support import TestSkipped, run_unittest
17: 
18: import email
19: from email import __file__ as testfile
20: from email.iterators import _structure
21: 
22: def openfile(filename):
23:     from os.path import join, dirname, abspath
24:     path = abspath(join(dirname(testfile), os.pardir, 'moredata', filename))
25:     return open(path, 'r')
26: 
27: # Prevent this test from running in the Python distro
28: try:
29:     openfile('crispin-torture.txt')
30: except IOError:
31:     raise TestSkipped
32: 
33: 
34: 
35: class TortureBase(TestEmailBase):
36:     def _msgobj(self, filename):
37:         fp = openfile(filename)
38:         try:
39:             msg = email.message_from_file(fp)
40:         finally:
41:             fp.close()
42:         return msg
43: 
44: 
45: 
46: class TestCrispinTorture(TortureBase):
47:     # Mark Crispin's torture test from the SquirrelMail project
48:     def test_mondo_message(self):
49:         eq = self.assertEqual
50:         neq = self.ndiffAssertEqual
51:         msg = self._msgobj('crispin-torture.txt')
52:         payload = msg.get_payload()
53:         eq(type(payload), ListType)
54:         eq(len(payload), 12)
55:         eq(msg.preamble, None)
56:         eq(msg.epilogue, '\n')
57:         # Probably the best way to verify the message is parsed correctly is to
58:         # dump its structure and compare it against the known structure.
59:         fp = StringIO()
60:         _structure(msg, fp=fp)
61:         neq(fp.getvalue(), '''\
62: multipart/mixed
63:     text/plain
64:     message/rfc822
65:         multipart/alternative
66:             text/plain
67:             multipart/mixed
68:                 text/richtext
69:             application/andrew-inset
70:     message/rfc822
71:         audio/basic
72:     audio/basic
73:     image/pbm
74:     message/rfc822
75:         multipart/mixed
76:             multipart/mixed
77:                 text/plain
78:                 audio/x-sun
79:             multipart/mixed
80:                 image/gif
81:                 image/gif
82:                 application/x-be2
83:                 application/atomicmail
84:             audio/x-sun
85:     message/rfc822
86:         multipart/mixed
87:             text/plain
88:             image/pgm
89:             text/plain
90:     message/rfc822
91:         multipart/mixed
92:             text/plain
93:             image/pbm
94:     message/rfc822
95:         application/postscript
96:     image/gif
97:     message/rfc822
98:         multipart/mixed
99:             audio/basic
100:             audio/basic
101:     message/rfc822
102:         multipart/mixed
103:             application/postscript
104:             text/plain
105:             message/rfc822
106:                 multipart/mixed
107:                     text/plain
108:                     multipart/parallel
109:                         image/gif
110:                         audio/basic
111:                     application/atomicmail
112:                     message/rfc822
113:                         audio/x-sun
114: ''')
115: 
116: 
117: def _testclasses():
118:     mod = sys.modules[__name__]
119:     return [getattr(mod, name) for name in dir(mod) if name.startswith('Test')]
120: 
121: 
122: def suite():
123:     suite = unittest.TestSuite()
124:     for testclass in _testclasses():
125:         suite.addTest(unittest.makeSuite(testclass))
126:     return suite
127: 
128: 
129: def test_main():
130:     for testclass in _testclasses():
131:         run_unittest(testclass)
132: 
133: 
134: 
135: if __name__ == '__main__':
136:     unittest.main(defaultTest='suite')
137: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import sys' statement (line 9)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import os' statement (line 10)
import os

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import unittest' statement (line 11)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from cStringIO import StringIO' statement (line 12)
try:
    from cStringIO import StringIO

except:
    StringIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'cStringIO', None, module_type_store, ['StringIO'], [StringIO])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from types import ListType' statement (line 13)
try:
    from types import ListType

except:
    ListType = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'types', None, module_type_store, ['ListType'], [ListType])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from email.test.test_email import TestEmailBase' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/email/test/')
import_33630 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'email.test.test_email')

if (type(import_33630) is not StypyTypeError):

    if (import_33630 != 'pyd_module'):
        __import__(import_33630)
        sys_modules_33631 = sys.modules[import_33630]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'email.test.test_email', sys_modules_33631.module_type_store, module_type_store, ['TestEmailBase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_33631, sys_modules_33631.module_type_store, module_type_store)
    else:
        from email.test.test_email import TestEmailBase

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'email.test.test_email', None, module_type_store, ['TestEmailBase'], [TestEmailBase])

else:
    # Assigning a type to the variable 'email.test.test_email' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'email.test.test_email', import_33630)

remove_current_file_folder_from_path('C:/Python27/lib/email/test/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from test.test_support import TestSkipped, run_unittest' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/email/test/')
import_33632 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'test.test_support')

if (type(import_33632) is not StypyTypeError):

    if (import_33632 != 'pyd_module'):
        __import__(import_33632)
        sys_modules_33633 = sys.modules[import_33632]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'test.test_support', sys_modules_33633.module_type_store, module_type_store, ['TestSkipped', 'run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_33633, sys_modules_33633.module_type_store, module_type_store)
    else:
        from test.test_support import TestSkipped, run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'test.test_support', None, module_type_store, ['TestSkipped', 'run_unittest'], [TestSkipped, run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'test.test_support', import_33632)

remove_current_file_folder_from_path('C:/Python27/lib/email/test/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import email' statement (line 18)
import email

import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'email', email, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from email import testfile' statement (line 19)
try:
    from email import __file__ as testfile

except:
    testfile = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'email', None, module_type_store, ['__file__'], [testfile])
# Adding an alias
module_type_store.add_alias('testfile', '__file__')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from email.iterators import _structure' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/email/test/')
import_33634 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'email.iterators')

if (type(import_33634) is not StypyTypeError):

    if (import_33634 != 'pyd_module'):
        __import__(import_33634)
        sys_modules_33635 = sys.modules[import_33634]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'email.iterators', sys_modules_33635.module_type_store, module_type_store, ['_structure'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_33635, sys_modules_33635.module_type_store, module_type_store)
    else:
        from email.iterators import _structure

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'email.iterators', None, module_type_store, ['_structure'], [_structure])

else:
    # Assigning a type to the variable 'email.iterators' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'email.iterators', import_33634)

remove_current_file_folder_from_path('C:/Python27/lib/email/test/')


@norecursion
def openfile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'openfile'
    module_type_store = module_type_store.open_function_context('openfile', 22, 0, False)
    
    # Passed parameters checking function
    openfile.stypy_localization = localization
    openfile.stypy_type_of_self = None
    openfile.stypy_type_store = module_type_store
    openfile.stypy_function_name = 'openfile'
    openfile.stypy_param_names_list = ['filename']
    openfile.stypy_varargs_param_name = None
    openfile.stypy_kwargs_param_name = None
    openfile.stypy_call_defaults = defaults
    openfile.stypy_call_varargs = varargs
    openfile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'openfile', ['filename'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'openfile', localization, ['filename'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'openfile(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 4))
    
    # 'from os.path import join, dirname, abspath' statement (line 23)
    update_path_to_current_file_folder('C:/Python27/lib/email/test/')
    import_33636 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 4), 'os.path')

    if (type(import_33636) is not StypyTypeError):

        if (import_33636 != 'pyd_module'):
            __import__(import_33636)
            sys_modules_33637 = sys.modules[import_33636]
            import_from_module(stypy.reporting.localization.Localization(__file__, 23, 4), 'os.path', sys_modules_33637.module_type_store, module_type_store, ['join', 'dirname', 'abspath'])
            nest_module(stypy.reporting.localization.Localization(__file__, 23, 4), __file__, sys_modules_33637, sys_modules_33637.module_type_store, module_type_store)
        else:
            from os.path import join, dirname, abspath

            import_from_module(stypy.reporting.localization.Localization(__file__, 23, 4), 'os.path', None, module_type_store, ['join', 'dirname', 'abspath'], [join, dirname, abspath])

    else:
        # Assigning a type to the variable 'os.path' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'os.path', import_33636)

    remove_current_file_folder_from_path('C:/Python27/lib/email/test/')
    
    
    # Assigning a Call to a Name (line 24):
    
    # Call to abspath(...): (line 24)
    # Processing the call arguments (line 24)
    
    # Call to join(...): (line 24)
    # Processing the call arguments (line 24)
    
    # Call to dirname(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'testfile' (line 24)
    testfile_33641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 32), 'testfile', False)
    # Processing the call keyword arguments (line 24)
    kwargs_33642 = {}
    # Getting the type of 'dirname' (line 24)
    dirname_33640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 24), 'dirname', False)
    # Calling dirname(args, kwargs) (line 24)
    dirname_call_result_33643 = invoke(stypy.reporting.localization.Localization(__file__, 24, 24), dirname_33640, *[testfile_33641], **kwargs_33642)
    
    # Getting the type of 'os' (line 24)
    os_33644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 43), 'os', False)
    # Obtaining the member 'pardir' of a type (line 24)
    pardir_33645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 43), os_33644, 'pardir')
    str_33646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 54), 'str', 'moredata')
    # Getting the type of 'filename' (line 24)
    filename_33647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 66), 'filename', False)
    # Processing the call keyword arguments (line 24)
    kwargs_33648 = {}
    # Getting the type of 'join' (line 24)
    join_33639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 19), 'join', False)
    # Calling join(args, kwargs) (line 24)
    join_call_result_33649 = invoke(stypy.reporting.localization.Localization(__file__, 24, 19), join_33639, *[dirname_call_result_33643, pardir_33645, str_33646, filename_33647], **kwargs_33648)
    
    # Processing the call keyword arguments (line 24)
    kwargs_33650 = {}
    # Getting the type of 'abspath' (line 24)
    abspath_33638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), 'abspath', False)
    # Calling abspath(args, kwargs) (line 24)
    abspath_call_result_33651 = invoke(stypy.reporting.localization.Localization(__file__, 24, 11), abspath_33638, *[join_call_result_33649], **kwargs_33650)
    
    # Assigning a type to the variable 'path' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'path', abspath_call_result_33651)
    
    # Call to open(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'path' (line 25)
    path_33653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'path', False)
    str_33654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 22), 'str', 'r')
    # Processing the call keyword arguments (line 25)
    kwargs_33655 = {}
    # Getting the type of 'open' (line 25)
    open_33652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'open', False)
    # Calling open(args, kwargs) (line 25)
    open_call_result_33656 = invoke(stypy.reporting.localization.Localization(__file__, 25, 11), open_33652, *[path_33653, str_33654], **kwargs_33655)
    
    # Assigning a type to the variable 'stypy_return_type' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type', open_call_result_33656)
    
    # ################# End of 'openfile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'openfile' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_33657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33657)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'openfile'
    return stypy_return_type_33657

# Assigning a type to the variable 'openfile' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'openfile', openfile)


# SSA begins for try-except statement (line 28)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Call to openfile(...): (line 29)
# Processing the call arguments (line 29)
str_33659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 13), 'str', 'crispin-torture.txt')
# Processing the call keyword arguments (line 29)
kwargs_33660 = {}
# Getting the type of 'openfile' (line 29)
openfile_33658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'openfile', False)
# Calling openfile(args, kwargs) (line 29)
openfile_call_result_33661 = invoke(stypy.reporting.localization.Localization(__file__, 29, 4), openfile_33658, *[str_33659], **kwargs_33660)

# SSA branch for the except part of a try statement (line 28)
# SSA branch for the except 'IOError' branch of a try statement (line 28)
module_type_store.open_ssa_branch('except')
# Getting the type of 'TestSkipped' (line 31)
TestSkipped_33662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 10), 'TestSkipped')
ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 31, 4), TestSkipped_33662, 'raise parameter', BaseException)
# SSA join for try-except statement (line 28)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'TortureBase' class
# Getting the type of 'TestEmailBase' (line 35)
TestEmailBase_33663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 18), 'TestEmailBase')

class TortureBase(TestEmailBase_33663, ):

    @norecursion
    def _msgobj(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_msgobj'
        module_type_store = module_type_store.open_function_context('_msgobj', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TortureBase._msgobj.__dict__.__setitem__('stypy_localization', localization)
        TortureBase._msgobj.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TortureBase._msgobj.__dict__.__setitem__('stypy_type_store', module_type_store)
        TortureBase._msgobj.__dict__.__setitem__('stypy_function_name', 'TortureBase._msgobj')
        TortureBase._msgobj.__dict__.__setitem__('stypy_param_names_list', ['filename'])
        TortureBase._msgobj.__dict__.__setitem__('stypy_varargs_param_name', None)
        TortureBase._msgobj.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TortureBase._msgobj.__dict__.__setitem__('stypy_call_defaults', defaults)
        TortureBase._msgobj.__dict__.__setitem__('stypy_call_varargs', varargs)
        TortureBase._msgobj.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TortureBase._msgobj.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TortureBase._msgobj', ['filename'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_msgobj', localization, ['filename'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_msgobj(...)' code ##################

        
        # Assigning a Call to a Name (line 37):
        
        # Call to openfile(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'filename' (line 37)
        filename_33665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 22), 'filename', False)
        # Processing the call keyword arguments (line 37)
        kwargs_33666 = {}
        # Getting the type of 'openfile' (line 37)
        openfile_33664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 13), 'openfile', False)
        # Calling openfile(args, kwargs) (line 37)
        openfile_call_result_33667 = invoke(stypy.reporting.localization.Localization(__file__, 37, 13), openfile_33664, *[filename_33665], **kwargs_33666)
        
        # Assigning a type to the variable 'fp' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'fp', openfile_call_result_33667)
        
        # Try-finally block (line 38)
        
        # Assigning a Call to a Name (line 39):
        
        # Call to message_from_file(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'fp' (line 39)
        fp_33670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 42), 'fp', False)
        # Processing the call keyword arguments (line 39)
        kwargs_33671 = {}
        # Getting the type of 'email' (line 39)
        email_33668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 18), 'email', False)
        # Obtaining the member 'message_from_file' of a type (line 39)
        message_from_file_33669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 18), email_33668, 'message_from_file')
        # Calling message_from_file(args, kwargs) (line 39)
        message_from_file_call_result_33672 = invoke(stypy.reporting.localization.Localization(__file__, 39, 18), message_from_file_33669, *[fp_33670], **kwargs_33671)
        
        # Assigning a type to the variable 'msg' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'msg', message_from_file_call_result_33672)
        
        # finally branch of the try-finally block (line 38)
        
        # Call to close(...): (line 41)
        # Processing the call keyword arguments (line 41)
        kwargs_33675 = {}
        # Getting the type of 'fp' (line 41)
        fp_33673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'fp', False)
        # Obtaining the member 'close' of a type (line 41)
        close_33674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), fp_33673, 'close')
        # Calling close(args, kwargs) (line 41)
        close_call_result_33676 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), close_33674, *[], **kwargs_33675)
        
        
        # Getting the type of 'msg' (line 42)
        msg_33677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'msg')
        # Assigning a type to the variable 'stypy_return_type' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'stypy_return_type', msg_33677)
        
        # ################# End of '_msgobj(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_msgobj' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_33678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33678)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_msgobj'
        return stypy_return_type_33678


# Assigning a type to the variable 'TortureBase' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'TortureBase', TortureBase)
# Declaration of the 'TestCrispinTorture' class
# Getting the type of 'TortureBase' (line 46)
TortureBase_33679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 25), 'TortureBase')

class TestCrispinTorture(TortureBase_33679, ):

    @norecursion
    def test_mondo_message(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_mondo_message'
        module_type_store = module_type_store.open_function_context('test_mondo_message', 48, 4, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCrispinTorture.test_mondo_message.__dict__.__setitem__('stypy_localization', localization)
        TestCrispinTorture.test_mondo_message.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCrispinTorture.test_mondo_message.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCrispinTorture.test_mondo_message.__dict__.__setitem__('stypy_function_name', 'TestCrispinTorture.test_mondo_message')
        TestCrispinTorture.test_mondo_message.__dict__.__setitem__('stypy_param_names_list', [])
        TestCrispinTorture.test_mondo_message.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCrispinTorture.test_mondo_message.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCrispinTorture.test_mondo_message.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCrispinTorture.test_mondo_message.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCrispinTorture.test_mondo_message.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCrispinTorture.test_mondo_message.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCrispinTorture.test_mondo_message', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_mondo_message', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_mondo_message(...)' code ##################

        
        # Assigning a Attribute to a Name (line 49):
        # Getting the type of 'self' (line 49)
        self_33680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 13), 'self')
        # Obtaining the member 'assertEqual' of a type (line 49)
        assertEqual_33681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 13), self_33680, 'assertEqual')
        # Assigning a type to the variable 'eq' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'eq', assertEqual_33681)
        
        # Assigning a Attribute to a Name (line 50):
        # Getting the type of 'self' (line 50)
        self_33682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 14), 'self')
        # Obtaining the member 'ndiffAssertEqual' of a type (line 50)
        ndiffAssertEqual_33683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 14), self_33682, 'ndiffAssertEqual')
        # Assigning a type to the variable 'neq' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'neq', ndiffAssertEqual_33683)
        
        # Assigning a Call to a Name (line 51):
        
        # Call to _msgobj(...): (line 51)
        # Processing the call arguments (line 51)
        str_33686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 27), 'str', 'crispin-torture.txt')
        # Processing the call keyword arguments (line 51)
        kwargs_33687 = {}
        # Getting the type of 'self' (line 51)
        self_33684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 14), 'self', False)
        # Obtaining the member '_msgobj' of a type (line 51)
        _msgobj_33685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 14), self_33684, '_msgobj')
        # Calling _msgobj(args, kwargs) (line 51)
        _msgobj_call_result_33688 = invoke(stypy.reporting.localization.Localization(__file__, 51, 14), _msgobj_33685, *[str_33686], **kwargs_33687)
        
        # Assigning a type to the variable 'msg' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'msg', _msgobj_call_result_33688)
        
        # Assigning a Call to a Name (line 52):
        
        # Call to get_payload(...): (line 52)
        # Processing the call keyword arguments (line 52)
        kwargs_33691 = {}
        # Getting the type of 'msg' (line 52)
        msg_33689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 18), 'msg', False)
        # Obtaining the member 'get_payload' of a type (line 52)
        get_payload_33690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 18), msg_33689, 'get_payload')
        # Calling get_payload(args, kwargs) (line 52)
        get_payload_call_result_33692 = invoke(stypy.reporting.localization.Localization(__file__, 52, 18), get_payload_33690, *[], **kwargs_33691)
        
        # Assigning a type to the variable 'payload' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'payload', get_payload_call_result_33692)
        
        # Call to eq(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Call to type(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'payload' (line 53)
        payload_33695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'payload', False)
        # Processing the call keyword arguments (line 53)
        kwargs_33696 = {}
        # Getting the type of 'type' (line 53)
        type_33694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'type', False)
        # Calling type(args, kwargs) (line 53)
        type_call_result_33697 = invoke(stypy.reporting.localization.Localization(__file__, 53, 11), type_33694, *[payload_33695], **kwargs_33696)
        
        # Getting the type of 'ListType' (line 53)
        ListType_33698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 26), 'ListType', False)
        # Processing the call keyword arguments (line 53)
        kwargs_33699 = {}
        # Getting the type of 'eq' (line 53)
        eq_33693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'eq', False)
        # Calling eq(args, kwargs) (line 53)
        eq_call_result_33700 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), eq_33693, *[type_call_result_33697, ListType_33698], **kwargs_33699)
        
        
        # Call to eq(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Call to len(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'payload' (line 54)
        payload_33703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'payload', False)
        # Processing the call keyword arguments (line 54)
        kwargs_33704 = {}
        # Getting the type of 'len' (line 54)
        len_33702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'len', False)
        # Calling len(args, kwargs) (line 54)
        len_call_result_33705 = invoke(stypy.reporting.localization.Localization(__file__, 54, 11), len_33702, *[payload_33703], **kwargs_33704)
        
        int_33706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 25), 'int')
        # Processing the call keyword arguments (line 54)
        kwargs_33707 = {}
        # Getting the type of 'eq' (line 54)
        eq_33701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'eq', False)
        # Calling eq(args, kwargs) (line 54)
        eq_call_result_33708 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), eq_33701, *[len_call_result_33705, int_33706], **kwargs_33707)
        
        
        # Call to eq(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'msg' (line 55)
        msg_33710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'msg', False)
        # Obtaining the member 'preamble' of a type (line 55)
        preamble_33711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 11), msg_33710, 'preamble')
        # Getting the type of 'None' (line 55)
        None_33712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'None', False)
        # Processing the call keyword arguments (line 55)
        kwargs_33713 = {}
        # Getting the type of 'eq' (line 55)
        eq_33709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'eq', False)
        # Calling eq(args, kwargs) (line 55)
        eq_call_result_33714 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), eq_33709, *[preamble_33711, None_33712], **kwargs_33713)
        
        
        # Call to eq(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'msg' (line 56)
        msg_33716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'msg', False)
        # Obtaining the member 'epilogue' of a type (line 56)
        epilogue_33717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 11), msg_33716, 'epilogue')
        str_33718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'str', '\n')
        # Processing the call keyword arguments (line 56)
        kwargs_33719 = {}
        # Getting the type of 'eq' (line 56)
        eq_33715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'eq', False)
        # Calling eq(args, kwargs) (line 56)
        eq_call_result_33720 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), eq_33715, *[epilogue_33717, str_33718], **kwargs_33719)
        
        
        # Assigning a Call to a Name (line 59):
        
        # Call to StringIO(...): (line 59)
        # Processing the call keyword arguments (line 59)
        kwargs_33722 = {}
        # Getting the type of 'StringIO' (line 59)
        StringIO_33721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 13), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 59)
        StringIO_call_result_33723 = invoke(stypy.reporting.localization.Localization(__file__, 59, 13), StringIO_33721, *[], **kwargs_33722)
        
        # Assigning a type to the variable 'fp' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'fp', StringIO_call_result_33723)
        
        # Call to _structure(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'msg' (line 60)
        msg_33725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 19), 'msg', False)
        # Processing the call keyword arguments (line 60)
        # Getting the type of 'fp' (line 60)
        fp_33726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 27), 'fp', False)
        keyword_33727 = fp_33726
        kwargs_33728 = {'fp': keyword_33727}
        # Getting the type of '_structure' (line 60)
        _structure_33724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), '_structure', False)
        # Calling _structure(args, kwargs) (line 60)
        _structure_call_result_33729 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), _structure_33724, *[msg_33725], **kwargs_33728)
        
        
        # Call to neq(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Call to getvalue(...): (line 61)
        # Processing the call keyword arguments (line 61)
        kwargs_33733 = {}
        # Getting the type of 'fp' (line 61)
        fp_33731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'fp', False)
        # Obtaining the member 'getvalue' of a type (line 61)
        getvalue_33732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), fp_33731, 'getvalue')
        # Calling getvalue(args, kwargs) (line 61)
        getvalue_call_result_33734 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), getvalue_33732, *[], **kwargs_33733)
        
        str_33735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, (-1)), 'str', 'multipart/mixed\n    text/plain\n    message/rfc822\n        multipart/alternative\n            text/plain\n            multipart/mixed\n                text/richtext\n            application/andrew-inset\n    message/rfc822\n        audio/basic\n    audio/basic\n    image/pbm\n    message/rfc822\n        multipart/mixed\n            multipart/mixed\n                text/plain\n                audio/x-sun\n            multipart/mixed\n                image/gif\n                image/gif\n                application/x-be2\n                application/atomicmail\n            audio/x-sun\n    message/rfc822\n        multipart/mixed\n            text/plain\n            image/pgm\n            text/plain\n    message/rfc822\n        multipart/mixed\n            text/plain\n            image/pbm\n    message/rfc822\n        application/postscript\n    image/gif\n    message/rfc822\n        multipart/mixed\n            audio/basic\n            audio/basic\n    message/rfc822\n        multipart/mixed\n            application/postscript\n            text/plain\n            message/rfc822\n                multipart/mixed\n                    text/plain\n                    multipart/parallel\n                        image/gif\n                        audio/basic\n                    application/atomicmail\n                    message/rfc822\n                        audio/x-sun\n')
        # Processing the call keyword arguments (line 61)
        kwargs_33736 = {}
        # Getting the type of 'neq' (line 61)
        neq_33730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'neq', False)
        # Calling neq(args, kwargs) (line 61)
        neq_call_result_33737 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), neq_33730, *[getvalue_call_result_33734, str_33735], **kwargs_33736)
        
        
        # ################# End of 'test_mondo_message(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_mondo_message' in the type store
        # Getting the type of 'stypy_return_type' (line 48)
        stypy_return_type_33738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33738)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_mondo_message'
        return stypy_return_type_33738


# Assigning a type to the variable 'TestCrispinTorture' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'TestCrispinTorture', TestCrispinTorture)

@norecursion
def _testclasses(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_testclasses'
    module_type_store = module_type_store.open_function_context('_testclasses', 117, 0, False)
    
    # Passed parameters checking function
    _testclasses.stypy_localization = localization
    _testclasses.stypy_type_of_self = None
    _testclasses.stypy_type_store = module_type_store
    _testclasses.stypy_function_name = '_testclasses'
    _testclasses.stypy_param_names_list = []
    _testclasses.stypy_varargs_param_name = None
    _testclasses.stypy_kwargs_param_name = None
    _testclasses.stypy_call_defaults = defaults
    _testclasses.stypy_call_varargs = varargs
    _testclasses.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_testclasses', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_testclasses', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_testclasses(...)' code ##################

    
    # Assigning a Subscript to a Name (line 118):
    
    # Obtaining the type of the subscript
    # Getting the type of '__name__' (line 118)
    name___33739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 22), '__name__')
    # Getting the type of 'sys' (line 118)
    sys_33740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 10), 'sys')
    # Obtaining the member 'modules' of a type (line 118)
    modules_33741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 10), sys_33740, 'modules')
    # Obtaining the member '__getitem__' of a type (line 118)
    getitem___33742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 10), modules_33741, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 118)
    subscript_call_result_33743 = invoke(stypy.reporting.localization.Localization(__file__, 118, 10), getitem___33742, name___33739)
    
    # Assigning a type to the variable 'mod' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'mod', subscript_call_result_33743)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to dir(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'mod' (line 119)
    mod_33755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 47), 'mod', False)
    # Processing the call keyword arguments (line 119)
    kwargs_33756 = {}
    # Getting the type of 'dir' (line 119)
    dir_33754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 43), 'dir', False)
    # Calling dir(args, kwargs) (line 119)
    dir_call_result_33757 = invoke(stypy.reporting.localization.Localization(__file__, 119, 43), dir_33754, *[mod_33755], **kwargs_33756)
    
    comprehension_33758 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 12), dir_call_result_33757)
    # Assigning a type to the variable 'name' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'name', comprehension_33758)
    
    # Call to startswith(...): (line 119)
    # Processing the call arguments (line 119)
    str_33751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 71), 'str', 'Test')
    # Processing the call keyword arguments (line 119)
    kwargs_33752 = {}
    # Getting the type of 'name' (line 119)
    name_33749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 55), 'name', False)
    # Obtaining the member 'startswith' of a type (line 119)
    startswith_33750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 55), name_33749, 'startswith')
    # Calling startswith(args, kwargs) (line 119)
    startswith_call_result_33753 = invoke(stypy.reporting.localization.Localization(__file__, 119, 55), startswith_33750, *[str_33751], **kwargs_33752)
    
    
    # Call to getattr(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'mod' (line 119)
    mod_33745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 20), 'mod', False)
    # Getting the type of 'name' (line 119)
    name_33746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 25), 'name', False)
    # Processing the call keyword arguments (line 119)
    kwargs_33747 = {}
    # Getting the type of 'getattr' (line 119)
    getattr_33744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'getattr', False)
    # Calling getattr(args, kwargs) (line 119)
    getattr_call_result_33748 = invoke(stypy.reporting.localization.Localization(__file__, 119, 12), getattr_33744, *[mod_33745, name_33746], **kwargs_33747)
    
    list_33759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 12), list_33759, getattr_call_result_33748)
    # Assigning a type to the variable 'stypy_return_type' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type', list_33759)
    
    # ################# End of '_testclasses(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_testclasses' in the type store
    # Getting the type of 'stypy_return_type' (line 117)
    stypy_return_type_33760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33760)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_testclasses'
    return stypy_return_type_33760

# Assigning a type to the variable '_testclasses' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), '_testclasses', _testclasses)

@norecursion
def suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'suite'
    module_type_store = module_type_store.open_function_context('suite', 122, 0, False)
    
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

    
    # Assigning a Call to a Name (line 123):
    
    # Call to TestSuite(...): (line 123)
    # Processing the call keyword arguments (line 123)
    kwargs_33763 = {}
    # Getting the type of 'unittest' (line 123)
    unittest_33761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'unittest', False)
    # Obtaining the member 'TestSuite' of a type (line 123)
    TestSuite_33762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 12), unittest_33761, 'TestSuite')
    # Calling TestSuite(args, kwargs) (line 123)
    TestSuite_call_result_33764 = invoke(stypy.reporting.localization.Localization(__file__, 123, 12), TestSuite_33762, *[], **kwargs_33763)
    
    # Assigning a type to the variable 'suite' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'suite', TestSuite_call_result_33764)
    
    
    # Call to _testclasses(...): (line 124)
    # Processing the call keyword arguments (line 124)
    kwargs_33766 = {}
    # Getting the type of '_testclasses' (line 124)
    _testclasses_33765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 21), '_testclasses', False)
    # Calling _testclasses(args, kwargs) (line 124)
    _testclasses_call_result_33767 = invoke(stypy.reporting.localization.Localization(__file__, 124, 21), _testclasses_33765, *[], **kwargs_33766)
    
    # Assigning a type to the variable '_testclasses_call_result_33767' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), '_testclasses_call_result_33767', _testclasses_call_result_33767)
    # Testing if the for loop is going to be iterated (line 124)
    # Testing the type of a for loop iterable (line 124)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 124, 4), _testclasses_call_result_33767)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 124, 4), _testclasses_call_result_33767):
        # Getting the type of the for loop variable (line 124)
        for_loop_var_33768 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 124, 4), _testclasses_call_result_33767)
        # Assigning a type to the variable 'testclass' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'testclass', for_loop_var_33768)
        # SSA begins for a for statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to addTest(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Call to makeSuite(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'testclass' (line 125)
        testclass_33773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 41), 'testclass', False)
        # Processing the call keyword arguments (line 125)
        kwargs_33774 = {}
        # Getting the type of 'unittest' (line 125)
        unittest_33771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 22), 'unittest', False)
        # Obtaining the member 'makeSuite' of a type (line 125)
        makeSuite_33772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 22), unittest_33771, 'makeSuite')
        # Calling makeSuite(args, kwargs) (line 125)
        makeSuite_call_result_33775 = invoke(stypy.reporting.localization.Localization(__file__, 125, 22), makeSuite_33772, *[testclass_33773], **kwargs_33774)
        
        # Processing the call keyword arguments (line 125)
        kwargs_33776 = {}
        # Getting the type of 'suite' (line 125)
        suite_33769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'suite', False)
        # Obtaining the member 'addTest' of a type (line 125)
        addTest_33770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), suite_33769, 'addTest')
        # Calling addTest(args, kwargs) (line 125)
        addTest_call_result_33777 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), addTest_33770, *[makeSuite_call_result_33775], **kwargs_33776)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'suite' (line 126)
    suite_33778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), 'suite')
    # Assigning a type to the variable 'stypy_return_type' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type', suite_33778)
    
    # ################# End of 'suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'suite' in the type store
    # Getting the type of 'stypy_return_type' (line 122)
    stypy_return_type_33779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33779)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'suite'
    return stypy_return_type_33779

# Assigning a type to the variable 'suite' (line 122)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'suite', suite)

@norecursion
def test_main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_main'
    module_type_store = module_type_store.open_function_context('test_main', 129, 0, False)
    
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

    
    
    # Call to _testclasses(...): (line 130)
    # Processing the call keyword arguments (line 130)
    kwargs_33781 = {}
    # Getting the type of '_testclasses' (line 130)
    _testclasses_33780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 21), '_testclasses', False)
    # Calling _testclasses(args, kwargs) (line 130)
    _testclasses_call_result_33782 = invoke(stypy.reporting.localization.Localization(__file__, 130, 21), _testclasses_33780, *[], **kwargs_33781)
    
    # Assigning a type to the variable '_testclasses_call_result_33782' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), '_testclasses_call_result_33782', _testclasses_call_result_33782)
    # Testing if the for loop is going to be iterated (line 130)
    # Testing the type of a for loop iterable (line 130)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 130, 4), _testclasses_call_result_33782)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 130, 4), _testclasses_call_result_33782):
        # Getting the type of the for loop variable (line 130)
        for_loop_var_33783 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 130, 4), _testclasses_call_result_33782)
        # Assigning a type to the variable 'testclass' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'testclass', for_loop_var_33783)
        # SSA begins for a for statement (line 130)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to run_unittest(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'testclass' (line 131)
        testclass_33785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 21), 'testclass', False)
        # Processing the call keyword arguments (line 131)
        kwargs_33786 = {}
        # Getting the type of 'run_unittest' (line 131)
        run_unittest_33784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'run_unittest', False)
        # Calling run_unittest(args, kwargs) (line 131)
        run_unittest_call_result_33787 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), run_unittest_33784, *[testclass_33785], **kwargs_33786)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'test_main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_main' in the type store
    # Getting the type of 'stypy_return_type' (line 129)
    stypy_return_type_33788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33788)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_main'
    return stypy_return_type_33788

# Assigning a type to the variable 'test_main' (line 129)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 0), 'test_main', test_main)

if (__name__ == '__main__'):
    
    # Call to main(...): (line 136)
    # Processing the call keyword arguments (line 136)
    str_33791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 30), 'str', 'suite')
    keyword_33792 = str_33791
    kwargs_33793 = {'defaultTest': keyword_33792}
    # Getting the type of 'unittest' (line 136)
    unittest_33789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'unittest', False)
    # Obtaining the member 'main' of a type (line 136)
    main_33790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 4), unittest_33789, 'main')
    # Calling main(args, kwargs) (line 136)
    main_call_result_33794 = invoke(stypy.reporting.localization.Localization(__file__, 136, 4), main_33790, *[], **kwargs_33793)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
