
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2001-2006 Python Software Foundation
2: # Author: Barry Warsaw
3: # Contact: email-sig@python.org
4: 
5: '''Class representing text/* type MIME documents.'''
6: 
7: __all__ = ['MIMEText']
8: 
9: from email.encoders import encode_7or8bit
10: from email.mime.nonmultipart import MIMENonMultipart
11: 
12: 
13: 
14: class MIMEText(MIMENonMultipart):
15:     '''Class for generating text/* type MIME documents.'''
16: 
17:     def __init__(self, _text, _subtype='plain', _charset='us-ascii'):
18:         '''Create a text/* type MIME document.
19: 
20:         _text is the string for this message object.
21: 
22:         _subtype is the MIME sub content type, defaulting to "plain".
23: 
24:         _charset is the character set parameter added to the Content-Type
25:         header.  This defaults to "us-ascii".  Note that as a side-effect, the
26:         Content-Transfer-Encoding header will also be set.
27:         '''
28:         MIMENonMultipart.__init__(self, 'text', _subtype,
29:                                   **{'charset': _charset})
30:         self.set_payload(_text, _charset)
31: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_21036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'str', 'Class representing text/* type MIME documents.')

# Assigning a List to a Name (line 7):
__all__ = ['MIMEText']
module_type_store.set_exportable_members(['MIMEText'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_21037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_21038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'MIMEText')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_21037, str_21038)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_21037)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from email.encoders import encode_7or8bit' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/email/mime/')
import_21039 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'email.encoders')

if (type(import_21039) is not StypyTypeError):

    if (import_21039 != 'pyd_module'):
        __import__(import_21039)
        sys_modules_21040 = sys.modules[import_21039]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'email.encoders', sys_modules_21040.module_type_store, module_type_store, ['encode_7or8bit'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_21040, sys_modules_21040.module_type_store, module_type_store)
    else:
        from email.encoders import encode_7or8bit

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'email.encoders', None, module_type_store, ['encode_7or8bit'], [encode_7or8bit])

else:
    # Assigning a type to the variable 'email.encoders' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'email.encoders', import_21039)

remove_current_file_folder_from_path('C:/Python27/lib/email/mime/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from email.mime.nonmultipart import MIMENonMultipart' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/email/mime/')
import_21041 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'email.mime.nonmultipart')

if (type(import_21041) is not StypyTypeError):

    if (import_21041 != 'pyd_module'):
        __import__(import_21041)
        sys_modules_21042 = sys.modules[import_21041]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'email.mime.nonmultipart', sys_modules_21042.module_type_store, module_type_store, ['MIMENonMultipart'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_21042, sys_modules_21042.module_type_store, module_type_store)
    else:
        from email.mime.nonmultipart import MIMENonMultipart

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'email.mime.nonmultipart', None, module_type_store, ['MIMENonMultipart'], [MIMENonMultipart])

else:
    # Assigning a type to the variable 'email.mime.nonmultipart' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'email.mime.nonmultipart', import_21041)

remove_current_file_folder_from_path('C:/Python27/lib/email/mime/')

# Declaration of the 'MIMEText' class
# Getting the type of 'MIMENonMultipart' (line 14)
MIMENonMultipart_21043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'MIMENonMultipart')

class MIMEText(MIMENonMultipart_21043, ):
    str_21044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 4), 'str', 'Class for generating text/* type MIME documents.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_21045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 39), 'str', 'plain')
        str_21046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 57), 'str', 'us-ascii')
        defaults = [str_21045, str_21046]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MIMEText.__init__', ['_text', '_subtype', '_charset'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['_text', '_subtype', '_charset'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_21047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, (-1)), 'str', 'Create a text/* type MIME document.\n\n        _text is the string for this message object.\n\n        _subtype is the MIME sub content type, defaulting to "plain".\n\n        _charset is the character set parameter added to the Content-Type\n        header.  This defaults to "us-ascii".  Note that as a side-effect, the\n        Content-Transfer-Encoding header will also be set.\n        ')
        
        # Call to __init__(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'self' (line 28)
        self_21050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 34), 'self', False)
        str_21051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 40), 'str', 'text')
        # Getting the type of '_subtype' (line 28)
        _subtype_21052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 48), '_subtype', False)
        # Processing the call keyword arguments (line 28)
        
        # Obtaining an instance of the builtin type 'dict' (line 29)
        dict_21053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 36), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 29)
        # Adding element type (key, value) (line 29)
        str_21054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 37), 'str', 'charset')
        # Getting the type of '_charset' (line 29)
        _charset_21055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 48), '_charset', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 36), dict_21053, (str_21054, _charset_21055))
        
        kwargs_21056 = {'dict_21053': dict_21053}
        # Getting the type of 'MIMENonMultipart' (line 28)
        MIMENonMultipart_21048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'MIMENonMultipart', False)
        # Obtaining the member '__init__' of a type (line 28)
        init___21049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), MIMENonMultipart_21048, '__init__')
        # Calling __init__(args, kwargs) (line 28)
        init___call_result_21057 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), init___21049, *[self_21050, str_21051, _subtype_21052], **kwargs_21056)
        
        
        # Call to set_payload(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of '_text' (line 30)
        _text_21060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 25), '_text', False)
        # Getting the type of '_charset' (line 30)
        _charset_21061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 32), '_charset', False)
        # Processing the call keyword arguments (line 30)
        kwargs_21062 = {}
        # Getting the type of 'self' (line 30)
        self_21058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self', False)
        # Obtaining the member 'set_payload' of a type (line 30)
        set_payload_21059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_21058, 'set_payload')
        # Calling set_payload(args, kwargs) (line 30)
        set_payload_call_result_21063 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), set_payload_21059, *[_text_21060, _charset_21061], **kwargs_21062)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'MIMEText' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'MIMEText', MIMEText)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
