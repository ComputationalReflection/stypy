
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2001-2006 Python Software Foundation
2: # Author: Keith Dart
3: # Contact: email-sig@python.org
4: 
5: '''Class representing application/* type MIME documents.'''
6: 
7: __all__ = ["MIMEApplication"]
8: 
9: from email import encoders
10: from email.mime.nonmultipart import MIMENonMultipart
11: 
12: 
13: class MIMEApplication(MIMENonMultipart):
14:     '''Class for generating application/* MIME documents.'''
15: 
16:     def __init__(self, _data, _subtype='octet-stream',
17:                  _encoder=encoders.encode_base64, **_params):
18:         '''Create an application/* type MIME document.
19: 
20:         _data is a string containing the raw application data.
21: 
22:         _subtype is the MIME content type subtype, defaulting to
23:         'octet-stream'.
24: 
25:         _encoder is a function which will perform the actual encoding for
26:         transport of the application data, defaulting to base64 encoding.
27: 
28:         Any additional keyword arguments are passed to the base class
29:         constructor, which turns them into parameters on the Content-Type
30:         header.
31:         '''
32:         if _subtype is None:
33:             raise TypeError('Invalid application MIME subtype')
34:         MIMENonMultipart.__init__(self, 'application', _subtype, **_params)
35:         self.set_payload(_data)
36:         _encoder(self)
37: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_20748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'str', 'Class representing application/* type MIME documents.')

# Assigning a List to a Name (line 7):
__all__ = ['MIMEApplication']
module_type_store.set_exportable_members(['MIMEApplication'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_20749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_20750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'MIMEApplication')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_20749, str_20750)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_20749)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from email import encoders' statement (line 9)
try:
    from email import encoders

except:
    encoders = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'email', None, module_type_store, ['encoders'], [encoders])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from email.mime.nonmultipart import MIMENonMultipart' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/email/mime/')
import_20751 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'email.mime.nonmultipart')

if (type(import_20751) is not StypyTypeError):

    if (import_20751 != 'pyd_module'):
        __import__(import_20751)
        sys_modules_20752 = sys.modules[import_20751]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'email.mime.nonmultipart', sys_modules_20752.module_type_store, module_type_store, ['MIMENonMultipart'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_20752, sys_modules_20752.module_type_store, module_type_store)
    else:
        from email.mime.nonmultipart import MIMENonMultipart

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'email.mime.nonmultipart', None, module_type_store, ['MIMENonMultipart'], [MIMENonMultipart])

else:
    # Assigning a type to the variable 'email.mime.nonmultipart' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'email.mime.nonmultipart', import_20751)

remove_current_file_folder_from_path('C:/Python27/lib/email/mime/')

# Declaration of the 'MIMEApplication' class
# Getting the type of 'MIMENonMultipart' (line 13)
MIMENonMultipart_20753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 22), 'MIMENonMultipart')

class MIMEApplication(MIMENonMultipart_20753, ):
    str_20754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'str', 'Class for generating application/* MIME documents.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_20755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 39), 'str', 'octet-stream')
        # Getting the type of 'encoders' (line 17)
        encoders_20756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 26), 'encoders')
        # Obtaining the member 'encode_base64' of a type (line 17)
        encode_base64_20757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 26), encoders_20756, 'encode_base64')
        defaults = [str_20755, encode_base64_20757]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 16, 4, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MIMEApplication.__init__', ['_data', '_subtype', '_encoder'], None, '_params', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['_data', '_subtype', '_encoder'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_20758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, (-1)), 'str', "Create an application/* type MIME document.\n\n        _data is a string containing the raw application data.\n\n        _subtype is the MIME content type subtype, defaulting to\n        'octet-stream'.\n\n        _encoder is a function which will perform the actual encoding for\n        transport of the application data, defaulting to base64 encoding.\n\n        Any additional keyword arguments are passed to the base class\n        constructor, which turns them into parameters on the Content-Type\n        header.\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 32)
        # Getting the type of '_subtype' (line 32)
        _subtype_20759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), '_subtype')
        # Getting the type of 'None' (line 32)
        None_20760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 23), 'None')
        
        (may_be_20761, more_types_in_union_20762) = may_be_none(_subtype_20759, None_20760)

        if may_be_20761:

            if more_types_in_union_20762:
                # Runtime conditional SSA (line 32)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to TypeError(...): (line 33)
            # Processing the call arguments (line 33)
            str_20764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 28), 'str', 'Invalid application MIME subtype')
            # Processing the call keyword arguments (line 33)
            kwargs_20765 = {}
            # Getting the type of 'TypeError' (line 33)
            TypeError_20763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 18), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 33)
            TypeError_call_result_20766 = invoke(stypy.reporting.localization.Localization(__file__, 33, 18), TypeError_20763, *[str_20764], **kwargs_20765)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 33, 12), TypeError_call_result_20766, 'raise parameter', BaseException)

            if more_types_in_union_20762:
                # SSA join for if statement (line 32)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of '_subtype' (line 32)
        _subtype_20767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), '_subtype')
        # Assigning a type to the variable '_subtype' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), '_subtype', remove_type_from_union(_subtype_20767, types.NoneType))
        
        # Call to __init__(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'self' (line 34)
        self_20770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 34), 'self', False)
        str_20771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 40), 'str', 'application')
        # Getting the type of '_subtype' (line 34)
        _subtype_20772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 55), '_subtype', False)
        # Processing the call keyword arguments (line 34)
        # Getting the type of '_params' (line 34)
        _params_20773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 67), '_params', False)
        kwargs_20774 = {'_params_20773': _params_20773}
        # Getting the type of 'MIMENonMultipart' (line 34)
        MIMENonMultipart_20768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'MIMENonMultipart', False)
        # Obtaining the member '__init__' of a type (line 34)
        init___20769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), MIMENonMultipart_20768, '__init__')
        # Calling __init__(args, kwargs) (line 34)
        init___call_result_20775 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), init___20769, *[self_20770, str_20771, _subtype_20772], **kwargs_20774)
        
        
        # Call to set_payload(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of '_data' (line 35)
        _data_20778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 25), '_data', False)
        # Processing the call keyword arguments (line 35)
        kwargs_20779 = {}
        # Getting the type of 'self' (line 35)
        self_20776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self', False)
        # Obtaining the member 'set_payload' of a type (line 35)
        set_payload_20777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_20776, 'set_payload')
        # Calling set_payload(args, kwargs) (line 35)
        set_payload_call_result_20780 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), set_payload_20777, *[_data_20778], **kwargs_20779)
        
        
        # Call to _encoder(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'self' (line 36)
        self_20782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'self', False)
        # Processing the call keyword arguments (line 36)
        kwargs_20783 = {}
        # Getting the type of '_encoder' (line 36)
        _encoder_20781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), '_encoder', False)
        # Calling _encoder(args, kwargs) (line 36)
        _encoder_call_result_20784 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), _encoder_20781, *[self_20782], **kwargs_20783)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'MIMEApplication' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'MIMEApplication', MIMEApplication)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
