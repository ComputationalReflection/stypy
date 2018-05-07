
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2001-2006 Python Software Foundation
2: # Author: Barry Warsaw
3: # Contact: email-sig@python.org
4: 
5: '''Class representing image/* type MIME documents.'''
6: 
7: __all__ = ['MIMEImage']
8: 
9: import imghdr
10: 
11: from email import encoders
12: from email.mime.nonmultipart import MIMENonMultipart
13: 
14: 
15: 
16: class MIMEImage(MIMENonMultipart):
17:     '''Class for generating image/* type MIME documents.'''
18: 
19:     def __init__(self, _imagedata, _subtype=None,
20:                  _encoder=encoders.encode_base64, **_params):
21:         '''Create an image/* type MIME document.
22: 
23:         _imagedata is a string containing the raw image data.  If this data
24:         can be decoded by the standard Python `imghdr' module, then the
25:         subtype will be automatically included in the Content-Type header.
26:         Otherwise, you can specify the specific image subtype via the _subtype
27:         parameter.
28: 
29:         _encoder is a function which will perform the actual encoding for
30:         transport of the image data.  It takes one argument, which is this
31:         Image instance.  It should use get_payload() and set_payload() to
32:         change the payload to the encoded form.  It should also add any
33:         Content-Transfer-Encoding or other headers to the message as
34:         necessary.  The default encoding is Base64.
35: 
36:         Any additional keyword arguments are passed to the base class
37:         constructor, which turns them into parameters on the Content-Type
38:         header.
39:         '''
40:         if _subtype is None:
41:             _subtype = imghdr.what(None, _imagedata)
42:         if _subtype is None:
43:             raise TypeError('Could not guess image MIME subtype')
44:         MIMENonMultipart.__init__(self, 'image', _subtype, **_params)
45:         self.set_payload(_imagedata)
46:         _encoder(self)
47: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_20899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'str', 'Class representing image/* type MIME documents.')

# Assigning a List to a Name (line 7):
__all__ = ['MIMEImage']
module_type_store.set_exportable_members(['MIMEImage'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_20900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_20901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'MIMEImage')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_20900, str_20901)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_20900)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import imghdr' statement (line 9)
import imghdr

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'imghdr', imghdr, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from email import encoders' statement (line 11)
try:
    from email import encoders

except:
    encoders = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'email', None, module_type_store, ['encoders'], [encoders])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from email.mime.nonmultipart import MIMENonMultipart' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/email/mime/')
import_20902 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'email.mime.nonmultipart')

if (type(import_20902) is not StypyTypeError):

    if (import_20902 != 'pyd_module'):
        __import__(import_20902)
        sys_modules_20903 = sys.modules[import_20902]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'email.mime.nonmultipart', sys_modules_20903.module_type_store, module_type_store, ['MIMENonMultipart'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_20903, sys_modules_20903.module_type_store, module_type_store)
    else:
        from email.mime.nonmultipart import MIMENonMultipart

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'email.mime.nonmultipart', None, module_type_store, ['MIMENonMultipart'], [MIMENonMultipart])

else:
    # Assigning a type to the variable 'email.mime.nonmultipart' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'email.mime.nonmultipart', import_20902)

remove_current_file_folder_from_path('C:/Python27/lib/email/mime/')

# Declaration of the 'MIMEImage' class
# Getting the type of 'MIMENonMultipart' (line 16)
MIMENonMultipart_20904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'MIMENonMultipart')

class MIMEImage(MIMENonMultipart_20904, ):
    str_20905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 4), 'str', 'Class for generating image/* type MIME documents.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 19)
        None_20906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 44), 'None')
        # Getting the type of 'encoders' (line 20)
        encoders_20907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 26), 'encoders')
        # Obtaining the member 'encode_base64' of a type (line 20)
        encode_base64_20908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 26), encoders_20907, 'encode_base64')
        defaults = [None_20906, encode_base64_20908]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MIMEImage.__init__', ['_imagedata', '_subtype', '_encoder'], None, '_params', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['_imagedata', '_subtype', '_encoder'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_20909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, (-1)), 'str', "Create an image/* type MIME document.\n\n        _imagedata is a string containing the raw image data.  If this data\n        can be decoded by the standard Python `imghdr' module, then the\n        subtype will be automatically included in the Content-Type header.\n        Otherwise, you can specify the specific image subtype via the _subtype\n        parameter.\n\n        _encoder is a function which will perform the actual encoding for\n        transport of the image data.  It takes one argument, which is this\n        Image instance.  It should use get_payload() and set_payload() to\n        change the payload to the encoded form.  It should also add any\n        Content-Transfer-Encoding or other headers to the message as\n        necessary.  The default encoding is Base64.\n\n        Any additional keyword arguments are passed to the base class\n        constructor, which turns them into parameters on the Content-Type\n        header.\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 40)
        # Getting the type of '_subtype' (line 40)
        _subtype_20910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), '_subtype')
        # Getting the type of 'None' (line 40)
        None_20911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 23), 'None')
        
        (may_be_20912, more_types_in_union_20913) = may_be_none(_subtype_20910, None_20911)

        if may_be_20912:

            if more_types_in_union_20913:
                # Runtime conditional SSA (line 40)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 41):
            
            # Call to what(...): (line 41)
            # Processing the call arguments (line 41)
            # Getting the type of 'None' (line 41)
            None_20916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 35), 'None', False)
            # Getting the type of '_imagedata' (line 41)
            _imagedata_20917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 41), '_imagedata', False)
            # Processing the call keyword arguments (line 41)
            kwargs_20918 = {}
            # Getting the type of 'imghdr' (line 41)
            imghdr_20914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 23), 'imghdr', False)
            # Obtaining the member 'what' of a type (line 41)
            what_20915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 23), imghdr_20914, 'what')
            # Calling what(args, kwargs) (line 41)
            what_call_result_20919 = invoke(stypy.reporting.localization.Localization(__file__, 41, 23), what_20915, *[None_20916, _imagedata_20917], **kwargs_20918)
            
            # Assigning a type to the variable '_subtype' (line 41)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), '_subtype', what_call_result_20919)

            if more_types_in_union_20913:
                # SSA join for if statement (line 40)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 42)
        # Getting the type of '_subtype' (line 42)
        _subtype_20920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), '_subtype')
        # Getting the type of 'None' (line 42)
        None_20921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 23), 'None')
        
        (may_be_20922, more_types_in_union_20923) = may_be_none(_subtype_20920, None_20921)

        if may_be_20922:

            if more_types_in_union_20923:
                # Runtime conditional SSA (line 42)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to TypeError(...): (line 43)
            # Processing the call arguments (line 43)
            str_20925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 28), 'str', 'Could not guess image MIME subtype')
            # Processing the call keyword arguments (line 43)
            kwargs_20926 = {}
            # Getting the type of 'TypeError' (line 43)
            TypeError_20924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 18), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 43)
            TypeError_call_result_20927 = invoke(stypy.reporting.localization.Localization(__file__, 43, 18), TypeError_20924, *[str_20925], **kwargs_20926)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 43, 12), TypeError_call_result_20927, 'raise parameter', BaseException)

            if more_types_in_union_20923:
                # SSA join for if statement (line 42)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of '_subtype' (line 42)
        _subtype_20928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), '_subtype')
        # Assigning a type to the variable '_subtype' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), '_subtype', remove_type_from_union(_subtype_20928, types.NoneType))
        
        # Call to __init__(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'self' (line 44)
        self_20931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 34), 'self', False)
        str_20932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 40), 'str', 'image')
        # Getting the type of '_subtype' (line 44)
        _subtype_20933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 49), '_subtype', False)
        # Processing the call keyword arguments (line 44)
        # Getting the type of '_params' (line 44)
        _params_20934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 61), '_params', False)
        kwargs_20935 = {'_params_20934': _params_20934}
        # Getting the type of 'MIMENonMultipart' (line 44)
        MIMENonMultipart_20929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'MIMENonMultipart', False)
        # Obtaining the member '__init__' of a type (line 44)
        init___20930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), MIMENonMultipart_20929, '__init__')
        # Calling __init__(args, kwargs) (line 44)
        init___call_result_20936 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), init___20930, *[self_20931, str_20932, _subtype_20933], **kwargs_20935)
        
        
        # Call to set_payload(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of '_imagedata' (line 45)
        _imagedata_20939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), '_imagedata', False)
        # Processing the call keyword arguments (line 45)
        kwargs_20940 = {}
        # Getting the type of 'self' (line 45)
        self_20937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self', False)
        # Obtaining the member 'set_payload' of a type (line 45)
        set_payload_20938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_20937, 'set_payload')
        # Calling set_payload(args, kwargs) (line 45)
        set_payload_call_result_20941 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), set_payload_20938, *[_imagedata_20939], **kwargs_20940)
        
        
        # Call to _encoder(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'self' (line 46)
        self_20943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 17), 'self', False)
        # Processing the call keyword arguments (line 46)
        kwargs_20944 = {}
        # Getting the type of '_encoder' (line 46)
        _encoder_20942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), '_encoder', False)
        # Calling _encoder(args, kwargs) (line 46)
        _encoder_call_result_20945 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), _encoder_20942, *[self_20943], **kwargs_20944)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'MIMEImage' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'MIMEImage', MIMEImage)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
