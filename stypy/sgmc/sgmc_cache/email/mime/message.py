
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2001-2006 Python Software Foundation
2: # Author: Barry Warsaw
3: # Contact: email-sig@python.org
4: 
5: '''Class representing message/* MIME documents.'''
6: 
7: __all__ = ['MIMEMessage']
8: 
9: from email import message
10: from email.mime.nonmultipart import MIMENonMultipart
11: 
12: 
13: 
14: class MIMEMessage(MIMENonMultipart):
15:     '''Class representing message/* MIME documents.'''
16: 
17:     def __init__(self, _msg, _subtype='rfc822'):
18:         '''Create a message/* type MIME document.
19: 
20:         _msg is a message object and must be an instance of Message, or a
21:         derived class of Message, otherwise a TypeError is raised.
22: 
23:         Optional _subtype defines the subtype of the contained message.  The
24:         default is "rfc822" (this is defined by the MIME standard, even though
25:         the term "rfc822" is technically outdated by RFC 2822).
26:         '''
27:         MIMENonMultipart.__init__(self, 'message', _subtype)
28:         if not isinstance(_msg, message.Message):
29:             raise TypeError('Argument is not an instance of Message')
30:         # It's convenient to use this base class method.  We need to do it
31:         # this way or we'll get an exception
32:         message.Message.attach(self, _msg)
33:         # And be sure our default type is set correctly
34:         self.set_default_type('message/rfc822')
35: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_20946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'str', 'Class representing message/* MIME documents.')

# Assigning a List to a Name (line 7):
__all__ = ['MIMEMessage']
module_type_store.set_exportable_members(['MIMEMessage'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_20947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_20948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'MIMEMessage')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_20947, str_20948)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_20947)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from email import message' statement (line 9)
try:
    from email import message

except:
    message = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'email', None, module_type_store, ['message'], [message])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from email.mime.nonmultipart import MIMENonMultipart' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/email/mime/')
import_20949 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'email.mime.nonmultipart')

if (type(import_20949) is not StypyTypeError):

    if (import_20949 != 'pyd_module'):
        __import__(import_20949)
        sys_modules_20950 = sys.modules[import_20949]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'email.mime.nonmultipart', sys_modules_20950.module_type_store, module_type_store, ['MIMENonMultipart'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_20950, sys_modules_20950.module_type_store, module_type_store)
    else:
        from email.mime.nonmultipart import MIMENonMultipart

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'email.mime.nonmultipart', None, module_type_store, ['MIMENonMultipart'], [MIMENonMultipart])

else:
    # Assigning a type to the variable 'email.mime.nonmultipart' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'email.mime.nonmultipart', import_20949)

remove_current_file_folder_from_path('C:/Python27/lib/email/mime/')

# Declaration of the 'MIMEMessage' class
# Getting the type of 'MIMENonMultipart' (line 14)
MIMENonMultipart_20951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 18), 'MIMENonMultipart')

class MIMEMessage(MIMENonMultipart_20951, ):
    str_20952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 4), 'str', 'Class representing message/* MIME documents.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_20953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 38), 'str', 'rfc822')
        defaults = [str_20953]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MIMEMessage.__init__', ['_msg', '_subtype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['_msg', '_subtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_20954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, (-1)), 'str', 'Create a message/* type MIME document.\n\n        _msg is a message object and must be an instance of Message, or a\n        derived class of Message, otherwise a TypeError is raised.\n\n        Optional _subtype defines the subtype of the contained message.  The\n        default is "rfc822" (this is defined by the MIME standard, even though\n        the term "rfc822" is technically outdated by RFC 2822).\n        ')
        
        # Call to __init__(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'self' (line 27)
        self_20957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 34), 'self', False)
        str_20958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 40), 'str', 'message')
        # Getting the type of '_subtype' (line 27)
        _subtype_20959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 51), '_subtype', False)
        # Processing the call keyword arguments (line 27)
        kwargs_20960 = {}
        # Getting the type of 'MIMENonMultipart' (line 27)
        MIMENonMultipart_20955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'MIMENonMultipart', False)
        # Obtaining the member '__init__' of a type (line 27)
        init___20956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), MIMENonMultipart_20955, '__init__')
        # Calling __init__(args, kwargs) (line 27)
        init___call_result_20961 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), init___20956, *[self_20957, str_20958, _subtype_20959], **kwargs_20960)
        
        
        
        # Call to isinstance(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of '_msg' (line 28)
        _msg_20963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 26), '_msg', False)
        # Getting the type of 'message' (line 28)
        message_20964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 32), 'message', False)
        # Obtaining the member 'Message' of a type (line 28)
        Message_20965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 32), message_20964, 'Message')
        # Processing the call keyword arguments (line 28)
        kwargs_20966 = {}
        # Getting the type of 'isinstance' (line 28)
        isinstance_20962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 28)
        isinstance_call_result_20967 = invoke(stypy.reporting.localization.Localization(__file__, 28, 15), isinstance_20962, *[_msg_20963, Message_20965], **kwargs_20966)
        
        # Applying the 'not' unary operator (line 28)
        result_not__20968 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 11), 'not', isinstance_call_result_20967)
        
        # Testing if the type of an if condition is none (line 28)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 28, 8), result_not__20968):
            pass
        else:
            
            # Testing the type of an if condition (line 28)
            if_condition_20969 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 8), result_not__20968)
            # Assigning a type to the variable 'if_condition_20969' (line 28)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'if_condition_20969', if_condition_20969)
            # SSA begins for if statement (line 28)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 29)
            # Processing the call arguments (line 29)
            str_20971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 28), 'str', 'Argument is not an instance of Message')
            # Processing the call keyword arguments (line 29)
            kwargs_20972 = {}
            # Getting the type of 'TypeError' (line 29)
            TypeError_20970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 18), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 29)
            TypeError_call_result_20973 = invoke(stypy.reporting.localization.Localization(__file__, 29, 18), TypeError_20970, *[str_20971], **kwargs_20972)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 29, 12), TypeError_call_result_20973, 'raise parameter', BaseException)
            # SSA join for if statement (line 28)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to attach(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'self' (line 32)
        self_20977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 31), 'self', False)
        # Getting the type of '_msg' (line 32)
        _msg_20978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 37), '_msg', False)
        # Processing the call keyword arguments (line 32)
        kwargs_20979 = {}
        # Getting the type of 'message' (line 32)
        message_20974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'message', False)
        # Obtaining the member 'Message' of a type (line 32)
        Message_20975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), message_20974, 'Message')
        # Obtaining the member 'attach' of a type (line 32)
        attach_20976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), Message_20975, 'attach')
        # Calling attach(args, kwargs) (line 32)
        attach_call_result_20980 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), attach_20976, *[self_20977, _msg_20978], **kwargs_20979)
        
        
        # Call to set_default_type(...): (line 34)
        # Processing the call arguments (line 34)
        str_20983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 30), 'str', 'message/rfc822')
        # Processing the call keyword arguments (line 34)
        kwargs_20984 = {}
        # Getting the type of 'self' (line 34)
        self_20981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self', False)
        # Obtaining the member 'set_default_type' of a type (line 34)
        set_default_type_20982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_20981, 'set_default_type')
        # Calling set_default_type(args, kwargs) (line 34)
        set_default_type_call_result_20985 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), set_default_type_20982, *[str_20983], **kwargs_20984)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'MIMEMessage' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'MIMEMessage', MIMEMessage)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
