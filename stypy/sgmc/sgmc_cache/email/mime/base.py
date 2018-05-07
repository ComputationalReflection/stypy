
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2001-2006 Python Software Foundation
2: # Author: Barry Warsaw
3: # Contact: email-sig@python.org
4: 
5: '''Base class for MIME specializations.'''
6: 
7: __all__ = ['MIMEBase']
8: 
9: from email import message
10: 
11: 
12: 
13: class MIMEBase(message.Message):
14:     '''Base class for MIME specializations.'''
15: 
16:     def __init__(self, _maintype, _subtype, **_params):
17:         '''This constructor adds a Content-Type: and a MIME-Version: header.
18: 
19:         The Content-Type: header is taken from the _maintype and _subtype
20:         arguments.  Additional parameters for this header are taken from the
21:         keyword arguments.
22:         '''
23:         message.Message.__init__(self)
24:         ctype = '%s/%s' % (_maintype, _subtype)
25:         self.add_header('Content-Type', ctype, **_params)
26:         self['MIME-Version'] = '1.0'
27: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_20871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'str', 'Base class for MIME specializations.')

# Assigning a List to a Name (line 7):
__all__ = ['MIMEBase']
module_type_store.set_exportable_members(['MIMEBase'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_20872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_20873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'MIMEBase')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_20872, str_20873)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_20872)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from email import message' statement (line 9)
try:
    from email import message

except:
    message = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'email', None, module_type_store, ['message'], [message])

# Declaration of the 'MIMEBase' class
# Getting the type of 'message' (line 13)
message_20874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 15), 'message')
# Obtaining the member 'Message' of a type (line 13)
Message_20875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 15), message_20874, 'Message')

class MIMEBase(Message_20875, ):
    str_20876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'str', 'Base class for MIME specializations.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 16, 4, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MIMEBase.__init__', ['_maintype', '_subtype'], None, '_params', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['_maintype', '_subtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_20877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, (-1)), 'str', 'This constructor adds a Content-Type: and a MIME-Version: header.\n\n        The Content-Type: header is taken from the _maintype and _subtype\n        arguments.  Additional parameters for this header are taken from the\n        keyword arguments.\n        ')
        
        # Call to __init__(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'self' (line 23)
        self_20881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 33), 'self', False)
        # Processing the call keyword arguments (line 23)
        kwargs_20882 = {}
        # Getting the type of 'message' (line 23)
        message_20878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'message', False)
        # Obtaining the member 'Message' of a type (line 23)
        Message_20879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), message_20878, 'Message')
        # Obtaining the member '__init__' of a type (line 23)
        init___20880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), Message_20879, '__init__')
        # Calling __init__(args, kwargs) (line 23)
        init___call_result_20883 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), init___20880, *[self_20881], **kwargs_20882)
        
        
        # Assigning a BinOp to a Name (line 24):
        str_20884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 16), 'str', '%s/%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 24)
        tuple_20885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 24)
        # Adding element type (line 24)
        # Getting the type of '_maintype' (line 24)
        _maintype_20886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 27), '_maintype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 27), tuple_20885, _maintype_20886)
        # Adding element type (line 24)
        # Getting the type of '_subtype' (line 24)
        _subtype_20887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 38), '_subtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 27), tuple_20885, _subtype_20887)
        
        # Applying the binary operator '%' (line 24)
        result_mod_20888 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 16), '%', str_20884, tuple_20885)
        
        # Assigning a type to the variable 'ctype' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'ctype', result_mod_20888)
        
        # Call to add_header(...): (line 25)
        # Processing the call arguments (line 25)
        str_20891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 24), 'str', 'Content-Type')
        # Getting the type of 'ctype' (line 25)
        ctype_20892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 40), 'ctype', False)
        # Processing the call keyword arguments (line 25)
        # Getting the type of '_params' (line 25)
        _params_20893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 49), '_params', False)
        kwargs_20894 = {'_params_20893': _params_20893}
        # Getting the type of 'self' (line 25)
        self_20889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self', False)
        # Obtaining the member 'add_header' of a type (line 25)
        add_header_20890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_20889, 'add_header')
        # Calling add_header(args, kwargs) (line 25)
        add_header_call_result_20895 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), add_header_20890, *[str_20891, ctype_20892], **kwargs_20894)
        
        
        # Assigning a Str to a Subscript (line 26):
        str_20896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 31), 'str', '1.0')
        # Getting the type of 'self' (line 26)
        self_20897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self')
        str_20898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 13), 'str', 'MIME-Version')
        # Storing an element on a container (line 26)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 8), self_20897, (str_20898, str_20896))
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'MIMEBase' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'MIMEBase', MIMEBase)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
