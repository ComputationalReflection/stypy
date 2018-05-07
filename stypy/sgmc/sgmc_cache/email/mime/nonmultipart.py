
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2002-2006 Python Software Foundation
2: # Author: Barry Warsaw
3: # Contact: email-sig@python.org
4: 
5: '''Base class for MIME type messages that are not multipart.'''
6: 
7: __all__ = ['MIMENonMultipart']
8: 
9: from email import errors
10: from email.mime.base import MIMEBase
11: 
12: 
13: 
14: class MIMENonMultipart(MIMEBase):
15:     '''Base class for MIME non-multipart type messages.'''
16: 
17:     def attach(self, payload):
18:         # The public API prohibits attaching multiple subparts to MIMEBase
19:         # derived subtypes since none of them are, by definition, of content
20:         # type multipart/*
21:         raise errors.MultipartConversionError(
22:             'Cannot attach additional subparts to non-multipart/*')
23: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_21023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'str', 'Base class for MIME type messages that are not multipart.')

# Assigning a List to a Name (line 7):
__all__ = ['MIMENonMultipart']
module_type_store.set_exportable_members(['MIMENonMultipart'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_21024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_21025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'MIMENonMultipart')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_21024, str_21025)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_21024)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from email import errors' statement (line 9)
try:
    from email import errors

except:
    errors = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'email', None, module_type_store, ['errors'], [errors])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from email.mime.base import MIMEBase' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/email/mime/')
import_21026 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'email.mime.base')

if (type(import_21026) is not StypyTypeError):

    if (import_21026 != 'pyd_module'):
        __import__(import_21026)
        sys_modules_21027 = sys.modules[import_21026]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'email.mime.base', sys_modules_21027.module_type_store, module_type_store, ['MIMEBase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_21027, sys_modules_21027.module_type_store, module_type_store)
    else:
        from email.mime.base import MIMEBase

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'email.mime.base', None, module_type_store, ['MIMEBase'], [MIMEBase])

else:
    # Assigning a type to the variable 'email.mime.base' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'email.mime.base', import_21026)

remove_current_file_folder_from_path('C:/Python27/lib/email/mime/')

# Declaration of the 'MIMENonMultipart' class
# Getting the type of 'MIMEBase' (line 14)
MIMEBase_21028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 23), 'MIMEBase')

class MIMENonMultipart(MIMEBase_21028, ):
    str_21029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 4), 'str', 'Base class for MIME non-multipart type messages.')

    @norecursion
    def attach(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'attach'
        module_type_store = module_type_store.open_function_context('attach', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MIMENonMultipart.attach.__dict__.__setitem__('stypy_localization', localization)
        MIMENonMultipart.attach.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MIMENonMultipart.attach.__dict__.__setitem__('stypy_type_store', module_type_store)
        MIMENonMultipart.attach.__dict__.__setitem__('stypy_function_name', 'MIMENonMultipart.attach')
        MIMENonMultipart.attach.__dict__.__setitem__('stypy_param_names_list', ['payload'])
        MIMENonMultipart.attach.__dict__.__setitem__('stypy_varargs_param_name', None)
        MIMENonMultipart.attach.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MIMENonMultipart.attach.__dict__.__setitem__('stypy_call_defaults', defaults)
        MIMENonMultipart.attach.__dict__.__setitem__('stypy_call_varargs', varargs)
        MIMENonMultipart.attach.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MIMENonMultipart.attach.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MIMENonMultipart.attach', ['payload'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'attach', localization, ['payload'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'attach(...)' code ##################

        
        # Call to MultipartConversionError(...): (line 21)
        # Processing the call arguments (line 21)
        str_21032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 12), 'str', 'Cannot attach additional subparts to non-multipart/*')
        # Processing the call keyword arguments (line 21)
        kwargs_21033 = {}
        # Getting the type of 'errors' (line 21)
        errors_21030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 14), 'errors', False)
        # Obtaining the member 'MultipartConversionError' of a type (line 21)
        MultipartConversionError_21031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 14), errors_21030, 'MultipartConversionError')
        # Calling MultipartConversionError(args, kwargs) (line 21)
        MultipartConversionError_call_result_21034 = invoke(stypy.reporting.localization.Localization(__file__, 21, 14), MultipartConversionError_21031, *[str_21032], **kwargs_21033)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 21, 8), MultipartConversionError_call_result_21034, 'raise parameter', BaseException)
        
        # ################# End of 'attach(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'attach' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_21035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21035)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'attach'
        return stypy_return_type_21035


# Assigning a type to the variable 'MIMENonMultipart' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'MIMENonMultipart', MIMENonMultipart)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
