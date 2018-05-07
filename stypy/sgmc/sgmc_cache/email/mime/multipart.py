
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2002-2006 Python Software Foundation
2: # Author: Barry Warsaw
3: # Contact: email-sig@python.org
4: 
5: '''Base class for MIME multipart/* type messages.'''
6: 
7: __all__ = ['MIMEMultipart']
8: 
9: from email.mime.base import MIMEBase
10: 
11: 
12: 
13: class MIMEMultipart(MIMEBase):
14:     '''Base class for MIME multipart/* type messages.'''
15: 
16:     def __init__(self, _subtype='mixed', boundary=None, _subparts=None,
17:                  **_params):
18:         '''Creates a multipart/* type message.
19: 
20:         By default, creates a multipart/mixed message, with proper
21:         Content-Type and MIME-Version headers.
22: 
23:         _subtype is the subtype of the multipart content type, defaulting to
24:         `mixed'.
25: 
26:         boundary is the multipart boundary string.  By default it is
27:         calculated as needed.
28: 
29:         _subparts is a sequence of initial subparts for the payload.  It
30:         must be an iterable object, such as a list.  You can always
31:         attach new subparts to the message by using the attach() method.
32: 
33:         Additional parameters for the Content-Type header are taken from the
34:         keyword arguments (or passed into the _params argument).
35:         '''
36:         MIMEBase.__init__(self, 'multipart', _subtype, **_params)
37: 
38:         # Initialise _payload to an empty list as the Message superclass's
39:         # implementation of is_multipart assumes that _payload is a list for
40:         # multipart messages.
41:         self._payload = []
42: 
43:         if _subparts:
44:             for p in _subparts:
45:                 self.attach(p)
46:         if boundary:
47:             self.set_boundary(boundary)
48: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_20986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'str', 'Base class for MIME multipart/* type messages.')

# Assigning a List to a Name (line 7):
__all__ = ['MIMEMultipart']
module_type_store.set_exportable_members(['MIMEMultipart'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_20987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_20988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'MIMEMultipart')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_20987, str_20988)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_20987)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from email.mime.base import MIMEBase' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/email/mime/')
import_20989 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'email.mime.base')

if (type(import_20989) is not StypyTypeError):

    if (import_20989 != 'pyd_module'):
        __import__(import_20989)
        sys_modules_20990 = sys.modules[import_20989]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'email.mime.base', sys_modules_20990.module_type_store, module_type_store, ['MIMEBase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_20990, sys_modules_20990.module_type_store, module_type_store)
    else:
        from email.mime.base import MIMEBase

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'email.mime.base', None, module_type_store, ['MIMEBase'], [MIMEBase])

else:
    # Assigning a type to the variable 'email.mime.base' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'email.mime.base', import_20989)

remove_current_file_folder_from_path('C:/Python27/lib/email/mime/')

# Declaration of the 'MIMEMultipart' class
# Getting the type of 'MIMEBase' (line 13)
MIMEBase_20991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 20), 'MIMEBase')

class MIMEMultipart(MIMEBase_20991, ):
    str_20992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'str', 'Base class for MIME multipart/* type messages.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_20993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 32), 'str', 'mixed')
        # Getting the type of 'None' (line 16)
        None_20994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 50), 'None')
        # Getting the type of 'None' (line 16)
        None_20995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 66), 'None')
        defaults = [str_20993, None_20994, None_20995]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 16, 4, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MIMEMultipart.__init__', ['_subtype', 'boundary', '_subparts'], None, '_params', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['_subtype', 'boundary', '_subparts'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_20996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, (-1)), 'str', "Creates a multipart/* type message.\n\n        By default, creates a multipart/mixed message, with proper\n        Content-Type and MIME-Version headers.\n\n        _subtype is the subtype of the multipart content type, defaulting to\n        `mixed'.\n\n        boundary is the multipart boundary string.  By default it is\n        calculated as needed.\n\n        _subparts is a sequence of initial subparts for the payload.  It\n        must be an iterable object, such as a list.  You can always\n        attach new subparts to the message by using the attach() method.\n\n        Additional parameters for the Content-Type header are taken from the\n        keyword arguments (or passed into the _params argument).\n        ")
        
        # Call to __init__(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'self' (line 36)
        self_20999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 26), 'self', False)
        str_21000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 32), 'str', 'multipart')
        # Getting the type of '_subtype' (line 36)
        _subtype_21001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 45), '_subtype', False)
        # Processing the call keyword arguments (line 36)
        # Getting the type of '_params' (line 36)
        _params_21002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 57), '_params', False)
        kwargs_21003 = {'_params_21002': _params_21002}
        # Getting the type of 'MIMEBase' (line 36)
        MIMEBase_20997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'MIMEBase', False)
        # Obtaining the member '__init__' of a type (line 36)
        init___20998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), MIMEBase_20997, '__init__')
        # Calling __init__(args, kwargs) (line 36)
        init___call_result_21004 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), init___20998, *[self_20999, str_21000, _subtype_21001], **kwargs_21003)
        
        
        # Assigning a List to a Attribute (line 41):
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_21005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        
        # Getting the type of 'self' (line 41)
        self_21006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self')
        # Setting the type of the member '_payload' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_21006, '_payload', list_21005)
        # Getting the type of '_subparts' (line 43)
        _subparts_21007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), '_subparts')
        # Testing if the type of an if condition is none (line 43)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 43, 8), _subparts_21007):
            pass
        else:
            
            # Testing the type of an if condition (line 43)
            if_condition_21008 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 8), _subparts_21007)
            # Assigning a type to the variable 'if_condition_21008' (line 43)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'if_condition_21008', if_condition_21008)
            # SSA begins for if statement (line 43)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of '_subparts' (line 44)
            _subparts_21009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 21), '_subparts')
            # Assigning a type to the variable '_subparts_21009' (line 44)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), '_subparts_21009', _subparts_21009)
            # Testing if the for loop is going to be iterated (line 44)
            # Testing the type of a for loop iterable (line 44)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 44, 12), _subparts_21009)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 44, 12), _subparts_21009):
                # Getting the type of the for loop variable (line 44)
                for_loop_var_21010 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 44, 12), _subparts_21009)
                # Assigning a type to the variable 'p' (line 44)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'p', for_loop_var_21010)
                # SSA begins for a for statement (line 44)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to attach(...): (line 45)
                # Processing the call arguments (line 45)
                # Getting the type of 'p' (line 45)
                p_21013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 28), 'p', False)
                # Processing the call keyword arguments (line 45)
                kwargs_21014 = {}
                # Getting the type of 'self' (line 45)
                self_21011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'self', False)
                # Obtaining the member 'attach' of a type (line 45)
                attach_21012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 16), self_21011, 'attach')
                # Calling attach(args, kwargs) (line 45)
                attach_call_result_21015 = invoke(stypy.reporting.localization.Localization(__file__, 45, 16), attach_21012, *[p_21013], **kwargs_21014)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 43)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'boundary' (line 46)
        boundary_21016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'boundary')
        # Testing if the type of an if condition is none (line 46)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 46, 8), boundary_21016):
            pass
        else:
            
            # Testing the type of an if condition (line 46)
            if_condition_21017 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 8), boundary_21016)
            # Assigning a type to the variable 'if_condition_21017' (line 46)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'if_condition_21017', if_condition_21017)
            # SSA begins for if statement (line 46)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to set_boundary(...): (line 47)
            # Processing the call arguments (line 47)
            # Getting the type of 'boundary' (line 47)
            boundary_21020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 30), 'boundary', False)
            # Processing the call keyword arguments (line 47)
            kwargs_21021 = {}
            # Getting the type of 'self' (line 47)
            self_21018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'self', False)
            # Obtaining the member 'set_boundary' of a type (line 47)
            set_boundary_21019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), self_21018, 'set_boundary')
            # Calling set_boundary(args, kwargs) (line 47)
            set_boundary_call_result_21022 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), set_boundary_21019, *[boundary_21020], **kwargs_21021)
            
            # SSA join for if statement (line 46)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'MIMEMultipart' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'MIMEMultipart', MIMEMultipart)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
