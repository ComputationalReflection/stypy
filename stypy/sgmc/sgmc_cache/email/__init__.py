
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2001-2006 Python Software Foundation
2: # Author: Barry Warsaw
3: # Contact: email-sig@python.org
4: 
5: '''A package for parsing, handling, and generating email messages.'''
6: 
7: __version__ = '4.0.3'
8: 
9: __all__ = [
10:     # Old names
11:     'base64MIME',
12:     'Charset',
13:     'Encoders',
14:     'Errors',
15:     'Generator',
16:     'Header',
17:     'Iterators',
18:     'Message',
19:     'MIMEAudio',
20:     'MIMEBase',
21:     'MIMEImage',
22:     'MIMEMessage',
23:     'MIMEMultipart',
24:     'MIMENonMultipart',
25:     'MIMEText',
26:     'Parser',
27:     'quopriMIME',
28:     'Utils',
29:     'message_from_string',
30:     'message_from_file',
31:     # new names
32:     'base64mime',
33:     'charset',
34:     'encoders',
35:     'errors',
36:     'generator',
37:     'header',
38:     'iterators',
39:     'message',
40:     'mime',
41:     'parser',
42:     'quoprimime',
43:     'utils',
44:     ]
45: 
46: 
47: 
48: # Some convenience routines.  Don't import Parser and Message as side-effects
49: # of importing email since those cascadingly import most of the rest of the
50: # email package.
51: def message_from_string(s, *args, **kws):
52:     '''Parse a string into a Message object model.
53: 
54:     Optional _class and strict are passed to the Parser constructor.
55:     '''
56:     from email.parser import Parser
57:     return Parser(*args, **kws).parsestr(s)
58: 
59: 
60: def message_from_file(fp, *args, **kws):
61:     '''Read a file and parse its contents into a Message object model.
62: 
63:     Optional _class and strict are passed to the Parser constructor.
64:     '''
65:     from email.parser import Parser
66:     return Parser(*args, **kws).parse(fp)
67: 
68: 
69: 
70: # Lazy loading to provide name mapping from new-style names (PEP 8 compatible
71: # email 4.0 module names), to old-style names (email 3.0 module names).
72: import sys
73: 
74: class LazyImporter(object):
75:     def __init__(self, module_name):
76:         self.__name__ = 'email.' + module_name
77: 
78:     def __getattr__(self, name):
79:         __import__(self.__name__)
80:         mod = sys.modules[self.__name__]
81:         self.__dict__.update(mod.__dict__)
82:         return getattr(mod, name)
83: 
84: 
85: _LOWERNAMES = [
86:     # email.<old name> -> email.<new name is lowercased old name>
87:     'Charset',
88:     'Encoders',
89:     'Errors',
90:     'FeedParser',
91:     'Generator',
92:     'Header',
93:     'Iterators',
94:     'Message',
95:     'Parser',
96:     'Utils',
97:     'base64MIME',
98:     'quopriMIME',
99:     ]
100: 
101: _MIMENAMES = [
102:     # email.MIME<old name> -> email.mime.<new name is lowercased old name>
103:     'Audio',
104:     'Base',
105:     'Image',
106:     'Message',
107:     'Multipart',
108:     'NonMultipart',
109:     'Text',
110:     ]
111: 
112: for _name in _LOWERNAMES:
113:     importer = LazyImporter(_name.lower())
114:     sys.modules['email.' + _name] = importer
115:     setattr(sys.modules['email'], _name, importer)
116: 
117: 
118: import email.mime
119: for _name in _MIMENAMES:
120:     importer = LazyImporter('mime.' + _name.lower())
121:     sys.modules['email.MIME' + _name] = importer
122:     setattr(sys.modules['email'], 'MIME' + _name, importer)
123:     setattr(sys.modules['email.mime'], _name, importer)
124: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_20572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'str', 'A package for parsing, handling, and generating email messages.')

# Assigning a Str to a Name (line 7):
str_20573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'str', '4.0.3')
# Assigning a type to the variable '__version__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__version__', str_20573)

# Assigning a List to a Name (line 9):
__all__ = ['base64MIME', 'Charset', 'Encoders', 'Errors', 'Generator', 'Header', 'Iterators', 'Message', 'MIMEAudio', 'MIMEBase', 'MIMEImage', 'MIMEMessage', 'MIMEMultipart', 'MIMENonMultipart', 'MIMEText', 'Parser', 'quopriMIME', 'Utils', 'message_from_string', 'message_from_file', 'base64mime', 'charset', 'encoders', 'errors', 'generator', 'header', 'iterators', 'message', 'mime', 'parser', 'quoprimime', 'utils']
module_type_store.set_exportable_members(['base64MIME', 'Charset', 'Encoders', 'Errors', 'Generator', 'Header', 'Iterators', 'Message', 'MIMEAudio', 'MIMEBase', 'MIMEImage', 'MIMEMessage', 'MIMEMultipart', 'MIMENonMultipart', 'MIMEText', 'Parser', 'quopriMIME', 'Utils', 'message_from_string', 'message_from_file', 'base64mime', 'charset', 'encoders', 'errors', 'generator', 'header', 'iterators', 'message', 'mime', 'parser', 'quoprimime', 'utils'])

# Obtaining an instance of the builtin type 'list' (line 9)
list_20574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
str_20575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 4), 'str', 'base64MIME')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20575)
# Adding element type (line 9)
str_20576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 4), 'str', 'Charset')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20576)
# Adding element type (line 9)
str_20577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 4), 'str', 'Encoders')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20577)
# Adding element type (line 9)
str_20578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'str', 'Errors')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20578)
# Adding element type (line 9)
str_20579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 4), 'str', 'Generator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20579)
# Adding element type (line 9)
str_20580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'str', 'Header')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20580)
# Adding element type (line 9)
str_20581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 4), 'str', 'Iterators')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20581)
# Adding element type (line 9)
str_20582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 4), 'str', 'Message')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20582)
# Adding element type (line 9)
str_20583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 4), 'str', 'MIMEAudio')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20583)
# Adding element type (line 9)
str_20584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 4), 'str', 'MIMEBase')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20584)
# Adding element type (line 9)
str_20585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 4), 'str', 'MIMEImage')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20585)
# Adding element type (line 9)
str_20586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 4), 'str', 'MIMEMessage')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20586)
# Adding element type (line 9)
str_20587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 4), 'str', 'MIMEMultipart')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20587)
# Adding element type (line 9)
str_20588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 4), 'str', 'MIMENonMultipart')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20588)
# Adding element type (line 9)
str_20589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'str', 'MIMEText')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20589)
# Adding element type (line 9)
str_20590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 4), 'str', 'Parser')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20590)
# Adding element type (line 9)
str_20591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'str', 'quopriMIME')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20591)
# Adding element type (line 9)
str_20592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 4), 'str', 'Utils')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20592)
# Adding element type (line 9)
str_20593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'str', 'message_from_string')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20593)
# Adding element type (line 9)
str_20594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 4), 'str', 'message_from_file')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20594)
# Adding element type (line 9)
str_20595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 4), 'str', 'base64mime')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20595)
# Adding element type (line 9)
str_20596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 4), 'str', 'charset')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20596)
# Adding element type (line 9)
str_20597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 4), 'str', 'encoders')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20597)
# Adding element type (line 9)
str_20598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 4), 'str', 'errors')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20598)
# Adding element type (line 9)
str_20599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'str', 'generator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20599)
# Adding element type (line 9)
str_20600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 4), 'str', 'header')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20600)
# Adding element type (line 9)
str_20601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 4), 'str', 'iterators')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20601)
# Adding element type (line 9)
str_20602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 4), 'str', 'message')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20602)
# Adding element type (line 9)
str_20603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 4), 'str', 'mime')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20603)
# Adding element type (line 9)
str_20604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 4), 'str', 'parser')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20604)
# Adding element type (line 9)
str_20605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 4), 'str', 'quoprimime')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20605)
# Adding element type (line 9)
str_20606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 4), 'str', 'utils')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_20574, str_20606)

# Assigning a type to the variable '__all__' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), '__all__', list_20574)

@norecursion
def message_from_string(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'message_from_string'
    module_type_store = module_type_store.open_function_context('message_from_string', 51, 0, False)
    
    # Passed parameters checking function
    message_from_string.stypy_localization = localization
    message_from_string.stypy_type_of_self = None
    message_from_string.stypy_type_store = module_type_store
    message_from_string.stypy_function_name = 'message_from_string'
    message_from_string.stypy_param_names_list = ['s']
    message_from_string.stypy_varargs_param_name = 'args'
    message_from_string.stypy_kwargs_param_name = 'kws'
    message_from_string.stypy_call_defaults = defaults
    message_from_string.stypy_call_varargs = varargs
    message_from_string.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'message_from_string', ['s'], 'args', 'kws', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'message_from_string', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'message_from_string(...)' code ##################

    str_20607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, (-1)), 'str', 'Parse a string into a Message object model.\n\n    Optional _class and strict are passed to the Parser constructor.\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 56, 4))
    
    # 'from email.parser import Parser' statement (line 56)
    update_path_to_current_file_folder('C:/Python27/lib/email/')
    import_20608 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 56, 4), 'email.parser')

    if (type(import_20608) is not StypyTypeError):

        if (import_20608 != 'pyd_module'):
            __import__(import_20608)
            sys_modules_20609 = sys.modules[import_20608]
            import_from_module(stypy.reporting.localization.Localization(__file__, 56, 4), 'email.parser', sys_modules_20609.module_type_store, module_type_store, ['Parser'])
            nest_module(stypy.reporting.localization.Localization(__file__, 56, 4), __file__, sys_modules_20609, sys_modules_20609.module_type_store, module_type_store)
        else:
            from email.parser import Parser

            import_from_module(stypy.reporting.localization.Localization(__file__, 56, 4), 'email.parser', None, module_type_store, ['Parser'], [Parser])

    else:
        # Assigning a type to the variable 'email.parser' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'email.parser', import_20608)

    remove_current_file_folder_from_path('C:/Python27/lib/email/')
    
    
    # Call to parsestr(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 's' (line 57)
    s_20616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 41), 's', False)
    # Processing the call keyword arguments (line 57)
    kwargs_20617 = {}
    
    # Call to Parser(...): (line 57)
    # Getting the type of 'args' (line 57)
    args_20611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'args', False)
    # Processing the call keyword arguments (line 57)
    # Getting the type of 'kws' (line 57)
    kws_20612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 27), 'kws', False)
    kwargs_20613 = {'kws_20612': kws_20612}
    # Getting the type of 'Parser' (line 57)
    Parser_20610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'Parser', False)
    # Calling Parser(args, kwargs) (line 57)
    Parser_call_result_20614 = invoke(stypy.reporting.localization.Localization(__file__, 57, 11), Parser_20610, *[args_20611], **kwargs_20613)
    
    # Obtaining the member 'parsestr' of a type (line 57)
    parsestr_20615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 11), Parser_call_result_20614, 'parsestr')
    # Calling parsestr(args, kwargs) (line 57)
    parsestr_call_result_20618 = invoke(stypy.reporting.localization.Localization(__file__, 57, 11), parsestr_20615, *[s_20616], **kwargs_20617)
    
    # Assigning a type to the variable 'stypy_return_type' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type', parsestr_call_result_20618)
    
    # ################# End of 'message_from_string(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'message_from_string' in the type store
    # Getting the type of 'stypy_return_type' (line 51)
    stypy_return_type_20619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20619)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'message_from_string'
    return stypy_return_type_20619

# Assigning a type to the variable 'message_from_string' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'message_from_string', message_from_string)

@norecursion
def message_from_file(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'message_from_file'
    module_type_store = module_type_store.open_function_context('message_from_file', 60, 0, False)
    
    # Passed parameters checking function
    message_from_file.stypy_localization = localization
    message_from_file.stypy_type_of_self = None
    message_from_file.stypy_type_store = module_type_store
    message_from_file.stypy_function_name = 'message_from_file'
    message_from_file.stypy_param_names_list = ['fp']
    message_from_file.stypy_varargs_param_name = 'args'
    message_from_file.stypy_kwargs_param_name = 'kws'
    message_from_file.stypy_call_defaults = defaults
    message_from_file.stypy_call_varargs = varargs
    message_from_file.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'message_from_file', ['fp'], 'args', 'kws', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'message_from_file', localization, ['fp'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'message_from_file(...)' code ##################

    str_20620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, (-1)), 'str', 'Read a file and parse its contents into a Message object model.\n\n    Optional _class and strict are passed to the Parser constructor.\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 65, 4))
    
    # 'from email.parser import Parser' statement (line 65)
    update_path_to_current_file_folder('C:/Python27/lib/email/')
    import_20621 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 65, 4), 'email.parser')

    if (type(import_20621) is not StypyTypeError):

        if (import_20621 != 'pyd_module'):
            __import__(import_20621)
            sys_modules_20622 = sys.modules[import_20621]
            import_from_module(stypy.reporting.localization.Localization(__file__, 65, 4), 'email.parser', sys_modules_20622.module_type_store, module_type_store, ['Parser'])
            nest_module(stypy.reporting.localization.Localization(__file__, 65, 4), __file__, sys_modules_20622, sys_modules_20622.module_type_store, module_type_store)
        else:
            from email.parser import Parser

            import_from_module(stypy.reporting.localization.Localization(__file__, 65, 4), 'email.parser', None, module_type_store, ['Parser'], [Parser])

    else:
        # Assigning a type to the variable 'email.parser' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'email.parser', import_20621)

    remove_current_file_folder_from_path('C:/Python27/lib/email/')
    
    
    # Call to parse(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'fp' (line 66)
    fp_20629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 38), 'fp', False)
    # Processing the call keyword arguments (line 66)
    kwargs_20630 = {}
    
    # Call to Parser(...): (line 66)
    # Getting the type of 'args' (line 66)
    args_20624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 19), 'args', False)
    # Processing the call keyword arguments (line 66)
    # Getting the type of 'kws' (line 66)
    kws_20625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 27), 'kws', False)
    kwargs_20626 = {'kws_20625': kws_20625}
    # Getting the type of 'Parser' (line 66)
    Parser_20623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'Parser', False)
    # Calling Parser(args, kwargs) (line 66)
    Parser_call_result_20627 = invoke(stypy.reporting.localization.Localization(__file__, 66, 11), Parser_20623, *[args_20624], **kwargs_20626)
    
    # Obtaining the member 'parse' of a type (line 66)
    parse_20628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 11), Parser_call_result_20627, 'parse')
    # Calling parse(args, kwargs) (line 66)
    parse_call_result_20631 = invoke(stypy.reporting.localization.Localization(__file__, 66, 11), parse_20628, *[fp_20629], **kwargs_20630)
    
    # Assigning a type to the variable 'stypy_return_type' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type', parse_call_result_20631)
    
    # ################# End of 'message_from_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'message_from_file' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_20632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20632)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'message_from_file'
    return stypy_return_type_20632

# Assigning a type to the variable 'message_from_file' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'message_from_file', message_from_file)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 72, 0))

# 'import sys' statement (line 72)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 72, 0), 'sys', sys, module_type_store)

# Declaration of the 'LazyImporter' class

class LazyImporter(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 75, 4, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LazyImporter.__init__', ['module_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['module_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a BinOp to a Attribute (line 76):
        str_20633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 24), 'str', 'email.')
        # Getting the type of 'module_name' (line 76)
        module_name_20634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 35), 'module_name')
        # Applying the binary operator '+' (line 76)
        result_add_20635 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 24), '+', str_20633, module_name_20634)
        
        # Getting the type of 'self' (line 76)
        self_20636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self')
        # Setting the type of the member '__name__' of a type (line 76)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_20636, '__name__', result_add_20635)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __getattr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getattr__'
        module_type_store = module_type_store.open_function_context('__getattr__', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LazyImporter.__getattr__.__dict__.__setitem__('stypy_localization', localization)
        LazyImporter.__getattr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LazyImporter.__getattr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LazyImporter.__getattr__.__dict__.__setitem__('stypy_function_name', 'LazyImporter.__getattr__')
        LazyImporter.__getattr__.__dict__.__setitem__('stypy_param_names_list', ['name'])
        LazyImporter.__getattr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LazyImporter.__getattr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LazyImporter.__getattr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LazyImporter.__getattr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LazyImporter.__getattr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LazyImporter.__getattr__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LazyImporter.__getattr__', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getattr__', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getattr__(...)' code ##################

        
        # Call to __import__(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'self' (line 79)
        self_20638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), 'self', False)
        # Obtaining the member '__name__' of a type (line 79)
        name___20639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 19), self_20638, '__name__')
        # Processing the call keyword arguments (line 79)
        kwargs_20640 = {}
        # Getting the type of '__import__' (line 79)
        import___20637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), '__import__', False)
        # Calling __import__(args, kwargs) (line 79)
        import___call_result_20641 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), import___20637, *[name___20639], **kwargs_20640)
        
        
        # Assigning a Subscript to a Name (line 80):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 80)
        self_20642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 26), 'self')
        # Obtaining the member '__name__' of a type (line 80)
        name___20643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 26), self_20642, '__name__')
        # Getting the type of 'sys' (line 80)
        sys_20644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 14), 'sys')
        # Obtaining the member 'modules' of a type (line 80)
        modules_20645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 14), sys_20644, 'modules')
        # Obtaining the member '__getitem__' of a type (line 80)
        getitem___20646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 14), modules_20645, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 80)
        subscript_call_result_20647 = invoke(stypy.reporting.localization.Localization(__file__, 80, 14), getitem___20646, name___20643)
        
        # Assigning a type to the variable 'mod' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'mod', subscript_call_result_20647)
        
        # Call to update(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'mod' (line 81)
        mod_20651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 29), 'mod', False)
        # Obtaining the member '__dict__' of a type (line 81)
        dict___20652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 29), mod_20651, '__dict__')
        # Processing the call keyword arguments (line 81)
        kwargs_20653 = {}
        # Getting the type of 'self' (line 81)
        self_20648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'self', False)
        # Obtaining the member '__dict__' of a type (line 81)
        dict___20649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), self_20648, '__dict__')
        # Obtaining the member 'update' of a type (line 81)
        update_20650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), dict___20649, 'update')
        # Calling update(args, kwargs) (line 81)
        update_call_result_20654 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), update_20650, *[dict___20652], **kwargs_20653)
        
        
        # Call to getattr(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'mod' (line 82)
        mod_20656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 23), 'mod', False)
        # Getting the type of 'name' (line 82)
        name_20657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 28), 'name', False)
        # Processing the call keyword arguments (line 82)
        kwargs_20658 = {}
        # Getting the type of 'getattr' (line 82)
        getattr_20655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 82)
        getattr_call_result_20659 = invoke(stypy.reporting.localization.Localization(__file__, 82, 15), getattr_20655, *[mod_20656, name_20657], **kwargs_20658)
        
        # Assigning a type to the variable 'stypy_return_type' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stypy_return_type', getattr_call_result_20659)
        
        # ################# End of '__getattr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getattr__' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_20660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20660)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getattr__'
        return stypy_return_type_20660


# Assigning a type to the variable 'LazyImporter' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'LazyImporter', LazyImporter)

# Assigning a List to a Name (line 85):

# Obtaining an instance of the builtin type 'list' (line 85)
list_20661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 85)
# Adding element type (line 85)
str_20662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 4), 'str', 'Charset')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 14), list_20661, str_20662)
# Adding element type (line 85)
str_20663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 4), 'str', 'Encoders')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 14), list_20661, str_20663)
# Adding element type (line 85)
str_20664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 4), 'str', 'Errors')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 14), list_20661, str_20664)
# Adding element type (line 85)
str_20665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 4), 'str', 'FeedParser')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 14), list_20661, str_20665)
# Adding element type (line 85)
str_20666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 4), 'str', 'Generator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 14), list_20661, str_20666)
# Adding element type (line 85)
str_20667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 4), 'str', 'Header')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 14), list_20661, str_20667)
# Adding element type (line 85)
str_20668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 4), 'str', 'Iterators')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 14), list_20661, str_20668)
# Adding element type (line 85)
str_20669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 4), 'str', 'Message')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 14), list_20661, str_20669)
# Adding element type (line 85)
str_20670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 4), 'str', 'Parser')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 14), list_20661, str_20670)
# Adding element type (line 85)
str_20671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 4), 'str', 'Utils')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 14), list_20661, str_20671)
# Adding element type (line 85)
str_20672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 4), 'str', 'base64MIME')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 14), list_20661, str_20672)
# Adding element type (line 85)
str_20673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'str', 'quopriMIME')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 14), list_20661, str_20673)

# Assigning a type to the variable '_LOWERNAMES' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), '_LOWERNAMES', list_20661)

# Assigning a List to a Name (line 101):

# Obtaining an instance of the builtin type 'list' (line 101)
list_20674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 101)
# Adding element type (line 101)
str_20675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 4), 'str', 'Audio')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 13), list_20674, str_20675)
# Adding element type (line 101)
str_20676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 4), 'str', 'Base')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 13), list_20674, str_20676)
# Adding element type (line 101)
str_20677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 4), 'str', 'Image')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 13), list_20674, str_20677)
# Adding element type (line 101)
str_20678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 4), 'str', 'Message')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 13), list_20674, str_20678)
# Adding element type (line 101)
str_20679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 4), 'str', 'Multipart')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 13), list_20674, str_20679)
# Adding element type (line 101)
str_20680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 4), 'str', 'NonMultipart')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 13), list_20674, str_20680)
# Adding element type (line 101)
str_20681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 4), 'str', 'Text')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 13), list_20674, str_20681)

# Assigning a type to the variable '_MIMENAMES' (line 101)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), '_MIMENAMES', list_20674)

# Getting the type of '_LOWERNAMES' (line 112)
_LOWERNAMES_20682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 13), '_LOWERNAMES')
# Assigning a type to the variable '_LOWERNAMES_20682' (line 112)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 0), '_LOWERNAMES_20682', _LOWERNAMES_20682)
# Testing if the for loop is going to be iterated (line 112)
# Testing the type of a for loop iterable (line 112)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 112, 0), _LOWERNAMES_20682)

if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 112, 0), _LOWERNAMES_20682):
    # Getting the type of the for loop variable (line 112)
    for_loop_var_20683 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 112, 0), _LOWERNAMES_20682)
    # Assigning a type to the variable '_name' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 0), '_name', for_loop_var_20683)
    # SSA begins for a for statement (line 112)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 113):
    
    # Call to LazyImporter(...): (line 113)
    # Processing the call arguments (line 113)
    
    # Call to lower(...): (line 113)
    # Processing the call keyword arguments (line 113)
    kwargs_20687 = {}
    # Getting the type of '_name' (line 113)
    _name_20685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 28), '_name', False)
    # Obtaining the member 'lower' of a type (line 113)
    lower_20686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 28), _name_20685, 'lower')
    # Calling lower(args, kwargs) (line 113)
    lower_call_result_20688 = invoke(stypy.reporting.localization.Localization(__file__, 113, 28), lower_20686, *[], **kwargs_20687)
    
    # Processing the call keyword arguments (line 113)
    kwargs_20689 = {}
    # Getting the type of 'LazyImporter' (line 113)
    LazyImporter_20684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'LazyImporter', False)
    # Calling LazyImporter(args, kwargs) (line 113)
    LazyImporter_call_result_20690 = invoke(stypy.reporting.localization.Localization(__file__, 113, 15), LazyImporter_20684, *[lower_call_result_20688], **kwargs_20689)
    
    # Assigning a type to the variable 'importer' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'importer', LazyImporter_call_result_20690)
    
    # Assigning a Name to a Subscript (line 114):
    # Getting the type of 'importer' (line 114)
    importer_20691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 36), 'importer')
    # Getting the type of 'sys' (line 114)
    sys_20692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'sys')
    # Obtaining the member 'modules' of a type (line 114)
    modules_20693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 4), sys_20692, 'modules')
    str_20694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 16), 'str', 'email.')
    # Getting the type of '_name' (line 114)
    _name_20695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 27), '_name')
    # Applying the binary operator '+' (line 114)
    result_add_20696 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 16), '+', str_20694, _name_20695)
    
    # Storing an element on a container (line 114)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 4), modules_20693, (result_add_20696, importer_20691))
    
    # Call to setattr(...): (line 115)
    # Processing the call arguments (line 115)
    
    # Obtaining the type of the subscript
    str_20698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 24), 'str', 'email')
    # Getting the type of 'sys' (line 115)
    sys_20699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'sys', False)
    # Obtaining the member 'modules' of a type (line 115)
    modules_20700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), sys_20699, 'modules')
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___20701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), modules_20700, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 115)
    subscript_call_result_20702 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), getitem___20701, str_20698)
    
    # Getting the type of '_name' (line 115)
    _name_20703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 34), '_name', False)
    # Getting the type of 'importer' (line 115)
    importer_20704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 41), 'importer', False)
    # Processing the call keyword arguments (line 115)
    kwargs_20705 = {}
    # Getting the type of 'setattr' (line 115)
    setattr_20697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'setattr', False)
    # Calling setattr(args, kwargs) (line 115)
    setattr_call_result_20706 = invoke(stypy.reporting.localization.Localization(__file__, 115, 4), setattr_20697, *[subscript_call_result_20702, _name_20703, importer_20704], **kwargs_20705)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()


stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 118, 0))

# 'import email.mime' statement (line 118)
update_path_to_current_file_folder('C:/Python27/lib/email/')
import_20707 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 118, 0), 'email.mime')

if (type(import_20707) is not StypyTypeError):

    if (import_20707 != 'pyd_module'):
        __import__(import_20707)
        sys_modules_20708 = sys.modules[import_20707]
        import_module(stypy.reporting.localization.Localization(__file__, 118, 0), 'email.mime', sys_modules_20708.module_type_store, module_type_store)
    else:
        import email.mime

        import_module(stypy.reporting.localization.Localization(__file__, 118, 0), 'email.mime', email.mime, module_type_store)

else:
    # Assigning a type to the variable 'email.mime' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'email.mime', import_20707)

remove_current_file_folder_from_path('C:/Python27/lib/email/')


# Getting the type of '_MIMENAMES' (line 119)
_MIMENAMES_20709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 13), '_MIMENAMES')
# Assigning a type to the variable '_MIMENAMES_20709' (line 119)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), '_MIMENAMES_20709', _MIMENAMES_20709)
# Testing if the for loop is going to be iterated (line 119)
# Testing the type of a for loop iterable (line 119)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 119, 0), _MIMENAMES_20709)

if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 119, 0), _MIMENAMES_20709):
    # Getting the type of the for loop variable (line 119)
    for_loop_var_20710 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 119, 0), _MIMENAMES_20709)
    # Assigning a type to the variable '_name' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), '_name', for_loop_var_20710)
    # SSA begins for a for statement (line 119)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 120):
    
    # Call to LazyImporter(...): (line 120)
    # Processing the call arguments (line 120)
    str_20712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 28), 'str', 'mime.')
    
    # Call to lower(...): (line 120)
    # Processing the call keyword arguments (line 120)
    kwargs_20715 = {}
    # Getting the type of '_name' (line 120)
    _name_20713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 38), '_name', False)
    # Obtaining the member 'lower' of a type (line 120)
    lower_20714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 38), _name_20713, 'lower')
    # Calling lower(args, kwargs) (line 120)
    lower_call_result_20716 = invoke(stypy.reporting.localization.Localization(__file__, 120, 38), lower_20714, *[], **kwargs_20715)
    
    # Applying the binary operator '+' (line 120)
    result_add_20717 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 28), '+', str_20712, lower_call_result_20716)
    
    # Processing the call keyword arguments (line 120)
    kwargs_20718 = {}
    # Getting the type of 'LazyImporter' (line 120)
    LazyImporter_20711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'LazyImporter', False)
    # Calling LazyImporter(args, kwargs) (line 120)
    LazyImporter_call_result_20719 = invoke(stypy.reporting.localization.Localization(__file__, 120, 15), LazyImporter_20711, *[result_add_20717], **kwargs_20718)
    
    # Assigning a type to the variable 'importer' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'importer', LazyImporter_call_result_20719)
    
    # Assigning a Name to a Subscript (line 121):
    # Getting the type of 'importer' (line 121)
    importer_20720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 40), 'importer')
    # Getting the type of 'sys' (line 121)
    sys_20721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'sys')
    # Obtaining the member 'modules' of a type (line 121)
    modules_20722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 4), sys_20721, 'modules')
    str_20723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 16), 'str', 'email.MIME')
    # Getting the type of '_name' (line 121)
    _name_20724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 31), '_name')
    # Applying the binary operator '+' (line 121)
    result_add_20725 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 16), '+', str_20723, _name_20724)
    
    # Storing an element on a container (line 121)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 4), modules_20722, (result_add_20725, importer_20720))
    
    # Call to setattr(...): (line 122)
    # Processing the call arguments (line 122)
    
    # Obtaining the type of the subscript
    str_20727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 24), 'str', 'email')
    # Getting the type of 'sys' (line 122)
    sys_20728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'sys', False)
    # Obtaining the member 'modules' of a type (line 122)
    modules_20729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 12), sys_20728, 'modules')
    # Obtaining the member '__getitem__' of a type (line 122)
    getitem___20730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 12), modules_20729, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 122)
    subscript_call_result_20731 = invoke(stypy.reporting.localization.Localization(__file__, 122, 12), getitem___20730, str_20727)
    
    str_20732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 34), 'str', 'MIME')
    # Getting the type of '_name' (line 122)
    _name_20733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 43), '_name', False)
    # Applying the binary operator '+' (line 122)
    result_add_20734 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 34), '+', str_20732, _name_20733)
    
    # Getting the type of 'importer' (line 122)
    importer_20735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 50), 'importer', False)
    # Processing the call keyword arguments (line 122)
    kwargs_20736 = {}
    # Getting the type of 'setattr' (line 122)
    setattr_20726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'setattr', False)
    # Calling setattr(args, kwargs) (line 122)
    setattr_call_result_20737 = invoke(stypy.reporting.localization.Localization(__file__, 122, 4), setattr_20726, *[subscript_call_result_20731, result_add_20734, importer_20735], **kwargs_20736)
    
    
    # Call to setattr(...): (line 123)
    # Processing the call arguments (line 123)
    
    # Obtaining the type of the subscript
    str_20739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 24), 'str', 'email.mime')
    # Getting the type of 'sys' (line 123)
    sys_20740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'sys', False)
    # Obtaining the member 'modules' of a type (line 123)
    modules_20741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 12), sys_20740, 'modules')
    # Obtaining the member '__getitem__' of a type (line 123)
    getitem___20742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 12), modules_20741, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 123)
    subscript_call_result_20743 = invoke(stypy.reporting.localization.Localization(__file__, 123, 12), getitem___20742, str_20739)
    
    # Getting the type of '_name' (line 123)
    _name_20744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 39), '_name', False)
    # Getting the type of 'importer' (line 123)
    importer_20745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 46), 'importer', False)
    # Processing the call keyword arguments (line 123)
    kwargs_20746 = {}
    # Getting the type of 'setattr' (line 123)
    setattr_20738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'setattr', False)
    # Calling setattr(args, kwargs) (line 123)
    setattr_call_result_20747 = invoke(stypy.reporting.localization.Localization(__file__, 123, 4), setattr_20738, *[subscript_call_result_20743, _name_20744, importer_20745], **kwargs_20746)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()



# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
