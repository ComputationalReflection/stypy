
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2001-2006 Python Software Foundation
2: # Author: Barry Warsaw
3: # Contact: email-sig@python.org
4: 
5: '''email package exception classes.'''
6: 
7: 
8: 
9: class MessageError(Exception):
10:     '''Base class for errors in the email package.'''
11: 
12: 
13: class MessageParseError(MessageError):
14:     '''Base class for message parsing errors.'''
15: 
16: 
17: class HeaderParseError(MessageParseError):
18:     '''Error while parsing headers.'''
19: 
20: 
21: class BoundaryError(MessageParseError):
22:     '''Couldn't find terminating boundary.'''
23: 
24: 
25: class MultipartConversionError(MessageError, TypeError):
26:     '''Conversion to a multipart is prohibited.'''
27: 
28: 
29: class CharsetError(MessageError):
30:     '''An illegal charset was given.'''
31: 
32: 
33: 
34: # These are parsing defects which the parser was able to work around.
35: class MessageDefect:
36:     '''Base class for a message defect.'''
37: 
38:     def __init__(self, line=None):
39:         self.line = line
40: 
41: class NoBoundaryInMultipartDefect(MessageDefect):
42:     '''A message claimed to be a multipart but had no boundary parameter.'''
43: 
44: class StartBoundaryNotFoundDefect(MessageDefect):
45:     '''The claimed start boundary was never found.'''
46: 
47: class FirstHeaderLineIsContinuationDefect(MessageDefect):
48:     '''A message had a continuation line as its first header line.'''
49: 
50: class MisplacedEnvelopeHeaderDefect(MessageDefect):
51:     '''A 'Unix-from' header was found in the middle of a header block.'''
52: 
53: class MalformedHeaderDefect(MessageDefect):
54:     '''Found a header that was missing a colon, or was otherwise malformed.'''
55: 
56: class MultipartInvariantViolationDefect(MessageDefect):
57:     '''A message claimed to be a multipart but no subparts were found.'''
58: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_12915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'str', 'email package exception classes.')
# Declaration of the 'MessageError' class
# Getting the type of 'Exception' (line 9)
Exception_12916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 19), 'Exception')

class MessageError(Exception_12916, ):
    str_12917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 4), 'str', 'Base class for errors in the email package.')

# Assigning a type to the variable 'MessageError' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'MessageError', MessageError)
# Declaration of the 'MessageParseError' class
# Getting the type of 'MessageError' (line 13)
MessageError_12918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 24), 'MessageError')

class MessageParseError(MessageError_12918, ):
    str_12919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'str', 'Base class for message parsing errors.')

# Assigning a type to the variable 'MessageParseError' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'MessageParseError', MessageParseError)
# Declaration of the 'HeaderParseError' class
# Getting the type of 'MessageParseError' (line 17)
MessageParseError_12920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 23), 'MessageParseError')

class HeaderParseError(MessageParseError_12920, ):
    str_12921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 4), 'str', 'Error while parsing headers.')

# Assigning a type to the variable 'HeaderParseError' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'HeaderParseError', HeaderParseError)
# Declaration of the 'BoundaryError' class
# Getting the type of 'MessageParseError' (line 21)
MessageParseError_12922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'MessageParseError')

class BoundaryError(MessageParseError_12922, ):
    str_12923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 4), 'str', "Couldn't find terminating boundary.")

# Assigning a type to the variable 'BoundaryError' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'BoundaryError', BoundaryError)
# Declaration of the 'MultipartConversionError' class
# Getting the type of 'MessageError' (line 25)
MessageError_12924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 31), 'MessageError')
# Getting the type of 'TypeError' (line 25)
TypeError_12925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 45), 'TypeError')

class MultipartConversionError(MessageError_12924, TypeError_12925, ):
    str_12926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 4), 'str', 'Conversion to a multipart is prohibited.')

# Assigning a type to the variable 'MultipartConversionError' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'MultipartConversionError', MultipartConversionError)
# Declaration of the 'CharsetError' class
# Getting the type of 'MessageError' (line 29)
MessageError_12927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'MessageError')

class CharsetError(MessageError_12927, ):
    str_12928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 4), 'str', 'An illegal charset was given.')

# Assigning a type to the variable 'CharsetError' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'CharsetError', CharsetError)
# Declaration of the 'MessageDefect' class

class MessageDefect:
    str_12929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'str', 'Base class for a message defect.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 38)
        None_12930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 28), 'None')
        defaults = [None_12930]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MessageDefect.__init__', ['line'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['line'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 39):
        # Getting the type of 'line' (line 39)
        line_12931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'line')
        # Getting the type of 'self' (line 39)
        self_12932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'self')
        # Setting the type of the member 'line' of a type (line 39)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), self_12932, 'line', line_12931)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'MessageDefect' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'MessageDefect', MessageDefect)
# Declaration of the 'NoBoundaryInMultipartDefect' class
# Getting the type of 'MessageDefect' (line 41)
MessageDefect_12933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 34), 'MessageDefect')

class NoBoundaryInMultipartDefect(MessageDefect_12933, ):
    str_12934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 4), 'str', 'A message claimed to be a multipart but had no boundary parameter.')

# Assigning a type to the variable 'NoBoundaryInMultipartDefect' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'NoBoundaryInMultipartDefect', NoBoundaryInMultipartDefect)
# Declaration of the 'StartBoundaryNotFoundDefect' class
# Getting the type of 'MessageDefect' (line 44)
MessageDefect_12935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 34), 'MessageDefect')

class StartBoundaryNotFoundDefect(MessageDefect_12935, ):
    str_12936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 4), 'str', 'The claimed start boundary was never found.')

# Assigning a type to the variable 'StartBoundaryNotFoundDefect' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'StartBoundaryNotFoundDefect', StartBoundaryNotFoundDefect)
# Declaration of the 'FirstHeaderLineIsContinuationDefect' class
# Getting the type of 'MessageDefect' (line 47)
MessageDefect_12937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 42), 'MessageDefect')

class FirstHeaderLineIsContinuationDefect(MessageDefect_12937, ):
    str_12938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 4), 'str', 'A message had a continuation line as its first header line.')

# Assigning a type to the variable 'FirstHeaderLineIsContinuationDefect' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'FirstHeaderLineIsContinuationDefect', FirstHeaderLineIsContinuationDefect)
# Declaration of the 'MisplacedEnvelopeHeaderDefect' class
# Getting the type of 'MessageDefect' (line 50)
MessageDefect_12939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 36), 'MessageDefect')

class MisplacedEnvelopeHeaderDefect(MessageDefect_12939, ):
    str_12940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 4), 'str', "A 'Unix-from' header was found in the middle of a header block.")

# Assigning a type to the variable 'MisplacedEnvelopeHeaderDefect' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'MisplacedEnvelopeHeaderDefect', MisplacedEnvelopeHeaderDefect)
# Declaration of the 'MalformedHeaderDefect' class
# Getting the type of 'MessageDefect' (line 53)
MessageDefect_12941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 28), 'MessageDefect')

class MalformedHeaderDefect(MessageDefect_12941, ):
    str_12942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 4), 'str', 'Found a header that was missing a colon, or was otherwise malformed.')

# Assigning a type to the variable 'MalformedHeaderDefect' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'MalformedHeaderDefect', MalformedHeaderDefect)
# Declaration of the 'MultipartInvariantViolationDefect' class
# Getting the type of 'MessageDefect' (line 56)
MessageDefect_12943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 40), 'MessageDefect')

class MultipartInvariantViolationDefect(MessageDefect_12943, ):
    str_12944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 4), 'str', 'A message claimed to be a multipart but no subparts were found.')

# Assigning a type to the variable 'MultipartInvariantViolationDefect' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'MultipartInvariantViolationDefect', MultipartInvariantViolationDefect)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
