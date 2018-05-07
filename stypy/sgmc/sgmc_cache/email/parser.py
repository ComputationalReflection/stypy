
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2001-2006 Python Software Foundation
2: # Author: Barry Warsaw, Thomas Wouters, Anthony Baxter
3: # Contact: email-sig@python.org
4: 
5: '''A parser of RFC 2822 and MIME email messages.'''
6: 
7: __all__ = ['Parser', 'HeaderParser']
8: 
9: import warnings
10: from cStringIO import StringIO
11: 
12: from email.feedparser import FeedParser
13: from email.message import Message
14: 
15: 
16: 
17: class Parser:
18:     def __init__(self, *args, **kws):
19:         '''Parser of RFC 2822 and MIME email messages.
20: 
21:         Creates an in-memory object tree representing the email message, which
22:         can then be manipulated and turned over to a Generator to return the
23:         textual representation of the message.
24: 
25:         The string must be formatted as a block of RFC 2822 headers and header
26:         continuation lines, optionally preceded by a `Unix-from' header.  The
27:         header block is terminated either by the end of the string or by a
28:         blank line.
29: 
30:         _class is the class to instantiate for new message objects when they
31:         must be created.  This class must have a constructor that can take
32:         zero arguments.  Default is Message.Message.
33:         '''
34:         if len(args) >= 1:
35:             if '_class' in kws:
36:                 raise TypeError("Multiple values for keyword arg '_class'")
37:             kws['_class'] = args[0]
38:         if len(args) == 2:
39:             if 'strict' in kws:
40:                 raise TypeError("Multiple values for keyword arg 'strict'")
41:             kws['strict'] = args[1]
42:         if len(args) > 2:
43:             raise TypeError('Too many arguments')
44:         if '_class' in kws:
45:             self._class = kws['_class']
46:             del kws['_class']
47:         else:
48:             self._class = Message
49:         if 'strict' in kws:
50:             warnings.warn("'strict' argument is deprecated (and ignored)",
51:                           DeprecationWarning, 2)
52:             del kws['strict']
53:         if kws:
54:             raise TypeError('Unexpected keyword arguments')
55: 
56:     def parse(self, fp, headersonly=False):
57:         '''Create a message structure from the data in a file.
58: 
59:         Reads all the data from the file and returns the root of the message
60:         structure.  Optional headersonly is a flag specifying whether to stop
61:         parsing after reading the headers or not.  The default is False,
62:         meaning it parses the entire contents of the file.
63:         '''
64:         feedparser = FeedParser(self._class)
65:         if headersonly:
66:             feedparser._set_headersonly()
67:         while True:
68:             data = fp.read(8192)
69:             if not data:
70:                 break
71:             feedparser.feed(data)
72:         return feedparser.close()
73: 
74:     def parsestr(self, text, headersonly=False):
75:         '''Create a message structure from a string.
76: 
77:         Returns the root of the message structure.  Optional headersonly is a
78:         flag specifying whether to stop parsing after reading the headers or
79:         not.  The default is False, meaning it parses the entire contents of
80:         the file.
81:         '''
82:         return self.parse(StringIO(text), headersonly=headersonly)
83: 
84: 
85: 
86: class HeaderParser(Parser):
87:     def parse(self, fp, headersonly=True):
88:         return Parser.parse(self, fp, True)
89: 
90:     def parsestr(self, text, headersonly=True):
91:         return Parser.parsestr(self, text, True)
92: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_17567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'str', 'A parser of RFC 2822 and MIME email messages.')

# Assigning a List to a Name (line 7):
__all__ = ['Parser', 'HeaderParser']
module_type_store.set_exportable_members(['Parser', 'HeaderParser'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_17568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_17569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'Parser')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_17568, str_17569)
# Adding element type (line 7)
str_17570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 21), 'str', 'HeaderParser')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_17568, str_17570)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_17568)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import warnings' statement (line 9)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from cStringIO import StringIO' statement (line 10)
try:
    from cStringIO import StringIO

except:
    StringIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'cStringIO', None, module_type_store, ['StringIO'], [StringIO])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from email.feedparser import FeedParser' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/email/')
import_17571 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'email.feedparser')

if (type(import_17571) is not StypyTypeError):

    if (import_17571 != 'pyd_module'):
        __import__(import_17571)
        sys_modules_17572 = sys.modules[import_17571]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'email.feedparser', sys_modules_17572.module_type_store, module_type_store, ['FeedParser'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_17572, sys_modules_17572.module_type_store, module_type_store)
    else:
        from email.feedparser import FeedParser

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'email.feedparser', None, module_type_store, ['FeedParser'], [FeedParser])

else:
    # Assigning a type to the variable 'email.feedparser' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'email.feedparser', import_17571)

remove_current_file_folder_from_path('C:/Python27/lib/email/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from email.message import Message' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/email/')
import_17573 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'email.message')

if (type(import_17573) is not StypyTypeError):

    if (import_17573 != 'pyd_module'):
        __import__(import_17573)
        sys_modules_17574 = sys.modules[import_17573]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'email.message', sys_modules_17574.module_type_store, module_type_store, ['Message'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_17574, sys_modules_17574.module_type_store, module_type_store)
    else:
        from email.message import Message

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'email.message', None, module_type_store, ['Message'], [Message])

else:
    # Assigning a type to the variable 'email.message' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'email.message', import_17573)

remove_current_file_folder_from_path('C:/Python27/lib/email/')

# Declaration of the 'Parser' class

class Parser:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Parser.__init__', [], 'args', 'kws', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_17575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, (-1)), 'str', "Parser of RFC 2822 and MIME email messages.\n\n        Creates an in-memory object tree representing the email message, which\n        can then be manipulated and turned over to a Generator to return the\n        textual representation of the message.\n\n        The string must be formatted as a block of RFC 2822 headers and header\n        continuation lines, optionally preceded by a `Unix-from' header.  The\n        header block is terminated either by the end of the string or by a\n        blank line.\n\n        _class is the class to instantiate for new message objects when they\n        must be created.  This class must have a constructor that can take\n        zero arguments.  Default is Message.Message.\n        ")
        
        
        # Call to len(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'args' (line 34)
        args_17577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'args', False)
        # Processing the call keyword arguments (line 34)
        kwargs_17578 = {}
        # Getting the type of 'len' (line 34)
        len_17576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'len', False)
        # Calling len(args, kwargs) (line 34)
        len_call_result_17579 = invoke(stypy.reporting.localization.Localization(__file__, 34, 11), len_17576, *[args_17577], **kwargs_17578)
        
        int_17580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 24), 'int')
        # Applying the binary operator '>=' (line 34)
        result_ge_17581 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 11), '>=', len_call_result_17579, int_17580)
        
        # Testing if the type of an if condition is none (line 34)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 34, 8), result_ge_17581):
            pass
        else:
            
            # Testing the type of an if condition (line 34)
            if_condition_17582 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 8), result_ge_17581)
            # Assigning a type to the variable 'if_condition_17582' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'if_condition_17582', if_condition_17582)
            # SSA begins for if statement (line 34)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            str_17583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 15), 'str', '_class')
            # Getting the type of 'kws' (line 35)
            kws_17584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 27), 'kws')
            # Applying the binary operator 'in' (line 35)
            result_contains_17585 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 15), 'in', str_17583, kws_17584)
            
            # Testing if the type of an if condition is none (line 35)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 35, 12), result_contains_17585):
                pass
            else:
                
                # Testing the type of an if condition (line 35)
                if_condition_17586 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 12), result_contains_17585)
                # Assigning a type to the variable 'if_condition_17586' (line 35)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'if_condition_17586', if_condition_17586)
                # SSA begins for if statement (line 35)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to TypeError(...): (line 36)
                # Processing the call arguments (line 36)
                str_17588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 32), 'str', "Multiple values for keyword arg '_class'")
                # Processing the call keyword arguments (line 36)
                kwargs_17589 = {}
                # Getting the type of 'TypeError' (line 36)
                TypeError_17587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 22), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 36)
                TypeError_call_result_17590 = invoke(stypy.reporting.localization.Localization(__file__, 36, 22), TypeError_17587, *[str_17588], **kwargs_17589)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 36, 16), TypeError_call_result_17590, 'raise parameter', BaseException)
                # SSA join for if statement (line 35)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Subscript to a Subscript (line 37):
            
            # Obtaining the type of the subscript
            int_17591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 33), 'int')
            # Getting the type of 'args' (line 37)
            args_17592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 28), 'args')
            # Obtaining the member '__getitem__' of a type (line 37)
            getitem___17593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 28), args_17592, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 37)
            subscript_call_result_17594 = invoke(stypy.reporting.localization.Localization(__file__, 37, 28), getitem___17593, int_17591)
            
            # Getting the type of 'kws' (line 37)
            kws_17595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'kws')
            str_17596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 16), 'str', '_class')
            # Storing an element on a container (line 37)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 12), kws_17595, (str_17596, subscript_call_result_17594))
            # SSA join for if statement (line 34)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to len(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'args' (line 38)
        args_17598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'args', False)
        # Processing the call keyword arguments (line 38)
        kwargs_17599 = {}
        # Getting the type of 'len' (line 38)
        len_17597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'len', False)
        # Calling len(args, kwargs) (line 38)
        len_call_result_17600 = invoke(stypy.reporting.localization.Localization(__file__, 38, 11), len_17597, *[args_17598], **kwargs_17599)
        
        int_17601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 24), 'int')
        # Applying the binary operator '==' (line 38)
        result_eq_17602 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 11), '==', len_call_result_17600, int_17601)
        
        # Testing if the type of an if condition is none (line 38)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 38, 8), result_eq_17602):
            pass
        else:
            
            # Testing the type of an if condition (line 38)
            if_condition_17603 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 8), result_eq_17602)
            # Assigning a type to the variable 'if_condition_17603' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'if_condition_17603', if_condition_17603)
            # SSA begins for if statement (line 38)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            str_17604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 15), 'str', 'strict')
            # Getting the type of 'kws' (line 39)
            kws_17605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), 'kws')
            # Applying the binary operator 'in' (line 39)
            result_contains_17606 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 15), 'in', str_17604, kws_17605)
            
            # Testing if the type of an if condition is none (line 39)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 39, 12), result_contains_17606):
                pass
            else:
                
                # Testing the type of an if condition (line 39)
                if_condition_17607 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 12), result_contains_17606)
                # Assigning a type to the variable 'if_condition_17607' (line 39)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'if_condition_17607', if_condition_17607)
                # SSA begins for if statement (line 39)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to TypeError(...): (line 40)
                # Processing the call arguments (line 40)
                str_17609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 32), 'str', "Multiple values for keyword arg 'strict'")
                # Processing the call keyword arguments (line 40)
                kwargs_17610 = {}
                # Getting the type of 'TypeError' (line 40)
                TypeError_17608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 22), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 40)
                TypeError_call_result_17611 = invoke(stypy.reporting.localization.Localization(__file__, 40, 22), TypeError_17608, *[str_17609], **kwargs_17610)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 40, 16), TypeError_call_result_17611, 'raise parameter', BaseException)
                # SSA join for if statement (line 39)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Subscript to a Subscript (line 41):
            
            # Obtaining the type of the subscript
            int_17612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 33), 'int')
            # Getting the type of 'args' (line 41)
            args_17613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 28), 'args')
            # Obtaining the member '__getitem__' of a type (line 41)
            getitem___17614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 28), args_17613, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 41)
            subscript_call_result_17615 = invoke(stypy.reporting.localization.Localization(__file__, 41, 28), getitem___17614, int_17612)
            
            # Getting the type of 'kws' (line 41)
            kws_17616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'kws')
            str_17617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 16), 'str', 'strict')
            # Storing an element on a container (line 41)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 12), kws_17616, (str_17617, subscript_call_result_17615))
            # SSA join for if statement (line 38)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to len(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'args' (line 42)
        args_17619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'args', False)
        # Processing the call keyword arguments (line 42)
        kwargs_17620 = {}
        # Getting the type of 'len' (line 42)
        len_17618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'len', False)
        # Calling len(args, kwargs) (line 42)
        len_call_result_17621 = invoke(stypy.reporting.localization.Localization(__file__, 42, 11), len_17618, *[args_17619], **kwargs_17620)
        
        int_17622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 23), 'int')
        # Applying the binary operator '>' (line 42)
        result_gt_17623 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 11), '>', len_call_result_17621, int_17622)
        
        # Testing if the type of an if condition is none (line 42)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 42, 8), result_gt_17623):
            pass
        else:
            
            # Testing the type of an if condition (line 42)
            if_condition_17624 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 8), result_gt_17623)
            # Assigning a type to the variable 'if_condition_17624' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'if_condition_17624', if_condition_17624)
            # SSA begins for if statement (line 42)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 43)
            # Processing the call arguments (line 43)
            str_17626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 28), 'str', 'Too many arguments')
            # Processing the call keyword arguments (line 43)
            kwargs_17627 = {}
            # Getting the type of 'TypeError' (line 43)
            TypeError_17625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 18), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 43)
            TypeError_call_result_17628 = invoke(stypy.reporting.localization.Localization(__file__, 43, 18), TypeError_17625, *[str_17626], **kwargs_17627)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 43, 12), TypeError_call_result_17628, 'raise parameter', BaseException)
            # SSA join for if statement (line 42)
            module_type_store = module_type_store.join_ssa_context()
            

        
        str_17629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 11), 'str', '_class')
        # Getting the type of 'kws' (line 44)
        kws_17630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'kws')
        # Applying the binary operator 'in' (line 44)
        result_contains_17631 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 11), 'in', str_17629, kws_17630)
        
        # Testing if the type of an if condition is none (line 44)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 44, 8), result_contains_17631):
            
            # Assigning a Name to a Attribute (line 48):
            # Getting the type of 'Message' (line 48)
            Message_17643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'Message')
            # Getting the type of 'self' (line 48)
            self_17644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'self')
            # Setting the type of the member '_class' of a type (line 48)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), self_17644, '_class', Message_17643)
        else:
            
            # Testing the type of an if condition (line 44)
            if_condition_17632 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 8), result_contains_17631)
            # Assigning a type to the variable 'if_condition_17632' (line 44)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'if_condition_17632', if_condition_17632)
            # SSA begins for if statement (line 44)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Attribute (line 45):
            
            # Obtaining the type of the subscript
            str_17633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 30), 'str', '_class')
            # Getting the type of 'kws' (line 45)
            kws_17634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 26), 'kws')
            # Obtaining the member '__getitem__' of a type (line 45)
            getitem___17635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 26), kws_17634, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 45)
            subscript_call_result_17636 = invoke(stypy.reporting.localization.Localization(__file__, 45, 26), getitem___17635, str_17633)
            
            # Getting the type of 'self' (line 45)
            self_17637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'self')
            # Setting the type of the member '_class' of a type (line 45)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), self_17637, '_class', subscript_call_result_17636)
            # Deleting a member
            # Getting the type of 'kws' (line 46)
            kws_17638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'kws')
            
            # Obtaining the type of the subscript
            str_17639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 20), 'str', '_class')
            # Getting the type of 'kws' (line 46)
            kws_17640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'kws')
            # Obtaining the member '__getitem__' of a type (line 46)
            getitem___17641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 16), kws_17640, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 46)
            subscript_call_result_17642 = invoke(stypy.reporting.localization.Localization(__file__, 46, 16), getitem___17641, str_17639)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 12), kws_17638, subscript_call_result_17642)
            # SSA branch for the else part of an if statement (line 44)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Attribute (line 48):
            # Getting the type of 'Message' (line 48)
            Message_17643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'Message')
            # Getting the type of 'self' (line 48)
            self_17644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'self')
            # Setting the type of the member '_class' of a type (line 48)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), self_17644, '_class', Message_17643)
            # SSA join for if statement (line 44)
            module_type_store = module_type_store.join_ssa_context()
            

        
        str_17645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 11), 'str', 'strict')
        # Getting the type of 'kws' (line 49)
        kws_17646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 23), 'kws')
        # Applying the binary operator 'in' (line 49)
        result_contains_17647 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 11), 'in', str_17645, kws_17646)
        
        # Testing if the type of an if condition is none (line 49)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 49, 8), result_contains_17647):
            pass
        else:
            
            # Testing the type of an if condition (line 49)
            if_condition_17648 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 8), result_contains_17647)
            # Assigning a type to the variable 'if_condition_17648' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'if_condition_17648', if_condition_17648)
            # SSA begins for if statement (line 49)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to warn(...): (line 50)
            # Processing the call arguments (line 50)
            str_17651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 26), 'str', "'strict' argument is deprecated (and ignored)")
            # Getting the type of 'DeprecationWarning' (line 51)
            DeprecationWarning_17652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 26), 'DeprecationWarning', False)
            int_17653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 46), 'int')
            # Processing the call keyword arguments (line 50)
            kwargs_17654 = {}
            # Getting the type of 'warnings' (line 50)
            warnings_17649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'warnings', False)
            # Obtaining the member 'warn' of a type (line 50)
            warn_17650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), warnings_17649, 'warn')
            # Calling warn(args, kwargs) (line 50)
            warn_call_result_17655 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), warn_17650, *[str_17651, DeprecationWarning_17652, int_17653], **kwargs_17654)
            
            # Deleting a member
            # Getting the type of 'kws' (line 52)
            kws_17656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'kws')
            
            # Obtaining the type of the subscript
            str_17657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 20), 'str', 'strict')
            # Getting the type of 'kws' (line 52)
            kws_17658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'kws')
            # Obtaining the member '__getitem__' of a type (line 52)
            getitem___17659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 16), kws_17658, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 52)
            subscript_call_result_17660 = invoke(stypy.reporting.localization.Localization(__file__, 52, 16), getitem___17659, str_17657)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 12), kws_17656, subscript_call_result_17660)
            # SSA join for if statement (line 49)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'kws' (line 53)
        kws_17661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'kws')
        # Testing if the type of an if condition is none (line 53)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 53, 8), kws_17661):
            pass
        else:
            
            # Testing the type of an if condition (line 53)
            if_condition_17662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 8), kws_17661)
            # Assigning a type to the variable 'if_condition_17662' (line 53)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'if_condition_17662', if_condition_17662)
            # SSA begins for if statement (line 53)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 54)
            # Processing the call arguments (line 54)
            str_17664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 28), 'str', 'Unexpected keyword arguments')
            # Processing the call keyword arguments (line 54)
            kwargs_17665 = {}
            # Getting the type of 'TypeError' (line 54)
            TypeError_17663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 18), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 54)
            TypeError_call_result_17666 = invoke(stypy.reporting.localization.Localization(__file__, 54, 18), TypeError_17663, *[str_17664], **kwargs_17665)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 54, 12), TypeError_call_result_17666, 'raise parameter', BaseException)
            # SSA join for if statement (line 53)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def parse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 56)
        False_17667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 36), 'False')
        defaults = [False_17667]
        # Create a new context for function 'parse'
        module_type_store = module_type_store.open_function_context('parse', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Parser.parse.__dict__.__setitem__('stypy_localization', localization)
        Parser.parse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Parser.parse.__dict__.__setitem__('stypy_type_store', module_type_store)
        Parser.parse.__dict__.__setitem__('stypy_function_name', 'Parser.parse')
        Parser.parse.__dict__.__setitem__('stypy_param_names_list', ['fp', 'headersonly'])
        Parser.parse.__dict__.__setitem__('stypy_varargs_param_name', None)
        Parser.parse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Parser.parse.__dict__.__setitem__('stypy_call_defaults', defaults)
        Parser.parse.__dict__.__setitem__('stypy_call_varargs', varargs)
        Parser.parse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Parser.parse.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Parser.parse', ['fp', 'headersonly'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'parse', localization, ['fp', 'headersonly'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'parse(...)' code ##################

        str_17668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, (-1)), 'str', 'Create a message structure from the data in a file.\n\n        Reads all the data from the file and returns the root of the message\n        structure.  Optional headersonly is a flag specifying whether to stop\n        parsing after reading the headers or not.  The default is False,\n        meaning it parses the entire contents of the file.\n        ')
        
        # Assigning a Call to a Name (line 64):
        
        # Call to FeedParser(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'self' (line 64)
        self_17670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 32), 'self', False)
        # Obtaining the member '_class' of a type (line 64)
        _class_17671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 32), self_17670, '_class')
        # Processing the call keyword arguments (line 64)
        kwargs_17672 = {}
        # Getting the type of 'FeedParser' (line 64)
        FeedParser_17669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'FeedParser', False)
        # Calling FeedParser(args, kwargs) (line 64)
        FeedParser_call_result_17673 = invoke(stypy.reporting.localization.Localization(__file__, 64, 21), FeedParser_17669, *[_class_17671], **kwargs_17672)
        
        # Assigning a type to the variable 'feedparser' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'feedparser', FeedParser_call_result_17673)
        # Getting the type of 'headersonly' (line 65)
        headersonly_17674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'headersonly')
        # Testing if the type of an if condition is none (line 65)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 65, 8), headersonly_17674):
            pass
        else:
            
            # Testing the type of an if condition (line 65)
            if_condition_17675 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 8), headersonly_17674)
            # Assigning a type to the variable 'if_condition_17675' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'if_condition_17675', if_condition_17675)
            # SSA begins for if statement (line 65)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _set_headersonly(...): (line 66)
            # Processing the call keyword arguments (line 66)
            kwargs_17678 = {}
            # Getting the type of 'feedparser' (line 66)
            feedparser_17676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'feedparser', False)
            # Obtaining the member '_set_headersonly' of a type (line 66)
            _set_headersonly_17677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), feedparser_17676, '_set_headersonly')
            # Calling _set_headersonly(args, kwargs) (line 66)
            _set_headersonly_call_result_17679 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), _set_headersonly_17677, *[], **kwargs_17678)
            
            # SSA join for if statement (line 65)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'True' (line 67)
        True_17680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 14), 'True')
        # Assigning a type to the variable 'True_17680' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'True_17680', True_17680)
        # Testing if the while is going to be iterated (line 67)
        # Testing the type of an if condition (line 67)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 8), True_17680)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 67, 8), True_17680):
            
            # Assigning a Call to a Name (line 68):
            
            # Call to read(...): (line 68)
            # Processing the call arguments (line 68)
            int_17683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 27), 'int')
            # Processing the call keyword arguments (line 68)
            kwargs_17684 = {}
            # Getting the type of 'fp' (line 68)
            fp_17681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'fp', False)
            # Obtaining the member 'read' of a type (line 68)
            read_17682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 19), fp_17681, 'read')
            # Calling read(args, kwargs) (line 68)
            read_call_result_17685 = invoke(stypy.reporting.localization.Localization(__file__, 68, 19), read_17682, *[int_17683], **kwargs_17684)
            
            # Assigning a type to the variable 'data' (line 68)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'data', read_call_result_17685)
            
            # Getting the type of 'data' (line 69)
            data_17686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'data')
            # Applying the 'not' unary operator (line 69)
            result_not__17687 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 15), 'not', data_17686)
            
            # Testing if the type of an if condition is none (line 69)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 69, 12), result_not__17687):
                pass
            else:
                
                # Testing the type of an if condition (line 69)
                if_condition_17688 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 12), result_not__17687)
                # Assigning a type to the variable 'if_condition_17688' (line 69)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'if_condition_17688', if_condition_17688)
                # SSA begins for if statement (line 69)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 69)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to feed(...): (line 71)
            # Processing the call arguments (line 71)
            # Getting the type of 'data' (line 71)
            data_17691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 28), 'data', False)
            # Processing the call keyword arguments (line 71)
            kwargs_17692 = {}
            # Getting the type of 'feedparser' (line 71)
            feedparser_17689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'feedparser', False)
            # Obtaining the member 'feed' of a type (line 71)
            feed_17690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), feedparser_17689, 'feed')
            # Calling feed(args, kwargs) (line 71)
            feed_call_result_17693 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), feed_17690, *[data_17691], **kwargs_17692)
            

        
        
        # Call to close(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_17696 = {}
        # Getting the type of 'feedparser' (line 72)
        feedparser_17694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'feedparser', False)
        # Obtaining the member 'close' of a type (line 72)
        close_17695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 15), feedparser_17694, 'close')
        # Calling close(args, kwargs) (line 72)
        close_call_result_17697 = invoke(stypy.reporting.localization.Localization(__file__, 72, 15), close_17695, *[], **kwargs_17696)
        
        # Assigning a type to the variable 'stypy_return_type' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'stypy_return_type', close_call_result_17697)
        
        # ################# End of 'parse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'parse' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_17698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17698)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'parse'
        return stypy_return_type_17698


    @norecursion
    def parsestr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 74)
        False_17699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 41), 'False')
        defaults = [False_17699]
        # Create a new context for function 'parsestr'
        module_type_store = module_type_store.open_function_context('parsestr', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Parser.parsestr.__dict__.__setitem__('stypy_localization', localization)
        Parser.parsestr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Parser.parsestr.__dict__.__setitem__('stypy_type_store', module_type_store)
        Parser.parsestr.__dict__.__setitem__('stypy_function_name', 'Parser.parsestr')
        Parser.parsestr.__dict__.__setitem__('stypy_param_names_list', ['text', 'headersonly'])
        Parser.parsestr.__dict__.__setitem__('stypy_varargs_param_name', None)
        Parser.parsestr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Parser.parsestr.__dict__.__setitem__('stypy_call_defaults', defaults)
        Parser.parsestr.__dict__.__setitem__('stypy_call_varargs', varargs)
        Parser.parsestr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Parser.parsestr.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Parser.parsestr', ['text', 'headersonly'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'parsestr', localization, ['text', 'headersonly'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'parsestr(...)' code ##################

        str_17700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, (-1)), 'str', 'Create a message structure from a string.\n\n        Returns the root of the message structure.  Optional headersonly is a\n        flag specifying whether to stop parsing after reading the headers or\n        not.  The default is False, meaning it parses the entire contents of\n        the file.\n        ')
        
        # Call to parse(...): (line 82)
        # Processing the call arguments (line 82)
        
        # Call to StringIO(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'text' (line 82)
        text_17704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 35), 'text', False)
        # Processing the call keyword arguments (line 82)
        kwargs_17705 = {}
        # Getting the type of 'StringIO' (line 82)
        StringIO_17703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 82)
        StringIO_call_result_17706 = invoke(stypy.reporting.localization.Localization(__file__, 82, 26), StringIO_17703, *[text_17704], **kwargs_17705)
        
        # Processing the call keyword arguments (line 82)
        # Getting the type of 'headersonly' (line 82)
        headersonly_17707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 54), 'headersonly', False)
        keyword_17708 = headersonly_17707
        kwargs_17709 = {'headersonly': keyword_17708}
        # Getting the type of 'self' (line 82)
        self_17701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'self', False)
        # Obtaining the member 'parse' of a type (line 82)
        parse_17702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), self_17701, 'parse')
        # Calling parse(args, kwargs) (line 82)
        parse_call_result_17710 = invoke(stypy.reporting.localization.Localization(__file__, 82, 15), parse_17702, *[StringIO_call_result_17706], **kwargs_17709)
        
        # Assigning a type to the variable 'stypy_return_type' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stypy_return_type', parse_call_result_17710)
        
        # ################# End of 'parsestr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'parsestr' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_17711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17711)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'parsestr'
        return stypy_return_type_17711


# Assigning a type to the variable 'Parser' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'Parser', Parser)
# Declaration of the 'HeaderParser' class
# Getting the type of 'Parser' (line 86)
Parser_17712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'Parser')

class HeaderParser(Parser_17712, ):

    @norecursion
    def parse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 87)
        True_17713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 36), 'True')
        defaults = [True_17713]
        # Create a new context for function 'parse'
        module_type_store = module_type_store.open_function_context('parse', 87, 4, False)
        # Assigning a type to the variable 'self' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HeaderParser.parse.__dict__.__setitem__('stypy_localization', localization)
        HeaderParser.parse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HeaderParser.parse.__dict__.__setitem__('stypy_type_store', module_type_store)
        HeaderParser.parse.__dict__.__setitem__('stypy_function_name', 'HeaderParser.parse')
        HeaderParser.parse.__dict__.__setitem__('stypy_param_names_list', ['fp', 'headersonly'])
        HeaderParser.parse.__dict__.__setitem__('stypy_varargs_param_name', None)
        HeaderParser.parse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HeaderParser.parse.__dict__.__setitem__('stypy_call_defaults', defaults)
        HeaderParser.parse.__dict__.__setitem__('stypy_call_varargs', varargs)
        HeaderParser.parse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HeaderParser.parse.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HeaderParser.parse', ['fp', 'headersonly'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'parse', localization, ['fp', 'headersonly'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'parse(...)' code ##################

        
        # Call to parse(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'self' (line 88)
        self_17716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 28), 'self', False)
        # Getting the type of 'fp' (line 88)
        fp_17717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 34), 'fp', False)
        # Getting the type of 'True' (line 88)
        True_17718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 38), 'True', False)
        # Processing the call keyword arguments (line 88)
        kwargs_17719 = {}
        # Getting the type of 'Parser' (line 88)
        Parser_17714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'Parser', False)
        # Obtaining the member 'parse' of a type (line 88)
        parse_17715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 15), Parser_17714, 'parse')
        # Calling parse(args, kwargs) (line 88)
        parse_call_result_17720 = invoke(stypy.reporting.localization.Localization(__file__, 88, 15), parse_17715, *[self_17716, fp_17717, True_17718], **kwargs_17719)
        
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'stypy_return_type', parse_call_result_17720)
        
        # ################# End of 'parse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'parse' in the type store
        # Getting the type of 'stypy_return_type' (line 87)
        stypy_return_type_17721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17721)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'parse'
        return stypy_return_type_17721


    @norecursion
    def parsestr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 90)
        True_17722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 41), 'True')
        defaults = [True_17722]
        # Create a new context for function 'parsestr'
        module_type_store = module_type_store.open_function_context('parsestr', 90, 4, False)
        # Assigning a type to the variable 'self' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HeaderParser.parsestr.__dict__.__setitem__('stypy_localization', localization)
        HeaderParser.parsestr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HeaderParser.parsestr.__dict__.__setitem__('stypy_type_store', module_type_store)
        HeaderParser.parsestr.__dict__.__setitem__('stypy_function_name', 'HeaderParser.parsestr')
        HeaderParser.parsestr.__dict__.__setitem__('stypy_param_names_list', ['text', 'headersonly'])
        HeaderParser.parsestr.__dict__.__setitem__('stypy_varargs_param_name', None)
        HeaderParser.parsestr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HeaderParser.parsestr.__dict__.__setitem__('stypy_call_defaults', defaults)
        HeaderParser.parsestr.__dict__.__setitem__('stypy_call_varargs', varargs)
        HeaderParser.parsestr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HeaderParser.parsestr.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HeaderParser.parsestr', ['text', 'headersonly'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'parsestr', localization, ['text', 'headersonly'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'parsestr(...)' code ##################

        
        # Call to parsestr(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'self' (line 91)
        self_17725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 31), 'self', False)
        # Getting the type of 'text' (line 91)
        text_17726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 37), 'text', False)
        # Getting the type of 'True' (line 91)
        True_17727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 43), 'True', False)
        # Processing the call keyword arguments (line 91)
        kwargs_17728 = {}
        # Getting the type of 'Parser' (line 91)
        Parser_17723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'Parser', False)
        # Obtaining the member 'parsestr' of a type (line 91)
        parsestr_17724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 15), Parser_17723, 'parsestr')
        # Calling parsestr(args, kwargs) (line 91)
        parsestr_call_result_17729 = invoke(stypy.reporting.localization.Localization(__file__, 91, 15), parsestr_17724, *[self_17725, text_17726, True_17727], **kwargs_17728)
        
        # Assigning a type to the variable 'stypy_return_type' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'stypy_return_type', parsestr_call_result_17729)
        
        # ################# End of 'parsestr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'parsestr' in the type store
        # Getting the type of 'stypy_return_type' (line 90)
        stypy_return_type_17730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17730)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'parsestr'
        return stypy_return_type_17730


# Assigning a type to the variable 'HeaderParser' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'HeaderParser', HeaderParser)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
