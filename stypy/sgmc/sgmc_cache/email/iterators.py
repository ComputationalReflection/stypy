
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2001-2006 Python Software Foundation
2: # Author: Barry Warsaw
3: # Contact: email-sig@python.org
4: 
5: '''Various types of useful iterators and generators.'''
6: 
7: __all__ = [
8:     'body_line_iterator',
9:     'typed_subpart_iterator',
10:     'walk',
11:     # Do not include _structure() since it's part of the debugging API.
12:     ]
13: 
14: import sys
15: from cStringIO import StringIO
16: 
17: 
18: 
19: # This function will become a method of the Message class
20: def walk(self):
21:     '''Walk over the message tree, yielding each subpart.
22: 
23:     The walk is performed in depth-first order.  This method is a
24:     generator.
25:     '''
26:     yield self
27:     if self.is_multipart():
28:         for subpart in self.get_payload():
29:             for subsubpart in subpart.walk():
30:                 yield subsubpart
31: 
32: 
33: 
34: # These two functions are imported into the Iterators.py interface module.
35: def body_line_iterator(msg, decode=False):
36:     '''Iterate over the parts, returning string payloads line-by-line.
37: 
38:     Optional decode (default False) is passed through to .get_payload().
39:     '''
40:     for subpart in msg.walk():
41:         payload = subpart.get_payload(decode=decode)
42:         if isinstance(payload, basestring):
43:             for line in StringIO(payload):
44:                 yield line
45: 
46: 
47: def typed_subpart_iterator(msg, maintype='text', subtype=None):
48:     '''Iterate over the subparts with a given MIME type.
49: 
50:     Use `maintype' as the main MIME type to match against; this defaults to
51:     "text".  Optional `subtype' is the MIME subtype to match against; if
52:     omitted, only the main type is matched.
53:     '''
54:     for subpart in msg.walk():
55:         if subpart.get_content_maintype() == maintype:
56:             if subtype is None or subpart.get_content_subtype() == subtype:
57:                 yield subpart
58: 
59: 
60: 
61: def _structure(msg, fp=None, level=0, include_default=False):
62:     '''A handy debugging aid'''
63:     if fp is None:
64:         fp = sys.stdout
65:     tab = ' ' * (level * 4)
66:     print >> fp, tab + msg.get_content_type(),
67:     if include_default:
68:         print >> fp, '[%s]' % msg.get_default_type()
69:     else:
70:         print >> fp
71:     if msg.is_multipart():
72:         for subpart in msg.get_payload():
73:             _structure(subpart, fp, level+1, include_default)
74: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_15913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'str', 'Various types of useful iterators and generators.')

# Assigning a List to a Name (line 7):
__all__ = ['body_line_iterator', 'typed_subpart_iterator', 'walk']
module_type_store.set_exportable_members(['body_line_iterator', 'typed_subpart_iterator', 'walk'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_15914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_15915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 4), 'str', 'body_line_iterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_15914, str_15915)
# Adding element type (line 7)
str_15916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 4), 'str', 'typed_subpart_iterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_15914, str_15916)
# Adding element type (line 7)
str_15917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 4), 'str', 'walk')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_15914, str_15917)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_15914)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import sys' statement (line 14)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from cStringIO import StringIO' statement (line 15)
try:
    from cStringIO import StringIO

except:
    StringIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'cStringIO', None, module_type_store, ['StringIO'], [StringIO])


@norecursion
def walk(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'walk'
    module_type_store = module_type_store.open_function_context('walk', 20, 0, False)
    
    # Passed parameters checking function
    walk.stypy_localization = localization
    walk.stypy_type_of_self = None
    walk.stypy_type_store = module_type_store
    walk.stypy_function_name = 'walk'
    walk.stypy_param_names_list = ['self']
    walk.stypy_varargs_param_name = None
    walk.stypy_kwargs_param_name = None
    walk.stypy_call_defaults = defaults
    walk.stypy_call_varargs = varargs
    walk.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'walk', ['self'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'walk', localization, ['self'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'walk(...)' code ##################

    str_15918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, (-1)), 'str', 'Walk over the message tree, yielding each subpart.\n\n    The walk is performed in depth-first order.  This method is a\n    generator.\n    ')
    # Creating a generator
    # Getting the type of 'self' (line 26)
    self_15919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 10), 'self')
    GeneratorType_15920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 4), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 4), GeneratorType_15920, self_15919)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type', GeneratorType_15920)
    
    # Call to is_multipart(...): (line 27)
    # Processing the call keyword arguments (line 27)
    kwargs_15923 = {}
    # Getting the type of 'self' (line 27)
    self_15921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 7), 'self', False)
    # Obtaining the member 'is_multipart' of a type (line 27)
    is_multipart_15922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 7), self_15921, 'is_multipart')
    # Calling is_multipart(args, kwargs) (line 27)
    is_multipart_call_result_15924 = invoke(stypy.reporting.localization.Localization(__file__, 27, 7), is_multipart_15922, *[], **kwargs_15923)
    
    # Testing if the type of an if condition is none (line 27)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 27, 4), is_multipart_call_result_15924):
        pass
    else:
        
        # Testing the type of an if condition (line 27)
        if_condition_15925 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 4), is_multipart_call_result_15924)
        # Assigning a type to the variable 'if_condition_15925' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'if_condition_15925', if_condition_15925)
        # SSA begins for if statement (line 27)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to get_payload(...): (line 28)
        # Processing the call keyword arguments (line 28)
        kwargs_15928 = {}
        # Getting the type of 'self' (line 28)
        self_15926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 23), 'self', False)
        # Obtaining the member 'get_payload' of a type (line 28)
        get_payload_15927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 23), self_15926, 'get_payload')
        # Calling get_payload(args, kwargs) (line 28)
        get_payload_call_result_15929 = invoke(stypy.reporting.localization.Localization(__file__, 28, 23), get_payload_15927, *[], **kwargs_15928)
        
        # Assigning a type to the variable 'get_payload_call_result_15929' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'get_payload_call_result_15929', get_payload_call_result_15929)
        # Testing if the for loop is going to be iterated (line 28)
        # Testing the type of a for loop iterable (line 28)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 28, 8), get_payload_call_result_15929)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 28, 8), get_payload_call_result_15929):
            # Getting the type of the for loop variable (line 28)
            for_loop_var_15930 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 28, 8), get_payload_call_result_15929)
            # Assigning a type to the variable 'subpart' (line 28)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'subpart', for_loop_var_15930)
            # SSA begins for a for statement (line 28)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to walk(...): (line 29)
            # Processing the call keyword arguments (line 29)
            kwargs_15933 = {}
            # Getting the type of 'subpart' (line 29)
            subpart_15931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 30), 'subpart', False)
            # Obtaining the member 'walk' of a type (line 29)
            walk_15932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 30), subpart_15931, 'walk')
            # Calling walk(args, kwargs) (line 29)
            walk_call_result_15934 = invoke(stypy.reporting.localization.Localization(__file__, 29, 30), walk_15932, *[], **kwargs_15933)
            
            # Assigning a type to the variable 'walk_call_result_15934' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'walk_call_result_15934', walk_call_result_15934)
            # Testing if the for loop is going to be iterated (line 29)
            # Testing the type of a for loop iterable (line 29)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 29, 12), walk_call_result_15934)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 29, 12), walk_call_result_15934):
                # Getting the type of the for loop variable (line 29)
                for_loop_var_15935 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 29, 12), walk_call_result_15934)
                # Assigning a type to the variable 'subsubpart' (line 29)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'subsubpart', for_loop_var_15935)
                # SSA begins for a for statement (line 29)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                # Creating a generator
                # Getting the type of 'subsubpart' (line 30)
                subsubpart_15936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'subsubpart')
                GeneratorType_15937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 16), 'GeneratorType')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 16), GeneratorType_15937, subsubpart_15936)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'stypy_return_type', GeneratorType_15937)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for if statement (line 27)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'walk(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'walk' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_15938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15938)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'walk'
    return stypy_return_type_15938

# Assigning a type to the variable 'walk' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'walk', walk)

@norecursion
def body_line_iterator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 35)
    False_15939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 35), 'False')
    defaults = [False_15939]
    # Create a new context for function 'body_line_iterator'
    module_type_store = module_type_store.open_function_context('body_line_iterator', 35, 0, False)
    
    # Passed parameters checking function
    body_line_iterator.stypy_localization = localization
    body_line_iterator.stypy_type_of_self = None
    body_line_iterator.stypy_type_store = module_type_store
    body_line_iterator.stypy_function_name = 'body_line_iterator'
    body_line_iterator.stypy_param_names_list = ['msg', 'decode']
    body_line_iterator.stypy_varargs_param_name = None
    body_line_iterator.stypy_kwargs_param_name = None
    body_line_iterator.stypy_call_defaults = defaults
    body_line_iterator.stypy_call_varargs = varargs
    body_line_iterator.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'body_line_iterator', ['msg', 'decode'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'body_line_iterator', localization, ['msg', 'decode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'body_line_iterator(...)' code ##################

    str_15940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, (-1)), 'str', 'Iterate over the parts, returning string payloads line-by-line.\n\n    Optional decode (default False) is passed through to .get_payload().\n    ')
    
    
    # Call to walk(...): (line 40)
    # Processing the call keyword arguments (line 40)
    kwargs_15943 = {}
    # Getting the type of 'msg' (line 40)
    msg_15941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'msg', False)
    # Obtaining the member 'walk' of a type (line 40)
    walk_15942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 19), msg_15941, 'walk')
    # Calling walk(args, kwargs) (line 40)
    walk_call_result_15944 = invoke(stypy.reporting.localization.Localization(__file__, 40, 19), walk_15942, *[], **kwargs_15943)
    
    # Assigning a type to the variable 'walk_call_result_15944' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'walk_call_result_15944', walk_call_result_15944)
    # Testing if the for loop is going to be iterated (line 40)
    # Testing the type of a for loop iterable (line 40)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 40, 4), walk_call_result_15944)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 40, 4), walk_call_result_15944):
        # Getting the type of the for loop variable (line 40)
        for_loop_var_15945 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 40, 4), walk_call_result_15944)
        # Assigning a type to the variable 'subpart' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'subpart', for_loop_var_15945)
        # SSA begins for a for statement (line 40)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 41):
        
        # Call to get_payload(...): (line 41)
        # Processing the call keyword arguments (line 41)
        # Getting the type of 'decode' (line 41)
        decode_15948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 45), 'decode', False)
        keyword_15949 = decode_15948
        kwargs_15950 = {'decode': keyword_15949}
        # Getting the type of 'subpart' (line 41)
        subpart_15946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 18), 'subpart', False)
        # Obtaining the member 'get_payload' of a type (line 41)
        get_payload_15947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 18), subpart_15946, 'get_payload')
        # Calling get_payload(args, kwargs) (line 41)
        get_payload_call_result_15951 = invoke(stypy.reporting.localization.Localization(__file__, 41, 18), get_payload_15947, *[], **kwargs_15950)
        
        # Assigning a type to the variable 'payload' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'payload', get_payload_call_result_15951)
        
        # Type idiom detected: calculating its left and rigth part (line 42)
        # Getting the type of 'basestring' (line 42)
        basestring_15952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 31), 'basestring')
        # Getting the type of 'payload' (line 42)
        payload_15953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 22), 'payload')
        
        (may_be_15954, more_types_in_union_15955) = may_be_subtype(basestring_15952, payload_15953)

        if may_be_15954:

            if more_types_in_union_15955:
                # Runtime conditional SSA (line 42)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'payload' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'payload', remove_not_subtype_from_union(payload_15953, basestring))
            
            
            # Call to StringIO(...): (line 43)
            # Processing the call arguments (line 43)
            # Getting the type of 'payload' (line 43)
            payload_15957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 33), 'payload', False)
            # Processing the call keyword arguments (line 43)
            kwargs_15958 = {}
            # Getting the type of 'StringIO' (line 43)
            StringIO_15956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'StringIO', False)
            # Calling StringIO(args, kwargs) (line 43)
            StringIO_call_result_15959 = invoke(stypy.reporting.localization.Localization(__file__, 43, 24), StringIO_15956, *[payload_15957], **kwargs_15958)
            
            # Assigning a type to the variable 'StringIO_call_result_15959' (line 43)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'StringIO_call_result_15959', StringIO_call_result_15959)
            # Testing if the for loop is going to be iterated (line 43)
            # Testing the type of a for loop iterable (line 43)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 43, 12), StringIO_call_result_15959)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 43, 12), StringIO_call_result_15959):
                # Getting the type of the for loop variable (line 43)
                for_loop_var_15960 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 43, 12), StringIO_call_result_15959)
                # Assigning a type to the variable 'line' (line 43)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'line', for_loop_var_15960)
                # SSA begins for a for statement (line 43)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                # Creating a generator
                # Getting the type of 'line' (line 44)
                line_15961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'line')
                GeneratorType_15962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 16), 'GeneratorType')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 16), GeneratorType_15962, line_15961)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'stypy_return_type', GeneratorType_15962)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            

            if more_types_in_union_15955:
                # SSA join for if statement (line 42)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'body_line_iterator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'body_line_iterator' in the type store
    # Getting the type of 'stypy_return_type' (line 35)
    stypy_return_type_15963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15963)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'body_line_iterator'
    return stypy_return_type_15963

# Assigning a type to the variable 'body_line_iterator' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'body_line_iterator', body_line_iterator)

@norecursion
def typed_subpart_iterator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_15964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 41), 'str', 'text')
    # Getting the type of 'None' (line 47)
    None_15965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 57), 'None')
    defaults = [str_15964, None_15965]
    # Create a new context for function 'typed_subpart_iterator'
    module_type_store = module_type_store.open_function_context('typed_subpart_iterator', 47, 0, False)
    
    # Passed parameters checking function
    typed_subpart_iterator.stypy_localization = localization
    typed_subpart_iterator.stypy_type_of_self = None
    typed_subpart_iterator.stypy_type_store = module_type_store
    typed_subpart_iterator.stypy_function_name = 'typed_subpart_iterator'
    typed_subpart_iterator.stypy_param_names_list = ['msg', 'maintype', 'subtype']
    typed_subpart_iterator.stypy_varargs_param_name = None
    typed_subpart_iterator.stypy_kwargs_param_name = None
    typed_subpart_iterator.stypy_call_defaults = defaults
    typed_subpart_iterator.stypy_call_varargs = varargs
    typed_subpart_iterator.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'typed_subpart_iterator', ['msg', 'maintype', 'subtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'typed_subpart_iterator', localization, ['msg', 'maintype', 'subtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'typed_subpart_iterator(...)' code ##################

    str_15966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, (-1)), 'str', 'Iterate over the subparts with a given MIME type.\n\n    Use `maintype\' as the main MIME type to match against; this defaults to\n    "text".  Optional `subtype\' is the MIME subtype to match against; if\n    omitted, only the main type is matched.\n    ')
    
    
    # Call to walk(...): (line 54)
    # Processing the call keyword arguments (line 54)
    kwargs_15969 = {}
    # Getting the type of 'msg' (line 54)
    msg_15967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 19), 'msg', False)
    # Obtaining the member 'walk' of a type (line 54)
    walk_15968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 19), msg_15967, 'walk')
    # Calling walk(args, kwargs) (line 54)
    walk_call_result_15970 = invoke(stypy.reporting.localization.Localization(__file__, 54, 19), walk_15968, *[], **kwargs_15969)
    
    # Assigning a type to the variable 'walk_call_result_15970' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'walk_call_result_15970', walk_call_result_15970)
    # Testing if the for loop is going to be iterated (line 54)
    # Testing the type of a for loop iterable (line 54)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 54, 4), walk_call_result_15970)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 54, 4), walk_call_result_15970):
        # Getting the type of the for loop variable (line 54)
        for_loop_var_15971 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 54, 4), walk_call_result_15970)
        # Assigning a type to the variable 'subpart' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'subpart', for_loop_var_15971)
        # SSA begins for a for statement (line 54)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to get_content_maintype(...): (line 55)
        # Processing the call keyword arguments (line 55)
        kwargs_15974 = {}
        # Getting the type of 'subpart' (line 55)
        subpart_15972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'subpart', False)
        # Obtaining the member 'get_content_maintype' of a type (line 55)
        get_content_maintype_15973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 11), subpart_15972, 'get_content_maintype')
        # Calling get_content_maintype(args, kwargs) (line 55)
        get_content_maintype_call_result_15975 = invoke(stypy.reporting.localization.Localization(__file__, 55, 11), get_content_maintype_15973, *[], **kwargs_15974)
        
        # Getting the type of 'maintype' (line 55)
        maintype_15976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 45), 'maintype')
        # Applying the binary operator '==' (line 55)
        result_eq_15977 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 11), '==', get_content_maintype_call_result_15975, maintype_15976)
        
        # Testing if the type of an if condition is none (line 55)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 55, 8), result_eq_15977):
            pass
        else:
            
            # Testing the type of an if condition (line 55)
            if_condition_15978 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 8), result_eq_15977)
            # Assigning a type to the variable 'if_condition_15978' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'if_condition_15978', if_condition_15978)
            # SSA begins for if statement (line 55)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Evaluating a boolean operation
            
            # Getting the type of 'subtype' (line 56)
            subtype_15979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 'subtype')
            # Getting the type of 'None' (line 56)
            None_15980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 26), 'None')
            # Applying the binary operator 'is' (line 56)
            result_is__15981 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 15), 'is', subtype_15979, None_15980)
            
            
            
            # Call to get_content_subtype(...): (line 56)
            # Processing the call keyword arguments (line 56)
            kwargs_15984 = {}
            # Getting the type of 'subpart' (line 56)
            subpart_15982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 34), 'subpart', False)
            # Obtaining the member 'get_content_subtype' of a type (line 56)
            get_content_subtype_15983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 34), subpart_15982, 'get_content_subtype')
            # Calling get_content_subtype(args, kwargs) (line 56)
            get_content_subtype_call_result_15985 = invoke(stypy.reporting.localization.Localization(__file__, 56, 34), get_content_subtype_15983, *[], **kwargs_15984)
            
            # Getting the type of 'subtype' (line 56)
            subtype_15986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 67), 'subtype')
            # Applying the binary operator '==' (line 56)
            result_eq_15987 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 34), '==', get_content_subtype_call_result_15985, subtype_15986)
            
            # Applying the binary operator 'or' (line 56)
            result_or_keyword_15988 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 15), 'or', result_is__15981, result_eq_15987)
            
            # Testing if the type of an if condition is none (line 56)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 56, 12), result_or_keyword_15988):
                pass
            else:
                
                # Testing the type of an if condition (line 56)
                if_condition_15989 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 12), result_or_keyword_15988)
                # Assigning a type to the variable 'if_condition_15989' (line 56)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'if_condition_15989', if_condition_15989)
                # SSA begins for if statement (line 56)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Creating a generator
                # Getting the type of 'subpart' (line 57)
                subpart_15990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 22), 'subpart')
                GeneratorType_15991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 16), 'GeneratorType')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 16), GeneratorType_15991, subpart_15990)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'stypy_return_type', GeneratorType_15991)
                # SSA join for if statement (line 56)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 55)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'typed_subpart_iterator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'typed_subpart_iterator' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_15992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15992)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'typed_subpart_iterator'
    return stypy_return_type_15992

# Assigning a type to the variable 'typed_subpart_iterator' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'typed_subpart_iterator', typed_subpart_iterator)

@norecursion
def _structure(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 61)
    None_15993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 23), 'None')
    int_15994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 35), 'int')
    # Getting the type of 'False' (line 61)
    False_15995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 54), 'False')
    defaults = [None_15993, int_15994, False_15995]
    # Create a new context for function '_structure'
    module_type_store = module_type_store.open_function_context('_structure', 61, 0, False)
    
    # Passed parameters checking function
    _structure.stypy_localization = localization
    _structure.stypy_type_of_self = None
    _structure.stypy_type_store = module_type_store
    _structure.stypy_function_name = '_structure'
    _structure.stypy_param_names_list = ['msg', 'fp', 'level', 'include_default']
    _structure.stypy_varargs_param_name = None
    _structure.stypy_kwargs_param_name = None
    _structure.stypy_call_defaults = defaults
    _structure.stypy_call_varargs = varargs
    _structure.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_structure', ['msg', 'fp', 'level', 'include_default'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_structure', localization, ['msg', 'fp', 'level', 'include_default'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_structure(...)' code ##################

    str_15996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 4), 'str', 'A handy debugging aid')
    
    # Type idiom detected: calculating its left and rigth part (line 63)
    # Getting the type of 'fp' (line 63)
    fp_15997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 7), 'fp')
    # Getting the type of 'None' (line 63)
    None_15998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 13), 'None')
    
    (may_be_15999, more_types_in_union_16000) = may_be_none(fp_15997, None_15998)

    if may_be_15999:

        if more_types_in_union_16000:
            # Runtime conditional SSA (line 63)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 64):
        # Getting the type of 'sys' (line 64)
        sys_16001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 13), 'sys')
        # Obtaining the member 'stdout' of a type (line 64)
        stdout_16002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 13), sys_16001, 'stdout')
        # Assigning a type to the variable 'fp' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'fp', stdout_16002)

        if more_types_in_union_16000:
            # SSA join for if statement (line 63)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 65):
    str_16003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 10), 'str', ' ')
    # Getting the type of 'level' (line 65)
    level_16004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 17), 'level')
    int_16005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 25), 'int')
    # Applying the binary operator '*' (line 65)
    result_mul_16006 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 17), '*', level_16004, int_16005)
    
    # Applying the binary operator '*' (line 65)
    result_mul_16007 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 10), '*', str_16003, result_mul_16006)
    
    # Assigning a type to the variable 'tab' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tab', result_mul_16007)
    # Getting the type of 'tab' (line 66)
    tab_16008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'tab')
    
    # Call to get_content_type(...): (line 66)
    # Processing the call keyword arguments (line 66)
    kwargs_16011 = {}
    # Getting the type of 'msg' (line 66)
    msg_16009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 23), 'msg', False)
    # Obtaining the member 'get_content_type' of a type (line 66)
    get_content_type_16010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 23), msg_16009, 'get_content_type')
    # Calling get_content_type(args, kwargs) (line 66)
    get_content_type_call_result_16012 = invoke(stypy.reporting.localization.Localization(__file__, 66, 23), get_content_type_16010, *[], **kwargs_16011)
    
    # Applying the binary operator '+' (line 66)
    result_add_16013 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 17), '+', tab_16008, get_content_type_call_result_16012)
    
    # Getting the type of 'include_default' (line 67)
    include_default_16014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 7), 'include_default')
    # Testing if the type of an if condition is none (line 67)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 67, 4), include_default_16014):
        pass
    else:
        
        # Testing the type of an if condition (line 67)
        if_condition_16015 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 4), include_default_16014)
        # Assigning a type to the variable 'if_condition_16015' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'if_condition_16015', if_condition_16015)
        # SSA begins for if statement (line 67)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_16016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 21), 'str', '[%s]')
        
        # Call to get_default_type(...): (line 68)
        # Processing the call keyword arguments (line 68)
        kwargs_16019 = {}
        # Getting the type of 'msg' (line 68)
        msg_16017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 30), 'msg', False)
        # Obtaining the member 'get_default_type' of a type (line 68)
        get_default_type_16018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 30), msg_16017, 'get_default_type')
        # Calling get_default_type(args, kwargs) (line 68)
        get_default_type_call_result_16020 = invoke(stypy.reporting.localization.Localization(__file__, 68, 30), get_default_type_16018, *[], **kwargs_16019)
        
        # Applying the binary operator '%' (line 68)
        result_mod_16021 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 21), '%', str_16016, get_default_type_call_result_16020)
        
        # SSA branch for the else part of an if statement (line 67)
        module_type_store.open_ssa_branch('else')
        # SSA join for if statement (line 67)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to is_multipart(...): (line 71)
    # Processing the call keyword arguments (line 71)
    kwargs_16024 = {}
    # Getting the type of 'msg' (line 71)
    msg_16022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 7), 'msg', False)
    # Obtaining the member 'is_multipart' of a type (line 71)
    is_multipart_16023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 7), msg_16022, 'is_multipart')
    # Calling is_multipart(args, kwargs) (line 71)
    is_multipart_call_result_16025 = invoke(stypy.reporting.localization.Localization(__file__, 71, 7), is_multipart_16023, *[], **kwargs_16024)
    
    # Testing if the type of an if condition is none (line 71)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 71, 4), is_multipart_call_result_16025):
        pass
    else:
        
        # Testing the type of an if condition (line 71)
        if_condition_16026 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 4), is_multipart_call_result_16025)
        # Assigning a type to the variable 'if_condition_16026' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'if_condition_16026', if_condition_16026)
        # SSA begins for if statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to get_payload(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_16029 = {}
        # Getting the type of 'msg' (line 72)
        msg_16027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 'msg', False)
        # Obtaining the member 'get_payload' of a type (line 72)
        get_payload_16028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 23), msg_16027, 'get_payload')
        # Calling get_payload(args, kwargs) (line 72)
        get_payload_call_result_16030 = invoke(stypy.reporting.localization.Localization(__file__, 72, 23), get_payload_16028, *[], **kwargs_16029)
        
        # Assigning a type to the variable 'get_payload_call_result_16030' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'get_payload_call_result_16030', get_payload_call_result_16030)
        # Testing if the for loop is going to be iterated (line 72)
        # Testing the type of a for loop iterable (line 72)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 72, 8), get_payload_call_result_16030)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 72, 8), get_payload_call_result_16030):
            # Getting the type of the for loop variable (line 72)
            for_loop_var_16031 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 72, 8), get_payload_call_result_16030)
            # Assigning a type to the variable 'subpart' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'subpart', for_loop_var_16031)
            # SSA begins for a for statement (line 72)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to _structure(...): (line 73)
            # Processing the call arguments (line 73)
            # Getting the type of 'subpart' (line 73)
            subpart_16033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'subpart', False)
            # Getting the type of 'fp' (line 73)
            fp_16034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 32), 'fp', False)
            # Getting the type of 'level' (line 73)
            level_16035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 36), 'level', False)
            int_16036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 42), 'int')
            # Applying the binary operator '+' (line 73)
            result_add_16037 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 36), '+', level_16035, int_16036)
            
            # Getting the type of 'include_default' (line 73)
            include_default_16038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 45), 'include_default', False)
            # Processing the call keyword arguments (line 73)
            kwargs_16039 = {}
            # Getting the type of '_structure' (line 73)
            _structure_16032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), '_structure', False)
            # Calling _structure(args, kwargs) (line 73)
            _structure_call_result_16040 = invoke(stypy.reporting.localization.Localization(__file__, 73, 12), _structure_16032, *[subpart_16033, fp_16034, result_add_16037, include_default_16038], **kwargs_16039)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for if statement (line 71)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of '_structure(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_structure' in the type store
    # Getting the type of 'stypy_return_type' (line 61)
    stypy_return_type_16041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16041)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_structure'
    return stypy_return_type_16041

# Assigning a type to the variable '_structure' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), '_structure', _structure)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
