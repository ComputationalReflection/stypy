
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: from matplotlib import cbook
7: import sys
8: import types
9: 
10: 
11: class Substitution(object):
12:     '''
13:     A decorator to take a function's docstring and perform string
14:     substitution on it.
15: 
16:     This decorator should be robust even if func.__doc__ is None
17:     (for example, if -OO was passed to the interpreter)
18: 
19:     Usage: construct a docstring.Substitution with a sequence or
20:     dictionary suitable for performing substitution; then
21:     decorate a suitable function with the constructed object. e.g.
22: 
23:     sub_author_name = Substitution(author='Jason')
24: 
25:     @sub_author_name
26:     def some_function(x):
27:         "%(author)s wrote this function"
28: 
29:     # note that some_function.__doc__ is now "Jason wrote this function"
30: 
31:     One can also use positional arguments.
32: 
33:     sub_first_last_names = Substitution('Edgar Allen', 'Poe')
34: 
35:     @sub_first_last_names
36:     def some_function(x):
37:         "%s %s wrote the Raven"
38:     '''
39:     def __init__(self, *args, **kwargs):
40:         assert not (len(args) and len(kwargs)), \
41:                 "Only positional or keyword args are allowed"
42:         self.params = args or kwargs
43: 
44:     def __call__(self, func):
45:         func.__doc__ = func.__doc__ and func.__doc__ % self.params
46:         return func
47: 
48:     def update(self, *args, **kwargs):
49:         "Assume self.params is a dict and update it with supplied args"
50:         self.params.update(*args, **kwargs)
51: 
52:     @classmethod
53:     def from_params(cls, params):
54:         '''
55:         In the case where the params is a mutable sequence (list or
56:         dictionary) and it may change before this class is called, one may
57:         explicitly use a reference to the params rather than using *args or
58:         **kwargs which will copy the values and not reference them.
59:         '''
60:         result = cls()
61:         result.params = params
62:         return result
63: 
64: 
65: class Appender(object):
66:     '''
67:     A function decorator that will append an addendum to the docstring
68:     of the target function.
69: 
70:     This decorator should be robust even if func.__doc__ is None
71:     (for example, if -OO was passed to the interpreter).
72: 
73:     Usage: construct a docstring.Appender with a string to be joined to
74:     the original docstring. An optional 'join' parameter may be supplied
75:     which will be used to join the docstring and addendum. e.g.
76: 
77:     add_copyright = Appender("Copyright (c) 2009", join='\n')
78: 
79:     @add_copyright
80:     def my_dog(has='fleas'):
81:         "This docstring will have a copyright below"
82:         pass
83:     '''
84:     def __init__(self, addendum, join=''):
85:         self.addendum = addendum
86:         self.join = join
87: 
88:     def __call__(self, func):
89:         docitems = [func.__doc__, self.addendum]
90:         func.__doc__ = func.__doc__ and self.join.join(docitems)
91:         return func
92: 
93: 
94: def dedent(func):
95:     "Dedent a docstring (if present)"
96:     func.__doc__ = func.__doc__ and cbook.dedent(func.__doc__)
97:     return func
98: 
99: 
100: def copy(source):
101:     "Copy a docstring from another source function (if present)"
102:     def do_copy(target):
103:         if source.__doc__:
104:             target.__doc__ = source.__doc__
105:         return target
106:     return do_copy
107: 
108: # create a decorator that will house the various documentation that
109: #  is reused throughout matplotlib
110: interpd = Substitution()
111: 
112: 
113: def dedent_interpd(func):
114:     '''A special case of the interpd that first performs a dedent on
115:     the incoming docstring'''
116:     if isinstance(func, types.MethodType) and not six.PY3:
117:         func = func.im_func
118:     return interpd(dedent(func))
119: 
120: 
121: def copy_dedent(source):
122:     '''A decorator that will copy the docstring from the source and
123:     then dedent it'''
124:     # note the following is ugly because "Python is not a functional
125:     # language" - GVR. Perhaps one day, functools.compose will exist.
126:     #  or perhaps not.
127:     #  http://mail.python.org/pipermail/patches/2007-February/021687.html
128:     return lambda target: dedent(copy(source)(target))
129: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_47623 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_47623) is not StypyTypeError):

    if (import_47623 != 'pyd_module'):
        __import__(import_47623)
        sys_modules_47624 = sys.modules[import_47623]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_47624.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_47623)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from matplotlib import cbook' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_47625 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib')

if (type(import_47625) is not StypyTypeError):

    if (import_47625 != 'pyd_module'):
        __import__(import_47625)
        sys_modules_47626 = sys.modules[import_47625]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib', sys_modules_47626.module_type_store, module_type_store, ['cbook'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_47626, sys_modules_47626.module_type_store, module_type_store)
    else:
        from matplotlib import cbook

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib', None, module_type_store, ['cbook'], [cbook])

else:
    # Assigning a type to the variable 'matplotlib' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib', import_47625)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import sys' statement (line 7)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import types' statement (line 8)
import types

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'types', types, module_type_store)

# Declaration of the 'Substitution' class

class Substitution(object, ):
    unicode_47627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, (-1)), 'unicode', u'\n    A decorator to take a function\'s docstring and perform string\n    substitution on it.\n\n    This decorator should be robust even if func.__doc__ is None\n    (for example, if -OO was passed to the interpreter)\n\n    Usage: construct a docstring.Substitution with a sequence or\n    dictionary suitable for performing substitution; then\n    decorate a suitable function with the constructed object. e.g.\n\n    sub_author_name = Substitution(author=\'Jason\')\n\n    @sub_author_name\n    def some_function(x):\n        "%(author)s wrote this function"\n\n    # note that some_function.__doc__ is now "Jason wrote this function"\n\n    One can also use positional arguments.\n\n    sub_first_last_names = Substitution(\'Edgar Allen\', \'Poe\')\n\n    @sub_first_last_names\n    def some_function(x):\n        "%s %s wrote the Raven"\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Substitution.__init__', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        # Evaluating assert statement condition
        
        
        # Evaluating a boolean operation
        
        # Call to len(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'args' (line 40)
        args_47629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'args', False)
        # Processing the call keyword arguments (line 40)
        kwargs_47630 = {}
        # Getting the type of 'len' (line 40)
        len_47628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'len', False)
        # Calling len(args, kwargs) (line 40)
        len_call_result_47631 = invoke(stypy.reporting.localization.Localization(__file__, 40, 20), len_47628, *[args_47629], **kwargs_47630)
        
        
        # Call to len(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'kwargs' (line 40)
        kwargs_47633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 38), 'kwargs', False)
        # Processing the call keyword arguments (line 40)
        kwargs_47634 = {}
        # Getting the type of 'len' (line 40)
        len_47632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 34), 'len', False)
        # Calling len(args, kwargs) (line 40)
        len_call_result_47635 = invoke(stypy.reporting.localization.Localization(__file__, 40, 34), len_47632, *[kwargs_47633], **kwargs_47634)
        
        # Applying the binary operator 'and' (line 40)
        result_and_keyword_47636 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 20), 'and', len_call_result_47631, len_call_result_47635)
        
        # Applying the 'not' unary operator (line 40)
        result_not__47637 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 15), 'not', result_and_keyword_47636)
        
        
        # Assigning a BoolOp to a Attribute (line 42):
        
        # Evaluating a boolean operation
        # Getting the type of 'args' (line 42)
        args_47638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 22), 'args')
        # Getting the type of 'kwargs' (line 42)
        kwargs_47639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 30), 'kwargs')
        # Applying the binary operator 'or' (line 42)
        result_or_keyword_47640 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 22), 'or', args_47638, kwargs_47639)
        
        # Getting the type of 'self' (line 42)
        self_47641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self')
        # Setting the type of the member 'params' of a type (line 42)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_47641, 'params', result_or_keyword_47640)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Substitution.__call__.__dict__.__setitem__('stypy_localization', localization)
        Substitution.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Substitution.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Substitution.__call__.__dict__.__setitem__('stypy_function_name', 'Substitution.__call__')
        Substitution.__call__.__dict__.__setitem__('stypy_param_names_list', ['func'])
        Substitution.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Substitution.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Substitution.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Substitution.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Substitution.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Substitution.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Substitution.__call__', ['func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Assigning a BoolOp to a Attribute (line 45):
        
        # Evaluating a boolean operation
        # Getting the type of 'func' (line 45)
        func_47642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 23), 'func')
        # Obtaining the member '__doc__' of a type (line 45)
        doc___47643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 23), func_47642, '__doc__')
        # Getting the type of 'func' (line 45)
        func_47644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 40), 'func')
        # Obtaining the member '__doc__' of a type (line 45)
        doc___47645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 40), func_47644, '__doc__')
        # Getting the type of 'self' (line 45)
        self_47646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 55), 'self')
        # Obtaining the member 'params' of a type (line 45)
        params_47647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 55), self_47646, 'params')
        # Applying the binary operator '%' (line 45)
        result_mod_47648 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 40), '%', doc___47645, params_47647)
        
        # Applying the binary operator 'and' (line 45)
        result_and_keyword_47649 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 23), 'and', doc___47643, result_mod_47648)
        
        # Getting the type of 'func' (line 45)
        func_47650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'func')
        # Setting the type of the member '__doc__' of a type (line 45)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), func_47650, '__doc__', result_and_keyword_47649)
        # Getting the type of 'func' (line 46)
        func_47651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 15), 'func')
        # Assigning a type to the variable 'stypy_return_type' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'stypy_return_type', func_47651)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_47652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_47652)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_47652


    @norecursion
    def update(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update'
        module_type_store = module_type_store.open_function_context('update', 48, 4, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Substitution.update.__dict__.__setitem__('stypy_localization', localization)
        Substitution.update.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Substitution.update.__dict__.__setitem__('stypy_type_store', module_type_store)
        Substitution.update.__dict__.__setitem__('stypy_function_name', 'Substitution.update')
        Substitution.update.__dict__.__setitem__('stypy_param_names_list', [])
        Substitution.update.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        Substitution.update.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        Substitution.update.__dict__.__setitem__('stypy_call_defaults', defaults)
        Substitution.update.__dict__.__setitem__('stypy_call_varargs', varargs)
        Substitution.update.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Substitution.update.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Substitution.update', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update(...)' code ##################

        unicode_47653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 8), 'unicode', u'Assume self.params is a dict and update it with supplied args')
        
        # Call to update(...): (line 50)
        # Getting the type of 'args' (line 50)
        args_47657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 28), 'args', False)
        # Processing the call keyword arguments (line 50)
        # Getting the type of 'kwargs' (line 50)
        kwargs_47658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 36), 'kwargs', False)
        kwargs_47659 = {'kwargs_47658': kwargs_47658}
        # Getting the type of 'self' (line 50)
        self_47654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'self', False)
        # Obtaining the member 'params' of a type (line 50)
        params_47655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), self_47654, 'params')
        # Obtaining the member 'update' of a type (line 50)
        update_47656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), params_47655, 'update')
        # Calling update(args, kwargs) (line 50)
        update_call_result_47660 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), update_47656, *[args_47657], **kwargs_47659)
        
        
        # ################# End of 'update(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update' in the type store
        # Getting the type of 'stypy_return_type' (line 48)
        stypy_return_type_47661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_47661)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update'
        return stypy_return_type_47661


    @norecursion
    def from_params(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'from_params'
        module_type_store = module_type_store.open_function_context('from_params', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Substitution.from_params.__dict__.__setitem__('stypy_localization', localization)
        Substitution.from_params.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Substitution.from_params.__dict__.__setitem__('stypy_type_store', module_type_store)
        Substitution.from_params.__dict__.__setitem__('stypy_function_name', 'Substitution.from_params')
        Substitution.from_params.__dict__.__setitem__('stypy_param_names_list', ['params'])
        Substitution.from_params.__dict__.__setitem__('stypy_varargs_param_name', None)
        Substitution.from_params.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Substitution.from_params.__dict__.__setitem__('stypy_call_defaults', defaults)
        Substitution.from_params.__dict__.__setitem__('stypy_call_varargs', varargs)
        Substitution.from_params.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Substitution.from_params.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Substitution.from_params', ['params'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'from_params', localization, ['params'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'from_params(...)' code ##################

        unicode_47662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'unicode', u'\n        In the case where the params is a mutable sequence (list or\n        dictionary) and it may change before this class is called, one may\n        explicitly use a reference to the params rather than using *args or\n        **kwargs which will copy the values and not reference them.\n        ')
        
        # Assigning a Call to a Name (line 60):
        
        # Call to cls(...): (line 60)
        # Processing the call keyword arguments (line 60)
        kwargs_47664 = {}
        # Getting the type of 'cls' (line 60)
        cls_47663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 17), 'cls', False)
        # Calling cls(args, kwargs) (line 60)
        cls_call_result_47665 = invoke(stypy.reporting.localization.Localization(__file__, 60, 17), cls_47663, *[], **kwargs_47664)
        
        # Assigning a type to the variable 'result' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'result', cls_call_result_47665)
        
        # Assigning a Name to a Attribute (line 61):
        # Getting the type of 'params' (line 61)
        params_47666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'params')
        # Getting the type of 'result' (line 61)
        result_47667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'result')
        # Setting the type of the member 'params' of a type (line 61)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), result_47667, 'params', params_47666)
        # Getting the type of 'result' (line 62)
        result_47668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'stypy_return_type', result_47668)
        
        # ################# End of 'from_params(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'from_params' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_47669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_47669)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'from_params'
        return stypy_return_type_47669


# Assigning a type to the variable 'Substitution' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'Substitution', Substitution)
# Declaration of the 'Appender' class

class Appender(object, ):
    unicode_47670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, (-1)), 'unicode', u'\n    A function decorator that will append an addendum to the docstring\n    of the target function.\n\n    This decorator should be robust even if func.__doc__ is None\n    (for example, if -OO was passed to the interpreter).\n\n    Usage: construct a docstring.Appender with a string to be joined to\n    the original docstring. An optional \'join\' parameter may be supplied\n    which will be used to join the docstring and addendum. e.g.\n\n    add_copyright = Appender("Copyright (c) 2009", join=\'\n\')\n\n    @add_copyright\n    def my_dog(has=\'fleas\'):\n        "This docstring will have a copyright below"\n        pass\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        unicode_47671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 38), 'unicode', u'')
        defaults = [unicode_47671]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 84, 4, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Appender.__init__', ['addendum', 'join'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['addendum', 'join'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 85):
        # Getting the type of 'addendum' (line 85)
        addendum_47672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 24), 'addendum')
        # Getting the type of 'self' (line 85)
        self_47673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'self')
        # Setting the type of the member 'addendum' of a type (line 85)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), self_47673, 'addendum', addendum_47672)
        
        # Assigning a Name to a Attribute (line 86):
        # Getting the type of 'join' (line 86)
        join_47674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'join')
        # Getting the type of 'self' (line 86)
        self_47675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'self')
        # Setting the type of the member 'join' of a type (line 86)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), self_47675, 'join', join_47674)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Appender.__call__.__dict__.__setitem__('stypy_localization', localization)
        Appender.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Appender.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Appender.__call__.__dict__.__setitem__('stypy_function_name', 'Appender.__call__')
        Appender.__call__.__dict__.__setitem__('stypy_param_names_list', ['func'])
        Appender.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Appender.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Appender.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Appender.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Appender.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Appender.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Appender.__call__', ['func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Assigning a List to a Name (line 89):
        
        # Obtaining an instance of the builtin type 'list' (line 89)
        list_47676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 89)
        # Adding element type (line 89)
        # Getting the type of 'func' (line 89)
        func_47677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'func')
        # Obtaining the member '__doc__' of a type (line 89)
        doc___47678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 20), func_47677, '__doc__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 19), list_47676, doc___47678)
        # Adding element type (line 89)
        # Getting the type of 'self' (line 89)
        self_47679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 34), 'self')
        # Obtaining the member 'addendum' of a type (line 89)
        addendum_47680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 34), self_47679, 'addendum')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 19), list_47676, addendum_47680)
        
        # Assigning a type to the variable 'docitems' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'docitems', list_47676)
        
        # Assigning a BoolOp to a Attribute (line 90):
        
        # Evaluating a boolean operation
        # Getting the type of 'func' (line 90)
        func_47681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 23), 'func')
        # Obtaining the member '__doc__' of a type (line 90)
        doc___47682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 23), func_47681, '__doc__')
        
        # Call to join(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'docitems' (line 90)
        docitems_47686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 55), 'docitems', False)
        # Processing the call keyword arguments (line 90)
        kwargs_47687 = {}
        # Getting the type of 'self' (line 90)
        self_47683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 40), 'self', False)
        # Obtaining the member 'join' of a type (line 90)
        join_47684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 40), self_47683, 'join')
        # Obtaining the member 'join' of a type (line 90)
        join_47685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 40), join_47684, 'join')
        # Calling join(args, kwargs) (line 90)
        join_call_result_47688 = invoke(stypy.reporting.localization.Localization(__file__, 90, 40), join_47685, *[docitems_47686], **kwargs_47687)
        
        # Applying the binary operator 'and' (line 90)
        result_and_keyword_47689 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 23), 'and', doc___47682, join_call_result_47688)
        
        # Getting the type of 'func' (line 90)
        func_47690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'func')
        # Setting the type of the member '__doc__' of a type (line 90)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), func_47690, '__doc__', result_and_keyword_47689)
        # Getting the type of 'func' (line 91)
        func_47691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'func')
        # Assigning a type to the variable 'stypy_return_type' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'stypy_return_type', func_47691)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_47692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_47692)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_47692


# Assigning a type to the variable 'Appender' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'Appender', Appender)

@norecursion
def dedent(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dedent'
    module_type_store = module_type_store.open_function_context('dedent', 94, 0, False)
    
    # Passed parameters checking function
    dedent.stypy_localization = localization
    dedent.stypy_type_of_self = None
    dedent.stypy_type_store = module_type_store
    dedent.stypy_function_name = 'dedent'
    dedent.stypy_param_names_list = ['func']
    dedent.stypy_varargs_param_name = None
    dedent.stypy_kwargs_param_name = None
    dedent.stypy_call_defaults = defaults
    dedent.stypy_call_varargs = varargs
    dedent.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dedent', ['func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dedent', localization, ['func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dedent(...)' code ##################

    unicode_47693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 4), 'unicode', u'Dedent a docstring (if present)')
    
    # Assigning a BoolOp to a Attribute (line 96):
    
    # Evaluating a boolean operation
    # Getting the type of 'func' (line 96)
    func_47694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 19), 'func')
    # Obtaining the member '__doc__' of a type (line 96)
    doc___47695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 19), func_47694, '__doc__')
    
    # Call to dedent(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'func' (line 96)
    func_47698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 49), 'func', False)
    # Obtaining the member '__doc__' of a type (line 96)
    doc___47699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 49), func_47698, '__doc__')
    # Processing the call keyword arguments (line 96)
    kwargs_47700 = {}
    # Getting the type of 'cbook' (line 96)
    cbook_47696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 36), 'cbook', False)
    # Obtaining the member 'dedent' of a type (line 96)
    dedent_47697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 36), cbook_47696, 'dedent')
    # Calling dedent(args, kwargs) (line 96)
    dedent_call_result_47701 = invoke(stypy.reporting.localization.Localization(__file__, 96, 36), dedent_47697, *[doc___47699], **kwargs_47700)
    
    # Applying the binary operator 'and' (line 96)
    result_and_keyword_47702 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 19), 'and', doc___47695, dedent_call_result_47701)
    
    # Getting the type of 'func' (line 96)
    func_47703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'func')
    # Setting the type of the member '__doc__' of a type (line 96)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 4), func_47703, '__doc__', result_and_keyword_47702)
    # Getting the type of 'func' (line 97)
    func_47704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'func')
    # Assigning a type to the variable 'stypy_return_type' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type', func_47704)
    
    # ################# End of 'dedent(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dedent' in the type store
    # Getting the type of 'stypy_return_type' (line 94)
    stypy_return_type_47705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_47705)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dedent'
    return stypy_return_type_47705

# Assigning a type to the variable 'dedent' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'dedent', dedent)

@norecursion
def copy(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'copy'
    module_type_store = module_type_store.open_function_context('copy', 100, 0, False)
    
    # Passed parameters checking function
    copy.stypy_localization = localization
    copy.stypy_type_of_self = None
    copy.stypy_type_store = module_type_store
    copy.stypy_function_name = 'copy'
    copy.stypy_param_names_list = ['source']
    copy.stypy_varargs_param_name = None
    copy.stypy_kwargs_param_name = None
    copy.stypy_call_defaults = defaults
    copy.stypy_call_varargs = varargs
    copy.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'copy', ['source'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'copy', localization, ['source'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'copy(...)' code ##################

    unicode_47706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 4), 'unicode', u'Copy a docstring from another source function (if present)')

    @norecursion
    def do_copy(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'do_copy'
        module_type_store = module_type_store.open_function_context('do_copy', 102, 4, False)
        
        # Passed parameters checking function
        do_copy.stypy_localization = localization
        do_copy.stypy_type_of_self = None
        do_copy.stypy_type_store = module_type_store
        do_copy.stypy_function_name = 'do_copy'
        do_copy.stypy_param_names_list = ['target']
        do_copy.stypy_varargs_param_name = None
        do_copy.stypy_kwargs_param_name = None
        do_copy.stypy_call_defaults = defaults
        do_copy.stypy_call_varargs = varargs
        do_copy.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'do_copy', ['target'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'do_copy', localization, ['target'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'do_copy(...)' code ##################

        
        # Getting the type of 'source' (line 103)
        source_47707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'source')
        # Obtaining the member '__doc__' of a type (line 103)
        doc___47708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 11), source_47707, '__doc__')
        # Testing the type of an if condition (line 103)
        if_condition_47709 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 8), doc___47708)
        # Assigning a type to the variable 'if_condition_47709' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'if_condition_47709', if_condition_47709)
        # SSA begins for if statement (line 103)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 104):
        # Getting the type of 'source' (line 104)
        source_47710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 'source')
        # Obtaining the member '__doc__' of a type (line 104)
        doc___47711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 29), source_47710, '__doc__')
        # Getting the type of 'target' (line 104)
        target_47712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'target')
        # Setting the type of the member '__doc__' of a type (line 104)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), target_47712, '__doc__', doc___47711)
        # SSA join for if statement (line 103)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'target' (line 105)
        target_47713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'target')
        # Assigning a type to the variable 'stypy_return_type' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'stypy_return_type', target_47713)
        
        # ################# End of 'do_copy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'do_copy' in the type store
        # Getting the type of 'stypy_return_type' (line 102)
        stypy_return_type_47714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_47714)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'do_copy'
        return stypy_return_type_47714

    # Assigning a type to the variable 'do_copy' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'do_copy', do_copy)
    # Getting the type of 'do_copy' (line 106)
    do_copy_47715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'do_copy')
    # Assigning a type to the variable 'stypy_return_type' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type', do_copy_47715)
    
    # ################# End of 'copy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'copy' in the type store
    # Getting the type of 'stypy_return_type' (line 100)
    stypy_return_type_47716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_47716)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'copy'
    return stypy_return_type_47716

# Assigning a type to the variable 'copy' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'copy', copy)

# Assigning a Call to a Name (line 110):

# Call to Substitution(...): (line 110)
# Processing the call keyword arguments (line 110)
kwargs_47718 = {}
# Getting the type of 'Substitution' (line 110)
Substitution_47717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 10), 'Substitution', False)
# Calling Substitution(args, kwargs) (line 110)
Substitution_call_result_47719 = invoke(stypy.reporting.localization.Localization(__file__, 110, 10), Substitution_47717, *[], **kwargs_47718)

# Assigning a type to the variable 'interpd' (line 110)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'interpd', Substitution_call_result_47719)

@norecursion
def dedent_interpd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dedent_interpd'
    module_type_store = module_type_store.open_function_context('dedent_interpd', 113, 0, False)
    
    # Passed parameters checking function
    dedent_interpd.stypy_localization = localization
    dedent_interpd.stypy_type_of_self = None
    dedent_interpd.stypy_type_store = module_type_store
    dedent_interpd.stypy_function_name = 'dedent_interpd'
    dedent_interpd.stypy_param_names_list = ['func']
    dedent_interpd.stypy_varargs_param_name = None
    dedent_interpd.stypy_kwargs_param_name = None
    dedent_interpd.stypy_call_defaults = defaults
    dedent_interpd.stypy_call_varargs = varargs
    dedent_interpd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dedent_interpd', ['func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dedent_interpd', localization, ['func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dedent_interpd(...)' code ##################

    unicode_47720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, (-1)), 'unicode', u'A special case of the interpd that first performs a dedent on\n    the incoming docstring')
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'func' (line 116)
    func_47722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 18), 'func', False)
    # Getting the type of 'types' (line 116)
    types_47723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'types', False)
    # Obtaining the member 'MethodType' of a type (line 116)
    MethodType_47724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 24), types_47723, 'MethodType')
    # Processing the call keyword arguments (line 116)
    kwargs_47725 = {}
    # Getting the type of 'isinstance' (line 116)
    isinstance_47721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 116)
    isinstance_call_result_47726 = invoke(stypy.reporting.localization.Localization(__file__, 116, 7), isinstance_47721, *[func_47722, MethodType_47724], **kwargs_47725)
    
    
    # Getting the type of 'six' (line 116)
    six_47727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 50), 'six')
    # Obtaining the member 'PY3' of a type (line 116)
    PY3_47728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 50), six_47727, 'PY3')
    # Applying the 'not' unary operator (line 116)
    result_not__47729 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 46), 'not', PY3_47728)
    
    # Applying the binary operator 'and' (line 116)
    result_and_keyword_47730 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 7), 'and', isinstance_call_result_47726, result_not__47729)
    
    # Testing the type of an if condition (line 116)
    if_condition_47731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 4), result_and_keyword_47730)
    # Assigning a type to the variable 'if_condition_47731' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'if_condition_47731', if_condition_47731)
    # SSA begins for if statement (line 116)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 117):
    # Getting the type of 'func' (line 117)
    func_47732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'func')
    # Obtaining the member 'im_func' of a type (line 117)
    im_func_47733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 15), func_47732, 'im_func')
    # Assigning a type to the variable 'func' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'func', im_func_47733)
    # SSA join for if statement (line 116)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to interpd(...): (line 118)
    # Processing the call arguments (line 118)
    
    # Call to dedent(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'func' (line 118)
    func_47736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 26), 'func', False)
    # Processing the call keyword arguments (line 118)
    kwargs_47737 = {}
    # Getting the type of 'dedent' (line 118)
    dedent_47735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 19), 'dedent', False)
    # Calling dedent(args, kwargs) (line 118)
    dedent_call_result_47738 = invoke(stypy.reporting.localization.Localization(__file__, 118, 19), dedent_47735, *[func_47736], **kwargs_47737)
    
    # Processing the call keyword arguments (line 118)
    kwargs_47739 = {}
    # Getting the type of 'interpd' (line 118)
    interpd_47734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 11), 'interpd', False)
    # Calling interpd(args, kwargs) (line 118)
    interpd_call_result_47740 = invoke(stypy.reporting.localization.Localization(__file__, 118, 11), interpd_47734, *[dedent_call_result_47738], **kwargs_47739)
    
    # Assigning a type to the variable 'stypy_return_type' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type', interpd_call_result_47740)
    
    # ################# End of 'dedent_interpd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dedent_interpd' in the type store
    # Getting the type of 'stypy_return_type' (line 113)
    stypy_return_type_47741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_47741)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dedent_interpd'
    return stypy_return_type_47741

# Assigning a type to the variable 'dedent_interpd' (line 113)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), 'dedent_interpd', dedent_interpd)

@norecursion
def copy_dedent(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'copy_dedent'
    module_type_store = module_type_store.open_function_context('copy_dedent', 121, 0, False)
    
    # Passed parameters checking function
    copy_dedent.stypy_localization = localization
    copy_dedent.stypy_type_of_self = None
    copy_dedent.stypy_type_store = module_type_store
    copy_dedent.stypy_function_name = 'copy_dedent'
    copy_dedent.stypy_param_names_list = ['source']
    copy_dedent.stypy_varargs_param_name = None
    copy_dedent.stypy_kwargs_param_name = None
    copy_dedent.stypy_call_defaults = defaults
    copy_dedent.stypy_call_varargs = varargs
    copy_dedent.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'copy_dedent', ['source'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'copy_dedent', localization, ['source'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'copy_dedent(...)' code ##################

    unicode_47742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, (-1)), 'unicode', u'A decorator that will copy the docstring from the source and\n    then dedent it')

    @norecursion
    def _stypy_temp_lambda_11(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_11'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_11', 128, 11, True)
        # Passed parameters checking function
        _stypy_temp_lambda_11.stypy_localization = localization
        _stypy_temp_lambda_11.stypy_type_of_self = None
        _stypy_temp_lambda_11.stypy_type_store = module_type_store
        _stypy_temp_lambda_11.stypy_function_name = '_stypy_temp_lambda_11'
        _stypy_temp_lambda_11.stypy_param_names_list = ['target']
        _stypy_temp_lambda_11.stypy_varargs_param_name = None
        _stypy_temp_lambda_11.stypy_kwargs_param_name = None
        _stypy_temp_lambda_11.stypy_call_defaults = defaults
        _stypy_temp_lambda_11.stypy_call_varargs = varargs
        _stypy_temp_lambda_11.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_11', ['target'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_11', ['target'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to dedent(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Call to (...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'target' (line 128)
        target_47748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 46), 'target', False)
        # Processing the call keyword arguments (line 128)
        kwargs_47749 = {}
        
        # Call to copy(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'source' (line 128)
        source_47745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 38), 'source', False)
        # Processing the call keyword arguments (line 128)
        kwargs_47746 = {}
        # Getting the type of 'copy' (line 128)
        copy_47744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 33), 'copy', False)
        # Calling copy(args, kwargs) (line 128)
        copy_call_result_47747 = invoke(stypy.reporting.localization.Localization(__file__, 128, 33), copy_47744, *[source_47745], **kwargs_47746)
        
        # Calling (args, kwargs) (line 128)
        _call_result_47750 = invoke(stypy.reporting.localization.Localization(__file__, 128, 33), copy_call_result_47747, *[target_47748], **kwargs_47749)
        
        # Processing the call keyword arguments (line 128)
        kwargs_47751 = {}
        # Getting the type of 'dedent' (line 128)
        dedent_47743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 26), 'dedent', False)
        # Calling dedent(args, kwargs) (line 128)
        dedent_call_result_47752 = invoke(stypy.reporting.localization.Localization(__file__, 128, 26), dedent_47743, *[_call_result_47750], **kwargs_47751)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'stypy_return_type', dedent_call_result_47752)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_11' in the type store
        # Getting the type of 'stypy_return_type' (line 128)
        stypy_return_type_47753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_47753)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_11'
        return stypy_return_type_47753

    # Assigning a type to the variable '_stypy_temp_lambda_11' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), '_stypy_temp_lambda_11', _stypy_temp_lambda_11)
    # Getting the type of '_stypy_temp_lambda_11' (line 128)
    _stypy_temp_lambda_11_47754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), '_stypy_temp_lambda_11')
    # Assigning a type to the variable 'stypy_return_type' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type', _stypy_temp_lambda_11_47754)
    
    # ################# End of 'copy_dedent(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'copy_dedent' in the type store
    # Getting the type of 'stypy_return_type' (line 121)
    stypy_return_type_47755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_47755)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'copy_dedent'
    return stypy_return_type_47755

# Assigning a type to the variable 'copy_dedent' (line 121)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'copy_dedent', copy_dedent)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
