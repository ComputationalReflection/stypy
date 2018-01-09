
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Originally from astropy project (http://astropy.org), under BSD
2: # 3-clause license.
3: 
4: from __future__ import (absolute_import, division, print_function,
5:                         unicode_literals)
6: 
7: import contextlib
8: import socket
9: 
10: from six.moves import urllib
11: 
12: # save original socket method for restoration
13: # These are global so that re-calling the turn_off_internet function doesn't
14: # overwrite them again
15: socket_original = socket.socket
16: socket_create_connection = socket.create_connection
17: socket_bind = socket.socket.bind
18: socket_connect = socket.socket.connect
19: 
20: 
21: INTERNET_OFF = False
22: 
23: # urllib2 uses a global variable to cache its default "opener" for opening
24: # connections for various protocols; we store it off here so we can restore to
25: # the default after re-enabling internet use
26: _orig_opener = None
27: 
28: 
29: # ::1 is apparently another valid name for localhost?
30: # it is returned by getaddrinfo when that function is given localhost
31: 
32: def check_internet_off(original_function):
33:     '''
34:     Wraps ``original_function``, which in most cases is assumed
35:     to be a `socket.socket` method, to raise an `IOError` for any operations
36:     on non-local AF_INET sockets.
37:     '''
38: 
39:     def new_function(*args, **kwargs):
40:         if isinstance(args[0], socket.socket):
41:             if not args[0].family in (socket.AF_INET, socket.AF_INET6):
42:                 # Should be fine in all but some very obscure cases
43:                 # More to the point, we don't want to affect AF_UNIX
44:                 # sockets.
45:                 return original_function(*args, **kwargs)
46:             host = args[1][0]
47:             addr_arg = 1
48:             valid_hosts = ('localhost', '127.0.0.1', '::1')
49:         else:
50:             # The only other function this is used to wrap currently is
51:             # socket.create_connection, which should be passed a 2-tuple, but
52:             # we'll check just in case
53:             if not (isinstance(args[0], tuple) and len(args[0]) == 2):
54:                 return original_function(*args, **kwargs)
55: 
56:             host = args[0][0]
57:             addr_arg = 0
58:             valid_hosts = ('localhost', '127.0.0.1')
59: 
60:         hostname = socket.gethostname()
61:         fqdn = socket.getfqdn()
62: 
63:         if host in (hostname, fqdn):
64:             host = 'localhost'
65:             new_addr = (host, args[addr_arg][1])
66:             args = args[:addr_arg] + (new_addr,) + args[addr_arg + 1:]
67: 
68:         if any([h in host for h in valid_hosts]):
69:             return original_function(*args, **kwargs)
70:         else:
71:             raise IOError("An attempt was made to connect to the internet "
72:                           "by a test that was not marked `remote_data`.")
73:     return new_function
74: 
75: 
76: def turn_off_internet(verbose=False):
77:     '''
78:     Disable internet access via python by preventing connections from being
79:     created using the socket module.  Presumably this could be worked around by
80:     using some other means of accessing the internet, but all default python
81:     modules (urllib, requests, etc.) use socket [citation needed].
82:     '''
83: 
84:     global INTERNET_OFF
85:     global _orig_opener
86: 
87:     if INTERNET_OFF:
88:         return
89: 
90:     INTERNET_OFF = True
91: 
92:     __tracebackhide__ = True
93:     if verbose:
94:         print("Internet access disabled")
95: 
96:     # Update urllib2 to force it not to use any proxies
97:     # Must use {} here (the default of None will kick off an automatic search
98:     # for proxies)
99:     _orig_opener = urllib.request.build_opener()
100:     no_proxy_handler = urllib.request.ProxyHandler({})
101:     opener = urllib.request.build_opener(no_proxy_handler)
102:     urllib.request.install_opener(opener)
103: 
104:     socket.create_connection = check_internet_off(socket_create_connection)
105:     socket.socket.bind = check_internet_off(socket_bind)
106:     socket.socket.connect = check_internet_off(socket_connect)
107: 
108:     return socket
109: 
110: 
111: def turn_on_internet(verbose=False):
112:     '''
113:     Restore internet access.  Not used, but kept in case it is needed.
114:     '''
115: 
116:     global INTERNET_OFF
117:     global _orig_opener
118: 
119:     if not INTERNET_OFF:
120:         return
121: 
122:     INTERNET_OFF = False
123: 
124:     if verbose:
125:         print("Internet access enabled")
126: 
127:     urllib.request.install_opener(_orig_opener)
128: 
129:     socket.create_connection = socket_create_connection
130:     socket.socket.bind = socket_bind
131:     socket.socket.connect = socket_connect
132:     return socket
133: 
134: 
135: @contextlib.contextmanager
136: def no_internet(verbose=False):
137:     '''Context manager to temporarily disable internet access (if not already
138:     disabled).  If it was already disabled before entering the context manager
139:     (i.e. `turn_off_internet` was called previously) then this is a no-op and
140:     leaves internet access disabled until a manual call to `turn_on_internet`.
141:     '''
142: 
143:     already_disabled = INTERNET_OFF
144: 
145:     turn_off_internet(verbose=verbose)
146:     try:
147:         yield
148:     finally:
149:         if not already_disabled:
150:             turn_on_internet(verbose=verbose)
151: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import contextlib' statement (line 7)
import contextlib

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'contextlib', contextlib, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import socket' statement (line 8)
import socket

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'socket', socket, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from six.moves import urllib' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_291699 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'six.moves')

if (type(import_291699) is not StypyTypeError):

    if (import_291699 != 'pyd_module'):
        __import__(import_291699)
        sys_modules_291700 = sys.modules[import_291699]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'six.moves', sys_modules_291700.module_type_store, module_type_store, ['urllib'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_291700, sys_modules_291700.module_type_store, module_type_store)
    else:
        from six.moves import urllib

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'six.moves', None, module_type_store, ['urllib'], [urllib])

else:
    # Assigning a type to the variable 'six.moves' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'six.moves', import_291699)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')


# Assigning a Attribute to a Name (line 15):
# Getting the type of 'socket' (line 15)
socket_291701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 18), 'socket')
# Obtaining the member 'socket' of a type (line 15)
socket_291702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 18), socket_291701, 'socket')
# Assigning a type to the variable 'socket_original' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'socket_original', socket_291702)

# Assigning a Attribute to a Name (line 16):
# Getting the type of 'socket' (line 16)
socket_291703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 27), 'socket')
# Obtaining the member 'create_connection' of a type (line 16)
create_connection_291704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 27), socket_291703, 'create_connection')
# Assigning a type to the variable 'socket_create_connection' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'socket_create_connection', create_connection_291704)

# Assigning a Attribute to a Name (line 17):
# Getting the type of 'socket' (line 17)
socket_291705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 14), 'socket')
# Obtaining the member 'socket' of a type (line 17)
socket_291706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 14), socket_291705, 'socket')
# Obtaining the member 'bind' of a type (line 17)
bind_291707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 14), socket_291706, 'bind')
# Assigning a type to the variable 'socket_bind' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'socket_bind', bind_291707)

# Assigning a Attribute to a Name (line 18):
# Getting the type of 'socket' (line 18)
socket_291708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), 'socket')
# Obtaining the member 'socket' of a type (line 18)
socket_291709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 17), socket_291708, 'socket')
# Obtaining the member 'connect' of a type (line 18)
connect_291710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 17), socket_291709, 'connect')
# Assigning a type to the variable 'socket_connect' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'socket_connect', connect_291710)

# Assigning a Name to a Name (line 21):
# Getting the type of 'False' (line 21)
False_291711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'False')
# Assigning a type to the variable 'INTERNET_OFF' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'INTERNET_OFF', False_291711)

# Assigning a Name to a Name (line 26):
# Getting the type of 'None' (line 26)
None_291712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'None')
# Assigning a type to the variable '_orig_opener' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '_orig_opener', None_291712)

@norecursion
def check_internet_off(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_internet_off'
    module_type_store = module_type_store.open_function_context('check_internet_off', 32, 0, False)
    
    # Passed parameters checking function
    check_internet_off.stypy_localization = localization
    check_internet_off.stypy_type_of_self = None
    check_internet_off.stypy_type_store = module_type_store
    check_internet_off.stypy_function_name = 'check_internet_off'
    check_internet_off.stypy_param_names_list = ['original_function']
    check_internet_off.stypy_varargs_param_name = None
    check_internet_off.stypy_kwargs_param_name = None
    check_internet_off.stypy_call_defaults = defaults
    check_internet_off.stypy_call_varargs = varargs
    check_internet_off.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_internet_off', ['original_function'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_internet_off', localization, ['original_function'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_internet_off(...)' code ##################

    unicode_291713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, (-1)), 'unicode', u'\n    Wraps ``original_function``, which in most cases is assumed\n    to be a `socket.socket` method, to raise an `IOError` for any operations\n    on non-local AF_INET sockets.\n    ')

    @norecursion
    def new_function(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_function'
        module_type_store = module_type_store.open_function_context('new_function', 39, 4, False)
        
        # Passed parameters checking function
        new_function.stypy_localization = localization
        new_function.stypy_type_of_self = None
        new_function.stypy_type_store = module_type_store
        new_function.stypy_function_name = 'new_function'
        new_function.stypy_param_names_list = []
        new_function.stypy_varargs_param_name = 'args'
        new_function.stypy_kwargs_param_name = 'kwargs'
        new_function.stypy_call_defaults = defaults
        new_function.stypy_call_varargs = varargs
        new_function.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'new_function', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'new_function', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'new_function(...)' code ##################

        
        
        # Call to isinstance(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Obtaining the type of the subscript
        int_291715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 27), 'int')
        # Getting the type of 'args' (line 40)
        args_291716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 22), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 40)
        getitem___291717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 22), args_291716, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 40)
        subscript_call_result_291718 = invoke(stypy.reporting.localization.Localization(__file__, 40, 22), getitem___291717, int_291715)
        
        # Getting the type of 'socket' (line 40)
        socket_291719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 31), 'socket', False)
        # Obtaining the member 'socket' of a type (line 40)
        socket_291720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 31), socket_291719, 'socket')
        # Processing the call keyword arguments (line 40)
        kwargs_291721 = {}
        # Getting the type of 'isinstance' (line 40)
        isinstance_291714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 40)
        isinstance_call_result_291722 = invoke(stypy.reporting.localization.Localization(__file__, 40, 11), isinstance_291714, *[subscript_call_result_291718, socket_291720], **kwargs_291721)
        
        # Testing the type of an if condition (line 40)
        if_condition_291723 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 8), isinstance_call_result_291722)
        # Assigning a type to the variable 'if_condition_291723' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'if_condition_291723', if_condition_291723)
        # SSA begins for if statement (line 40)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        
        # Obtaining the type of the subscript
        int_291724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 24), 'int')
        # Getting the type of 'args' (line 41)
        args_291725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 19), 'args')
        # Obtaining the member '__getitem__' of a type (line 41)
        getitem___291726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 19), args_291725, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 41)
        subscript_call_result_291727 = invoke(stypy.reporting.localization.Localization(__file__, 41, 19), getitem___291726, int_291724)
        
        # Obtaining the member 'family' of a type (line 41)
        family_291728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 19), subscript_call_result_291727, 'family')
        
        # Obtaining an instance of the builtin type 'tuple' (line 41)
        tuple_291729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 41)
        # Adding element type (line 41)
        # Getting the type of 'socket' (line 41)
        socket_291730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 38), 'socket')
        # Obtaining the member 'AF_INET' of a type (line 41)
        AF_INET_291731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 38), socket_291730, 'AF_INET')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 38), tuple_291729, AF_INET_291731)
        # Adding element type (line 41)
        # Getting the type of 'socket' (line 41)
        socket_291732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 54), 'socket')
        # Obtaining the member 'AF_INET6' of a type (line 41)
        AF_INET6_291733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 54), socket_291732, 'AF_INET6')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 38), tuple_291729, AF_INET6_291733)
        
        # Applying the binary operator 'in' (line 41)
        result_contains_291734 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 19), 'in', family_291728, tuple_291729)
        
        # Applying the 'not' unary operator (line 41)
        result_not__291735 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 15), 'not', result_contains_291734)
        
        # Testing the type of an if condition (line 41)
        if_condition_291736 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 12), result_not__291735)
        # Assigning a type to the variable 'if_condition_291736' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'if_condition_291736', if_condition_291736)
        # SSA begins for if statement (line 41)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to original_function(...): (line 45)
        # Getting the type of 'args' (line 45)
        args_291738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 42), 'args', False)
        # Processing the call keyword arguments (line 45)
        # Getting the type of 'kwargs' (line 45)
        kwargs_291739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 50), 'kwargs', False)
        kwargs_291740 = {'kwargs_291739': kwargs_291739}
        # Getting the type of 'original_function' (line 45)
        original_function_291737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 23), 'original_function', False)
        # Calling original_function(args, kwargs) (line 45)
        original_function_call_result_291741 = invoke(stypy.reporting.localization.Localization(__file__, 45, 23), original_function_291737, *[args_291738], **kwargs_291740)
        
        # Assigning a type to the variable 'stypy_return_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'stypy_return_type', original_function_call_result_291741)
        # SSA join for if statement (line 41)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 46):
        
        # Obtaining the type of the subscript
        int_291742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 27), 'int')
        
        # Obtaining the type of the subscript
        int_291743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 24), 'int')
        # Getting the type of 'args' (line 46)
        args_291744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 19), 'args')
        # Obtaining the member '__getitem__' of a type (line 46)
        getitem___291745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 19), args_291744, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 46)
        subscript_call_result_291746 = invoke(stypy.reporting.localization.Localization(__file__, 46, 19), getitem___291745, int_291743)
        
        # Obtaining the member '__getitem__' of a type (line 46)
        getitem___291747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 19), subscript_call_result_291746, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 46)
        subscript_call_result_291748 = invoke(stypy.reporting.localization.Localization(__file__, 46, 19), getitem___291747, int_291742)
        
        # Assigning a type to the variable 'host' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'host', subscript_call_result_291748)
        
        # Assigning a Num to a Name (line 47):
        int_291749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 23), 'int')
        # Assigning a type to the variable 'addr_arg' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'addr_arg', int_291749)
        
        # Assigning a Tuple to a Name (line 48):
        
        # Obtaining an instance of the builtin type 'tuple' (line 48)
        tuple_291750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 48)
        # Adding element type (line 48)
        unicode_291751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 27), 'unicode', u'localhost')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 27), tuple_291750, unicode_291751)
        # Adding element type (line 48)
        unicode_291752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 40), 'unicode', u'127.0.0.1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 27), tuple_291750, unicode_291752)
        # Adding element type (line 48)
        unicode_291753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 53), 'unicode', u'::1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 27), tuple_291750, unicode_291753)
        
        # Assigning a type to the variable 'valid_hosts' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'valid_hosts', tuple_291750)
        # SSA branch for the else part of an if statement (line 40)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Obtaining the type of the subscript
        int_291755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 36), 'int')
        # Getting the type of 'args' (line 53)
        args_291756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 31), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 53)
        getitem___291757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 31), args_291756, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 53)
        subscript_call_result_291758 = invoke(stypy.reporting.localization.Localization(__file__, 53, 31), getitem___291757, int_291755)
        
        # Getting the type of 'tuple' (line 53)
        tuple_291759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 40), 'tuple', False)
        # Processing the call keyword arguments (line 53)
        kwargs_291760 = {}
        # Getting the type of 'isinstance' (line 53)
        isinstance_291754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 53)
        isinstance_call_result_291761 = invoke(stypy.reporting.localization.Localization(__file__, 53, 20), isinstance_291754, *[subscript_call_result_291758, tuple_291759], **kwargs_291760)
        
        
        
        # Call to len(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Obtaining the type of the subscript
        int_291763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 60), 'int')
        # Getting the type of 'args' (line 53)
        args_291764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 55), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 53)
        getitem___291765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 55), args_291764, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 53)
        subscript_call_result_291766 = invoke(stypy.reporting.localization.Localization(__file__, 53, 55), getitem___291765, int_291763)
        
        # Processing the call keyword arguments (line 53)
        kwargs_291767 = {}
        # Getting the type of 'len' (line 53)
        len_291762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 51), 'len', False)
        # Calling len(args, kwargs) (line 53)
        len_call_result_291768 = invoke(stypy.reporting.localization.Localization(__file__, 53, 51), len_291762, *[subscript_call_result_291766], **kwargs_291767)
        
        int_291769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 67), 'int')
        # Applying the binary operator '==' (line 53)
        result_eq_291770 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 51), '==', len_call_result_291768, int_291769)
        
        # Applying the binary operator 'and' (line 53)
        result_and_keyword_291771 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 20), 'and', isinstance_call_result_291761, result_eq_291770)
        
        # Applying the 'not' unary operator (line 53)
        result_not__291772 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 15), 'not', result_and_keyword_291771)
        
        # Testing the type of an if condition (line 53)
        if_condition_291773 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 12), result_not__291772)
        # Assigning a type to the variable 'if_condition_291773' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'if_condition_291773', if_condition_291773)
        # SSA begins for if statement (line 53)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to original_function(...): (line 54)
        # Getting the type of 'args' (line 54)
        args_291775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 42), 'args', False)
        # Processing the call keyword arguments (line 54)
        # Getting the type of 'kwargs' (line 54)
        kwargs_291776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 50), 'kwargs', False)
        kwargs_291777 = {'kwargs_291776': kwargs_291776}
        # Getting the type of 'original_function' (line 54)
        original_function_291774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 23), 'original_function', False)
        # Calling original_function(args, kwargs) (line 54)
        original_function_call_result_291778 = invoke(stypy.reporting.localization.Localization(__file__, 54, 23), original_function_291774, *[args_291775], **kwargs_291777)
        
        # Assigning a type to the variable 'stypy_return_type' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'stypy_return_type', original_function_call_result_291778)
        # SSA join for if statement (line 53)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 56):
        
        # Obtaining the type of the subscript
        int_291779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 27), 'int')
        
        # Obtaining the type of the subscript
        int_291780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 24), 'int')
        # Getting the type of 'args' (line 56)
        args_291781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 19), 'args')
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___291782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 19), args_291781, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_291783 = invoke(stypy.reporting.localization.Localization(__file__, 56, 19), getitem___291782, int_291780)
        
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___291784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 19), subscript_call_result_291783, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_291785 = invoke(stypy.reporting.localization.Localization(__file__, 56, 19), getitem___291784, int_291779)
        
        # Assigning a type to the variable 'host' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'host', subscript_call_result_291785)
        
        # Assigning a Num to a Name (line 57):
        int_291786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 23), 'int')
        # Assigning a type to the variable 'addr_arg' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'addr_arg', int_291786)
        
        # Assigning a Tuple to a Name (line 58):
        
        # Obtaining an instance of the builtin type 'tuple' (line 58)
        tuple_291787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 58)
        # Adding element type (line 58)
        unicode_291788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 27), 'unicode', u'localhost')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 27), tuple_291787, unicode_291788)
        # Adding element type (line 58)
        unicode_291789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 40), 'unicode', u'127.0.0.1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 27), tuple_291787, unicode_291789)
        
        # Assigning a type to the variable 'valid_hosts' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'valid_hosts', tuple_291787)
        # SSA join for if statement (line 40)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 60):
        
        # Call to gethostname(...): (line 60)
        # Processing the call keyword arguments (line 60)
        kwargs_291792 = {}
        # Getting the type of 'socket' (line 60)
        socket_291790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 19), 'socket', False)
        # Obtaining the member 'gethostname' of a type (line 60)
        gethostname_291791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 19), socket_291790, 'gethostname')
        # Calling gethostname(args, kwargs) (line 60)
        gethostname_call_result_291793 = invoke(stypy.reporting.localization.Localization(__file__, 60, 19), gethostname_291791, *[], **kwargs_291792)
        
        # Assigning a type to the variable 'hostname' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'hostname', gethostname_call_result_291793)
        
        # Assigning a Call to a Name (line 61):
        
        # Call to getfqdn(...): (line 61)
        # Processing the call keyword arguments (line 61)
        kwargs_291796 = {}
        # Getting the type of 'socket' (line 61)
        socket_291794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'socket', False)
        # Obtaining the member 'getfqdn' of a type (line 61)
        getfqdn_291795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 15), socket_291794, 'getfqdn')
        # Calling getfqdn(args, kwargs) (line 61)
        getfqdn_call_result_291797 = invoke(stypy.reporting.localization.Localization(__file__, 61, 15), getfqdn_291795, *[], **kwargs_291796)
        
        # Assigning a type to the variable 'fqdn' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'fqdn', getfqdn_call_result_291797)
        
        
        # Getting the type of 'host' (line 63)
        host_291798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 11), 'host')
        
        # Obtaining an instance of the builtin type 'tuple' (line 63)
        tuple_291799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 63)
        # Adding element type (line 63)
        # Getting the type of 'hostname' (line 63)
        hostname_291800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'hostname')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 20), tuple_291799, hostname_291800)
        # Adding element type (line 63)
        # Getting the type of 'fqdn' (line 63)
        fqdn_291801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'fqdn')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 20), tuple_291799, fqdn_291801)
        
        # Applying the binary operator 'in' (line 63)
        result_contains_291802 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 11), 'in', host_291798, tuple_291799)
        
        # Testing the type of an if condition (line 63)
        if_condition_291803 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 63, 8), result_contains_291802)
        # Assigning a type to the variable 'if_condition_291803' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'if_condition_291803', if_condition_291803)
        # SSA begins for if statement (line 63)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 64):
        unicode_291804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 19), 'unicode', u'localhost')
        # Assigning a type to the variable 'host' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'host', unicode_291804)
        
        # Assigning a Tuple to a Name (line 65):
        
        # Obtaining an instance of the builtin type 'tuple' (line 65)
        tuple_291805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 65)
        # Adding element type (line 65)
        # Getting the type of 'host' (line 65)
        host_291806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 24), 'host')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 24), tuple_291805, host_291806)
        # Adding element type (line 65)
        
        # Obtaining the type of the subscript
        int_291807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 45), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'addr_arg' (line 65)
        addr_arg_291808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 35), 'addr_arg')
        # Getting the type of 'args' (line 65)
        args_291809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'args')
        # Obtaining the member '__getitem__' of a type (line 65)
        getitem___291810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 30), args_291809, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 65)
        subscript_call_result_291811 = invoke(stypy.reporting.localization.Localization(__file__, 65, 30), getitem___291810, addr_arg_291808)
        
        # Obtaining the member '__getitem__' of a type (line 65)
        getitem___291812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 30), subscript_call_result_291811, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 65)
        subscript_call_result_291813 = invoke(stypy.reporting.localization.Localization(__file__, 65, 30), getitem___291812, int_291807)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 24), tuple_291805, subscript_call_result_291813)
        
        # Assigning a type to the variable 'new_addr' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'new_addr', tuple_291805)
        
        # Assigning a BinOp to a Name (line 66):
        
        # Obtaining the type of the subscript
        # Getting the type of 'addr_arg' (line 66)
        addr_arg_291814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'addr_arg')
        slice_291815 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 66, 19), None, addr_arg_291814, None)
        # Getting the type of 'args' (line 66)
        args_291816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 19), 'args')
        # Obtaining the member '__getitem__' of a type (line 66)
        getitem___291817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 19), args_291816, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 66)
        subscript_call_result_291818 = invoke(stypy.reporting.localization.Localization(__file__, 66, 19), getitem___291817, slice_291815)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 66)
        tuple_291819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 66)
        # Adding element type (line 66)
        # Getting the type of 'new_addr' (line 66)
        new_addr_291820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 38), 'new_addr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 38), tuple_291819, new_addr_291820)
        
        # Applying the binary operator '+' (line 66)
        result_add_291821 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 19), '+', subscript_call_result_291818, tuple_291819)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'addr_arg' (line 66)
        addr_arg_291822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 56), 'addr_arg')
        int_291823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 67), 'int')
        # Applying the binary operator '+' (line 66)
        result_add_291824 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 56), '+', addr_arg_291822, int_291823)
        
        slice_291825 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 66, 51), result_add_291824, None, None)
        # Getting the type of 'args' (line 66)
        args_291826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 51), 'args')
        # Obtaining the member '__getitem__' of a type (line 66)
        getitem___291827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 51), args_291826, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 66)
        subscript_call_result_291828 = invoke(stypy.reporting.localization.Localization(__file__, 66, 51), getitem___291827, slice_291825)
        
        # Applying the binary operator '+' (line 66)
        result_add_291829 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 49), '+', result_add_291821, subscript_call_result_291828)
        
        # Assigning a type to the variable 'args' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'args', result_add_291829)
        # SSA join for if statement (line 63)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to any(...): (line 68)
        # Processing the call arguments (line 68)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'valid_hosts' (line 68)
        valid_hosts_291834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 35), 'valid_hosts', False)
        comprehension_291835 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 16), valid_hosts_291834)
        # Assigning a type to the variable 'h' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'h', comprehension_291835)
        
        # Getting the type of 'h' (line 68)
        h_291831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'h', False)
        # Getting the type of 'host' (line 68)
        host_291832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 21), 'host', False)
        # Applying the binary operator 'in' (line 68)
        result_contains_291833 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 16), 'in', h_291831, host_291832)
        
        list_291836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 16), list_291836, result_contains_291833)
        # Processing the call keyword arguments (line 68)
        kwargs_291837 = {}
        # Getting the type of 'any' (line 68)
        any_291830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'any', False)
        # Calling any(args, kwargs) (line 68)
        any_call_result_291838 = invoke(stypy.reporting.localization.Localization(__file__, 68, 11), any_291830, *[list_291836], **kwargs_291837)
        
        # Testing the type of an if condition (line 68)
        if_condition_291839 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 8), any_call_result_291838)
        # Assigning a type to the variable 'if_condition_291839' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'if_condition_291839', if_condition_291839)
        # SSA begins for if statement (line 68)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to original_function(...): (line 69)
        # Getting the type of 'args' (line 69)
        args_291841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 38), 'args', False)
        # Processing the call keyword arguments (line 69)
        # Getting the type of 'kwargs' (line 69)
        kwargs_291842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 46), 'kwargs', False)
        kwargs_291843 = {'kwargs_291842': kwargs_291842}
        # Getting the type of 'original_function' (line 69)
        original_function_291840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'original_function', False)
        # Calling original_function(args, kwargs) (line 69)
        original_function_call_result_291844 = invoke(stypy.reporting.localization.Localization(__file__, 69, 19), original_function_291840, *[args_291841], **kwargs_291843)
        
        # Assigning a type to the variable 'stypy_return_type' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'stypy_return_type', original_function_call_result_291844)
        # SSA branch for the else part of an if statement (line 68)
        module_type_store.open_ssa_branch('else')
        
        # Call to IOError(...): (line 71)
        # Processing the call arguments (line 71)
        unicode_291846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 26), 'unicode', u'An attempt was made to connect to the internet by a test that was not marked `remote_data`.')
        # Processing the call keyword arguments (line 71)
        kwargs_291847 = {}
        # Getting the type of 'IOError' (line 71)
        IOError_291845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 18), 'IOError', False)
        # Calling IOError(args, kwargs) (line 71)
        IOError_call_result_291848 = invoke(stypy.reporting.localization.Localization(__file__, 71, 18), IOError_291845, *[unicode_291846], **kwargs_291847)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 71, 12), IOError_call_result_291848, 'raise parameter', BaseException)
        # SSA join for if statement (line 68)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'new_function(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_function' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_291849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_291849)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_function'
        return stypy_return_type_291849

    # Assigning a type to the variable 'new_function' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'new_function', new_function)
    # Getting the type of 'new_function' (line 73)
    new_function_291850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'new_function')
    # Assigning a type to the variable 'stypy_return_type' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'stypy_return_type', new_function_291850)
    
    # ################# End of 'check_internet_off(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_internet_off' in the type store
    # Getting the type of 'stypy_return_type' (line 32)
    stypy_return_type_291851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_291851)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_internet_off'
    return stypy_return_type_291851

# Assigning a type to the variable 'check_internet_off' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'check_internet_off', check_internet_off)

@norecursion
def turn_off_internet(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 76)
    False_291852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 30), 'False')
    defaults = [False_291852]
    # Create a new context for function 'turn_off_internet'
    module_type_store = module_type_store.open_function_context('turn_off_internet', 76, 0, False)
    
    # Passed parameters checking function
    turn_off_internet.stypy_localization = localization
    turn_off_internet.stypy_type_of_self = None
    turn_off_internet.stypy_type_store = module_type_store
    turn_off_internet.stypy_function_name = 'turn_off_internet'
    turn_off_internet.stypy_param_names_list = ['verbose']
    turn_off_internet.stypy_varargs_param_name = None
    turn_off_internet.stypy_kwargs_param_name = None
    turn_off_internet.stypy_call_defaults = defaults
    turn_off_internet.stypy_call_varargs = varargs
    turn_off_internet.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'turn_off_internet', ['verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'turn_off_internet', localization, ['verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'turn_off_internet(...)' code ##################

    unicode_291853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, (-1)), 'unicode', u'\n    Disable internet access via python by preventing connections from being\n    created using the socket module.  Presumably this could be worked around by\n    using some other means of accessing the internet, but all default python\n    modules (urllib, requests, etc.) use socket [citation needed].\n    ')
    # Marking variables as global (line 84)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 84, 4), 'INTERNET_OFF')
    # Marking variables as global (line 85)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 85, 4), '_orig_opener')
    
    # Getting the type of 'INTERNET_OFF' (line 87)
    INTERNET_OFF_291854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 7), 'INTERNET_OFF')
    # Testing the type of an if condition (line 87)
    if_condition_291855 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 4), INTERNET_OFF_291854)
    # Assigning a type to the variable 'if_condition_291855' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'if_condition_291855', if_condition_291855)
    # SSA begins for if statement (line 87)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 87)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 90):
    # Getting the type of 'True' (line 90)
    True_291856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'True')
    # Assigning a type to the variable 'INTERNET_OFF' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'INTERNET_OFF', True_291856)
    
    # Assigning a Name to a Name (line 92):
    # Getting the type of 'True' (line 92)
    True_291857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'True')
    # Assigning a type to the variable '__tracebackhide__' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), '__tracebackhide__', True_291857)
    
    # Getting the type of 'verbose' (line 93)
    verbose_291858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 7), 'verbose')
    # Testing the type of an if condition (line 93)
    if_condition_291859 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 4), verbose_291858)
    # Assigning a type to the variable 'if_condition_291859' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'if_condition_291859', if_condition_291859)
    # SSA begins for if statement (line 93)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 94)
    # Processing the call arguments (line 94)
    unicode_291861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 14), 'unicode', u'Internet access disabled')
    # Processing the call keyword arguments (line 94)
    kwargs_291862 = {}
    # Getting the type of 'print' (line 94)
    print_291860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'print', False)
    # Calling print(args, kwargs) (line 94)
    print_call_result_291863 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), print_291860, *[unicode_291861], **kwargs_291862)
    
    # SSA join for if statement (line 93)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 99):
    
    # Call to build_opener(...): (line 99)
    # Processing the call keyword arguments (line 99)
    kwargs_291867 = {}
    # Getting the type of 'urllib' (line 99)
    urllib_291864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 19), 'urllib', False)
    # Obtaining the member 'request' of a type (line 99)
    request_291865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 19), urllib_291864, 'request')
    # Obtaining the member 'build_opener' of a type (line 99)
    build_opener_291866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 19), request_291865, 'build_opener')
    # Calling build_opener(args, kwargs) (line 99)
    build_opener_call_result_291868 = invoke(stypy.reporting.localization.Localization(__file__, 99, 19), build_opener_291866, *[], **kwargs_291867)
    
    # Assigning a type to the variable '_orig_opener' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), '_orig_opener', build_opener_call_result_291868)
    
    # Assigning a Call to a Name (line 100):
    
    # Call to ProxyHandler(...): (line 100)
    # Processing the call arguments (line 100)
    
    # Obtaining an instance of the builtin type 'dict' (line 100)
    dict_291872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 51), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 100)
    
    # Processing the call keyword arguments (line 100)
    kwargs_291873 = {}
    # Getting the type of 'urllib' (line 100)
    urllib_291869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 23), 'urllib', False)
    # Obtaining the member 'request' of a type (line 100)
    request_291870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 23), urllib_291869, 'request')
    # Obtaining the member 'ProxyHandler' of a type (line 100)
    ProxyHandler_291871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 23), request_291870, 'ProxyHandler')
    # Calling ProxyHandler(args, kwargs) (line 100)
    ProxyHandler_call_result_291874 = invoke(stypy.reporting.localization.Localization(__file__, 100, 23), ProxyHandler_291871, *[dict_291872], **kwargs_291873)
    
    # Assigning a type to the variable 'no_proxy_handler' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'no_proxy_handler', ProxyHandler_call_result_291874)
    
    # Assigning a Call to a Name (line 101):
    
    # Call to build_opener(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'no_proxy_handler' (line 101)
    no_proxy_handler_291878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 41), 'no_proxy_handler', False)
    # Processing the call keyword arguments (line 101)
    kwargs_291879 = {}
    # Getting the type of 'urllib' (line 101)
    urllib_291875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 13), 'urllib', False)
    # Obtaining the member 'request' of a type (line 101)
    request_291876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 13), urllib_291875, 'request')
    # Obtaining the member 'build_opener' of a type (line 101)
    build_opener_291877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 13), request_291876, 'build_opener')
    # Calling build_opener(args, kwargs) (line 101)
    build_opener_call_result_291880 = invoke(stypy.reporting.localization.Localization(__file__, 101, 13), build_opener_291877, *[no_proxy_handler_291878], **kwargs_291879)
    
    # Assigning a type to the variable 'opener' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'opener', build_opener_call_result_291880)
    
    # Call to install_opener(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'opener' (line 102)
    opener_291884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 34), 'opener', False)
    # Processing the call keyword arguments (line 102)
    kwargs_291885 = {}
    # Getting the type of 'urllib' (line 102)
    urllib_291881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'urllib', False)
    # Obtaining the member 'request' of a type (line 102)
    request_291882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 4), urllib_291881, 'request')
    # Obtaining the member 'install_opener' of a type (line 102)
    install_opener_291883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 4), request_291882, 'install_opener')
    # Calling install_opener(args, kwargs) (line 102)
    install_opener_call_result_291886 = invoke(stypy.reporting.localization.Localization(__file__, 102, 4), install_opener_291883, *[opener_291884], **kwargs_291885)
    
    
    # Assigning a Call to a Attribute (line 104):
    
    # Call to check_internet_off(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'socket_create_connection' (line 104)
    socket_create_connection_291888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 50), 'socket_create_connection', False)
    # Processing the call keyword arguments (line 104)
    kwargs_291889 = {}
    # Getting the type of 'check_internet_off' (line 104)
    check_internet_off_291887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 31), 'check_internet_off', False)
    # Calling check_internet_off(args, kwargs) (line 104)
    check_internet_off_call_result_291890 = invoke(stypy.reporting.localization.Localization(__file__, 104, 31), check_internet_off_291887, *[socket_create_connection_291888], **kwargs_291889)
    
    # Getting the type of 'socket' (line 104)
    socket_291891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'socket')
    # Setting the type of the member 'create_connection' of a type (line 104)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 4), socket_291891, 'create_connection', check_internet_off_call_result_291890)
    
    # Assigning a Call to a Attribute (line 105):
    
    # Call to check_internet_off(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'socket_bind' (line 105)
    socket_bind_291893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 44), 'socket_bind', False)
    # Processing the call keyword arguments (line 105)
    kwargs_291894 = {}
    # Getting the type of 'check_internet_off' (line 105)
    check_internet_off_291892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 25), 'check_internet_off', False)
    # Calling check_internet_off(args, kwargs) (line 105)
    check_internet_off_call_result_291895 = invoke(stypy.reporting.localization.Localization(__file__, 105, 25), check_internet_off_291892, *[socket_bind_291893], **kwargs_291894)
    
    # Getting the type of 'socket' (line 105)
    socket_291896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'socket')
    # Obtaining the member 'socket' of a type (line 105)
    socket_291897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 4), socket_291896, 'socket')
    # Setting the type of the member 'bind' of a type (line 105)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 4), socket_291897, 'bind', check_internet_off_call_result_291895)
    
    # Assigning a Call to a Attribute (line 106):
    
    # Call to check_internet_off(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'socket_connect' (line 106)
    socket_connect_291899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 47), 'socket_connect', False)
    # Processing the call keyword arguments (line 106)
    kwargs_291900 = {}
    # Getting the type of 'check_internet_off' (line 106)
    check_internet_off_291898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 28), 'check_internet_off', False)
    # Calling check_internet_off(args, kwargs) (line 106)
    check_internet_off_call_result_291901 = invoke(stypy.reporting.localization.Localization(__file__, 106, 28), check_internet_off_291898, *[socket_connect_291899], **kwargs_291900)
    
    # Getting the type of 'socket' (line 106)
    socket_291902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'socket')
    # Obtaining the member 'socket' of a type (line 106)
    socket_291903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 4), socket_291902, 'socket')
    # Setting the type of the member 'connect' of a type (line 106)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 4), socket_291903, 'connect', check_internet_off_call_result_291901)
    # Getting the type of 'socket' (line 108)
    socket_291904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 11), 'socket')
    # Assigning a type to the variable 'stypy_return_type' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type', socket_291904)
    
    # ################# End of 'turn_off_internet(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'turn_off_internet' in the type store
    # Getting the type of 'stypy_return_type' (line 76)
    stypy_return_type_291905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_291905)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'turn_off_internet'
    return stypy_return_type_291905

# Assigning a type to the variable 'turn_off_internet' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'turn_off_internet', turn_off_internet)

@norecursion
def turn_on_internet(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 111)
    False_291906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 29), 'False')
    defaults = [False_291906]
    # Create a new context for function 'turn_on_internet'
    module_type_store = module_type_store.open_function_context('turn_on_internet', 111, 0, False)
    
    # Passed parameters checking function
    turn_on_internet.stypy_localization = localization
    turn_on_internet.stypy_type_of_self = None
    turn_on_internet.stypy_type_store = module_type_store
    turn_on_internet.stypy_function_name = 'turn_on_internet'
    turn_on_internet.stypy_param_names_list = ['verbose']
    turn_on_internet.stypy_varargs_param_name = None
    turn_on_internet.stypy_kwargs_param_name = None
    turn_on_internet.stypy_call_defaults = defaults
    turn_on_internet.stypy_call_varargs = varargs
    turn_on_internet.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'turn_on_internet', ['verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'turn_on_internet', localization, ['verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'turn_on_internet(...)' code ##################

    unicode_291907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, (-1)), 'unicode', u'\n    Restore internet access.  Not used, but kept in case it is needed.\n    ')
    # Marking variables as global (line 116)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 116, 4), 'INTERNET_OFF')
    # Marking variables as global (line 117)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 117, 4), '_orig_opener')
    
    
    # Getting the type of 'INTERNET_OFF' (line 119)
    INTERNET_OFF_291908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'INTERNET_OFF')
    # Applying the 'not' unary operator (line 119)
    result_not__291909 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 7), 'not', INTERNET_OFF_291908)
    
    # Testing the type of an if condition (line 119)
    if_condition_291910 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 4), result_not__291909)
    # Assigning a type to the variable 'if_condition_291910' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'if_condition_291910', if_condition_291910)
    # SSA begins for if statement (line 119)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 119)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 122):
    # Getting the type of 'False' (line 122)
    False_291911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 19), 'False')
    # Assigning a type to the variable 'INTERNET_OFF' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'INTERNET_OFF', False_291911)
    
    # Getting the type of 'verbose' (line 124)
    verbose_291912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 7), 'verbose')
    # Testing the type of an if condition (line 124)
    if_condition_291913 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 4), verbose_291912)
    # Assigning a type to the variable 'if_condition_291913' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'if_condition_291913', if_condition_291913)
    # SSA begins for if statement (line 124)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 125)
    # Processing the call arguments (line 125)
    unicode_291915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 14), 'unicode', u'Internet access enabled')
    # Processing the call keyword arguments (line 125)
    kwargs_291916 = {}
    # Getting the type of 'print' (line 125)
    print_291914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'print', False)
    # Calling print(args, kwargs) (line 125)
    print_call_result_291917 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), print_291914, *[unicode_291915], **kwargs_291916)
    
    # SSA join for if statement (line 124)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to install_opener(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of '_orig_opener' (line 127)
    _orig_opener_291921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 34), '_orig_opener', False)
    # Processing the call keyword arguments (line 127)
    kwargs_291922 = {}
    # Getting the type of 'urllib' (line 127)
    urllib_291918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'urllib', False)
    # Obtaining the member 'request' of a type (line 127)
    request_291919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 4), urllib_291918, 'request')
    # Obtaining the member 'install_opener' of a type (line 127)
    install_opener_291920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 4), request_291919, 'install_opener')
    # Calling install_opener(args, kwargs) (line 127)
    install_opener_call_result_291923 = invoke(stypy.reporting.localization.Localization(__file__, 127, 4), install_opener_291920, *[_orig_opener_291921], **kwargs_291922)
    
    
    # Assigning a Name to a Attribute (line 129):
    # Getting the type of 'socket_create_connection' (line 129)
    socket_create_connection_291924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 31), 'socket_create_connection')
    # Getting the type of 'socket' (line 129)
    socket_291925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'socket')
    # Setting the type of the member 'create_connection' of a type (line 129)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 4), socket_291925, 'create_connection', socket_create_connection_291924)
    
    # Assigning a Name to a Attribute (line 130):
    # Getting the type of 'socket_bind' (line 130)
    socket_bind_291926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'socket_bind')
    # Getting the type of 'socket' (line 130)
    socket_291927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'socket')
    # Obtaining the member 'socket' of a type (line 130)
    socket_291928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 4), socket_291927, 'socket')
    # Setting the type of the member 'bind' of a type (line 130)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 4), socket_291928, 'bind', socket_bind_291926)
    
    # Assigning a Name to a Attribute (line 131):
    # Getting the type of 'socket_connect' (line 131)
    socket_connect_291929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 28), 'socket_connect')
    # Getting the type of 'socket' (line 131)
    socket_291930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'socket')
    # Obtaining the member 'socket' of a type (line 131)
    socket_291931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 4), socket_291930, 'socket')
    # Setting the type of the member 'connect' of a type (line 131)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 4), socket_291931, 'connect', socket_connect_291929)
    # Getting the type of 'socket' (line 132)
    socket_291932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 11), 'socket')
    # Assigning a type to the variable 'stypy_return_type' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type', socket_291932)
    
    # ################# End of 'turn_on_internet(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'turn_on_internet' in the type store
    # Getting the type of 'stypy_return_type' (line 111)
    stypy_return_type_291933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_291933)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'turn_on_internet'
    return stypy_return_type_291933

# Assigning a type to the variable 'turn_on_internet' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'turn_on_internet', turn_on_internet)

@norecursion
def no_internet(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 136)
    False_291934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'False')
    defaults = [False_291934]
    # Create a new context for function 'no_internet'
    module_type_store = module_type_store.open_function_context('no_internet', 135, 0, False)
    
    # Passed parameters checking function
    no_internet.stypy_localization = localization
    no_internet.stypy_type_of_self = None
    no_internet.stypy_type_store = module_type_store
    no_internet.stypy_function_name = 'no_internet'
    no_internet.stypy_param_names_list = ['verbose']
    no_internet.stypy_varargs_param_name = None
    no_internet.stypy_kwargs_param_name = None
    no_internet.stypy_call_defaults = defaults
    no_internet.stypy_call_varargs = varargs
    no_internet.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'no_internet', ['verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'no_internet', localization, ['verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'no_internet(...)' code ##################

    unicode_291935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, (-1)), 'unicode', u'Context manager to temporarily disable internet access (if not already\n    disabled).  If it was already disabled before entering the context manager\n    (i.e. `turn_off_internet` was called previously) then this is a no-op and\n    leaves internet access disabled until a manual call to `turn_on_internet`.\n    ')
    
    # Assigning a Name to a Name (line 143):
    # Getting the type of 'INTERNET_OFF' (line 143)
    INTERNET_OFF_291936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'INTERNET_OFF')
    # Assigning a type to the variable 'already_disabled' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'already_disabled', INTERNET_OFF_291936)
    
    # Call to turn_off_internet(...): (line 145)
    # Processing the call keyword arguments (line 145)
    # Getting the type of 'verbose' (line 145)
    verbose_291938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 30), 'verbose', False)
    keyword_291939 = verbose_291938
    kwargs_291940 = {'verbose': keyword_291939}
    # Getting the type of 'turn_off_internet' (line 145)
    turn_off_internet_291937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'turn_off_internet', False)
    # Calling turn_off_internet(args, kwargs) (line 145)
    turn_off_internet_call_result_291941 = invoke(stypy.reporting.localization.Localization(__file__, 145, 4), turn_off_internet_291937, *[], **kwargs_291940)
    
    
    # Try-finally block (line 146)
    # Creating a generator
    GeneratorType_291942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 8), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 8), GeneratorType_291942, None)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'stypy_return_type', GeneratorType_291942)
    
    # finally branch of the try-finally block (line 146)
    
    
    # Getting the type of 'already_disabled' (line 149)
    already_disabled_291943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'already_disabled')
    # Applying the 'not' unary operator (line 149)
    result_not__291944 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 11), 'not', already_disabled_291943)
    
    # Testing the type of an if condition (line 149)
    if_condition_291945 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 8), result_not__291944)
    # Assigning a type to the variable 'if_condition_291945' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'if_condition_291945', if_condition_291945)
    # SSA begins for if statement (line 149)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to turn_on_internet(...): (line 150)
    # Processing the call keyword arguments (line 150)
    # Getting the type of 'verbose' (line 150)
    verbose_291947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 37), 'verbose', False)
    keyword_291948 = verbose_291947
    kwargs_291949 = {'verbose': keyword_291948}
    # Getting the type of 'turn_on_internet' (line 150)
    turn_on_internet_291946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'turn_on_internet', False)
    # Calling turn_on_internet(args, kwargs) (line 150)
    turn_on_internet_call_result_291950 = invoke(stypy.reporting.localization.Localization(__file__, 150, 12), turn_on_internet_291946, *[], **kwargs_291949)
    
    # SSA join for if statement (line 149)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # ################# End of 'no_internet(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'no_internet' in the type store
    # Getting the type of 'stypy_return_type' (line 135)
    stypy_return_type_291951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_291951)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'no_internet'
    return stypy_return_type_291951

# Assigning a type to the variable 'no_internet' (line 135)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'no_internet', no_internet)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
