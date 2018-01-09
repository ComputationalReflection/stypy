
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.pypirc
2: 
3: Provides the PyPIRCCommand class, the base class for the command classes
4: that uses .pypirc in the distutils.command package.
5: '''
6: import os
7: from ConfigParser import ConfigParser
8: 
9: from distutils.cmd import Command
10: 
11: DEFAULT_PYPIRC = '''\
12: [distutils]
13: index-servers =
14:     pypi
15: 
16: [pypi]
17: username:%s
18: password:%s
19: '''
20: 
21: class PyPIRCCommand(Command):
22:     '''Base command that knows how to handle the .pypirc file
23:     '''
24:     DEFAULT_REPOSITORY = 'https://upload.pypi.org/legacy/'
25:     DEFAULT_REALM = 'pypi'
26:     repository = None
27:     realm = None
28: 
29:     user_options = [
30:         ('repository=', 'r',
31:          "url of repository [default: %s]" % \
32:             DEFAULT_REPOSITORY),
33:         ('show-response', None,
34:          'display full response text from server')]
35: 
36:     boolean_options = ['show-response']
37: 
38:     def _get_rc_file(self):
39:         '''Returns rc file path.'''
40:         return os.path.join(os.path.expanduser('~'), '.pypirc')
41: 
42:     def _store_pypirc(self, username, password):
43:         '''Creates a default .pypirc file.'''
44:         rc = self._get_rc_file()
45:         f = os.fdopen(os.open(rc, os.O_CREAT | os.O_WRONLY, 0600), 'w')
46:         try:
47:             f.write(DEFAULT_PYPIRC % (username, password))
48:         finally:
49:             f.close()
50: 
51:     def _read_pypirc(self):
52:         '''Reads the .pypirc file.'''
53:         rc = self._get_rc_file()
54:         if os.path.exists(rc):
55:             self.announce('Using PyPI login from %s' % rc)
56:             repository = self.repository or self.DEFAULT_REPOSITORY
57:             config = ConfigParser()
58:             config.read(rc)
59:             sections = config.sections()
60:             if 'distutils' in sections:
61:                 # let's get the list of servers
62:                 index_servers = config.get('distutils', 'index-servers')
63:                 _servers = [server.strip() for server in
64:                             index_servers.split('\n')
65:                             if server.strip() != '']
66:                 if _servers == []:
67:                     # nothing set, let's try to get the default pypi
68:                     if 'pypi' in sections:
69:                         _servers = ['pypi']
70:                     else:
71:                         # the file is not properly defined, returning
72:                         # an empty dict
73:                         return {}
74:                 for server in _servers:
75:                     current = {'server': server}
76:                     current['username'] = config.get(server, 'username')
77: 
78:                     # optional params
79:                     for key, default in (('repository',
80:                                           self.DEFAULT_REPOSITORY),
81:                                          ('realm', self.DEFAULT_REALM),
82:                                          ('password', None)):
83:                         if config.has_option(server, key):
84:                             current[key] = config.get(server, key)
85:                         else:
86:                             current[key] = default
87:                     if (current['server'] == repository or
88:                         current['repository'] == repository):
89:                         return current
90:             elif 'server-login' in sections:
91:                 # old format
92:                 server = 'server-login'
93:                 if config.has_option(server, 'repository'):
94:                     repository = config.get(server, 'repository')
95:                 else:
96:                     repository = self.DEFAULT_REPOSITORY
97:                 return {'username': config.get(server, 'username'),
98:                         'password': config.get(server, 'password'),
99:                         'repository': repository,
100:                         'server': server,
101:                         'realm': self.DEFAULT_REALM}
102: 
103:         return {}
104: 
105:     def initialize_options(self):
106:         '''Initialize options.'''
107:         self.repository = None
108:         self.realm = None
109:         self.show_response = 0
110: 
111:     def finalize_options(self):
112:         '''Finalizes options.'''
113:         if self.repository is None:
114:             self.repository = self.DEFAULT_REPOSITORY
115:         if self.realm is None:
116:             self.realm = self.DEFAULT_REALM
117: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_306136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', 'distutils.pypirc\n\nProvides the PyPIRCCommand class, the base class for the command classes\nthat uses .pypirc in the distutils.command package.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import os' statement (line 6)
import os

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from ConfigParser import ConfigParser' statement (line 7)
try:
    from ConfigParser import ConfigParser

except:
    ConfigParser = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'ConfigParser', None, module_type_store, ['ConfigParser'], [ConfigParser])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils.cmd import Command' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_306137 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.cmd')

if (type(import_306137) is not StypyTypeError):

    if (import_306137 != 'pyd_module'):
        __import__(import_306137)
        sys_modules_306138 = sys.modules[import_306137]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.cmd', sys_modules_306138.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_306138, sys_modules_306138.module_type_store, module_type_store)
    else:
        from distutils.cmd import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.cmd', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.cmd' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.cmd', import_306137)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')


# Assigning a Str to a Name (line 11):
str_306139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'str', '[distutils]\nindex-servers =\n    pypi\n\n[pypi]\nusername:%s\npassword:%s\n')
# Assigning a type to the variable 'DEFAULT_PYPIRC' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'DEFAULT_PYPIRC', str_306139)
# Declaration of the 'PyPIRCCommand' class
# Getting the type of 'Command' (line 21)
Command_306140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'Command')

class PyPIRCCommand(Command_306140, ):
    str_306141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, (-1)), 'str', 'Base command that knows how to handle the .pypirc file\n    ')

    @norecursion
    def _get_rc_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_rc_file'
        module_type_store = module_type_store.open_function_context('_get_rc_file', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PyPIRCCommand._get_rc_file.__dict__.__setitem__('stypy_localization', localization)
        PyPIRCCommand._get_rc_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PyPIRCCommand._get_rc_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        PyPIRCCommand._get_rc_file.__dict__.__setitem__('stypy_function_name', 'PyPIRCCommand._get_rc_file')
        PyPIRCCommand._get_rc_file.__dict__.__setitem__('stypy_param_names_list', [])
        PyPIRCCommand._get_rc_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        PyPIRCCommand._get_rc_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PyPIRCCommand._get_rc_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        PyPIRCCommand._get_rc_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        PyPIRCCommand._get_rc_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PyPIRCCommand._get_rc_file.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PyPIRCCommand._get_rc_file', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_rc_file', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_rc_file(...)' code ##################

        str_306142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 8), 'str', 'Returns rc file path.')
        
        # Call to join(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Call to expanduser(...): (line 40)
        # Processing the call arguments (line 40)
        str_306149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 47), 'str', '~')
        # Processing the call keyword arguments (line 40)
        kwargs_306150 = {}
        # Getting the type of 'os' (line 40)
        os_306146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 40)
        path_306147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 28), os_306146, 'path')
        # Obtaining the member 'expanduser' of a type (line 40)
        expanduser_306148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 28), path_306147, 'expanduser')
        # Calling expanduser(args, kwargs) (line 40)
        expanduser_call_result_306151 = invoke(stypy.reporting.localization.Localization(__file__, 40, 28), expanduser_306148, *[str_306149], **kwargs_306150)
        
        str_306152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 53), 'str', '.pypirc')
        # Processing the call keyword arguments (line 40)
        kwargs_306153 = {}
        # Getting the type of 'os' (line 40)
        os_306143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 40)
        path_306144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 15), os_306143, 'path')
        # Obtaining the member 'join' of a type (line 40)
        join_306145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 15), path_306144, 'join')
        # Calling join(args, kwargs) (line 40)
        join_call_result_306154 = invoke(stypy.reporting.localization.Localization(__file__, 40, 15), join_306145, *[expanduser_call_result_306151, str_306152], **kwargs_306153)
        
        # Assigning a type to the variable 'stypy_return_type' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'stypy_return_type', join_call_result_306154)
        
        # ################# End of '_get_rc_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_rc_file' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_306155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_306155)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_rc_file'
        return stypy_return_type_306155


    @norecursion
    def _store_pypirc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_store_pypirc'
        module_type_store = module_type_store.open_function_context('_store_pypirc', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PyPIRCCommand._store_pypirc.__dict__.__setitem__('stypy_localization', localization)
        PyPIRCCommand._store_pypirc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PyPIRCCommand._store_pypirc.__dict__.__setitem__('stypy_type_store', module_type_store)
        PyPIRCCommand._store_pypirc.__dict__.__setitem__('stypy_function_name', 'PyPIRCCommand._store_pypirc')
        PyPIRCCommand._store_pypirc.__dict__.__setitem__('stypy_param_names_list', ['username', 'password'])
        PyPIRCCommand._store_pypirc.__dict__.__setitem__('stypy_varargs_param_name', None)
        PyPIRCCommand._store_pypirc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PyPIRCCommand._store_pypirc.__dict__.__setitem__('stypy_call_defaults', defaults)
        PyPIRCCommand._store_pypirc.__dict__.__setitem__('stypy_call_varargs', varargs)
        PyPIRCCommand._store_pypirc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PyPIRCCommand._store_pypirc.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PyPIRCCommand._store_pypirc', ['username', 'password'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_store_pypirc', localization, ['username', 'password'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_store_pypirc(...)' code ##################

        str_306156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 8), 'str', 'Creates a default .pypirc file.')
        
        # Assigning a Call to a Name (line 44):
        
        # Call to _get_rc_file(...): (line 44)
        # Processing the call keyword arguments (line 44)
        kwargs_306159 = {}
        # Getting the type of 'self' (line 44)
        self_306157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'self', False)
        # Obtaining the member '_get_rc_file' of a type (line 44)
        _get_rc_file_306158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 13), self_306157, '_get_rc_file')
        # Calling _get_rc_file(args, kwargs) (line 44)
        _get_rc_file_call_result_306160 = invoke(stypy.reporting.localization.Localization(__file__, 44, 13), _get_rc_file_306158, *[], **kwargs_306159)
        
        # Assigning a type to the variable 'rc' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'rc', _get_rc_file_call_result_306160)
        
        # Assigning a Call to a Name (line 45):
        
        # Call to fdopen(...): (line 45)
        # Processing the call arguments (line 45)
        
        # Call to open(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'rc' (line 45)
        rc_306165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 30), 'rc', False)
        # Getting the type of 'os' (line 45)
        os_306166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 34), 'os', False)
        # Obtaining the member 'O_CREAT' of a type (line 45)
        O_CREAT_306167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 34), os_306166, 'O_CREAT')
        # Getting the type of 'os' (line 45)
        os_306168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 47), 'os', False)
        # Obtaining the member 'O_WRONLY' of a type (line 45)
        O_WRONLY_306169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 47), os_306168, 'O_WRONLY')
        # Applying the binary operator '|' (line 45)
        result_or__306170 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 34), '|', O_CREAT_306167, O_WRONLY_306169)
        
        int_306171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 60), 'int')
        # Processing the call keyword arguments (line 45)
        kwargs_306172 = {}
        # Getting the type of 'os' (line 45)
        os_306163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 22), 'os', False)
        # Obtaining the member 'open' of a type (line 45)
        open_306164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 22), os_306163, 'open')
        # Calling open(args, kwargs) (line 45)
        open_call_result_306173 = invoke(stypy.reporting.localization.Localization(__file__, 45, 22), open_306164, *[rc_306165, result_or__306170, int_306171], **kwargs_306172)
        
        str_306174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 67), 'str', 'w')
        # Processing the call keyword arguments (line 45)
        kwargs_306175 = {}
        # Getting the type of 'os' (line 45)
        os_306161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'os', False)
        # Obtaining the member 'fdopen' of a type (line 45)
        fdopen_306162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), os_306161, 'fdopen')
        # Calling fdopen(args, kwargs) (line 45)
        fdopen_call_result_306176 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), fdopen_306162, *[open_call_result_306173, str_306174], **kwargs_306175)
        
        # Assigning a type to the variable 'f' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'f', fdopen_call_result_306176)
        
        # Try-finally block (line 46)
        
        # Call to write(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'DEFAULT_PYPIRC' (line 47)
        DEFAULT_PYPIRC_306179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 20), 'DEFAULT_PYPIRC', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 47)
        tuple_306180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 47)
        # Adding element type (line 47)
        # Getting the type of 'username' (line 47)
        username_306181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 38), 'username', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 38), tuple_306180, username_306181)
        # Adding element type (line 47)
        # Getting the type of 'password' (line 47)
        password_306182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 48), 'password', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 38), tuple_306180, password_306182)
        
        # Applying the binary operator '%' (line 47)
        result_mod_306183 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 20), '%', DEFAULT_PYPIRC_306179, tuple_306180)
        
        # Processing the call keyword arguments (line 47)
        kwargs_306184 = {}
        # Getting the type of 'f' (line 47)
        f_306177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 47)
        write_306178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), f_306177, 'write')
        # Calling write(args, kwargs) (line 47)
        write_call_result_306185 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), write_306178, *[result_mod_306183], **kwargs_306184)
        
        
        # finally branch of the try-finally block (line 46)
        
        # Call to close(...): (line 49)
        # Processing the call keyword arguments (line 49)
        kwargs_306188 = {}
        # Getting the type of 'f' (line 49)
        f_306186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 49)
        close_306187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), f_306186, 'close')
        # Calling close(args, kwargs) (line 49)
        close_call_result_306189 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), close_306187, *[], **kwargs_306188)
        
        
        
        # ################# End of '_store_pypirc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_store_pypirc' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_306190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_306190)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_store_pypirc'
        return stypy_return_type_306190


    @norecursion
    def _read_pypirc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_read_pypirc'
        module_type_store = module_type_store.open_function_context('_read_pypirc', 51, 4, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PyPIRCCommand._read_pypirc.__dict__.__setitem__('stypy_localization', localization)
        PyPIRCCommand._read_pypirc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PyPIRCCommand._read_pypirc.__dict__.__setitem__('stypy_type_store', module_type_store)
        PyPIRCCommand._read_pypirc.__dict__.__setitem__('stypy_function_name', 'PyPIRCCommand._read_pypirc')
        PyPIRCCommand._read_pypirc.__dict__.__setitem__('stypy_param_names_list', [])
        PyPIRCCommand._read_pypirc.__dict__.__setitem__('stypy_varargs_param_name', None)
        PyPIRCCommand._read_pypirc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PyPIRCCommand._read_pypirc.__dict__.__setitem__('stypy_call_defaults', defaults)
        PyPIRCCommand._read_pypirc.__dict__.__setitem__('stypy_call_varargs', varargs)
        PyPIRCCommand._read_pypirc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PyPIRCCommand._read_pypirc.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PyPIRCCommand._read_pypirc', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_read_pypirc', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_read_pypirc(...)' code ##################

        str_306191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 8), 'str', 'Reads the .pypirc file.')
        
        # Assigning a Call to a Name (line 53):
        
        # Call to _get_rc_file(...): (line 53)
        # Processing the call keyword arguments (line 53)
        kwargs_306194 = {}
        # Getting the type of 'self' (line 53)
        self_306192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 13), 'self', False)
        # Obtaining the member '_get_rc_file' of a type (line 53)
        _get_rc_file_306193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 13), self_306192, '_get_rc_file')
        # Calling _get_rc_file(args, kwargs) (line 53)
        _get_rc_file_call_result_306195 = invoke(stypy.reporting.localization.Localization(__file__, 53, 13), _get_rc_file_306193, *[], **kwargs_306194)
        
        # Assigning a type to the variable 'rc' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'rc', _get_rc_file_call_result_306195)
        
        
        # Call to exists(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'rc' (line 54)
        rc_306199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 26), 'rc', False)
        # Processing the call keyword arguments (line 54)
        kwargs_306200 = {}
        # Getting the type of 'os' (line 54)
        os_306196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'os', False)
        # Obtaining the member 'path' of a type (line 54)
        path_306197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 11), os_306196, 'path')
        # Obtaining the member 'exists' of a type (line 54)
        exists_306198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 11), path_306197, 'exists')
        # Calling exists(args, kwargs) (line 54)
        exists_call_result_306201 = invoke(stypy.reporting.localization.Localization(__file__, 54, 11), exists_306198, *[rc_306199], **kwargs_306200)
        
        # Testing the type of an if condition (line 54)
        if_condition_306202 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 8), exists_call_result_306201)
        # Assigning a type to the variable 'if_condition_306202' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'if_condition_306202', if_condition_306202)
        # SSA begins for if statement (line 54)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to announce(...): (line 55)
        # Processing the call arguments (line 55)
        str_306205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 26), 'str', 'Using PyPI login from %s')
        # Getting the type of 'rc' (line 55)
        rc_306206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 55), 'rc', False)
        # Applying the binary operator '%' (line 55)
        result_mod_306207 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 26), '%', str_306205, rc_306206)
        
        # Processing the call keyword arguments (line 55)
        kwargs_306208 = {}
        # Getting the type of 'self' (line 55)
        self_306203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'self', False)
        # Obtaining the member 'announce' of a type (line 55)
        announce_306204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 12), self_306203, 'announce')
        # Calling announce(args, kwargs) (line 55)
        announce_call_result_306209 = invoke(stypy.reporting.localization.Localization(__file__, 55, 12), announce_306204, *[result_mod_306207], **kwargs_306208)
        
        
        # Assigning a BoolOp to a Name (line 56):
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 56)
        self_306210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 25), 'self')
        # Obtaining the member 'repository' of a type (line 56)
        repository_306211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 25), self_306210, 'repository')
        # Getting the type of 'self' (line 56)
        self_306212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 44), 'self')
        # Obtaining the member 'DEFAULT_REPOSITORY' of a type (line 56)
        DEFAULT_REPOSITORY_306213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 44), self_306212, 'DEFAULT_REPOSITORY')
        # Applying the binary operator 'or' (line 56)
        result_or_keyword_306214 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 25), 'or', repository_306211, DEFAULT_REPOSITORY_306213)
        
        # Assigning a type to the variable 'repository' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'repository', result_or_keyword_306214)
        
        # Assigning a Call to a Name (line 57):
        
        # Call to ConfigParser(...): (line 57)
        # Processing the call keyword arguments (line 57)
        kwargs_306216 = {}
        # Getting the type of 'ConfigParser' (line 57)
        ConfigParser_306215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 21), 'ConfigParser', False)
        # Calling ConfigParser(args, kwargs) (line 57)
        ConfigParser_call_result_306217 = invoke(stypy.reporting.localization.Localization(__file__, 57, 21), ConfigParser_306215, *[], **kwargs_306216)
        
        # Assigning a type to the variable 'config' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'config', ConfigParser_call_result_306217)
        
        # Call to read(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'rc' (line 58)
        rc_306220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 24), 'rc', False)
        # Processing the call keyword arguments (line 58)
        kwargs_306221 = {}
        # Getting the type of 'config' (line 58)
        config_306218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'config', False)
        # Obtaining the member 'read' of a type (line 58)
        read_306219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 12), config_306218, 'read')
        # Calling read(args, kwargs) (line 58)
        read_call_result_306222 = invoke(stypy.reporting.localization.Localization(__file__, 58, 12), read_306219, *[rc_306220], **kwargs_306221)
        
        
        # Assigning a Call to a Name (line 59):
        
        # Call to sections(...): (line 59)
        # Processing the call keyword arguments (line 59)
        kwargs_306225 = {}
        # Getting the type of 'config' (line 59)
        config_306223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 23), 'config', False)
        # Obtaining the member 'sections' of a type (line 59)
        sections_306224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 23), config_306223, 'sections')
        # Calling sections(args, kwargs) (line 59)
        sections_call_result_306226 = invoke(stypy.reporting.localization.Localization(__file__, 59, 23), sections_306224, *[], **kwargs_306225)
        
        # Assigning a type to the variable 'sections' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'sections', sections_call_result_306226)
        
        
        str_306227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 15), 'str', 'distutils')
        # Getting the type of 'sections' (line 60)
        sections_306228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 30), 'sections')
        # Applying the binary operator 'in' (line 60)
        result_contains_306229 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 15), 'in', str_306227, sections_306228)
        
        # Testing the type of an if condition (line 60)
        if_condition_306230 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 12), result_contains_306229)
        # Assigning a type to the variable 'if_condition_306230' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'if_condition_306230', if_condition_306230)
        # SSA begins for if statement (line 60)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 62):
        
        # Call to get(...): (line 62)
        # Processing the call arguments (line 62)
        str_306233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 43), 'str', 'distutils')
        str_306234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 56), 'str', 'index-servers')
        # Processing the call keyword arguments (line 62)
        kwargs_306235 = {}
        # Getting the type of 'config' (line 62)
        config_306231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 32), 'config', False)
        # Obtaining the member 'get' of a type (line 62)
        get_306232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 32), config_306231, 'get')
        # Calling get(args, kwargs) (line 62)
        get_call_result_306236 = invoke(stypy.reporting.localization.Localization(__file__, 62, 32), get_306232, *[str_306233, str_306234], **kwargs_306235)
        
        # Assigning a type to the variable 'index_servers' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'index_servers', get_call_result_306236)
        
        # Assigning a ListComp to a Name (line 63):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to split(...): (line 64)
        # Processing the call arguments (line 64)
        str_306249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 48), 'str', '\n')
        # Processing the call keyword arguments (line 64)
        kwargs_306250 = {}
        # Getting the type of 'index_servers' (line 64)
        index_servers_306247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 28), 'index_servers', False)
        # Obtaining the member 'split' of a type (line 64)
        split_306248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 28), index_servers_306247, 'split')
        # Calling split(args, kwargs) (line 64)
        split_call_result_306251 = invoke(stypy.reporting.localization.Localization(__file__, 64, 28), split_306248, *[str_306249], **kwargs_306250)
        
        comprehension_306252 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 28), split_call_result_306251)
        # Assigning a type to the variable 'server' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 28), 'server', comprehension_306252)
        
        
        # Call to strip(...): (line 65)
        # Processing the call keyword arguments (line 65)
        kwargs_306243 = {}
        # Getting the type of 'server' (line 65)
        server_306241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 31), 'server', False)
        # Obtaining the member 'strip' of a type (line 65)
        strip_306242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 31), server_306241, 'strip')
        # Calling strip(args, kwargs) (line 65)
        strip_call_result_306244 = invoke(stypy.reporting.localization.Localization(__file__, 65, 31), strip_306242, *[], **kwargs_306243)
        
        str_306245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 49), 'str', '')
        # Applying the binary operator '!=' (line 65)
        result_ne_306246 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 31), '!=', strip_call_result_306244, str_306245)
        
        
        # Call to strip(...): (line 63)
        # Processing the call keyword arguments (line 63)
        kwargs_306239 = {}
        # Getting the type of 'server' (line 63)
        server_306237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 28), 'server', False)
        # Obtaining the member 'strip' of a type (line 63)
        strip_306238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 28), server_306237, 'strip')
        # Calling strip(args, kwargs) (line 63)
        strip_call_result_306240 = invoke(stypy.reporting.localization.Localization(__file__, 63, 28), strip_306238, *[], **kwargs_306239)
        
        list_306253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 28), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 28), list_306253, strip_call_result_306240)
        # Assigning a type to the variable '_servers' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), '_servers', list_306253)
        
        
        # Getting the type of '_servers' (line 66)
        _servers_306254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 19), '_servers')
        
        # Obtaining an instance of the builtin type 'list' (line 66)
        list_306255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 66)
        
        # Applying the binary operator '==' (line 66)
        result_eq_306256 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 19), '==', _servers_306254, list_306255)
        
        # Testing the type of an if condition (line 66)
        if_condition_306257 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 16), result_eq_306256)
        # Assigning a type to the variable 'if_condition_306257' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'if_condition_306257', if_condition_306257)
        # SSA begins for if statement (line 66)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        str_306258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 23), 'str', 'pypi')
        # Getting the type of 'sections' (line 68)
        sections_306259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 33), 'sections')
        # Applying the binary operator 'in' (line 68)
        result_contains_306260 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 23), 'in', str_306258, sections_306259)
        
        # Testing the type of an if condition (line 68)
        if_condition_306261 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 20), result_contains_306260)
        # Assigning a type to the variable 'if_condition_306261' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 20), 'if_condition_306261', if_condition_306261)
        # SSA begins for if statement (line 68)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 69):
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_306262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        str_306263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 36), 'str', 'pypi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 35), list_306262, str_306263)
        
        # Assigning a type to the variable '_servers' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 24), '_servers', list_306262)
        # SSA branch for the else part of an if statement (line 68)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining an instance of the builtin type 'dict' (line 73)
        dict_306264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 31), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 73)
        
        # Assigning a type to the variable 'stypy_return_type' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 24), 'stypy_return_type', dict_306264)
        # SSA join for if statement (line 68)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 66)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of '_servers' (line 74)
        _servers_306265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 30), '_servers')
        # Testing the type of a for loop iterable (line 74)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 74, 16), _servers_306265)
        # Getting the type of the for loop variable (line 74)
        for_loop_var_306266 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 74, 16), _servers_306265)
        # Assigning a type to the variable 'server' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'server', for_loop_var_306266)
        # SSA begins for a for statement (line 74)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Dict to a Name (line 75):
        
        # Obtaining an instance of the builtin type 'dict' (line 75)
        dict_306267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 30), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 75)
        # Adding element type (key, value) (line 75)
        str_306268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 31), 'str', 'server')
        # Getting the type of 'server' (line 75)
        server_306269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 41), 'server')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 30), dict_306267, (str_306268, server_306269))
        
        # Assigning a type to the variable 'current' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'current', dict_306267)
        
        # Assigning a Call to a Subscript (line 76):
        
        # Call to get(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'server' (line 76)
        server_306272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 53), 'server', False)
        str_306273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 61), 'str', 'username')
        # Processing the call keyword arguments (line 76)
        kwargs_306274 = {}
        # Getting the type of 'config' (line 76)
        config_306270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 42), 'config', False)
        # Obtaining the member 'get' of a type (line 76)
        get_306271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 42), config_306270, 'get')
        # Calling get(args, kwargs) (line 76)
        get_call_result_306275 = invoke(stypy.reporting.localization.Localization(__file__, 76, 42), get_306271, *[server_306272, str_306273], **kwargs_306274)
        
        # Getting the type of 'current' (line 76)
        current_306276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), 'current')
        str_306277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 28), 'str', 'username')
        # Storing an element on a container (line 76)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 20), current_306276, (str_306277, get_call_result_306275))
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 79)
        tuple_306278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 79)
        # Adding element type (line 79)
        
        # Obtaining an instance of the builtin type 'tuple' (line 79)
        tuple_306279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 79)
        # Adding element type (line 79)
        str_306280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 42), 'str', 'repository')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 42), tuple_306279, str_306280)
        # Adding element type (line 79)
        # Getting the type of 'self' (line 80)
        self_306281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 42), 'self')
        # Obtaining the member 'DEFAULT_REPOSITORY' of a type (line 80)
        DEFAULT_REPOSITORY_306282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 42), self_306281, 'DEFAULT_REPOSITORY')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 42), tuple_306279, DEFAULT_REPOSITORY_306282)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 41), tuple_306278, tuple_306279)
        # Adding element type (line 79)
        
        # Obtaining an instance of the builtin type 'tuple' (line 81)
        tuple_306283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 81)
        # Adding element type (line 81)
        str_306284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 42), 'str', 'realm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 42), tuple_306283, str_306284)
        # Adding element type (line 81)
        # Getting the type of 'self' (line 81)
        self_306285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 51), 'self')
        # Obtaining the member 'DEFAULT_REALM' of a type (line 81)
        DEFAULT_REALM_306286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 51), self_306285, 'DEFAULT_REALM')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 42), tuple_306283, DEFAULT_REALM_306286)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 41), tuple_306278, tuple_306283)
        # Adding element type (line 79)
        
        # Obtaining an instance of the builtin type 'tuple' (line 82)
        tuple_306287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 82)
        # Adding element type (line 82)
        str_306288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 42), 'str', 'password')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 42), tuple_306287, str_306288)
        # Adding element type (line 82)
        # Getting the type of 'None' (line 82)
        None_306289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 54), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 42), tuple_306287, None_306289)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 41), tuple_306278, tuple_306287)
        
        # Testing the type of a for loop iterable (line 79)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 79, 20), tuple_306278)
        # Getting the type of the for loop variable (line 79)
        for_loop_var_306290 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 79, 20), tuple_306278)
        # Assigning a type to the variable 'key' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 20), for_loop_var_306290))
        # Assigning a type to the variable 'default' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'default', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 20), for_loop_var_306290))
        # SSA begins for a for statement (line 79)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to has_option(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'server' (line 83)
        server_306293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 45), 'server', False)
        # Getting the type of 'key' (line 83)
        key_306294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 53), 'key', False)
        # Processing the call keyword arguments (line 83)
        kwargs_306295 = {}
        # Getting the type of 'config' (line 83)
        config_306291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 27), 'config', False)
        # Obtaining the member 'has_option' of a type (line 83)
        has_option_306292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 27), config_306291, 'has_option')
        # Calling has_option(args, kwargs) (line 83)
        has_option_call_result_306296 = invoke(stypy.reporting.localization.Localization(__file__, 83, 27), has_option_306292, *[server_306293, key_306294], **kwargs_306295)
        
        # Testing the type of an if condition (line 83)
        if_condition_306297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 24), has_option_call_result_306296)
        # Assigning a type to the variable 'if_condition_306297' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'if_condition_306297', if_condition_306297)
        # SSA begins for if statement (line 83)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Subscript (line 84):
        
        # Call to get(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'server' (line 84)
        server_306300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 54), 'server', False)
        # Getting the type of 'key' (line 84)
        key_306301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 62), 'key', False)
        # Processing the call keyword arguments (line 84)
        kwargs_306302 = {}
        # Getting the type of 'config' (line 84)
        config_306298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 43), 'config', False)
        # Obtaining the member 'get' of a type (line 84)
        get_306299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 43), config_306298, 'get')
        # Calling get(args, kwargs) (line 84)
        get_call_result_306303 = invoke(stypy.reporting.localization.Localization(__file__, 84, 43), get_306299, *[server_306300, key_306301], **kwargs_306302)
        
        # Getting the type of 'current' (line 84)
        current_306304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 28), 'current')
        # Getting the type of 'key' (line 84)
        key_306305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 36), 'key')
        # Storing an element on a container (line 84)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 28), current_306304, (key_306305, get_call_result_306303))
        # SSA branch for the else part of an if statement (line 83)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Subscript (line 86):
        # Getting the type of 'default' (line 86)
        default_306306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 43), 'default')
        # Getting the type of 'current' (line 86)
        current_306307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 28), 'current')
        # Getting the type of 'key' (line 86)
        key_306308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 36), 'key')
        # Storing an element on a container (line 86)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 28), current_306307, (key_306308, default_306306))
        # SSA join for if statement (line 83)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        
        # Obtaining the type of the subscript
        str_306309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 32), 'str', 'server')
        # Getting the type of 'current' (line 87)
        current_306310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), 'current')
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___306311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 24), current_306310, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_306312 = invoke(stypy.reporting.localization.Localization(__file__, 87, 24), getitem___306311, str_306309)
        
        # Getting the type of 'repository' (line 87)
        repository_306313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 45), 'repository')
        # Applying the binary operator '==' (line 87)
        result_eq_306314 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 24), '==', subscript_call_result_306312, repository_306313)
        
        
        
        # Obtaining the type of the subscript
        str_306315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 32), 'str', 'repository')
        # Getting the type of 'current' (line 88)
        current_306316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'current')
        # Obtaining the member '__getitem__' of a type (line 88)
        getitem___306317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 24), current_306316, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 88)
        subscript_call_result_306318 = invoke(stypy.reporting.localization.Localization(__file__, 88, 24), getitem___306317, str_306315)
        
        # Getting the type of 'repository' (line 88)
        repository_306319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 49), 'repository')
        # Applying the binary operator '==' (line 88)
        result_eq_306320 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 24), '==', subscript_call_result_306318, repository_306319)
        
        # Applying the binary operator 'or' (line 87)
        result_or_keyword_306321 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 24), 'or', result_eq_306314, result_eq_306320)
        
        # Testing the type of an if condition (line 87)
        if_condition_306322 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 20), result_or_keyword_306321)
        # Assigning a type to the variable 'if_condition_306322' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'if_condition_306322', if_condition_306322)
        # SSA begins for if statement (line 87)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'current' (line 89)
        current_306323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 31), 'current')
        # Assigning a type to the variable 'stypy_return_type' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 24), 'stypy_return_type', current_306323)
        # SSA join for if statement (line 87)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 60)
        module_type_store.open_ssa_branch('else')
        
        
        str_306324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 17), 'str', 'server-login')
        # Getting the type of 'sections' (line 90)
        sections_306325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 35), 'sections')
        # Applying the binary operator 'in' (line 90)
        result_contains_306326 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 17), 'in', str_306324, sections_306325)
        
        # Testing the type of an if condition (line 90)
        if_condition_306327 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 17), result_contains_306326)
        # Assigning a type to the variable 'if_condition_306327' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 17), 'if_condition_306327', if_condition_306327)
        # SSA begins for if statement (line 90)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 92):
        str_306328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 25), 'str', 'server-login')
        # Assigning a type to the variable 'server' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'server', str_306328)
        
        
        # Call to has_option(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'server' (line 93)
        server_306331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 37), 'server', False)
        str_306332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 45), 'str', 'repository')
        # Processing the call keyword arguments (line 93)
        kwargs_306333 = {}
        # Getting the type of 'config' (line 93)
        config_306329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 19), 'config', False)
        # Obtaining the member 'has_option' of a type (line 93)
        has_option_306330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 19), config_306329, 'has_option')
        # Calling has_option(args, kwargs) (line 93)
        has_option_call_result_306334 = invoke(stypy.reporting.localization.Localization(__file__, 93, 19), has_option_306330, *[server_306331, str_306332], **kwargs_306333)
        
        # Testing the type of an if condition (line 93)
        if_condition_306335 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 16), has_option_call_result_306334)
        # Assigning a type to the variable 'if_condition_306335' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'if_condition_306335', if_condition_306335)
        # SSA begins for if statement (line 93)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 94):
        
        # Call to get(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'server' (line 94)
        server_306338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 44), 'server', False)
        str_306339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 52), 'str', 'repository')
        # Processing the call keyword arguments (line 94)
        kwargs_306340 = {}
        # Getting the type of 'config' (line 94)
        config_306336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 33), 'config', False)
        # Obtaining the member 'get' of a type (line 94)
        get_306337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 33), config_306336, 'get')
        # Calling get(args, kwargs) (line 94)
        get_call_result_306341 = invoke(stypy.reporting.localization.Localization(__file__, 94, 33), get_306337, *[server_306338, str_306339], **kwargs_306340)
        
        # Assigning a type to the variable 'repository' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'repository', get_call_result_306341)
        # SSA branch for the else part of an if statement (line 93)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 96):
        # Getting the type of 'self' (line 96)
        self_306342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 33), 'self')
        # Obtaining the member 'DEFAULT_REPOSITORY' of a type (line 96)
        DEFAULT_REPOSITORY_306343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 33), self_306342, 'DEFAULT_REPOSITORY')
        # Assigning a type to the variable 'repository' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 20), 'repository', DEFAULT_REPOSITORY_306343)
        # SSA join for if statement (line 93)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'dict' (line 97)
        dict_306344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 23), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 97)
        # Adding element type (key, value) (line 97)
        str_306345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 24), 'str', 'username')
        
        # Call to get(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'server' (line 97)
        server_306348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 47), 'server', False)
        str_306349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 55), 'str', 'username')
        # Processing the call keyword arguments (line 97)
        kwargs_306350 = {}
        # Getting the type of 'config' (line 97)
        config_306346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 36), 'config', False)
        # Obtaining the member 'get' of a type (line 97)
        get_306347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 36), config_306346, 'get')
        # Calling get(args, kwargs) (line 97)
        get_call_result_306351 = invoke(stypy.reporting.localization.Localization(__file__, 97, 36), get_306347, *[server_306348, str_306349], **kwargs_306350)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 23), dict_306344, (str_306345, get_call_result_306351))
        # Adding element type (key, value) (line 97)
        str_306352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 24), 'str', 'password')
        
        # Call to get(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'server' (line 98)
        server_306355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 47), 'server', False)
        str_306356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 55), 'str', 'password')
        # Processing the call keyword arguments (line 98)
        kwargs_306357 = {}
        # Getting the type of 'config' (line 98)
        config_306353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 36), 'config', False)
        # Obtaining the member 'get' of a type (line 98)
        get_306354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 36), config_306353, 'get')
        # Calling get(args, kwargs) (line 98)
        get_call_result_306358 = invoke(stypy.reporting.localization.Localization(__file__, 98, 36), get_306354, *[server_306355, str_306356], **kwargs_306357)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 23), dict_306344, (str_306352, get_call_result_306358))
        # Adding element type (key, value) (line 97)
        str_306359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 24), 'str', 'repository')
        # Getting the type of 'repository' (line 99)
        repository_306360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 38), 'repository')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 23), dict_306344, (str_306359, repository_306360))
        # Adding element type (key, value) (line 97)
        str_306361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 24), 'str', 'server')
        # Getting the type of 'server' (line 100)
        server_306362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 34), 'server')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 23), dict_306344, (str_306361, server_306362))
        # Adding element type (key, value) (line 97)
        str_306363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 24), 'str', 'realm')
        # Getting the type of 'self' (line 101)
        self_306364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 33), 'self')
        # Obtaining the member 'DEFAULT_REALM' of a type (line 101)
        DEFAULT_REALM_306365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 33), self_306364, 'DEFAULT_REALM')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 23), dict_306344, (str_306363, DEFAULT_REALM_306365))
        
        # Assigning a type to the variable 'stypy_return_type' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'stypy_return_type', dict_306344)
        # SSA join for if statement (line 90)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 60)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 54)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'dict' (line 103)
        dict_306366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 15), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 103)
        
        # Assigning a type to the variable 'stypy_return_type' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'stypy_return_type', dict_306366)
        
        # ################# End of '_read_pypirc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_read_pypirc' in the type store
        # Getting the type of 'stypy_return_type' (line 51)
        stypy_return_type_306367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_306367)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_read_pypirc'
        return stypy_return_type_306367


    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PyPIRCCommand.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        PyPIRCCommand.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PyPIRCCommand.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        PyPIRCCommand.initialize_options.__dict__.__setitem__('stypy_function_name', 'PyPIRCCommand.initialize_options')
        PyPIRCCommand.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        PyPIRCCommand.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        PyPIRCCommand.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PyPIRCCommand.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        PyPIRCCommand.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        PyPIRCCommand.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PyPIRCCommand.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PyPIRCCommand.initialize_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'initialize_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'initialize_options(...)' code ##################

        str_306368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'str', 'Initialize options.')
        
        # Assigning a Name to a Attribute (line 107):
        # Getting the type of 'None' (line 107)
        None_306369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 26), 'None')
        # Getting the type of 'self' (line 107)
        self_306370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'self')
        # Setting the type of the member 'repository' of a type (line 107)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 8), self_306370, 'repository', None_306369)
        
        # Assigning a Name to a Attribute (line 108):
        # Getting the type of 'None' (line 108)
        None_306371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 21), 'None')
        # Getting the type of 'self' (line 108)
        self_306372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'self')
        # Setting the type of the member 'realm' of a type (line 108)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), self_306372, 'realm', None_306371)
        
        # Assigning a Num to a Attribute (line 109):
        int_306373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 29), 'int')
        # Getting the type of 'self' (line 109)
        self_306374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'self')
        # Setting the type of the member 'show_response' of a type (line 109)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), self_306374, 'show_response', int_306373)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_306375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_306375)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_306375


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 111, 4, False)
        # Assigning a type to the variable 'self' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PyPIRCCommand.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        PyPIRCCommand.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PyPIRCCommand.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        PyPIRCCommand.finalize_options.__dict__.__setitem__('stypy_function_name', 'PyPIRCCommand.finalize_options')
        PyPIRCCommand.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        PyPIRCCommand.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        PyPIRCCommand.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PyPIRCCommand.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        PyPIRCCommand.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        PyPIRCCommand.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PyPIRCCommand.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PyPIRCCommand.finalize_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'finalize_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'finalize_options(...)' code ##################

        str_306376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 8), 'str', 'Finalizes options.')
        
        # Type idiom detected: calculating its left and rigth part (line 113)
        # Getting the type of 'self' (line 113)
        self_306377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 11), 'self')
        # Obtaining the member 'repository' of a type (line 113)
        repository_306378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 11), self_306377, 'repository')
        # Getting the type of 'None' (line 113)
        None_306379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 30), 'None')
        
        (may_be_306380, more_types_in_union_306381) = may_be_none(repository_306378, None_306379)

        if may_be_306380:

            if more_types_in_union_306381:
                # Runtime conditional SSA (line 113)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Attribute (line 114):
            # Getting the type of 'self' (line 114)
            self_306382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 30), 'self')
            # Obtaining the member 'DEFAULT_REPOSITORY' of a type (line 114)
            DEFAULT_REPOSITORY_306383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 30), self_306382, 'DEFAULT_REPOSITORY')
            # Getting the type of 'self' (line 114)
            self_306384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'self')
            # Setting the type of the member 'repository' of a type (line 114)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), self_306384, 'repository', DEFAULT_REPOSITORY_306383)

            if more_types_in_union_306381:
                # SSA join for if statement (line 113)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 115)
        # Getting the type of 'self' (line 115)
        self_306385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'self')
        # Obtaining the member 'realm' of a type (line 115)
        realm_306386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 11), self_306385, 'realm')
        # Getting the type of 'None' (line 115)
        None_306387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 25), 'None')
        
        (may_be_306388, more_types_in_union_306389) = may_be_none(realm_306386, None_306387)

        if may_be_306388:

            if more_types_in_union_306389:
                # Runtime conditional SSA (line 115)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Attribute (line 116):
            # Getting the type of 'self' (line 116)
            self_306390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 25), 'self')
            # Obtaining the member 'DEFAULT_REALM' of a type (line 116)
            DEFAULT_REALM_306391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 25), self_306390, 'DEFAULT_REALM')
            # Getting the type of 'self' (line 116)
            self_306392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'self')
            # Setting the type of the member 'realm' of a type (line 116)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 12), self_306392, 'realm', DEFAULT_REALM_306391)

            if more_types_in_union_306389:
                # SSA join for if statement (line 115)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 111)
        stypy_return_type_306393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_306393)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_306393


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 21, 0, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PyPIRCCommand.__init__', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'PyPIRCCommand' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'PyPIRCCommand', PyPIRCCommand)

# Assigning a Str to a Name (line 24):
str_306394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'str', 'https://upload.pypi.org/legacy/')
# Getting the type of 'PyPIRCCommand'
PyPIRCCommand_306395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PyPIRCCommand')
# Setting the type of the member 'DEFAULT_REPOSITORY' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PyPIRCCommand_306395, 'DEFAULT_REPOSITORY', str_306394)

# Assigning a Str to a Name (line 25):
str_306396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 20), 'str', 'pypi')
# Getting the type of 'PyPIRCCommand'
PyPIRCCommand_306397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PyPIRCCommand')
# Setting the type of the member 'DEFAULT_REALM' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PyPIRCCommand_306397, 'DEFAULT_REALM', str_306396)

# Assigning a Name to a Name (line 26):
# Getting the type of 'None' (line 26)
None_306398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 17), 'None')
# Getting the type of 'PyPIRCCommand'
PyPIRCCommand_306399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PyPIRCCommand')
# Setting the type of the member 'repository' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PyPIRCCommand_306399, 'repository', None_306398)

# Assigning a Name to a Name (line 27):
# Getting the type of 'None' (line 27)
None_306400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'None')
# Getting the type of 'PyPIRCCommand'
PyPIRCCommand_306401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PyPIRCCommand')
# Setting the type of the member 'realm' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PyPIRCCommand_306401, 'realm', None_306400)

# Assigning a List to a Name (line 29):

# Obtaining an instance of the builtin type 'list' (line 29)
list_306402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 29)
# Adding element type (line 29)

# Obtaining an instance of the builtin type 'tuple' (line 30)
tuple_306403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 30)
# Adding element type (line 30)
str_306404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 9), 'str', 'repository=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 9), tuple_306403, str_306404)
# Adding element type (line 30)
str_306405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 24), 'str', 'r')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 9), tuple_306403, str_306405)
# Adding element type (line 30)
str_306406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 9), 'str', 'url of repository [default: %s]')
# Getting the type of 'PyPIRCCommand'
PyPIRCCommand_306407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PyPIRCCommand')
# Obtaining the member 'DEFAULT_REPOSITORY' of a type
DEFAULT_REPOSITORY_306408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PyPIRCCommand_306407, 'DEFAULT_REPOSITORY')
# Applying the binary operator '%' (line 31)
result_mod_306409 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 9), '%', str_306406, DEFAULT_REPOSITORY_306408)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 9), tuple_306403, result_mod_306409)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 19), list_306402, tuple_306403)
# Adding element type (line 29)

# Obtaining an instance of the builtin type 'tuple' (line 33)
tuple_306410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 33)
# Adding element type (line 33)
str_306411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 9), 'str', 'show-response')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_306410, str_306411)
# Adding element type (line 33)
# Getting the type of 'None' (line 33)
None_306412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_306410, None_306412)
# Adding element type (line 33)
str_306413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 9), 'str', 'display full response text from server')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_306410, str_306413)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 19), list_306402, tuple_306410)

# Getting the type of 'PyPIRCCommand'
PyPIRCCommand_306414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PyPIRCCommand')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PyPIRCCommand_306414, 'user_options', list_306402)

# Assigning a List to a Name (line 36):

# Obtaining an instance of the builtin type 'list' (line 36)
list_306415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 36)
# Adding element type (line 36)
str_306416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 23), 'str', 'show-response')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 22), list_306415, str_306416)

# Getting the type of 'PyPIRCCommand'
PyPIRCCommand_306417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PyPIRCCommand')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PyPIRCCommand_306417, 'boolean_options', list_306415)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
