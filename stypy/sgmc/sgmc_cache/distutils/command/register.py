
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.register
2: 
3: Implements the Distutils 'register' command (register with the repository).
4: '''
5: 
6: # created 2002/10/21, Richard Jones
7: 
8: __revision__ = "$Id$"
9: 
10: import urllib2
11: import getpass
12: import urlparse
13: from warnings import warn
14: 
15: from distutils.core import PyPIRCCommand
16: from distutils import log
17: 
18: class register(PyPIRCCommand):
19: 
20:     description = ("register the distribution with the Python package index")
21:     user_options = PyPIRCCommand.user_options + [
22:         ('list-classifiers', None,
23:          'list the valid Trove classifiers'),
24:         ('strict', None ,
25:          'Will stop the registering if the meta-data are not fully compliant')
26:         ]
27:     boolean_options = PyPIRCCommand.boolean_options + [
28:         'verify', 'list-classifiers', 'strict']
29: 
30:     sub_commands = [('check', lambda self: True)]
31: 
32:     def initialize_options(self):
33:         PyPIRCCommand.initialize_options(self)
34:         self.list_classifiers = 0
35:         self.strict = 0
36: 
37:     def finalize_options(self):
38:         PyPIRCCommand.finalize_options(self)
39:         # setting options for the `check` subcommand
40:         check_options = {'strict': ('register', self.strict),
41:                          'restructuredtext': ('register', 1)}
42:         self.distribution.command_options['check'] = check_options
43: 
44:     def run(self):
45:         self.finalize_options()
46:         self._set_config()
47: 
48:         # Run sub commands
49:         for cmd_name in self.get_sub_commands():
50:             self.run_command(cmd_name)
51: 
52:         if self.dry_run:
53:             self.verify_metadata()
54:         elif self.list_classifiers:
55:             self.classifiers()
56:         else:
57:             self.send_metadata()
58: 
59:     def check_metadata(self):
60:         '''Deprecated API.'''
61:         warn("distutils.command.register.check_metadata is deprecated, \
62:               use the check command instead", PendingDeprecationWarning)
63:         check = self.distribution.get_command_obj('check')
64:         check.ensure_finalized()
65:         check.strict = self.strict
66:         check.restructuredtext = 1
67:         check.run()
68: 
69:     def _set_config(self):
70:         ''' Reads the configuration file and set attributes.
71:         '''
72:         config = self._read_pypirc()
73:         if config != {}:
74:             self.username = config['username']
75:             self.password = config['password']
76:             self.repository = config['repository']
77:             self.realm = config['realm']
78:             self.has_config = True
79:         else:
80:             if self.repository not in ('pypi', self.DEFAULT_REPOSITORY):
81:                 raise ValueError('%s not found in .pypirc' % self.repository)
82:             if self.repository == 'pypi':
83:                 self.repository = self.DEFAULT_REPOSITORY
84:             self.has_config = False
85: 
86:     def classifiers(self):
87:         ''' Fetch the list of classifiers from the server.
88:         '''
89:         response = urllib2.urlopen(self.repository+'?:action=list_classifiers')
90:         log.info(response.read())
91: 
92:     def verify_metadata(self):
93:         ''' Send the metadata to the package index server to be checked.
94:         '''
95:         # send the info to the server and report the result
96:         (code, result) = self.post_to_server(self.build_post_data('verify'))
97:         log.info('Server response (%s): %s' % (code, result))
98: 
99: 
100:     def send_metadata(self):
101:         ''' Send the metadata to the package index server.
102: 
103:             Well, do the following:
104:             1. figure who the user is, and then
105:             2. send the data as a Basic auth'ed POST.
106: 
107:             First we try to read the username/password from $HOME/.pypirc,
108:             which is a ConfigParser-formatted file with a section
109:             [distutils] containing username and password entries (both
110:             in clear text). Eg:
111: 
112:                 [distutils]
113:                 index-servers =
114:                     pypi
115: 
116:                 [pypi]
117:                 username: fred
118:                 password: sekrit
119: 
120:             Otherwise, to figure who the user is, we offer the user three
121:             choices:
122: 
123:              1. use existing login,
124:              2. register as a new user, or
125:              3. set the password to a random string and email the user.
126: 
127:         '''
128:         # see if we can short-cut and get the username/password from the
129:         # config
130:         if self.has_config:
131:             choice = '1'
132:             username = self.username
133:             password = self.password
134:         else:
135:             choice = 'x'
136:             username = password = ''
137: 
138:         # get the user's login info
139:         choices = '1 2 3 4'.split()
140:         while choice not in choices:
141:             self.announce('''\
142: We need to know who you are, so please choose either:
143:  1. use your existing login,
144:  2. register as a new user,
145:  3. have the server generate a new password for you (and email it to you), or
146:  4. quit
147: Your selection [default 1]: ''', log.INFO)
148: 
149:             choice = raw_input()
150:             if not choice:
151:                 choice = '1'
152:             elif choice not in choices:
153:                 print 'Please choose one of the four options!'
154: 
155:         if choice == '1':
156:             # get the username and password
157:             while not username:
158:                 username = raw_input('Username: ')
159:             while not password:
160:                 password = getpass.getpass('Password: ')
161: 
162:             # set up the authentication
163:             auth = urllib2.HTTPPasswordMgr()
164:             host = urlparse.urlparse(self.repository)[1]
165:             auth.add_password(self.realm, host, username, password)
166:             # send the info to the server and report the result
167:             code, result = self.post_to_server(self.build_post_data('submit'),
168:                 auth)
169:             self.announce('Server response (%s): %s' % (code, result),
170:                           log.INFO)
171: 
172:             # possibly save the login
173:             if code == 200:
174:                 if self.has_config:
175:                     # sharing the password in the distribution instance
176:                     # so the upload command can reuse it
177:                     self.distribution.password = password
178:                 else:
179:                     self.announce(('I can store your PyPI login so future '
180:                                    'submissions will be faster.'), log.INFO)
181:                     self.announce('(the login will be stored in %s)' % \
182:                                   self._get_rc_file(), log.INFO)
183:                     choice = 'X'
184:                     while choice.lower() not in 'yn':
185:                         choice = raw_input('Save your login (y/N)?')
186:                         if not choice:
187:                             choice = 'n'
188:                     if choice.lower() == 'y':
189:                         self._store_pypirc(username, password)
190: 
191:         elif choice == '2':
192:             data = {':action': 'user'}
193:             data['name'] = data['password'] = data['email'] = ''
194:             data['confirm'] = None
195:             while not data['name']:
196:                 data['name'] = raw_input('Username: ')
197:             while data['password'] != data['confirm']:
198:                 while not data['password']:
199:                     data['password'] = getpass.getpass('Password: ')
200:                 while not data['confirm']:
201:                     data['confirm'] = getpass.getpass(' Confirm: ')
202:                 if data['password'] != data['confirm']:
203:                     data['password'] = ''
204:                     data['confirm'] = None
205:                     print "Password and confirm don't match!"
206:             while not data['email']:
207:                 data['email'] = raw_input('   EMail: ')
208:             code, result = self.post_to_server(data)
209:             if code != 200:
210:                 log.info('Server response (%s): %s' % (code, result))
211:             else:
212:                 log.info('You will receive an email shortly.')
213:                 log.info(('Follow the instructions in it to '
214:                           'complete registration.'))
215:         elif choice == '3':
216:             data = {':action': 'password_reset'}
217:             data['email'] = ''
218:             while not data['email']:
219:                 data['email'] = raw_input('Your email address: ')
220:             code, result = self.post_to_server(data)
221:             log.info('Server response (%s): %s' % (code, result))
222: 
223:     def build_post_data(self, action):
224:         # figure the data to send - the metadata plus some additional
225:         # information used by the package server
226:         meta = self.distribution.metadata
227:         data = {
228:             ':action': action,
229:             'metadata_version' : '1.0',
230:             'name': meta.get_name(),
231:             'version': meta.get_version(),
232:             'summary': meta.get_description(),
233:             'home_page': meta.get_url(),
234:             'author': meta.get_contact(),
235:             'author_email': meta.get_contact_email(),
236:             'license': meta.get_licence(),
237:             'description': meta.get_long_description(),
238:             'keywords': meta.get_keywords(),
239:             'platform': meta.get_platforms(),
240:             'classifiers': meta.get_classifiers(),
241:             'download_url': meta.get_download_url(),
242:             # PEP 314
243:             'provides': meta.get_provides(),
244:             'requires': meta.get_requires(),
245:             'obsoletes': meta.get_obsoletes(),
246:         }
247:         if data['provides'] or data['requires'] or data['obsoletes']:
248:             data['metadata_version'] = '1.1'
249:         return data
250: 
251:     def post_to_server(self, data, auth=None):
252:         ''' Post a query to the server, and return a string response.
253:         '''
254:         if 'name' in data:
255:             self.announce('Registering %s to %s' % (data['name'],
256:                                                    self.repository),
257:                                                    log.INFO)
258:         # Build up the MIME payload for the urllib2 POST data
259:         boundary = '--------------GHSKFJDLGDS7543FJKLFHRE75642756743254'
260:         sep_boundary = '\n--' + boundary
261:         end_boundary = sep_boundary + '--'
262:         chunks = []
263:         for key, value in data.items():
264:             # handle multiple entries for the same name
265:             if type(value) not in (type([]), type( () )):
266:                 value = [value]
267:             for value in value:
268:                 chunks.append(sep_boundary)
269:                 chunks.append('\nContent-Disposition: form-data; name="%s"'%key)
270:                 chunks.append("\n\n")
271:                 chunks.append(value)
272:                 if value and value[-1] == '\r':
273:                     chunks.append('\n')  # write an extra newline (lurve Macs)
274:         chunks.append(end_boundary)
275:         chunks.append("\n")
276: 
277:         # chunks may be bytes (str) or unicode objects that we need to encode
278:         body = []
279:         for chunk in chunks:
280:             if isinstance(chunk, unicode):
281:                 body.append(chunk.encode('utf-8'))
282:             else:
283:                 body.append(chunk)
284: 
285:         body = ''.join(body)
286: 
287:         # build the Request
288:         headers = {
289:             'Content-type': 'multipart/form-data; boundary=%s; charset=utf-8'%boundary,
290:             'Content-length': str(len(body))
291:         }
292:         req = urllib2.Request(self.repository, body, headers)
293: 
294:         # handle HTTP and include the Basic Auth handler
295:         opener = urllib2.build_opener(
296:             urllib2.HTTPBasicAuthHandler(password_mgr=auth)
297:         )
298:         data = ''
299:         try:
300:             result = opener.open(req)
301:         except urllib2.HTTPError, e:
302:             if self.show_response:
303:                 data = e.fp.read()
304:             result = e.code, e.msg
305:         except urllib2.URLError, e:
306:             result = 500, str(e)
307:         else:
308:             if self.show_response:
309:                 data = result.read()
310:             result = 200, 'OK'
311:         if self.show_response:
312:             dashes = '-' * 75
313:             self.announce('%s%s%s' % (dashes, data, dashes))
314: 
315:         return result
316: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_24729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', "distutils.command.register\n\nImplements the Distutils 'register' command (register with the repository).\n")

# Assigning a Str to a Name (line 8):

# Assigning a Str to a Name (line 8):
str_24730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__revision__', str_24730)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import urllib2' statement (line 10)
import urllib2

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'urllib2', urllib2, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import getpass' statement (line 11)
import getpass

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'getpass', getpass, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import urlparse' statement (line 12)
import urlparse

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'urlparse', urlparse, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from warnings import warn' statement (line 13)
try:
    from warnings import warn

except:
    warn = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'warnings', None, module_type_store, ['warn'], [warn])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from distutils.core import PyPIRCCommand' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_24731 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.core')

if (type(import_24731) is not StypyTypeError):

    if (import_24731 != 'pyd_module'):
        __import__(import_24731)
        sys_modules_24732 = sys.modules[import_24731]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.core', sys_modules_24732.module_type_store, module_type_store, ['PyPIRCCommand'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_24732, sys_modules_24732.module_type_store, module_type_store)
    else:
        from distutils.core import PyPIRCCommand

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.core', None, module_type_store, ['PyPIRCCommand'], [PyPIRCCommand])

else:
    # Assigning a type to the variable 'distutils.core' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.core', import_24731)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from distutils import log' statement (line 16)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils', None, module_type_store, ['log'], [log])

# Declaration of the 'register' class
# Getting the type of 'PyPIRCCommand' (line 18)
PyPIRCCommand_24733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 15), 'PyPIRCCommand')

class register(PyPIRCCommand_24733, ):
    
    # Assigning a Str to a Name (line 20):
    
    # Assigning a BinOp to a Name (line 21):
    
    # Assigning a BinOp to a Name (line 27):
    
    # Assigning a List to a Name (line 30):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        register.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        register.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        register.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        register.initialize_options.__dict__.__setitem__('stypy_function_name', 'register.initialize_options')
        register.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        register.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        register.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        register.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        register.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        register.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        register.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'register.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to initialize_options(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'self' (line 33)
        self_24736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 41), 'self', False)
        # Processing the call keyword arguments (line 33)
        kwargs_24737 = {}
        # Getting the type of 'PyPIRCCommand' (line 33)
        PyPIRCCommand_24734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'PyPIRCCommand', False)
        # Obtaining the member 'initialize_options' of a type (line 33)
        initialize_options_24735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), PyPIRCCommand_24734, 'initialize_options')
        # Calling initialize_options(args, kwargs) (line 33)
        initialize_options_call_result_24738 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), initialize_options_24735, *[self_24736], **kwargs_24737)
        
        
        # Assigning a Num to a Attribute (line 34):
        
        # Assigning a Num to a Attribute (line 34):
        int_24739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 32), 'int')
        # Getting the type of 'self' (line 34)
        self_24740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'list_classifiers' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_24740, 'list_classifiers', int_24739)
        
        # Assigning a Num to a Attribute (line 35):
        
        # Assigning a Num to a Attribute (line 35):
        int_24741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 22), 'int')
        # Getting the type of 'self' (line 35)
        self_24742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self')
        # Setting the type of the member 'strict' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_24742, 'strict', int_24741)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_24743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24743)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_24743


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        register.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        register.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        register.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        register.finalize_options.__dict__.__setitem__('stypy_function_name', 'register.finalize_options')
        register.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        register.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        register.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        register.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        register.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        register.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        register.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'register.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to finalize_options(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'self' (line 38)
        self_24746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 39), 'self', False)
        # Processing the call keyword arguments (line 38)
        kwargs_24747 = {}
        # Getting the type of 'PyPIRCCommand' (line 38)
        PyPIRCCommand_24744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'PyPIRCCommand', False)
        # Obtaining the member 'finalize_options' of a type (line 38)
        finalize_options_24745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), PyPIRCCommand_24744, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 38)
        finalize_options_call_result_24748 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), finalize_options_24745, *[self_24746], **kwargs_24747)
        
        
        # Assigning a Dict to a Name (line 40):
        
        # Assigning a Dict to a Name (line 40):
        
        # Obtaining an instance of the builtin type 'dict' (line 40)
        dict_24749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 24), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 40)
        # Adding element type (key, value) (line 40)
        str_24750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 25), 'str', 'strict')
        
        # Obtaining an instance of the builtin type 'tuple' (line 40)
        tuple_24751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 40)
        # Adding element type (line 40)
        str_24752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 36), 'str', 'register')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 36), tuple_24751, str_24752)
        # Adding element type (line 40)
        # Getting the type of 'self' (line 40)
        self_24753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 48), 'self')
        # Obtaining the member 'strict' of a type (line 40)
        strict_24754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 48), self_24753, 'strict')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 36), tuple_24751, strict_24754)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 24), dict_24749, (str_24750, tuple_24751))
        # Adding element type (key, value) (line 40)
        str_24755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 25), 'str', 'restructuredtext')
        
        # Obtaining an instance of the builtin type 'tuple' (line 41)
        tuple_24756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 41)
        # Adding element type (line 41)
        str_24757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 46), 'str', 'register')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 46), tuple_24756, str_24757)
        # Adding element type (line 41)
        int_24758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 46), tuple_24756, int_24758)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 24), dict_24749, (str_24755, tuple_24756))
        
        # Assigning a type to the variable 'check_options' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'check_options', dict_24749)
        
        # Assigning a Name to a Subscript (line 42):
        
        # Assigning a Name to a Subscript (line 42):
        # Getting the type of 'check_options' (line 42)
        check_options_24759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 53), 'check_options')
        # Getting the type of 'self' (line 42)
        self_24760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self')
        # Obtaining the member 'distribution' of a type (line 42)
        distribution_24761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_24760, 'distribution')
        # Obtaining the member 'command_options' of a type (line 42)
        command_options_24762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), distribution_24761, 'command_options')
        str_24763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 42), 'str', 'check')
        # Storing an element on a container (line 42)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 8), command_options_24762, (str_24763, check_options_24759))
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_24764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24764)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_24764


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        register.run.__dict__.__setitem__('stypy_localization', localization)
        register.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        register.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        register.run.__dict__.__setitem__('stypy_function_name', 'register.run')
        register.run.__dict__.__setitem__('stypy_param_names_list', [])
        register.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        register.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        register.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        register.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        register.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        register.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'register.run', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run(...)' code ##################

        
        # Call to finalize_options(...): (line 45)
        # Processing the call keyword arguments (line 45)
        kwargs_24767 = {}
        # Getting the type of 'self' (line 45)
        self_24765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self', False)
        # Obtaining the member 'finalize_options' of a type (line 45)
        finalize_options_24766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_24765, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 45)
        finalize_options_call_result_24768 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), finalize_options_24766, *[], **kwargs_24767)
        
        
        # Call to _set_config(...): (line 46)
        # Processing the call keyword arguments (line 46)
        kwargs_24771 = {}
        # Getting the type of 'self' (line 46)
        self_24769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self', False)
        # Obtaining the member '_set_config' of a type (line 46)
        _set_config_24770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_24769, '_set_config')
        # Calling _set_config(args, kwargs) (line 46)
        _set_config_call_result_24772 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), _set_config_24770, *[], **kwargs_24771)
        
        
        
        # Call to get_sub_commands(...): (line 49)
        # Processing the call keyword arguments (line 49)
        kwargs_24775 = {}
        # Getting the type of 'self' (line 49)
        self_24773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 24), 'self', False)
        # Obtaining the member 'get_sub_commands' of a type (line 49)
        get_sub_commands_24774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 24), self_24773, 'get_sub_commands')
        # Calling get_sub_commands(args, kwargs) (line 49)
        get_sub_commands_call_result_24776 = invoke(stypy.reporting.localization.Localization(__file__, 49, 24), get_sub_commands_24774, *[], **kwargs_24775)
        
        # Testing the type of a for loop iterable (line 49)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 49, 8), get_sub_commands_call_result_24776)
        # Getting the type of the for loop variable (line 49)
        for_loop_var_24777 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 49, 8), get_sub_commands_call_result_24776)
        # Assigning a type to the variable 'cmd_name' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'cmd_name', for_loop_var_24777)
        # SSA begins for a for statement (line 49)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to run_command(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'cmd_name' (line 50)
        cmd_name_24780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 29), 'cmd_name', False)
        # Processing the call keyword arguments (line 50)
        kwargs_24781 = {}
        # Getting the type of 'self' (line 50)
        self_24778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'self', False)
        # Obtaining the member 'run_command' of a type (line 50)
        run_command_24779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), self_24778, 'run_command')
        # Calling run_command(args, kwargs) (line 50)
        run_command_call_result_24782 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), run_command_24779, *[cmd_name_24780], **kwargs_24781)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 52)
        self_24783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'self')
        # Obtaining the member 'dry_run' of a type (line 52)
        dry_run_24784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 11), self_24783, 'dry_run')
        # Testing the type of an if condition (line 52)
        if_condition_24785 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 8), dry_run_24784)
        # Assigning a type to the variable 'if_condition_24785' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'if_condition_24785', if_condition_24785)
        # SSA begins for if statement (line 52)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to verify_metadata(...): (line 53)
        # Processing the call keyword arguments (line 53)
        kwargs_24788 = {}
        # Getting the type of 'self' (line 53)
        self_24786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'self', False)
        # Obtaining the member 'verify_metadata' of a type (line 53)
        verify_metadata_24787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 12), self_24786, 'verify_metadata')
        # Calling verify_metadata(args, kwargs) (line 53)
        verify_metadata_call_result_24789 = invoke(stypy.reporting.localization.Localization(__file__, 53, 12), verify_metadata_24787, *[], **kwargs_24788)
        
        # SSA branch for the else part of an if statement (line 52)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 54)
        self_24790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'self')
        # Obtaining the member 'list_classifiers' of a type (line 54)
        list_classifiers_24791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 13), self_24790, 'list_classifiers')
        # Testing the type of an if condition (line 54)
        if_condition_24792 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 13), list_classifiers_24791)
        # Assigning a type to the variable 'if_condition_24792' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'if_condition_24792', if_condition_24792)
        # SSA begins for if statement (line 54)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to classifiers(...): (line 55)
        # Processing the call keyword arguments (line 55)
        kwargs_24795 = {}
        # Getting the type of 'self' (line 55)
        self_24793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'self', False)
        # Obtaining the member 'classifiers' of a type (line 55)
        classifiers_24794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 12), self_24793, 'classifiers')
        # Calling classifiers(args, kwargs) (line 55)
        classifiers_call_result_24796 = invoke(stypy.reporting.localization.Localization(__file__, 55, 12), classifiers_24794, *[], **kwargs_24795)
        
        # SSA branch for the else part of an if statement (line 54)
        module_type_store.open_ssa_branch('else')
        
        # Call to send_metadata(...): (line 57)
        # Processing the call keyword arguments (line 57)
        kwargs_24799 = {}
        # Getting the type of 'self' (line 57)
        self_24797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'self', False)
        # Obtaining the member 'send_metadata' of a type (line 57)
        send_metadata_24798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), self_24797, 'send_metadata')
        # Calling send_metadata(args, kwargs) (line 57)
        send_metadata_call_result_24800 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), send_metadata_24798, *[], **kwargs_24799)
        
        # SSA join for if statement (line 54)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 52)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_24801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24801)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_24801


    @norecursion
    def check_metadata(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_metadata'
        module_type_store = module_type_store.open_function_context('check_metadata', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        register.check_metadata.__dict__.__setitem__('stypy_localization', localization)
        register.check_metadata.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        register.check_metadata.__dict__.__setitem__('stypy_type_store', module_type_store)
        register.check_metadata.__dict__.__setitem__('stypy_function_name', 'register.check_metadata')
        register.check_metadata.__dict__.__setitem__('stypy_param_names_list', [])
        register.check_metadata.__dict__.__setitem__('stypy_varargs_param_name', None)
        register.check_metadata.__dict__.__setitem__('stypy_kwargs_param_name', None)
        register.check_metadata.__dict__.__setitem__('stypy_call_defaults', defaults)
        register.check_metadata.__dict__.__setitem__('stypy_call_varargs', varargs)
        register.check_metadata.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        register.check_metadata.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'register.check_metadata', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_metadata', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_metadata(...)' code ##################

        str_24802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'str', 'Deprecated API.')
        
        # Call to warn(...): (line 61)
        # Processing the call arguments (line 61)
        str_24804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, (-1)), 'str', 'distutils.command.register.check_metadata is deprecated,               use the check command instead')
        # Getting the type of 'PendingDeprecationWarning' (line 62)
        PendingDeprecationWarning_24805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 46), 'PendingDeprecationWarning', False)
        # Processing the call keyword arguments (line 61)
        kwargs_24806 = {}
        # Getting the type of 'warn' (line 61)
        warn_24803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'warn', False)
        # Calling warn(args, kwargs) (line 61)
        warn_call_result_24807 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), warn_24803, *[str_24804, PendingDeprecationWarning_24805], **kwargs_24806)
        
        
        # Assigning a Call to a Name (line 63):
        
        # Assigning a Call to a Name (line 63):
        
        # Call to get_command_obj(...): (line 63)
        # Processing the call arguments (line 63)
        str_24811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 50), 'str', 'check')
        # Processing the call keyword arguments (line 63)
        kwargs_24812 = {}
        # Getting the type of 'self' (line 63)
        self_24808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'self', False)
        # Obtaining the member 'distribution' of a type (line 63)
        distribution_24809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 16), self_24808, 'distribution')
        # Obtaining the member 'get_command_obj' of a type (line 63)
        get_command_obj_24810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 16), distribution_24809, 'get_command_obj')
        # Calling get_command_obj(args, kwargs) (line 63)
        get_command_obj_call_result_24813 = invoke(stypy.reporting.localization.Localization(__file__, 63, 16), get_command_obj_24810, *[str_24811], **kwargs_24812)
        
        # Assigning a type to the variable 'check' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'check', get_command_obj_call_result_24813)
        
        # Call to ensure_finalized(...): (line 64)
        # Processing the call keyword arguments (line 64)
        kwargs_24816 = {}
        # Getting the type of 'check' (line 64)
        check_24814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'check', False)
        # Obtaining the member 'ensure_finalized' of a type (line 64)
        ensure_finalized_24815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), check_24814, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 64)
        ensure_finalized_call_result_24817 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), ensure_finalized_24815, *[], **kwargs_24816)
        
        
        # Assigning a Attribute to a Attribute (line 65):
        
        # Assigning a Attribute to a Attribute (line 65):
        # Getting the type of 'self' (line 65)
        self_24818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 23), 'self')
        # Obtaining the member 'strict' of a type (line 65)
        strict_24819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 23), self_24818, 'strict')
        # Getting the type of 'check' (line 65)
        check_24820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'check')
        # Setting the type of the member 'strict' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), check_24820, 'strict', strict_24819)
        
        # Assigning a Num to a Attribute (line 66):
        
        # Assigning a Num to a Attribute (line 66):
        int_24821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 33), 'int')
        # Getting the type of 'check' (line 66)
        check_24822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'check')
        # Setting the type of the member 'restructuredtext' of a type (line 66)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), check_24822, 'restructuredtext', int_24821)
        
        # Call to run(...): (line 67)
        # Processing the call keyword arguments (line 67)
        kwargs_24825 = {}
        # Getting the type of 'check' (line 67)
        check_24823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'check', False)
        # Obtaining the member 'run' of a type (line 67)
        run_24824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), check_24823, 'run')
        # Calling run(args, kwargs) (line 67)
        run_call_result_24826 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), run_24824, *[], **kwargs_24825)
        
        
        # ################# End of 'check_metadata(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_metadata' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_24827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24827)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_metadata'
        return stypy_return_type_24827


    @norecursion
    def _set_config(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_set_config'
        module_type_store = module_type_store.open_function_context('_set_config', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        register._set_config.__dict__.__setitem__('stypy_localization', localization)
        register._set_config.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        register._set_config.__dict__.__setitem__('stypy_type_store', module_type_store)
        register._set_config.__dict__.__setitem__('stypy_function_name', 'register._set_config')
        register._set_config.__dict__.__setitem__('stypy_param_names_list', [])
        register._set_config.__dict__.__setitem__('stypy_varargs_param_name', None)
        register._set_config.__dict__.__setitem__('stypy_kwargs_param_name', None)
        register._set_config.__dict__.__setitem__('stypy_call_defaults', defaults)
        register._set_config.__dict__.__setitem__('stypy_call_varargs', varargs)
        register._set_config.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        register._set_config.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'register._set_config', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_config', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_config(...)' code ##################

        str_24828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, (-1)), 'str', ' Reads the configuration file and set attributes.\n        ')
        
        # Assigning a Call to a Name (line 72):
        
        # Assigning a Call to a Name (line 72):
        
        # Call to _read_pypirc(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_24831 = {}
        # Getting the type of 'self' (line 72)
        self_24829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'self', False)
        # Obtaining the member '_read_pypirc' of a type (line 72)
        _read_pypirc_24830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 17), self_24829, '_read_pypirc')
        # Calling _read_pypirc(args, kwargs) (line 72)
        _read_pypirc_call_result_24832 = invoke(stypy.reporting.localization.Localization(__file__, 72, 17), _read_pypirc_24830, *[], **kwargs_24831)
        
        # Assigning a type to the variable 'config' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'config', _read_pypirc_call_result_24832)
        
        
        # Getting the type of 'config' (line 73)
        config_24833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'config')
        
        # Obtaining an instance of the builtin type 'dict' (line 73)
        dict_24834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 21), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 73)
        
        # Applying the binary operator '!=' (line 73)
        result_ne_24835 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 11), '!=', config_24833, dict_24834)
        
        # Testing the type of an if condition (line 73)
        if_condition_24836 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 8), result_ne_24835)
        # Assigning a type to the variable 'if_condition_24836' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'if_condition_24836', if_condition_24836)
        # SSA begins for if statement (line 73)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Attribute (line 74):
        
        # Assigning a Subscript to a Attribute (line 74):
        
        # Obtaining the type of the subscript
        str_24837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 35), 'str', 'username')
        # Getting the type of 'config' (line 74)
        config_24838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'config')
        # Obtaining the member '__getitem__' of a type (line 74)
        getitem___24839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 28), config_24838, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 74)
        subscript_call_result_24840 = invoke(stypy.reporting.localization.Localization(__file__, 74, 28), getitem___24839, str_24837)
        
        # Getting the type of 'self' (line 74)
        self_24841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'self')
        # Setting the type of the member 'username' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 12), self_24841, 'username', subscript_call_result_24840)
        
        # Assigning a Subscript to a Attribute (line 75):
        
        # Assigning a Subscript to a Attribute (line 75):
        
        # Obtaining the type of the subscript
        str_24842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 35), 'str', 'password')
        # Getting the type of 'config' (line 75)
        config_24843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 28), 'config')
        # Obtaining the member '__getitem__' of a type (line 75)
        getitem___24844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 28), config_24843, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 75)
        subscript_call_result_24845 = invoke(stypy.reporting.localization.Localization(__file__, 75, 28), getitem___24844, str_24842)
        
        # Getting the type of 'self' (line 75)
        self_24846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'self')
        # Setting the type of the member 'password' of a type (line 75)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), self_24846, 'password', subscript_call_result_24845)
        
        # Assigning a Subscript to a Attribute (line 76):
        
        # Assigning a Subscript to a Attribute (line 76):
        
        # Obtaining the type of the subscript
        str_24847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 37), 'str', 'repository')
        # Getting the type of 'config' (line 76)
        config_24848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 30), 'config')
        # Obtaining the member '__getitem__' of a type (line 76)
        getitem___24849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 30), config_24848, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 76)
        subscript_call_result_24850 = invoke(stypy.reporting.localization.Localization(__file__, 76, 30), getitem___24849, str_24847)
        
        # Getting the type of 'self' (line 76)
        self_24851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'self')
        # Setting the type of the member 'repository' of a type (line 76)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), self_24851, 'repository', subscript_call_result_24850)
        
        # Assigning a Subscript to a Attribute (line 77):
        
        # Assigning a Subscript to a Attribute (line 77):
        
        # Obtaining the type of the subscript
        str_24852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 32), 'str', 'realm')
        # Getting the type of 'config' (line 77)
        config_24853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), 'config')
        # Obtaining the member '__getitem__' of a type (line 77)
        getitem___24854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 25), config_24853, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 77)
        subscript_call_result_24855 = invoke(stypy.reporting.localization.Localization(__file__, 77, 25), getitem___24854, str_24852)
        
        # Getting the type of 'self' (line 77)
        self_24856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'self')
        # Setting the type of the member 'realm' of a type (line 77)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 12), self_24856, 'realm', subscript_call_result_24855)
        
        # Assigning a Name to a Attribute (line 78):
        
        # Assigning a Name to a Attribute (line 78):
        # Getting the type of 'True' (line 78)
        True_24857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), 'True')
        # Getting the type of 'self' (line 78)
        self_24858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'self')
        # Setting the type of the member 'has_config' of a type (line 78)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), self_24858, 'has_config', True_24857)
        # SSA branch for the else part of an if statement (line 73)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 80)
        self_24859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'self')
        # Obtaining the member 'repository' of a type (line 80)
        repository_24860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 15), self_24859, 'repository')
        
        # Obtaining an instance of the builtin type 'tuple' (line 80)
        tuple_24861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 80)
        # Adding element type (line 80)
        str_24862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 39), 'str', 'pypi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 39), tuple_24861, str_24862)
        # Adding element type (line 80)
        # Getting the type of 'self' (line 80)
        self_24863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 47), 'self')
        # Obtaining the member 'DEFAULT_REPOSITORY' of a type (line 80)
        DEFAULT_REPOSITORY_24864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 47), self_24863, 'DEFAULT_REPOSITORY')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 39), tuple_24861, DEFAULT_REPOSITORY_24864)
        
        # Applying the binary operator 'notin' (line 80)
        result_contains_24865 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 15), 'notin', repository_24860, tuple_24861)
        
        # Testing the type of an if condition (line 80)
        if_condition_24866 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 12), result_contains_24865)
        # Assigning a type to the variable 'if_condition_24866' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'if_condition_24866', if_condition_24866)
        # SSA begins for if statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 81)
        # Processing the call arguments (line 81)
        str_24868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 33), 'str', '%s not found in .pypirc')
        # Getting the type of 'self' (line 81)
        self_24869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 61), 'self', False)
        # Obtaining the member 'repository' of a type (line 81)
        repository_24870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 61), self_24869, 'repository')
        # Applying the binary operator '%' (line 81)
        result_mod_24871 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 33), '%', str_24868, repository_24870)
        
        # Processing the call keyword arguments (line 81)
        kwargs_24872 = {}
        # Getting the type of 'ValueError' (line 81)
        ValueError_24867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 81)
        ValueError_call_result_24873 = invoke(stypy.reporting.localization.Localization(__file__, 81, 22), ValueError_24867, *[result_mod_24871], **kwargs_24872)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 81, 16), ValueError_call_result_24873, 'raise parameter', BaseException)
        # SSA join for if statement (line 80)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 82)
        self_24874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'self')
        # Obtaining the member 'repository' of a type (line 82)
        repository_24875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), self_24874, 'repository')
        str_24876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 34), 'str', 'pypi')
        # Applying the binary operator '==' (line 82)
        result_eq_24877 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 15), '==', repository_24875, str_24876)
        
        # Testing the type of an if condition (line 82)
        if_condition_24878 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 12), result_eq_24877)
        # Assigning a type to the variable 'if_condition_24878' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'if_condition_24878', if_condition_24878)
        # SSA begins for if statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 83):
        
        # Assigning a Attribute to a Attribute (line 83):
        # Getting the type of 'self' (line 83)
        self_24879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 34), 'self')
        # Obtaining the member 'DEFAULT_REPOSITORY' of a type (line 83)
        DEFAULT_REPOSITORY_24880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 34), self_24879, 'DEFAULT_REPOSITORY')
        # Getting the type of 'self' (line 83)
        self_24881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'self')
        # Setting the type of the member 'repository' of a type (line 83)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 16), self_24881, 'repository', DEFAULT_REPOSITORY_24880)
        # SSA join for if statement (line 82)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 84):
        
        # Assigning a Name to a Attribute (line 84):
        # Getting the type of 'False' (line 84)
        False_24882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 30), 'False')
        # Getting the type of 'self' (line 84)
        self_24883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'self')
        # Setting the type of the member 'has_config' of a type (line 84)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), self_24883, 'has_config', False_24882)
        # SSA join for if statement (line 73)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_set_config(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_config' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_24884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24884)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_config'
        return stypy_return_type_24884


    @norecursion
    def classifiers(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'classifiers'
        module_type_store = module_type_store.open_function_context('classifiers', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        register.classifiers.__dict__.__setitem__('stypy_localization', localization)
        register.classifiers.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        register.classifiers.__dict__.__setitem__('stypy_type_store', module_type_store)
        register.classifiers.__dict__.__setitem__('stypy_function_name', 'register.classifiers')
        register.classifiers.__dict__.__setitem__('stypy_param_names_list', [])
        register.classifiers.__dict__.__setitem__('stypy_varargs_param_name', None)
        register.classifiers.__dict__.__setitem__('stypy_kwargs_param_name', None)
        register.classifiers.__dict__.__setitem__('stypy_call_defaults', defaults)
        register.classifiers.__dict__.__setitem__('stypy_call_varargs', varargs)
        register.classifiers.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        register.classifiers.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'register.classifiers', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'classifiers', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'classifiers(...)' code ##################

        str_24885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, (-1)), 'str', ' Fetch the list of classifiers from the server.\n        ')
        
        # Assigning a Call to a Name (line 89):
        
        # Assigning a Call to a Name (line 89):
        
        # Call to urlopen(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'self' (line 89)
        self_24888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 35), 'self', False)
        # Obtaining the member 'repository' of a type (line 89)
        repository_24889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 35), self_24888, 'repository')
        str_24890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 51), 'str', '?:action=list_classifiers')
        # Applying the binary operator '+' (line 89)
        result_add_24891 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 35), '+', repository_24889, str_24890)
        
        # Processing the call keyword arguments (line 89)
        kwargs_24892 = {}
        # Getting the type of 'urllib2' (line 89)
        urllib2_24886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 19), 'urllib2', False)
        # Obtaining the member 'urlopen' of a type (line 89)
        urlopen_24887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 19), urllib2_24886, 'urlopen')
        # Calling urlopen(args, kwargs) (line 89)
        urlopen_call_result_24893 = invoke(stypy.reporting.localization.Localization(__file__, 89, 19), urlopen_24887, *[result_add_24891], **kwargs_24892)
        
        # Assigning a type to the variable 'response' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'response', urlopen_call_result_24893)
        
        # Call to info(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Call to read(...): (line 90)
        # Processing the call keyword arguments (line 90)
        kwargs_24898 = {}
        # Getting the type of 'response' (line 90)
        response_24896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 17), 'response', False)
        # Obtaining the member 'read' of a type (line 90)
        read_24897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 17), response_24896, 'read')
        # Calling read(args, kwargs) (line 90)
        read_call_result_24899 = invoke(stypy.reporting.localization.Localization(__file__, 90, 17), read_24897, *[], **kwargs_24898)
        
        # Processing the call keyword arguments (line 90)
        kwargs_24900 = {}
        # Getting the type of 'log' (line 90)
        log_24894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 90)
        info_24895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), log_24894, 'info')
        # Calling info(args, kwargs) (line 90)
        info_call_result_24901 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), info_24895, *[read_call_result_24899], **kwargs_24900)
        
        
        # ################# End of 'classifiers(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'classifiers' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_24902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24902)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'classifiers'
        return stypy_return_type_24902


    @norecursion
    def verify_metadata(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'verify_metadata'
        module_type_store = module_type_store.open_function_context('verify_metadata', 92, 4, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        register.verify_metadata.__dict__.__setitem__('stypy_localization', localization)
        register.verify_metadata.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        register.verify_metadata.__dict__.__setitem__('stypy_type_store', module_type_store)
        register.verify_metadata.__dict__.__setitem__('stypy_function_name', 'register.verify_metadata')
        register.verify_metadata.__dict__.__setitem__('stypy_param_names_list', [])
        register.verify_metadata.__dict__.__setitem__('stypy_varargs_param_name', None)
        register.verify_metadata.__dict__.__setitem__('stypy_kwargs_param_name', None)
        register.verify_metadata.__dict__.__setitem__('stypy_call_defaults', defaults)
        register.verify_metadata.__dict__.__setitem__('stypy_call_varargs', varargs)
        register.verify_metadata.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        register.verify_metadata.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'register.verify_metadata', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'verify_metadata', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'verify_metadata(...)' code ##################

        str_24903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, (-1)), 'str', ' Send the metadata to the package index server to be checked.\n        ')
        
        # Assigning a Call to a Tuple (line 96):
        
        # Assigning a Subscript to a Name (line 96):
        
        # Obtaining the type of the subscript
        int_24904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'int')
        
        # Call to post_to_server(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to build_post_data(...): (line 96)
        # Processing the call arguments (line 96)
        str_24909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 66), 'str', 'verify')
        # Processing the call keyword arguments (line 96)
        kwargs_24910 = {}
        # Getting the type of 'self' (line 96)
        self_24907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 45), 'self', False)
        # Obtaining the member 'build_post_data' of a type (line 96)
        build_post_data_24908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 45), self_24907, 'build_post_data')
        # Calling build_post_data(args, kwargs) (line 96)
        build_post_data_call_result_24911 = invoke(stypy.reporting.localization.Localization(__file__, 96, 45), build_post_data_24908, *[str_24909], **kwargs_24910)
        
        # Processing the call keyword arguments (line 96)
        kwargs_24912 = {}
        # Getting the type of 'self' (line 96)
        self_24905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 25), 'self', False)
        # Obtaining the member 'post_to_server' of a type (line 96)
        post_to_server_24906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 25), self_24905, 'post_to_server')
        # Calling post_to_server(args, kwargs) (line 96)
        post_to_server_call_result_24913 = invoke(stypy.reporting.localization.Localization(__file__, 96, 25), post_to_server_24906, *[build_post_data_call_result_24911], **kwargs_24912)
        
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___24914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), post_to_server_call_result_24913, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_24915 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), getitem___24914, int_24904)
        
        # Assigning a type to the variable 'tuple_var_assignment_24721' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'tuple_var_assignment_24721', subscript_call_result_24915)
        
        # Assigning a Subscript to a Name (line 96):
        
        # Obtaining the type of the subscript
        int_24916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'int')
        
        # Call to post_to_server(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to build_post_data(...): (line 96)
        # Processing the call arguments (line 96)
        str_24921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 66), 'str', 'verify')
        # Processing the call keyword arguments (line 96)
        kwargs_24922 = {}
        # Getting the type of 'self' (line 96)
        self_24919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 45), 'self', False)
        # Obtaining the member 'build_post_data' of a type (line 96)
        build_post_data_24920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 45), self_24919, 'build_post_data')
        # Calling build_post_data(args, kwargs) (line 96)
        build_post_data_call_result_24923 = invoke(stypy.reporting.localization.Localization(__file__, 96, 45), build_post_data_24920, *[str_24921], **kwargs_24922)
        
        # Processing the call keyword arguments (line 96)
        kwargs_24924 = {}
        # Getting the type of 'self' (line 96)
        self_24917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 25), 'self', False)
        # Obtaining the member 'post_to_server' of a type (line 96)
        post_to_server_24918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 25), self_24917, 'post_to_server')
        # Calling post_to_server(args, kwargs) (line 96)
        post_to_server_call_result_24925 = invoke(stypy.reporting.localization.Localization(__file__, 96, 25), post_to_server_24918, *[build_post_data_call_result_24923], **kwargs_24924)
        
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___24926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), post_to_server_call_result_24925, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_24927 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), getitem___24926, int_24916)
        
        # Assigning a type to the variable 'tuple_var_assignment_24722' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'tuple_var_assignment_24722', subscript_call_result_24927)
        
        # Assigning a Name to a Name (line 96):
        # Getting the type of 'tuple_var_assignment_24721' (line 96)
        tuple_var_assignment_24721_24928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'tuple_var_assignment_24721')
        # Assigning a type to the variable 'code' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 9), 'code', tuple_var_assignment_24721_24928)
        
        # Assigning a Name to a Name (line 96):
        # Getting the type of 'tuple_var_assignment_24722' (line 96)
        tuple_var_assignment_24722_24929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'tuple_var_assignment_24722')
        # Assigning a type to the variable 'result' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'result', tuple_var_assignment_24722_24929)
        
        # Call to info(...): (line 97)
        # Processing the call arguments (line 97)
        str_24932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 17), 'str', 'Server response (%s): %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 97)
        tuple_24933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 97)
        # Adding element type (line 97)
        # Getting the type of 'code' (line 97)
        code_24934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 47), 'code', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 47), tuple_24933, code_24934)
        # Adding element type (line 97)
        # Getting the type of 'result' (line 97)
        result_24935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 53), 'result', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 47), tuple_24933, result_24935)
        
        # Applying the binary operator '%' (line 97)
        result_mod_24936 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 17), '%', str_24932, tuple_24933)
        
        # Processing the call keyword arguments (line 97)
        kwargs_24937 = {}
        # Getting the type of 'log' (line 97)
        log_24930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 97)
        info_24931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), log_24930, 'info')
        # Calling info(args, kwargs) (line 97)
        info_call_result_24938 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), info_24931, *[result_mod_24936], **kwargs_24937)
        
        
        # ################# End of 'verify_metadata(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'verify_metadata' in the type store
        # Getting the type of 'stypy_return_type' (line 92)
        stypy_return_type_24939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24939)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'verify_metadata'
        return stypy_return_type_24939


    @norecursion
    def send_metadata(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'send_metadata'
        module_type_store = module_type_store.open_function_context('send_metadata', 100, 4, False)
        # Assigning a type to the variable 'self' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        register.send_metadata.__dict__.__setitem__('stypy_localization', localization)
        register.send_metadata.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        register.send_metadata.__dict__.__setitem__('stypy_type_store', module_type_store)
        register.send_metadata.__dict__.__setitem__('stypy_function_name', 'register.send_metadata')
        register.send_metadata.__dict__.__setitem__('stypy_param_names_list', [])
        register.send_metadata.__dict__.__setitem__('stypy_varargs_param_name', None)
        register.send_metadata.__dict__.__setitem__('stypy_kwargs_param_name', None)
        register.send_metadata.__dict__.__setitem__('stypy_call_defaults', defaults)
        register.send_metadata.__dict__.__setitem__('stypy_call_varargs', varargs)
        register.send_metadata.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        register.send_metadata.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'register.send_metadata', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'send_metadata', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'send_metadata(...)' code ##################

        str_24940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, (-1)), 'str', " Send the metadata to the package index server.\n\n            Well, do the following:\n            1. figure who the user is, and then\n            2. send the data as a Basic auth'ed POST.\n\n            First we try to read the username/password from $HOME/.pypirc,\n            which is a ConfigParser-formatted file with a section\n            [distutils] containing username and password entries (both\n            in clear text). Eg:\n\n                [distutils]\n                index-servers =\n                    pypi\n\n                [pypi]\n                username: fred\n                password: sekrit\n\n            Otherwise, to figure who the user is, we offer the user three\n            choices:\n\n             1. use existing login,\n             2. register as a new user, or\n             3. set the password to a random string and email the user.\n\n        ")
        
        # Getting the type of 'self' (line 130)
        self_24941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), 'self')
        # Obtaining the member 'has_config' of a type (line 130)
        has_config_24942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 11), self_24941, 'has_config')
        # Testing the type of an if condition (line 130)
        if_condition_24943 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 8), has_config_24942)
        # Assigning a type to the variable 'if_condition_24943' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'if_condition_24943', if_condition_24943)
        # SSA begins for if statement (line 130)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 131):
        
        # Assigning a Str to a Name (line 131):
        str_24944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 21), 'str', '1')
        # Assigning a type to the variable 'choice' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'choice', str_24944)
        
        # Assigning a Attribute to a Name (line 132):
        
        # Assigning a Attribute to a Name (line 132):
        # Getting the type of 'self' (line 132)
        self_24945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 23), 'self')
        # Obtaining the member 'username' of a type (line 132)
        username_24946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 23), self_24945, 'username')
        # Assigning a type to the variable 'username' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'username', username_24946)
        
        # Assigning a Attribute to a Name (line 133):
        
        # Assigning a Attribute to a Name (line 133):
        # Getting the type of 'self' (line 133)
        self_24947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'self')
        # Obtaining the member 'password' of a type (line 133)
        password_24948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 23), self_24947, 'password')
        # Assigning a type to the variable 'password' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'password', password_24948)
        # SSA branch for the else part of an if statement (line 130)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 135):
        
        # Assigning a Str to a Name (line 135):
        str_24949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 21), 'str', 'x')
        # Assigning a type to the variable 'choice' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'choice', str_24949)
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Str to a Name (line 136):
        str_24950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 34), 'str', '')
        # Assigning a type to the variable 'password' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 23), 'password', str_24950)
        
        # Assigning a Name to a Name (line 136):
        # Getting the type of 'password' (line 136)
        password_24951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 23), 'password')
        # Assigning a type to the variable 'username' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'username', password_24951)
        # SSA join for if statement (line 130)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 139):
        
        # Assigning a Call to a Name (line 139):
        
        # Call to split(...): (line 139)
        # Processing the call keyword arguments (line 139)
        kwargs_24954 = {}
        str_24952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 18), 'str', '1 2 3 4')
        # Obtaining the member 'split' of a type (line 139)
        split_24953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 18), str_24952, 'split')
        # Calling split(args, kwargs) (line 139)
        split_call_result_24955 = invoke(stypy.reporting.localization.Localization(__file__, 139, 18), split_24953, *[], **kwargs_24954)
        
        # Assigning a type to the variable 'choices' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'choices', split_call_result_24955)
        
        
        # Getting the type of 'choice' (line 140)
        choice_24956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 14), 'choice')
        # Getting the type of 'choices' (line 140)
        choices_24957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 28), 'choices')
        # Applying the binary operator 'notin' (line 140)
        result_contains_24958 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 14), 'notin', choice_24956, choices_24957)
        
        # Testing the type of an if condition (line 140)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 8), result_contains_24958)
        # SSA begins for while statement (line 140)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Call to announce(...): (line 141)
        # Processing the call arguments (line 141)
        str_24961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, (-1)), 'str', 'We need to know who you are, so please choose either:\n 1. use your existing login,\n 2. register as a new user,\n 3. have the server generate a new password for you (and email it to you), or\n 4. quit\nYour selection [default 1]: ')
        # Getting the type of 'log' (line 147)
        log_24962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 33), 'log', False)
        # Obtaining the member 'INFO' of a type (line 147)
        INFO_24963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 33), log_24962, 'INFO')
        # Processing the call keyword arguments (line 141)
        kwargs_24964 = {}
        # Getting the type of 'self' (line 141)
        self_24959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'self', False)
        # Obtaining the member 'announce' of a type (line 141)
        announce_24960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 12), self_24959, 'announce')
        # Calling announce(args, kwargs) (line 141)
        announce_call_result_24965 = invoke(stypy.reporting.localization.Localization(__file__, 141, 12), announce_24960, *[str_24961, INFO_24963], **kwargs_24964)
        
        
        # Assigning a Call to a Name (line 149):
        
        # Assigning a Call to a Name (line 149):
        
        # Call to raw_input(...): (line 149)
        # Processing the call keyword arguments (line 149)
        kwargs_24967 = {}
        # Getting the type of 'raw_input' (line 149)
        raw_input_24966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 21), 'raw_input', False)
        # Calling raw_input(args, kwargs) (line 149)
        raw_input_call_result_24968 = invoke(stypy.reporting.localization.Localization(__file__, 149, 21), raw_input_24966, *[], **kwargs_24967)
        
        # Assigning a type to the variable 'choice' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'choice', raw_input_call_result_24968)
        
        
        # Getting the type of 'choice' (line 150)
        choice_24969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 19), 'choice')
        # Applying the 'not' unary operator (line 150)
        result_not__24970 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 15), 'not', choice_24969)
        
        # Testing the type of an if condition (line 150)
        if_condition_24971 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 12), result_not__24970)
        # Assigning a type to the variable 'if_condition_24971' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'if_condition_24971', if_condition_24971)
        # SSA begins for if statement (line 150)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 151):
        
        # Assigning a Str to a Name (line 151):
        str_24972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 25), 'str', '1')
        # Assigning a type to the variable 'choice' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'choice', str_24972)
        # SSA branch for the else part of an if statement (line 150)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'choice' (line 152)
        choice_24973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 17), 'choice')
        # Getting the type of 'choices' (line 152)
        choices_24974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 31), 'choices')
        # Applying the binary operator 'notin' (line 152)
        result_contains_24975 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 17), 'notin', choice_24973, choices_24974)
        
        # Testing the type of an if condition (line 152)
        if_condition_24976 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 17), result_contains_24975)
        # Assigning a type to the variable 'if_condition_24976' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 17), 'if_condition_24976', if_condition_24976)
        # SSA begins for if statement (line 152)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_24977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 22), 'str', 'Please choose one of the four options!')
        # SSA join for if statement (line 152)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 150)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 140)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'choice' (line 155)
        choice_24978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'choice')
        str_24979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 21), 'str', '1')
        # Applying the binary operator '==' (line 155)
        result_eq_24980 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 11), '==', choice_24978, str_24979)
        
        # Testing the type of an if condition (line 155)
        if_condition_24981 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 8), result_eq_24980)
        # Assigning a type to the variable 'if_condition_24981' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'if_condition_24981', if_condition_24981)
        # SSA begins for if statement (line 155)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'username' (line 157)
        username_24982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 22), 'username')
        # Applying the 'not' unary operator (line 157)
        result_not__24983 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 18), 'not', username_24982)
        
        # Testing the type of an if condition (line 157)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 12), result_not__24983)
        # SSA begins for while statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 158):
        
        # Assigning a Call to a Name (line 158):
        
        # Call to raw_input(...): (line 158)
        # Processing the call arguments (line 158)
        str_24985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 37), 'str', 'Username: ')
        # Processing the call keyword arguments (line 158)
        kwargs_24986 = {}
        # Getting the type of 'raw_input' (line 158)
        raw_input_24984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 27), 'raw_input', False)
        # Calling raw_input(args, kwargs) (line 158)
        raw_input_call_result_24987 = invoke(stypy.reporting.localization.Localization(__file__, 158, 27), raw_input_24984, *[str_24985], **kwargs_24986)
        
        # Assigning a type to the variable 'username' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'username', raw_input_call_result_24987)
        # SSA join for while statement (line 157)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'password' (line 159)
        password_24988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), 'password')
        # Applying the 'not' unary operator (line 159)
        result_not__24989 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 18), 'not', password_24988)
        
        # Testing the type of an if condition (line 159)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 12), result_not__24989)
        # SSA begins for while statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 160):
        
        # Assigning a Call to a Name (line 160):
        
        # Call to getpass(...): (line 160)
        # Processing the call arguments (line 160)
        str_24992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 43), 'str', 'Password: ')
        # Processing the call keyword arguments (line 160)
        kwargs_24993 = {}
        # Getting the type of 'getpass' (line 160)
        getpass_24990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 27), 'getpass', False)
        # Obtaining the member 'getpass' of a type (line 160)
        getpass_24991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 27), getpass_24990, 'getpass')
        # Calling getpass(args, kwargs) (line 160)
        getpass_call_result_24994 = invoke(stypy.reporting.localization.Localization(__file__, 160, 27), getpass_24991, *[str_24992], **kwargs_24993)
        
        # Assigning a type to the variable 'password' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'password', getpass_call_result_24994)
        # SSA join for while statement (line 159)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 163):
        
        # Assigning a Call to a Name (line 163):
        
        # Call to HTTPPasswordMgr(...): (line 163)
        # Processing the call keyword arguments (line 163)
        kwargs_24997 = {}
        # Getting the type of 'urllib2' (line 163)
        urllib2_24995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 19), 'urllib2', False)
        # Obtaining the member 'HTTPPasswordMgr' of a type (line 163)
        HTTPPasswordMgr_24996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 19), urllib2_24995, 'HTTPPasswordMgr')
        # Calling HTTPPasswordMgr(args, kwargs) (line 163)
        HTTPPasswordMgr_call_result_24998 = invoke(stypy.reporting.localization.Localization(__file__, 163, 19), HTTPPasswordMgr_24996, *[], **kwargs_24997)
        
        # Assigning a type to the variable 'auth' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'auth', HTTPPasswordMgr_call_result_24998)
        
        # Assigning a Subscript to a Name (line 164):
        
        # Assigning a Subscript to a Name (line 164):
        
        # Obtaining the type of the subscript
        int_24999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 54), 'int')
        
        # Call to urlparse(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'self' (line 164)
        self_25002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 37), 'self', False)
        # Obtaining the member 'repository' of a type (line 164)
        repository_25003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 37), self_25002, 'repository')
        # Processing the call keyword arguments (line 164)
        kwargs_25004 = {}
        # Getting the type of 'urlparse' (line 164)
        urlparse_25000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 19), 'urlparse', False)
        # Obtaining the member 'urlparse' of a type (line 164)
        urlparse_25001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 19), urlparse_25000, 'urlparse')
        # Calling urlparse(args, kwargs) (line 164)
        urlparse_call_result_25005 = invoke(stypy.reporting.localization.Localization(__file__, 164, 19), urlparse_25001, *[repository_25003], **kwargs_25004)
        
        # Obtaining the member '__getitem__' of a type (line 164)
        getitem___25006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 19), urlparse_call_result_25005, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 164)
        subscript_call_result_25007 = invoke(stypy.reporting.localization.Localization(__file__, 164, 19), getitem___25006, int_24999)
        
        # Assigning a type to the variable 'host' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'host', subscript_call_result_25007)
        
        # Call to add_password(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'self' (line 165)
        self_25010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 30), 'self', False)
        # Obtaining the member 'realm' of a type (line 165)
        realm_25011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 30), self_25010, 'realm')
        # Getting the type of 'host' (line 165)
        host_25012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 42), 'host', False)
        # Getting the type of 'username' (line 165)
        username_25013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 48), 'username', False)
        # Getting the type of 'password' (line 165)
        password_25014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 58), 'password', False)
        # Processing the call keyword arguments (line 165)
        kwargs_25015 = {}
        # Getting the type of 'auth' (line 165)
        auth_25008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'auth', False)
        # Obtaining the member 'add_password' of a type (line 165)
        add_password_25009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 12), auth_25008, 'add_password')
        # Calling add_password(args, kwargs) (line 165)
        add_password_call_result_25016 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), add_password_25009, *[realm_25011, host_25012, username_25013, password_25014], **kwargs_25015)
        
        
        # Assigning a Call to a Tuple (line 167):
        
        # Assigning a Subscript to a Name (line 167):
        
        # Obtaining the type of the subscript
        int_25017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 12), 'int')
        
        # Call to post_to_server(...): (line 167)
        # Processing the call arguments (line 167)
        
        # Call to build_post_data(...): (line 167)
        # Processing the call arguments (line 167)
        str_25022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 68), 'str', 'submit')
        # Processing the call keyword arguments (line 167)
        kwargs_25023 = {}
        # Getting the type of 'self' (line 167)
        self_25020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 47), 'self', False)
        # Obtaining the member 'build_post_data' of a type (line 167)
        build_post_data_25021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 47), self_25020, 'build_post_data')
        # Calling build_post_data(args, kwargs) (line 167)
        build_post_data_call_result_25024 = invoke(stypy.reporting.localization.Localization(__file__, 167, 47), build_post_data_25021, *[str_25022], **kwargs_25023)
        
        # Getting the type of 'auth' (line 168)
        auth_25025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'auth', False)
        # Processing the call keyword arguments (line 167)
        kwargs_25026 = {}
        # Getting the type of 'self' (line 167)
        self_25018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 27), 'self', False)
        # Obtaining the member 'post_to_server' of a type (line 167)
        post_to_server_25019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 27), self_25018, 'post_to_server')
        # Calling post_to_server(args, kwargs) (line 167)
        post_to_server_call_result_25027 = invoke(stypy.reporting.localization.Localization(__file__, 167, 27), post_to_server_25019, *[build_post_data_call_result_25024, auth_25025], **kwargs_25026)
        
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___25028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 12), post_to_server_call_result_25027, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_25029 = invoke(stypy.reporting.localization.Localization(__file__, 167, 12), getitem___25028, int_25017)
        
        # Assigning a type to the variable 'tuple_var_assignment_24723' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'tuple_var_assignment_24723', subscript_call_result_25029)
        
        # Assigning a Subscript to a Name (line 167):
        
        # Obtaining the type of the subscript
        int_25030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 12), 'int')
        
        # Call to post_to_server(...): (line 167)
        # Processing the call arguments (line 167)
        
        # Call to build_post_data(...): (line 167)
        # Processing the call arguments (line 167)
        str_25035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 68), 'str', 'submit')
        # Processing the call keyword arguments (line 167)
        kwargs_25036 = {}
        # Getting the type of 'self' (line 167)
        self_25033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 47), 'self', False)
        # Obtaining the member 'build_post_data' of a type (line 167)
        build_post_data_25034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 47), self_25033, 'build_post_data')
        # Calling build_post_data(args, kwargs) (line 167)
        build_post_data_call_result_25037 = invoke(stypy.reporting.localization.Localization(__file__, 167, 47), build_post_data_25034, *[str_25035], **kwargs_25036)
        
        # Getting the type of 'auth' (line 168)
        auth_25038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'auth', False)
        # Processing the call keyword arguments (line 167)
        kwargs_25039 = {}
        # Getting the type of 'self' (line 167)
        self_25031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 27), 'self', False)
        # Obtaining the member 'post_to_server' of a type (line 167)
        post_to_server_25032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 27), self_25031, 'post_to_server')
        # Calling post_to_server(args, kwargs) (line 167)
        post_to_server_call_result_25040 = invoke(stypy.reporting.localization.Localization(__file__, 167, 27), post_to_server_25032, *[build_post_data_call_result_25037, auth_25038], **kwargs_25039)
        
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___25041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 12), post_to_server_call_result_25040, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_25042 = invoke(stypy.reporting.localization.Localization(__file__, 167, 12), getitem___25041, int_25030)
        
        # Assigning a type to the variable 'tuple_var_assignment_24724' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'tuple_var_assignment_24724', subscript_call_result_25042)
        
        # Assigning a Name to a Name (line 167):
        # Getting the type of 'tuple_var_assignment_24723' (line 167)
        tuple_var_assignment_24723_25043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'tuple_var_assignment_24723')
        # Assigning a type to the variable 'code' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'code', tuple_var_assignment_24723_25043)
        
        # Assigning a Name to a Name (line 167):
        # Getting the type of 'tuple_var_assignment_24724' (line 167)
        tuple_var_assignment_24724_25044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'tuple_var_assignment_24724')
        # Assigning a type to the variable 'result' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'result', tuple_var_assignment_24724_25044)
        
        # Call to announce(...): (line 169)
        # Processing the call arguments (line 169)
        str_25047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 26), 'str', 'Server response (%s): %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 169)
        tuple_25048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 169)
        # Adding element type (line 169)
        # Getting the type of 'code' (line 169)
        code_25049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 56), 'code', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 56), tuple_25048, code_25049)
        # Adding element type (line 169)
        # Getting the type of 'result' (line 169)
        result_25050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 62), 'result', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 56), tuple_25048, result_25050)
        
        # Applying the binary operator '%' (line 169)
        result_mod_25051 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 26), '%', str_25047, tuple_25048)
        
        # Getting the type of 'log' (line 170)
        log_25052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 26), 'log', False)
        # Obtaining the member 'INFO' of a type (line 170)
        INFO_25053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 26), log_25052, 'INFO')
        # Processing the call keyword arguments (line 169)
        kwargs_25054 = {}
        # Getting the type of 'self' (line 169)
        self_25045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'self', False)
        # Obtaining the member 'announce' of a type (line 169)
        announce_25046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 12), self_25045, 'announce')
        # Calling announce(args, kwargs) (line 169)
        announce_call_result_25055 = invoke(stypy.reporting.localization.Localization(__file__, 169, 12), announce_25046, *[result_mod_25051, INFO_25053], **kwargs_25054)
        
        
        
        # Getting the type of 'code' (line 173)
        code_25056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'code')
        int_25057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 23), 'int')
        # Applying the binary operator '==' (line 173)
        result_eq_25058 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 15), '==', code_25056, int_25057)
        
        # Testing the type of an if condition (line 173)
        if_condition_25059 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 12), result_eq_25058)
        # Assigning a type to the variable 'if_condition_25059' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'if_condition_25059', if_condition_25059)
        # SSA begins for if statement (line 173)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 174)
        self_25060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 19), 'self')
        # Obtaining the member 'has_config' of a type (line 174)
        has_config_25061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 19), self_25060, 'has_config')
        # Testing the type of an if condition (line 174)
        if_condition_25062 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 16), has_config_25061)
        # Assigning a type to the variable 'if_condition_25062' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'if_condition_25062', if_condition_25062)
        # SSA begins for if statement (line 174)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 177):
        
        # Assigning a Name to a Attribute (line 177):
        # Getting the type of 'password' (line 177)
        password_25063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 49), 'password')
        # Getting the type of 'self' (line 177)
        self_25064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 20), 'self')
        # Obtaining the member 'distribution' of a type (line 177)
        distribution_25065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 20), self_25064, 'distribution')
        # Setting the type of the member 'password' of a type (line 177)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 20), distribution_25065, 'password', password_25063)
        # SSA branch for the else part of an if statement (line 174)
        module_type_store.open_ssa_branch('else')
        
        # Call to announce(...): (line 179)
        # Processing the call arguments (line 179)
        str_25068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 35), 'str', 'I can store your PyPI login so future submissions will be faster.')
        # Getting the type of 'log' (line 180)
        log_25069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 67), 'log', False)
        # Obtaining the member 'INFO' of a type (line 180)
        INFO_25070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 67), log_25069, 'INFO')
        # Processing the call keyword arguments (line 179)
        kwargs_25071 = {}
        # Getting the type of 'self' (line 179)
        self_25066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 20), 'self', False)
        # Obtaining the member 'announce' of a type (line 179)
        announce_25067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 20), self_25066, 'announce')
        # Calling announce(args, kwargs) (line 179)
        announce_call_result_25072 = invoke(stypy.reporting.localization.Localization(__file__, 179, 20), announce_25067, *[str_25068, INFO_25070], **kwargs_25071)
        
        
        # Call to announce(...): (line 181)
        # Processing the call arguments (line 181)
        str_25075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 34), 'str', '(the login will be stored in %s)')
        
        # Call to _get_rc_file(...): (line 182)
        # Processing the call keyword arguments (line 182)
        kwargs_25078 = {}
        # Getting the type of 'self' (line 182)
        self_25076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 34), 'self', False)
        # Obtaining the member '_get_rc_file' of a type (line 182)
        _get_rc_file_25077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 34), self_25076, '_get_rc_file')
        # Calling _get_rc_file(args, kwargs) (line 182)
        _get_rc_file_call_result_25079 = invoke(stypy.reporting.localization.Localization(__file__, 182, 34), _get_rc_file_25077, *[], **kwargs_25078)
        
        # Applying the binary operator '%' (line 181)
        result_mod_25080 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 34), '%', str_25075, _get_rc_file_call_result_25079)
        
        # Getting the type of 'log' (line 182)
        log_25081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 55), 'log', False)
        # Obtaining the member 'INFO' of a type (line 182)
        INFO_25082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 55), log_25081, 'INFO')
        # Processing the call keyword arguments (line 181)
        kwargs_25083 = {}
        # Getting the type of 'self' (line 181)
        self_25073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 20), 'self', False)
        # Obtaining the member 'announce' of a type (line 181)
        announce_25074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 20), self_25073, 'announce')
        # Calling announce(args, kwargs) (line 181)
        announce_call_result_25084 = invoke(stypy.reporting.localization.Localization(__file__, 181, 20), announce_25074, *[result_mod_25080, INFO_25082], **kwargs_25083)
        
        
        # Assigning a Str to a Name (line 183):
        
        # Assigning a Str to a Name (line 183):
        str_25085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 29), 'str', 'X')
        # Assigning a type to the variable 'choice' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 20), 'choice', str_25085)
        
        
        
        # Call to lower(...): (line 184)
        # Processing the call keyword arguments (line 184)
        kwargs_25088 = {}
        # Getting the type of 'choice' (line 184)
        choice_25086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 26), 'choice', False)
        # Obtaining the member 'lower' of a type (line 184)
        lower_25087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 26), choice_25086, 'lower')
        # Calling lower(args, kwargs) (line 184)
        lower_call_result_25089 = invoke(stypy.reporting.localization.Localization(__file__, 184, 26), lower_25087, *[], **kwargs_25088)
        
        str_25090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 48), 'str', 'yn')
        # Applying the binary operator 'notin' (line 184)
        result_contains_25091 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 26), 'notin', lower_call_result_25089, str_25090)
        
        # Testing the type of an if condition (line 184)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 20), result_contains_25091)
        # SSA begins for while statement (line 184)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 185):
        
        # Assigning a Call to a Name (line 185):
        
        # Call to raw_input(...): (line 185)
        # Processing the call arguments (line 185)
        str_25093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 43), 'str', 'Save your login (y/N)?')
        # Processing the call keyword arguments (line 185)
        kwargs_25094 = {}
        # Getting the type of 'raw_input' (line 185)
        raw_input_25092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 33), 'raw_input', False)
        # Calling raw_input(args, kwargs) (line 185)
        raw_input_call_result_25095 = invoke(stypy.reporting.localization.Localization(__file__, 185, 33), raw_input_25092, *[str_25093], **kwargs_25094)
        
        # Assigning a type to the variable 'choice' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 24), 'choice', raw_input_call_result_25095)
        
        
        # Getting the type of 'choice' (line 186)
        choice_25096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 31), 'choice')
        # Applying the 'not' unary operator (line 186)
        result_not__25097 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 27), 'not', choice_25096)
        
        # Testing the type of an if condition (line 186)
        if_condition_25098 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 24), result_not__25097)
        # Assigning a type to the variable 'if_condition_25098' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 24), 'if_condition_25098', if_condition_25098)
        # SSA begins for if statement (line 186)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 187):
        
        # Assigning a Str to a Name (line 187):
        str_25099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 37), 'str', 'n')
        # Assigning a type to the variable 'choice' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 28), 'choice', str_25099)
        # SSA join for if statement (line 186)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 184)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to lower(...): (line 188)
        # Processing the call keyword arguments (line 188)
        kwargs_25102 = {}
        # Getting the type of 'choice' (line 188)
        choice_25100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 23), 'choice', False)
        # Obtaining the member 'lower' of a type (line 188)
        lower_25101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 23), choice_25100, 'lower')
        # Calling lower(args, kwargs) (line 188)
        lower_call_result_25103 = invoke(stypy.reporting.localization.Localization(__file__, 188, 23), lower_25101, *[], **kwargs_25102)
        
        str_25104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 41), 'str', 'y')
        # Applying the binary operator '==' (line 188)
        result_eq_25105 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 23), '==', lower_call_result_25103, str_25104)
        
        # Testing the type of an if condition (line 188)
        if_condition_25106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 20), result_eq_25105)
        # Assigning a type to the variable 'if_condition_25106' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 20), 'if_condition_25106', if_condition_25106)
        # SSA begins for if statement (line 188)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _store_pypirc(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'username' (line 189)
        username_25109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 43), 'username', False)
        # Getting the type of 'password' (line 189)
        password_25110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 53), 'password', False)
        # Processing the call keyword arguments (line 189)
        kwargs_25111 = {}
        # Getting the type of 'self' (line 189)
        self_25107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 24), 'self', False)
        # Obtaining the member '_store_pypirc' of a type (line 189)
        _store_pypirc_25108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 24), self_25107, '_store_pypirc')
        # Calling _store_pypirc(args, kwargs) (line 189)
        _store_pypirc_call_result_25112 = invoke(stypy.reporting.localization.Localization(__file__, 189, 24), _store_pypirc_25108, *[username_25109, password_25110], **kwargs_25111)
        
        # SSA join for if statement (line 188)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 174)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 173)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 155)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'choice' (line 191)
        choice_25113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 13), 'choice')
        str_25114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 23), 'str', '2')
        # Applying the binary operator '==' (line 191)
        result_eq_25115 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 13), '==', choice_25113, str_25114)
        
        # Testing the type of an if condition (line 191)
        if_condition_25116 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 13), result_eq_25115)
        # Assigning a type to the variable 'if_condition_25116' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 13), 'if_condition_25116', if_condition_25116)
        # SSA begins for if statement (line 191)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Dict to a Name (line 192):
        
        # Assigning a Dict to a Name (line 192):
        
        # Obtaining an instance of the builtin type 'dict' (line 192)
        dict_25117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 19), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 192)
        # Adding element type (key, value) (line 192)
        str_25118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 20), 'str', ':action')
        str_25119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 31), 'str', 'user')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 19), dict_25117, (str_25118, str_25119))
        
        # Assigning a type to the variable 'data' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'data', dict_25117)
        
        # Multiple assignment of 3 elements.
        
        # Assigning a Str to a Subscript (line 193):
        str_25120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 62), 'str', '')
        # Getting the type of 'data' (line 193)
        data_25121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 46), 'data')
        str_25122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 51), 'str', 'email')
        # Storing an element on a container (line 193)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 46), data_25121, (str_25122, str_25120))
        
        # Assigning a Subscript to a Subscript (line 193):
        
        # Obtaining the type of the subscript
        str_25123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 51), 'str', 'email')
        # Getting the type of 'data' (line 193)
        data_25124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 46), 'data')
        # Obtaining the member '__getitem__' of a type (line 193)
        getitem___25125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 46), data_25124, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 193)
        subscript_call_result_25126 = invoke(stypy.reporting.localization.Localization(__file__, 193, 46), getitem___25125, str_25123)
        
        # Getting the type of 'data' (line 193)
        data_25127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 27), 'data')
        str_25128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 32), 'str', 'password')
        # Storing an element on a container (line 193)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 27), data_25127, (str_25128, subscript_call_result_25126))
        
        # Assigning a Subscript to a Subscript (line 193):
        
        # Obtaining the type of the subscript
        str_25129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 32), 'str', 'password')
        # Getting the type of 'data' (line 193)
        data_25130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 27), 'data')
        # Obtaining the member '__getitem__' of a type (line 193)
        getitem___25131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 27), data_25130, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 193)
        subscript_call_result_25132 = invoke(stypy.reporting.localization.Localization(__file__, 193, 27), getitem___25131, str_25129)
        
        # Getting the type of 'data' (line 193)
        data_25133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'data')
        str_25134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 17), 'str', 'name')
        # Storing an element on a container (line 193)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 12), data_25133, (str_25134, subscript_call_result_25132))
        
        # Assigning a Name to a Subscript (line 194):
        
        # Assigning a Name to a Subscript (line 194):
        # Getting the type of 'None' (line 194)
        None_25135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 30), 'None')
        # Getting the type of 'data' (line 194)
        data_25136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'data')
        str_25137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 17), 'str', 'confirm')
        # Storing an element on a container (line 194)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 12), data_25136, (str_25137, None_25135))
        
        
        
        # Obtaining the type of the subscript
        str_25138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 27), 'str', 'name')
        # Getting the type of 'data' (line 195)
        data_25139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 22), 'data')
        # Obtaining the member '__getitem__' of a type (line 195)
        getitem___25140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 22), data_25139, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 195)
        subscript_call_result_25141 = invoke(stypy.reporting.localization.Localization(__file__, 195, 22), getitem___25140, str_25138)
        
        # Applying the 'not' unary operator (line 195)
        result_not__25142 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 18), 'not', subscript_call_result_25141)
        
        # Testing the type of an if condition (line 195)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 12), result_not__25142)
        # SSA begins for while statement (line 195)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Subscript (line 196):
        
        # Assigning a Call to a Subscript (line 196):
        
        # Call to raw_input(...): (line 196)
        # Processing the call arguments (line 196)
        str_25144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 41), 'str', 'Username: ')
        # Processing the call keyword arguments (line 196)
        kwargs_25145 = {}
        # Getting the type of 'raw_input' (line 196)
        raw_input_25143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 31), 'raw_input', False)
        # Calling raw_input(args, kwargs) (line 196)
        raw_input_call_result_25146 = invoke(stypy.reporting.localization.Localization(__file__, 196, 31), raw_input_25143, *[str_25144], **kwargs_25145)
        
        # Getting the type of 'data' (line 196)
        data_25147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'data')
        str_25148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 21), 'str', 'name')
        # Storing an element on a container (line 196)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 16), data_25147, (str_25148, raw_input_call_result_25146))
        # SSA join for while statement (line 195)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        str_25149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 23), 'str', 'password')
        # Getting the type of 'data' (line 197)
        data_25150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 18), 'data')
        # Obtaining the member '__getitem__' of a type (line 197)
        getitem___25151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 18), data_25150, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 197)
        subscript_call_result_25152 = invoke(stypy.reporting.localization.Localization(__file__, 197, 18), getitem___25151, str_25149)
        
        
        # Obtaining the type of the subscript
        str_25153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 43), 'str', 'confirm')
        # Getting the type of 'data' (line 197)
        data_25154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 38), 'data')
        # Obtaining the member '__getitem__' of a type (line 197)
        getitem___25155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 38), data_25154, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 197)
        subscript_call_result_25156 = invoke(stypy.reporting.localization.Localization(__file__, 197, 38), getitem___25155, str_25153)
        
        # Applying the binary operator '!=' (line 197)
        result_ne_25157 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 18), '!=', subscript_call_result_25152, subscript_call_result_25156)
        
        # Testing the type of an if condition (line 197)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 12), result_ne_25157)
        # SSA begins for while statement (line 197)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        
        
        # Obtaining the type of the subscript
        str_25158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 31), 'str', 'password')
        # Getting the type of 'data' (line 198)
        data_25159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 26), 'data')
        # Obtaining the member '__getitem__' of a type (line 198)
        getitem___25160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 26), data_25159, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 198)
        subscript_call_result_25161 = invoke(stypy.reporting.localization.Localization(__file__, 198, 26), getitem___25160, str_25158)
        
        # Applying the 'not' unary operator (line 198)
        result_not__25162 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 22), 'not', subscript_call_result_25161)
        
        # Testing the type of an if condition (line 198)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 16), result_not__25162)
        # SSA begins for while statement (line 198)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Subscript (line 199):
        
        # Assigning a Call to a Subscript (line 199):
        
        # Call to getpass(...): (line 199)
        # Processing the call arguments (line 199)
        str_25165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 55), 'str', 'Password: ')
        # Processing the call keyword arguments (line 199)
        kwargs_25166 = {}
        # Getting the type of 'getpass' (line 199)
        getpass_25163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 39), 'getpass', False)
        # Obtaining the member 'getpass' of a type (line 199)
        getpass_25164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 39), getpass_25163, 'getpass')
        # Calling getpass(args, kwargs) (line 199)
        getpass_call_result_25167 = invoke(stypy.reporting.localization.Localization(__file__, 199, 39), getpass_25164, *[str_25165], **kwargs_25166)
        
        # Getting the type of 'data' (line 199)
        data_25168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'data')
        str_25169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 25), 'str', 'password')
        # Storing an element on a container (line 199)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 20), data_25168, (str_25169, getpass_call_result_25167))
        # SSA join for while statement (line 198)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        str_25170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 31), 'str', 'confirm')
        # Getting the type of 'data' (line 200)
        data_25171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 26), 'data')
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___25172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 26), data_25171, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_25173 = invoke(stypy.reporting.localization.Localization(__file__, 200, 26), getitem___25172, str_25170)
        
        # Applying the 'not' unary operator (line 200)
        result_not__25174 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 22), 'not', subscript_call_result_25173)
        
        # Testing the type of an if condition (line 200)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 16), result_not__25174)
        # SSA begins for while statement (line 200)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Subscript (line 201):
        
        # Assigning a Call to a Subscript (line 201):
        
        # Call to getpass(...): (line 201)
        # Processing the call arguments (line 201)
        str_25177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 54), 'str', ' Confirm: ')
        # Processing the call keyword arguments (line 201)
        kwargs_25178 = {}
        # Getting the type of 'getpass' (line 201)
        getpass_25175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 38), 'getpass', False)
        # Obtaining the member 'getpass' of a type (line 201)
        getpass_25176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 38), getpass_25175, 'getpass')
        # Calling getpass(args, kwargs) (line 201)
        getpass_call_result_25179 = invoke(stypy.reporting.localization.Localization(__file__, 201, 38), getpass_25176, *[str_25177], **kwargs_25178)
        
        # Getting the type of 'data' (line 201)
        data_25180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 20), 'data')
        str_25181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 25), 'str', 'confirm')
        # Storing an element on a container (line 201)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 20), data_25180, (str_25181, getpass_call_result_25179))
        # SSA join for while statement (line 200)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        str_25182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 24), 'str', 'password')
        # Getting the type of 'data' (line 202)
        data_25183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 19), 'data')
        # Obtaining the member '__getitem__' of a type (line 202)
        getitem___25184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 19), data_25183, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 202)
        subscript_call_result_25185 = invoke(stypy.reporting.localization.Localization(__file__, 202, 19), getitem___25184, str_25182)
        
        
        # Obtaining the type of the subscript
        str_25186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 44), 'str', 'confirm')
        # Getting the type of 'data' (line 202)
        data_25187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 39), 'data')
        # Obtaining the member '__getitem__' of a type (line 202)
        getitem___25188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 39), data_25187, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 202)
        subscript_call_result_25189 = invoke(stypy.reporting.localization.Localization(__file__, 202, 39), getitem___25188, str_25186)
        
        # Applying the binary operator '!=' (line 202)
        result_ne_25190 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 19), '!=', subscript_call_result_25185, subscript_call_result_25189)
        
        # Testing the type of an if condition (line 202)
        if_condition_25191 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 16), result_ne_25190)
        # Assigning a type to the variable 'if_condition_25191' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'if_condition_25191', if_condition_25191)
        # SSA begins for if statement (line 202)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Subscript (line 203):
        
        # Assigning a Str to a Subscript (line 203):
        str_25192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 39), 'str', '')
        # Getting the type of 'data' (line 203)
        data_25193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 20), 'data')
        str_25194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 25), 'str', 'password')
        # Storing an element on a container (line 203)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 20), data_25193, (str_25194, str_25192))
        
        # Assigning a Name to a Subscript (line 204):
        
        # Assigning a Name to a Subscript (line 204):
        # Getting the type of 'None' (line 204)
        None_25195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 38), 'None')
        # Getting the type of 'data' (line 204)
        data_25196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 20), 'data')
        str_25197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 25), 'str', 'confirm')
        # Storing an element on a container (line 204)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 20), data_25196, (str_25197, None_25195))
        str_25198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 26), 'str', "Password and confirm don't match!")
        # SSA join for if statement (line 202)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 197)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        str_25199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 27), 'str', 'email')
        # Getting the type of 'data' (line 206)
        data_25200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 22), 'data')
        # Obtaining the member '__getitem__' of a type (line 206)
        getitem___25201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 22), data_25200, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 206)
        subscript_call_result_25202 = invoke(stypy.reporting.localization.Localization(__file__, 206, 22), getitem___25201, str_25199)
        
        # Applying the 'not' unary operator (line 206)
        result_not__25203 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 18), 'not', subscript_call_result_25202)
        
        # Testing the type of an if condition (line 206)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 12), result_not__25203)
        # SSA begins for while statement (line 206)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Subscript (line 207):
        
        # Assigning a Call to a Subscript (line 207):
        
        # Call to raw_input(...): (line 207)
        # Processing the call arguments (line 207)
        str_25205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 42), 'str', '   EMail: ')
        # Processing the call keyword arguments (line 207)
        kwargs_25206 = {}
        # Getting the type of 'raw_input' (line 207)
        raw_input_25204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 32), 'raw_input', False)
        # Calling raw_input(args, kwargs) (line 207)
        raw_input_call_result_25207 = invoke(stypy.reporting.localization.Localization(__file__, 207, 32), raw_input_25204, *[str_25205], **kwargs_25206)
        
        # Getting the type of 'data' (line 207)
        data_25208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'data')
        str_25209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 21), 'str', 'email')
        # Storing an element on a container (line 207)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 16), data_25208, (str_25209, raw_input_call_result_25207))
        # SSA join for while statement (line 206)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 208):
        
        # Assigning a Subscript to a Name (line 208):
        
        # Obtaining the type of the subscript
        int_25210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 12), 'int')
        
        # Call to post_to_server(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'data' (line 208)
        data_25213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 47), 'data', False)
        # Processing the call keyword arguments (line 208)
        kwargs_25214 = {}
        # Getting the type of 'self' (line 208)
        self_25211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 27), 'self', False)
        # Obtaining the member 'post_to_server' of a type (line 208)
        post_to_server_25212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 27), self_25211, 'post_to_server')
        # Calling post_to_server(args, kwargs) (line 208)
        post_to_server_call_result_25215 = invoke(stypy.reporting.localization.Localization(__file__, 208, 27), post_to_server_25212, *[data_25213], **kwargs_25214)
        
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___25216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 12), post_to_server_call_result_25215, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_25217 = invoke(stypy.reporting.localization.Localization(__file__, 208, 12), getitem___25216, int_25210)
        
        # Assigning a type to the variable 'tuple_var_assignment_24725' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'tuple_var_assignment_24725', subscript_call_result_25217)
        
        # Assigning a Subscript to a Name (line 208):
        
        # Obtaining the type of the subscript
        int_25218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 12), 'int')
        
        # Call to post_to_server(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'data' (line 208)
        data_25221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 47), 'data', False)
        # Processing the call keyword arguments (line 208)
        kwargs_25222 = {}
        # Getting the type of 'self' (line 208)
        self_25219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 27), 'self', False)
        # Obtaining the member 'post_to_server' of a type (line 208)
        post_to_server_25220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 27), self_25219, 'post_to_server')
        # Calling post_to_server(args, kwargs) (line 208)
        post_to_server_call_result_25223 = invoke(stypy.reporting.localization.Localization(__file__, 208, 27), post_to_server_25220, *[data_25221], **kwargs_25222)
        
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___25224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 12), post_to_server_call_result_25223, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_25225 = invoke(stypy.reporting.localization.Localization(__file__, 208, 12), getitem___25224, int_25218)
        
        # Assigning a type to the variable 'tuple_var_assignment_24726' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'tuple_var_assignment_24726', subscript_call_result_25225)
        
        # Assigning a Name to a Name (line 208):
        # Getting the type of 'tuple_var_assignment_24725' (line 208)
        tuple_var_assignment_24725_25226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'tuple_var_assignment_24725')
        # Assigning a type to the variable 'code' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'code', tuple_var_assignment_24725_25226)
        
        # Assigning a Name to a Name (line 208):
        # Getting the type of 'tuple_var_assignment_24726' (line 208)
        tuple_var_assignment_24726_25227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'tuple_var_assignment_24726')
        # Assigning a type to the variable 'result' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 18), 'result', tuple_var_assignment_24726_25227)
        
        
        # Getting the type of 'code' (line 209)
        code_25228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), 'code')
        int_25229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 23), 'int')
        # Applying the binary operator '!=' (line 209)
        result_ne_25230 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 15), '!=', code_25228, int_25229)
        
        # Testing the type of an if condition (line 209)
        if_condition_25231 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 209, 12), result_ne_25230)
        # Assigning a type to the variable 'if_condition_25231' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'if_condition_25231', if_condition_25231)
        # SSA begins for if statement (line 209)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 210)
        # Processing the call arguments (line 210)
        str_25234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 25), 'str', 'Server response (%s): %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 210)
        tuple_25235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 210)
        # Adding element type (line 210)
        # Getting the type of 'code' (line 210)
        code_25236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 55), 'code', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 55), tuple_25235, code_25236)
        # Adding element type (line 210)
        # Getting the type of 'result' (line 210)
        result_25237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 61), 'result', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 55), tuple_25235, result_25237)
        
        # Applying the binary operator '%' (line 210)
        result_mod_25238 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 25), '%', str_25234, tuple_25235)
        
        # Processing the call keyword arguments (line 210)
        kwargs_25239 = {}
        # Getting the type of 'log' (line 210)
        log_25232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'log', False)
        # Obtaining the member 'info' of a type (line 210)
        info_25233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 16), log_25232, 'info')
        # Calling info(args, kwargs) (line 210)
        info_call_result_25240 = invoke(stypy.reporting.localization.Localization(__file__, 210, 16), info_25233, *[result_mod_25238], **kwargs_25239)
        
        # SSA branch for the else part of an if statement (line 209)
        module_type_store.open_ssa_branch('else')
        
        # Call to info(...): (line 212)
        # Processing the call arguments (line 212)
        str_25243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 25), 'str', 'You will receive an email shortly.')
        # Processing the call keyword arguments (line 212)
        kwargs_25244 = {}
        # Getting the type of 'log' (line 212)
        log_25241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 16), 'log', False)
        # Obtaining the member 'info' of a type (line 212)
        info_25242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 16), log_25241, 'info')
        # Calling info(args, kwargs) (line 212)
        info_call_result_25245 = invoke(stypy.reporting.localization.Localization(__file__, 212, 16), info_25242, *[str_25243], **kwargs_25244)
        
        
        # Call to info(...): (line 213)
        # Processing the call arguments (line 213)
        str_25248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 26), 'str', 'Follow the instructions in it to complete registration.')
        # Processing the call keyword arguments (line 213)
        kwargs_25249 = {}
        # Getting the type of 'log' (line 213)
        log_25246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'log', False)
        # Obtaining the member 'info' of a type (line 213)
        info_25247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 16), log_25246, 'info')
        # Calling info(args, kwargs) (line 213)
        info_call_result_25250 = invoke(stypy.reporting.localization.Localization(__file__, 213, 16), info_25247, *[str_25248], **kwargs_25249)
        
        # SSA join for if statement (line 209)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 191)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'choice' (line 215)
        choice_25251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 13), 'choice')
        str_25252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 23), 'str', '3')
        # Applying the binary operator '==' (line 215)
        result_eq_25253 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 13), '==', choice_25251, str_25252)
        
        # Testing the type of an if condition (line 215)
        if_condition_25254 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 13), result_eq_25253)
        # Assigning a type to the variable 'if_condition_25254' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 13), 'if_condition_25254', if_condition_25254)
        # SSA begins for if statement (line 215)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Dict to a Name (line 216):
        
        # Assigning a Dict to a Name (line 216):
        
        # Obtaining an instance of the builtin type 'dict' (line 216)
        dict_25255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 19), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 216)
        # Adding element type (key, value) (line 216)
        str_25256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 20), 'str', ':action')
        str_25257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 31), 'str', 'password_reset')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 19), dict_25255, (str_25256, str_25257))
        
        # Assigning a type to the variable 'data' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'data', dict_25255)
        
        # Assigning a Str to a Subscript (line 217):
        
        # Assigning a Str to a Subscript (line 217):
        str_25258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 28), 'str', '')
        # Getting the type of 'data' (line 217)
        data_25259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'data')
        str_25260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 17), 'str', 'email')
        # Storing an element on a container (line 217)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 12), data_25259, (str_25260, str_25258))
        
        
        
        # Obtaining the type of the subscript
        str_25261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 27), 'str', 'email')
        # Getting the type of 'data' (line 218)
        data_25262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 22), 'data')
        # Obtaining the member '__getitem__' of a type (line 218)
        getitem___25263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 22), data_25262, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 218)
        subscript_call_result_25264 = invoke(stypy.reporting.localization.Localization(__file__, 218, 22), getitem___25263, str_25261)
        
        # Applying the 'not' unary operator (line 218)
        result_not__25265 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 18), 'not', subscript_call_result_25264)
        
        # Testing the type of an if condition (line 218)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 218, 12), result_not__25265)
        # SSA begins for while statement (line 218)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Subscript (line 219):
        
        # Assigning a Call to a Subscript (line 219):
        
        # Call to raw_input(...): (line 219)
        # Processing the call arguments (line 219)
        str_25267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 42), 'str', 'Your email address: ')
        # Processing the call keyword arguments (line 219)
        kwargs_25268 = {}
        # Getting the type of 'raw_input' (line 219)
        raw_input_25266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 32), 'raw_input', False)
        # Calling raw_input(args, kwargs) (line 219)
        raw_input_call_result_25269 = invoke(stypy.reporting.localization.Localization(__file__, 219, 32), raw_input_25266, *[str_25267], **kwargs_25268)
        
        # Getting the type of 'data' (line 219)
        data_25270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'data')
        str_25271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 21), 'str', 'email')
        # Storing an element on a container (line 219)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 16), data_25270, (str_25271, raw_input_call_result_25269))
        # SSA join for while statement (line 218)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 220):
        
        # Assigning a Subscript to a Name (line 220):
        
        # Obtaining the type of the subscript
        int_25272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 12), 'int')
        
        # Call to post_to_server(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'data' (line 220)
        data_25275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 47), 'data', False)
        # Processing the call keyword arguments (line 220)
        kwargs_25276 = {}
        # Getting the type of 'self' (line 220)
        self_25273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 27), 'self', False)
        # Obtaining the member 'post_to_server' of a type (line 220)
        post_to_server_25274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 27), self_25273, 'post_to_server')
        # Calling post_to_server(args, kwargs) (line 220)
        post_to_server_call_result_25277 = invoke(stypy.reporting.localization.Localization(__file__, 220, 27), post_to_server_25274, *[data_25275], **kwargs_25276)
        
        # Obtaining the member '__getitem__' of a type (line 220)
        getitem___25278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 12), post_to_server_call_result_25277, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 220)
        subscript_call_result_25279 = invoke(stypy.reporting.localization.Localization(__file__, 220, 12), getitem___25278, int_25272)
        
        # Assigning a type to the variable 'tuple_var_assignment_24727' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'tuple_var_assignment_24727', subscript_call_result_25279)
        
        # Assigning a Subscript to a Name (line 220):
        
        # Obtaining the type of the subscript
        int_25280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 12), 'int')
        
        # Call to post_to_server(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'data' (line 220)
        data_25283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 47), 'data', False)
        # Processing the call keyword arguments (line 220)
        kwargs_25284 = {}
        # Getting the type of 'self' (line 220)
        self_25281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 27), 'self', False)
        # Obtaining the member 'post_to_server' of a type (line 220)
        post_to_server_25282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 27), self_25281, 'post_to_server')
        # Calling post_to_server(args, kwargs) (line 220)
        post_to_server_call_result_25285 = invoke(stypy.reporting.localization.Localization(__file__, 220, 27), post_to_server_25282, *[data_25283], **kwargs_25284)
        
        # Obtaining the member '__getitem__' of a type (line 220)
        getitem___25286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 12), post_to_server_call_result_25285, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 220)
        subscript_call_result_25287 = invoke(stypy.reporting.localization.Localization(__file__, 220, 12), getitem___25286, int_25280)
        
        # Assigning a type to the variable 'tuple_var_assignment_24728' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'tuple_var_assignment_24728', subscript_call_result_25287)
        
        # Assigning a Name to a Name (line 220):
        # Getting the type of 'tuple_var_assignment_24727' (line 220)
        tuple_var_assignment_24727_25288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'tuple_var_assignment_24727')
        # Assigning a type to the variable 'code' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'code', tuple_var_assignment_24727_25288)
        
        # Assigning a Name to a Name (line 220):
        # Getting the type of 'tuple_var_assignment_24728' (line 220)
        tuple_var_assignment_24728_25289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'tuple_var_assignment_24728')
        # Assigning a type to the variable 'result' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 18), 'result', tuple_var_assignment_24728_25289)
        
        # Call to info(...): (line 221)
        # Processing the call arguments (line 221)
        str_25292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 21), 'str', 'Server response (%s): %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 221)
        tuple_25293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 51), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 221)
        # Adding element type (line 221)
        # Getting the type of 'code' (line 221)
        code_25294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 51), 'code', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 51), tuple_25293, code_25294)
        # Adding element type (line 221)
        # Getting the type of 'result' (line 221)
        result_25295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 57), 'result', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 51), tuple_25293, result_25295)
        
        # Applying the binary operator '%' (line 221)
        result_mod_25296 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 21), '%', str_25292, tuple_25293)
        
        # Processing the call keyword arguments (line 221)
        kwargs_25297 = {}
        # Getting the type of 'log' (line 221)
        log_25290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 221)
        info_25291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 12), log_25290, 'info')
        # Calling info(args, kwargs) (line 221)
        info_call_result_25298 = invoke(stypy.reporting.localization.Localization(__file__, 221, 12), info_25291, *[result_mod_25296], **kwargs_25297)
        
        # SSA join for if statement (line 215)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 191)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 155)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'send_metadata(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'send_metadata' in the type store
        # Getting the type of 'stypy_return_type' (line 100)
        stypy_return_type_25299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25299)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'send_metadata'
        return stypy_return_type_25299


    @norecursion
    def build_post_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'build_post_data'
        module_type_store = module_type_store.open_function_context('build_post_data', 223, 4, False)
        # Assigning a type to the variable 'self' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        register.build_post_data.__dict__.__setitem__('stypy_localization', localization)
        register.build_post_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        register.build_post_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        register.build_post_data.__dict__.__setitem__('stypy_function_name', 'register.build_post_data')
        register.build_post_data.__dict__.__setitem__('stypy_param_names_list', ['action'])
        register.build_post_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        register.build_post_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        register.build_post_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        register.build_post_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        register.build_post_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        register.build_post_data.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'register.build_post_data', ['action'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'build_post_data', localization, ['action'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'build_post_data(...)' code ##################

        
        # Assigning a Attribute to a Name (line 226):
        
        # Assigning a Attribute to a Name (line 226):
        # Getting the type of 'self' (line 226)
        self_25300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 15), 'self')
        # Obtaining the member 'distribution' of a type (line 226)
        distribution_25301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 15), self_25300, 'distribution')
        # Obtaining the member 'metadata' of a type (line 226)
        metadata_25302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 15), distribution_25301, 'metadata')
        # Assigning a type to the variable 'meta' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'meta', metadata_25302)
        
        # Assigning a Dict to a Name (line 227):
        
        # Assigning a Dict to a Name (line 227):
        
        # Obtaining an instance of the builtin type 'dict' (line 227)
        dict_25303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 15), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 227)
        # Adding element type (key, value) (line 227)
        str_25304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 12), 'str', ':action')
        # Getting the type of 'action' (line 228)
        action_25305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 23), 'action')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 15), dict_25303, (str_25304, action_25305))
        # Adding element type (key, value) (line 227)
        str_25306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 12), 'str', 'metadata_version')
        str_25307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 33), 'str', '1.0')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 15), dict_25303, (str_25306, str_25307))
        # Adding element type (key, value) (line 227)
        str_25308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 12), 'str', 'name')
        
        # Call to get_name(...): (line 230)
        # Processing the call keyword arguments (line 230)
        kwargs_25311 = {}
        # Getting the type of 'meta' (line 230)
        meta_25309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 20), 'meta', False)
        # Obtaining the member 'get_name' of a type (line 230)
        get_name_25310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 20), meta_25309, 'get_name')
        # Calling get_name(args, kwargs) (line 230)
        get_name_call_result_25312 = invoke(stypy.reporting.localization.Localization(__file__, 230, 20), get_name_25310, *[], **kwargs_25311)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 15), dict_25303, (str_25308, get_name_call_result_25312))
        # Adding element type (key, value) (line 227)
        str_25313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 12), 'str', 'version')
        
        # Call to get_version(...): (line 231)
        # Processing the call keyword arguments (line 231)
        kwargs_25316 = {}
        # Getting the type of 'meta' (line 231)
        meta_25314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 23), 'meta', False)
        # Obtaining the member 'get_version' of a type (line 231)
        get_version_25315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 23), meta_25314, 'get_version')
        # Calling get_version(args, kwargs) (line 231)
        get_version_call_result_25317 = invoke(stypy.reporting.localization.Localization(__file__, 231, 23), get_version_25315, *[], **kwargs_25316)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 15), dict_25303, (str_25313, get_version_call_result_25317))
        # Adding element type (key, value) (line 227)
        str_25318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 12), 'str', 'summary')
        
        # Call to get_description(...): (line 232)
        # Processing the call keyword arguments (line 232)
        kwargs_25321 = {}
        # Getting the type of 'meta' (line 232)
        meta_25319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 23), 'meta', False)
        # Obtaining the member 'get_description' of a type (line 232)
        get_description_25320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 23), meta_25319, 'get_description')
        # Calling get_description(args, kwargs) (line 232)
        get_description_call_result_25322 = invoke(stypy.reporting.localization.Localization(__file__, 232, 23), get_description_25320, *[], **kwargs_25321)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 15), dict_25303, (str_25318, get_description_call_result_25322))
        # Adding element type (key, value) (line 227)
        str_25323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 12), 'str', 'home_page')
        
        # Call to get_url(...): (line 233)
        # Processing the call keyword arguments (line 233)
        kwargs_25326 = {}
        # Getting the type of 'meta' (line 233)
        meta_25324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 25), 'meta', False)
        # Obtaining the member 'get_url' of a type (line 233)
        get_url_25325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 25), meta_25324, 'get_url')
        # Calling get_url(args, kwargs) (line 233)
        get_url_call_result_25327 = invoke(stypy.reporting.localization.Localization(__file__, 233, 25), get_url_25325, *[], **kwargs_25326)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 15), dict_25303, (str_25323, get_url_call_result_25327))
        # Adding element type (key, value) (line 227)
        str_25328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 12), 'str', 'author')
        
        # Call to get_contact(...): (line 234)
        # Processing the call keyword arguments (line 234)
        kwargs_25331 = {}
        # Getting the type of 'meta' (line 234)
        meta_25329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 22), 'meta', False)
        # Obtaining the member 'get_contact' of a type (line 234)
        get_contact_25330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 22), meta_25329, 'get_contact')
        # Calling get_contact(args, kwargs) (line 234)
        get_contact_call_result_25332 = invoke(stypy.reporting.localization.Localization(__file__, 234, 22), get_contact_25330, *[], **kwargs_25331)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 15), dict_25303, (str_25328, get_contact_call_result_25332))
        # Adding element type (key, value) (line 227)
        str_25333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 12), 'str', 'author_email')
        
        # Call to get_contact_email(...): (line 235)
        # Processing the call keyword arguments (line 235)
        kwargs_25336 = {}
        # Getting the type of 'meta' (line 235)
        meta_25334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 28), 'meta', False)
        # Obtaining the member 'get_contact_email' of a type (line 235)
        get_contact_email_25335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 28), meta_25334, 'get_contact_email')
        # Calling get_contact_email(args, kwargs) (line 235)
        get_contact_email_call_result_25337 = invoke(stypy.reporting.localization.Localization(__file__, 235, 28), get_contact_email_25335, *[], **kwargs_25336)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 15), dict_25303, (str_25333, get_contact_email_call_result_25337))
        # Adding element type (key, value) (line 227)
        str_25338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 12), 'str', 'license')
        
        # Call to get_licence(...): (line 236)
        # Processing the call keyword arguments (line 236)
        kwargs_25341 = {}
        # Getting the type of 'meta' (line 236)
        meta_25339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 23), 'meta', False)
        # Obtaining the member 'get_licence' of a type (line 236)
        get_licence_25340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 23), meta_25339, 'get_licence')
        # Calling get_licence(args, kwargs) (line 236)
        get_licence_call_result_25342 = invoke(stypy.reporting.localization.Localization(__file__, 236, 23), get_licence_25340, *[], **kwargs_25341)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 15), dict_25303, (str_25338, get_licence_call_result_25342))
        # Adding element type (key, value) (line 227)
        str_25343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 12), 'str', 'description')
        
        # Call to get_long_description(...): (line 237)
        # Processing the call keyword arguments (line 237)
        kwargs_25346 = {}
        # Getting the type of 'meta' (line 237)
        meta_25344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 27), 'meta', False)
        # Obtaining the member 'get_long_description' of a type (line 237)
        get_long_description_25345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 27), meta_25344, 'get_long_description')
        # Calling get_long_description(args, kwargs) (line 237)
        get_long_description_call_result_25347 = invoke(stypy.reporting.localization.Localization(__file__, 237, 27), get_long_description_25345, *[], **kwargs_25346)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 15), dict_25303, (str_25343, get_long_description_call_result_25347))
        # Adding element type (key, value) (line 227)
        str_25348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 12), 'str', 'keywords')
        
        # Call to get_keywords(...): (line 238)
        # Processing the call keyword arguments (line 238)
        kwargs_25351 = {}
        # Getting the type of 'meta' (line 238)
        meta_25349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 24), 'meta', False)
        # Obtaining the member 'get_keywords' of a type (line 238)
        get_keywords_25350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 24), meta_25349, 'get_keywords')
        # Calling get_keywords(args, kwargs) (line 238)
        get_keywords_call_result_25352 = invoke(stypy.reporting.localization.Localization(__file__, 238, 24), get_keywords_25350, *[], **kwargs_25351)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 15), dict_25303, (str_25348, get_keywords_call_result_25352))
        # Adding element type (key, value) (line 227)
        str_25353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 12), 'str', 'platform')
        
        # Call to get_platforms(...): (line 239)
        # Processing the call keyword arguments (line 239)
        kwargs_25356 = {}
        # Getting the type of 'meta' (line 239)
        meta_25354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 24), 'meta', False)
        # Obtaining the member 'get_platforms' of a type (line 239)
        get_platforms_25355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 24), meta_25354, 'get_platforms')
        # Calling get_platforms(args, kwargs) (line 239)
        get_platforms_call_result_25357 = invoke(stypy.reporting.localization.Localization(__file__, 239, 24), get_platforms_25355, *[], **kwargs_25356)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 15), dict_25303, (str_25353, get_platforms_call_result_25357))
        # Adding element type (key, value) (line 227)
        str_25358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 12), 'str', 'classifiers')
        
        # Call to get_classifiers(...): (line 240)
        # Processing the call keyword arguments (line 240)
        kwargs_25361 = {}
        # Getting the type of 'meta' (line 240)
        meta_25359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 27), 'meta', False)
        # Obtaining the member 'get_classifiers' of a type (line 240)
        get_classifiers_25360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 27), meta_25359, 'get_classifiers')
        # Calling get_classifiers(args, kwargs) (line 240)
        get_classifiers_call_result_25362 = invoke(stypy.reporting.localization.Localization(__file__, 240, 27), get_classifiers_25360, *[], **kwargs_25361)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 15), dict_25303, (str_25358, get_classifiers_call_result_25362))
        # Adding element type (key, value) (line 227)
        str_25363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 12), 'str', 'download_url')
        
        # Call to get_download_url(...): (line 241)
        # Processing the call keyword arguments (line 241)
        kwargs_25366 = {}
        # Getting the type of 'meta' (line 241)
        meta_25364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 28), 'meta', False)
        # Obtaining the member 'get_download_url' of a type (line 241)
        get_download_url_25365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 28), meta_25364, 'get_download_url')
        # Calling get_download_url(args, kwargs) (line 241)
        get_download_url_call_result_25367 = invoke(stypy.reporting.localization.Localization(__file__, 241, 28), get_download_url_25365, *[], **kwargs_25366)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 15), dict_25303, (str_25363, get_download_url_call_result_25367))
        # Adding element type (key, value) (line 227)
        str_25368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 12), 'str', 'provides')
        
        # Call to get_provides(...): (line 243)
        # Processing the call keyword arguments (line 243)
        kwargs_25371 = {}
        # Getting the type of 'meta' (line 243)
        meta_25369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 24), 'meta', False)
        # Obtaining the member 'get_provides' of a type (line 243)
        get_provides_25370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 24), meta_25369, 'get_provides')
        # Calling get_provides(args, kwargs) (line 243)
        get_provides_call_result_25372 = invoke(stypy.reporting.localization.Localization(__file__, 243, 24), get_provides_25370, *[], **kwargs_25371)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 15), dict_25303, (str_25368, get_provides_call_result_25372))
        # Adding element type (key, value) (line 227)
        str_25373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 12), 'str', 'requires')
        
        # Call to get_requires(...): (line 244)
        # Processing the call keyword arguments (line 244)
        kwargs_25376 = {}
        # Getting the type of 'meta' (line 244)
        meta_25374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'meta', False)
        # Obtaining the member 'get_requires' of a type (line 244)
        get_requires_25375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 24), meta_25374, 'get_requires')
        # Calling get_requires(args, kwargs) (line 244)
        get_requires_call_result_25377 = invoke(stypy.reporting.localization.Localization(__file__, 244, 24), get_requires_25375, *[], **kwargs_25376)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 15), dict_25303, (str_25373, get_requires_call_result_25377))
        # Adding element type (key, value) (line 227)
        str_25378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 12), 'str', 'obsoletes')
        
        # Call to get_obsoletes(...): (line 245)
        # Processing the call keyword arguments (line 245)
        kwargs_25381 = {}
        # Getting the type of 'meta' (line 245)
        meta_25379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 25), 'meta', False)
        # Obtaining the member 'get_obsoletes' of a type (line 245)
        get_obsoletes_25380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 25), meta_25379, 'get_obsoletes')
        # Calling get_obsoletes(args, kwargs) (line 245)
        get_obsoletes_call_result_25382 = invoke(stypy.reporting.localization.Localization(__file__, 245, 25), get_obsoletes_25380, *[], **kwargs_25381)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 15), dict_25303, (str_25378, get_obsoletes_call_result_25382))
        
        # Assigning a type to the variable 'data' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'data', dict_25303)
        
        
        # Evaluating a boolean operation
        
        # Obtaining the type of the subscript
        str_25383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 16), 'str', 'provides')
        # Getting the type of 'data' (line 247)
        data_25384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 11), 'data')
        # Obtaining the member '__getitem__' of a type (line 247)
        getitem___25385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 11), data_25384, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 247)
        subscript_call_result_25386 = invoke(stypy.reporting.localization.Localization(__file__, 247, 11), getitem___25385, str_25383)
        
        
        # Obtaining the type of the subscript
        str_25387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 36), 'str', 'requires')
        # Getting the type of 'data' (line 247)
        data_25388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 31), 'data')
        # Obtaining the member '__getitem__' of a type (line 247)
        getitem___25389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 31), data_25388, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 247)
        subscript_call_result_25390 = invoke(stypy.reporting.localization.Localization(__file__, 247, 31), getitem___25389, str_25387)
        
        # Applying the binary operator 'or' (line 247)
        result_or_keyword_25391 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 11), 'or', subscript_call_result_25386, subscript_call_result_25390)
        
        # Obtaining the type of the subscript
        str_25392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 56), 'str', 'obsoletes')
        # Getting the type of 'data' (line 247)
        data_25393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 51), 'data')
        # Obtaining the member '__getitem__' of a type (line 247)
        getitem___25394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 51), data_25393, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 247)
        subscript_call_result_25395 = invoke(stypy.reporting.localization.Localization(__file__, 247, 51), getitem___25394, str_25392)
        
        # Applying the binary operator 'or' (line 247)
        result_or_keyword_25396 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 11), 'or', result_or_keyword_25391, subscript_call_result_25395)
        
        # Testing the type of an if condition (line 247)
        if_condition_25397 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 247, 8), result_or_keyword_25396)
        # Assigning a type to the variable 'if_condition_25397' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'if_condition_25397', if_condition_25397)
        # SSA begins for if statement (line 247)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Subscript (line 248):
        
        # Assigning a Str to a Subscript (line 248):
        str_25398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 39), 'str', '1.1')
        # Getting the type of 'data' (line 248)
        data_25399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'data')
        str_25400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 17), 'str', 'metadata_version')
        # Storing an element on a container (line 248)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 12), data_25399, (str_25400, str_25398))
        # SSA join for if statement (line 247)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'data' (line 249)
        data_25401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 15), 'data')
        # Assigning a type to the variable 'stypy_return_type' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'stypy_return_type', data_25401)
        
        # ################# End of 'build_post_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'build_post_data' in the type store
        # Getting the type of 'stypy_return_type' (line 223)
        stypy_return_type_25402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25402)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'build_post_data'
        return stypy_return_type_25402


    @norecursion
    def post_to_server(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 251)
        None_25403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 40), 'None')
        defaults = [None_25403]
        # Create a new context for function 'post_to_server'
        module_type_store = module_type_store.open_function_context('post_to_server', 251, 4, False)
        # Assigning a type to the variable 'self' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        register.post_to_server.__dict__.__setitem__('stypy_localization', localization)
        register.post_to_server.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        register.post_to_server.__dict__.__setitem__('stypy_type_store', module_type_store)
        register.post_to_server.__dict__.__setitem__('stypy_function_name', 'register.post_to_server')
        register.post_to_server.__dict__.__setitem__('stypy_param_names_list', ['data', 'auth'])
        register.post_to_server.__dict__.__setitem__('stypy_varargs_param_name', None)
        register.post_to_server.__dict__.__setitem__('stypy_kwargs_param_name', None)
        register.post_to_server.__dict__.__setitem__('stypy_call_defaults', defaults)
        register.post_to_server.__dict__.__setitem__('stypy_call_varargs', varargs)
        register.post_to_server.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        register.post_to_server.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'register.post_to_server', ['data', 'auth'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'post_to_server', localization, ['data', 'auth'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'post_to_server(...)' code ##################

        str_25404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, (-1)), 'str', ' Post a query to the server, and return a string response.\n        ')
        
        
        str_25405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 11), 'str', 'name')
        # Getting the type of 'data' (line 254)
        data_25406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 21), 'data')
        # Applying the binary operator 'in' (line 254)
        result_contains_25407 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 11), 'in', str_25405, data_25406)
        
        # Testing the type of an if condition (line 254)
        if_condition_25408 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 254, 8), result_contains_25407)
        # Assigning a type to the variable 'if_condition_25408' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'if_condition_25408', if_condition_25408)
        # SSA begins for if statement (line 254)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to announce(...): (line 255)
        # Processing the call arguments (line 255)
        str_25411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 26), 'str', 'Registering %s to %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 255)
        tuple_25412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 255)
        # Adding element type (line 255)
        
        # Obtaining the type of the subscript
        str_25413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 57), 'str', 'name')
        # Getting the type of 'data' (line 255)
        data_25414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 52), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 255)
        getitem___25415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 52), data_25414, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 255)
        subscript_call_result_25416 = invoke(stypy.reporting.localization.Localization(__file__, 255, 52), getitem___25415, str_25413)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 52), tuple_25412, subscript_call_result_25416)
        # Adding element type (line 255)
        # Getting the type of 'self' (line 256)
        self_25417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 51), 'self', False)
        # Obtaining the member 'repository' of a type (line 256)
        repository_25418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 51), self_25417, 'repository')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 52), tuple_25412, repository_25418)
        
        # Applying the binary operator '%' (line 255)
        result_mod_25419 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 26), '%', str_25411, tuple_25412)
        
        # Getting the type of 'log' (line 257)
        log_25420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 51), 'log', False)
        # Obtaining the member 'INFO' of a type (line 257)
        INFO_25421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 51), log_25420, 'INFO')
        # Processing the call keyword arguments (line 255)
        kwargs_25422 = {}
        # Getting the type of 'self' (line 255)
        self_25409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'self', False)
        # Obtaining the member 'announce' of a type (line 255)
        announce_25410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 12), self_25409, 'announce')
        # Calling announce(args, kwargs) (line 255)
        announce_call_result_25423 = invoke(stypy.reporting.localization.Localization(__file__, 255, 12), announce_25410, *[result_mod_25419, INFO_25421], **kwargs_25422)
        
        # SSA join for if statement (line 254)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Str to a Name (line 259):
        
        # Assigning a Str to a Name (line 259):
        str_25424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 19), 'str', '--------------GHSKFJDLGDS7543FJKLFHRE75642756743254')
        # Assigning a type to the variable 'boundary' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'boundary', str_25424)
        
        # Assigning a BinOp to a Name (line 260):
        
        # Assigning a BinOp to a Name (line 260):
        str_25425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 23), 'str', '\n--')
        # Getting the type of 'boundary' (line 260)
        boundary_25426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 32), 'boundary')
        # Applying the binary operator '+' (line 260)
        result_add_25427 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 23), '+', str_25425, boundary_25426)
        
        # Assigning a type to the variable 'sep_boundary' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'sep_boundary', result_add_25427)
        
        # Assigning a BinOp to a Name (line 261):
        
        # Assigning a BinOp to a Name (line 261):
        # Getting the type of 'sep_boundary' (line 261)
        sep_boundary_25428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 23), 'sep_boundary')
        str_25429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 38), 'str', '--')
        # Applying the binary operator '+' (line 261)
        result_add_25430 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 23), '+', sep_boundary_25428, str_25429)
        
        # Assigning a type to the variable 'end_boundary' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'end_boundary', result_add_25430)
        
        # Assigning a List to a Name (line 262):
        
        # Assigning a List to a Name (line 262):
        
        # Obtaining an instance of the builtin type 'list' (line 262)
        list_25431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 262)
        
        # Assigning a type to the variable 'chunks' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'chunks', list_25431)
        
        
        # Call to items(...): (line 263)
        # Processing the call keyword arguments (line 263)
        kwargs_25434 = {}
        # Getting the type of 'data' (line 263)
        data_25432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 26), 'data', False)
        # Obtaining the member 'items' of a type (line 263)
        items_25433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 26), data_25432, 'items')
        # Calling items(args, kwargs) (line 263)
        items_call_result_25435 = invoke(stypy.reporting.localization.Localization(__file__, 263, 26), items_25433, *[], **kwargs_25434)
        
        # Testing the type of a for loop iterable (line 263)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 263, 8), items_call_result_25435)
        # Getting the type of the for loop variable (line 263)
        for_loop_var_25436 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 263, 8), items_call_result_25435)
        # Assigning a type to the variable 'key' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 8), for_loop_var_25436))
        # Assigning a type to the variable 'value' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 8), for_loop_var_25436))
        # SSA begins for a for statement (line 263)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to type(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'value' (line 265)
        value_25438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 20), 'value', False)
        # Processing the call keyword arguments (line 265)
        kwargs_25439 = {}
        # Getting the type of 'type' (line 265)
        type_25437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 15), 'type', False)
        # Calling type(args, kwargs) (line 265)
        type_call_result_25440 = invoke(stypy.reporting.localization.Localization(__file__, 265, 15), type_25437, *[value_25438], **kwargs_25439)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 265)
        tuple_25441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 265)
        # Adding element type (line 265)
        
        # Call to type(...): (line 265)
        # Processing the call arguments (line 265)
        
        # Obtaining an instance of the builtin type 'list' (line 265)
        list_25443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 265)
        
        # Processing the call keyword arguments (line 265)
        kwargs_25444 = {}
        # Getting the type of 'type' (line 265)
        type_25442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 35), 'type', False)
        # Calling type(args, kwargs) (line 265)
        type_call_result_25445 = invoke(stypy.reporting.localization.Localization(__file__, 265, 35), type_25442, *[list_25443], **kwargs_25444)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 35), tuple_25441, type_call_result_25445)
        # Adding element type (line 265)
        
        # Call to type(...): (line 265)
        # Processing the call arguments (line 265)
        
        # Obtaining an instance of the builtin type 'tuple' (line 265)
        tuple_25447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 51), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 265)
        
        # Processing the call keyword arguments (line 265)
        kwargs_25448 = {}
        # Getting the type of 'type' (line 265)
        type_25446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 45), 'type', False)
        # Calling type(args, kwargs) (line 265)
        type_call_result_25449 = invoke(stypy.reporting.localization.Localization(__file__, 265, 45), type_25446, *[tuple_25447], **kwargs_25448)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 35), tuple_25441, type_call_result_25449)
        
        # Applying the binary operator 'notin' (line 265)
        result_contains_25450 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 15), 'notin', type_call_result_25440, tuple_25441)
        
        # Testing the type of an if condition (line 265)
        if_condition_25451 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 12), result_contains_25450)
        # Assigning a type to the variable 'if_condition_25451' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'if_condition_25451', if_condition_25451)
        # SSA begins for if statement (line 265)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 266):
        
        # Assigning a List to a Name (line 266):
        
        # Obtaining an instance of the builtin type 'list' (line 266)
        list_25452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 266)
        # Adding element type (line 266)
        # Getting the type of 'value' (line 266)
        value_25453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 25), 'value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 24), list_25452, value_25453)
        
        # Assigning a type to the variable 'value' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'value', list_25452)
        # SSA join for if statement (line 265)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'value' (line 267)
        value_25454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 25), 'value')
        # Testing the type of a for loop iterable (line 267)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 267, 12), value_25454)
        # Getting the type of the for loop variable (line 267)
        for_loop_var_25455 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 267, 12), value_25454)
        # Assigning a type to the variable 'value' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'value', for_loop_var_25455)
        # SSA begins for a for statement (line 267)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'sep_boundary' (line 268)
        sep_boundary_25458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 30), 'sep_boundary', False)
        # Processing the call keyword arguments (line 268)
        kwargs_25459 = {}
        # Getting the type of 'chunks' (line 268)
        chunks_25456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 16), 'chunks', False)
        # Obtaining the member 'append' of a type (line 268)
        append_25457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 16), chunks_25456, 'append')
        # Calling append(args, kwargs) (line 268)
        append_call_result_25460 = invoke(stypy.reporting.localization.Localization(__file__, 268, 16), append_25457, *[sep_boundary_25458], **kwargs_25459)
        
        
        # Call to append(...): (line 269)
        # Processing the call arguments (line 269)
        str_25463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 30), 'str', '\nContent-Disposition: form-data; name="%s"')
        # Getting the type of 'key' (line 269)
        key_25464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 76), 'key', False)
        # Applying the binary operator '%' (line 269)
        result_mod_25465 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 30), '%', str_25463, key_25464)
        
        # Processing the call keyword arguments (line 269)
        kwargs_25466 = {}
        # Getting the type of 'chunks' (line 269)
        chunks_25461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 16), 'chunks', False)
        # Obtaining the member 'append' of a type (line 269)
        append_25462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 16), chunks_25461, 'append')
        # Calling append(args, kwargs) (line 269)
        append_call_result_25467 = invoke(stypy.reporting.localization.Localization(__file__, 269, 16), append_25462, *[result_mod_25465], **kwargs_25466)
        
        
        # Call to append(...): (line 270)
        # Processing the call arguments (line 270)
        str_25470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 30), 'str', '\n\n')
        # Processing the call keyword arguments (line 270)
        kwargs_25471 = {}
        # Getting the type of 'chunks' (line 270)
        chunks_25468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 16), 'chunks', False)
        # Obtaining the member 'append' of a type (line 270)
        append_25469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 16), chunks_25468, 'append')
        # Calling append(args, kwargs) (line 270)
        append_call_result_25472 = invoke(stypy.reporting.localization.Localization(__file__, 270, 16), append_25469, *[str_25470], **kwargs_25471)
        
        
        # Call to append(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'value' (line 271)
        value_25475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 30), 'value', False)
        # Processing the call keyword arguments (line 271)
        kwargs_25476 = {}
        # Getting the type of 'chunks' (line 271)
        chunks_25473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'chunks', False)
        # Obtaining the member 'append' of a type (line 271)
        append_25474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 16), chunks_25473, 'append')
        # Calling append(args, kwargs) (line 271)
        append_call_result_25477 = invoke(stypy.reporting.localization.Localization(__file__, 271, 16), append_25474, *[value_25475], **kwargs_25476)
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'value' (line 272)
        value_25478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 19), 'value')
        
        
        # Obtaining the type of the subscript
        int_25479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 35), 'int')
        # Getting the type of 'value' (line 272)
        value_25480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 29), 'value')
        # Obtaining the member '__getitem__' of a type (line 272)
        getitem___25481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 29), value_25480, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 272)
        subscript_call_result_25482 = invoke(stypy.reporting.localization.Localization(__file__, 272, 29), getitem___25481, int_25479)
        
        str_25483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 42), 'str', '\r')
        # Applying the binary operator '==' (line 272)
        result_eq_25484 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 29), '==', subscript_call_result_25482, str_25483)
        
        # Applying the binary operator 'and' (line 272)
        result_and_keyword_25485 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 19), 'and', value_25478, result_eq_25484)
        
        # Testing the type of an if condition (line 272)
        if_condition_25486 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 16), result_and_keyword_25485)
        # Assigning a type to the variable 'if_condition_25486' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), 'if_condition_25486', if_condition_25486)
        # SSA begins for if statement (line 272)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 273)
        # Processing the call arguments (line 273)
        str_25489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 34), 'str', '\n')
        # Processing the call keyword arguments (line 273)
        kwargs_25490 = {}
        # Getting the type of 'chunks' (line 273)
        chunks_25487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 20), 'chunks', False)
        # Obtaining the member 'append' of a type (line 273)
        append_25488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 20), chunks_25487, 'append')
        # Calling append(args, kwargs) (line 273)
        append_call_result_25491 = invoke(stypy.reporting.localization.Localization(__file__, 273, 20), append_25488, *[str_25489], **kwargs_25490)
        
        # SSA join for if statement (line 272)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'end_boundary' (line 274)
        end_boundary_25494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 22), 'end_boundary', False)
        # Processing the call keyword arguments (line 274)
        kwargs_25495 = {}
        # Getting the type of 'chunks' (line 274)
        chunks_25492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'chunks', False)
        # Obtaining the member 'append' of a type (line 274)
        append_25493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), chunks_25492, 'append')
        # Calling append(args, kwargs) (line 274)
        append_call_result_25496 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), append_25493, *[end_boundary_25494], **kwargs_25495)
        
        
        # Call to append(...): (line 275)
        # Processing the call arguments (line 275)
        str_25499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 22), 'str', '\n')
        # Processing the call keyword arguments (line 275)
        kwargs_25500 = {}
        # Getting the type of 'chunks' (line 275)
        chunks_25497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'chunks', False)
        # Obtaining the member 'append' of a type (line 275)
        append_25498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), chunks_25497, 'append')
        # Calling append(args, kwargs) (line 275)
        append_call_result_25501 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), append_25498, *[str_25499], **kwargs_25500)
        
        
        # Assigning a List to a Name (line 278):
        
        # Assigning a List to a Name (line 278):
        
        # Obtaining an instance of the builtin type 'list' (line 278)
        list_25502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 278)
        
        # Assigning a type to the variable 'body' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'body', list_25502)
        
        # Getting the type of 'chunks' (line 279)
        chunks_25503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 21), 'chunks')
        # Testing the type of a for loop iterable (line 279)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 279, 8), chunks_25503)
        # Getting the type of the for loop variable (line 279)
        for_loop_var_25504 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 279, 8), chunks_25503)
        # Assigning a type to the variable 'chunk' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'chunk', for_loop_var_25504)
        # SSA begins for a for statement (line 279)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 280)
        # Getting the type of 'unicode' (line 280)
        unicode_25505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 33), 'unicode')
        # Getting the type of 'chunk' (line 280)
        chunk_25506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 26), 'chunk')
        
        (may_be_25507, more_types_in_union_25508) = may_be_subtype(unicode_25505, chunk_25506)

        if may_be_25507:

            if more_types_in_union_25508:
                # Runtime conditional SSA (line 280)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'chunk' (line 280)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'chunk', remove_not_subtype_from_union(chunk_25506, unicode))
            
            # Call to append(...): (line 281)
            # Processing the call arguments (line 281)
            
            # Call to encode(...): (line 281)
            # Processing the call arguments (line 281)
            str_25513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 41), 'str', 'utf-8')
            # Processing the call keyword arguments (line 281)
            kwargs_25514 = {}
            # Getting the type of 'chunk' (line 281)
            chunk_25511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 28), 'chunk', False)
            # Obtaining the member 'encode' of a type (line 281)
            encode_25512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 28), chunk_25511, 'encode')
            # Calling encode(args, kwargs) (line 281)
            encode_call_result_25515 = invoke(stypy.reporting.localization.Localization(__file__, 281, 28), encode_25512, *[str_25513], **kwargs_25514)
            
            # Processing the call keyword arguments (line 281)
            kwargs_25516 = {}
            # Getting the type of 'body' (line 281)
            body_25509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'body', False)
            # Obtaining the member 'append' of a type (line 281)
            append_25510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 16), body_25509, 'append')
            # Calling append(args, kwargs) (line 281)
            append_call_result_25517 = invoke(stypy.reporting.localization.Localization(__file__, 281, 16), append_25510, *[encode_call_result_25515], **kwargs_25516)
            

            if more_types_in_union_25508:
                # Runtime conditional SSA for else branch (line 280)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_25507) or more_types_in_union_25508):
            # Assigning a type to the variable 'chunk' (line 280)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'chunk', remove_subtype_from_union(chunk_25506, unicode))
            
            # Call to append(...): (line 283)
            # Processing the call arguments (line 283)
            # Getting the type of 'chunk' (line 283)
            chunk_25520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 28), 'chunk', False)
            # Processing the call keyword arguments (line 283)
            kwargs_25521 = {}
            # Getting the type of 'body' (line 283)
            body_25518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'body', False)
            # Obtaining the member 'append' of a type (line 283)
            append_25519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 16), body_25518, 'append')
            # Calling append(args, kwargs) (line 283)
            append_call_result_25522 = invoke(stypy.reporting.localization.Localization(__file__, 283, 16), append_25519, *[chunk_25520], **kwargs_25521)
            

            if (may_be_25507 and more_types_in_union_25508):
                # SSA join for if statement (line 280)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 285):
        
        # Assigning a Call to a Name (line 285):
        
        # Call to join(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 'body' (line 285)
        body_25525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 23), 'body', False)
        # Processing the call keyword arguments (line 285)
        kwargs_25526 = {}
        str_25523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 15), 'str', '')
        # Obtaining the member 'join' of a type (line 285)
        join_25524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 15), str_25523, 'join')
        # Calling join(args, kwargs) (line 285)
        join_call_result_25527 = invoke(stypy.reporting.localization.Localization(__file__, 285, 15), join_25524, *[body_25525], **kwargs_25526)
        
        # Assigning a type to the variable 'body' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'body', join_call_result_25527)
        
        # Assigning a Dict to a Name (line 288):
        
        # Assigning a Dict to a Name (line 288):
        
        # Obtaining an instance of the builtin type 'dict' (line 288)
        dict_25528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 18), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 288)
        # Adding element type (key, value) (line 288)
        str_25529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 12), 'str', 'Content-type')
        str_25530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 28), 'str', 'multipart/form-data; boundary=%s; charset=utf-8')
        # Getting the type of 'boundary' (line 289)
        boundary_25531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 78), 'boundary')
        # Applying the binary operator '%' (line 289)
        result_mod_25532 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 28), '%', str_25530, boundary_25531)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 18), dict_25528, (str_25529, result_mod_25532))
        # Adding element type (key, value) (line 288)
        str_25533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 12), 'str', 'Content-length')
        
        # Call to str(...): (line 290)
        # Processing the call arguments (line 290)
        
        # Call to len(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'body' (line 290)
        body_25536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 38), 'body', False)
        # Processing the call keyword arguments (line 290)
        kwargs_25537 = {}
        # Getting the type of 'len' (line 290)
        len_25535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 34), 'len', False)
        # Calling len(args, kwargs) (line 290)
        len_call_result_25538 = invoke(stypy.reporting.localization.Localization(__file__, 290, 34), len_25535, *[body_25536], **kwargs_25537)
        
        # Processing the call keyword arguments (line 290)
        kwargs_25539 = {}
        # Getting the type of 'str' (line 290)
        str_25534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 30), 'str', False)
        # Calling str(args, kwargs) (line 290)
        str_call_result_25540 = invoke(stypy.reporting.localization.Localization(__file__, 290, 30), str_25534, *[len_call_result_25538], **kwargs_25539)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 18), dict_25528, (str_25533, str_call_result_25540))
        
        # Assigning a type to the variable 'headers' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'headers', dict_25528)
        
        # Assigning a Call to a Name (line 292):
        
        # Assigning a Call to a Name (line 292):
        
        # Call to Request(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'self' (line 292)
        self_25543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 30), 'self', False)
        # Obtaining the member 'repository' of a type (line 292)
        repository_25544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 30), self_25543, 'repository')
        # Getting the type of 'body' (line 292)
        body_25545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 47), 'body', False)
        # Getting the type of 'headers' (line 292)
        headers_25546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 53), 'headers', False)
        # Processing the call keyword arguments (line 292)
        kwargs_25547 = {}
        # Getting the type of 'urllib2' (line 292)
        urllib2_25541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 14), 'urllib2', False)
        # Obtaining the member 'Request' of a type (line 292)
        Request_25542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 14), urllib2_25541, 'Request')
        # Calling Request(args, kwargs) (line 292)
        Request_call_result_25548 = invoke(stypy.reporting.localization.Localization(__file__, 292, 14), Request_25542, *[repository_25544, body_25545, headers_25546], **kwargs_25547)
        
        # Assigning a type to the variable 'req' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'req', Request_call_result_25548)
        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Call to build_opener(...): (line 295)
        # Processing the call arguments (line 295)
        
        # Call to HTTPBasicAuthHandler(...): (line 296)
        # Processing the call keyword arguments (line 296)
        # Getting the type of 'auth' (line 296)
        auth_25553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 54), 'auth', False)
        keyword_25554 = auth_25553
        kwargs_25555 = {'password_mgr': keyword_25554}
        # Getting the type of 'urllib2' (line 296)
        urllib2_25551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'urllib2', False)
        # Obtaining the member 'HTTPBasicAuthHandler' of a type (line 296)
        HTTPBasicAuthHandler_25552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 12), urllib2_25551, 'HTTPBasicAuthHandler')
        # Calling HTTPBasicAuthHandler(args, kwargs) (line 296)
        HTTPBasicAuthHandler_call_result_25556 = invoke(stypy.reporting.localization.Localization(__file__, 296, 12), HTTPBasicAuthHandler_25552, *[], **kwargs_25555)
        
        # Processing the call keyword arguments (line 295)
        kwargs_25557 = {}
        # Getting the type of 'urllib2' (line 295)
        urllib2_25549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 17), 'urllib2', False)
        # Obtaining the member 'build_opener' of a type (line 295)
        build_opener_25550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 17), urllib2_25549, 'build_opener')
        # Calling build_opener(args, kwargs) (line 295)
        build_opener_call_result_25558 = invoke(stypy.reporting.localization.Localization(__file__, 295, 17), build_opener_25550, *[HTTPBasicAuthHandler_call_result_25556], **kwargs_25557)
        
        # Assigning a type to the variable 'opener' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'opener', build_opener_call_result_25558)
        
        # Assigning a Str to a Name (line 298):
        
        # Assigning a Str to a Name (line 298):
        str_25559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 15), 'str', '')
        # Assigning a type to the variable 'data' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'data', str_25559)
        
        
        # SSA begins for try-except statement (line 299)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 300):
        
        # Assigning a Call to a Name (line 300):
        
        # Call to open(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'req' (line 300)
        req_25562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 33), 'req', False)
        # Processing the call keyword arguments (line 300)
        kwargs_25563 = {}
        # Getting the type of 'opener' (line 300)
        opener_25560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 21), 'opener', False)
        # Obtaining the member 'open' of a type (line 300)
        open_25561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 21), opener_25560, 'open')
        # Calling open(args, kwargs) (line 300)
        open_call_result_25564 = invoke(stypy.reporting.localization.Localization(__file__, 300, 21), open_25561, *[req_25562], **kwargs_25563)
        
        # Assigning a type to the variable 'result' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'result', open_call_result_25564)
        # SSA branch for the except part of a try statement (line 299)
        # SSA branch for the except 'Attribute' branch of a try statement (line 299)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'urllib2' (line 301)
        urllib2_25565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 15), 'urllib2')
        # Obtaining the member 'HTTPError' of a type (line 301)
        HTTPError_25566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 15), urllib2_25565, 'HTTPError')
        # Assigning a type to the variable 'e' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'e', HTTPError_25566)
        
        # Getting the type of 'self' (line 302)
        self_25567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 15), 'self')
        # Obtaining the member 'show_response' of a type (line 302)
        show_response_25568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 15), self_25567, 'show_response')
        # Testing the type of an if condition (line 302)
        if_condition_25569 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 302, 12), show_response_25568)
        # Assigning a type to the variable 'if_condition_25569' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'if_condition_25569', if_condition_25569)
        # SSA begins for if statement (line 302)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 303):
        
        # Assigning a Call to a Name (line 303):
        
        # Call to read(...): (line 303)
        # Processing the call keyword arguments (line 303)
        kwargs_25573 = {}
        # Getting the type of 'e' (line 303)
        e_25570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 23), 'e', False)
        # Obtaining the member 'fp' of a type (line 303)
        fp_25571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 23), e_25570, 'fp')
        # Obtaining the member 'read' of a type (line 303)
        read_25572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 23), fp_25571, 'read')
        # Calling read(args, kwargs) (line 303)
        read_call_result_25574 = invoke(stypy.reporting.localization.Localization(__file__, 303, 23), read_25572, *[], **kwargs_25573)
        
        # Assigning a type to the variable 'data' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'data', read_call_result_25574)
        # SSA join for if statement (line 302)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Name (line 304):
        
        # Assigning a Tuple to a Name (line 304):
        
        # Obtaining an instance of the builtin type 'tuple' (line 304)
        tuple_25575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 304)
        # Adding element type (line 304)
        # Getting the type of 'e' (line 304)
        e_25576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 21), 'e')
        # Obtaining the member 'code' of a type (line 304)
        code_25577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 21), e_25576, 'code')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 21), tuple_25575, code_25577)
        # Adding element type (line 304)
        # Getting the type of 'e' (line 304)
        e_25578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 29), 'e')
        # Obtaining the member 'msg' of a type (line 304)
        msg_25579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 29), e_25578, 'msg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 21), tuple_25575, msg_25579)
        
        # Assigning a type to the variable 'result' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'result', tuple_25575)
        # SSA branch for the except 'Attribute' branch of a try statement (line 299)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'urllib2' (line 305)
        urllib2_25580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 15), 'urllib2')
        # Obtaining the member 'URLError' of a type (line 305)
        URLError_25581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 15), urllib2_25580, 'URLError')
        # Assigning a type to the variable 'e' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'e', URLError_25581)
        
        # Assigning a Tuple to a Name (line 306):
        
        # Assigning a Tuple to a Name (line 306):
        
        # Obtaining an instance of the builtin type 'tuple' (line 306)
        tuple_25582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 306)
        # Adding element type (line 306)
        int_25583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 21), tuple_25582, int_25583)
        # Adding element type (line 306)
        
        # Call to str(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 'e' (line 306)
        e_25585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 30), 'e', False)
        # Processing the call keyword arguments (line 306)
        kwargs_25586 = {}
        # Getting the type of 'str' (line 306)
        str_25584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 26), 'str', False)
        # Calling str(args, kwargs) (line 306)
        str_call_result_25587 = invoke(stypy.reporting.localization.Localization(__file__, 306, 26), str_25584, *[e_25585], **kwargs_25586)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 21), tuple_25582, str_call_result_25587)
        
        # Assigning a type to the variable 'result' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'result', tuple_25582)
        # SSA branch for the else branch of a try statement (line 299)
        module_type_store.open_ssa_branch('except else')
        
        # Getting the type of 'self' (line 308)
        self_25588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 15), 'self')
        # Obtaining the member 'show_response' of a type (line 308)
        show_response_25589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 15), self_25588, 'show_response')
        # Testing the type of an if condition (line 308)
        if_condition_25590 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 308, 12), show_response_25589)
        # Assigning a type to the variable 'if_condition_25590' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'if_condition_25590', if_condition_25590)
        # SSA begins for if statement (line 308)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 309):
        
        # Assigning a Call to a Name (line 309):
        
        # Call to read(...): (line 309)
        # Processing the call keyword arguments (line 309)
        kwargs_25593 = {}
        # Getting the type of 'result' (line 309)
        result_25591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 23), 'result', False)
        # Obtaining the member 'read' of a type (line 309)
        read_25592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 23), result_25591, 'read')
        # Calling read(args, kwargs) (line 309)
        read_call_result_25594 = invoke(stypy.reporting.localization.Localization(__file__, 309, 23), read_25592, *[], **kwargs_25593)
        
        # Assigning a type to the variable 'data' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'data', read_call_result_25594)
        # SSA join for if statement (line 308)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Name (line 310):
        
        # Assigning a Tuple to a Name (line 310):
        
        # Obtaining an instance of the builtin type 'tuple' (line 310)
        tuple_25595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 310)
        # Adding element type (line 310)
        int_25596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 21), tuple_25595, int_25596)
        # Adding element type (line 310)
        str_25597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 26), 'str', 'OK')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 21), tuple_25595, str_25597)
        
        # Assigning a type to the variable 'result' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'result', tuple_25595)
        # SSA join for try-except statement (line 299)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 311)
        self_25598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 11), 'self')
        # Obtaining the member 'show_response' of a type (line 311)
        show_response_25599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 11), self_25598, 'show_response')
        # Testing the type of an if condition (line 311)
        if_condition_25600 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 311, 8), show_response_25599)
        # Assigning a type to the variable 'if_condition_25600' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'if_condition_25600', if_condition_25600)
        # SSA begins for if statement (line 311)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 312):
        
        # Assigning a BinOp to a Name (line 312):
        str_25601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 21), 'str', '-')
        int_25602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 27), 'int')
        # Applying the binary operator '*' (line 312)
        result_mul_25603 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 21), '*', str_25601, int_25602)
        
        # Assigning a type to the variable 'dashes' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'dashes', result_mul_25603)
        
        # Call to announce(...): (line 313)
        # Processing the call arguments (line 313)
        str_25606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 26), 'str', '%s%s%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 313)
        tuple_25607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 313)
        # Adding element type (line 313)
        # Getting the type of 'dashes' (line 313)
        dashes_25608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 38), 'dashes', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 38), tuple_25607, dashes_25608)
        # Adding element type (line 313)
        # Getting the type of 'data' (line 313)
        data_25609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 46), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 38), tuple_25607, data_25609)
        # Adding element type (line 313)
        # Getting the type of 'dashes' (line 313)
        dashes_25610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 52), 'dashes', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 38), tuple_25607, dashes_25610)
        
        # Applying the binary operator '%' (line 313)
        result_mod_25611 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 26), '%', str_25606, tuple_25607)
        
        # Processing the call keyword arguments (line 313)
        kwargs_25612 = {}
        # Getting the type of 'self' (line 313)
        self_25604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'self', False)
        # Obtaining the member 'announce' of a type (line 313)
        announce_25605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 12), self_25604, 'announce')
        # Calling announce(args, kwargs) (line 313)
        announce_call_result_25613 = invoke(stypy.reporting.localization.Localization(__file__, 313, 12), announce_25605, *[result_mod_25611], **kwargs_25612)
        
        # SSA join for if statement (line 311)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'result' (line 315)
        result_25614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'stypy_return_type', result_25614)
        
        # ################# End of 'post_to_server(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'post_to_server' in the type store
        # Getting the type of 'stypy_return_type' (line 251)
        stypy_return_type_25615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25615)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'post_to_server'
        return stypy_return_type_25615


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 18, 0, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'register.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'register' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'register', register)

# Assigning a Str to a Name (line 20):
str_25616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 19), 'str', 'register the distribution with the Python package index')
# Getting the type of 'register'
register_25617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'register')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), register_25617, 'description', str_25616)

# Assigning a BinOp to a Name (line 21):
# Getting the type of 'PyPIRCCommand' (line 21)
PyPIRCCommand_25618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'PyPIRCCommand')
# Obtaining the member 'user_options' of a type (line 21)
user_options_25619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 19), PyPIRCCommand_25618, 'user_options')

# Obtaining an instance of the builtin type 'list' (line 21)
list_25620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 48), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)

# Obtaining an instance of the builtin type 'tuple' (line 22)
tuple_25621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 22)
# Adding element type (line 22)
str_25622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 9), 'str', 'list-classifiers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_25621, str_25622)
# Adding element type (line 22)
# Getting the type of 'None' (line 22)
None_25623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 29), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_25621, None_25623)
# Adding element type (line 22)
str_25624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'str', 'list the valid Trove classifiers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_25621, str_25624)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 48), list_25620, tuple_25621)
# Adding element type (line 21)

# Obtaining an instance of the builtin type 'tuple' (line 24)
tuple_25625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 24)
# Adding element type (line 24)
str_25626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'str', 'strict')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_25625, str_25626)
# Adding element type (line 24)
# Getting the type of 'None' (line 24)
None_25627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 19), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_25625, None_25627)
# Adding element type (line 24)
str_25628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 9), 'str', 'Will stop the registering if the meta-data are not fully compliant')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_25625, str_25628)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 48), list_25620, tuple_25625)

# Applying the binary operator '+' (line 21)
result_add_25629 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 19), '+', user_options_25619, list_25620)

# Getting the type of 'register'
register_25630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'register')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), register_25630, 'user_options', result_add_25629)

# Assigning a BinOp to a Name (line 27):
# Getting the type of 'PyPIRCCommand' (line 27)
PyPIRCCommand_25631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 22), 'PyPIRCCommand')
# Obtaining the member 'boolean_options' of a type (line 27)
boolean_options_25632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 22), PyPIRCCommand_25631, 'boolean_options')

# Obtaining an instance of the builtin type 'list' (line 27)
list_25633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 54), 'list')
# Adding type elements to the builtin type 'list' instance (line 27)
# Adding element type (line 27)
str_25634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'str', 'verify')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 54), list_25633, str_25634)
# Adding element type (line 27)
str_25635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 18), 'str', 'list-classifiers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 54), list_25633, str_25635)
# Adding element type (line 27)
str_25636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 38), 'str', 'strict')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 54), list_25633, str_25636)

# Applying the binary operator '+' (line 27)
result_add_25637 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 22), '+', boolean_options_25632, list_25633)

# Getting the type of 'register'
register_25638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'register')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), register_25638, 'boolean_options', result_add_25637)

# Assigning a List to a Name (line 30):

# Obtaining an instance of the builtin type 'list' (line 30)
list_25639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 30)
# Adding element type (line 30)

# Obtaining an instance of the builtin type 'tuple' (line 30)
tuple_25640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 30)
# Adding element type (line 30)
str_25641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 21), 'str', 'check')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 21), tuple_25640, str_25641)
# Adding element type (line 30)

@norecursion
def _stypy_temp_lambda_4(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_4'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_4', 30, 30, True)
    # Passed parameters checking function
    _stypy_temp_lambda_4.stypy_localization = localization
    _stypy_temp_lambda_4.stypy_type_of_self = None
    _stypy_temp_lambda_4.stypy_type_store = module_type_store
    _stypy_temp_lambda_4.stypy_function_name = '_stypy_temp_lambda_4'
    _stypy_temp_lambda_4.stypy_param_names_list = ['self']
    _stypy_temp_lambda_4.stypy_varargs_param_name = None
    _stypy_temp_lambda_4.stypy_kwargs_param_name = None
    _stypy_temp_lambda_4.stypy_call_defaults = defaults
    _stypy_temp_lambda_4.stypy_call_varargs = varargs
    _stypy_temp_lambda_4.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_4', ['self'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_4', ['self'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of 'True' (line 30)
    True_25642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 43), 'True')
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 30), 'stypy_return_type', True_25642)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_4' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_25643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 30), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25643)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_4'
    return stypy_return_type_25643

# Assigning a type to the variable '_stypy_temp_lambda_4' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 30), '_stypy_temp_lambda_4', _stypy_temp_lambda_4)
# Getting the type of '_stypy_temp_lambda_4' (line 30)
_stypy_temp_lambda_4_25644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 30), '_stypy_temp_lambda_4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 21), tuple_25640, _stypy_temp_lambda_4_25644)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 19), list_25639, tuple_25640)

# Getting the type of 'register'
register_25645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'register')
# Setting the type of the member 'sub_commands' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), register_25645, 'sub_commands', list_25639)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
