
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.upload
2: 
3: Implements the Distutils 'upload' subcommand (upload package to PyPI).'''
4: import os
5: import socket
6: import platform
7: from urllib2 import urlopen, Request, HTTPError
8: from base64 import standard_b64encode
9: import urlparse
10: import cStringIO as StringIO
11: from hashlib import md5
12: 
13: from distutils.errors import DistutilsError, DistutilsOptionError
14: from distutils.core import PyPIRCCommand
15: from distutils.spawn import spawn
16: from distutils import log
17: 
18: class upload(PyPIRCCommand):
19: 
20:     description = "upload binary package to PyPI"
21: 
22:     user_options = PyPIRCCommand.user_options + [
23:         ('sign', 's',
24:          'sign files to upload using gpg'),
25:         ('identity=', 'i', 'GPG identity used to sign files'),
26:         ]
27: 
28:     boolean_options = PyPIRCCommand.boolean_options + ['sign']
29: 
30:     def initialize_options(self):
31:         PyPIRCCommand.initialize_options(self)
32:         self.username = ''
33:         self.password = ''
34:         self.show_response = 0
35:         self.sign = False
36:         self.identity = None
37: 
38:     def finalize_options(self):
39:         PyPIRCCommand.finalize_options(self)
40:         if self.identity and not self.sign:
41:             raise DistutilsOptionError(
42:                 "Must use --sign for --identity to have meaning"
43:             )
44:         config = self._read_pypirc()
45:         if config != {}:
46:             self.username = config['username']
47:             self.password = config['password']
48:             self.repository = config['repository']
49:             self.realm = config['realm']
50: 
51:         # getting the password from the distribution
52:         # if previously set by the register command
53:         if not self.password and self.distribution.password:
54:             self.password = self.distribution.password
55: 
56:     def run(self):
57:         if not self.distribution.dist_files:
58:             raise DistutilsOptionError("No dist file created in earlier command")
59:         for command, pyversion, filename in self.distribution.dist_files:
60:             self.upload_file(command, pyversion, filename)
61: 
62:     def upload_file(self, command, pyversion, filename):
63:         # Makes sure the repository URL is compliant
64:         schema, netloc, url, params, query, fragments = \
65:             urlparse.urlparse(self.repository)
66:         if params or query or fragments:
67:             raise AssertionError("Incompatible url %s" % self.repository)
68: 
69:         if schema not in ('http', 'https'):
70:             raise AssertionError("unsupported schema " + schema)
71: 
72:         # Sign if requested
73:         if self.sign:
74:             gpg_args = ["gpg", "--detach-sign", "-a", filename]
75:             if self.identity:
76:                 gpg_args[2:2] = ["--local-user", self.identity]
77:             spawn(gpg_args,
78:                   dry_run=self.dry_run)
79: 
80:         # Fill in the data - send all the meta-data in case we need to
81:         # register a new release
82:         f = open(filename,'rb')
83:         try:
84:             content = f.read()
85:         finally:
86:             f.close()
87:         meta = self.distribution.metadata
88:         data = {
89:             # action
90:             ':action': 'file_upload',
91:             'protcol_version': '1',
92: 
93:             # identify release
94:             'name': meta.get_name(),
95:             'version': meta.get_version(),
96: 
97:             # file content
98:             'content': (os.path.basename(filename),content),
99:             'filetype': command,
100:             'pyversion': pyversion,
101:             'md5_digest': md5(content).hexdigest(),
102: 
103:             # additional meta-data
104:             'metadata_version' : '1.0',
105:             'summary': meta.get_description(),
106:             'home_page': meta.get_url(),
107:             'author': meta.get_contact(),
108:             'author_email': meta.get_contact_email(),
109:             'license': meta.get_licence(),
110:             'description': meta.get_long_description(),
111:             'keywords': meta.get_keywords(),
112:             'platform': meta.get_platforms(),
113:             'classifiers': meta.get_classifiers(),
114:             'download_url': meta.get_download_url(),
115:             # PEP 314
116:             'provides': meta.get_provides(),
117:             'requires': meta.get_requires(),
118:             'obsoletes': meta.get_obsoletes(),
119:             }
120:         comment = ''
121:         if command == 'bdist_rpm':
122:             dist, version, id = platform.dist()
123:             if dist:
124:                 comment = 'built for %s %s' % (dist, version)
125:         elif command == 'bdist_dumb':
126:             comment = 'built for %s' % platform.platform(terse=1)
127:         data['comment'] = comment
128: 
129:         if self.sign:
130:             data['gpg_signature'] = (os.path.basename(filename) + ".asc",
131:                                      open(filename+".asc").read())
132: 
133:         # set up the authentication
134:         auth = "Basic " + standard_b64encode(self.username + ":" +
135:                                              self.password)
136: 
137:         # Build up the MIME payload for the POST data
138:         boundary = '--------------GHSKFJDLGDS7543FJKLFHRE75642756743254'
139:         sep_boundary = '\r\n--' + boundary
140:         end_boundary = sep_boundary + '--\r\n'
141:         body = StringIO.StringIO()
142:         for key, value in data.items():
143:             # handle multiple entries for the same name
144:             if not isinstance(value, list):
145:                 value = [value]
146:             for value in value:
147:                 if isinstance(value, tuple):
148:                     fn = ';filename="%s"' % value[0]
149:                     value = value[1]
150:                 else:
151:                     fn = ""
152: 
153:                 body.write(sep_boundary)
154:                 body.write('\r\nContent-Disposition: form-data; name="%s"' % key)
155:                 body.write(fn)
156:                 body.write("\r\n\r\n")
157:                 body.write(value)
158:                 if value and value[-1] == '\r':
159:                     body.write('\n')  # write an extra newline (lurve Macs)
160:         body.write(end_boundary)
161:         body = body.getvalue()
162: 
163:         self.announce("Submitting %s to %s" % (filename, self.repository), log.INFO)
164: 
165:         # build the Request
166:         headers = {'Content-type':
167:                         'multipart/form-data; boundary=%s' % boundary,
168:                    'Content-length': str(len(body)),
169:                    'Authorization': auth}
170: 
171:         request = Request(self.repository, data=body,
172:                           headers=headers)
173:         # send the data
174:         try:
175:             result = urlopen(request)
176:             status = result.getcode()
177:             reason = result.msg
178:             if self.show_response:
179:                 msg = '\n'.join(('-' * 75, result.read(), '-' * 75))
180:                 self.announce(msg, log.INFO)
181:         except socket.error, e:
182:             self.announce(str(e), log.ERROR)
183:             raise
184:         except HTTPError, e:
185:             status = e.code
186:             reason = e.msg
187: 
188:         if status == 200:
189:             self.announce('Server response (%s): %s' % (status, reason),
190:                           log.INFO)
191:         else:
192:             msg = 'Upload failed (%s): %s' % (status, reason)
193:             self.announce(msg, log.ERROR)
194:             raise DistutilsError(msg)
195: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_26669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', "distutils.command.upload\n\nImplements the Distutils 'upload' subcommand (upload package to PyPI).")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import socket' statement (line 5)
import socket

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'socket', socket, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import platform' statement (line 6)
import platform

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'platform', platform, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from urllib2 import urlopen, Request, HTTPError' statement (line 7)
try:
    from urllib2 import urlopen, Request, HTTPError

except:
    urlopen = UndefinedType
    Request = UndefinedType
    HTTPError = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'urllib2', None, module_type_store, ['urlopen', 'Request', 'HTTPError'], [urlopen, Request, HTTPError])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from base64 import standard_b64encode' statement (line 8)
try:
    from base64 import standard_b64encode

except:
    standard_b64encode = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'base64', None, module_type_store, ['standard_b64encode'], [standard_b64encode])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import urlparse' statement (line 9)
import urlparse

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'urlparse', urlparse, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import cStringIO' statement (line 10)
import cStringIO as StringIO

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'StringIO', StringIO, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from hashlib import md5' statement (line 11)
try:
    from hashlib import md5

except:
    md5 = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'hashlib', None, module_type_store, ['md5'], [md5])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from distutils.errors import DistutilsError, DistutilsOptionError' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_26670 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.errors')

if (type(import_26670) is not StypyTypeError):

    if (import_26670 != 'pyd_module'):
        __import__(import_26670)
        sys_modules_26671 = sys.modules[import_26670]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.errors', sys_modules_26671.module_type_store, module_type_store, ['DistutilsError', 'DistutilsOptionError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_26671, sys_modules_26671.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsError, DistutilsOptionError

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.errors', None, module_type_store, ['DistutilsError', 'DistutilsOptionError'], [DistutilsError, DistutilsOptionError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.errors', import_26670)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from distutils.core import PyPIRCCommand' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_26672 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.core')

if (type(import_26672) is not StypyTypeError):

    if (import_26672 != 'pyd_module'):
        __import__(import_26672)
        sys_modules_26673 = sys.modules[import_26672]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.core', sys_modules_26673.module_type_store, module_type_store, ['PyPIRCCommand'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_26673, sys_modules_26673.module_type_store, module_type_store)
    else:
        from distutils.core import PyPIRCCommand

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.core', None, module_type_store, ['PyPIRCCommand'], [PyPIRCCommand])

else:
    # Assigning a type to the variable 'distutils.core' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.core', import_26672)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from distutils.spawn import spawn' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_26674 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.spawn')

if (type(import_26674) is not StypyTypeError):

    if (import_26674 != 'pyd_module'):
        __import__(import_26674)
        sys_modules_26675 = sys.modules[import_26674]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.spawn', sys_modules_26675.module_type_store, module_type_store, ['spawn'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_26675, sys_modules_26675.module_type_store, module_type_store)
    else:
        from distutils.spawn import spawn

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.spawn', None, module_type_store, ['spawn'], [spawn])

else:
    # Assigning a type to the variable 'distutils.spawn' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.spawn', import_26674)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from distutils import log' statement (line 16)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils', None, module_type_store, ['log'], [log])

# Declaration of the 'upload' class
# Getting the type of 'PyPIRCCommand' (line 18)
PyPIRCCommand_26676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 13), 'PyPIRCCommand')

class upload(PyPIRCCommand_26676, ):
    
    # Assigning a Str to a Name (line 20):
    
    # Assigning a BinOp to a Name (line 22):
    
    # Assigning a BinOp to a Name (line 28):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        upload.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        upload.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        upload.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        upload.initialize_options.__dict__.__setitem__('stypy_function_name', 'upload.initialize_options')
        upload.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        upload.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        upload.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        upload.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        upload.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        upload.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        upload.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'upload.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to initialize_options(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'self' (line 31)
        self_26679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 41), 'self', False)
        # Processing the call keyword arguments (line 31)
        kwargs_26680 = {}
        # Getting the type of 'PyPIRCCommand' (line 31)
        PyPIRCCommand_26677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'PyPIRCCommand', False)
        # Obtaining the member 'initialize_options' of a type (line 31)
        initialize_options_26678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), PyPIRCCommand_26677, 'initialize_options')
        # Calling initialize_options(args, kwargs) (line 31)
        initialize_options_call_result_26681 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), initialize_options_26678, *[self_26679], **kwargs_26680)
        
        
        # Assigning a Str to a Attribute (line 32):
        
        # Assigning a Str to a Attribute (line 32):
        str_26682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 24), 'str', '')
        # Getting the type of 'self' (line 32)
        self_26683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self')
        # Setting the type of the member 'username' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_26683, 'username', str_26682)
        
        # Assigning a Str to a Attribute (line 33):
        
        # Assigning a Str to a Attribute (line 33):
        str_26684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 24), 'str', '')
        # Getting the type of 'self' (line 33)
        self_26685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Setting the type of the member 'password' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_26685, 'password', str_26684)
        
        # Assigning a Num to a Attribute (line 34):
        
        # Assigning a Num to a Attribute (line 34):
        int_26686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 29), 'int')
        # Getting the type of 'self' (line 34)
        self_26687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'show_response' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_26687, 'show_response', int_26686)
        
        # Assigning a Name to a Attribute (line 35):
        
        # Assigning a Name to a Attribute (line 35):
        # Getting the type of 'False' (line 35)
        False_26688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'False')
        # Getting the type of 'self' (line 35)
        self_26689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self')
        # Setting the type of the member 'sign' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_26689, 'sign', False_26688)
        
        # Assigning a Name to a Attribute (line 36):
        
        # Assigning a Name to a Attribute (line 36):
        # Getting the type of 'None' (line 36)
        None_26690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'None')
        # Getting the type of 'self' (line 36)
        self_26691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self')
        # Setting the type of the member 'identity' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_26691, 'identity', None_26690)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_26692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26692)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_26692


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        upload.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        upload.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        upload.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        upload.finalize_options.__dict__.__setitem__('stypy_function_name', 'upload.finalize_options')
        upload.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        upload.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        upload.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        upload.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        upload.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        upload.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        upload.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'upload.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to finalize_options(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'self' (line 39)
        self_26695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 39), 'self', False)
        # Processing the call keyword arguments (line 39)
        kwargs_26696 = {}
        # Getting the type of 'PyPIRCCommand' (line 39)
        PyPIRCCommand_26693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'PyPIRCCommand', False)
        # Obtaining the member 'finalize_options' of a type (line 39)
        finalize_options_26694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), PyPIRCCommand_26693, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 39)
        finalize_options_call_result_26697 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), finalize_options_26694, *[self_26695], **kwargs_26696)
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 40)
        self_26698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'self')
        # Obtaining the member 'identity' of a type (line 40)
        identity_26699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 11), self_26698, 'identity')
        
        # Getting the type of 'self' (line 40)
        self_26700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 33), 'self')
        # Obtaining the member 'sign' of a type (line 40)
        sign_26701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 33), self_26700, 'sign')
        # Applying the 'not' unary operator (line 40)
        result_not__26702 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 29), 'not', sign_26701)
        
        # Applying the binary operator 'and' (line 40)
        result_and_keyword_26703 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 11), 'and', identity_26699, result_not__26702)
        
        # Testing the type of an if condition (line 40)
        if_condition_26704 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 8), result_and_keyword_26703)
        # Assigning a type to the variable 'if_condition_26704' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'if_condition_26704', if_condition_26704)
        # SSA begins for if statement (line 40)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsOptionError(...): (line 41)
        # Processing the call arguments (line 41)
        str_26706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 16), 'str', 'Must use --sign for --identity to have meaning')
        # Processing the call keyword arguments (line 41)
        kwargs_26707 = {}
        # Getting the type of 'DistutilsOptionError' (line 41)
        DistutilsOptionError_26705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 18), 'DistutilsOptionError', False)
        # Calling DistutilsOptionError(args, kwargs) (line 41)
        DistutilsOptionError_call_result_26708 = invoke(stypy.reporting.localization.Localization(__file__, 41, 18), DistutilsOptionError_26705, *[str_26706], **kwargs_26707)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 41, 12), DistutilsOptionError_call_result_26708, 'raise parameter', BaseException)
        # SSA join for if statement (line 40)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 44):
        
        # Assigning a Call to a Name (line 44):
        
        # Call to _read_pypirc(...): (line 44)
        # Processing the call keyword arguments (line 44)
        kwargs_26711 = {}
        # Getting the type of 'self' (line 44)
        self_26709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 17), 'self', False)
        # Obtaining the member '_read_pypirc' of a type (line 44)
        _read_pypirc_26710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 17), self_26709, '_read_pypirc')
        # Calling _read_pypirc(args, kwargs) (line 44)
        _read_pypirc_call_result_26712 = invoke(stypy.reporting.localization.Localization(__file__, 44, 17), _read_pypirc_26710, *[], **kwargs_26711)
        
        # Assigning a type to the variable 'config' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'config', _read_pypirc_call_result_26712)
        
        
        # Getting the type of 'config' (line 45)
        config_26713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'config')
        
        # Obtaining an instance of the builtin type 'dict' (line 45)
        dict_26714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 21), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 45)
        
        # Applying the binary operator '!=' (line 45)
        result_ne_26715 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 11), '!=', config_26713, dict_26714)
        
        # Testing the type of an if condition (line 45)
        if_condition_26716 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 8), result_ne_26715)
        # Assigning a type to the variable 'if_condition_26716' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'if_condition_26716', if_condition_26716)
        # SSA begins for if statement (line 45)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Attribute (line 46):
        
        # Assigning a Subscript to a Attribute (line 46):
        
        # Obtaining the type of the subscript
        str_26717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 35), 'str', 'username')
        # Getting the type of 'config' (line 46)
        config_26718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 28), 'config')
        # Obtaining the member '__getitem__' of a type (line 46)
        getitem___26719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 28), config_26718, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 46)
        subscript_call_result_26720 = invoke(stypy.reporting.localization.Localization(__file__, 46, 28), getitem___26719, str_26717)
        
        # Getting the type of 'self' (line 46)
        self_26721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'self')
        # Setting the type of the member 'username' of a type (line 46)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 12), self_26721, 'username', subscript_call_result_26720)
        
        # Assigning a Subscript to a Attribute (line 47):
        
        # Assigning a Subscript to a Attribute (line 47):
        
        # Obtaining the type of the subscript
        str_26722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 35), 'str', 'password')
        # Getting the type of 'config' (line 47)
        config_26723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 28), 'config')
        # Obtaining the member '__getitem__' of a type (line 47)
        getitem___26724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 28), config_26723, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 47)
        subscript_call_result_26725 = invoke(stypy.reporting.localization.Localization(__file__, 47, 28), getitem___26724, str_26722)
        
        # Getting the type of 'self' (line 47)
        self_26726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'self')
        # Setting the type of the member 'password' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), self_26726, 'password', subscript_call_result_26725)
        
        # Assigning a Subscript to a Attribute (line 48):
        
        # Assigning a Subscript to a Attribute (line 48):
        
        # Obtaining the type of the subscript
        str_26727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 37), 'str', 'repository')
        # Getting the type of 'config' (line 48)
        config_26728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 30), 'config')
        # Obtaining the member '__getitem__' of a type (line 48)
        getitem___26729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 30), config_26728, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 48)
        subscript_call_result_26730 = invoke(stypy.reporting.localization.Localization(__file__, 48, 30), getitem___26729, str_26727)
        
        # Getting the type of 'self' (line 48)
        self_26731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'self')
        # Setting the type of the member 'repository' of a type (line 48)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), self_26731, 'repository', subscript_call_result_26730)
        
        # Assigning a Subscript to a Attribute (line 49):
        
        # Assigning a Subscript to a Attribute (line 49):
        
        # Obtaining the type of the subscript
        str_26732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 32), 'str', 'realm')
        # Getting the type of 'config' (line 49)
        config_26733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'config')
        # Obtaining the member '__getitem__' of a type (line 49)
        getitem___26734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 25), config_26733, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
        subscript_call_result_26735 = invoke(stypy.reporting.localization.Localization(__file__, 49, 25), getitem___26734, str_26732)
        
        # Getting the type of 'self' (line 49)
        self_26736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'self')
        # Setting the type of the member 'realm' of a type (line 49)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), self_26736, 'realm', subscript_call_result_26735)
        # SSA join for if statement (line 45)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 53)
        self_26737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 15), 'self')
        # Obtaining the member 'password' of a type (line 53)
        password_26738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 15), self_26737, 'password')
        # Applying the 'not' unary operator (line 53)
        result_not__26739 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 11), 'not', password_26738)
        
        # Getting the type of 'self' (line 53)
        self_26740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 33), 'self')
        # Obtaining the member 'distribution' of a type (line 53)
        distribution_26741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 33), self_26740, 'distribution')
        # Obtaining the member 'password' of a type (line 53)
        password_26742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 33), distribution_26741, 'password')
        # Applying the binary operator 'and' (line 53)
        result_and_keyword_26743 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 11), 'and', result_not__26739, password_26742)
        
        # Testing the type of an if condition (line 53)
        if_condition_26744 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 8), result_and_keyword_26743)
        # Assigning a type to the variable 'if_condition_26744' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'if_condition_26744', if_condition_26744)
        # SSA begins for if statement (line 53)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 54):
        
        # Assigning a Attribute to a Attribute (line 54):
        # Getting the type of 'self' (line 54)
        self_26745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'self')
        # Obtaining the member 'distribution' of a type (line 54)
        distribution_26746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 28), self_26745, 'distribution')
        # Obtaining the member 'password' of a type (line 54)
        password_26747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 28), distribution_26746, 'password')
        # Getting the type of 'self' (line 54)
        self_26748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'self')
        # Setting the type of the member 'password' of a type (line 54)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), self_26748, 'password', password_26747)
        # SSA join for if statement (line 53)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_26749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26749)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_26749


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        upload.run.__dict__.__setitem__('stypy_localization', localization)
        upload.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        upload.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        upload.run.__dict__.__setitem__('stypy_function_name', 'upload.run')
        upload.run.__dict__.__setitem__('stypy_param_names_list', [])
        upload.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        upload.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        upload.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        upload.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        upload.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        upload.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'upload.run', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'self' (line 57)
        self_26750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'self')
        # Obtaining the member 'distribution' of a type (line 57)
        distribution_26751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 15), self_26750, 'distribution')
        # Obtaining the member 'dist_files' of a type (line 57)
        dist_files_26752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 15), distribution_26751, 'dist_files')
        # Applying the 'not' unary operator (line 57)
        result_not__26753 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 11), 'not', dist_files_26752)
        
        # Testing the type of an if condition (line 57)
        if_condition_26754 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 8), result_not__26753)
        # Assigning a type to the variable 'if_condition_26754' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'if_condition_26754', if_condition_26754)
        # SSA begins for if statement (line 57)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsOptionError(...): (line 58)
        # Processing the call arguments (line 58)
        str_26756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 39), 'str', 'No dist file created in earlier command')
        # Processing the call keyword arguments (line 58)
        kwargs_26757 = {}
        # Getting the type of 'DistutilsOptionError' (line 58)
        DistutilsOptionError_26755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 18), 'DistutilsOptionError', False)
        # Calling DistutilsOptionError(args, kwargs) (line 58)
        DistutilsOptionError_call_result_26758 = invoke(stypy.reporting.localization.Localization(__file__, 58, 18), DistutilsOptionError_26755, *[str_26756], **kwargs_26757)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 58, 12), DistutilsOptionError_call_result_26758, 'raise parameter', BaseException)
        # SSA join for if statement (line 57)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 59)
        self_26759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 44), 'self')
        # Obtaining the member 'distribution' of a type (line 59)
        distribution_26760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 44), self_26759, 'distribution')
        # Obtaining the member 'dist_files' of a type (line 59)
        dist_files_26761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 44), distribution_26760, 'dist_files')
        # Testing the type of a for loop iterable (line 59)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 59, 8), dist_files_26761)
        # Getting the type of the for loop variable (line 59)
        for_loop_var_26762 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 59, 8), dist_files_26761)
        # Assigning a type to the variable 'command' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'command', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 8), for_loop_var_26762))
        # Assigning a type to the variable 'pyversion' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'pyversion', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 8), for_loop_var_26762))
        # Assigning a type to the variable 'filename' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'filename', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 8), for_loop_var_26762))
        # SSA begins for a for statement (line 59)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to upload_file(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'command' (line 60)
        command_26765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 29), 'command', False)
        # Getting the type of 'pyversion' (line 60)
        pyversion_26766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 38), 'pyversion', False)
        # Getting the type of 'filename' (line 60)
        filename_26767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 49), 'filename', False)
        # Processing the call keyword arguments (line 60)
        kwargs_26768 = {}
        # Getting the type of 'self' (line 60)
        self_26763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'self', False)
        # Obtaining the member 'upload_file' of a type (line 60)
        upload_file_26764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), self_26763, 'upload_file')
        # Calling upload_file(args, kwargs) (line 60)
        upload_file_call_result_26769 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), upload_file_26764, *[command_26765, pyversion_26766, filename_26767], **kwargs_26768)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_26770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26770)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_26770


    @norecursion
    def upload_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'upload_file'
        module_type_store = module_type_store.open_function_context('upload_file', 62, 4, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        upload.upload_file.__dict__.__setitem__('stypy_localization', localization)
        upload.upload_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        upload.upload_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        upload.upload_file.__dict__.__setitem__('stypy_function_name', 'upload.upload_file')
        upload.upload_file.__dict__.__setitem__('stypy_param_names_list', ['command', 'pyversion', 'filename'])
        upload.upload_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        upload.upload_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        upload.upload_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        upload.upload_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        upload.upload_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        upload.upload_file.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'upload.upload_file', ['command', 'pyversion', 'filename'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'upload_file', localization, ['command', 'pyversion', 'filename'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'upload_file(...)' code ##################

        
        # Assigning a Call to a Tuple (line 64):
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_26771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        
        # Call to urlparse(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'self' (line 65)
        self_26774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'self', False)
        # Obtaining the member 'repository' of a type (line 65)
        repository_26775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 30), self_26774, 'repository')
        # Processing the call keyword arguments (line 65)
        kwargs_26776 = {}
        # Getting the type of 'urlparse' (line 65)
        urlparse_26772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'urlparse', False)
        # Obtaining the member 'urlparse' of a type (line 65)
        urlparse_26773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), urlparse_26772, 'urlparse')
        # Calling urlparse(args, kwargs) (line 65)
        urlparse_call_result_26777 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), urlparse_26773, *[repository_26775], **kwargs_26776)
        
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___26778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), urlparse_call_result_26777, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_26779 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___26778, int_26771)
        
        # Assigning a type to the variable 'tuple_var_assignment_26660' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_26660', subscript_call_result_26779)
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_26780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        
        # Call to urlparse(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'self' (line 65)
        self_26783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'self', False)
        # Obtaining the member 'repository' of a type (line 65)
        repository_26784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 30), self_26783, 'repository')
        # Processing the call keyword arguments (line 65)
        kwargs_26785 = {}
        # Getting the type of 'urlparse' (line 65)
        urlparse_26781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'urlparse', False)
        # Obtaining the member 'urlparse' of a type (line 65)
        urlparse_26782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), urlparse_26781, 'urlparse')
        # Calling urlparse(args, kwargs) (line 65)
        urlparse_call_result_26786 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), urlparse_26782, *[repository_26784], **kwargs_26785)
        
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___26787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), urlparse_call_result_26786, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_26788 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___26787, int_26780)
        
        # Assigning a type to the variable 'tuple_var_assignment_26661' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_26661', subscript_call_result_26788)
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_26789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        
        # Call to urlparse(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'self' (line 65)
        self_26792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'self', False)
        # Obtaining the member 'repository' of a type (line 65)
        repository_26793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 30), self_26792, 'repository')
        # Processing the call keyword arguments (line 65)
        kwargs_26794 = {}
        # Getting the type of 'urlparse' (line 65)
        urlparse_26790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'urlparse', False)
        # Obtaining the member 'urlparse' of a type (line 65)
        urlparse_26791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), urlparse_26790, 'urlparse')
        # Calling urlparse(args, kwargs) (line 65)
        urlparse_call_result_26795 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), urlparse_26791, *[repository_26793], **kwargs_26794)
        
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___26796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), urlparse_call_result_26795, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_26797 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___26796, int_26789)
        
        # Assigning a type to the variable 'tuple_var_assignment_26662' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_26662', subscript_call_result_26797)
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_26798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        
        # Call to urlparse(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'self' (line 65)
        self_26801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'self', False)
        # Obtaining the member 'repository' of a type (line 65)
        repository_26802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 30), self_26801, 'repository')
        # Processing the call keyword arguments (line 65)
        kwargs_26803 = {}
        # Getting the type of 'urlparse' (line 65)
        urlparse_26799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'urlparse', False)
        # Obtaining the member 'urlparse' of a type (line 65)
        urlparse_26800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), urlparse_26799, 'urlparse')
        # Calling urlparse(args, kwargs) (line 65)
        urlparse_call_result_26804 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), urlparse_26800, *[repository_26802], **kwargs_26803)
        
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___26805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), urlparse_call_result_26804, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_26806 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___26805, int_26798)
        
        # Assigning a type to the variable 'tuple_var_assignment_26663' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_26663', subscript_call_result_26806)
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_26807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        
        # Call to urlparse(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'self' (line 65)
        self_26810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'self', False)
        # Obtaining the member 'repository' of a type (line 65)
        repository_26811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 30), self_26810, 'repository')
        # Processing the call keyword arguments (line 65)
        kwargs_26812 = {}
        # Getting the type of 'urlparse' (line 65)
        urlparse_26808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'urlparse', False)
        # Obtaining the member 'urlparse' of a type (line 65)
        urlparse_26809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), urlparse_26808, 'urlparse')
        # Calling urlparse(args, kwargs) (line 65)
        urlparse_call_result_26813 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), urlparse_26809, *[repository_26811], **kwargs_26812)
        
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___26814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), urlparse_call_result_26813, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_26815 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___26814, int_26807)
        
        # Assigning a type to the variable 'tuple_var_assignment_26664' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_26664', subscript_call_result_26815)
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_26816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        
        # Call to urlparse(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'self' (line 65)
        self_26819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'self', False)
        # Obtaining the member 'repository' of a type (line 65)
        repository_26820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 30), self_26819, 'repository')
        # Processing the call keyword arguments (line 65)
        kwargs_26821 = {}
        # Getting the type of 'urlparse' (line 65)
        urlparse_26817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'urlparse', False)
        # Obtaining the member 'urlparse' of a type (line 65)
        urlparse_26818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), urlparse_26817, 'urlparse')
        # Calling urlparse(args, kwargs) (line 65)
        urlparse_call_result_26822 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), urlparse_26818, *[repository_26820], **kwargs_26821)
        
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___26823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), urlparse_call_result_26822, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_26824 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___26823, int_26816)
        
        # Assigning a type to the variable 'tuple_var_assignment_26665' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_26665', subscript_call_result_26824)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_26660' (line 64)
        tuple_var_assignment_26660_26825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_26660')
        # Assigning a type to the variable 'schema' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'schema', tuple_var_assignment_26660_26825)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_26661' (line 64)
        tuple_var_assignment_26661_26826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_26661')
        # Assigning a type to the variable 'netloc' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'netloc', tuple_var_assignment_26661_26826)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_26662' (line 64)
        tuple_var_assignment_26662_26827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_26662')
        # Assigning a type to the variable 'url' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 24), 'url', tuple_var_assignment_26662_26827)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_26663' (line 64)
        tuple_var_assignment_26663_26828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_26663')
        # Assigning a type to the variable 'params' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 29), 'params', tuple_var_assignment_26663_26828)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_26664' (line 64)
        tuple_var_assignment_26664_26829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_26664')
        # Assigning a type to the variable 'query' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 37), 'query', tuple_var_assignment_26664_26829)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_26665' (line 64)
        tuple_var_assignment_26665_26830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_26665')
        # Assigning a type to the variable 'fragments' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 44), 'fragments', tuple_var_assignment_26665_26830)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'params' (line 66)
        params_26831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'params')
        # Getting the type of 'query' (line 66)
        query_26832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 21), 'query')
        # Applying the binary operator 'or' (line 66)
        result_or_keyword_26833 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 11), 'or', params_26831, query_26832)
        # Getting the type of 'fragments' (line 66)
        fragments_26834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 30), 'fragments')
        # Applying the binary operator 'or' (line 66)
        result_or_keyword_26835 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 11), 'or', result_or_keyword_26833, fragments_26834)
        
        # Testing the type of an if condition (line 66)
        if_condition_26836 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 8), result_or_keyword_26835)
        # Assigning a type to the variable 'if_condition_26836' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'if_condition_26836', if_condition_26836)
        # SSA begins for if statement (line 66)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to AssertionError(...): (line 67)
        # Processing the call arguments (line 67)
        str_26838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 33), 'str', 'Incompatible url %s')
        # Getting the type of 'self' (line 67)
        self_26839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 57), 'self', False)
        # Obtaining the member 'repository' of a type (line 67)
        repository_26840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 57), self_26839, 'repository')
        # Applying the binary operator '%' (line 67)
        result_mod_26841 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 33), '%', str_26838, repository_26840)
        
        # Processing the call keyword arguments (line 67)
        kwargs_26842 = {}
        # Getting the type of 'AssertionError' (line 67)
        AssertionError_26837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'AssertionError', False)
        # Calling AssertionError(args, kwargs) (line 67)
        AssertionError_call_result_26843 = invoke(stypy.reporting.localization.Localization(__file__, 67, 18), AssertionError_26837, *[result_mod_26841], **kwargs_26842)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 67, 12), AssertionError_call_result_26843, 'raise parameter', BaseException)
        # SSA join for if statement (line 66)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'schema' (line 69)
        schema_26844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'schema')
        
        # Obtaining an instance of the builtin type 'tuple' (line 69)
        tuple_26845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 69)
        # Adding element type (line 69)
        str_26846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 26), 'str', 'http')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 26), tuple_26845, str_26846)
        # Adding element type (line 69)
        str_26847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 34), 'str', 'https')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 26), tuple_26845, str_26847)
        
        # Applying the binary operator 'notin' (line 69)
        result_contains_26848 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 11), 'notin', schema_26844, tuple_26845)
        
        # Testing the type of an if condition (line 69)
        if_condition_26849 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 8), result_contains_26848)
        # Assigning a type to the variable 'if_condition_26849' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'if_condition_26849', if_condition_26849)
        # SSA begins for if statement (line 69)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to AssertionError(...): (line 70)
        # Processing the call arguments (line 70)
        str_26851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 33), 'str', 'unsupported schema ')
        # Getting the type of 'schema' (line 70)
        schema_26852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 57), 'schema', False)
        # Applying the binary operator '+' (line 70)
        result_add_26853 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 33), '+', str_26851, schema_26852)
        
        # Processing the call keyword arguments (line 70)
        kwargs_26854 = {}
        # Getting the type of 'AssertionError' (line 70)
        AssertionError_26850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 18), 'AssertionError', False)
        # Calling AssertionError(args, kwargs) (line 70)
        AssertionError_call_result_26855 = invoke(stypy.reporting.localization.Localization(__file__, 70, 18), AssertionError_26850, *[result_add_26853], **kwargs_26854)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 70, 12), AssertionError_call_result_26855, 'raise parameter', BaseException)
        # SSA join for if statement (line 69)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 73)
        self_26856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'self')
        # Obtaining the member 'sign' of a type (line 73)
        sign_26857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 11), self_26856, 'sign')
        # Testing the type of an if condition (line 73)
        if_condition_26858 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 8), sign_26857)
        # Assigning a type to the variable 'if_condition_26858' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'if_condition_26858', if_condition_26858)
        # SSA begins for if statement (line 73)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 74):
        
        # Assigning a List to a Name (line 74):
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_26859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        # Adding element type (line 74)
        str_26860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 24), 'str', 'gpg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 23), list_26859, str_26860)
        # Adding element type (line 74)
        str_26861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 31), 'str', '--detach-sign')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 23), list_26859, str_26861)
        # Adding element type (line 74)
        str_26862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 48), 'str', '-a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 23), list_26859, str_26862)
        # Adding element type (line 74)
        # Getting the type of 'filename' (line 74)
        filename_26863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 54), 'filename')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 23), list_26859, filename_26863)
        
        # Assigning a type to the variable 'gpg_args' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'gpg_args', list_26859)
        
        # Getting the type of 'self' (line 75)
        self_26864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'self')
        # Obtaining the member 'identity' of a type (line 75)
        identity_26865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), self_26864, 'identity')
        # Testing the type of an if condition (line 75)
        if_condition_26866 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 12), identity_26865)
        # Assigning a type to the variable 'if_condition_26866' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'if_condition_26866', if_condition_26866)
        # SSA begins for if statement (line 75)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Subscript (line 76):
        
        # Assigning a List to a Subscript (line 76):
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_26867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        str_26868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'str', '--local-user')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 32), list_26867, str_26868)
        # Adding element type (line 76)
        # Getting the type of 'self' (line 76)
        self_26869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 49), 'self')
        # Obtaining the member 'identity' of a type (line 76)
        identity_26870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 49), self_26869, 'identity')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 32), list_26867, identity_26870)
        
        # Getting the type of 'gpg_args' (line 76)
        gpg_args_26871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'gpg_args')
        int_26872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 25), 'int')
        int_26873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 27), 'int')
        slice_26874 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 76, 16), int_26872, int_26873, None)
        # Storing an element on a container (line 76)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 16), gpg_args_26871, (slice_26874, list_26867))
        # SSA join for if statement (line 75)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to spawn(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'gpg_args' (line 77)
        gpg_args_26876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), 'gpg_args', False)
        # Processing the call keyword arguments (line 77)
        # Getting the type of 'self' (line 78)
        self_26877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 78)
        dry_run_26878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 26), self_26877, 'dry_run')
        keyword_26879 = dry_run_26878
        kwargs_26880 = {'dry_run': keyword_26879}
        # Getting the type of 'spawn' (line 77)
        spawn_26875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'spawn', False)
        # Calling spawn(args, kwargs) (line 77)
        spawn_call_result_26881 = invoke(stypy.reporting.localization.Localization(__file__, 77, 12), spawn_26875, *[gpg_args_26876], **kwargs_26880)
        
        # SSA join for if statement (line 73)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 82):
        
        # Assigning a Call to a Name (line 82):
        
        # Call to open(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'filename' (line 82)
        filename_26883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 17), 'filename', False)
        str_26884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 26), 'str', 'rb')
        # Processing the call keyword arguments (line 82)
        kwargs_26885 = {}
        # Getting the type of 'open' (line 82)
        open_26882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'open', False)
        # Calling open(args, kwargs) (line 82)
        open_call_result_26886 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), open_26882, *[filename_26883, str_26884], **kwargs_26885)
        
        # Assigning a type to the variable 'f' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'f', open_call_result_26886)
        
        # Try-finally block (line 83)
        
        # Assigning a Call to a Name (line 84):
        
        # Assigning a Call to a Name (line 84):
        
        # Call to read(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_26889 = {}
        # Getting the type of 'f' (line 84)
        f_26887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 22), 'f', False)
        # Obtaining the member 'read' of a type (line 84)
        read_26888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 22), f_26887, 'read')
        # Calling read(args, kwargs) (line 84)
        read_call_result_26890 = invoke(stypy.reporting.localization.Localization(__file__, 84, 22), read_26888, *[], **kwargs_26889)
        
        # Assigning a type to the variable 'content' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'content', read_call_result_26890)
        
        # finally branch of the try-finally block (line 83)
        
        # Call to close(...): (line 86)
        # Processing the call keyword arguments (line 86)
        kwargs_26893 = {}
        # Getting the type of 'f' (line 86)
        f_26891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 86)
        close_26892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), f_26891, 'close')
        # Calling close(args, kwargs) (line 86)
        close_call_result_26894 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), close_26892, *[], **kwargs_26893)
        
        
        
        # Assigning a Attribute to a Name (line 87):
        
        # Assigning a Attribute to a Name (line 87):
        # Getting the type of 'self' (line 87)
        self_26895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'self')
        # Obtaining the member 'distribution' of a type (line 87)
        distribution_26896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 15), self_26895, 'distribution')
        # Obtaining the member 'metadata' of a type (line 87)
        metadata_26897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 15), distribution_26896, 'metadata')
        # Assigning a type to the variable 'meta' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'meta', metadata_26897)
        
        # Assigning a Dict to a Name (line 88):
        
        # Assigning a Dict to a Name (line 88):
        
        # Obtaining an instance of the builtin type 'dict' (line 88)
        dict_26898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 15), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 88)
        # Adding element type (key, value) (line 88)
        str_26899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 12), 'str', ':action')
        str_26900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 23), 'str', 'file_upload')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26899, str_26900))
        # Adding element type (key, value) (line 88)
        str_26901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 12), 'str', 'protcol_version')
        str_26902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 31), 'str', '1')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26901, str_26902))
        # Adding element type (key, value) (line 88)
        str_26903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 12), 'str', 'name')
        
        # Call to get_name(...): (line 94)
        # Processing the call keyword arguments (line 94)
        kwargs_26906 = {}
        # Getting the type of 'meta' (line 94)
        meta_26904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'meta', False)
        # Obtaining the member 'get_name' of a type (line 94)
        get_name_26905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 20), meta_26904, 'get_name')
        # Calling get_name(args, kwargs) (line 94)
        get_name_call_result_26907 = invoke(stypy.reporting.localization.Localization(__file__, 94, 20), get_name_26905, *[], **kwargs_26906)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26903, get_name_call_result_26907))
        # Adding element type (key, value) (line 88)
        str_26908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 12), 'str', 'version')
        
        # Call to get_version(...): (line 95)
        # Processing the call keyword arguments (line 95)
        kwargs_26911 = {}
        # Getting the type of 'meta' (line 95)
        meta_26909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 23), 'meta', False)
        # Obtaining the member 'get_version' of a type (line 95)
        get_version_26910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 23), meta_26909, 'get_version')
        # Calling get_version(args, kwargs) (line 95)
        get_version_call_result_26912 = invoke(stypy.reporting.localization.Localization(__file__, 95, 23), get_version_26910, *[], **kwargs_26911)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26908, get_version_call_result_26912))
        # Adding element type (key, value) (line 88)
        str_26913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 12), 'str', 'content')
        
        # Obtaining an instance of the builtin type 'tuple' (line 98)
        tuple_26914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 98)
        # Adding element type (line 98)
        
        # Call to basename(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'filename' (line 98)
        filename_26918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 41), 'filename', False)
        # Processing the call keyword arguments (line 98)
        kwargs_26919 = {}
        # Getting the type of 'os' (line 98)
        os_26915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 98)
        path_26916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 24), os_26915, 'path')
        # Obtaining the member 'basename' of a type (line 98)
        basename_26917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 24), path_26916, 'basename')
        # Calling basename(args, kwargs) (line 98)
        basename_call_result_26920 = invoke(stypy.reporting.localization.Localization(__file__, 98, 24), basename_26917, *[filename_26918], **kwargs_26919)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 24), tuple_26914, basename_call_result_26920)
        # Adding element type (line 98)
        # Getting the type of 'content' (line 98)
        content_26921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 51), 'content')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 24), tuple_26914, content_26921)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26913, tuple_26914))
        # Adding element type (key, value) (line 88)
        str_26922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 12), 'str', 'filetype')
        # Getting the type of 'command' (line 99)
        command_26923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 24), 'command')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26922, command_26923))
        # Adding element type (key, value) (line 88)
        str_26924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 12), 'str', 'pyversion')
        # Getting the type of 'pyversion' (line 100)
        pyversion_26925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'pyversion')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26924, pyversion_26925))
        # Adding element type (key, value) (line 88)
        str_26926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 12), 'str', 'md5_digest')
        
        # Call to hexdigest(...): (line 101)
        # Processing the call keyword arguments (line 101)
        kwargs_26932 = {}
        
        # Call to md5(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'content' (line 101)
        content_26928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'content', False)
        # Processing the call keyword arguments (line 101)
        kwargs_26929 = {}
        # Getting the type of 'md5' (line 101)
        md5_26927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 26), 'md5', False)
        # Calling md5(args, kwargs) (line 101)
        md5_call_result_26930 = invoke(stypy.reporting.localization.Localization(__file__, 101, 26), md5_26927, *[content_26928], **kwargs_26929)
        
        # Obtaining the member 'hexdigest' of a type (line 101)
        hexdigest_26931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 26), md5_call_result_26930, 'hexdigest')
        # Calling hexdigest(args, kwargs) (line 101)
        hexdigest_call_result_26933 = invoke(stypy.reporting.localization.Localization(__file__, 101, 26), hexdigest_26931, *[], **kwargs_26932)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26926, hexdigest_call_result_26933))
        # Adding element type (key, value) (line 88)
        str_26934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 12), 'str', 'metadata_version')
        str_26935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 33), 'str', '1.0')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26934, str_26935))
        # Adding element type (key, value) (line 88)
        str_26936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 12), 'str', 'summary')
        
        # Call to get_description(...): (line 105)
        # Processing the call keyword arguments (line 105)
        kwargs_26939 = {}
        # Getting the type of 'meta' (line 105)
        meta_26937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 23), 'meta', False)
        # Obtaining the member 'get_description' of a type (line 105)
        get_description_26938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 23), meta_26937, 'get_description')
        # Calling get_description(args, kwargs) (line 105)
        get_description_call_result_26940 = invoke(stypy.reporting.localization.Localization(__file__, 105, 23), get_description_26938, *[], **kwargs_26939)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26936, get_description_call_result_26940))
        # Adding element type (key, value) (line 88)
        str_26941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 12), 'str', 'home_page')
        
        # Call to get_url(...): (line 106)
        # Processing the call keyword arguments (line 106)
        kwargs_26944 = {}
        # Getting the type of 'meta' (line 106)
        meta_26942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 25), 'meta', False)
        # Obtaining the member 'get_url' of a type (line 106)
        get_url_26943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 25), meta_26942, 'get_url')
        # Calling get_url(args, kwargs) (line 106)
        get_url_call_result_26945 = invoke(stypy.reporting.localization.Localization(__file__, 106, 25), get_url_26943, *[], **kwargs_26944)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26941, get_url_call_result_26945))
        # Adding element type (key, value) (line 88)
        str_26946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 12), 'str', 'author')
        
        # Call to get_contact(...): (line 107)
        # Processing the call keyword arguments (line 107)
        kwargs_26949 = {}
        # Getting the type of 'meta' (line 107)
        meta_26947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 22), 'meta', False)
        # Obtaining the member 'get_contact' of a type (line 107)
        get_contact_26948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 22), meta_26947, 'get_contact')
        # Calling get_contact(args, kwargs) (line 107)
        get_contact_call_result_26950 = invoke(stypy.reporting.localization.Localization(__file__, 107, 22), get_contact_26948, *[], **kwargs_26949)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26946, get_contact_call_result_26950))
        # Adding element type (key, value) (line 88)
        str_26951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 12), 'str', 'author_email')
        
        # Call to get_contact_email(...): (line 108)
        # Processing the call keyword arguments (line 108)
        kwargs_26954 = {}
        # Getting the type of 'meta' (line 108)
        meta_26952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 28), 'meta', False)
        # Obtaining the member 'get_contact_email' of a type (line 108)
        get_contact_email_26953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 28), meta_26952, 'get_contact_email')
        # Calling get_contact_email(args, kwargs) (line 108)
        get_contact_email_call_result_26955 = invoke(stypy.reporting.localization.Localization(__file__, 108, 28), get_contact_email_26953, *[], **kwargs_26954)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26951, get_contact_email_call_result_26955))
        # Adding element type (key, value) (line 88)
        str_26956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 12), 'str', 'license')
        
        # Call to get_licence(...): (line 109)
        # Processing the call keyword arguments (line 109)
        kwargs_26959 = {}
        # Getting the type of 'meta' (line 109)
        meta_26957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 23), 'meta', False)
        # Obtaining the member 'get_licence' of a type (line 109)
        get_licence_26958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 23), meta_26957, 'get_licence')
        # Calling get_licence(args, kwargs) (line 109)
        get_licence_call_result_26960 = invoke(stypy.reporting.localization.Localization(__file__, 109, 23), get_licence_26958, *[], **kwargs_26959)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26956, get_licence_call_result_26960))
        # Adding element type (key, value) (line 88)
        str_26961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 12), 'str', 'description')
        
        # Call to get_long_description(...): (line 110)
        # Processing the call keyword arguments (line 110)
        kwargs_26964 = {}
        # Getting the type of 'meta' (line 110)
        meta_26962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 27), 'meta', False)
        # Obtaining the member 'get_long_description' of a type (line 110)
        get_long_description_26963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 27), meta_26962, 'get_long_description')
        # Calling get_long_description(args, kwargs) (line 110)
        get_long_description_call_result_26965 = invoke(stypy.reporting.localization.Localization(__file__, 110, 27), get_long_description_26963, *[], **kwargs_26964)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26961, get_long_description_call_result_26965))
        # Adding element type (key, value) (line 88)
        str_26966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 12), 'str', 'keywords')
        
        # Call to get_keywords(...): (line 111)
        # Processing the call keyword arguments (line 111)
        kwargs_26969 = {}
        # Getting the type of 'meta' (line 111)
        meta_26967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'meta', False)
        # Obtaining the member 'get_keywords' of a type (line 111)
        get_keywords_26968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 24), meta_26967, 'get_keywords')
        # Calling get_keywords(args, kwargs) (line 111)
        get_keywords_call_result_26970 = invoke(stypy.reporting.localization.Localization(__file__, 111, 24), get_keywords_26968, *[], **kwargs_26969)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26966, get_keywords_call_result_26970))
        # Adding element type (key, value) (line 88)
        str_26971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 12), 'str', 'platform')
        
        # Call to get_platforms(...): (line 112)
        # Processing the call keyword arguments (line 112)
        kwargs_26974 = {}
        # Getting the type of 'meta' (line 112)
        meta_26972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 24), 'meta', False)
        # Obtaining the member 'get_platforms' of a type (line 112)
        get_platforms_26973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 24), meta_26972, 'get_platforms')
        # Calling get_platforms(args, kwargs) (line 112)
        get_platforms_call_result_26975 = invoke(stypy.reporting.localization.Localization(__file__, 112, 24), get_platforms_26973, *[], **kwargs_26974)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26971, get_platforms_call_result_26975))
        # Adding element type (key, value) (line 88)
        str_26976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 12), 'str', 'classifiers')
        
        # Call to get_classifiers(...): (line 113)
        # Processing the call keyword arguments (line 113)
        kwargs_26979 = {}
        # Getting the type of 'meta' (line 113)
        meta_26977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), 'meta', False)
        # Obtaining the member 'get_classifiers' of a type (line 113)
        get_classifiers_26978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 27), meta_26977, 'get_classifiers')
        # Calling get_classifiers(args, kwargs) (line 113)
        get_classifiers_call_result_26980 = invoke(stypy.reporting.localization.Localization(__file__, 113, 27), get_classifiers_26978, *[], **kwargs_26979)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26976, get_classifiers_call_result_26980))
        # Adding element type (key, value) (line 88)
        str_26981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 12), 'str', 'download_url')
        
        # Call to get_download_url(...): (line 114)
        # Processing the call keyword arguments (line 114)
        kwargs_26984 = {}
        # Getting the type of 'meta' (line 114)
        meta_26982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 28), 'meta', False)
        # Obtaining the member 'get_download_url' of a type (line 114)
        get_download_url_26983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 28), meta_26982, 'get_download_url')
        # Calling get_download_url(args, kwargs) (line 114)
        get_download_url_call_result_26985 = invoke(stypy.reporting.localization.Localization(__file__, 114, 28), get_download_url_26983, *[], **kwargs_26984)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26981, get_download_url_call_result_26985))
        # Adding element type (key, value) (line 88)
        str_26986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 12), 'str', 'provides')
        
        # Call to get_provides(...): (line 116)
        # Processing the call keyword arguments (line 116)
        kwargs_26989 = {}
        # Getting the type of 'meta' (line 116)
        meta_26987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'meta', False)
        # Obtaining the member 'get_provides' of a type (line 116)
        get_provides_26988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 24), meta_26987, 'get_provides')
        # Calling get_provides(args, kwargs) (line 116)
        get_provides_call_result_26990 = invoke(stypy.reporting.localization.Localization(__file__, 116, 24), get_provides_26988, *[], **kwargs_26989)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26986, get_provides_call_result_26990))
        # Adding element type (key, value) (line 88)
        str_26991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 12), 'str', 'requires')
        
        # Call to get_requires(...): (line 117)
        # Processing the call keyword arguments (line 117)
        kwargs_26994 = {}
        # Getting the type of 'meta' (line 117)
        meta_26992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 24), 'meta', False)
        # Obtaining the member 'get_requires' of a type (line 117)
        get_requires_26993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 24), meta_26992, 'get_requires')
        # Calling get_requires(args, kwargs) (line 117)
        get_requires_call_result_26995 = invoke(stypy.reporting.localization.Localization(__file__, 117, 24), get_requires_26993, *[], **kwargs_26994)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26991, get_requires_call_result_26995))
        # Adding element type (key, value) (line 88)
        str_26996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 12), 'str', 'obsoletes')
        
        # Call to get_obsoletes(...): (line 118)
        # Processing the call keyword arguments (line 118)
        kwargs_26999 = {}
        # Getting the type of 'meta' (line 118)
        meta_26997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 25), 'meta', False)
        # Obtaining the member 'get_obsoletes' of a type (line 118)
        get_obsoletes_26998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 25), meta_26997, 'get_obsoletes')
        # Calling get_obsoletes(args, kwargs) (line 118)
        get_obsoletes_call_result_27000 = invoke(stypy.reporting.localization.Localization(__file__, 118, 25), get_obsoletes_26998, *[], **kwargs_26999)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), dict_26898, (str_26996, get_obsoletes_call_result_27000))
        
        # Assigning a type to the variable 'data' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'data', dict_26898)
        
        # Assigning a Str to a Name (line 120):
        
        # Assigning a Str to a Name (line 120):
        str_27001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 18), 'str', '')
        # Assigning a type to the variable 'comment' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'comment', str_27001)
        
        
        # Getting the type of 'command' (line 121)
        command_27002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'command')
        str_27003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 22), 'str', 'bdist_rpm')
        # Applying the binary operator '==' (line 121)
        result_eq_27004 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 11), '==', command_27002, str_27003)
        
        # Testing the type of an if condition (line 121)
        if_condition_27005 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 8), result_eq_27004)
        # Assigning a type to the variable 'if_condition_27005' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'if_condition_27005', if_condition_27005)
        # SSA begins for if statement (line 121)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 122):
        
        # Assigning a Subscript to a Name (line 122):
        
        # Obtaining the type of the subscript
        int_27006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 12), 'int')
        
        # Call to dist(...): (line 122)
        # Processing the call keyword arguments (line 122)
        kwargs_27009 = {}
        # Getting the type of 'platform' (line 122)
        platform_27007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 32), 'platform', False)
        # Obtaining the member 'dist' of a type (line 122)
        dist_27008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 32), platform_27007, 'dist')
        # Calling dist(args, kwargs) (line 122)
        dist_call_result_27010 = invoke(stypy.reporting.localization.Localization(__file__, 122, 32), dist_27008, *[], **kwargs_27009)
        
        # Obtaining the member '__getitem__' of a type (line 122)
        getitem___27011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 12), dist_call_result_27010, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
        subscript_call_result_27012 = invoke(stypy.reporting.localization.Localization(__file__, 122, 12), getitem___27011, int_27006)
        
        # Assigning a type to the variable 'tuple_var_assignment_26666' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'tuple_var_assignment_26666', subscript_call_result_27012)
        
        # Assigning a Subscript to a Name (line 122):
        
        # Obtaining the type of the subscript
        int_27013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 12), 'int')
        
        # Call to dist(...): (line 122)
        # Processing the call keyword arguments (line 122)
        kwargs_27016 = {}
        # Getting the type of 'platform' (line 122)
        platform_27014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 32), 'platform', False)
        # Obtaining the member 'dist' of a type (line 122)
        dist_27015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 32), platform_27014, 'dist')
        # Calling dist(args, kwargs) (line 122)
        dist_call_result_27017 = invoke(stypy.reporting.localization.Localization(__file__, 122, 32), dist_27015, *[], **kwargs_27016)
        
        # Obtaining the member '__getitem__' of a type (line 122)
        getitem___27018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 12), dist_call_result_27017, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
        subscript_call_result_27019 = invoke(stypy.reporting.localization.Localization(__file__, 122, 12), getitem___27018, int_27013)
        
        # Assigning a type to the variable 'tuple_var_assignment_26667' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'tuple_var_assignment_26667', subscript_call_result_27019)
        
        # Assigning a Subscript to a Name (line 122):
        
        # Obtaining the type of the subscript
        int_27020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 12), 'int')
        
        # Call to dist(...): (line 122)
        # Processing the call keyword arguments (line 122)
        kwargs_27023 = {}
        # Getting the type of 'platform' (line 122)
        platform_27021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 32), 'platform', False)
        # Obtaining the member 'dist' of a type (line 122)
        dist_27022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 32), platform_27021, 'dist')
        # Calling dist(args, kwargs) (line 122)
        dist_call_result_27024 = invoke(stypy.reporting.localization.Localization(__file__, 122, 32), dist_27022, *[], **kwargs_27023)
        
        # Obtaining the member '__getitem__' of a type (line 122)
        getitem___27025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 12), dist_call_result_27024, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
        subscript_call_result_27026 = invoke(stypy.reporting.localization.Localization(__file__, 122, 12), getitem___27025, int_27020)
        
        # Assigning a type to the variable 'tuple_var_assignment_26668' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'tuple_var_assignment_26668', subscript_call_result_27026)
        
        # Assigning a Name to a Name (line 122):
        # Getting the type of 'tuple_var_assignment_26666' (line 122)
        tuple_var_assignment_26666_27027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'tuple_var_assignment_26666')
        # Assigning a type to the variable 'dist' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'dist', tuple_var_assignment_26666_27027)
        
        # Assigning a Name to a Name (line 122):
        # Getting the type of 'tuple_var_assignment_26667' (line 122)
        tuple_var_assignment_26667_27028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'tuple_var_assignment_26667')
        # Assigning a type to the variable 'version' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 18), 'version', tuple_var_assignment_26667_27028)
        
        # Assigning a Name to a Name (line 122):
        # Getting the type of 'tuple_var_assignment_26668' (line 122)
        tuple_var_assignment_26668_27029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'tuple_var_assignment_26668')
        # Assigning a type to the variable 'id' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 27), 'id', tuple_var_assignment_26668_27029)
        
        # Getting the type of 'dist' (line 123)
        dist_27030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'dist')
        # Testing the type of an if condition (line 123)
        if_condition_27031 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 12), dist_27030)
        # Assigning a type to the variable 'if_condition_27031' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'if_condition_27031', if_condition_27031)
        # SSA begins for if statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 124):
        
        # Assigning a BinOp to a Name (line 124):
        str_27032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 26), 'str', 'built for %s %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 124)
        tuple_27033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 124)
        # Adding element type (line 124)
        # Getting the type of 'dist' (line 124)
        dist_27034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 47), 'dist')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 47), tuple_27033, dist_27034)
        # Adding element type (line 124)
        # Getting the type of 'version' (line 124)
        version_27035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 53), 'version')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 47), tuple_27033, version_27035)
        
        # Applying the binary operator '%' (line 124)
        result_mod_27036 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 26), '%', str_27032, tuple_27033)
        
        # Assigning a type to the variable 'comment' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'comment', result_mod_27036)
        # SSA join for if statement (line 123)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 121)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'command' (line 125)
        command_27037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 13), 'command')
        str_27038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 24), 'str', 'bdist_dumb')
        # Applying the binary operator '==' (line 125)
        result_eq_27039 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 13), '==', command_27037, str_27038)
        
        # Testing the type of an if condition (line 125)
        if_condition_27040 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 13), result_eq_27039)
        # Assigning a type to the variable 'if_condition_27040' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 13), 'if_condition_27040', if_condition_27040)
        # SSA begins for if statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 126):
        
        # Assigning a BinOp to a Name (line 126):
        str_27041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 22), 'str', 'built for %s')
        
        # Call to platform(...): (line 126)
        # Processing the call keyword arguments (line 126)
        int_27044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 63), 'int')
        keyword_27045 = int_27044
        kwargs_27046 = {'terse': keyword_27045}
        # Getting the type of 'platform' (line 126)
        platform_27042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 39), 'platform', False)
        # Obtaining the member 'platform' of a type (line 126)
        platform_27043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 39), platform_27042, 'platform')
        # Calling platform(args, kwargs) (line 126)
        platform_call_result_27047 = invoke(stypy.reporting.localization.Localization(__file__, 126, 39), platform_27043, *[], **kwargs_27046)
        
        # Applying the binary operator '%' (line 126)
        result_mod_27048 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 22), '%', str_27041, platform_call_result_27047)
        
        # Assigning a type to the variable 'comment' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'comment', result_mod_27048)
        # SSA join for if statement (line 125)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 121)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 127):
        
        # Assigning a Name to a Subscript (line 127):
        # Getting the type of 'comment' (line 127)
        comment_27049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 26), 'comment')
        # Getting the type of 'data' (line 127)
        data_27050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'data')
        str_27051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 13), 'str', 'comment')
        # Storing an element on a container (line 127)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 8), data_27050, (str_27051, comment_27049))
        
        # Getting the type of 'self' (line 129)
        self_27052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 11), 'self')
        # Obtaining the member 'sign' of a type (line 129)
        sign_27053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 11), self_27052, 'sign')
        # Testing the type of an if condition (line 129)
        if_condition_27054 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 8), sign_27053)
        # Assigning a type to the variable 'if_condition_27054' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'if_condition_27054', if_condition_27054)
        # SSA begins for if statement (line 129)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Subscript (line 130):
        
        # Assigning a Tuple to a Subscript (line 130):
        
        # Obtaining an instance of the builtin type 'tuple' (line 130)
        tuple_27055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 130)
        # Adding element type (line 130)
        
        # Call to basename(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'filename' (line 130)
        filename_27059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 54), 'filename', False)
        # Processing the call keyword arguments (line 130)
        kwargs_27060 = {}
        # Getting the type of 'os' (line 130)
        os_27056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 37), 'os', False)
        # Obtaining the member 'path' of a type (line 130)
        path_27057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 37), os_27056, 'path')
        # Obtaining the member 'basename' of a type (line 130)
        basename_27058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 37), path_27057, 'basename')
        # Calling basename(args, kwargs) (line 130)
        basename_call_result_27061 = invoke(stypy.reporting.localization.Localization(__file__, 130, 37), basename_27058, *[filename_27059], **kwargs_27060)
        
        str_27062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 66), 'str', '.asc')
        # Applying the binary operator '+' (line 130)
        result_add_27063 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 37), '+', basename_call_result_27061, str_27062)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 37), tuple_27055, result_add_27063)
        # Adding element type (line 130)
        
        # Call to read(...): (line 131)
        # Processing the call keyword arguments (line 131)
        kwargs_27071 = {}
        
        # Call to open(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'filename' (line 131)
        filename_27065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 42), 'filename', False)
        str_27066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 51), 'str', '.asc')
        # Applying the binary operator '+' (line 131)
        result_add_27067 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 42), '+', filename_27065, str_27066)
        
        # Processing the call keyword arguments (line 131)
        kwargs_27068 = {}
        # Getting the type of 'open' (line 131)
        open_27064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 37), 'open', False)
        # Calling open(args, kwargs) (line 131)
        open_call_result_27069 = invoke(stypy.reporting.localization.Localization(__file__, 131, 37), open_27064, *[result_add_27067], **kwargs_27068)
        
        # Obtaining the member 'read' of a type (line 131)
        read_27070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 37), open_call_result_27069, 'read')
        # Calling read(args, kwargs) (line 131)
        read_call_result_27072 = invoke(stypy.reporting.localization.Localization(__file__, 131, 37), read_27070, *[], **kwargs_27071)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 37), tuple_27055, read_call_result_27072)
        
        # Getting the type of 'data' (line 130)
        data_27073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'data')
        str_27074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 17), 'str', 'gpg_signature')
        # Storing an element on a container (line 130)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 12), data_27073, (str_27074, tuple_27055))
        # SSA join for if statement (line 129)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 134):
        
        # Assigning a BinOp to a Name (line 134):
        str_27075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 15), 'str', 'Basic ')
        
        # Call to standard_b64encode(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'self' (line 134)
        self_27077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 45), 'self', False)
        # Obtaining the member 'username' of a type (line 134)
        username_27078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 45), self_27077, 'username')
        str_27079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 61), 'str', ':')
        # Applying the binary operator '+' (line 134)
        result_add_27080 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 45), '+', username_27078, str_27079)
        
        # Getting the type of 'self' (line 135)
        self_27081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 45), 'self', False)
        # Obtaining the member 'password' of a type (line 135)
        password_27082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 45), self_27081, 'password')
        # Applying the binary operator '+' (line 134)
        result_add_27083 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 65), '+', result_add_27080, password_27082)
        
        # Processing the call keyword arguments (line 134)
        kwargs_27084 = {}
        # Getting the type of 'standard_b64encode' (line 134)
        standard_b64encode_27076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 26), 'standard_b64encode', False)
        # Calling standard_b64encode(args, kwargs) (line 134)
        standard_b64encode_call_result_27085 = invoke(stypy.reporting.localization.Localization(__file__, 134, 26), standard_b64encode_27076, *[result_add_27083], **kwargs_27084)
        
        # Applying the binary operator '+' (line 134)
        result_add_27086 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 15), '+', str_27075, standard_b64encode_call_result_27085)
        
        # Assigning a type to the variable 'auth' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'auth', result_add_27086)
        
        # Assigning a Str to a Name (line 138):
        
        # Assigning a Str to a Name (line 138):
        str_27087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 19), 'str', '--------------GHSKFJDLGDS7543FJKLFHRE75642756743254')
        # Assigning a type to the variable 'boundary' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'boundary', str_27087)
        
        # Assigning a BinOp to a Name (line 139):
        
        # Assigning a BinOp to a Name (line 139):
        str_27088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 23), 'str', '\r\n--')
        # Getting the type of 'boundary' (line 139)
        boundary_27089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 34), 'boundary')
        # Applying the binary operator '+' (line 139)
        result_add_27090 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 23), '+', str_27088, boundary_27089)
        
        # Assigning a type to the variable 'sep_boundary' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'sep_boundary', result_add_27090)
        
        # Assigning a BinOp to a Name (line 140):
        
        # Assigning a BinOp to a Name (line 140):
        # Getting the type of 'sep_boundary' (line 140)
        sep_boundary_27091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 23), 'sep_boundary')
        str_27092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 38), 'str', '--\r\n')
        # Applying the binary operator '+' (line 140)
        result_add_27093 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 23), '+', sep_boundary_27091, str_27092)
        
        # Assigning a type to the variable 'end_boundary' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'end_boundary', result_add_27093)
        
        # Assigning a Call to a Name (line 141):
        
        # Assigning a Call to a Name (line 141):
        
        # Call to StringIO(...): (line 141)
        # Processing the call keyword arguments (line 141)
        kwargs_27096 = {}
        # Getting the type of 'StringIO' (line 141)
        StringIO_27094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'StringIO', False)
        # Obtaining the member 'StringIO' of a type (line 141)
        StringIO_27095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 15), StringIO_27094, 'StringIO')
        # Calling StringIO(args, kwargs) (line 141)
        StringIO_call_result_27097 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), StringIO_27095, *[], **kwargs_27096)
        
        # Assigning a type to the variable 'body' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'body', StringIO_call_result_27097)
        
        
        # Call to items(...): (line 142)
        # Processing the call keyword arguments (line 142)
        kwargs_27100 = {}
        # Getting the type of 'data' (line 142)
        data_27098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 26), 'data', False)
        # Obtaining the member 'items' of a type (line 142)
        items_27099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 26), data_27098, 'items')
        # Calling items(args, kwargs) (line 142)
        items_call_result_27101 = invoke(stypy.reporting.localization.Localization(__file__, 142, 26), items_27099, *[], **kwargs_27100)
        
        # Testing the type of a for loop iterable (line 142)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 142, 8), items_call_result_27101)
        # Getting the type of the for loop variable (line 142)
        for_loop_var_27102 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 142, 8), items_call_result_27101)
        # Assigning a type to the variable 'key' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 8), for_loop_var_27102))
        # Assigning a type to the variable 'value' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 8), for_loop_var_27102))
        # SSA begins for a for statement (line 142)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 144)
        # Getting the type of 'list' (line 144)
        list_27103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 37), 'list')
        # Getting the type of 'value' (line 144)
        value_27104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 30), 'value')
        
        (may_be_27105, more_types_in_union_27106) = may_not_be_subtype(list_27103, value_27104)

        if may_be_27105:

            if more_types_in_union_27106:
                # Runtime conditional SSA (line 144)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'value' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'value', remove_subtype_from_union(value_27104, list))
            
            # Assigning a List to a Name (line 145):
            
            # Assigning a List to a Name (line 145):
            
            # Obtaining an instance of the builtin type 'list' (line 145)
            list_27107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 24), 'list')
            # Adding type elements to the builtin type 'list' instance (line 145)
            # Adding element type (line 145)
            # Getting the type of 'value' (line 145)
            value_27108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'value')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 24), list_27107, value_27108)
            
            # Assigning a type to the variable 'value' (line 145)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'value', list_27107)

            if more_types_in_union_27106:
                # SSA join for if statement (line 144)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'value' (line 146)
        value_27109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 25), 'value')
        # Testing the type of a for loop iterable (line 146)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 146, 12), value_27109)
        # Getting the type of the for loop variable (line 146)
        for_loop_var_27110 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 146, 12), value_27109)
        # Assigning a type to the variable 'value' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'value', for_loop_var_27110)
        # SSA begins for a for statement (line 146)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 147)
        # Getting the type of 'tuple' (line 147)
        tuple_27111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 37), 'tuple')
        # Getting the type of 'value' (line 147)
        value_27112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 30), 'value')
        
        (may_be_27113, more_types_in_union_27114) = may_be_subtype(tuple_27111, value_27112)

        if may_be_27113:

            if more_types_in_union_27114:
                # Runtime conditional SSA (line 147)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'value' (line 147)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'value', remove_not_subtype_from_union(value_27112, tuple))
            
            # Assigning a BinOp to a Name (line 148):
            
            # Assigning a BinOp to a Name (line 148):
            str_27115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 25), 'str', ';filename="%s"')
            
            # Obtaining the type of the subscript
            int_27116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 50), 'int')
            # Getting the type of 'value' (line 148)
            value_27117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 44), 'value')
            # Obtaining the member '__getitem__' of a type (line 148)
            getitem___27118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 44), value_27117, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 148)
            subscript_call_result_27119 = invoke(stypy.reporting.localization.Localization(__file__, 148, 44), getitem___27118, int_27116)
            
            # Applying the binary operator '%' (line 148)
            result_mod_27120 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 25), '%', str_27115, subscript_call_result_27119)
            
            # Assigning a type to the variable 'fn' (line 148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'fn', result_mod_27120)
            
            # Assigning a Subscript to a Name (line 149):
            
            # Assigning a Subscript to a Name (line 149):
            
            # Obtaining the type of the subscript
            int_27121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 34), 'int')
            # Getting the type of 'value' (line 149)
            value_27122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 28), 'value')
            # Obtaining the member '__getitem__' of a type (line 149)
            getitem___27123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 28), value_27122, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 149)
            subscript_call_result_27124 = invoke(stypy.reporting.localization.Localization(__file__, 149, 28), getitem___27123, int_27121)
            
            # Assigning a type to the variable 'value' (line 149)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'value', subscript_call_result_27124)

            if more_types_in_union_27114:
                # Runtime conditional SSA for else branch (line 147)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_27113) or more_types_in_union_27114):
            # Assigning a type to the variable 'value' (line 147)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'value', remove_subtype_from_union(value_27112, tuple))
            
            # Assigning a Str to a Name (line 151):
            
            # Assigning a Str to a Name (line 151):
            str_27125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 25), 'str', '')
            # Assigning a type to the variable 'fn' (line 151)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 20), 'fn', str_27125)

            if (may_be_27113 and more_types_in_union_27114):
                # SSA join for if statement (line 147)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to write(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'sep_boundary' (line 153)
        sep_boundary_27128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 27), 'sep_boundary', False)
        # Processing the call keyword arguments (line 153)
        kwargs_27129 = {}
        # Getting the type of 'body' (line 153)
        body_27126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'body', False)
        # Obtaining the member 'write' of a type (line 153)
        write_27127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 16), body_27126, 'write')
        # Calling write(args, kwargs) (line 153)
        write_call_result_27130 = invoke(stypy.reporting.localization.Localization(__file__, 153, 16), write_27127, *[sep_boundary_27128], **kwargs_27129)
        
        
        # Call to write(...): (line 154)
        # Processing the call arguments (line 154)
        str_27133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 27), 'str', '\r\nContent-Disposition: form-data; name="%s"')
        # Getting the type of 'key' (line 154)
        key_27134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 77), 'key', False)
        # Applying the binary operator '%' (line 154)
        result_mod_27135 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 27), '%', str_27133, key_27134)
        
        # Processing the call keyword arguments (line 154)
        kwargs_27136 = {}
        # Getting the type of 'body' (line 154)
        body_27131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'body', False)
        # Obtaining the member 'write' of a type (line 154)
        write_27132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 16), body_27131, 'write')
        # Calling write(args, kwargs) (line 154)
        write_call_result_27137 = invoke(stypy.reporting.localization.Localization(__file__, 154, 16), write_27132, *[result_mod_27135], **kwargs_27136)
        
        
        # Call to write(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'fn' (line 155)
        fn_27140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 27), 'fn', False)
        # Processing the call keyword arguments (line 155)
        kwargs_27141 = {}
        # Getting the type of 'body' (line 155)
        body_27138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'body', False)
        # Obtaining the member 'write' of a type (line 155)
        write_27139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 16), body_27138, 'write')
        # Calling write(args, kwargs) (line 155)
        write_call_result_27142 = invoke(stypy.reporting.localization.Localization(__file__, 155, 16), write_27139, *[fn_27140], **kwargs_27141)
        
        
        # Call to write(...): (line 156)
        # Processing the call arguments (line 156)
        str_27145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 27), 'str', '\r\n\r\n')
        # Processing the call keyword arguments (line 156)
        kwargs_27146 = {}
        # Getting the type of 'body' (line 156)
        body_27143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'body', False)
        # Obtaining the member 'write' of a type (line 156)
        write_27144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 16), body_27143, 'write')
        # Calling write(args, kwargs) (line 156)
        write_call_result_27147 = invoke(stypy.reporting.localization.Localization(__file__, 156, 16), write_27144, *[str_27145], **kwargs_27146)
        
        
        # Call to write(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'value' (line 157)
        value_27150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 27), 'value', False)
        # Processing the call keyword arguments (line 157)
        kwargs_27151 = {}
        # Getting the type of 'body' (line 157)
        body_27148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'body', False)
        # Obtaining the member 'write' of a type (line 157)
        write_27149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 16), body_27148, 'write')
        # Calling write(args, kwargs) (line 157)
        write_call_result_27152 = invoke(stypy.reporting.localization.Localization(__file__, 157, 16), write_27149, *[value_27150], **kwargs_27151)
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'value' (line 158)
        value_27153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 19), 'value')
        
        
        # Obtaining the type of the subscript
        int_27154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 35), 'int')
        # Getting the type of 'value' (line 158)
        value_27155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 29), 'value')
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___27156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 29), value_27155, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_27157 = invoke(stypy.reporting.localization.Localization(__file__, 158, 29), getitem___27156, int_27154)
        
        str_27158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 42), 'str', '\r')
        # Applying the binary operator '==' (line 158)
        result_eq_27159 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 29), '==', subscript_call_result_27157, str_27158)
        
        # Applying the binary operator 'and' (line 158)
        result_and_keyword_27160 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 19), 'and', value_27153, result_eq_27159)
        
        # Testing the type of an if condition (line 158)
        if_condition_27161 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 16), result_and_keyword_27160)
        # Assigning a type to the variable 'if_condition_27161' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'if_condition_27161', if_condition_27161)
        # SSA begins for if statement (line 158)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 159)
        # Processing the call arguments (line 159)
        str_27164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 31), 'str', '\n')
        # Processing the call keyword arguments (line 159)
        kwargs_27165 = {}
        # Getting the type of 'body' (line 159)
        body_27162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 20), 'body', False)
        # Obtaining the member 'write' of a type (line 159)
        write_27163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 20), body_27162, 'write')
        # Calling write(args, kwargs) (line 159)
        write_call_result_27166 = invoke(stypy.reporting.localization.Localization(__file__, 159, 20), write_27163, *[str_27164], **kwargs_27165)
        
        # SSA join for if statement (line 158)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'end_boundary' (line 160)
        end_boundary_27169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 19), 'end_boundary', False)
        # Processing the call keyword arguments (line 160)
        kwargs_27170 = {}
        # Getting the type of 'body' (line 160)
        body_27167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'body', False)
        # Obtaining the member 'write' of a type (line 160)
        write_27168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), body_27167, 'write')
        # Calling write(args, kwargs) (line 160)
        write_call_result_27171 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), write_27168, *[end_boundary_27169], **kwargs_27170)
        
        
        # Assigning a Call to a Name (line 161):
        
        # Assigning a Call to a Name (line 161):
        
        # Call to getvalue(...): (line 161)
        # Processing the call keyword arguments (line 161)
        kwargs_27174 = {}
        # Getting the type of 'body' (line 161)
        body_27172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'body', False)
        # Obtaining the member 'getvalue' of a type (line 161)
        getvalue_27173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 15), body_27172, 'getvalue')
        # Calling getvalue(args, kwargs) (line 161)
        getvalue_call_result_27175 = invoke(stypy.reporting.localization.Localization(__file__, 161, 15), getvalue_27173, *[], **kwargs_27174)
        
        # Assigning a type to the variable 'body' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'body', getvalue_call_result_27175)
        
        # Call to announce(...): (line 163)
        # Processing the call arguments (line 163)
        str_27178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 22), 'str', 'Submitting %s to %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 163)
        tuple_27179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 163)
        # Adding element type (line 163)
        # Getting the type of 'filename' (line 163)
        filename_27180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 47), 'filename', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 47), tuple_27179, filename_27180)
        # Adding element type (line 163)
        # Getting the type of 'self' (line 163)
        self_27181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 57), 'self', False)
        # Obtaining the member 'repository' of a type (line 163)
        repository_27182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 57), self_27181, 'repository')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 47), tuple_27179, repository_27182)
        
        # Applying the binary operator '%' (line 163)
        result_mod_27183 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 22), '%', str_27178, tuple_27179)
        
        # Getting the type of 'log' (line 163)
        log_27184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 75), 'log', False)
        # Obtaining the member 'INFO' of a type (line 163)
        INFO_27185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 75), log_27184, 'INFO')
        # Processing the call keyword arguments (line 163)
        kwargs_27186 = {}
        # Getting the type of 'self' (line 163)
        self_27176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'self', False)
        # Obtaining the member 'announce' of a type (line 163)
        announce_27177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), self_27176, 'announce')
        # Calling announce(args, kwargs) (line 163)
        announce_call_result_27187 = invoke(stypy.reporting.localization.Localization(__file__, 163, 8), announce_27177, *[result_mod_27183, INFO_27185], **kwargs_27186)
        
        
        # Assigning a Dict to a Name (line 166):
        
        # Assigning a Dict to a Name (line 166):
        
        # Obtaining an instance of the builtin type 'dict' (line 166)
        dict_27188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 18), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 166)
        # Adding element type (key, value) (line 166)
        str_27189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 19), 'str', 'Content-type')
        str_27190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 24), 'str', 'multipart/form-data; boundary=%s')
        # Getting the type of 'boundary' (line 167)
        boundary_27191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 61), 'boundary')
        # Applying the binary operator '%' (line 167)
        result_mod_27192 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 24), '%', str_27190, boundary_27191)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 18), dict_27188, (str_27189, result_mod_27192))
        # Adding element type (key, value) (line 166)
        str_27193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 19), 'str', 'Content-length')
        
        # Call to str(...): (line 168)
        # Processing the call arguments (line 168)
        
        # Call to len(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'body' (line 168)
        body_27196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 45), 'body', False)
        # Processing the call keyword arguments (line 168)
        kwargs_27197 = {}
        # Getting the type of 'len' (line 168)
        len_27195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 41), 'len', False)
        # Calling len(args, kwargs) (line 168)
        len_call_result_27198 = invoke(stypy.reporting.localization.Localization(__file__, 168, 41), len_27195, *[body_27196], **kwargs_27197)
        
        # Processing the call keyword arguments (line 168)
        kwargs_27199 = {}
        # Getting the type of 'str' (line 168)
        str_27194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 37), 'str', False)
        # Calling str(args, kwargs) (line 168)
        str_call_result_27200 = invoke(stypy.reporting.localization.Localization(__file__, 168, 37), str_27194, *[len_call_result_27198], **kwargs_27199)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 18), dict_27188, (str_27193, str_call_result_27200))
        # Adding element type (key, value) (line 166)
        str_27201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 19), 'str', 'Authorization')
        # Getting the type of 'auth' (line 169)
        auth_27202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 36), 'auth')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 18), dict_27188, (str_27201, auth_27202))
        
        # Assigning a type to the variable 'headers' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'headers', dict_27188)
        
        # Assigning a Call to a Name (line 171):
        
        # Assigning a Call to a Name (line 171):
        
        # Call to Request(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'self' (line 171)
        self_27204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 26), 'self', False)
        # Obtaining the member 'repository' of a type (line 171)
        repository_27205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 26), self_27204, 'repository')
        # Processing the call keyword arguments (line 171)
        # Getting the type of 'body' (line 171)
        body_27206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 48), 'body', False)
        keyword_27207 = body_27206
        # Getting the type of 'headers' (line 172)
        headers_27208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 34), 'headers', False)
        keyword_27209 = headers_27208
        kwargs_27210 = {'headers': keyword_27209, 'data': keyword_27207}
        # Getting the type of 'Request' (line 171)
        Request_27203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 18), 'Request', False)
        # Calling Request(args, kwargs) (line 171)
        Request_call_result_27211 = invoke(stypy.reporting.localization.Localization(__file__, 171, 18), Request_27203, *[repository_27205], **kwargs_27210)
        
        # Assigning a type to the variable 'request' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'request', Request_call_result_27211)
        
        
        # SSA begins for try-except statement (line 174)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 175):
        
        # Assigning a Call to a Name (line 175):
        
        # Call to urlopen(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'request' (line 175)
        request_27213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 29), 'request', False)
        # Processing the call keyword arguments (line 175)
        kwargs_27214 = {}
        # Getting the type of 'urlopen' (line 175)
        urlopen_27212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 21), 'urlopen', False)
        # Calling urlopen(args, kwargs) (line 175)
        urlopen_call_result_27215 = invoke(stypy.reporting.localization.Localization(__file__, 175, 21), urlopen_27212, *[request_27213], **kwargs_27214)
        
        # Assigning a type to the variable 'result' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'result', urlopen_call_result_27215)
        
        # Assigning a Call to a Name (line 176):
        
        # Assigning a Call to a Name (line 176):
        
        # Call to getcode(...): (line 176)
        # Processing the call keyword arguments (line 176)
        kwargs_27218 = {}
        # Getting the type of 'result' (line 176)
        result_27216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 21), 'result', False)
        # Obtaining the member 'getcode' of a type (line 176)
        getcode_27217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 21), result_27216, 'getcode')
        # Calling getcode(args, kwargs) (line 176)
        getcode_call_result_27219 = invoke(stypy.reporting.localization.Localization(__file__, 176, 21), getcode_27217, *[], **kwargs_27218)
        
        # Assigning a type to the variable 'status' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'status', getcode_call_result_27219)
        
        # Assigning a Attribute to a Name (line 177):
        
        # Assigning a Attribute to a Name (line 177):
        # Getting the type of 'result' (line 177)
        result_27220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 21), 'result')
        # Obtaining the member 'msg' of a type (line 177)
        msg_27221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 21), result_27220, 'msg')
        # Assigning a type to the variable 'reason' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'reason', msg_27221)
        
        # Getting the type of 'self' (line 178)
        self_27222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 15), 'self')
        # Obtaining the member 'show_response' of a type (line 178)
        show_response_27223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 15), self_27222, 'show_response')
        # Testing the type of an if condition (line 178)
        if_condition_27224 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 12), show_response_27223)
        # Assigning a type to the variable 'if_condition_27224' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'if_condition_27224', if_condition_27224)
        # SSA begins for if statement (line 178)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 179):
        
        # Assigning a Call to a Name (line 179):
        
        # Call to join(...): (line 179)
        # Processing the call arguments (line 179)
        
        # Obtaining an instance of the builtin type 'tuple' (line 179)
        tuple_27227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 179)
        # Adding element type (line 179)
        str_27228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 33), 'str', '-')
        int_27229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 39), 'int')
        # Applying the binary operator '*' (line 179)
        result_mul_27230 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 33), '*', str_27228, int_27229)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 33), tuple_27227, result_mul_27230)
        # Adding element type (line 179)
        
        # Call to read(...): (line 179)
        # Processing the call keyword arguments (line 179)
        kwargs_27233 = {}
        # Getting the type of 'result' (line 179)
        result_27231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 43), 'result', False)
        # Obtaining the member 'read' of a type (line 179)
        read_27232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 43), result_27231, 'read')
        # Calling read(args, kwargs) (line 179)
        read_call_result_27234 = invoke(stypy.reporting.localization.Localization(__file__, 179, 43), read_27232, *[], **kwargs_27233)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 33), tuple_27227, read_call_result_27234)
        # Adding element type (line 179)
        str_27235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 58), 'str', '-')
        int_27236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 64), 'int')
        # Applying the binary operator '*' (line 179)
        result_mul_27237 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 58), '*', str_27235, int_27236)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 33), tuple_27227, result_mul_27237)
        
        # Processing the call keyword arguments (line 179)
        kwargs_27238 = {}
        str_27225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 22), 'str', '\n')
        # Obtaining the member 'join' of a type (line 179)
        join_27226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 22), str_27225, 'join')
        # Calling join(args, kwargs) (line 179)
        join_call_result_27239 = invoke(stypy.reporting.localization.Localization(__file__, 179, 22), join_27226, *[tuple_27227], **kwargs_27238)
        
        # Assigning a type to the variable 'msg' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'msg', join_call_result_27239)
        
        # Call to announce(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'msg' (line 180)
        msg_27242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 30), 'msg', False)
        # Getting the type of 'log' (line 180)
        log_27243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 35), 'log', False)
        # Obtaining the member 'INFO' of a type (line 180)
        INFO_27244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 35), log_27243, 'INFO')
        # Processing the call keyword arguments (line 180)
        kwargs_27245 = {}
        # Getting the type of 'self' (line 180)
        self_27240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'self', False)
        # Obtaining the member 'announce' of a type (line 180)
        announce_27241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 16), self_27240, 'announce')
        # Calling announce(args, kwargs) (line 180)
        announce_call_result_27246 = invoke(stypy.reporting.localization.Localization(__file__, 180, 16), announce_27241, *[msg_27242, INFO_27244], **kwargs_27245)
        
        # SSA join for if statement (line 178)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except part of a try statement (line 174)
        # SSA branch for the except 'Attribute' branch of a try statement (line 174)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'socket' (line 181)
        socket_27247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 15), 'socket')
        # Obtaining the member 'error' of a type (line 181)
        error_27248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 15), socket_27247, 'error')
        # Assigning a type to the variable 'e' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'e', error_27248)
        
        # Call to announce(...): (line 182)
        # Processing the call arguments (line 182)
        
        # Call to str(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'e' (line 182)
        e_27252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 30), 'e', False)
        # Processing the call keyword arguments (line 182)
        kwargs_27253 = {}
        # Getting the type of 'str' (line 182)
        str_27251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 26), 'str', False)
        # Calling str(args, kwargs) (line 182)
        str_call_result_27254 = invoke(stypy.reporting.localization.Localization(__file__, 182, 26), str_27251, *[e_27252], **kwargs_27253)
        
        # Getting the type of 'log' (line 182)
        log_27255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 34), 'log', False)
        # Obtaining the member 'ERROR' of a type (line 182)
        ERROR_27256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 34), log_27255, 'ERROR')
        # Processing the call keyword arguments (line 182)
        kwargs_27257 = {}
        # Getting the type of 'self' (line 182)
        self_27249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'self', False)
        # Obtaining the member 'announce' of a type (line 182)
        announce_27250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 12), self_27249, 'announce')
        # Calling announce(args, kwargs) (line 182)
        announce_call_result_27258 = invoke(stypy.reporting.localization.Localization(__file__, 182, 12), announce_27250, *[str_call_result_27254, ERROR_27256], **kwargs_27257)
        
        # SSA branch for the except 'HTTPError' branch of a try statement (line 174)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'HTTPError' (line 184)
        HTTPError_27259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 15), 'HTTPError')
        # Assigning a type to the variable 'e' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'e', HTTPError_27259)
        
        # Assigning a Attribute to a Name (line 185):
        
        # Assigning a Attribute to a Name (line 185):
        # Getting the type of 'e' (line 185)
        e_27260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 21), 'e')
        # Obtaining the member 'code' of a type (line 185)
        code_27261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 21), e_27260, 'code')
        # Assigning a type to the variable 'status' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'status', code_27261)
        
        # Assigning a Attribute to a Name (line 186):
        
        # Assigning a Attribute to a Name (line 186):
        # Getting the type of 'e' (line 186)
        e_27262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 21), 'e')
        # Obtaining the member 'msg' of a type (line 186)
        msg_27263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 21), e_27262, 'msg')
        # Assigning a type to the variable 'reason' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'reason', msg_27263)
        # SSA join for try-except statement (line 174)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'status' (line 188)
        status_27264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 11), 'status')
        int_27265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 21), 'int')
        # Applying the binary operator '==' (line 188)
        result_eq_27266 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 11), '==', status_27264, int_27265)
        
        # Testing the type of an if condition (line 188)
        if_condition_27267 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 8), result_eq_27266)
        # Assigning a type to the variable 'if_condition_27267' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'if_condition_27267', if_condition_27267)
        # SSA begins for if statement (line 188)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to announce(...): (line 189)
        # Processing the call arguments (line 189)
        str_27270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 26), 'str', 'Server response (%s): %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 189)
        tuple_27271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 189)
        # Adding element type (line 189)
        # Getting the type of 'status' (line 189)
        status_27272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 56), 'status', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 56), tuple_27271, status_27272)
        # Adding element type (line 189)
        # Getting the type of 'reason' (line 189)
        reason_27273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 64), 'reason', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 56), tuple_27271, reason_27273)
        
        # Applying the binary operator '%' (line 189)
        result_mod_27274 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 26), '%', str_27270, tuple_27271)
        
        # Getting the type of 'log' (line 190)
        log_27275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 26), 'log', False)
        # Obtaining the member 'INFO' of a type (line 190)
        INFO_27276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 26), log_27275, 'INFO')
        # Processing the call keyword arguments (line 189)
        kwargs_27277 = {}
        # Getting the type of 'self' (line 189)
        self_27268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'self', False)
        # Obtaining the member 'announce' of a type (line 189)
        announce_27269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 12), self_27268, 'announce')
        # Calling announce(args, kwargs) (line 189)
        announce_call_result_27278 = invoke(stypy.reporting.localization.Localization(__file__, 189, 12), announce_27269, *[result_mod_27274, INFO_27276], **kwargs_27277)
        
        # SSA branch for the else part of an if statement (line 188)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 192):
        
        # Assigning a BinOp to a Name (line 192):
        str_27279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 18), 'str', 'Upload failed (%s): %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 192)
        tuple_27280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 192)
        # Adding element type (line 192)
        # Getting the type of 'status' (line 192)
        status_27281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 46), 'status')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 46), tuple_27280, status_27281)
        # Adding element type (line 192)
        # Getting the type of 'reason' (line 192)
        reason_27282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 54), 'reason')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 46), tuple_27280, reason_27282)
        
        # Applying the binary operator '%' (line 192)
        result_mod_27283 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 18), '%', str_27279, tuple_27280)
        
        # Assigning a type to the variable 'msg' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'msg', result_mod_27283)
        
        # Call to announce(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'msg' (line 193)
        msg_27286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 26), 'msg', False)
        # Getting the type of 'log' (line 193)
        log_27287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 31), 'log', False)
        # Obtaining the member 'ERROR' of a type (line 193)
        ERROR_27288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 31), log_27287, 'ERROR')
        # Processing the call keyword arguments (line 193)
        kwargs_27289 = {}
        # Getting the type of 'self' (line 193)
        self_27284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'self', False)
        # Obtaining the member 'announce' of a type (line 193)
        announce_27285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), self_27284, 'announce')
        # Calling announce(args, kwargs) (line 193)
        announce_call_result_27290 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), announce_27285, *[msg_27286, ERROR_27288], **kwargs_27289)
        
        
        # Call to DistutilsError(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'msg' (line 194)
        msg_27292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 33), 'msg', False)
        # Processing the call keyword arguments (line 194)
        kwargs_27293 = {}
        # Getting the type of 'DistutilsError' (line 194)
        DistutilsError_27291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 18), 'DistutilsError', False)
        # Calling DistutilsError(args, kwargs) (line 194)
        DistutilsError_call_result_27294 = invoke(stypy.reporting.localization.Localization(__file__, 194, 18), DistutilsError_27291, *[msg_27292], **kwargs_27293)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 194, 12), DistutilsError_call_result_27294, 'raise parameter', BaseException)
        # SSA join for if statement (line 188)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'upload_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'upload_file' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_27295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27295)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'upload_file'
        return stypy_return_type_27295


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'upload.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'upload' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'upload', upload)

# Assigning a Str to a Name (line 20):
str_27296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 18), 'str', 'upload binary package to PyPI')
# Getting the type of 'upload'
upload_27297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'upload')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), upload_27297, 'description', str_27296)

# Assigning a BinOp to a Name (line 22):
# Getting the type of 'PyPIRCCommand' (line 22)
PyPIRCCommand_27298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 19), 'PyPIRCCommand')
# Obtaining the member 'user_options' of a type (line 22)
user_options_27299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 19), PyPIRCCommand_27298, 'user_options')

# Obtaining an instance of the builtin type 'list' (line 22)
list_27300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 48), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_27301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)
# Adding element type (line 23)
str_27302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'str', 'sign')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_27301, str_27302)
# Adding element type (line 23)
str_27303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 17), 'str', 's')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_27301, str_27303)
# Adding element type (line 23)
str_27304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'str', 'sign files to upload using gpg')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_27301, str_27304)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 48), list_27300, tuple_27301)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'tuple' (line 25)
tuple_27305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 25)
# Adding element type (line 25)
str_27306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 9), 'str', 'identity=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 9), tuple_27305, str_27306)
# Adding element type (line 25)
str_27307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 22), 'str', 'i')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 9), tuple_27305, str_27307)
# Adding element type (line 25)
str_27308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 27), 'str', 'GPG identity used to sign files')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 9), tuple_27305, str_27308)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 48), list_27300, tuple_27305)

# Applying the binary operator '+' (line 22)
result_add_27309 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 19), '+', user_options_27299, list_27300)

# Getting the type of 'upload'
upload_27310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'upload')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), upload_27310, 'user_options', result_add_27309)

# Assigning a BinOp to a Name (line 28):
# Getting the type of 'PyPIRCCommand' (line 28)
PyPIRCCommand_27311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 22), 'PyPIRCCommand')
# Obtaining the member 'boolean_options' of a type (line 28)
boolean_options_27312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 22), PyPIRCCommand_27311, 'boolean_options')

# Obtaining an instance of the builtin type 'list' (line 28)
list_27313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 54), 'list')
# Adding type elements to the builtin type 'list' instance (line 28)
# Adding element type (line 28)
str_27314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 55), 'str', 'sign')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 54), list_27313, str_27314)

# Applying the binary operator '+' (line 28)
result_add_27315 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 22), '+', boolean_options_27312, list_27313)

# Getting the type of 'upload'
upload_27316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'upload')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), upload_27316, 'boolean_options', result_add_27315)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
