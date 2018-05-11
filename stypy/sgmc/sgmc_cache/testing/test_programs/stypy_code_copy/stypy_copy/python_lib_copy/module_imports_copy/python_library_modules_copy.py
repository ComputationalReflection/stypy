
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Output of the help("modules") command over a raw Python 2.7.9 installation. This table is used to determine if a
3: module is part of the Python library or it is an user-defined module.
4: '''
5: 
6: python_library_modules_str = '''
7: BaseHTTPServer      anydbm              imaplib             sgmllib
8: Bastion             argparse            imghdr              sha
9: CGIHTTPServer       array               imp                 shelve
10: Canvas              ast                 importlib           shlex
11: ConfigParser        asynchat            imputil             shutil
12: Cookie              asyncore            inspect             signal
13: Dialog              atexit              io                  site
14: DocXMLRPCServer     audiodev            itertools           smtpd
15: FileDialog          audioop             json                smtplib
16: FixTk               base64              keyword             sndhdr
17: HTMLParser          bdb                 lib2to3             socket
18: MimeWriter          binascii            linecache           sqlite3
19: Queue               binhex              locale              sre
20: ScrolledText        bisect              logging             sre_compile
21: SimpleDialog        bsddb               macpath             sre_constants
22: SimpleHTTPServer    bz2                 macurl2path         sre_parse
23: SimpleXMLRPCServer  cPickle             mailbox             ssl
24: SocketServer        cProfile            mailcap             stat
25: StringIO            cStringIO           markupbase          statvfs
26: Tix                 calendar            marshal             string
27: Tkconstants         cgi                 math                stringold
28: Tkdnd               cgitb               md5                 stringprep
29: Tkinter             chunk               mhlib               strop
30: UserDict            cmath               mimetools           struct
31: UserList            cmd                 mimetypes
32: UserString          code                mimify              subprocess
33: _LWPCookieJar       codecs              mmap                sunau
34: _MozillaCookieJar   codeop              modulefinder        sunaudio
35: __builtin__         collections         msilib              symbol
36: __future__          colorsys            msvcrt              symtable
37: _abcoll             commands            multifile           sys
38: _ast                compileall          multiprocessing     sysconfig
39: _bisect             compiler            mutex               tabnanny
40: _bsddb              contextlib          netrc               tarfile
41: _codecs             cookielib           new                 telnetlib
42: _codecs_cn          copy                nntplib             tempfile
43: _codecs_hk          copy_reg            nt                  test
44: _codecs_iso2022     csv                 ntpath              tests
45: _codecs_jp          ctypes              nturl2path          textwrap
46: _codecs_kr          curses              numbers             this
47: _codecs_tw          datetime            opcode              thread
48: _collections        dbhash              operator            threading
49: _csv                decimal             optparse            time
50: _ctypes             difflib             os                  timeit
51: _ctypes_test        dircache            os2emxpath          tkColorChooser
52: _elementtree        dis                 parser              tkCommonDialog
53: _functools          distutils           pdb                 tkFileDialog
54: _hashlib            doctest             pickle              tkFont
55: _heapq              dumbdbm             pickletools         tkMessageBox
56: _hotshot            dummy_thread        pip                 tkSimpleDialog
57: _io                 dummy_threading     pipes               toaiff
58: _json               easy_install        pkg_resources       token
59: _locale             email               pkgutil             tokenize
60: _lsprof             encodings           platform            trace
61: _markerlib          ensurepip           plistlib            traceback
62: _md5                errno               popen2              ttk
63: _msi                exceptions          poplib              tty
64: _multibytecodec     filecmp             posixfile           turtle
65: _multiprocessing    fileinput           posixpath           types
66: _osx_support        fnmatch             pprint              unicodedata
67: _pyio               formatter           profile             unit_testing
68: _random             fpformat            program             unittest
69: _sha                fractions           pstats              urllib
70: _sha256             ftplib              pty                 urllib2
71: _sha512             functools           py_compile          urlparse
72: _socket             future_builtins     pyclbr              user
73: _sqlite3            gc                  pydoc               uu
74: _sre                genericpath         pydoc_data          uuid
75: _ssl                getopt              pyexpat             warnings
76: _strptime           getpass             wave
77: _struct             gettext             quopri              weakref
78: _subprocess         glob                random              webbrowser
79: _symtable           gzip                re                  whichdb
80: _testcapi           hashlib             repr                winsound
81: _threading_local    heapq               rexec               wsgiref
82: _tkinter            hmac                rfc822              xdrlib
83: _warnings           hotshot             rlcompleter         xml
84: _weakref            htmlentitydefs      robotparser         xmllib
85: _weakrefset         htmllib             runpy               xmlrpclib
86: _winreg             httplib             sched               xxsubtype
87: abc                 idlelib             select              zipfile
88: aifc                ihooks              sets                zipimport
89: antigravity         imageop             setuptools          zlib
90: '''
91: 
92: python_library_modules = None
93: 
94: 
95: def is_python_library_module(module_name):
96:     '''
97:     Returns if the passed module name is part of the listed Python library modules
98:     :param module_name: module name
99:     :return: bool
100:     '''
101:     global python_library_modules
102: 
103:     if python_library_modules is None:
104:         temp = python_library_modules_str.replace("\n", " ")
105:         python_library_modules = temp.split(" ")
106:         python_library_modules = filter(lambda elem: not elem == "", python_library_modules)
107: 
108:     return module_name in python_library_modules
109: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_8668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', '\nOutput of the help("modules") command over a raw Python 2.7.9 installation. This table is used to determine if a\nmodule is part of the Python library or it is an user-defined module.\n')

# Assigning a Str to a Name (line 6):
str_8669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, (-1)), 'str', '\nBaseHTTPServer      anydbm              imaplib             sgmllib\nBastion             argparse            imghdr              sha\nCGIHTTPServer       array               imp                 shelve\nCanvas              ast                 importlib           shlex\nConfigParser        asynchat            imputil             shutil\nCookie              asyncore            inspect             signal\nDialog              atexit              io                  site\nDocXMLRPCServer     audiodev            itertools           smtpd\nFileDialog          audioop             json                smtplib\nFixTk               base64              keyword             sndhdr\nHTMLParser          bdb                 lib2to3             socket\nMimeWriter          binascii            linecache           sqlite3\nQueue               binhex              locale              sre\nScrolledText        bisect              logging             sre_compile\nSimpleDialog        bsddb               macpath             sre_constants\nSimpleHTTPServer    bz2                 macurl2path         sre_parse\nSimpleXMLRPCServer  cPickle             mailbox             ssl\nSocketServer        cProfile            mailcap             stat\nStringIO            cStringIO           markupbase          statvfs\nTix                 calendar            marshal             string\nTkconstants         cgi                 math                stringold\nTkdnd               cgitb               md5                 stringprep\nTkinter             chunk               mhlib               strop\nUserDict            cmath               mimetools           struct\nUserList            cmd                 mimetypes\nUserString          code                mimify              subprocess\n_LWPCookieJar       codecs              mmap                sunau\n_MozillaCookieJar   codeop              modulefinder        sunaudio\n__builtin__         collections         msilib              symbol\n__future__          colorsys            msvcrt              symtable\n_abcoll             commands            multifile           sys\n_ast                compileall          multiprocessing     sysconfig\n_bisect             compiler            mutex               tabnanny\n_bsddb              contextlib          netrc               tarfile\n_codecs             cookielib           new                 telnetlib\n_codecs_cn          copy                nntplib             tempfile\n_codecs_hk          copy_reg            nt                  test\n_codecs_iso2022     csv                 ntpath              tests\n_codecs_jp          ctypes              nturl2path          textwrap\n_codecs_kr          curses              numbers             this\n_codecs_tw          datetime            opcode              thread\n_collections        dbhash              operator            threading\n_csv                decimal             optparse            time\n_ctypes             difflib             os                  timeit\n_ctypes_test        dircache            os2emxpath          tkColorChooser\n_elementtree        dis                 parser              tkCommonDialog\n_functools          distutils           pdb                 tkFileDialog\n_hashlib            doctest             pickle              tkFont\n_heapq              dumbdbm             pickletools         tkMessageBox\n_hotshot            dummy_thread        pip                 tkSimpleDialog\n_io                 dummy_threading     pipes               toaiff\n_json               easy_install        pkg_resources       token\n_locale             email               pkgutil             tokenize\n_lsprof             encodings           platform            trace\n_markerlib          ensurepip           plistlib            traceback\n_md5                errno               popen2              ttk\n_msi                exceptions          poplib              tty\n_multibytecodec     filecmp             posixfile           turtle\n_multiprocessing    fileinput           posixpath           types\n_osx_support        fnmatch             pprint              unicodedata\n_pyio               formatter           profile             unit_testing\n_random             fpformat            program             unittest\n_sha                fractions           pstats              urllib\n_sha256             ftplib              pty                 urllib2\n_sha512             functools           py_compile          urlparse\n_socket             future_builtins     pyclbr              user\n_sqlite3            gc                  pydoc               uu\n_sre                genericpath         pydoc_data          uuid\n_ssl                getopt              pyexpat             warnings\n_strptime           getpass             wave\n_struct             gettext             quopri              weakref\n_subprocess         glob                random              webbrowser\n_symtable           gzip                re                  whichdb\n_testcapi           hashlib             repr                winsound\n_threading_local    heapq               rexec               wsgiref\n_tkinter            hmac                rfc822              xdrlib\n_warnings           hotshot             rlcompleter         xml\n_weakref            htmlentitydefs      robotparser         xmllib\n_weakrefset         htmllib             runpy               xmlrpclib\n_winreg             httplib             sched               xxsubtype\nabc                 idlelib             select              zipfile\naifc                ihooks              sets                zipimport\nantigravity         imageop             setuptools          zlib\n')
# Assigning a type to the variable 'python_library_modules_str' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'python_library_modules_str', str_8669)

# Assigning a Name to a Name (line 92):
# Getting the type of 'None' (line 92)
None_8670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'None')
# Assigning a type to the variable 'python_library_modules' (line 92)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'python_library_modules', None_8670)

@norecursion
def is_python_library_module(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_python_library_module'
    module_type_store = module_type_store.open_function_context('is_python_library_module', 95, 0, False)
    
    # Passed parameters checking function
    is_python_library_module.stypy_localization = localization
    is_python_library_module.stypy_type_of_self = None
    is_python_library_module.stypy_type_store = module_type_store
    is_python_library_module.stypy_function_name = 'is_python_library_module'
    is_python_library_module.stypy_param_names_list = ['module_name']
    is_python_library_module.stypy_varargs_param_name = None
    is_python_library_module.stypy_kwargs_param_name = None
    is_python_library_module.stypy_call_defaults = defaults
    is_python_library_module.stypy_call_varargs = varargs
    is_python_library_module.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_python_library_module', ['module_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_python_library_module', localization, ['module_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_python_library_module(...)' code ##################

    str_8671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, (-1)), 'str', '\n    Returns if the passed module name is part of the listed Python library modules\n    :param module_name: module name\n    :return: bool\n    ')
    # Marking variables as global (line 101)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 101, 4), 'python_library_modules')
    
    # Type idiom detected: calculating its left and rigth part (line 103)
    # Getting the type of 'python_library_modules' (line 103)
    python_library_modules_8672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 7), 'python_library_modules')
    # Getting the type of 'None' (line 103)
    None_8673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 33), 'None')
    
    (may_be_8674, more_types_in_union_8675) = may_be_none(python_library_modules_8672, None_8673)

    if may_be_8674:

        if more_types_in_union_8675:
            # Runtime conditional SSA (line 103)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 104):
        
        # Call to replace(...): (line 104)
        # Processing the call arguments (line 104)
        str_8678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 50), 'str', '\n')
        str_8679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 56), 'str', ' ')
        # Processing the call keyword arguments (line 104)
        kwargs_8680 = {}
        # Getting the type of 'python_library_modules_str' (line 104)
        python_library_modules_str_8676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'python_library_modules_str', False)
        # Obtaining the member 'replace' of a type (line 104)
        replace_8677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 15), python_library_modules_str_8676, 'replace')
        # Calling replace(args, kwargs) (line 104)
        replace_call_result_8681 = invoke(stypy.reporting.localization.Localization(__file__, 104, 15), replace_8677, *[str_8678, str_8679], **kwargs_8680)
        
        # Assigning a type to the variable 'temp' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'temp', replace_call_result_8681)
        
        # Assigning a Call to a Name (line 105):
        
        # Call to split(...): (line 105)
        # Processing the call arguments (line 105)
        str_8684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 44), 'str', ' ')
        # Processing the call keyword arguments (line 105)
        kwargs_8685 = {}
        # Getting the type of 'temp' (line 105)
        temp_8682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 33), 'temp', False)
        # Obtaining the member 'split' of a type (line 105)
        split_8683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 33), temp_8682, 'split')
        # Calling split(args, kwargs) (line 105)
        split_call_result_8686 = invoke(stypy.reporting.localization.Localization(__file__, 105, 33), split_8683, *[str_8684], **kwargs_8685)
        
        # Assigning a type to the variable 'python_library_modules' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'python_library_modules', split_call_result_8686)
        
        # Assigning a Call to a Name (line 106):
        
        # Call to filter(...): (line 106)
        # Processing the call arguments (line 106)

        @norecursion
        def _stypy_temp_lambda_17(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_17'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_17', 106, 40, True)
            # Passed parameters checking function
            _stypy_temp_lambda_17.stypy_localization = localization
            _stypy_temp_lambda_17.stypy_type_of_self = None
            _stypy_temp_lambda_17.stypy_type_store = module_type_store
            _stypy_temp_lambda_17.stypy_function_name = '_stypy_temp_lambda_17'
            _stypy_temp_lambda_17.stypy_param_names_list = ['elem']
            _stypy_temp_lambda_17.stypy_varargs_param_name = None
            _stypy_temp_lambda_17.stypy_kwargs_param_name = None
            _stypy_temp_lambda_17.stypy_call_defaults = defaults
            _stypy_temp_lambda_17.stypy_call_varargs = varargs
            _stypy_temp_lambda_17.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_17', ['elem'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_17', ['elem'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            
            # Getting the type of 'elem' (line 106)
            elem_8688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 57), 'elem', False)
            str_8689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 65), 'str', '')
            # Applying the binary operator '==' (line 106)
            result_eq_8690 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 57), '==', elem_8688, str_8689)
            
            # Applying the 'not' unary operator (line 106)
            result_not__8691 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 53), 'not', result_eq_8690)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 40), 'stypy_return_type', result_not__8691)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_17' in the type store
            # Getting the type of 'stypy_return_type' (line 106)
            stypy_return_type_8692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 40), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_8692)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_17'
            return stypy_return_type_8692

        # Assigning a type to the variable '_stypy_temp_lambda_17' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 40), '_stypy_temp_lambda_17', _stypy_temp_lambda_17)
        # Getting the type of '_stypy_temp_lambda_17' (line 106)
        _stypy_temp_lambda_17_8693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 40), '_stypy_temp_lambda_17')
        # Getting the type of 'python_library_modules' (line 106)
        python_library_modules_8694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 69), 'python_library_modules', False)
        # Processing the call keyword arguments (line 106)
        kwargs_8695 = {}
        # Getting the type of 'filter' (line 106)
        filter_8687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'filter', False)
        # Calling filter(args, kwargs) (line 106)
        filter_call_result_8696 = invoke(stypy.reporting.localization.Localization(__file__, 106, 33), filter_8687, *[_stypy_temp_lambda_17_8693, python_library_modules_8694], **kwargs_8695)
        
        # Assigning a type to the variable 'python_library_modules' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'python_library_modules', filter_call_result_8696)

        if more_types_in_union_8675:
            # SSA join for if statement (line 103)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'module_name' (line 108)
    module_name_8697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 11), 'module_name')
    # Getting the type of 'python_library_modules' (line 108)
    python_library_modules_8698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 26), 'python_library_modules')
    # Applying the binary operator 'in' (line 108)
    result_contains_8699 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 11), 'in', module_name_8697, python_library_modules_8698)
    
    # Assigning a type to the variable 'stypy_return_type' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type', result_contains_8699)
    
    # ################# End of 'is_python_library_module(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_python_library_module' in the type store
    # Getting the type of 'stypy_return_type' (line 95)
    stypy_return_type_8700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8700)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_python_library_module'
    return stypy_return_type_8700

# Assigning a type to the variable 'is_python_library_module' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'is_python_library_module', is_python_library_module)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
