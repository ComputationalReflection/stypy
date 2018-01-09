"""
Output of the help("modules") command over a raw Python 2.7.9 installation. This table is used to determine if a
module is part of the Python library or it is an user-defined module.
"""

python_library_modules_str = """
BaseHTTPServer      anydbm              imaplib             sgmllib
Bastion             argparse            imghdr              sha
CGIHTTPServer       array               imp                 shelve
Canvas              ast                 importlib           shlex
ConfigParser        asynchat            imputil             shutil
Cookie              asyncore            inspect             signal
Dialog              atexit              io                  site
DocXMLRPCServer     audiodev            itertools           smtpd
FileDialog          audioop             json                smtplib
FixTk               base64              keyword             sndhdr
HTMLParser          bdb                 lib2to3             socket
MimeWriter          binascii            linecache           sqlite3
Queue               binhex              locale              sre
ScrolledText        bisect              logging             sre_compile
SimpleDialog        bsddb               macpath             sre_constants
SimpleHTTPServer    bz2                 macurl2path         sre_parse
SimpleXMLRPCServer  cPickle             mailbox             ssl
SocketServer        cProfile            mailcap             stat
StringIO            cStringIO           markupbase          statvfs
Tix                 calendar            marshal             string
Tkconstants         cgi                 math                stringold
Tkdnd               cgitb               md5                 stringprep
Tkinter             chunk               mhlib               strop
UserDict            cmath               mimetools           struct
UserList            cmd                 mimetypes
UserString          code                mimify              subprocess
_LWPCookieJar       codecs              mmap                sunau
_MozillaCookieJar   codeop              modulefinder        sunaudio
__builtin__         collections         msilib              symbol
__future__          colorsys            msvcrt              symtable
_abcoll             commands            multifile           sys
_ast                compileall          multiprocessing     sysconfig
_bisect             compiler            mutex               tabnanny
_bsddb              contextlib          netrc               tarfile
_codecs             cookielib           new                 telnetlib
_codecs_cn          copy                nntplib             tempfile
_codecs_hk          copy_reg            nt                  test
_codecs_iso2022     csv                 ntpath              tests
_codecs_jp          ctypes              nturl2path          textwrap
_codecs_kr          curses              numbers             this
_codecs_tw          datetime            opcode              thread
_collections        dbhash              operator            threading
_csv                decimal             optparse            time
_ctypes             difflib             os                  timeit
_ctypes_test        dircache            os2emxpath          tkColorChooser
_elementtree        dis                 parser              tkCommonDialog
_functools          distutils           pdb                 tkFileDialog
_hashlib            doctest             pickle              tkFont
_heapq              dumbdbm             pickletools         tkMessageBox
_hotshot            dummy_thread        pip                 tkSimpleDialog
_io                 dummy_threading     pipes               toaiff
_json               easy_install        pkg_resources       token
_locale             email               pkgutil             tokenize
_lsprof             encodings           platform            trace
_markerlib          ensurepip           plistlib            traceback
_md5                errno               popen2              ttk
_msi                exceptions          poplib              tty
_multibytecodec     filecmp             posixfile           turtle
_multiprocessing    fileinput           posixpath           types
_osx_support        fnmatch             pprint              unicodedata
_pyio               formatter           profile             unit_testing
_random             fpformat            program             unittest
_sha                fractions           pstats              urllib
_sha256             ftplib              pty                 urllib2
_sha512             functools           py_compile          urlparse
_socket             future_builtins     pyclbr              user
_sqlite3            gc                  pydoc               uu
_sre                genericpath         pydoc_data          uuid
_ssl                getopt              pyexpat             warnings
_strptime           getpass             wave
_struct             gettext             quopri              weakref
_subprocess         glob                random              webbrowser
_symtable           gzip                re                  whichdb
_testcapi           hashlib             repr                winsound
_threading_local    heapq               rexec               wsgiref
_tkinter            hmac                rfc822              xdrlib
_warnings           hotshot             rlcompleter         xml
_weakref            htmlentitydefs      robotparser         xmllib
_weakrefset         htmllib             runpy               xmlrpclib
_winreg             httplib             sched               xxsubtype
abc                 idlelib             select              zipfile
aifc                ihooks              sets                zipimport
antigravity         imageop             setuptools          zlib
"""

python_library_modules = None


def is_python_library_module(module_name):
    """
    Returns if the passed module name is part of the listed Python library modules
    :param module_name: module name
    :return: bool
    """
    global python_library_modules

    if python_library_modules is None:
        temp = python_library_modules_str.replace("\n", " ")
        python_library_modules = temp.split(" ")
        python_library_modules = filter(lambda elem: not elem == "", python_library_modules)

    return module_name in python_library_modules
