
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.extension
2: 
3: Provides the Extension class, used to describe C/C++ extension
4: modules in setup scripts.
5: 
6: Overridden to support f2py.
7: 
8: '''
9: from __future__ import division, absolute_import, print_function
10: 
11: import sys
12: import re
13: from distutils.extension import Extension as old_Extension
14: 
15: if sys.version_info[0] >= 3:
16:     basestring = str
17: 
18: 
19: cxx_ext_re = re.compile(r'.*[.](cpp|cxx|cc)\Z', re.I).match
20: fortran_pyf_ext_re = re.compile(r'.*[.](f90|f95|f77|for|ftn|f|pyf)\Z', re.I).match
21: 
22: class Extension(old_Extension):
23:     def __init__ (self, name, sources,
24:                   include_dirs=None,
25:                   define_macros=None,
26:                   undef_macros=None,
27:                   library_dirs=None,
28:                   libraries=None,
29:                   runtime_library_dirs=None,
30:                   extra_objects=None,
31:                   extra_compile_args=None,
32:                   extra_link_args=None,
33:                   export_symbols=None,
34:                   swig_opts=None,
35:                   depends=None,
36:                   language=None,
37:                   f2py_options=None,
38:                   module_dirs=None,
39:                   extra_f77_compile_args=None,
40:                   extra_f90_compile_args=None,
41:                  ):
42:         old_Extension.__init__(self, name, [],
43:                                include_dirs,
44:                                define_macros,
45:                                undef_macros,
46:                                library_dirs,
47:                                libraries,
48:                                runtime_library_dirs,
49:                                extra_objects,
50:                                extra_compile_args,
51:                                extra_link_args,
52:                                export_symbols)
53:         # Avoid assert statements checking that sources contains strings:
54:         self.sources = sources
55: 
56:         # Python 2.4 distutils new features
57:         self.swig_opts = swig_opts or []
58:         # swig_opts is assumed to be a list. Here we handle the case where it
59:         # is specified as a string instead.
60:         if isinstance(self.swig_opts, basestring):
61:             import warnings
62:             msg = "swig_opts is specified as a string instead of a list"
63:             warnings.warn(msg, SyntaxWarning)
64:             self.swig_opts = self.swig_opts.split()
65: 
66:         # Python 2.3 distutils new features
67:         self.depends = depends or []
68:         self.language = language
69: 
70:         # numpy_distutils features
71:         self.f2py_options = f2py_options or []
72:         self.module_dirs = module_dirs or []
73:         self.extra_f77_compile_args = extra_f77_compile_args or []
74:         self.extra_f90_compile_args = extra_f90_compile_args or []
75: 
76:         return
77: 
78:     def has_cxx_sources(self):
79:         for source in self.sources:
80:             if cxx_ext_re(str(source)):
81:                 return True
82:         return False
83: 
84:     def has_f2py_sources(self):
85:         for source in self.sources:
86:             if fortran_pyf_ext_re(source):
87:                 return True
88:         return False
89: 
90: # class Extension
91: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_34959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, (-1)), 'str', 'distutils.extension\n\nProvides the Extension class, used to describe C/C++ extension\nmodules in setup scripts.\n\nOverridden to support f2py.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import sys' statement (line 11)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import re' statement (line 12)
import re

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from distutils.extension import old_Extension' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_34960 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.extension')

if (type(import_34960) is not StypyTypeError):

    if (import_34960 != 'pyd_module'):
        __import__(import_34960)
        sys_modules_34961 = sys.modules[import_34960]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.extension', sys_modules_34961.module_type_store, module_type_store, ['Extension'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_34961, sys_modules_34961.module_type_store, module_type_store)
    else:
        from distutils.extension import Extension as old_Extension

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.extension', None, module_type_store, ['Extension'], [old_Extension])

else:
    # Assigning a type to the variable 'distutils.extension' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.extension', import_34960)

# Adding an alias
module_type_store.add_alias('old_Extension', 'Extension')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')




# Obtaining the type of the subscript
int_34962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 20), 'int')
# Getting the type of 'sys' (line 15)
sys_34963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 15)
version_info_34964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 3), sys_34963, 'version_info')
# Obtaining the member '__getitem__' of a type (line 15)
getitem___34965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 3), version_info_34964, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 15)
subscript_call_result_34966 = invoke(stypy.reporting.localization.Localization(__file__, 15, 3), getitem___34965, int_34962)

int_34967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 26), 'int')
# Applying the binary operator '>=' (line 15)
result_ge_34968 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 3), '>=', subscript_call_result_34966, int_34967)

# Testing the type of an if condition (line 15)
if_condition_34969 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 15, 0), result_ge_34968)
# Assigning a type to the variable 'if_condition_34969' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'if_condition_34969', if_condition_34969)
# SSA begins for if statement (line 15)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 16):
# Getting the type of 'str' (line 16)
str_34970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 17), 'str')
# Assigning a type to the variable 'basestring' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'basestring', str_34970)
# SSA join for if statement (line 15)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Attribute to a Name (line 19):

# Call to compile(...): (line 19)
# Processing the call arguments (line 19)
str_34973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 24), 'str', '.*[.](cpp|cxx|cc)\\Z')
# Getting the type of 're' (line 19)
re_34974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 48), 're', False)
# Obtaining the member 'I' of a type (line 19)
I_34975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 48), re_34974, 'I')
# Processing the call keyword arguments (line 19)
kwargs_34976 = {}
# Getting the type of 're' (line 19)
re_34971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 13), 're', False)
# Obtaining the member 'compile' of a type (line 19)
compile_34972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 13), re_34971, 'compile')
# Calling compile(args, kwargs) (line 19)
compile_call_result_34977 = invoke(stypy.reporting.localization.Localization(__file__, 19, 13), compile_34972, *[str_34973, I_34975], **kwargs_34976)

# Obtaining the member 'match' of a type (line 19)
match_34978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 13), compile_call_result_34977, 'match')
# Assigning a type to the variable 'cxx_ext_re' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'cxx_ext_re', match_34978)

# Assigning a Attribute to a Name (line 20):

# Call to compile(...): (line 20)
# Processing the call arguments (line 20)
str_34981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 32), 'str', '.*[.](f90|f95|f77|for|ftn|f|pyf)\\Z')
# Getting the type of 're' (line 20)
re_34982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 71), 're', False)
# Obtaining the member 'I' of a type (line 20)
I_34983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 71), re_34982, 'I')
# Processing the call keyword arguments (line 20)
kwargs_34984 = {}
# Getting the type of 're' (line 20)
re_34979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 're', False)
# Obtaining the member 'compile' of a type (line 20)
compile_34980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 21), re_34979, 'compile')
# Calling compile(args, kwargs) (line 20)
compile_call_result_34985 = invoke(stypy.reporting.localization.Localization(__file__, 20, 21), compile_34980, *[str_34981, I_34983], **kwargs_34984)

# Obtaining the member 'match' of a type (line 20)
match_34986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 21), compile_call_result_34985, 'match')
# Assigning a type to the variable 'fortran_pyf_ext_re' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'fortran_pyf_ext_re', match_34986)
# Declaration of the 'Extension' class
# Getting the type of 'old_Extension' (line 22)
old_Extension_34987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'old_Extension')

class Extension(old_Extension_34987, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 24)
        None_34988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 31), 'None')
        # Getting the type of 'None' (line 25)
        None_34989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 32), 'None')
        # Getting the type of 'None' (line 26)
        None_34990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 31), 'None')
        # Getting the type of 'None' (line 27)
        None_34991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 31), 'None')
        # Getting the type of 'None' (line 28)
        None_34992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 28), 'None')
        # Getting the type of 'None' (line 29)
        None_34993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 39), 'None')
        # Getting the type of 'None' (line 30)
        None_34994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 32), 'None')
        # Getting the type of 'None' (line 31)
        None_34995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 37), 'None')
        # Getting the type of 'None' (line 32)
        None_34996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 34), 'None')
        # Getting the type of 'None' (line 33)
        None_34997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 33), 'None')
        # Getting the type of 'None' (line 34)
        None_34998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 28), 'None')
        # Getting the type of 'None' (line 35)
        None_34999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 26), 'None')
        # Getting the type of 'None' (line 36)
        None_35000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 27), 'None')
        # Getting the type of 'None' (line 37)
        None_35001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 31), 'None')
        # Getting the type of 'None' (line 38)
        None_35002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 30), 'None')
        # Getting the type of 'None' (line 39)
        None_35003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 41), 'None')
        # Getting the type of 'None' (line 40)
        None_35004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 41), 'None')
        defaults = [None_34988, None_34989, None_34990, None_34991, None_34992, None_34993, None_34994, None_34995, None_34996, None_34997, None_34998, None_34999, None_35000, None_35001, None_35002, None_35003, None_35004]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 23, 4, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Extension.__init__', ['name', 'sources', 'include_dirs', 'define_macros', 'undef_macros', 'library_dirs', 'libraries', 'runtime_library_dirs', 'extra_objects', 'extra_compile_args', 'extra_link_args', 'export_symbols', 'swig_opts', 'depends', 'language', 'f2py_options', 'module_dirs', 'extra_f77_compile_args', 'extra_f90_compile_args'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['name', 'sources', 'include_dirs', 'define_macros', 'undef_macros', 'library_dirs', 'libraries', 'runtime_library_dirs', 'extra_objects', 'extra_compile_args', 'extra_link_args', 'export_symbols', 'swig_opts', 'depends', 'language', 'f2py_options', 'module_dirs', 'extra_f77_compile_args', 'extra_f90_compile_args'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'self' (line 42)
        self_35007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 31), 'self', False)
        # Getting the type of 'name' (line 42)
        name_35008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 37), 'name', False)
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_35009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        
        # Getting the type of 'include_dirs' (line 43)
        include_dirs_35010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 31), 'include_dirs', False)
        # Getting the type of 'define_macros' (line 44)
        define_macros_35011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 31), 'define_macros', False)
        # Getting the type of 'undef_macros' (line 45)
        undef_macros_35012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 31), 'undef_macros', False)
        # Getting the type of 'library_dirs' (line 46)
        library_dirs_35013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 31), 'library_dirs', False)
        # Getting the type of 'libraries' (line 47)
        libraries_35014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 31), 'libraries', False)
        # Getting the type of 'runtime_library_dirs' (line 48)
        runtime_library_dirs_35015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 31), 'runtime_library_dirs', False)
        # Getting the type of 'extra_objects' (line 49)
        extra_objects_35016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 31), 'extra_objects', False)
        # Getting the type of 'extra_compile_args' (line 50)
        extra_compile_args_35017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 31), 'extra_compile_args', False)
        # Getting the type of 'extra_link_args' (line 51)
        extra_link_args_35018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 31), 'extra_link_args', False)
        # Getting the type of 'export_symbols' (line 52)
        export_symbols_35019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 31), 'export_symbols', False)
        # Processing the call keyword arguments (line 42)
        kwargs_35020 = {}
        # Getting the type of 'old_Extension' (line 42)
        old_Extension_35005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'old_Extension', False)
        # Obtaining the member '__init__' of a type (line 42)
        init___35006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), old_Extension_35005, '__init__')
        # Calling __init__(args, kwargs) (line 42)
        init___call_result_35021 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), init___35006, *[self_35007, name_35008, list_35009, include_dirs_35010, define_macros_35011, undef_macros_35012, library_dirs_35013, libraries_35014, runtime_library_dirs_35015, extra_objects_35016, extra_compile_args_35017, extra_link_args_35018, export_symbols_35019], **kwargs_35020)
        
        
        # Assigning a Name to a Attribute (line 54):
        # Getting the type of 'sources' (line 54)
        sources_35022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 23), 'sources')
        # Getting the type of 'self' (line 54)
        self_35023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self')
        # Setting the type of the member 'sources' of a type (line 54)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), self_35023, 'sources', sources_35022)
        
        # Assigning a BoolOp to a Attribute (line 57):
        
        # Evaluating a boolean operation
        # Getting the type of 'swig_opts' (line 57)
        swig_opts_35024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'swig_opts')
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_35025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        
        # Applying the binary operator 'or' (line 57)
        result_or_keyword_35026 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 25), 'or', swig_opts_35024, list_35025)
        
        # Getting the type of 'self' (line 57)
        self_35027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self')
        # Setting the type of the member 'swig_opts' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_35027, 'swig_opts', result_or_keyword_35026)
        
        # Type idiom detected: calculating its left and rigth part (line 60)
        # Getting the type of 'basestring' (line 60)
        basestring_35028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 38), 'basestring')
        # Getting the type of 'self' (line 60)
        self_35029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 22), 'self')
        # Obtaining the member 'swig_opts' of a type (line 60)
        swig_opts_35030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 22), self_35029, 'swig_opts')
        
        (may_be_35031, more_types_in_union_35032) = may_be_subtype(basestring_35028, swig_opts_35030)

        if may_be_35031:

            if more_types_in_union_35032:
                # Runtime conditional SSA (line 60)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 60)
            self_35033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self')
            # Obtaining the member 'swig_opts' of a type (line 60)
            swig_opts_35034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_35033, 'swig_opts')
            # Setting the type of the member 'swig_opts' of a type (line 60)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_35033, 'swig_opts', remove_not_subtype_from_union(swig_opts_35030, basestring))
            stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 61, 12))
            
            # 'import warnings' statement (line 61)
            import warnings

            import_module(stypy.reporting.localization.Localization(__file__, 61, 12), 'warnings', warnings, module_type_store)
            
            
            # Assigning a Str to a Name (line 62):
            str_35035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 18), 'str', 'swig_opts is specified as a string instead of a list')
            # Assigning a type to the variable 'msg' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'msg', str_35035)
            
            # Call to warn(...): (line 63)
            # Processing the call arguments (line 63)
            # Getting the type of 'msg' (line 63)
            msg_35038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 26), 'msg', False)
            # Getting the type of 'SyntaxWarning' (line 63)
            SyntaxWarning_35039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 31), 'SyntaxWarning', False)
            # Processing the call keyword arguments (line 63)
            kwargs_35040 = {}
            # Getting the type of 'warnings' (line 63)
            warnings_35036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'warnings', False)
            # Obtaining the member 'warn' of a type (line 63)
            warn_35037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), warnings_35036, 'warn')
            # Calling warn(args, kwargs) (line 63)
            warn_call_result_35041 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), warn_35037, *[msg_35038, SyntaxWarning_35039], **kwargs_35040)
            
            
            # Assigning a Call to a Attribute (line 64):
            
            # Call to split(...): (line 64)
            # Processing the call keyword arguments (line 64)
            kwargs_35045 = {}
            # Getting the type of 'self' (line 64)
            self_35042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 29), 'self', False)
            # Obtaining the member 'swig_opts' of a type (line 64)
            swig_opts_35043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 29), self_35042, 'swig_opts')
            # Obtaining the member 'split' of a type (line 64)
            split_35044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 29), swig_opts_35043, 'split')
            # Calling split(args, kwargs) (line 64)
            split_call_result_35046 = invoke(stypy.reporting.localization.Localization(__file__, 64, 29), split_35044, *[], **kwargs_35045)
            
            # Getting the type of 'self' (line 64)
            self_35047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'self')
            # Setting the type of the member 'swig_opts' of a type (line 64)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), self_35047, 'swig_opts', split_call_result_35046)

            if more_types_in_union_35032:
                # SSA join for if statement (line 60)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BoolOp to a Attribute (line 67):
        
        # Evaluating a boolean operation
        # Getting the type of 'depends' (line 67)
        depends_35048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 23), 'depends')
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_35049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        
        # Applying the binary operator 'or' (line 67)
        result_or_keyword_35050 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 23), 'or', depends_35048, list_35049)
        
        # Getting the type of 'self' (line 67)
        self_35051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self')
        # Setting the type of the member 'depends' of a type (line 67)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_35051, 'depends', result_or_keyword_35050)
        
        # Assigning a Name to a Attribute (line 68):
        # Getting the type of 'language' (line 68)
        language_35052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 24), 'language')
        # Getting the type of 'self' (line 68)
        self_35053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self')
        # Setting the type of the member 'language' of a type (line 68)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_35053, 'language', language_35052)
        
        # Assigning a BoolOp to a Attribute (line 71):
        
        # Evaluating a boolean operation
        # Getting the type of 'f2py_options' (line 71)
        f2py_options_35054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 28), 'f2py_options')
        
        # Obtaining an instance of the builtin type 'list' (line 71)
        list_35055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 71)
        
        # Applying the binary operator 'or' (line 71)
        result_or_keyword_35056 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 28), 'or', f2py_options_35054, list_35055)
        
        # Getting the type of 'self' (line 71)
        self_35057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'self')
        # Setting the type of the member 'f2py_options' of a type (line 71)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), self_35057, 'f2py_options', result_or_keyword_35056)
        
        # Assigning a BoolOp to a Attribute (line 72):
        
        # Evaluating a boolean operation
        # Getting the type of 'module_dirs' (line 72)
        module_dirs_35058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 27), 'module_dirs')
        
        # Obtaining an instance of the builtin type 'list' (line 72)
        list_35059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 72)
        
        # Applying the binary operator 'or' (line 72)
        result_or_keyword_35060 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 27), 'or', module_dirs_35058, list_35059)
        
        # Getting the type of 'self' (line 72)
        self_35061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self')
        # Setting the type of the member 'module_dirs' of a type (line 72)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_35061, 'module_dirs', result_or_keyword_35060)
        
        # Assigning a BoolOp to a Attribute (line 73):
        
        # Evaluating a boolean operation
        # Getting the type of 'extra_f77_compile_args' (line 73)
        extra_f77_compile_args_35062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 38), 'extra_f77_compile_args')
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_35063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 64), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        
        # Applying the binary operator 'or' (line 73)
        result_or_keyword_35064 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 38), 'or', extra_f77_compile_args_35062, list_35063)
        
        # Getting the type of 'self' (line 73)
        self_35065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self')
        # Setting the type of the member 'extra_f77_compile_args' of a type (line 73)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_35065, 'extra_f77_compile_args', result_or_keyword_35064)
        
        # Assigning a BoolOp to a Attribute (line 74):
        
        # Evaluating a boolean operation
        # Getting the type of 'extra_f90_compile_args' (line 74)
        extra_f90_compile_args_35066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 38), 'extra_f90_compile_args')
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_35067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 64), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        
        # Applying the binary operator 'or' (line 74)
        result_or_keyword_35068 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 38), 'or', extra_f90_compile_args_35066, list_35067)
        
        # Getting the type of 'self' (line 74)
        self_35069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self')
        # Setting the type of the member 'extra_f90_compile_args' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_35069, 'extra_f90_compile_args', result_or_keyword_35068)
        # Assigning a type to the variable 'stypy_return_type' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def has_cxx_sources(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_cxx_sources'
        module_type_store = module_type_store.open_function_context('has_cxx_sources', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Extension.has_cxx_sources.__dict__.__setitem__('stypy_localization', localization)
        Extension.has_cxx_sources.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Extension.has_cxx_sources.__dict__.__setitem__('stypy_type_store', module_type_store)
        Extension.has_cxx_sources.__dict__.__setitem__('stypy_function_name', 'Extension.has_cxx_sources')
        Extension.has_cxx_sources.__dict__.__setitem__('stypy_param_names_list', [])
        Extension.has_cxx_sources.__dict__.__setitem__('stypy_varargs_param_name', None)
        Extension.has_cxx_sources.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Extension.has_cxx_sources.__dict__.__setitem__('stypy_call_defaults', defaults)
        Extension.has_cxx_sources.__dict__.__setitem__('stypy_call_varargs', varargs)
        Extension.has_cxx_sources.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Extension.has_cxx_sources.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Extension.has_cxx_sources', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_cxx_sources', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_cxx_sources(...)' code ##################

        
        # Getting the type of 'self' (line 79)
        self_35070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 22), 'self')
        # Obtaining the member 'sources' of a type (line 79)
        sources_35071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 22), self_35070, 'sources')
        # Testing the type of a for loop iterable (line 79)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 79, 8), sources_35071)
        # Getting the type of the for loop variable (line 79)
        for_loop_var_35072 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 79, 8), sources_35071)
        # Assigning a type to the variable 'source' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'source', for_loop_var_35072)
        # SSA begins for a for statement (line 79)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to cxx_ext_re(...): (line 80)
        # Processing the call arguments (line 80)
        
        # Call to str(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'source' (line 80)
        source_35075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 30), 'source', False)
        # Processing the call keyword arguments (line 80)
        kwargs_35076 = {}
        # Getting the type of 'str' (line 80)
        str_35074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 26), 'str', False)
        # Calling str(args, kwargs) (line 80)
        str_call_result_35077 = invoke(stypy.reporting.localization.Localization(__file__, 80, 26), str_35074, *[source_35075], **kwargs_35076)
        
        # Processing the call keyword arguments (line 80)
        kwargs_35078 = {}
        # Getting the type of 'cxx_ext_re' (line 80)
        cxx_ext_re_35073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'cxx_ext_re', False)
        # Calling cxx_ext_re(args, kwargs) (line 80)
        cxx_ext_re_call_result_35079 = invoke(stypy.reporting.localization.Localization(__file__, 80, 15), cxx_ext_re_35073, *[str_call_result_35077], **kwargs_35078)
        
        # Testing the type of an if condition (line 80)
        if_condition_35080 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 12), cxx_ext_re_call_result_35079)
        # Assigning a type to the variable 'if_condition_35080' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'if_condition_35080', if_condition_35080)
        # SSA begins for if statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 81)
        True_35081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 23), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'stypy_return_type', True_35081)
        # SSA join for if statement (line 80)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'False' (line 82)
        False_35082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stypy_return_type', False_35082)
        
        # ################# End of 'has_cxx_sources(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_cxx_sources' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_35083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35083)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_cxx_sources'
        return stypy_return_type_35083


    @norecursion
    def has_f2py_sources(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_f2py_sources'
        module_type_store = module_type_store.open_function_context('has_f2py_sources', 84, 4, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Extension.has_f2py_sources.__dict__.__setitem__('stypy_localization', localization)
        Extension.has_f2py_sources.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Extension.has_f2py_sources.__dict__.__setitem__('stypy_type_store', module_type_store)
        Extension.has_f2py_sources.__dict__.__setitem__('stypy_function_name', 'Extension.has_f2py_sources')
        Extension.has_f2py_sources.__dict__.__setitem__('stypy_param_names_list', [])
        Extension.has_f2py_sources.__dict__.__setitem__('stypy_varargs_param_name', None)
        Extension.has_f2py_sources.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Extension.has_f2py_sources.__dict__.__setitem__('stypy_call_defaults', defaults)
        Extension.has_f2py_sources.__dict__.__setitem__('stypy_call_varargs', varargs)
        Extension.has_f2py_sources.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Extension.has_f2py_sources.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Extension.has_f2py_sources', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_f2py_sources', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_f2py_sources(...)' code ##################

        
        # Getting the type of 'self' (line 85)
        self_35084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 22), 'self')
        # Obtaining the member 'sources' of a type (line 85)
        sources_35085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 22), self_35084, 'sources')
        # Testing the type of a for loop iterable (line 85)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 85, 8), sources_35085)
        # Getting the type of the for loop variable (line 85)
        for_loop_var_35086 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 85, 8), sources_35085)
        # Assigning a type to the variable 'source' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'source', for_loop_var_35086)
        # SSA begins for a for statement (line 85)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to fortran_pyf_ext_re(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'source' (line 86)
        source_35088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 34), 'source', False)
        # Processing the call keyword arguments (line 86)
        kwargs_35089 = {}
        # Getting the type of 'fortran_pyf_ext_re' (line 86)
        fortran_pyf_ext_re_35087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'fortran_pyf_ext_re', False)
        # Calling fortran_pyf_ext_re(args, kwargs) (line 86)
        fortran_pyf_ext_re_call_result_35090 = invoke(stypy.reporting.localization.Localization(__file__, 86, 15), fortran_pyf_ext_re_35087, *[source_35088], **kwargs_35089)
        
        # Testing the type of an if condition (line 86)
        if_condition_35091 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 12), fortran_pyf_ext_re_call_result_35090)
        # Assigning a type to the variable 'if_condition_35091' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'if_condition_35091', if_condition_35091)
        # SSA begins for if statement (line 86)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 87)
        True_35092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'stypy_return_type', True_35092)
        # SSA join for if statement (line 86)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'False' (line 88)
        False_35093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'stypy_return_type', False_35093)
        
        # ################# End of 'has_f2py_sources(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_f2py_sources' in the type store
        # Getting the type of 'stypy_return_type' (line 84)
        stypy_return_type_35094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35094)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_f2py_sources'
        return stypy_return_type_35094


# Assigning a type to the variable 'Extension' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'Extension', Extension)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
