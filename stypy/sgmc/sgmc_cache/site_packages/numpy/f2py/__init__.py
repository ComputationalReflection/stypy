
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: '''Fortran to Python Interface Generator.
3: 
4: '''
5: from __future__ import division, absolute_import, print_function
6: 
7: __all__ = ['run_main', 'compile', 'f2py_testing']
8: 
9: import sys
10: 
11: from . import f2py2e
12: from . import f2py_testing
13: from . import diagnose
14: 
15: run_main = f2py2e.run_main
16: main = f2py2e.main
17: 
18: 
19: def compile(source,
20:             modulename='untitled',
21:             extra_args='',
22:             verbose=True,
23:             source_fn=None,
24:             extension='.f'
25:             ):
26:     ''' Build extension module from processing source with f2py.
27: 
28:     Parameters
29:     ----------
30:     source : str
31:         Fortran source of module / subroutine to compile
32:     modulename : str, optional
33:         the name of compiled python module
34:     extra_args: str, optional
35:         additional parameters passed to f2py
36:     verbose: bool, optional
37:         print f2py output to screen
38:     extension: {'.f', '.f90'}, optional
39:         filename extension influences the fortran compiler behavior
40: 
41:         .. versionadded:: 1.11.0
42: 
43:     '''
44:     from numpy.distutils.exec_command import exec_command
45:     import tempfile
46:     if source_fn is None:
47:         f = tempfile.NamedTemporaryFile(suffix=extension)
48:     else:
49:         f = open(source_fn, 'w')
50: 
51:     try:
52:         f.write(source)
53:         f.flush()
54: 
55:         args = ' -c -m {} {} {}'.format(modulename, f.name, extra_args)
56:         c = '{} -c "import numpy.f2py as f2py2e;f2py2e.main()" {}'
57:         c = c.format(sys.executable, args)
58:         status, output = exec_command(c)
59:         if verbose:
60:             print(output)
61:     finally:
62:         f.close()
63:     return status
64: 
65: from numpy.testing.nosetester import _numpy_tester
66: test = _numpy_tester().test
67: bench = _numpy_tester().bench
68: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_99799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', 'Fortran to Python Interface Generator.\n\n')

# Assigning a List to a Name (line 7):

# Assigning a List to a Name (line 7):
__all__ = ['run_main', 'compile', 'f2py_testing']
module_type_store.set_exportable_members(['run_main', 'compile', 'f2py_testing'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_99800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_99801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'run_main')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_99800, str_99801)
# Adding element type (line 7)
str_99802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 23), 'str', 'compile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_99800, str_99802)
# Adding element type (line 7)
str_99803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 34), 'str', 'f2py_testing')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_99800, str_99803)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_99800)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import sys' statement (line 9)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from numpy.f2py import f2py2e' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_99804 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.f2py')

if (type(import_99804) is not StypyTypeError):

    if (import_99804 != 'pyd_module'):
        __import__(import_99804)
        sys_modules_99805 = sys.modules[import_99804]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.f2py', sys_modules_99805.module_type_store, module_type_store, ['f2py2e'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_99805, sys_modules_99805.module_type_store, module_type_store)
    else:
        from numpy.f2py import f2py2e

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.f2py', None, module_type_store, ['f2py2e'], [f2py2e])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.f2py', import_99804)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy.f2py import f2py_testing' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_99806 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.f2py')

if (type(import_99806) is not StypyTypeError):

    if (import_99806 != 'pyd_module'):
        __import__(import_99806)
        sys_modules_99807 = sys.modules[import_99806]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.f2py', sys_modules_99807.module_type_store, module_type_store, ['f2py_testing'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_99807, sys_modules_99807.module_type_store, module_type_store)
    else:
        from numpy.f2py import f2py_testing

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.f2py', None, module_type_store, ['f2py_testing'], [f2py_testing])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.f2py', import_99806)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy.f2py import diagnose' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_99808 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.f2py')

if (type(import_99808) is not StypyTypeError):

    if (import_99808 != 'pyd_module'):
        __import__(import_99808)
        sys_modules_99809 = sys.modules[import_99808]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.f2py', sys_modules_99809.module_type_store, module_type_store, ['diagnose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_99809, sys_modules_99809.module_type_store, module_type_store)
    else:
        from numpy.f2py import diagnose

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.f2py', None, module_type_store, ['diagnose'], [diagnose])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.f2py', import_99808)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')


# Assigning a Attribute to a Name (line 15):

# Assigning a Attribute to a Name (line 15):
# Getting the type of 'f2py2e' (line 15)
f2py2e_99810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 11), 'f2py2e')
# Obtaining the member 'run_main' of a type (line 15)
run_main_99811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 11), f2py2e_99810, 'run_main')
# Assigning a type to the variable 'run_main' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'run_main', run_main_99811)

# Assigning a Attribute to a Name (line 16):

# Assigning a Attribute to a Name (line 16):
# Getting the type of 'f2py2e' (line 16)
f2py2e_99812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 7), 'f2py2e')
# Obtaining the member 'main' of a type (line 16)
main_99813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 7), f2py2e_99812, 'main')
# Assigning a type to the variable 'main' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'main', main_99813)

@norecursion
def compile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_99814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 23), 'str', 'untitled')
    str_99815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 23), 'str', '')
    # Getting the type of 'True' (line 22)
    True_99816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 20), 'True')
    # Getting the type of 'None' (line 23)
    None_99817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 22), 'None')
    str_99818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 22), 'str', '.f')
    defaults = [str_99814, str_99815, True_99816, None_99817, str_99818]
    # Create a new context for function 'compile'
    module_type_store = module_type_store.open_function_context('compile', 19, 0, False)
    
    # Passed parameters checking function
    compile.stypy_localization = localization
    compile.stypy_type_of_self = None
    compile.stypy_type_store = module_type_store
    compile.stypy_function_name = 'compile'
    compile.stypy_param_names_list = ['source', 'modulename', 'extra_args', 'verbose', 'source_fn', 'extension']
    compile.stypy_varargs_param_name = None
    compile.stypy_kwargs_param_name = None
    compile.stypy_call_defaults = defaults
    compile.stypy_call_varargs = varargs
    compile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'compile', ['source', 'modulename', 'extra_args', 'verbose', 'source_fn', 'extension'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'compile', localization, ['source', 'modulename', 'extra_args', 'verbose', 'source_fn', 'extension'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'compile(...)' code ##################

    str_99819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'str', " Build extension module from processing source with f2py.\n\n    Parameters\n    ----------\n    source : str\n        Fortran source of module / subroutine to compile\n    modulename : str, optional\n        the name of compiled python module\n    extra_args: str, optional\n        additional parameters passed to f2py\n    verbose: bool, optional\n        print f2py output to screen\n    extension: {'.f', '.f90'}, optional\n        filename extension influences the fortran compiler behavior\n\n        .. versionadded:: 1.11.0\n\n    ")
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 44, 4))
    
    # 'from numpy.distutils.exec_command import exec_command' statement (line 44)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_99820 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 44, 4), 'numpy.distutils.exec_command')

    if (type(import_99820) is not StypyTypeError):

        if (import_99820 != 'pyd_module'):
            __import__(import_99820)
            sys_modules_99821 = sys.modules[import_99820]
            import_from_module(stypy.reporting.localization.Localization(__file__, 44, 4), 'numpy.distutils.exec_command', sys_modules_99821.module_type_store, module_type_store, ['exec_command'])
            nest_module(stypy.reporting.localization.Localization(__file__, 44, 4), __file__, sys_modules_99821, sys_modules_99821.module_type_store, module_type_store)
        else:
            from numpy.distutils.exec_command import exec_command

            import_from_module(stypy.reporting.localization.Localization(__file__, 44, 4), 'numpy.distutils.exec_command', None, module_type_store, ['exec_command'], [exec_command])

    else:
        # Assigning a type to the variable 'numpy.distutils.exec_command' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'numpy.distutils.exec_command', import_99820)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 45, 4))
    
    # 'import tempfile' statement (line 45)
    import tempfile

    import_module(stypy.reporting.localization.Localization(__file__, 45, 4), 'tempfile', tempfile, module_type_store)
    
    
    # Type idiom detected: calculating its left and rigth part (line 46)
    # Getting the type of 'source_fn' (line 46)
    source_fn_99822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 7), 'source_fn')
    # Getting the type of 'None' (line 46)
    None_99823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'None')
    
    (may_be_99824, more_types_in_union_99825) = may_be_none(source_fn_99822, None_99823)

    if may_be_99824:

        if more_types_in_union_99825:
            # Runtime conditional SSA (line 46)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 47):
        
        # Assigning a Call to a Name (line 47):
        
        # Call to NamedTemporaryFile(...): (line 47)
        # Processing the call keyword arguments (line 47)
        # Getting the type of 'extension' (line 47)
        extension_99828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 47), 'extension', False)
        keyword_99829 = extension_99828
        kwargs_99830 = {'suffix': keyword_99829}
        # Getting the type of 'tempfile' (line 47)
        tempfile_99826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'tempfile', False)
        # Obtaining the member 'NamedTemporaryFile' of a type (line 47)
        NamedTemporaryFile_99827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), tempfile_99826, 'NamedTemporaryFile')
        # Calling NamedTemporaryFile(args, kwargs) (line 47)
        NamedTemporaryFile_call_result_99831 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), NamedTemporaryFile_99827, *[], **kwargs_99830)
        
        # Assigning a type to the variable 'f' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'f', NamedTemporaryFile_call_result_99831)

        if more_types_in_union_99825:
            # Runtime conditional SSA for else branch (line 46)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_99824) or more_types_in_union_99825):
        
        # Assigning a Call to a Name (line 49):
        
        # Assigning a Call to a Name (line 49):
        
        # Call to open(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'source_fn' (line 49)
        source_fn_99833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 17), 'source_fn', False)
        str_99834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 28), 'str', 'w')
        # Processing the call keyword arguments (line 49)
        kwargs_99835 = {}
        # Getting the type of 'open' (line 49)
        open_99832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'open', False)
        # Calling open(args, kwargs) (line 49)
        open_call_result_99836 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), open_99832, *[source_fn_99833, str_99834], **kwargs_99835)
        
        # Assigning a type to the variable 'f' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'f', open_call_result_99836)

        if (may_be_99824 and more_types_in_union_99825):
            # SSA join for if statement (line 46)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Try-finally block (line 51)
    
    # Call to write(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'source' (line 52)
    source_99839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'source', False)
    # Processing the call keyword arguments (line 52)
    kwargs_99840 = {}
    # Getting the type of 'f' (line 52)
    f_99837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'f', False)
    # Obtaining the member 'write' of a type (line 52)
    write_99838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), f_99837, 'write')
    # Calling write(args, kwargs) (line 52)
    write_call_result_99841 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), write_99838, *[source_99839], **kwargs_99840)
    
    
    # Call to flush(...): (line 53)
    # Processing the call keyword arguments (line 53)
    kwargs_99844 = {}
    # Getting the type of 'f' (line 53)
    f_99842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'f', False)
    # Obtaining the member 'flush' of a type (line 53)
    flush_99843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), f_99842, 'flush')
    # Calling flush(args, kwargs) (line 53)
    flush_call_result_99845 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), flush_99843, *[], **kwargs_99844)
    
    
    # Assigning a Call to a Name (line 55):
    
    # Assigning a Call to a Name (line 55):
    
    # Call to format(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'modulename' (line 55)
    modulename_99848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 40), 'modulename', False)
    # Getting the type of 'f' (line 55)
    f_99849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 52), 'f', False)
    # Obtaining the member 'name' of a type (line 55)
    name_99850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 52), f_99849, 'name')
    # Getting the type of 'extra_args' (line 55)
    extra_args_99851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 60), 'extra_args', False)
    # Processing the call keyword arguments (line 55)
    kwargs_99852 = {}
    str_99846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 15), 'str', ' -c -m {} {} {}')
    # Obtaining the member 'format' of a type (line 55)
    format_99847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 15), str_99846, 'format')
    # Calling format(args, kwargs) (line 55)
    format_call_result_99853 = invoke(stypy.reporting.localization.Localization(__file__, 55, 15), format_99847, *[modulename_99848, name_99850, extra_args_99851], **kwargs_99852)
    
    # Assigning a type to the variable 'args' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'args', format_call_result_99853)
    
    # Assigning a Str to a Name (line 56):
    
    # Assigning a Str to a Name (line 56):
    str_99854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 12), 'str', '{} -c "import numpy.f2py as f2py2e;f2py2e.main()" {}')
    # Assigning a type to the variable 'c' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'c', str_99854)
    
    # Assigning a Call to a Name (line 57):
    
    # Assigning a Call to a Name (line 57):
    
    # Call to format(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'sys' (line 57)
    sys_99857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 21), 'sys', False)
    # Obtaining the member 'executable' of a type (line 57)
    executable_99858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 21), sys_99857, 'executable')
    # Getting the type of 'args' (line 57)
    args_99859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 37), 'args', False)
    # Processing the call keyword arguments (line 57)
    kwargs_99860 = {}
    # Getting the type of 'c' (line 57)
    c_99855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'c', False)
    # Obtaining the member 'format' of a type (line 57)
    format_99856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), c_99855, 'format')
    # Calling format(args, kwargs) (line 57)
    format_call_result_99861 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), format_99856, *[executable_99858, args_99859], **kwargs_99860)
    
    # Assigning a type to the variable 'c' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'c', format_call_result_99861)
    
    # Assigning a Call to a Tuple (line 58):
    
    # Assigning a Call to a Name:
    
    # Call to exec_command(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'c' (line 58)
    c_99863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 38), 'c', False)
    # Processing the call keyword arguments (line 58)
    kwargs_99864 = {}
    # Getting the type of 'exec_command' (line 58)
    exec_command_99862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 25), 'exec_command', False)
    # Calling exec_command(args, kwargs) (line 58)
    exec_command_call_result_99865 = invoke(stypy.reporting.localization.Localization(__file__, 58, 25), exec_command_99862, *[c_99863], **kwargs_99864)
    
    # Assigning a type to the variable 'call_assignment_99796' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'call_assignment_99796', exec_command_call_result_99865)
    
    # Assigning a Call to a Name (line 58):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_99868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 8), 'int')
    # Processing the call keyword arguments
    kwargs_99869 = {}
    # Getting the type of 'call_assignment_99796' (line 58)
    call_assignment_99796_99866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'call_assignment_99796', False)
    # Obtaining the member '__getitem__' of a type (line 58)
    getitem___99867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), call_assignment_99796_99866, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_99870 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___99867, *[int_99868], **kwargs_99869)
    
    # Assigning a type to the variable 'call_assignment_99797' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'call_assignment_99797', getitem___call_result_99870)
    
    # Assigning a Name to a Name (line 58):
    # Getting the type of 'call_assignment_99797' (line 58)
    call_assignment_99797_99871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'call_assignment_99797')
    # Assigning a type to the variable 'status' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'status', call_assignment_99797_99871)
    
    # Assigning a Call to a Name (line 58):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_99874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 8), 'int')
    # Processing the call keyword arguments
    kwargs_99875 = {}
    # Getting the type of 'call_assignment_99796' (line 58)
    call_assignment_99796_99872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'call_assignment_99796', False)
    # Obtaining the member '__getitem__' of a type (line 58)
    getitem___99873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), call_assignment_99796_99872, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_99876 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___99873, *[int_99874], **kwargs_99875)
    
    # Assigning a type to the variable 'call_assignment_99798' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'call_assignment_99798', getitem___call_result_99876)
    
    # Assigning a Name to a Name (line 58):
    # Getting the type of 'call_assignment_99798' (line 58)
    call_assignment_99798_99877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'call_assignment_99798')
    # Assigning a type to the variable 'output' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'output', call_assignment_99798_99877)
    
    # Getting the type of 'verbose' (line 59)
    verbose_99878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'verbose')
    # Testing the type of an if condition (line 59)
    if_condition_99879 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 8), verbose_99878)
    # Assigning a type to the variable 'if_condition_99879' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'if_condition_99879', if_condition_99879)
    # SSA begins for if statement (line 59)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'output' (line 60)
    output_99881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 18), 'output', False)
    # Processing the call keyword arguments (line 60)
    kwargs_99882 = {}
    # Getting the type of 'print' (line 60)
    print_99880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'print', False)
    # Calling print(args, kwargs) (line 60)
    print_call_result_99883 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), print_99880, *[output_99881], **kwargs_99882)
    
    # SSA join for if statement (line 59)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 51)
    
    # Call to close(...): (line 62)
    # Processing the call keyword arguments (line 62)
    kwargs_99886 = {}
    # Getting the type of 'f' (line 62)
    f_99884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'f', False)
    # Obtaining the member 'close' of a type (line 62)
    close_99885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), f_99884, 'close')
    # Calling close(args, kwargs) (line 62)
    close_call_result_99887 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), close_99885, *[], **kwargs_99886)
    
    
    # Getting the type of 'status' (line 63)
    status_99888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 11), 'status')
    # Assigning a type to the variable 'stypy_return_type' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type', status_99888)
    
    # ################# End of 'compile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'compile' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_99889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_99889)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'compile'
    return stypy_return_type_99889

# Assigning a type to the variable 'compile' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'compile', compile)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 65, 0))

# 'from numpy.testing.nosetester import _numpy_tester' statement (line 65)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_99890 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 65, 0), 'numpy.testing.nosetester')

if (type(import_99890) is not StypyTypeError):

    if (import_99890 != 'pyd_module'):
        __import__(import_99890)
        sys_modules_99891 = sys.modules[import_99890]
        import_from_module(stypy.reporting.localization.Localization(__file__, 65, 0), 'numpy.testing.nosetester', sys_modules_99891.module_type_store, module_type_store, ['_numpy_tester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 65, 0), __file__, sys_modules_99891, sys_modules_99891.module_type_store, module_type_store)
    else:
        from numpy.testing.nosetester import _numpy_tester

        import_from_module(stypy.reporting.localization.Localization(__file__, 65, 0), 'numpy.testing.nosetester', None, module_type_store, ['_numpy_tester'], [_numpy_tester])

else:
    # Assigning a type to the variable 'numpy.testing.nosetester' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'numpy.testing.nosetester', import_99890)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')


# Assigning a Attribute to a Name (line 66):

# Assigning a Attribute to a Name (line 66):

# Call to _numpy_tester(...): (line 66)
# Processing the call keyword arguments (line 66)
kwargs_99893 = {}
# Getting the type of '_numpy_tester' (line 66)
_numpy_tester_99892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 7), '_numpy_tester', False)
# Calling _numpy_tester(args, kwargs) (line 66)
_numpy_tester_call_result_99894 = invoke(stypy.reporting.localization.Localization(__file__, 66, 7), _numpy_tester_99892, *[], **kwargs_99893)

# Obtaining the member 'test' of a type (line 66)
test_99895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 7), _numpy_tester_call_result_99894, 'test')
# Assigning a type to the variable 'test' (line 66)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'test', test_99895)

# Assigning a Attribute to a Name (line 67):

# Assigning a Attribute to a Name (line 67):

# Call to _numpy_tester(...): (line 67)
# Processing the call keyword arguments (line 67)
kwargs_99897 = {}
# Getting the type of '_numpy_tester' (line 67)
_numpy_tester_99896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), '_numpy_tester', False)
# Calling _numpy_tester(args, kwargs) (line 67)
_numpy_tester_call_result_99898 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), _numpy_tester_99896, *[], **kwargs_99897)

# Obtaining the member 'bench' of a type (line 67)
bench_99899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), _numpy_tester_call_result_99898, 'bench')
# Assigning a type to the variable 'bench' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'bench', bench_99899)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
