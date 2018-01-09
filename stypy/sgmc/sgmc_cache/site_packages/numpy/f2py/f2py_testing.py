
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import sys
4: import re
5: 
6: from numpy.testing.utils import jiffies, memusage
7: 
8: 
9: def cmdline():
10:     m = re.compile(r'\A\d+\Z')
11:     args = []
12:     repeat = 1
13:     for a in sys.argv[1:]:
14:         if m.match(a):
15:             repeat = eval(a)
16:         else:
17:             args.append(a)
18:     f2py_opts = ' '.join(args)
19:     return repeat, f2py_opts
20: 
21: 
22: def run(runtest, test_functions, repeat=1):
23:     l = [(t, repr(t.__doc__.split('\n')[1].strip())) for t in test_functions]
24:     start_memusage = memusage()
25:     diff_memusage = None
26:     start_jiffies = jiffies()
27:     i = 0
28:     while i < repeat:
29:         i += 1
30:         for t, fname in l:
31:             runtest(t)
32:             if start_memusage is None:
33:                 continue
34:             if diff_memusage is None:
35:                 diff_memusage = memusage() - start_memusage
36:             else:
37:                 diff_memusage2 = memusage() - start_memusage
38:                 if diff_memusage2 != diff_memusage:
39:                     print('memory usage change at step %i:' % i,
40:                           diff_memusage2 - diff_memusage,
41:                           fname)
42:                     diff_memusage = diff_memusage2
43:     current_memusage = memusage()
44:     print('run', repeat * len(test_functions), 'tests',
45:           'in %.2f seconds' % ((jiffies() - start_jiffies) / 100.0))
46:     if start_memusage:
47:         print('initial virtual memory size:', start_memusage, 'bytes')
48:         print('current virtual memory size:', current_memusage, 'bytes')
49: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import re' statement (line 4)
import re

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.testing.utils import jiffies, memusage' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_93008 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing.utils')

if (type(import_93008) is not StypyTypeError):

    if (import_93008 != 'pyd_module'):
        __import__(import_93008)
        sys_modules_93009 = sys.modules[import_93008]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing.utils', sys_modules_93009.module_type_store, module_type_store, ['jiffies', 'memusage'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_93009, sys_modules_93009.module_type_store, module_type_store)
    else:
        from numpy.testing.utils import jiffies, memusage

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing.utils', None, module_type_store, ['jiffies', 'memusage'], [jiffies, memusage])

else:
    # Assigning a type to the variable 'numpy.testing.utils' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing.utils', import_93008)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')


@norecursion
def cmdline(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'cmdline'
    module_type_store = module_type_store.open_function_context('cmdline', 9, 0, False)
    
    # Passed parameters checking function
    cmdline.stypy_localization = localization
    cmdline.stypy_type_of_self = None
    cmdline.stypy_type_store = module_type_store
    cmdline.stypy_function_name = 'cmdline'
    cmdline.stypy_param_names_list = []
    cmdline.stypy_varargs_param_name = None
    cmdline.stypy_kwargs_param_name = None
    cmdline.stypy_call_defaults = defaults
    cmdline.stypy_call_varargs = varargs
    cmdline.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cmdline', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cmdline', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cmdline(...)' code ##################

    
    # Assigning a Call to a Name (line 10):
    
    # Call to compile(...): (line 10)
    # Processing the call arguments (line 10)
    str_93012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 19), 'str', '\\A\\d+\\Z')
    # Processing the call keyword arguments (line 10)
    kwargs_93013 = {}
    # Getting the type of 're' (line 10)
    re_93010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 're', False)
    # Obtaining the member 'compile' of a type (line 10)
    compile_93011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 8), re_93010, 'compile')
    # Calling compile(args, kwargs) (line 10)
    compile_call_result_93014 = invoke(stypy.reporting.localization.Localization(__file__, 10, 8), compile_93011, *[str_93012], **kwargs_93013)
    
    # Assigning a type to the variable 'm' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'm', compile_call_result_93014)
    
    # Assigning a List to a Name (line 11):
    
    # Obtaining an instance of the builtin type 'list' (line 11)
    list_93015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 11)
    
    # Assigning a type to the variable 'args' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'args', list_93015)
    
    # Assigning a Num to a Name (line 12):
    int_93016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 13), 'int')
    # Assigning a type to the variable 'repeat' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'repeat', int_93016)
    
    
    # Obtaining the type of the subscript
    int_93017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 22), 'int')
    slice_93018 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 13, 13), int_93017, None, None)
    # Getting the type of 'sys' (line 13)
    sys_93019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 13), 'sys')
    # Obtaining the member 'argv' of a type (line 13)
    argv_93020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 13), sys_93019, 'argv')
    # Obtaining the member '__getitem__' of a type (line 13)
    getitem___93021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 13), argv_93020, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 13)
    subscript_call_result_93022 = invoke(stypy.reporting.localization.Localization(__file__, 13, 13), getitem___93021, slice_93018)
    
    # Testing the type of a for loop iterable (line 13)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 13, 4), subscript_call_result_93022)
    # Getting the type of the for loop variable (line 13)
    for_loop_var_93023 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 13, 4), subscript_call_result_93022)
    # Assigning a type to the variable 'a' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'a', for_loop_var_93023)
    # SSA begins for a for statement (line 13)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to match(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'a' (line 14)
    a_93026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'a', False)
    # Processing the call keyword arguments (line 14)
    kwargs_93027 = {}
    # Getting the type of 'm' (line 14)
    m_93024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'm', False)
    # Obtaining the member 'match' of a type (line 14)
    match_93025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 11), m_93024, 'match')
    # Calling match(args, kwargs) (line 14)
    match_call_result_93028 = invoke(stypy.reporting.localization.Localization(__file__, 14, 11), match_93025, *[a_93026], **kwargs_93027)
    
    # Testing the type of an if condition (line 14)
    if_condition_93029 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 8), match_call_result_93028)
    # Assigning a type to the variable 'if_condition_93029' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'if_condition_93029', if_condition_93029)
    # SSA begins for if statement (line 14)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 15):
    
    # Call to eval(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'a' (line 15)
    a_93031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 26), 'a', False)
    # Processing the call keyword arguments (line 15)
    kwargs_93032 = {}
    # Getting the type of 'eval' (line 15)
    eval_93030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 21), 'eval', False)
    # Calling eval(args, kwargs) (line 15)
    eval_call_result_93033 = invoke(stypy.reporting.localization.Localization(__file__, 15, 21), eval_93030, *[a_93031], **kwargs_93032)
    
    # Assigning a type to the variable 'repeat' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'repeat', eval_call_result_93033)
    # SSA branch for the else part of an if statement (line 14)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'a' (line 17)
    a_93036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 24), 'a', False)
    # Processing the call keyword arguments (line 17)
    kwargs_93037 = {}
    # Getting the type of 'args' (line 17)
    args_93034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'args', False)
    # Obtaining the member 'append' of a type (line 17)
    append_93035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 12), args_93034, 'append')
    # Calling append(args, kwargs) (line 17)
    append_call_result_93038 = invoke(stypy.reporting.localization.Localization(__file__, 17, 12), append_93035, *[a_93036], **kwargs_93037)
    
    # SSA join for if statement (line 14)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 18):
    
    # Call to join(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'args' (line 18)
    args_93041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'args', False)
    # Processing the call keyword arguments (line 18)
    kwargs_93042 = {}
    str_93039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 16), 'str', ' ')
    # Obtaining the member 'join' of a type (line 18)
    join_93040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 16), str_93039, 'join')
    # Calling join(args, kwargs) (line 18)
    join_call_result_93043 = invoke(stypy.reporting.localization.Localization(__file__, 18, 16), join_93040, *[args_93041], **kwargs_93042)
    
    # Assigning a type to the variable 'f2py_opts' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'f2py_opts', join_call_result_93043)
    
    # Obtaining an instance of the builtin type 'tuple' (line 19)
    tuple_93044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 19)
    # Adding element type (line 19)
    # Getting the type of 'repeat' (line 19)
    repeat_93045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'repeat')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 11), tuple_93044, repeat_93045)
    # Adding element type (line 19)
    # Getting the type of 'f2py_opts' (line 19)
    f2py_opts_93046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'f2py_opts')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 11), tuple_93044, f2py_opts_93046)
    
    # Assigning a type to the variable 'stypy_return_type' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type', tuple_93044)
    
    # ################# End of 'cmdline(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cmdline' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_93047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_93047)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cmdline'
    return stypy_return_type_93047

# Assigning a type to the variable 'cmdline' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'cmdline', cmdline)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_93048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 40), 'int')
    defaults = [int_93048]
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 22, 0, False)
    
    # Passed parameters checking function
    run.stypy_localization = localization
    run.stypy_type_of_self = None
    run.stypy_type_store = module_type_store
    run.stypy_function_name = 'run'
    run.stypy_param_names_list = ['runtest', 'test_functions', 'repeat']
    run.stypy_varargs_param_name = None
    run.stypy_kwargs_param_name = None
    run.stypy_call_defaults = defaults
    run.stypy_call_varargs = varargs
    run.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'run', ['runtest', 'test_functions', 'repeat'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'run', localization, ['runtest', 'test_functions', 'repeat'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'run(...)' code ##################

    
    # Assigning a ListComp to a Name (line 23):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'test_functions' (line 23)
    test_functions_93066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 62), 'test_functions')
    comprehension_93067 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), test_functions_93066)
    # Assigning a type to the variable 't' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 9), 't', comprehension_93067)
    
    # Obtaining an instance of the builtin type 'tuple' (line 23)
    tuple_93049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 23)
    # Adding element type (line 23)
    # Getting the type of 't' (line 23)
    t_93050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 't')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 10), tuple_93049, t_93050)
    # Adding element type (line 23)
    
    # Call to repr(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Call to strip(...): (line 23)
    # Processing the call keyword arguments (line 23)
    kwargs_93062 = {}
    
    # Obtaining the type of the subscript
    int_93052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 40), 'int')
    
    # Call to split(...): (line 23)
    # Processing the call arguments (line 23)
    str_93056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 34), 'str', '\n')
    # Processing the call keyword arguments (line 23)
    kwargs_93057 = {}
    # Getting the type of 't' (line 23)
    t_93053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 18), 't', False)
    # Obtaining the member '__doc__' of a type (line 23)
    doc___93054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 18), t_93053, '__doc__')
    # Obtaining the member 'split' of a type (line 23)
    split_93055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 18), doc___93054, 'split')
    # Calling split(args, kwargs) (line 23)
    split_call_result_93058 = invoke(stypy.reporting.localization.Localization(__file__, 23, 18), split_93055, *[str_93056], **kwargs_93057)
    
    # Obtaining the member '__getitem__' of a type (line 23)
    getitem___93059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 18), split_call_result_93058, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 23)
    subscript_call_result_93060 = invoke(stypy.reporting.localization.Localization(__file__, 23, 18), getitem___93059, int_93052)
    
    # Obtaining the member 'strip' of a type (line 23)
    strip_93061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 18), subscript_call_result_93060, 'strip')
    # Calling strip(args, kwargs) (line 23)
    strip_call_result_93063 = invoke(stypy.reporting.localization.Localization(__file__, 23, 18), strip_93061, *[], **kwargs_93062)
    
    # Processing the call keyword arguments (line 23)
    kwargs_93064 = {}
    # Getting the type of 'repr' (line 23)
    repr_93051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 13), 'repr', False)
    # Calling repr(args, kwargs) (line 23)
    repr_call_result_93065 = invoke(stypy.reporting.localization.Localization(__file__, 23, 13), repr_93051, *[strip_call_result_93063], **kwargs_93064)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 10), tuple_93049, repr_call_result_93065)
    
    list_93068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), list_93068, tuple_93049)
    # Assigning a type to the variable 'l' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'l', list_93068)
    
    # Assigning a Call to a Name (line 24):
    
    # Call to memusage(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_93070 = {}
    # Getting the type of 'memusage' (line 24)
    memusage_93069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 21), 'memusage', False)
    # Calling memusage(args, kwargs) (line 24)
    memusage_call_result_93071 = invoke(stypy.reporting.localization.Localization(__file__, 24, 21), memusage_93069, *[], **kwargs_93070)
    
    # Assigning a type to the variable 'start_memusage' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'start_memusage', memusage_call_result_93071)
    
    # Assigning a Name to a Name (line 25):
    # Getting the type of 'None' (line 25)
    None_93072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'None')
    # Assigning a type to the variable 'diff_memusage' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'diff_memusage', None_93072)
    
    # Assigning a Call to a Name (line 26):
    
    # Call to jiffies(...): (line 26)
    # Processing the call keyword arguments (line 26)
    kwargs_93074 = {}
    # Getting the type of 'jiffies' (line 26)
    jiffies_93073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 20), 'jiffies', False)
    # Calling jiffies(args, kwargs) (line 26)
    jiffies_call_result_93075 = invoke(stypy.reporting.localization.Localization(__file__, 26, 20), jiffies_93073, *[], **kwargs_93074)
    
    # Assigning a type to the variable 'start_jiffies' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'start_jiffies', jiffies_call_result_93075)
    
    # Assigning a Num to a Name (line 27):
    int_93076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 8), 'int')
    # Assigning a type to the variable 'i' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'i', int_93076)
    
    
    # Getting the type of 'i' (line 28)
    i_93077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 10), 'i')
    # Getting the type of 'repeat' (line 28)
    repeat_93078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 14), 'repeat')
    # Applying the binary operator '<' (line 28)
    result_lt_93079 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 10), '<', i_93077, repeat_93078)
    
    # Testing the type of an if condition (line 28)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 4), result_lt_93079)
    # SSA begins for while statement (line 28)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Getting the type of 'i' (line 29)
    i_93080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'i')
    int_93081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 13), 'int')
    # Applying the binary operator '+=' (line 29)
    result_iadd_93082 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 8), '+=', i_93080, int_93081)
    # Assigning a type to the variable 'i' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'i', result_iadd_93082)
    
    
    # Getting the type of 'l' (line 30)
    l_93083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'l')
    # Testing the type of a for loop iterable (line 30)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 30, 8), l_93083)
    # Getting the type of the for loop variable (line 30)
    for_loop_var_93084 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 30, 8), l_93083)
    # Assigning a type to the variable 't' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 't', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 8), for_loop_var_93084))
    # Assigning a type to the variable 'fname' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'fname', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 8), for_loop_var_93084))
    # SSA begins for a for statement (line 30)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to runtest(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 't' (line 31)
    t_93086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 't', False)
    # Processing the call keyword arguments (line 31)
    kwargs_93087 = {}
    # Getting the type of 'runtest' (line 31)
    runtest_93085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'runtest', False)
    # Calling runtest(args, kwargs) (line 31)
    runtest_call_result_93088 = invoke(stypy.reporting.localization.Localization(__file__, 31, 12), runtest_93085, *[t_93086], **kwargs_93087)
    
    
    # Type idiom detected: calculating its left and rigth part (line 32)
    # Getting the type of 'start_memusage' (line 32)
    start_memusage_93089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'start_memusage')
    # Getting the type of 'None' (line 32)
    None_93090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 33), 'None')
    
    (may_be_93091, more_types_in_union_93092) = may_be_none(start_memusage_93089, None_93090)

    if may_be_93091:

        if more_types_in_union_93092:
            # Runtime conditional SSA (line 32)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        if more_types_in_union_93092:
            # SSA join for if statement (line 32)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 34)
    # Getting the type of 'diff_memusage' (line 34)
    diff_memusage_93093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'diff_memusage')
    # Getting the type of 'None' (line 34)
    None_93094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 32), 'None')
    
    (may_be_93095, more_types_in_union_93096) = may_be_none(diff_memusage_93093, None_93094)

    if may_be_93095:

        if more_types_in_union_93096:
            # Runtime conditional SSA (line 34)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 35):
        
        # Call to memusage(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_93098 = {}
        # Getting the type of 'memusage' (line 35)
        memusage_93097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 32), 'memusage', False)
        # Calling memusage(args, kwargs) (line 35)
        memusage_call_result_93099 = invoke(stypy.reporting.localization.Localization(__file__, 35, 32), memusage_93097, *[], **kwargs_93098)
        
        # Getting the type of 'start_memusage' (line 35)
        start_memusage_93100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 45), 'start_memusage')
        # Applying the binary operator '-' (line 35)
        result_sub_93101 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 32), '-', memusage_call_result_93099, start_memusage_93100)
        
        # Assigning a type to the variable 'diff_memusage' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'diff_memusage', result_sub_93101)

        if more_types_in_union_93096:
            # Runtime conditional SSA for else branch (line 34)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_93095) or more_types_in_union_93096):
        
        # Assigning a BinOp to a Name (line 37):
        
        # Call to memusage(...): (line 37)
        # Processing the call keyword arguments (line 37)
        kwargs_93103 = {}
        # Getting the type of 'memusage' (line 37)
        memusage_93102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 33), 'memusage', False)
        # Calling memusage(args, kwargs) (line 37)
        memusage_call_result_93104 = invoke(stypy.reporting.localization.Localization(__file__, 37, 33), memusage_93102, *[], **kwargs_93103)
        
        # Getting the type of 'start_memusage' (line 37)
        start_memusage_93105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 46), 'start_memusage')
        # Applying the binary operator '-' (line 37)
        result_sub_93106 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 33), '-', memusage_call_result_93104, start_memusage_93105)
        
        # Assigning a type to the variable 'diff_memusage2' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'diff_memusage2', result_sub_93106)
        
        
        # Getting the type of 'diff_memusage2' (line 38)
        diff_memusage2_93107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 19), 'diff_memusage2')
        # Getting the type of 'diff_memusage' (line 38)
        diff_memusage_93108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 37), 'diff_memusage')
        # Applying the binary operator '!=' (line 38)
        result_ne_93109 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 19), '!=', diff_memusage2_93107, diff_memusage_93108)
        
        # Testing the type of an if condition (line 38)
        if_condition_93110 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 16), result_ne_93109)
        # Assigning a type to the variable 'if_condition_93110' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'if_condition_93110', if_condition_93110)
        # SSA begins for if statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 39)
        # Processing the call arguments (line 39)
        str_93112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 26), 'str', 'memory usage change at step %i:')
        # Getting the type of 'i' (line 39)
        i_93113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 62), 'i', False)
        # Applying the binary operator '%' (line 39)
        result_mod_93114 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 26), '%', str_93112, i_93113)
        
        # Getting the type of 'diff_memusage2' (line 40)
        diff_memusage2_93115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 26), 'diff_memusage2', False)
        # Getting the type of 'diff_memusage' (line 40)
        diff_memusage_93116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 43), 'diff_memusage', False)
        # Applying the binary operator '-' (line 40)
        result_sub_93117 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 26), '-', diff_memusage2_93115, diff_memusage_93116)
        
        # Getting the type of 'fname' (line 41)
        fname_93118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 26), 'fname', False)
        # Processing the call keyword arguments (line 39)
        kwargs_93119 = {}
        # Getting the type of 'print' (line 39)
        print_93111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'print', False)
        # Calling print(args, kwargs) (line 39)
        print_call_result_93120 = invoke(stypy.reporting.localization.Localization(__file__, 39, 20), print_93111, *[result_mod_93114, result_sub_93117, fname_93118], **kwargs_93119)
        
        
        # Assigning a Name to a Name (line 42):
        # Getting the type of 'diff_memusage2' (line 42)
        diff_memusage2_93121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 36), 'diff_memusage2')
        # Assigning a type to the variable 'diff_memusage' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'diff_memusage', diff_memusage2_93121)
        # SSA join for if statement (line 38)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_93095 and more_types_in_union_93096):
            # SSA join for if statement (line 34)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 28)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 43):
    
    # Call to memusage(...): (line 43)
    # Processing the call keyword arguments (line 43)
    kwargs_93123 = {}
    # Getting the type of 'memusage' (line 43)
    memusage_93122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 23), 'memusage', False)
    # Calling memusage(args, kwargs) (line 43)
    memusage_call_result_93124 = invoke(stypy.reporting.localization.Localization(__file__, 43, 23), memusage_93122, *[], **kwargs_93123)
    
    # Assigning a type to the variable 'current_memusage' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'current_memusage', memusage_call_result_93124)
    
    # Call to print(...): (line 44)
    # Processing the call arguments (line 44)
    str_93126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 10), 'str', 'run')
    # Getting the type of 'repeat' (line 44)
    repeat_93127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 17), 'repeat', False)
    
    # Call to len(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'test_functions' (line 44)
    test_functions_93129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 30), 'test_functions', False)
    # Processing the call keyword arguments (line 44)
    kwargs_93130 = {}
    # Getting the type of 'len' (line 44)
    len_93128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 26), 'len', False)
    # Calling len(args, kwargs) (line 44)
    len_call_result_93131 = invoke(stypy.reporting.localization.Localization(__file__, 44, 26), len_93128, *[test_functions_93129], **kwargs_93130)
    
    # Applying the binary operator '*' (line 44)
    result_mul_93132 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 17), '*', repeat_93127, len_call_result_93131)
    
    str_93133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 47), 'str', 'tests')
    str_93134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 10), 'str', 'in %.2f seconds')
    
    # Call to jiffies(...): (line 45)
    # Processing the call keyword arguments (line 45)
    kwargs_93136 = {}
    # Getting the type of 'jiffies' (line 45)
    jiffies_93135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 32), 'jiffies', False)
    # Calling jiffies(args, kwargs) (line 45)
    jiffies_call_result_93137 = invoke(stypy.reporting.localization.Localization(__file__, 45, 32), jiffies_93135, *[], **kwargs_93136)
    
    # Getting the type of 'start_jiffies' (line 45)
    start_jiffies_93138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 44), 'start_jiffies', False)
    # Applying the binary operator '-' (line 45)
    result_sub_93139 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 32), '-', jiffies_call_result_93137, start_jiffies_93138)
    
    float_93140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 61), 'float')
    # Applying the binary operator 'div' (line 45)
    result_div_93141 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 31), 'div', result_sub_93139, float_93140)
    
    # Applying the binary operator '%' (line 45)
    result_mod_93142 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 10), '%', str_93134, result_div_93141)
    
    # Processing the call keyword arguments (line 44)
    kwargs_93143 = {}
    # Getting the type of 'print' (line 44)
    print_93125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'print', False)
    # Calling print(args, kwargs) (line 44)
    print_call_result_93144 = invoke(stypy.reporting.localization.Localization(__file__, 44, 4), print_93125, *[str_93126, result_mul_93132, str_93133, result_mod_93142], **kwargs_93143)
    
    
    # Getting the type of 'start_memusage' (line 46)
    start_memusage_93145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 7), 'start_memusage')
    # Testing the type of an if condition (line 46)
    if_condition_93146 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 4), start_memusage_93145)
    # Assigning a type to the variable 'if_condition_93146' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'if_condition_93146', if_condition_93146)
    # SSA begins for if statement (line 46)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 47)
    # Processing the call arguments (line 47)
    str_93148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 14), 'str', 'initial virtual memory size:')
    # Getting the type of 'start_memusage' (line 47)
    start_memusage_93149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 46), 'start_memusage', False)
    str_93150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 62), 'str', 'bytes')
    # Processing the call keyword arguments (line 47)
    kwargs_93151 = {}
    # Getting the type of 'print' (line 47)
    print_93147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'print', False)
    # Calling print(args, kwargs) (line 47)
    print_call_result_93152 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), print_93147, *[str_93148, start_memusage_93149, str_93150], **kwargs_93151)
    
    
    # Call to print(...): (line 48)
    # Processing the call arguments (line 48)
    str_93154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 14), 'str', 'current virtual memory size:')
    # Getting the type of 'current_memusage' (line 48)
    current_memusage_93155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 46), 'current_memusage', False)
    str_93156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 64), 'str', 'bytes')
    # Processing the call keyword arguments (line 48)
    kwargs_93157 = {}
    # Getting the type of 'print' (line 48)
    print_93153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'print', False)
    # Calling print(args, kwargs) (line 48)
    print_call_result_93158 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), print_93153, *[str_93154, current_memusage_93155, str_93156], **kwargs_93157)
    
    # SSA join for if statement (line 46)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_93159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_93159)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_93159

# Assigning a type to the variable 'run' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'run', run)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
