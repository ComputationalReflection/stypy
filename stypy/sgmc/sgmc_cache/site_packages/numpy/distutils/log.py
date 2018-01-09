
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Colored log, requires Python 2.3 or up.
2: from __future__ import division, absolute_import, print_function
3: 
4: import sys
5: from distutils.log import *
6: from distutils.log import Log as old_Log
7: from distutils.log import _global_log
8: 
9: if sys.version_info[0] < 3:
10:     from .misc_util import (red_text, default_text, cyan_text, green_text,
11:             is_sequence, is_string)
12: else:
13:     from numpy.distutils.misc_util import (red_text, default_text, cyan_text,
14:             green_text, is_sequence, is_string)
15: 
16: 
17: def _fix_args(args,flag=1):
18:     if is_string(args):
19:         return args.replace('%', '%%')
20:     if flag and is_sequence(args):
21:         return tuple([_fix_args(a, flag=0) for a in args])
22:     return args
23: 
24: 
25: class Log(old_Log):
26:     def _log(self, level, msg, args):
27:         if level >= self.threshold:
28:             if args:
29:                 msg = msg % _fix_args(args)
30:             if 0:
31:                 if msg.startswith('copying ') and msg.find(' -> ') != -1:
32:                     return
33:                 if msg.startswith('byte-compiling '):
34:                     return
35:             print(_global_color_map[level](msg))
36:             sys.stdout.flush()
37: 
38:     def good(self, msg, *args):
39:         '''
40:         If we log WARN messages, log this message as a 'nice' anti-warn
41:         message.
42: 
43:         '''
44:         if WARN >= self.threshold:
45:             if args:
46:                 print(green_text(msg % _fix_args(args)))
47:             else:
48:                 print(green_text(msg))
49:             sys.stdout.flush()
50: 
51: 
52: _global_log.__class__ = Log
53: 
54: good = _global_log.good
55: 
56: def set_threshold(level, force=False):
57:     prev_level = _global_log.threshold
58:     if prev_level > DEBUG or force:
59:         # If we're running at DEBUG, don't change the threshold, as there's
60:         # likely a good reason why we're running at this level.
61:         _global_log.threshold = level
62:         if level <= DEBUG:
63:             info('set_threshold: setting threshold to DEBUG level,'
64:                     ' it can be changed only with force argument')
65:     else:
66:         info('set_threshold: not changing threshold from DEBUG level'
67:                 ' %s to %s' % (prev_level, level))
68:     return prev_level
69: 
70: 
71: def set_verbosity(v, force=False):
72:     prev_level = _global_log.threshold
73:     if v < 0:
74:         set_threshold(ERROR, force)
75:     elif v == 0:
76:         set_threshold(WARN, force)
77:     elif v == 1:
78:         set_threshold(INFO, force)
79:     elif v >= 2:
80:         set_threshold(DEBUG, force)
81:     return {FATAL:-2,ERROR:-1,WARN:0,INFO:1,DEBUG:2}.get(prev_level, 1)
82: 
83: 
84: _global_color_map = {
85:     DEBUG:cyan_text,
86:     INFO:default_text,
87:     WARN:red_text,
88:     ERROR:red_text,
89:     FATAL:red_text
90: }
91: 
92: # don't use INFO,.. flags in set_verbosity, these flags are for set_threshold.
93: set_verbosity(0, force=True)
94: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from distutils.log import ' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_36674 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.log')

if (type(import_36674) is not StypyTypeError):

    if (import_36674 != 'pyd_module'):
        __import__(import_36674)
        sys_modules_36675 = sys.modules[import_36674]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.log', sys_modules_36675.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_36675, sys_modules_36675.module_type_store, module_type_store)
    else:
        from distutils.log import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.log', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'distutils.log' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.log', import_36674)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from distutils.log import old_Log' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_36676 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.log')

if (type(import_36676) is not StypyTypeError):

    if (import_36676 != 'pyd_module'):
        __import__(import_36676)
        sys_modules_36677 = sys.modules[import_36676]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.log', sys_modules_36677.module_type_store, module_type_store, ['Log'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_36677, sys_modules_36677.module_type_store, module_type_store)
    else:
        from distutils.log import Log as old_Log

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.log', None, module_type_store, ['Log'], [old_Log])

else:
    # Assigning a type to the variable 'distutils.log' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.log', import_36676)

# Adding an alias
module_type_store.add_alias('old_Log', 'Log')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.log import _global_log' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_36678 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.log')

if (type(import_36678) is not StypyTypeError):

    if (import_36678 != 'pyd_module'):
        __import__(import_36678)
        sys_modules_36679 = sys.modules[import_36678]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.log', sys_modules_36679.module_type_store, module_type_store, ['_global_log'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_36679, sys_modules_36679.module_type_store, module_type_store)
    else:
        from distutils.log import _global_log

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.log', None, module_type_store, ['_global_log'], [_global_log])

else:
    # Assigning a type to the variable 'distutils.log' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.log', import_36678)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')




# Obtaining the type of the subscript
int_36680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 20), 'int')
# Getting the type of 'sys' (line 9)
sys_36681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 9)
version_info_36682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 3), sys_36681, 'version_info')
# Obtaining the member '__getitem__' of a type (line 9)
getitem___36683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 3), version_info_36682, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 9)
subscript_call_result_36684 = invoke(stypy.reporting.localization.Localization(__file__, 9, 3), getitem___36683, int_36680)

int_36685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 25), 'int')
# Applying the binary operator '<' (line 9)
result_lt_36686 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 3), '<', subscript_call_result_36684, int_36685)

# Testing the type of an if condition (line 9)
if_condition_36687 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 9, 0), result_lt_36686)
# Assigning a type to the variable 'if_condition_36687' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'if_condition_36687', if_condition_36687)
# SSA begins for if statement (line 9)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 4))

# 'from numpy.distutils.misc_util import red_text, default_text, cyan_text, green_text, is_sequence, is_string' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_36688 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.misc_util')

if (type(import_36688) is not StypyTypeError):

    if (import_36688 != 'pyd_module'):
        __import__(import_36688)
        sys_modules_36689 = sys.modules[import_36688]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.misc_util', sys_modules_36689.module_type_store, module_type_store, ['red_text', 'default_text', 'cyan_text', 'green_text', 'is_sequence', 'is_string'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 4), __file__, sys_modules_36689, sys_modules_36689.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import red_text, default_text, cyan_text, green_text, is_sequence, is_string

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.misc_util', None, module_type_store, ['red_text', 'default_text', 'cyan_text', 'green_text', 'is_sequence', 'is_string'], [red_text, default_text, cyan_text, green_text, is_sequence, is_string])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.misc_util', import_36688)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

# SSA branch for the else part of an if statement (line 9)
module_type_store.open_ssa_branch('else')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 4))

# 'from numpy.distutils.misc_util import red_text, default_text, cyan_text, green_text, is_sequence, is_string' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_36690 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.misc_util')

if (type(import_36690) is not StypyTypeError):

    if (import_36690 != 'pyd_module'):
        __import__(import_36690)
        sys_modules_36691 = sys.modules[import_36690]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.misc_util', sys_modules_36691.module_type_store, module_type_store, ['red_text', 'default_text', 'cyan_text', 'green_text', 'is_sequence', 'is_string'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 4), __file__, sys_modules_36691, sys_modules_36691.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import red_text, default_text, cyan_text, green_text, is_sequence, is_string

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.misc_util', None, module_type_store, ['red_text', 'default_text', 'cyan_text', 'green_text', 'is_sequence', 'is_string'], [red_text, default_text, cyan_text, green_text, is_sequence, is_string])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.misc_util', import_36690)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

# SSA join for if statement (line 9)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def _fix_args(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_36692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 24), 'int')
    defaults = [int_36692]
    # Create a new context for function '_fix_args'
    module_type_store = module_type_store.open_function_context('_fix_args', 17, 0, False)
    
    # Passed parameters checking function
    _fix_args.stypy_localization = localization
    _fix_args.stypy_type_of_self = None
    _fix_args.stypy_type_store = module_type_store
    _fix_args.stypy_function_name = '_fix_args'
    _fix_args.stypy_param_names_list = ['args', 'flag']
    _fix_args.stypy_varargs_param_name = None
    _fix_args.stypy_kwargs_param_name = None
    _fix_args.stypy_call_defaults = defaults
    _fix_args.stypy_call_varargs = varargs
    _fix_args.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fix_args', ['args', 'flag'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fix_args', localization, ['args', 'flag'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fix_args(...)' code ##################

    
    
    # Call to is_string(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'args' (line 18)
    args_36694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), 'args', False)
    # Processing the call keyword arguments (line 18)
    kwargs_36695 = {}
    # Getting the type of 'is_string' (line 18)
    is_string_36693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 7), 'is_string', False)
    # Calling is_string(args, kwargs) (line 18)
    is_string_call_result_36696 = invoke(stypy.reporting.localization.Localization(__file__, 18, 7), is_string_36693, *[args_36694], **kwargs_36695)
    
    # Testing the type of an if condition (line 18)
    if_condition_36697 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 18, 4), is_string_call_result_36696)
    # Assigning a type to the variable 'if_condition_36697' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'if_condition_36697', if_condition_36697)
    # SSA begins for if statement (line 18)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to replace(...): (line 19)
    # Processing the call arguments (line 19)
    str_36700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 28), 'str', '%')
    str_36701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 33), 'str', '%%')
    # Processing the call keyword arguments (line 19)
    kwargs_36702 = {}
    # Getting the type of 'args' (line 19)
    args_36698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'args', False)
    # Obtaining the member 'replace' of a type (line 19)
    replace_36699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 15), args_36698, 'replace')
    # Calling replace(args, kwargs) (line 19)
    replace_call_result_36703 = invoke(stypy.reporting.localization.Localization(__file__, 19, 15), replace_36699, *[str_36700, str_36701], **kwargs_36702)
    
    # Assigning a type to the variable 'stypy_return_type' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'stypy_return_type', replace_call_result_36703)
    # SSA join for if statement (line 18)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'flag' (line 20)
    flag_36704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 7), 'flag')
    
    # Call to is_sequence(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'args' (line 20)
    args_36706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 28), 'args', False)
    # Processing the call keyword arguments (line 20)
    kwargs_36707 = {}
    # Getting the type of 'is_sequence' (line 20)
    is_sequence_36705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'is_sequence', False)
    # Calling is_sequence(args, kwargs) (line 20)
    is_sequence_call_result_36708 = invoke(stypy.reporting.localization.Localization(__file__, 20, 16), is_sequence_36705, *[args_36706], **kwargs_36707)
    
    # Applying the binary operator 'and' (line 20)
    result_and_keyword_36709 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 7), 'and', flag_36704, is_sequence_call_result_36708)
    
    # Testing the type of an if condition (line 20)
    if_condition_36710 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 4), result_and_keyword_36709)
    # Assigning a type to the variable 'if_condition_36710' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'if_condition_36710', if_condition_36710)
    # SSA begins for if statement (line 20)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to tuple(...): (line 21)
    # Processing the call arguments (line 21)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'args' (line 21)
    args_36718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 52), 'args', False)
    comprehension_36719 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 22), args_36718)
    # Assigning a type to the variable 'a' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'a', comprehension_36719)
    
    # Call to _fix_args(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'a' (line 21)
    a_36713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 32), 'a', False)
    # Processing the call keyword arguments (line 21)
    int_36714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 40), 'int')
    keyword_36715 = int_36714
    kwargs_36716 = {'flag': keyword_36715}
    # Getting the type of '_fix_args' (line 21)
    _fix_args_36712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), '_fix_args', False)
    # Calling _fix_args(args, kwargs) (line 21)
    _fix_args_call_result_36717 = invoke(stypy.reporting.localization.Localization(__file__, 21, 22), _fix_args_36712, *[a_36713], **kwargs_36716)
    
    list_36720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 22), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 22), list_36720, _fix_args_call_result_36717)
    # Processing the call keyword arguments (line 21)
    kwargs_36721 = {}
    # Getting the type of 'tuple' (line 21)
    tuple_36711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 21)
    tuple_call_result_36722 = invoke(stypy.reporting.localization.Localization(__file__, 21, 15), tuple_36711, *[list_36720], **kwargs_36721)
    
    # Assigning a type to the variable 'stypy_return_type' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'stypy_return_type', tuple_call_result_36722)
    # SSA join for if statement (line 20)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'args' (line 22)
    args_36723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'args')
    # Assigning a type to the variable 'stypy_return_type' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type', args_36723)
    
    # ################# End of '_fix_args(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fix_args' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_36724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36724)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fix_args'
    return stypy_return_type_36724

# Assigning a type to the variable '_fix_args' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), '_fix_args', _fix_args)
# Declaration of the 'Log' class
# Getting the type of 'old_Log' (line 25)
old_Log_36725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 10), 'old_Log')

class Log(old_Log_36725, ):

    @norecursion
    def _log(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_log'
        module_type_store = module_type_store.open_function_context('_log', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Log._log.__dict__.__setitem__('stypy_localization', localization)
        Log._log.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Log._log.__dict__.__setitem__('stypy_type_store', module_type_store)
        Log._log.__dict__.__setitem__('stypy_function_name', 'Log._log')
        Log._log.__dict__.__setitem__('stypy_param_names_list', ['level', 'msg', 'args'])
        Log._log.__dict__.__setitem__('stypy_varargs_param_name', None)
        Log._log.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Log._log.__dict__.__setitem__('stypy_call_defaults', defaults)
        Log._log.__dict__.__setitem__('stypy_call_varargs', varargs)
        Log._log.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Log._log.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Log._log', ['level', 'msg', 'args'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_log', localization, ['level', 'msg', 'args'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_log(...)' code ##################

        
        
        # Getting the type of 'level' (line 27)
        level_36726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'level')
        # Getting the type of 'self' (line 27)
        self_36727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 20), 'self')
        # Obtaining the member 'threshold' of a type (line 27)
        threshold_36728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 20), self_36727, 'threshold')
        # Applying the binary operator '>=' (line 27)
        result_ge_36729 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 11), '>=', level_36726, threshold_36728)
        
        # Testing the type of an if condition (line 27)
        if_condition_36730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 8), result_ge_36729)
        # Assigning a type to the variable 'if_condition_36730' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'if_condition_36730', if_condition_36730)
        # SSA begins for if statement (line 27)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'args' (line 28)
        args_36731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'args')
        # Testing the type of an if condition (line 28)
        if_condition_36732 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 12), args_36731)
        # Assigning a type to the variable 'if_condition_36732' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'if_condition_36732', if_condition_36732)
        # SSA begins for if statement (line 28)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 29):
        # Getting the type of 'msg' (line 29)
        msg_36733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'msg')
        
        # Call to _fix_args(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'args' (line 29)
        args_36735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 38), 'args', False)
        # Processing the call keyword arguments (line 29)
        kwargs_36736 = {}
        # Getting the type of '_fix_args' (line 29)
        _fix_args_36734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 28), '_fix_args', False)
        # Calling _fix_args(args, kwargs) (line 29)
        _fix_args_call_result_36737 = invoke(stypy.reporting.localization.Localization(__file__, 29, 28), _fix_args_36734, *[args_36735], **kwargs_36736)
        
        # Applying the binary operator '%' (line 29)
        result_mod_36738 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 22), '%', msg_36733, _fix_args_call_result_36737)
        
        # Assigning a type to the variable 'msg' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 16), 'msg', result_mod_36738)
        # SSA join for if statement (line 28)
        module_type_store = module_type_store.join_ssa_context()
        
        
        int_36739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 15), 'int')
        # Testing the type of an if condition (line 30)
        if_condition_36740 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 12), int_36739)
        # Assigning a type to the variable 'if_condition_36740' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'if_condition_36740', if_condition_36740)
        # SSA begins for if statement (line 30)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        
        # Call to startswith(...): (line 31)
        # Processing the call arguments (line 31)
        str_36743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 34), 'str', 'copying ')
        # Processing the call keyword arguments (line 31)
        kwargs_36744 = {}
        # Getting the type of 'msg' (line 31)
        msg_36741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'msg', False)
        # Obtaining the member 'startswith' of a type (line 31)
        startswith_36742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 19), msg_36741, 'startswith')
        # Calling startswith(args, kwargs) (line 31)
        startswith_call_result_36745 = invoke(stypy.reporting.localization.Localization(__file__, 31, 19), startswith_36742, *[str_36743], **kwargs_36744)
        
        
        
        # Call to find(...): (line 31)
        # Processing the call arguments (line 31)
        str_36748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 59), 'str', ' -> ')
        # Processing the call keyword arguments (line 31)
        kwargs_36749 = {}
        # Getting the type of 'msg' (line 31)
        msg_36746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 50), 'msg', False)
        # Obtaining the member 'find' of a type (line 31)
        find_36747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 50), msg_36746, 'find')
        # Calling find(args, kwargs) (line 31)
        find_call_result_36750 = invoke(stypy.reporting.localization.Localization(__file__, 31, 50), find_36747, *[str_36748], **kwargs_36749)
        
        int_36751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 70), 'int')
        # Applying the binary operator '!=' (line 31)
        result_ne_36752 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 50), '!=', find_call_result_36750, int_36751)
        
        # Applying the binary operator 'and' (line 31)
        result_and_keyword_36753 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 19), 'and', startswith_call_result_36745, result_ne_36752)
        
        # Testing the type of an if condition (line 31)
        if_condition_36754 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 16), result_and_keyword_36753)
        # Assigning a type to the variable 'if_condition_36754' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'if_condition_36754', if_condition_36754)
        # SSA begins for if statement (line 31)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 20), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 31)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to startswith(...): (line 33)
        # Processing the call arguments (line 33)
        str_36757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 34), 'str', 'byte-compiling ')
        # Processing the call keyword arguments (line 33)
        kwargs_36758 = {}
        # Getting the type of 'msg' (line 33)
        msg_36755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'msg', False)
        # Obtaining the member 'startswith' of a type (line 33)
        startswith_36756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 19), msg_36755, 'startswith')
        # Calling startswith(args, kwargs) (line 33)
        startswith_call_result_36759 = invoke(stypy.reporting.localization.Localization(__file__, 33, 19), startswith_36756, *[str_36757], **kwargs_36758)
        
        # Testing the type of an if condition (line 33)
        if_condition_36760 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 16), startswith_call_result_36759)
        # Assigning a type to the variable 'if_condition_36760' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'if_condition_36760', if_condition_36760)
        # SSA begins for if statement (line 33)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 20), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 33)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 30)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to print(...): (line 35)
        # Processing the call arguments (line 35)
        
        # Call to (...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'msg' (line 35)
        msg_36766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 43), 'msg', False)
        # Processing the call keyword arguments (line 35)
        kwargs_36767 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'level' (line 35)
        level_36762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 36), 'level', False)
        # Getting the type of '_global_color_map' (line 35)
        _global_color_map_36763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 18), '_global_color_map', False)
        # Obtaining the member '__getitem__' of a type (line 35)
        getitem___36764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 18), _global_color_map_36763, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 35)
        subscript_call_result_36765 = invoke(stypy.reporting.localization.Localization(__file__, 35, 18), getitem___36764, level_36762)
        
        # Calling (args, kwargs) (line 35)
        _call_result_36768 = invoke(stypy.reporting.localization.Localization(__file__, 35, 18), subscript_call_result_36765, *[msg_36766], **kwargs_36767)
        
        # Processing the call keyword arguments (line 35)
        kwargs_36769 = {}
        # Getting the type of 'print' (line 35)
        print_36761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'print', False)
        # Calling print(args, kwargs) (line 35)
        print_call_result_36770 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), print_36761, *[_call_result_36768], **kwargs_36769)
        
        
        # Call to flush(...): (line 36)
        # Processing the call keyword arguments (line 36)
        kwargs_36774 = {}
        # Getting the type of 'sys' (line 36)
        sys_36771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'sys', False)
        # Obtaining the member 'stdout' of a type (line 36)
        stdout_36772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), sys_36771, 'stdout')
        # Obtaining the member 'flush' of a type (line 36)
        flush_36773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), stdout_36772, 'flush')
        # Calling flush(args, kwargs) (line 36)
        flush_call_result_36775 = invoke(stypy.reporting.localization.Localization(__file__, 36, 12), flush_36773, *[], **kwargs_36774)
        
        # SSA join for if statement (line 27)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_log(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_log' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_36776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36776)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_log'
        return stypy_return_type_36776


    @norecursion
    def good(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'good'
        module_type_store = module_type_store.open_function_context('good', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Log.good.__dict__.__setitem__('stypy_localization', localization)
        Log.good.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Log.good.__dict__.__setitem__('stypy_type_store', module_type_store)
        Log.good.__dict__.__setitem__('stypy_function_name', 'Log.good')
        Log.good.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        Log.good.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        Log.good.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Log.good.__dict__.__setitem__('stypy_call_defaults', defaults)
        Log.good.__dict__.__setitem__('stypy_call_varargs', varargs)
        Log.good.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Log.good.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Log.good', ['msg'], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'good', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'good(...)' code ##################

        str_36777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'str', "\n        If we log WARN messages, log this message as a 'nice' anti-warn\n        message.\n\n        ")
        
        
        # Getting the type of 'WARN' (line 44)
        WARN_36778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'WARN')
        # Getting the type of 'self' (line 44)
        self_36779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'self')
        # Obtaining the member 'threshold' of a type (line 44)
        threshold_36780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 19), self_36779, 'threshold')
        # Applying the binary operator '>=' (line 44)
        result_ge_36781 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 11), '>=', WARN_36778, threshold_36780)
        
        # Testing the type of an if condition (line 44)
        if_condition_36782 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 8), result_ge_36781)
        # Assigning a type to the variable 'if_condition_36782' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'if_condition_36782', if_condition_36782)
        # SSA begins for if statement (line 44)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'args' (line 45)
        args_36783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'args')
        # Testing the type of an if condition (line 45)
        if_condition_36784 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 12), args_36783)
        # Assigning a type to the variable 'if_condition_36784' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'if_condition_36784', if_condition_36784)
        # SSA begins for if statement (line 45)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Call to green_text(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'msg' (line 46)
        msg_36787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 33), 'msg', False)
        
        # Call to _fix_args(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'args' (line 46)
        args_36789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 49), 'args', False)
        # Processing the call keyword arguments (line 46)
        kwargs_36790 = {}
        # Getting the type of '_fix_args' (line 46)
        _fix_args_36788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 39), '_fix_args', False)
        # Calling _fix_args(args, kwargs) (line 46)
        _fix_args_call_result_36791 = invoke(stypy.reporting.localization.Localization(__file__, 46, 39), _fix_args_36788, *[args_36789], **kwargs_36790)
        
        # Applying the binary operator '%' (line 46)
        result_mod_36792 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 33), '%', msg_36787, _fix_args_call_result_36791)
        
        # Processing the call keyword arguments (line 46)
        kwargs_36793 = {}
        # Getting the type of 'green_text' (line 46)
        green_text_36786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'green_text', False)
        # Calling green_text(args, kwargs) (line 46)
        green_text_call_result_36794 = invoke(stypy.reporting.localization.Localization(__file__, 46, 22), green_text_36786, *[result_mod_36792], **kwargs_36793)
        
        # Processing the call keyword arguments (line 46)
        kwargs_36795 = {}
        # Getting the type of 'print' (line 46)
        print_36785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'print', False)
        # Calling print(args, kwargs) (line 46)
        print_call_result_36796 = invoke(stypy.reporting.localization.Localization(__file__, 46, 16), print_36785, *[green_text_call_result_36794], **kwargs_36795)
        
        # SSA branch for the else part of an if statement (line 45)
        module_type_store.open_ssa_branch('else')
        
        # Call to print(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Call to green_text(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'msg' (line 48)
        msg_36799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 33), 'msg', False)
        # Processing the call keyword arguments (line 48)
        kwargs_36800 = {}
        # Getting the type of 'green_text' (line 48)
        green_text_36798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'green_text', False)
        # Calling green_text(args, kwargs) (line 48)
        green_text_call_result_36801 = invoke(stypy.reporting.localization.Localization(__file__, 48, 22), green_text_36798, *[msg_36799], **kwargs_36800)
        
        # Processing the call keyword arguments (line 48)
        kwargs_36802 = {}
        # Getting the type of 'print' (line 48)
        print_36797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'print', False)
        # Calling print(args, kwargs) (line 48)
        print_call_result_36803 = invoke(stypy.reporting.localization.Localization(__file__, 48, 16), print_36797, *[green_text_call_result_36801], **kwargs_36802)
        
        # SSA join for if statement (line 45)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to flush(...): (line 49)
        # Processing the call keyword arguments (line 49)
        kwargs_36807 = {}
        # Getting the type of 'sys' (line 49)
        sys_36804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'sys', False)
        # Obtaining the member 'stdout' of a type (line 49)
        stdout_36805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), sys_36804, 'stdout')
        # Obtaining the member 'flush' of a type (line 49)
        flush_36806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), stdout_36805, 'flush')
        # Calling flush(args, kwargs) (line 49)
        flush_call_result_36808 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), flush_36806, *[], **kwargs_36807)
        
        # SSA join for if statement (line 44)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'good(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'good' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_36809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36809)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'good'
        return stypy_return_type_36809


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 25, 0, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Log.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Log' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'Log', Log)

# Assigning a Name to a Attribute (line 52):
# Getting the type of 'Log' (line 52)
Log_36810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 24), 'Log')
# Getting the type of '_global_log' (line 52)
_global_log_36811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), '_global_log')
# Setting the type of the member '__class__' of a type (line 52)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 0), _global_log_36811, '__class__', Log_36810)

# Assigning a Attribute to a Name (line 54):
# Getting the type of '_global_log' (line 54)
_global_log_36812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 7), '_global_log')
# Obtaining the member 'good' of a type (line 54)
good_36813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 7), _global_log_36812, 'good')
# Assigning a type to the variable 'good' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'good', good_36813)

@norecursion
def set_threshold(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 56)
    False_36814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 31), 'False')
    defaults = [False_36814]
    # Create a new context for function 'set_threshold'
    module_type_store = module_type_store.open_function_context('set_threshold', 56, 0, False)
    
    # Passed parameters checking function
    set_threshold.stypy_localization = localization
    set_threshold.stypy_type_of_self = None
    set_threshold.stypy_type_store = module_type_store
    set_threshold.stypy_function_name = 'set_threshold'
    set_threshold.stypy_param_names_list = ['level', 'force']
    set_threshold.stypy_varargs_param_name = None
    set_threshold.stypy_kwargs_param_name = None
    set_threshold.stypy_call_defaults = defaults
    set_threshold.stypy_call_varargs = varargs
    set_threshold.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'set_threshold', ['level', 'force'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'set_threshold', localization, ['level', 'force'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'set_threshold(...)' code ##################

    
    # Assigning a Attribute to a Name (line 57):
    # Getting the type of '_global_log' (line 57)
    _global_log_36815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 17), '_global_log')
    # Obtaining the member 'threshold' of a type (line 57)
    threshold_36816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 17), _global_log_36815, 'threshold')
    # Assigning a type to the variable 'prev_level' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'prev_level', threshold_36816)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'prev_level' (line 58)
    prev_level_36817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 7), 'prev_level')
    # Getting the type of 'DEBUG' (line 58)
    DEBUG_36818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 20), 'DEBUG')
    # Applying the binary operator '>' (line 58)
    result_gt_36819 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 7), '>', prev_level_36817, DEBUG_36818)
    
    # Getting the type of 'force' (line 58)
    force_36820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 29), 'force')
    # Applying the binary operator 'or' (line 58)
    result_or_keyword_36821 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 7), 'or', result_gt_36819, force_36820)
    
    # Testing the type of an if condition (line 58)
    if_condition_36822 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 4), result_or_keyword_36821)
    # Assigning a type to the variable 'if_condition_36822' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'if_condition_36822', if_condition_36822)
    # SSA begins for if statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Attribute (line 61):
    # Getting the type of 'level' (line 61)
    level_36823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 32), 'level')
    # Getting the type of '_global_log' (line 61)
    _global_log_36824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), '_global_log')
    # Setting the type of the member 'threshold' of a type (line 61)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), _global_log_36824, 'threshold', level_36823)
    
    
    # Getting the type of 'level' (line 62)
    level_36825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 11), 'level')
    # Getting the type of 'DEBUG' (line 62)
    DEBUG_36826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'DEBUG')
    # Applying the binary operator '<=' (line 62)
    result_le_36827 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 11), '<=', level_36825, DEBUG_36826)
    
    # Testing the type of an if condition (line 62)
    if_condition_36828 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 8), result_le_36827)
    # Assigning a type to the variable 'if_condition_36828' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'if_condition_36828', if_condition_36828)
    # SSA begins for if statement (line 62)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to info(...): (line 63)
    # Processing the call arguments (line 63)
    str_36830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 17), 'str', 'set_threshold: setting threshold to DEBUG level, it can be changed only with force argument')
    # Processing the call keyword arguments (line 63)
    kwargs_36831 = {}
    # Getting the type of 'info' (line 63)
    info_36829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'info', False)
    # Calling info(args, kwargs) (line 63)
    info_call_result_36832 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), info_36829, *[str_36830], **kwargs_36831)
    
    # SSA join for if statement (line 62)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 58)
    module_type_store.open_ssa_branch('else')
    
    # Call to info(...): (line 66)
    # Processing the call arguments (line 66)
    str_36834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 13), 'str', 'set_threshold: not changing threshold from DEBUG level %s to %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 67)
    tuple_36835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 67)
    # Adding element type (line 67)
    # Getting the type of 'prev_level' (line 67)
    prev_level_36836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 31), 'prev_level', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 31), tuple_36835, prev_level_36836)
    # Adding element type (line 67)
    # Getting the type of 'level' (line 67)
    level_36837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 43), 'level', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 31), tuple_36835, level_36837)
    
    # Applying the binary operator '%' (line 66)
    result_mod_36838 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 13), '%', str_36834, tuple_36835)
    
    # Processing the call keyword arguments (line 66)
    kwargs_36839 = {}
    # Getting the type of 'info' (line 66)
    info_36833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'info', False)
    # Calling info(args, kwargs) (line 66)
    info_call_result_36840 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), info_36833, *[result_mod_36838], **kwargs_36839)
    
    # SSA join for if statement (line 58)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'prev_level' (line 68)
    prev_level_36841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'prev_level')
    # Assigning a type to the variable 'stypy_return_type' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type', prev_level_36841)
    
    # ################# End of 'set_threshold(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'set_threshold' in the type store
    # Getting the type of 'stypy_return_type' (line 56)
    stypy_return_type_36842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36842)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'set_threshold'
    return stypy_return_type_36842

# Assigning a type to the variable 'set_threshold' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'set_threshold', set_threshold)

@norecursion
def set_verbosity(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 71)
    False_36843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), 'False')
    defaults = [False_36843]
    # Create a new context for function 'set_verbosity'
    module_type_store = module_type_store.open_function_context('set_verbosity', 71, 0, False)
    
    # Passed parameters checking function
    set_verbosity.stypy_localization = localization
    set_verbosity.stypy_type_of_self = None
    set_verbosity.stypy_type_store = module_type_store
    set_verbosity.stypy_function_name = 'set_verbosity'
    set_verbosity.stypy_param_names_list = ['v', 'force']
    set_verbosity.stypy_varargs_param_name = None
    set_verbosity.stypy_kwargs_param_name = None
    set_verbosity.stypy_call_defaults = defaults
    set_verbosity.stypy_call_varargs = varargs
    set_verbosity.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'set_verbosity', ['v', 'force'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'set_verbosity', localization, ['v', 'force'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'set_verbosity(...)' code ##################

    
    # Assigning a Attribute to a Name (line 72):
    # Getting the type of '_global_log' (line 72)
    _global_log_36844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), '_global_log')
    # Obtaining the member 'threshold' of a type (line 72)
    threshold_36845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 17), _global_log_36844, 'threshold')
    # Assigning a type to the variable 'prev_level' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'prev_level', threshold_36845)
    
    
    # Getting the type of 'v' (line 73)
    v_36846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 7), 'v')
    int_36847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 11), 'int')
    # Applying the binary operator '<' (line 73)
    result_lt_36848 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 7), '<', v_36846, int_36847)
    
    # Testing the type of an if condition (line 73)
    if_condition_36849 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 4), result_lt_36848)
    # Assigning a type to the variable 'if_condition_36849' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'if_condition_36849', if_condition_36849)
    # SSA begins for if statement (line 73)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to set_threshold(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'ERROR' (line 74)
    ERROR_36851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 22), 'ERROR', False)
    # Getting the type of 'force' (line 74)
    force_36852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 29), 'force', False)
    # Processing the call keyword arguments (line 74)
    kwargs_36853 = {}
    # Getting the type of 'set_threshold' (line 74)
    set_threshold_36850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'set_threshold', False)
    # Calling set_threshold(args, kwargs) (line 74)
    set_threshold_call_result_36854 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), set_threshold_36850, *[ERROR_36851, force_36852], **kwargs_36853)
    
    # SSA branch for the else part of an if statement (line 73)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'v' (line 75)
    v_36855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 9), 'v')
    int_36856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 14), 'int')
    # Applying the binary operator '==' (line 75)
    result_eq_36857 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 9), '==', v_36855, int_36856)
    
    # Testing the type of an if condition (line 75)
    if_condition_36858 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 9), result_eq_36857)
    # Assigning a type to the variable 'if_condition_36858' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 9), 'if_condition_36858', if_condition_36858)
    # SSA begins for if statement (line 75)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to set_threshold(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'WARN' (line 76)
    WARN_36860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 22), 'WARN', False)
    # Getting the type of 'force' (line 76)
    force_36861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 28), 'force', False)
    # Processing the call keyword arguments (line 76)
    kwargs_36862 = {}
    # Getting the type of 'set_threshold' (line 76)
    set_threshold_36859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'set_threshold', False)
    # Calling set_threshold(args, kwargs) (line 76)
    set_threshold_call_result_36863 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), set_threshold_36859, *[WARN_36860, force_36861], **kwargs_36862)
    
    # SSA branch for the else part of an if statement (line 75)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'v' (line 77)
    v_36864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 9), 'v')
    int_36865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 14), 'int')
    # Applying the binary operator '==' (line 77)
    result_eq_36866 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 9), '==', v_36864, int_36865)
    
    # Testing the type of an if condition (line 77)
    if_condition_36867 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 9), result_eq_36866)
    # Assigning a type to the variable 'if_condition_36867' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 9), 'if_condition_36867', if_condition_36867)
    # SSA begins for if statement (line 77)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to set_threshold(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'INFO' (line 78)
    INFO_36869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 22), 'INFO', False)
    # Getting the type of 'force' (line 78)
    force_36870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 28), 'force', False)
    # Processing the call keyword arguments (line 78)
    kwargs_36871 = {}
    # Getting the type of 'set_threshold' (line 78)
    set_threshold_36868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'set_threshold', False)
    # Calling set_threshold(args, kwargs) (line 78)
    set_threshold_call_result_36872 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), set_threshold_36868, *[INFO_36869, force_36870], **kwargs_36871)
    
    # SSA branch for the else part of an if statement (line 77)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'v' (line 79)
    v_36873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 9), 'v')
    int_36874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 14), 'int')
    # Applying the binary operator '>=' (line 79)
    result_ge_36875 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 9), '>=', v_36873, int_36874)
    
    # Testing the type of an if condition (line 79)
    if_condition_36876 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 9), result_ge_36875)
    # Assigning a type to the variable 'if_condition_36876' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 9), 'if_condition_36876', if_condition_36876)
    # SSA begins for if statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to set_threshold(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'DEBUG' (line 80)
    DEBUG_36878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 22), 'DEBUG', False)
    # Getting the type of 'force' (line 80)
    force_36879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 29), 'force', False)
    # Processing the call keyword arguments (line 80)
    kwargs_36880 = {}
    # Getting the type of 'set_threshold' (line 80)
    set_threshold_36877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'set_threshold', False)
    # Calling set_threshold(args, kwargs) (line 80)
    set_threshold_call_result_36881 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), set_threshold_36877, *[DEBUG_36878, force_36879], **kwargs_36880)
    
    # SSA join for if statement (line 79)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 77)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 75)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 73)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to get(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'prev_level' (line 81)
    prev_level_36894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 57), 'prev_level', False)
    int_36895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 69), 'int')
    # Processing the call keyword arguments (line 81)
    kwargs_36896 = {}
    
    # Obtaining an instance of the builtin type 'dict' (line 81)
    dict_36882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 11), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 81)
    # Adding element type (key, value) (line 81)
    # Getting the type of 'FATAL' (line 81)
    FATAL_36883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'FATAL', False)
    int_36884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 18), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 11), dict_36882, (FATAL_36883, int_36884))
    # Adding element type (key, value) (line 81)
    # Getting the type of 'ERROR' (line 81)
    ERROR_36885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 21), 'ERROR', False)
    int_36886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 27), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 11), dict_36882, (ERROR_36885, int_36886))
    # Adding element type (key, value) (line 81)
    # Getting the type of 'WARN' (line 81)
    WARN_36887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 30), 'WARN', False)
    int_36888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 35), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 11), dict_36882, (WARN_36887, int_36888))
    # Adding element type (key, value) (line 81)
    # Getting the type of 'INFO' (line 81)
    INFO_36889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 37), 'INFO', False)
    int_36890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 42), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 11), dict_36882, (INFO_36889, int_36890))
    # Adding element type (key, value) (line 81)
    # Getting the type of 'DEBUG' (line 81)
    DEBUG_36891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 44), 'DEBUG', False)
    int_36892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 50), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 11), dict_36882, (DEBUG_36891, int_36892))
    
    # Obtaining the member 'get' of a type (line 81)
    get_36893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 11), dict_36882, 'get')
    # Calling get(args, kwargs) (line 81)
    get_call_result_36897 = invoke(stypy.reporting.localization.Localization(__file__, 81, 11), get_36893, *[prev_level_36894, int_36895], **kwargs_36896)
    
    # Assigning a type to the variable 'stypy_return_type' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type', get_call_result_36897)
    
    # ################# End of 'set_verbosity(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'set_verbosity' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_36898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36898)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'set_verbosity'
    return stypy_return_type_36898

# Assigning a type to the variable 'set_verbosity' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'set_verbosity', set_verbosity)

# Assigning a Dict to a Name (line 84):

# Obtaining an instance of the builtin type 'dict' (line 84)
dict_36899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 20), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 84)
# Adding element type (key, value) (line 84)
# Getting the type of 'DEBUG' (line 85)
DEBUG_36900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'DEBUG')
# Getting the type of 'cyan_text' (line 85)
cyan_text_36901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 10), 'cyan_text')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 20), dict_36899, (DEBUG_36900, cyan_text_36901))
# Adding element type (key, value) (line 84)
# Getting the type of 'INFO' (line 86)
INFO_36902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'INFO')
# Getting the type of 'default_text' (line 86)
default_text_36903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 9), 'default_text')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 20), dict_36899, (INFO_36902, default_text_36903))
# Adding element type (key, value) (line 84)
# Getting the type of 'WARN' (line 87)
WARN_36904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'WARN')
# Getting the type of 'red_text' (line 87)
red_text_36905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 9), 'red_text')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 20), dict_36899, (WARN_36904, red_text_36905))
# Adding element type (key, value) (line 84)
# Getting the type of 'ERROR' (line 88)
ERROR_36906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'ERROR')
# Getting the type of 'red_text' (line 88)
red_text_36907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 10), 'red_text')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 20), dict_36899, (ERROR_36906, red_text_36907))
# Adding element type (key, value) (line 84)
# Getting the type of 'FATAL' (line 89)
FATAL_36908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'FATAL')
# Getting the type of 'red_text' (line 89)
red_text_36909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 10), 'red_text')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 20), dict_36899, (FATAL_36908, red_text_36909))

# Assigning a type to the variable '_global_color_map' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), '_global_color_map', dict_36899)

# Call to set_verbosity(...): (line 93)
# Processing the call arguments (line 93)
int_36911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 14), 'int')
# Processing the call keyword arguments (line 93)
# Getting the type of 'True' (line 93)
True_36912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 23), 'True', False)
keyword_36913 = True_36912
kwargs_36914 = {'force': keyword_36913}
# Getting the type of 'set_verbosity' (line 93)
set_verbosity_36910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'set_verbosity', False)
# Calling set_verbosity(args, kwargs) (line 93)
set_verbosity_call_result_36915 = invoke(stypy.reporting.localization.Localization(__file__, 93, 0), set_verbosity_36910, *[int_36911], **kwargs_36914)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
