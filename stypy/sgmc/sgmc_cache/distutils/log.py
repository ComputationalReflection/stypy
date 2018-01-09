
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''A simple log mechanism styled after PEP 282.'''
2: 
3: # The class here is styled after PEP 282 so that it could later be
4: # replaced with a standard Python logging implementation.
5: 
6: DEBUG = 1
7: INFO = 2
8: WARN = 3
9: ERROR = 4
10: FATAL = 5
11: 
12: import sys
13: 
14: class Log:
15: 
16:     def __init__(self, threshold=WARN):
17:         self.threshold = threshold
18: 
19:     def _log(self, level, msg, args):
20:         if level not in (DEBUG, INFO, WARN, ERROR, FATAL):
21:             raise ValueError('%s wrong log level' % str(level))
22: 
23:         if level >= self.threshold:
24:             if args:
25:                 msg = msg % args
26:             if level in (WARN, ERROR, FATAL):
27:                 stream = sys.stderr
28:             else:
29:                 stream = sys.stdout
30:             stream.write('%s\n' % msg)
31:             stream.flush()
32: 
33:     def log(self, level, msg, *args):
34:         self._log(level, msg, args)
35: 
36:     def debug(self, msg, *args):
37:         self._log(DEBUG, msg, args)
38: 
39:     def info(self, msg, *args):
40:         self._log(INFO, msg, args)
41: 
42:     def warn(self, msg, *args):
43:         self._log(WARN, msg, args)
44: 
45:     def error(self, msg, *args):
46:         self._log(ERROR, msg, args)
47: 
48:     def fatal(self, msg, *args):
49:         self._log(FATAL, msg, args)
50: 
51: _global_log = Log()
52: log = _global_log.log
53: debug = _global_log.debug
54: info = _global_log.info
55: warn = _global_log.warn
56: error = _global_log.error
57: fatal = _global_log.fatal
58: 
59: def set_threshold(level):
60:     # return the old threshold for use from tests
61:     old = _global_log.threshold
62:     _global_log.threshold = level
63:     return old
64: 
65: def set_verbosity(v):
66:     if v <= 0:
67:         set_threshold(WARN)
68:     elif v == 1:
69:         set_threshold(INFO)
70:     elif v >= 2:
71:         set_threshold(DEBUG)
72: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_2711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'A simple log mechanism styled after PEP 282.')

# Assigning a Num to a Name (line 6):
int_2712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 8), 'int')
# Assigning a type to the variable 'DEBUG' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'DEBUG', int_2712)

# Assigning a Num to a Name (line 7):
int_2713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 7), 'int')
# Assigning a type to the variable 'INFO' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'INFO', int_2713)

# Assigning a Num to a Name (line 8):
int_2714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 7), 'int')
# Assigning a type to the variable 'WARN' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'WARN', int_2714)

# Assigning a Num to a Name (line 9):
int_2715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 8), 'int')
# Assigning a type to the variable 'ERROR' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'ERROR', int_2715)

# Assigning a Num to a Name (line 10):
int_2716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'int')
# Assigning a type to the variable 'FATAL' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'FATAL', int_2716)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import sys' statement (line 12)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'sys', sys, module_type_store)

# Declaration of the 'Log' class

class Log:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'WARN' (line 16)
        WARN_2717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 33), 'WARN')
        defaults = [WARN_2717]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 16, 4, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Log.__init__', ['threshold'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['threshold'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 17):
        # Getting the type of 'threshold' (line 17)
        threshold_2718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 25), 'threshold')
        # Getting the type of 'self' (line 17)
        self_2719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'self')
        # Setting the type of the member 'threshold' of a type (line 17)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), self_2719, 'threshold', threshold_2718)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _log(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_log'
        module_type_store = module_type_store.open_function_context('_log', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
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

        
        
        # Getting the type of 'level' (line 20)
        level_2720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 11), 'level')
        
        # Obtaining an instance of the builtin type 'tuple' (line 20)
        tuple_2721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 20)
        # Adding element type (line 20)
        # Getting the type of 'DEBUG' (line 20)
        DEBUG_2722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 25), 'DEBUG')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), tuple_2721, DEBUG_2722)
        # Adding element type (line 20)
        # Getting the type of 'INFO' (line 20)
        INFO_2723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 32), 'INFO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), tuple_2721, INFO_2723)
        # Adding element type (line 20)
        # Getting the type of 'WARN' (line 20)
        WARN_2724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 38), 'WARN')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), tuple_2721, WARN_2724)
        # Adding element type (line 20)
        # Getting the type of 'ERROR' (line 20)
        ERROR_2725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 44), 'ERROR')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), tuple_2721, ERROR_2725)
        # Adding element type (line 20)
        # Getting the type of 'FATAL' (line 20)
        FATAL_2726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 51), 'FATAL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), tuple_2721, FATAL_2726)
        
        # Applying the binary operator 'notin' (line 20)
        result_contains_2727 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 11), 'notin', level_2720, tuple_2721)
        
        # Testing the type of an if condition (line 20)
        if_condition_2728 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 8), result_contains_2727)
        # Assigning a type to the variable 'if_condition_2728' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'if_condition_2728', if_condition_2728)
        # SSA begins for if statement (line 20)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 21)
        # Processing the call arguments (line 21)
        str_2730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 29), 'str', '%s wrong log level')
        
        # Call to str(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'level' (line 21)
        level_2732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 56), 'level', False)
        # Processing the call keyword arguments (line 21)
        kwargs_2733 = {}
        # Getting the type of 'str' (line 21)
        str_2731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 52), 'str', False)
        # Calling str(args, kwargs) (line 21)
        str_call_result_2734 = invoke(stypy.reporting.localization.Localization(__file__, 21, 52), str_2731, *[level_2732], **kwargs_2733)
        
        # Applying the binary operator '%' (line 21)
        result_mod_2735 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 29), '%', str_2730, str_call_result_2734)
        
        # Processing the call keyword arguments (line 21)
        kwargs_2736 = {}
        # Getting the type of 'ValueError' (line 21)
        ValueError_2729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 21)
        ValueError_call_result_2737 = invoke(stypy.reporting.localization.Localization(__file__, 21, 18), ValueError_2729, *[result_mod_2735], **kwargs_2736)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 21, 12), ValueError_call_result_2737, 'raise parameter', BaseException)
        # SSA join for if statement (line 20)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'level' (line 23)
        level_2738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'level')
        # Getting the type of 'self' (line 23)
        self_2739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 20), 'self')
        # Obtaining the member 'threshold' of a type (line 23)
        threshold_2740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 20), self_2739, 'threshold')
        # Applying the binary operator '>=' (line 23)
        result_ge_2741 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 11), '>=', level_2738, threshold_2740)
        
        # Testing the type of an if condition (line 23)
        if_condition_2742 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 23, 8), result_ge_2741)
        # Assigning a type to the variable 'if_condition_2742' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'if_condition_2742', if_condition_2742)
        # SSA begins for if statement (line 23)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'args' (line 24)
        args_2743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 15), 'args')
        # Testing the type of an if condition (line 24)
        if_condition_2744 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 12), args_2743)
        # Assigning a type to the variable 'if_condition_2744' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'if_condition_2744', if_condition_2744)
        # SSA begins for if statement (line 24)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 25):
        # Getting the type of 'msg' (line 25)
        msg_2745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 22), 'msg')
        # Getting the type of 'args' (line 25)
        args_2746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 28), 'args')
        # Applying the binary operator '%' (line 25)
        result_mod_2747 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 22), '%', msg_2745, args_2746)
        
        # Assigning a type to the variable 'msg' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'msg', result_mod_2747)
        # SSA join for if statement (line 24)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'level' (line 26)
        level_2748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'level')
        
        # Obtaining an instance of the builtin type 'tuple' (line 26)
        tuple_2749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 26)
        # Adding element type (line 26)
        # Getting the type of 'WARN' (line 26)
        WARN_2750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 25), 'WARN')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 25), tuple_2749, WARN_2750)
        # Adding element type (line 26)
        # Getting the type of 'ERROR' (line 26)
        ERROR_2751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 31), 'ERROR')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 25), tuple_2749, ERROR_2751)
        # Adding element type (line 26)
        # Getting the type of 'FATAL' (line 26)
        FATAL_2752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 38), 'FATAL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 25), tuple_2749, FATAL_2752)
        
        # Applying the binary operator 'in' (line 26)
        result_contains_2753 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 15), 'in', level_2748, tuple_2749)
        
        # Testing the type of an if condition (line 26)
        if_condition_2754 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 12), result_contains_2753)
        # Assigning a type to the variable 'if_condition_2754' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'if_condition_2754', if_condition_2754)
        # SSA begins for if statement (line 26)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 27):
        # Getting the type of 'sys' (line 27)
        sys_2755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 25), 'sys')
        # Obtaining the member 'stderr' of a type (line 27)
        stderr_2756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 25), sys_2755, 'stderr')
        # Assigning a type to the variable 'stream' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 16), 'stream', stderr_2756)
        # SSA branch for the else part of an if statement (line 26)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 29):
        # Getting the type of 'sys' (line 29)
        sys_2757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 25), 'sys')
        # Obtaining the member 'stdout' of a type (line 29)
        stdout_2758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 25), sys_2757, 'stdout')
        # Assigning a type to the variable 'stream' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 16), 'stream', stdout_2758)
        # SSA join for if statement (line 26)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 30)
        # Processing the call arguments (line 30)
        str_2761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 25), 'str', '%s\n')
        # Getting the type of 'msg' (line 30)
        msg_2762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 34), 'msg', False)
        # Applying the binary operator '%' (line 30)
        result_mod_2763 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 25), '%', str_2761, msg_2762)
        
        # Processing the call keyword arguments (line 30)
        kwargs_2764 = {}
        # Getting the type of 'stream' (line 30)
        stream_2759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'stream', False)
        # Obtaining the member 'write' of a type (line 30)
        write_2760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 12), stream_2759, 'write')
        # Calling write(args, kwargs) (line 30)
        write_call_result_2765 = invoke(stypy.reporting.localization.Localization(__file__, 30, 12), write_2760, *[result_mod_2763], **kwargs_2764)
        
        
        # Call to flush(...): (line 31)
        # Processing the call keyword arguments (line 31)
        kwargs_2768 = {}
        # Getting the type of 'stream' (line 31)
        stream_2766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'stream', False)
        # Obtaining the member 'flush' of a type (line 31)
        flush_2767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), stream_2766, 'flush')
        # Calling flush(args, kwargs) (line 31)
        flush_call_result_2769 = invoke(stypy.reporting.localization.Localization(__file__, 31, 12), flush_2767, *[], **kwargs_2768)
        
        # SSA join for if statement (line 23)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_log(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_log' in the type store
        # Getting the type of 'stypy_return_type' (line 19)
        stypy_return_type_2770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2770)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_log'
        return stypy_return_type_2770


    @norecursion
    def log(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'log'
        module_type_store = module_type_store.open_function_context('log', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Log.log.__dict__.__setitem__('stypy_localization', localization)
        Log.log.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Log.log.__dict__.__setitem__('stypy_type_store', module_type_store)
        Log.log.__dict__.__setitem__('stypy_function_name', 'Log.log')
        Log.log.__dict__.__setitem__('stypy_param_names_list', ['level', 'msg'])
        Log.log.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        Log.log.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Log.log.__dict__.__setitem__('stypy_call_defaults', defaults)
        Log.log.__dict__.__setitem__('stypy_call_varargs', varargs)
        Log.log.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Log.log.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Log.log', ['level', 'msg'], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'log', localization, ['level', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'log(...)' code ##################

        
        # Call to _log(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'level' (line 34)
        level_2773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 18), 'level', False)
        # Getting the type of 'msg' (line 34)
        msg_2774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'msg', False)
        # Getting the type of 'args' (line 34)
        args_2775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 30), 'args', False)
        # Processing the call keyword arguments (line 34)
        kwargs_2776 = {}
        # Getting the type of 'self' (line 34)
        self_2771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self', False)
        # Obtaining the member '_log' of a type (line 34)
        _log_2772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_2771, '_log')
        # Calling _log(args, kwargs) (line 34)
        _log_call_result_2777 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), _log_2772, *[level_2773, msg_2774, args_2775], **kwargs_2776)
        
        
        # ################# End of 'log(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'log' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_2778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2778)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'log'
        return stypy_return_type_2778


    @norecursion
    def debug(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'debug'
        module_type_store = module_type_store.open_function_context('debug', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Log.debug.__dict__.__setitem__('stypy_localization', localization)
        Log.debug.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Log.debug.__dict__.__setitem__('stypy_type_store', module_type_store)
        Log.debug.__dict__.__setitem__('stypy_function_name', 'Log.debug')
        Log.debug.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        Log.debug.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        Log.debug.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Log.debug.__dict__.__setitem__('stypy_call_defaults', defaults)
        Log.debug.__dict__.__setitem__('stypy_call_varargs', varargs)
        Log.debug.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Log.debug.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Log.debug', ['msg'], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'debug', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'debug(...)' code ##################

        
        # Call to _log(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'DEBUG' (line 37)
        DEBUG_2781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 18), 'DEBUG', False)
        # Getting the type of 'msg' (line 37)
        msg_2782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 25), 'msg', False)
        # Getting the type of 'args' (line 37)
        args_2783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 30), 'args', False)
        # Processing the call keyword arguments (line 37)
        kwargs_2784 = {}
        # Getting the type of 'self' (line 37)
        self_2779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self', False)
        # Obtaining the member '_log' of a type (line 37)
        _log_2780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_2779, '_log')
        # Calling _log(args, kwargs) (line 37)
        _log_call_result_2785 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), _log_2780, *[DEBUG_2781, msg_2782, args_2783], **kwargs_2784)
        
        
        # ################# End of 'debug(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'debug' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_2786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2786)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'debug'
        return stypy_return_type_2786


    @norecursion
    def info(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'info'
        module_type_store = module_type_store.open_function_context('info', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Log.info.__dict__.__setitem__('stypy_localization', localization)
        Log.info.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Log.info.__dict__.__setitem__('stypy_type_store', module_type_store)
        Log.info.__dict__.__setitem__('stypy_function_name', 'Log.info')
        Log.info.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        Log.info.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        Log.info.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Log.info.__dict__.__setitem__('stypy_call_defaults', defaults)
        Log.info.__dict__.__setitem__('stypy_call_varargs', varargs)
        Log.info.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Log.info.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Log.info', ['msg'], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'info', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'info(...)' code ##################

        
        # Call to _log(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'INFO' (line 40)
        INFO_2789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 18), 'INFO', False)
        # Getting the type of 'msg' (line 40)
        msg_2790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'msg', False)
        # Getting the type of 'args' (line 40)
        args_2791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 29), 'args', False)
        # Processing the call keyword arguments (line 40)
        kwargs_2792 = {}
        # Getting the type of 'self' (line 40)
        self_2787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self', False)
        # Obtaining the member '_log' of a type (line 40)
        _log_2788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_2787, '_log')
        # Calling _log(args, kwargs) (line 40)
        _log_call_result_2793 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), _log_2788, *[INFO_2789, msg_2790, args_2791], **kwargs_2792)
        
        
        # ################# End of 'info(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'info' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_2794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2794)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'info'
        return stypy_return_type_2794


    @norecursion
    def warn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'warn'
        module_type_store = module_type_store.open_function_context('warn', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Log.warn.__dict__.__setitem__('stypy_localization', localization)
        Log.warn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Log.warn.__dict__.__setitem__('stypy_type_store', module_type_store)
        Log.warn.__dict__.__setitem__('stypy_function_name', 'Log.warn')
        Log.warn.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        Log.warn.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        Log.warn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Log.warn.__dict__.__setitem__('stypy_call_defaults', defaults)
        Log.warn.__dict__.__setitem__('stypy_call_varargs', varargs)
        Log.warn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Log.warn.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Log.warn', ['msg'], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'warn', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'warn(...)' code ##################

        
        # Call to _log(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'WARN' (line 43)
        WARN_2797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 18), 'WARN', False)
        # Getting the type of 'msg' (line 43)
        msg_2798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'msg', False)
        # Getting the type of 'args' (line 43)
        args_2799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 29), 'args', False)
        # Processing the call keyword arguments (line 43)
        kwargs_2800 = {}
        # Getting the type of 'self' (line 43)
        self_2795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self', False)
        # Obtaining the member '_log' of a type (line 43)
        _log_2796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_2795, '_log')
        # Calling _log(args, kwargs) (line 43)
        _log_call_result_2801 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), _log_2796, *[WARN_2797, msg_2798, args_2799], **kwargs_2800)
        
        
        # ################# End of 'warn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'warn' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_2802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2802)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'warn'
        return stypy_return_type_2802


    @norecursion
    def error(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'error'
        module_type_store = module_type_store.open_function_context('error', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Log.error.__dict__.__setitem__('stypy_localization', localization)
        Log.error.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Log.error.__dict__.__setitem__('stypy_type_store', module_type_store)
        Log.error.__dict__.__setitem__('stypy_function_name', 'Log.error')
        Log.error.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        Log.error.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        Log.error.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Log.error.__dict__.__setitem__('stypy_call_defaults', defaults)
        Log.error.__dict__.__setitem__('stypy_call_varargs', varargs)
        Log.error.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Log.error.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Log.error', ['msg'], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'error', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'error(...)' code ##################

        
        # Call to _log(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'ERROR' (line 46)
        ERROR_2805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 18), 'ERROR', False)
        # Getting the type of 'msg' (line 46)
        msg_2806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 25), 'msg', False)
        # Getting the type of 'args' (line 46)
        args_2807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 30), 'args', False)
        # Processing the call keyword arguments (line 46)
        kwargs_2808 = {}
        # Getting the type of 'self' (line 46)
        self_2803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self', False)
        # Obtaining the member '_log' of a type (line 46)
        _log_2804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_2803, '_log')
        # Calling _log(args, kwargs) (line 46)
        _log_call_result_2809 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), _log_2804, *[ERROR_2805, msg_2806, args_2807], **kwargs_2808)
        
        
        # ################# End of 'error(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'error' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_2810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2810)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'error'
        return stypy_return_type_2810


    @norecursion
    def fatal(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fatal'
        module_type_store = module_type_store.open_function_context('fatal', 48, 4, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Log.fatal.__dict__.__setitem__('stypy_localization', localization)
        Log.fatal.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Log.fatal.__dict__.__setitem__('stypy_type_store', module_type_store)
        Log.fatal.__dict__.__setitem__('stypy_function_name', 'Log.fatal')
        Log.fatal.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        Log.fatal.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        Log.fatal.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Log.fatal.__dict__.__setitem__('stypy_call_defaults', defaults)
        Log.fatal.__dict__.__setitem__('stypy_call_varargs', varargs)
        Log.fatal.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Log.fatal.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Log.fatal', ['msg'], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fatal', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fatal(...)' code ##################

        
        # Call to _log(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'FATAL' (line 49)
        FATAL_2813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 18), 'FATAL', False)
        # Getting the type of 'msg' (line 49)
        msg_2814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'msg', False)
        # Getting the type of 'args' (line 49)
        args_2815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'args', False)
        # Processing the call keyword arguments (line 49)
        kwargs_2816 = {}
        # Getting the type of 'self' (line 49)
        self_2811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'self', False)
        # Obtaining the member '_log' of a type (line 49)
        _log_2812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), self_2811, '_log')
        # Calling _log(args, kwargs) (line 49)
        _log_call_result_2817 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), _log_2812, *[FATAL_2813, msg_2814, args_2815], **kwargs_2816)
        
        
        # ################# End of 'fatal(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fatal' in the type store
        # Getting the type of 'stypy_return_type' (line 48)
        stypy_return_type_2818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2818)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fatal'
        return stypy_return_type_2818


# Assigning a type to the variable 'Log' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'Log', Log)

# Assigning a Call to a Name (line 51):

# Call to Log(...): (line 51)
# Processing the call keyword arguments (line 51)
kwargs_2820 = {}
# Getting the type of 'Log' (line 51)
Log_2819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 14), 'Log', False)
# Calling Log(args, kwargs) (line 51)
Log_call_result_2821 = invoke(stypy.reporting.localization.Localization(__file__, 51, 14), Log_2819, *[], **kwargs_2820)

# Assigning a type to the variable '_global_log' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), '_global_log', Log_call_result_2821)

# Assigning a Attribute to a Name (line 52):
# Getting the type of '_global_log' (line 52)
_global_log_2822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 6), '_global_log')
# Obtaining the member 'log' of a type (line 52)
log_2823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 6), _global_log_2822, 'log')
# Assigning a type to the variable 'log' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'log', log_2823)

# Assigning a Attribute to a Name (line 53):
# Getting the type of '_global_log' (line 53)
_global_log_2824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), '_global_log')
# Obtaining the member 'debug' of a type (line 53)
debug_2825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), _global_log_2824, 'debug')
# Assigning a type to the variable 'debug' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'debug', debug_2825)

# Assigning a Attribute to a Name (line 54):
# Getting the type of '_global_log' (line 54)
_global_log_2826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 7), '_global_log')
# Obtaining the member 'info' of a type (line 54)
info_2827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 7), _global_log_2826, 'info')
# Assigning a type to the variable 'info' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'info', info_2827)

# Assigning a Attribute to a Name (line 55):
# Getting the type of '_global_log' (line 55)
_global_log_2828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 7), '_global_log')
# Obtaining the member 'warn' of a type (line 55)
warn_2829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 7), _global_log_2828, 'warn')
# Assigning a type to the variable 'warn' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'warn', warn_2829)

# Assigning a Attribute to a Name (line 56):
# Getting the type of '_global_log' (line 56)
_global_log_2830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), '_global_log')
# Obtaining the member 'error' of a type (line 56)
error_2831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), _global_log_2830, 'error')
# Assigning a type to the variable 'error' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'error', error_2831)

# Assigning a Attribute to a Name (line 57):
# Getting the type of '_global_log' (line 57)
_global_log_2832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), '_global_log')
# Obtaining the member 'fatal' of a type (line 57)
fatal_2833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), _global_log_2832, 'fatal')
# Assigning a type to the variable 'fatal' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'fatal', fatal_2833)

@norecursion
def set_threshold(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'set_threshold'
    module_type_store = module_type_store.open_function_context('set_threshold', 59, 0, False)
    
    # Passed parameters checking function
    set_threshold.stypy_localization = localization
    set_threshold.stypy_type_of_self = None
    set_threshold.stypy_type_store = module_type_store
    set_threshold.stypy_function_name = 'set_threshold'
    set_threshold.stypy_param_names_list = ['level']
    set_threshold.stypy_varargs_param_name = None
    set_threshold.stypy_kwargs_param_name = None
    set_threshold.stypy_call_defaults = defaults
    set_threshold.stypy_call_varargs = varargs
    set_threshold.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'set_threshold', ['level'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'set_threshold', localization, ['level'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'set_threshold(...)' code ##################

    
    # Assigning a Attribute to a Name (line 61):
    # Getting the type of '_global_log' (line 61)
    _global_log_2834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 10), '_global_log')
    # Obtaining the member 'threshold' of a type (line 61)
    threshold_2835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 10), _global_log_2834, 'threshold')
    # Assigning a type to the variable 'old' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'old', threshold_2835)
    
    # Assigning a Name to a Attribute (line 62):
    # Getting the type of 'level' (line 62)
    level_2836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 28), 'level')
    # Getting the type of '_global_log' (line 62)
    _global_log_2837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), '_global_log')
    # Setting the type of the member 'threshold' of a type (line 62)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 4), _global_log_2837, 'threshold', level_2836)
    # Getting the type of 'old' (line 63)
    old_2838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 11), 'old')
    # Assigning a type to the variable 'stypy_return_type' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type', old_2838)
    
    # ################# End of 'set_threshold(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'set_threshold' in the type store
    # Getting the type of 'stypy_return_type' (line 59)
    stypy_return_type_2839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2839)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'set_threshold'
    return stypy_return_type_2839

# Assigning a type to the variable 'set_threshold' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'set_threshold', set_threshold)

@norecursion
def set_verbosity(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'set_verbosity'
    module_type_store = module_type_store.open_function_context('set_verbosity', 65, 0, False)
    
    # Passed parameters checking function
    set_verbosity.stypy_localization = localization
    set_verbosity.stypy_type_of_self = None
    set_verbosity.stypy_type_store = module_type_store
    set_verbosity.stypy_function_name = 'set_verbosity'
    set_verbosity.stypy_param_names_list = ['v']
    set_verbosity.stypy_varargs_param_name = None
    set_verbosity.stypy_kwargs_param_name = None
    set_verbosity.stypy_call_defaults = defaults
    set_verbosity.stypy_call_varargs = varargs
    set_verbosity.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'set_verbosity', ['v'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'set_verbosity', localization, ['v'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'set_verbosity(...)' code ##################

    
    
    # Getting the type of 'v' (line 66)
    v_2840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 7), 'v')
    int_2841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 12), 'int')
    # Applying the binary operator '<=' (line 66)
    result_le_2842 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 7), '<=', v_2840, int_2841)
    
    # Testing the type of an if condition (line 66)
    if_condition_2843 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 4), result_le_2842)
    # Assigning a type to the variable 'if_condition_2843' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'if_condition_2843', if_condition_2843)
    # SSA begins for if statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to set_threshold(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'WARN' (line 67)
    WARN_2845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 22), 'WARN', False)
    # Processing the call keyword arguments (line 67)
    kwargs_2846 = {}
    # Getting the type of 'set_threshold' (line 67)
    set_threshold_2844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'set_threshold', False)
    # Calling set_threshold(args, kwargs) (line 67)
    set_threshold_call_result_2847 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), set_threshold_2844, *[WARN_2845], **kwargs_2846)
    
    # SSA branch for the else part of an if statement (line 66)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'v' (line 68)
    v_2848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 9), 'v')
    int_2849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 14), 'int')
    # Applying the binary operator '==' (line 68)
    result_eq_2850 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 9), '==', v_2848, int_2849)
    
    # Testing the type of an if condition (line 68)
    if_condition_2851 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 9), result_eq_2850)
    # Assigning a type to the variable 'if_condition_2851' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 9), 'if_condition_2851', if_condition_2851)
    # SSA begins for if statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to set_threshold(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'INFO' (line 69)
    INFO_2853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 22), 'INFO', False)
    # Processing the call keyword arguments (line 69)
    kwargs_2854 = {}
    # Getting the type of 'set_threshold' (line 69)
    set_threshold_2852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'set_threshold', False)
    # Calling set_threshold(args, kwargs) (line 69)
    set_threshold_call_result_2855 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), set_threshold_2852, *[INFO_2853], **kwargs_2854)
    
    # SSA branch for the else part of an if statement (line 68)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'v' (line 70)
    v_2856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 9), 'v')
    int_2857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 14), 'int')
    # Applying the binary operator '>=' (line 70)
    result_ge_2858 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 9), '>=', v_2856, int_2857)
    
    # Testing the type of an if condition (line 70)
    if_condition_2859 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 9), result_ge_2858)
    # Assigning a type to the variable 'if_condition_2859' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 9), 'if_condition_2859', if_condition_2859)
    # SSA begins for if statement (line 70)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to set_threshold(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'DEBUG' (line 71)
    DEBUG_2861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'DEBUG', False)
    # Processing the call keyword arguments (line 71)
    kwargs_2862 = {}
    # Getting the type of 'set_threshold' (line 71)
    set_threshold_2860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'set_threshold', False)
    # Calling set_threshold(args, kwargs) (line 71)
    set_threshold_call_result_2863 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), set_threshold_2860, *[DEBUG_2861], **kwargs_2862)
    
    # SSA join for if statement (line 70)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 68)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 66)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'set_verbosity(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'set_verbosity' in the type store
    # Getting the type of 'stypy_return_type' (line 65)
    stypy_return_type_2864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2864)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'set_verbosity'
    return stypy_return_type_2864

# Assigning a type to the variable 'set_verbosity' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'set_verbosity', set_verbosity)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
