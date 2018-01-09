
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: from numpy.distutils.cpuinfo import cpu
4: from numpy.distutils.fcompiler import FCompiler
5: 
6: compilers = ['MIPSFCompiler']
7: 
8: class MIPSFCompiler(FCompiler):
9: 
10:     compiler_type = 'mips'
11:     description = 'MIPSpro Fortran Compiler'
12:     version_pattern =  r'MIPSpro Compilers: Version (?P<version>[^\s*,]*)'
13: 
14:     executables = {
15:         'version_cmd'  : ["<F90>", "-version"],
16:         'compiler_f77' : ["f77", "-f77"],
17:         'compiler_fix' : ["f90", "-fixedform"],
18:         'compiler_f90' : ["f90"],
19:         'linker_so'    : ["f90", "-shared"],
20:         'archiver'     : ["ar", "-cr"],
21:         'ranlib'       : None
22:         }
23:     module_dir_switch = None #XXX: fix me
24:     module_include_switch = None #XXX: fix me
25:     pic_flags = ['-KPIC']
26: 
27:     def get_flags(self):
28:         return self.pic_flags + ['-n32']
29:     def get_flags_opt(self):
30:         return ['-O3']
31:     def get_flags_arch(self):
32:         opt = []
33:         for a in '19 20 21 22_4k 22_5k 24 25 26 27 28 30 32_5k 32_10k'.split():
34:             if getattr(cpu, 'is_IP%s'%a)():
35:                 opt.append('-TARG:platform=IP%s' % a)
36:                 break
37:         return opt
38:     def get_flags_arch_f77(self):
39:         r = None
40:         if cpu.is_r10000(): r = 10000
41:         elif cpu.is_r12000(): r = 12000
42:         elif cpu.is_r8000(): r = 8000
43:         elif cpu.is_r5000(): r = 5000
44:         elif cpu.is_r4000(): r = 4000
45:         if r is not None:
46:             return ['r%s' % (r)]
47:         return []
48:     def get_flags_arch_f90(self):
49:         r = self.get_flags_arch_f77()
50:         if r:
51:             r[0] = '-' + r[0]
52:         return r
53: 
54: if __name__ == '__main__':
55:     from numpy.distutils.fcompiler import new_fcompiler
56:     compiler = new_fcompiler(compiler='mips')
57:     compiler.customize()
58:     print(compiler.get_version())
59: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from numpy.distutils.cpuinfo import cpu' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_62726 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.distutils.cpuinfo')

if (type(import_62726) is not StypyTypeError):

    if (import_62726 != 'pyd_module'):
        __import__(import_62726)
        sys_modules_62727 = sys.modules[import_62726]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.distutils.cpuinfo', sys_modules_62727.module_type_store, module_type_store, ['cpu'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_62727, sys_modules_62727.module_type_store, module_type_store)
    else:
        from numpy.distutils.cpuinfo import cpu

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.distutils.cpuinfo', None, module_type_store, ['cpu'], [cpu])

else:
    # Assigning a type to the variable 'numpy.distutils.cpuinfo' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.distutils.cpuinfo', import_62726)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.distutils.fcompiler import FCompiler' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_62728 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.fcompiler')

if (type(import_62728) is not StypyTypeError):

    if (import_62728 != 'pyd_module'):
        __import__(import_62728)
        sys_modules_62729 = sys.modules[import_62728]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.fcompiler', sys_modules_62729.module_type_store, module_type_store, ['FCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_62729, sys_modules_62729.module_type_store, module_type_store)
    else:
        from numpy.distutils.fcompiler import FCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.fcompiler', None, module_type_store, ['FCompiler'], [FCompiler])

else:
    # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.fcompiler', import_62728)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')


# Assigning a List to a Name (line 6):

# Obtaining an instance of the builtin type 'list' (line 6)
list_62730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
str_62731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 13), 'str', 'MIPSFCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 12), list_62730, str_62731)

# Assigning a type to the variable 'compilers' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'compilers', list_62730)
# Declaration of the 'MIPSFCompiler' class
# Getting the type of 'FCompiler' (line 8)
FCompiler_62732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 20), 'FCompiler')

class MIPSFCompiler(FCompiler_62732, ):

    @norecursion
    def get_flags(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags'
        module_type_store = module_type_store.open_function_context('get_flags', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MIPSFCompiler.get_flags.__dict__.__setitem__('stypy_localization', localization)
        MIPSFCompiler.get_flags.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MIPSFCompiler.get_flags.__dict__.__setitem__('stypy_type_store', module_type_store)
        MIPSFCompiler.get_flags.__dict__.__setitem__('stypy_function_name', 'MIPSFCompiler.get_flags')
        MIPSFCompiler.get_flags.__dict__.__setitem__('stypy_param_names_list', [])
        MIPSFCompiler.get_flags.__dict__.__setitem__('stypy_varargs_param_name', None)
        MIPSFCompiler.get_flags.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MIPSFCompiler.get_flags.__dict__.__setitem__('stypy_call_defaults', defaults)
        MIPSFCompiler.get_flags.__dict__.__setitem__('stypy_call_varargs', varargs)
        MIPSFCompiler.get_flags.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MIPSFCompiler.get_flags.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MIPSFCompiler.get_flags', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags(...)' code ##################

        # Getting the type of 'self' (line 28)
        self_62733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'self')
        # Obtaining the member 'pic_flags' of a type (line 28)
        pic_flags_62734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 15), self_62733, 'pic_flags')
        
        # Obtaining an instance of the builtin type 'list' (line 28)
        list_62735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 28)
        # Adding element type (line 28)
        str_62736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 33), 'str', '-n32')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 32), list_62735, str_62736)
        
        # Applying the binary operator '+' (line 28)
        result_add_62737 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 15), '+', pic_flags_62734, list_62735)
        
        # Assigning a type to the variable 'stypy_return_type' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stypy_return_type', result_add_62737)
        
        # ################# End of 'get_flags(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_62738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62738)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags'
        return stypy_return_type_62738


    @norecursion
    def get_flags_opt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_opt'
        module_type_store = module_type_store.open_function_context('get_flags_opt', 29, 4, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MIPSFCompiler.get_flags_opt.__dict__.__setitem__('stypy_localization', localization)
        MIPSFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MIPSFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_store', module_type_store)
        MIPSFCompiler.get_flags_opt.__dict__.__setitem__('stypy_function_name', 'MIPSFCompiler.get_flags_opt')
        MIPSFCompiler.get_flags_opt.__dict__.__setitem__('stypy_param_names_list', [])
        MIPSFCompiler.get_flags_opt.__dict__.__setitem__('stypy_varargs_param_name', None)
        MIPSFCompiler.get_flags_opt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MIPSFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_defaults', defaults)
        MIPSFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_varargs', varargs)
        MIPSFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MIPSFCompiler.get_flags_opt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MIPSFCompiler.get_flags_opt', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_opt', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_opt(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 30)
        list_62739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 30)
        # Adding element type (line 30)
        str_62740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 16), 'str', '-O3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 15), list_62739, str_62740)
        
        # Assigning a type to the variable 'stypy_return_type' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'stypy_return_type', list_62739)
        
        # ################# End of 'get_flags_opt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_opt' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_62741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62741)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_opt'
        return stypy_return_type_62741


    @norecursion
    def get_flags_arch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_arch'
        module_type_store = module_type_store.open_function_context('get_flags_arch', 31, 4, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MIPSFCompiler.get_flags_arch.__dict__.__setitem__('stypy_localization', localization)
        MIPSFCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MIPSFCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_store', module_type_store)
        MIPSFCompiler.get_flags_arch.__dict__.__setitem__('stypy_function_name', 'MIPSFCompiler.get_flags_arch')
        MIPSFCompiler.get_flags_arch.__dict__.__setitem__('stypy_param_names_list', [])
        MIPSFCompiler.get_flags_arch.__dict__.__setitem__('stypy_varargs_param_name', None)
        MIPSFCompiler.get_flags_arch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MIPSFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_defaults', defaults)
        MIPSFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_varargs', varargs)
        MIPSFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MIPSFCompiler.get_flags_arch.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MIPSFCompiler.get_flags_arch', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_arch', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_arch(...)' code ##################

        
        # Assigning a List to a Name (line 32):
        
        # Obtaining an instance of the builtin type 'list' (line 32)
        list_62742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 32)
        
        # Assigning a type to the variable 'opt' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'opt', list_62742)
        
        
        # Call to split(...): (line 33)
        # Processing the call keyword arguments (line 33)
        kwargs_62745 = {}
        str_62743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 17), 'str', '19 20 21 22_4k 22_5k 24 25 26 27 28 30 32_5k 32_10k')
        # Obtaining the member 'split' of a type (line 33)
        split_62744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 17), str_62743, 'split')
        # Calling split(args, kwargs) (line 33)
        split_call_result_62746 = invoke(stypy.reporting.localization.Localization(__file__, 33, 17), split_62744, *[], **kwargs_62745)
        
        # Testing the type of a for loop iterable (line 33)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 33, 8), split_call_result_62746)
        # Getting the type of the for loop variable (line 33)
        for_loop_var_62747 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 33, 8), split_call_result_62746)
        # Assigning a type to the variable 'a' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'a', for_loop_var_62747)
        # SSA begins for a for statement (line 33)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to (...): (line 34)
        # Processing the call keyword arguments (line 34)
        kwargs_62755 = {}
        
        # Call to getattr(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'cpu' (line 34)
        cpu_62749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 23), 'cpu', False)
        str_62750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 28), 'str', 'is_IP%s')
        # Getting the type of 'a' (line 34)
        a_62751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 38), 'a', False)
        # Applying the binary operator '%' (line 34)
        result_mod_62752 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 28), '%', str_62750, a_62751)
        
        # Processing the call keyword arguments (line 34)
        kwargs_62753 = {}
        # Getting the type of 'getattr' (line 34)
        getattr_62748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 34)
        getattr_call_result_62754 = invoke(stypy.reporting.localization.Localization(__file__, 34, 15), getattr_62748, *[cpu_62749, result_mod_62752], **kwargs_62753)
        
        # Calling (args, kwargs) (line 34)
        _call_result_62756 = invoke(stypy.reporting.localization.Localization(__file__, 34, 15), getattr_call_result_62754, *[], **kwargs_62755)
        
        # Testing the type of an if condition (line 34)
        if_condition_62757 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 12), _call_result_62756)
        # Assigning a type to the variable 'if_condition_62757' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'if_condition_62757', if_condition_62757)
        # SSA begins for if statement (line 34)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 35)
        # Processing the call arguments (line 35)
        str_62760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 27), 'str', '-TARG:platform=IP%s')
        # Getting the type of 'a' (line 35)
        a_62761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 51), 'a', False)
        # Applying the binary operator '%' (line 35)
        result_mod_62762 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 27), '%', str_62760, a_62761)
        
        # Processing the call keyword arguments (line 35)
        kwargs_62763 = {}
        # Getting the type of 'opt' (line 35)
        opt_62758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'opt', False)
        # Obtaining the member 'append' of a type (line 35)
        append_62759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 16), opt_62758, 'append')
        # Calling append(args, kwargs) (line 35)
        append_call_result_62764 = invoke(stypy.reporting.localization.Localization(__file__, 35, 16), append_62759, *[result_mod_62762], **kwargs_62763)
        
        # SSA join for if statement (line 34)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'opt' (line 37)
        opt_62765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'stypy_return_type', opt_62765)
        
        # ################# End of 'get_flags_arch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_arch' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_62766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62766)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_arch'
        return stypy_return_type_62766


    @norecursion
    def get_flags_arch_f77(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_arch_f77'
        module_type_store = module_type_store.open_function_context('get_flags_arch_f77', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MIPSFCompiler.get_flags_arch_f77.__dict__.__setitem__('stypy_localization', localization)
        MIPSFCompiler.get_flags_arch_f77.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MIPSFCompiler.get_flags_arch_f77.__dict__.__setitem__('stypy_type_store', module_type_store)
        MIPSFCompiler.get_flags_arch_f77.__dict__.__setitem__('stypy_function_name', 'MIPSFCompiler.get_flags_arch_f77')
        MIPSFCompiler.get_flags_arch_f77.__dict__.__setitem__('stypy_param_names_list', [])
        MIPSFCompiler.get_flags_arch_f77.__dict__.__setitem__('stypy_varargs_param_name', None)
        MIPSFCompiler.get_flags_arch_f77.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MIPSFCompiler.get_flags_arch_f77.__dict__.__setitem__('stypy_call_defaults', defaults)
        MIPSFCompiler.get_flags_arch_f77.__dict__.__setitem__('stypy_call_varargs', varargs)
        MIPSFCompiler.get_flags_arch_f77.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MIPSFCompiler.get_flags_arch_f77.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MIPSFCompiler.get_flags_arch_f77', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_arch_f77', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_arch_f77(...)' code ##################

        
        # Assigning a Name to a Name (line 39):
        # Getting the type of 'None' (line 39)
        None_62767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'None')
        # Assigning a type to the variable 'r' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'r', None_62767)
        
        
        # Call to is_r10000(...): (line 40)
        # Processing the call keyword arguments (line 40)
        kwargs_62770 = {}
        # Getting the type of 'cpu' (line 40)
        cpu_62768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'cpu', False)
        # Obtaining the member 'is_r10000' of a type (line 40)
        is_r10000_62769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 11), cpu_62768, 'is_r10000')
        # Calling is_r10000(args, kwargs) (line 40)
        is_r10000_call_result_62771 = invoke(stypy.reporting.localization.Localization(__file__, 40, 11), is_r10000_62769, *[], **kwargs_62770)
        
        # Testing the type of an if condition (line 40)
        if_condition_62772 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 8), is_r10000_call_result_62771)
        # Assigning a type to the variable 'if_condition_62772' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'if_condition_62772', if_condition_62772)
        # SSA begins for if statement (line 40)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 40):
        int_62773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 32), 'int')
        # Assigning a type to the variable 'r' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 28), 'r', int_62773)
        # SSA branch for the else part of an if statement (line 40)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to is_r12000(...): (line 41)
        # Processing the call keyword arguments (line 41)
        kwargs_62776 = {}
        # Getting the type of 'cpu' (line 41)
        cpu_62774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 13), 'cpu', False)
        # Obtaining the member 'is_r12000' of a type (line 41)
        is_r12000_62775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 13), cpu_62774, 'is_r12000')
        # Calling is_r12000(args, kwargs) (line 41)
        is_r12000_call_result_62777 = invoke(stypy.reporting.localization.Localization(__file__, 41, 13), is_r12000_62775, *[], **kwargs_62776)
        
        # Testing the type of an if condition (line 41)
        if_condition_62778 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 13), is_r12000_call_result_62777)
        # Assigning a type to the variable 'if_condition_62778' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 13), 'if_condition_62778', if_condition_62778)
        # SSA begins for if statement (line 41)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 41):
        int_62779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 34), 'int')
        # Assigning a type to the variable 'r' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 30), 'r', int_62779)
        # SSA branch for the else part of an if statement (line 41)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to is_r8000(...): (line 42)
        # Processing the call keyword arguments (line 42)
        kwargs_62782 = {}
        # Getting the type of 'cpu' (line 42)
        cpu_62780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'cpu', False)
        # Obtaining the member 'is_r8000' of a type (line 42)
        is_r8000_62781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 13), cpu_62780, 'is_r8000')
        # Calling is_r8000(args, kwargs) (line 42)
        is_r8000_call_result_62783 = invoke(stypy.reporting.localization.Localization(__file__, 42, 13), is_r8000_62781, *[], **kwargs_62782)
        
        # Testing the type of an if condition (line 42)
        if_condition_62784 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 13), is_r8000_call_result_62783)
        # Assigning a type to the variable 'if_condition_62784' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'if_condition_62784', if_condition_62784)
        # SSA begins for if statement (line 42)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 42):
        int_62785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 33), 'int')
        # Assigning a type to the variable 'r' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 29), 'r', int_62785)
        # SSA branch for the else part of an if statement (line 42)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to is_r5000(...): (line 43)
        # Processing the call keyword arguments (line 43)
        kwargs_62788 = {}
        # Getting the type of 'cpu' (line 43)
        cpu_62786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), 'cpu', False)
        # Obtaining the member 'is_r5000' of a type (line 43)
        is_r5000_62787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 13), cpu_62786, 'is_r5000')
        # Calling is_r5000(args, kwargs) (line 43)
        is_r5000_call_result_62789 = invoke(stypy.reporting.localization.Localization(__file__, 43, 13), is_r5000_62787, *[], **kwargs_62788)
        
        # Testing the type of an if condition (line 43)
        if_condition_62790 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 13), is_r5000_call_result_62789)
        # Assigning a type to the variable 'if_condition_62790' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), 'if_condition_62790', if_condition_62790)
        # SSA begins for if statement (line 43)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 43):
        int_62791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 33), 'int')
        # Assigning a type to the variable 'r' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 29), 'r', int_62791)
        # SSA branch for the else part of an if statement (line 43)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to is_r4000(...): (line 44)
        # Processing the call keyword arguments (line 44)
        kwargs_62794 = {}
        # Getting the type of 'cpu' (line 44)
        cpu_62792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'cpu', False)
        # Obtaining the member 'is_r4000' of a type (line 44)
        is_r4000_62793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 13), cpu_62792, 'is_r4000')
        # Calling is_r4000(args, kwargs) (line 44)
        is_r4000_call_result_62795 = invoke(stypy.reporting.localization.Localization(__file__, 44, 13), is_r4000_62793, *[], **kwargs_62794)
        
        # Testing the type of an if condition (line 44)
        if_condition_62796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 13), is_r4000_call_result_62795)
        # Assigning a type to the variable 'if_condition_62796' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'if_condition_62796', if_condition_62796)
        # SSA begins for if statement (line 44)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 44):
        int_62797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 33), 'int')
        # Assigning a type to the variable 'r' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 29), 'r', int_62797)
        # SSA join for if statement (line 44)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 43)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 42)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 41)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 40)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 45)
        # Getting the type of 'r' (line 45)
        r_62798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'r')
        # Getting the type of 'None' (line 45)
        None_62799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 20), 'None')
        
        (may_be_62800, more_types_in_union_62801) = may_not_be_none(r_62798, None_62799)

        if may_be_62800:

            if more_types_in_union_62801:
                # Runtime conditional SSA (line 45)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Obtaining an instance of the builtin type 'list' (line 46)
            list_62802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 46)
            # Adding element type (line 46)
            str_62803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 20), 'str', 'r%s')
            # Getting the type of 'r' (line 46)
            r_62804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 29), 'r')
            # Applying the binary operator '%' (line 46)
            result_mod_62805 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 20), '%', str_62803, r_62804)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), list_62802, result_mod_62805)
            
            # Assigning a type to the variable 'stypy_return_type' (line 46)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'stypy_return_type', list_62802)

            if more_types_in_union_62801:
                # SSA join for if statement (line 45)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Obtaining an instance of the builtin type 'list' (line 47)
        list_62806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 47)
        
        # Assigning a type to the variable 'stypy_return_type' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'stypy_return_type', list_62806)
        
        # ################# End of 'get_flags_arch_f77(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_arch_f77' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_62807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62807)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_arch_f77'
        return stypy_return_type_62807


    @norecursion
    def get_flags_arch_f90(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_arch_f90'
        module_type_store = module_type_store.open_function_context('get_flags_arch_f90', 48, 4, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MIPSFCompiler.get_flags_arch_f90.__dict__.__setitem__('stypy_localization', localization)
        MIPSFCompiler.get_flags_arch_f90.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MIPSFCompiler.get_flags_arch_f90.__dict__.__setitem__('stypy_type_store', module_type_store)
        MIPSFCompiler.get_flags_arch_f90.__dict__.__setitem__('stypy_function_name', 'MIPSFCompiler.get_flags_arch_f90')
        MIPSFCompiler.get_flags_arch_f90.__dict__.__setitem__('stypy_param_names_list', [])
        MIPSFCompiler.get_flags_arch_f90.__dict__.__setitem__('stypy_varargs_param_name', None)
        MIPSFCompiler.get_flags_arch_f90.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MIPSFCompiler.get_flags_arch_f90.__dict__.__setitem__('stypy_call_defaults', defaults)
        MIPSFCompiler.get_flags_arch_f90.__dict__.__setitem__('stypy_call_varargs', varargs)
        MIPSFCompiler.get_flags_arch_f90.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MIPSFCompiler.get_flags_arch_f90.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MIPSFCompiler.get_flags_arch_f90', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_arch_f90', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_arch_f90(...)' code ##################

        
        # Assigning a Call to a Name (line 49):
        
        # Call to get_flags_arch_f77(...): (line 49)
        # Processing the call keyword arguments (line 49)
        kwargs_62810 = {}
        # Getting the type of 'self' (line 49)
        self_62808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'self', False)
        # Obtaining the member 'get_flags_arch_f77' of a type (line 49)
        get_flags_arch_f77_62809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), self_62808, 'get_flags_arch_f77')
        # Calling get_flags_arch_f77(args, kwargs) (line 49)
        get_flags_arch_f77_call_result_62811 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), get_flags_arch_f77_62809, *[], **kwargs_62810)
        
        # Assigning a type to the variable 'r' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'r', get_flags_arch_f77_call_result_62811)
        
        # Getting the type of 'r' (line 50)
        r_62812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'r')
        # Testing the type of an if condition (line 50)
        if_condition_62813 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 8), r_62812)
        # Assigning a type to the variable 'if_condition_62813' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'if_condition_62813', if_condition_62813)
        # SSA begins for if statement (line 50)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Subscript (line 51):
        str_62814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 19), 'str', '-')
        
        # Obtaining the type of the subscript
        int_62815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 27), 'int')
        # Getting the type of 'r' (line 51)
        r_62816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'r')
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___62817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 25), r_62816, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_62818 = invoke(stypy.reporting.localization.Localization(__file__, 51, 25), getitem___62817, int_62815)
        
        # Applying the binary operator '+' (line 51)
        result_add_62819 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 19), '+', str_62814, subscript_call_result_62818)
        
        # Getting the type of 'r' (line 51)
        r_62820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'r')
        int_62821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 14), 'int')
        # Storing an element on a container (line 51)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 12), r_62820, (int_62821, result_add_62819))
        # SSA join for if statement (line 50)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'r' (line 52)
        r_62822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'r')
        # Assigning a type to the variable 'stypy_return_type' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'stypy_return_type', r_62822)
        
        # ################# End of 'get_flags_arch_f90(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_arch_f90' in the type store
        # Getting the type of 'stypy_return_type' (line 48)
        stypy_return_type_62823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62823)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_arch_f90'
        return stypy_return_type_62823


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 8, 0, False)
        # Assigning a type to the variable 'self' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MIPSFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MIPSFCompiler' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'MIPSFCompiler', MIPSFCompiler)

# Assigning a Str to a Name (line 10):
str_62824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 20), 'str', 'mips')
# Getting the type of 'MIPSFCompiler'
MIPSFCompiler_62825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MIPSFCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MIPSFCompiler_62825, 'compiler_type', str_62824)

# Assigning a Str to a Name (line 11):
str_62826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 18), 'str', 'MIPSpro Fortran Compiler')
# Getting the type of 'MIPSFCompiler'
MIPSFCompiler_62827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MIPSFCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MIPSFCompiler_62827, 'description', str_62826)

# Assigning a Str to a Name (line 12):
str_62828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 23), 'str', 'MIPSpro Compilers: Version (?P<version>[^\\s*,]*)')
# Getting the type of 'MIPSFCompiler'
MIPSFCompiler_62829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MIPSFCompiler')
# Setting the type of the member 'version_pattern' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MIPSFCompiler_62829, 'version_pattern', str_62828)

# Assigning a Dict to a Name (line 14):

# Obtaining an instance of the builtin type 'dict' (line 14)
dict_62830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 14)
# Adding element type (key, value) (line 14)
str_62831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 8), 'str', 'version_cmd')

# Obtaining an instance of the builtin type 'list' (line 15)
list_62832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
str_62833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 26), 'str', '<F90>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 25), list_62832, str_62833)
# Adding element type (line 15)
str_62834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 35), 'str', '-version')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 25), list_62832, str_62834)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 18), dict_62830, (str_62831, list_62832))
# Adding element type (key, value) (line 14)
str_62835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 16)
list_62836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
str_62837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 26), 'str', 'f77')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 25), list_62836, str_62837)
# Adding element type (line 16)
str_62838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 33), 'str', '-f77')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 25), list_62836, str_62838)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 18), dict_62830, (str_62835, list_62836))
# Adding element type (key, value) (line 14)
str_62839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 17)
list_62840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
str_62841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 26), 'str', 'f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 25), list_62840, str_62841)
# Adding element type (line 17)
str_62842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 33), 'str', '-fixedform')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 25), list_62840, str_62842)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 18), dict_62830, (str_62839, list_62840))
# Adding element type (key, value) (line 14)
str_62843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 18)
list_62844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
str_62845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 26), 'str', 'f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 25), list_62844, str_62845)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 18), dict_62830, (str_62843, list_62844))
# Adding element type (key, value) (line 14)
str_62846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 19)
list_62847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
str_62848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'str', 'f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 25), list_62847, str_62848)
# Adding element type (line 19)
str_62849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 33), 'str', '-shared')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 25), list_62847, str_62849)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 18), dict_62830, (str_62846, list_62847))
# Adding element type (key, value) (line 14)
str_62850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 20)
list_62851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 20)
# Adding element type (line 20)
str_62852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), list_62851, str_62852)
# Adding element type (line 20)
str_62853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 32), 'str', '-cr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), list_62851, str_62853)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 18), dict_62830, (str_62850, list_62851))
# Adding element type (key, value) (line 14)
str_62854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 8), 'str', 'ranlib')
# Getting the type of 'None' (line 21)
None_62855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 25), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 18), dict_62830, (str_62854, None_62855))

# Getting the type of 'MIPSFCompiler'
MIPSFCompiler_62856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MIPSFCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MIPSFCompiler_62856, 'executables', dict_62830)

# Assigning a Name to a Name (line 23):
# Getting the type of 'None' (line 23)
None_62857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'None')
# Getting the type of 'MIPSFCompiler'
MIPSFCompiler_62858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MIPSFCompiler')
# Setting the type of the member 'module_dir_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MIPSFCompiler_62858, 'module_dir_switch', None_62857)

# Assigning a Name to a Name (line 24):
# Getting the type of 'None' (line 24)
None_62859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 28), 'None')
# Getting the type of 'MIPSFCompiler'
MIPSFCompiler_62860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MIPSFCompiler')
# Setting the type of the member 'module_include_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MIPSFCompiler_62860, 'module_include_switch', None_62859)

# Assigning a List to a Name (line 25):

# Obtaining an instance of the builtin type 'list' (line 25)
list_62861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
str_62862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 17), 'str', '-KPIC')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 16), list_62861, str_62862)

# Getting the type of 'MIPSFCompiler'
MIPSFCompiler_62863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MIPSFCompiler')
# Setting the type of the member 'pic_flags' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MIPSFCompiler_62863, 'pic_flags', list_62861)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 55, 4))
    
    # 'from numpy.distutils.fcompiler import new_fcompiler' statement (line 55)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    import_62864 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 55, 4), 'numpy.distutils.fcompiler')

    if (type(import_62864) is not StypyTypeError):

        if (import_62864 != 'pyd_module'):
            __import__(import_62864)
            sys_modules_62865 = sys.modules[import_62864]
            import_from_module(stypy.reporting.localization.Localization(__file__, 55, 4), 'numpy.distutils.fcompiler', sys_modules_62865.module_type_store, module_type_store, ['new_fcompiler'])
            nest_module(stypy.reporting.localization.Localization(__file__, 55, 4), __file__, sys_modules_62865, sys_modules_62865.module_type_store, module_type_store)
        else:
            from numpy.distutils.fcompiler import new_fcompiler

            import_from_module(stypy.reporting.localization.Localization(__file__, 55, 4), 'numpy.distutils.fcompiler', None, module_type_store, ['new_fcompiler'], [new_fcompiler])

    else:
        # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'numpy.distutils.fcompiler', import_62864)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    
    
    # Assigning a Call to a Name (line 56):
    
    # Call to new_fcompiler(...): (line 56)
    # Processing the call keyword arguments (line 56)
    str_62867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 38), 'str', 'mips')
    keyword_62868 = str_62867
    kwargs_62869 = {'compiler': keyword_62868}
    # Getting the type of 'new_fcompiler' (line 56)
    new_fcompiler_62866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 'new_fcompiler', False)
    # Calling new_fcompiler(args, kwargs) (line 56)
    new_fcompiler_call_result_62870 = invoke(stypy.reporting.localization.Localization(__file__, 56, 15), new_fcompiler_62866, *[], **kwargs_62869)
    
    # Assigning a type to the variable 'compiler' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'compiler', new_fcompiler_call_result_62870)
    
    # Call to customize(...): (line 57)
    # Processing the call keyword arguments (line 57)
    kwargs_62873 = {}
    # Getting the type of 'compiler' (line 57)
    compiler_62871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'compiler', False)
    # Obtaining the member 'customize' of a type (line 57)
    customize_62872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 4), compiler_62871, 'customize')
    # Calling customize(args, kwargs) (line 57)
    customize_call_result_62874 = invoke(stypy.reporting.localization.Localization(__file__, 57, 4), customize_62872, *[], **kwargs_62873)
    
    
    # Call to print(...): (line 58)
    # Processing the call arguments (line 58)
    
    # Call to get_version(...): (line 58)
    # Processing the call keyword arguments (line 58)
    kwargs_62878 = {}
    # Getting the type of 'compiler' (line 58)
    compiler_62876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 10), 'compiler', False)
    # Obtaining the member 'get_version' of a type (line 58)
    get_version_62877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 10), compiler_62876, 'get_version')
    # Calling get_version(args, kwargs) (line 58)
    get_version_call_result_62879 = invoke(stypy.reporting.localization.Localization(__file__, 58, 10), get_version_62877, *[], **kwargs_62878)
    
    # Processing the call keyword arguments (line 58)
    kwargs_62880 = {}
    # Getting the type of 'print' (line 58)
    print_62875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'print', False)
    # Calling print(args, kwargs) (line 58)
    print_call_result_62881 = invoke(stypy.reporting.localization.Localization(__file__, 58, 4), print_62875, *[get_version_call_result_62879], **kwargs_62880)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
