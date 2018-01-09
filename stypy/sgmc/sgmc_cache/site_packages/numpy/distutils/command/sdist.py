
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import sys
4: if 'setuptools' in sys.modules:
5:     from setuptools.command.sdist import sdist as old_sdist
6: else:
7:     from distutils.command.sdist import sdist as old_sdist
8: 
9: from numpy.distutils.misc_util import get_data_files
10: 
11: class sdist(old_sdist):
12: 
13:     def add_defaults (self):
14:         old_sdist.add_defaults(self)
15: 
16:         dist = self.distribution
17: 
18:         if dist.has_data_files():
19:             for data in dist.data_files:
20:                 self.filelist.extend(get_data_files(data))
21: 
22:         if dist.has_headers():
23:             headers = []
24:             for h in dist.headers:
25:                 if isinstance(h, str): headers.append(h)
26:                 else: headers.append(h[1])
27:             self.filelist.extend(headers)
28: 
29:         return
30: 

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



str_59722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 3), 'str', 'setuptools')
# Getting the type of 'sys' (line 4)
sys_59723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 19), 'sys')
# Obtaining the member 'modules' of a type (line 4)
modules_59724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 19), sys_59723, 'modules')
# Applying the binary operator 'in' (line 4)
result_contains_59725 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 3), 'in', str_59722, modules_59724)

# Testing the type of an if condition (line 4)
if_condition_59726 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 4, 0), result_contains_59725)
# Assigning a type to the variable 'if_condition_59726' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'if_condition_59726', if_condition_59726)
# SSA begins for if statement (line 4)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 4))

# 'from setuptools.command.sdist import old_sdist' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_59727 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'setuptools.command.sdist')

if (type(import_59727) is not StypyTypeError):

    if (import_59727 != 'pyd_module'):
        __import__(import_59727)
        sys_modules_59728 = sys.modules[import_59727]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'setuptools.command.sdist', sys_modules_59728.module_type_store, module_type_store, ['sdist'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 4), __file__, sys_modules_59728, sys_modules_59728.module_type_store, module_type_store)
    else:
        from setuptools.command.sdist import sdist as old_sdist

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'setuptools.command.sdist', None, module_type_store, ['sdist'], [old_sdist])

else:
    # Assigning a type to the variable 'setuptools.command.sdist' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'setuptools.command.sdist', import_59727)

# Adding an alias
module_type_store.add_alias('old_sdist', 'sdist')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

# SSA branch for the else part of an if statement (line 4)
module_type_store.open_ssa_branch('else')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 4))

# 'from distutils.command.sdist import old_sdist' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_59729 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'distutils.command.sdist')

if (type(import_59729) is not StypyTypeError):

    if (import_59729 != 'pyd_module'):
        __import__(import_59729)
        sys_modules_59730 = sys.modules[import_59729]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'distutils.command.sdist', sys_modules_59730.module_type_store, module_type_store, ['sdist'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 4), __file__, sys_modules_59730, sys_modules_59730.module_type_store, module_type_store)
    else:
        from distutils.command.sdist import sdist as old_sdist

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'distutils.command.sdist', None, module_type_store, ['sdist'], [old_sdist])

else:
    # Assigning a type to the variable 'distutils.command.sdist' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'distutils.command.sdist', import_59729)

# Adding an alias
module_type_store.add_alias('old_sdist', 'sdist')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

# SSA join for if statement (line 4)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.distutils.misc_util import get_data_files' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_59731 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.distutils.misc_util')

if (type(import_59731) is not StypyTypeError):

    if (import_59731 != 'pyd_module'):
        __import__(import_59731)
        sys_modules_59732 = sys.modules[import_59731]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.distutils.misc_util', sys_modules_59732.module_type_store, module_type_store, ['get_data_files'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_59732, sys_modules_59732.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import get_data_files

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.distutils.misc_util', None, module_type_store, ['get_data_files'], [get_data_files])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.distutils.misc_util', import_59731)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

# Declaration of the 'sdist' class
# Getting the type of 'old_sdist' (line 11)
old_sdist_59733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'old_sdist')

class sdist(old_sdist_59733, ):

    @norecursion
    def add_defaults(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_defaults'
        module_type_store = module_type_store.open_function_context('add_defaults', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sdist.add_defaults.__dict__.__setitem__('stypy_localization', localization)
        sdist.add_defaults.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sdist.add_defaults.__dict__.__setitem__('stypy_type_store', module_type_store)
        sdist.add_defaults.__dict__.__setitem__('stypy_function_name', 'sdist.add_defaults')
        sdist.add_defaults.__dict__.__setitem__('stypy_param_names_list', [])
        sdist.add_defaults.__dict__.__setitem__('stypy_varargs_param_name', None)
        sdist.add_defaults.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sdist.add_defaults.__dict__.__setitem__('stypy_call_defaults', defaults)
        sdist.add_defaults.__dict__.__setitem__('stypy_call_varargs', varargs)
        sdist.add_defaults.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sdist.add_defaults.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sdist.add_defaults', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_defaults', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_defaults(...)' code ##################

        
        # Call to add_defaults(...): (line 14)
        # Processing the call arguments (line 14)
        # Getting the type of 'self' (line 14)
        self_59736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 31), 'self', False)
        # Processing the call keyword arguments (line 14)
        kwargs_59737 = {}
        # Getting the type of 'old_sdist' (line 14)
        old_sdist_59734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'old_sdist', False)
        # Obtaining the member 'add_defaults' of a type (line 14)
        add_defaults_59735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), old_sdist_59734, 'add_defaults')
        # Calling add_defaults(args, kwargs) (line 14)
        add_defaults_call_result_59738 = invoke(stypy.reporting.localization.Localization(__file__, 14, 8), add_defaults_59735, *[self_59736], **kwargs_59737)
        
        
        # Assigning a Attribute to a Name (line 16):
        # Getting the type of 'self' (line 16)
        self_59739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'self')
        # Obtaining the member 'distribution' of a type (line 16)
        distribution_59740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 15), self_59739, 'distribution')
        # Assigning a type to the variable 'dist' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'dist', distribution_59740)
        
        
        # Call to has_data_files(...): (line 18)
        # Processing the call keyword arguments (line 18)
        kwargs_59743 = {}
        # Getting the type of 'dist' (line 18)
        dist_59741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'dist', False)
        # Obtaining the member 'has_data_files' of a type (line 18)
        has_data_files_59742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 11), dist_59741, 'has_data_files')
        # Calling has_data_files(args, kwargs) (line 18)
        has_data_files_call_result_59744 = invoke(stypy.reporting.localization.Localization(__file__, 18, 11), has_data_files_59742, *[], **kwargs_59743)
        
        # Testing the type of an if condition (line 18)
        if_condition_59745 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 18, 8), has_data_files_call_result_59744)
        # Assigning a type to the variable 'if_condition_59745' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'if_condition_59745', if_condition_59745)
        # SSA begins for if statement (line 18)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'dist' (line 19)
        dist_59746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 24), 'dist')
        # Obtaining the member 'data_files' of a type (line 19)
        data_files_59747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 24), dist_59746, 'data_files')
        # Testing the type of a for loop iterable (line 19)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 19, 12), data_files_59747)
        # Getting the type of the for loop variable (line 19)
        for_loop_var_59748 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 19, 12), data_files_59747)
        # Assigning a type to the variable 'data' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'data', for_loop_var_59748)
        # SSA begins for a for statement (line 19)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to extend(...): (line 20)
        # Processing the call arguments (line 20)
        
        # Call to get_data_files(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'data' (line 20)
        data_59753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 52), 'data', False)
        # Processing the call keyword arguments (line 20)
        kwargs_59754 = {}
        # Getting the type of 'get_data_files' (line 20)
        get_data_files_59752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 37), 'get_data_files', False)
        # Calling get_data_files(args, kwargs) (line 20)
        get_data_files_call_result_59755 = invoke(stypy.reporting.localization.Localization(__file__, 20, 37), get_data_files_59752, *[data_59753], **kwargs_59754)
        
        # Processing the call keyword arguments (line 20)
        kwargs_59756 = {}
        # Getting the type of 'self' (line 20)
        self_59749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'self', False)
        # Obtaining the member 'filelist' of a type (line 20)
        filelist_59750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 16), self_59749, 'filelist')
        # Obtaining the member 'extend' of a type (line 20)
        extend_59751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 16), filelist_59750, 'extend')
        # Calling extend(args, kwargs) (line 20)
        extend_call_result_59757 = invoke(stypy.reporting.localization.Localization(__file__, 20, 16), extend_59751, *[get_data_files_call_result_59755], **kwargs_59756)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 18)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to has_headers(...): (line 22)
        # Processing the call keyword arguments (line 22)
        kwargs_59760 = {}
        # Getting the type of 'dist' (line 22)
        dist_59758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'dist', False)
        # Obtaining the member 'has_headers' of a type (line 22)
        has_headers_59759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 11), dist_59758, 'has_headers')
        # Calling has_headers(args, kwargs) (line 22)
        has_headers_call_result_59761 = invoke(stypy.reporting.localization.Localization(__file__, 22, 11), has_headers_59759, *[], **kwargs_59760)
        
        # Testing the type of an if condition (line 22)
        if_condition_59762 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 8), has_headers_call_result_59761)
        # Assigning a type to the variable 'if_condition_59762' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'if_condition_59762', if_condition_59762)
        # SSA begins for if statement (line 22)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 23):
        
        # Obtaining an instance of the builtin type 'list' (line 23)
        list_59763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 23)
        
        # Assigning a type to the variable 'headers' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'headers', list_59763)
        
        # Getting the type of 'dist' (line 24)
        dist_59764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 21), 'dist')
        # Obtaining the member 'headers' of a type (line 24)
        headers_59765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 21), dist_59764, 'headers')
        # Testing the type of a for loop iterable (line 24)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 24, 12), headers_59765)
        # Getting the type of the for loop variable (line 24)
        for_loop_var_59766 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 24, 12), headers_59765)
        # Assigning a type to the variable 'h' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'h', for_loop_var_59766)
        # SSA begins for a for statement (line 24)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 25)
        # Getting the type of 'str' (line 25)
        str_59767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 33), 'str')
        # Getting the type of 'h' (line 25)
        h_59768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 30), 'h')
        
        (may_be_59769, more_types_in_union_59770) = may_be_subtype(str_59767, h_59768)

        if may_be_59769:

            if more_types_in_union_59770:
                # Runtime conditional SSA (line 25)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'h' (line 25)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'h', remove_not_subtype_from_union(h_59768, str))
            
            # Call to append(...): (line 25)
            # Processing the call arguments (line 25)
            # Getting the type of 'h' (line 25)
            h_59773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 54), 'h', False)
            # Processing the call keyword arguments (line 25)
            kwargs_59774 = {}
            # Getting the type of 'headers' (line 25)
            headers_59771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 39), 'headers', False)
            # Obtaining the member 'append' of a type (line 25)
            append_59772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 39), headers_59771, 'append')
            # Calling append(args, kwargs) (line 25)
            append_call_result_59775 = invoke(stypy.reporting.localization.Localization(__file__, 25, 39), append_59772, *[h_59773], **kwargs_59774)
            

            if more_types_in_union_59770:
                # Runtime conditional SSA for else branch (line 25)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_59769) or more_types_in_union_59770):
            # Assigning a type to the variable 'h' (line 25)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'h', remove_subtype_from_union(h_59768, str))
            
            # Call to append(...): (line 26)
            # Processing the call arguments (line 26)
            
            # Obtaining the type of the subscript
            int_59778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 39), 'int')
            # Getting the type of 'h' (line 26)
            h_59779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 37), 'h', False)
            # Obtaining the member '__getitem__' of a type (line 26)
            getitem___59780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 37), h_59779, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 26)
            subscript_call_result_59781 = invoke(stypy.reporting.localization.Localization(__file__, 26, 37), getitem___59780, int_59778)
            
            # Processing the call keyword arguments (line 26)
            kwargs_59782 = {}
            # Getting the type of 'headers' (line 26)
            headers_59776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 22), 'headers', False)
            # Obtaining the member 'append' of a type (line 26)
            append_59777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 22), headers_59776, 'append')
            # Calling append(args, kwargs) (line 26)
            append_call_result_59783 = invoke(stypy.reporting.localization.Localization(__file__, 26, 22), append_59777, *[subscript_call_result_59781], **kwargs_59782)
            

            if (may_be_59769 and more_types_in_union_59770):
                # SSA join for if statement (line 25)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to extend(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'headers' (line 27)
        headers_59787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 33), 'headers', False)
        # Processing the call keyword arguments (line 27)
        kwargs_59788 = {}
        # Getting the type of 'self' (line 27)
        self_59784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'self', False)
        # Obtaining the member 'filelist' of a type (line 27)
        filelist_59785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), self_59784, 'filelist')
        # Obtaining the member 'extend' of a type (line 27)
        extend_59786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), filelist_59785, 'extend')
        # Calling extend(args, kwargs) (line 27)
        extend_call_result_59789 = invoke(stypy.reporting.localization.Localization(__file__, 27, 12), extend_59786, *[headers_59787], **kwargs_59788)
        
        # SSA join for if statement (line 22)
        module_type_store = module_type_store.join_ssa_context()
        
        # Assigning a type to the variable 'stypy_return_type' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'add_defaults(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_defaults' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_59790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59790)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_defaults'
        return stypy_return_type_59790


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 11, 0, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sdist.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'sdist' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'sdist', sdist)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
