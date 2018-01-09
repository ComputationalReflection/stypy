
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import os
4: from distutils.command.install_headers import install_headers as old_install_headers
5: 
6: class install_headers (old_install_headers):
7: 
8:     def run (self):
9:         headers = self.distribution.headers
10:         if not headers:
11:             return
12: 
13:         prefix = os.path.dirname(self.install_dir)
14:         for header in headers:
15:             if isinstance(header, tuple):
16:                 # Kind of a hack, but I don't know where else to change this...
17:                 if header[0] == 'numpy.core':
18:                     header = ('numpy', header[1])
19:                     if os.path.splitext(header[1])[1] == '.inc':
20:                         continue
21:                 d = os.path.join(*([prefix]+header[0].split('.')))
22:                 header = header[1]
23:             else:
24:                 d = self.install_dir
25:             self.mkpath(d)
26:             (out, _) = self.copy_file(header, d)
27:             self.outfiles.append(out)
28: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from distutils.command.install_headers import old_install_headers' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_59620 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.command.install_headers')

if (type(import_59620) is not StypyTypeError):

    if (import_59620 != 'pyd_module'):
        __import__(import_59620)
        sys_modules_59621 = sys.modules[import_59620]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.command.install_headers', sys_modules_59621.module_type_store, module_type_store, ['install_headers'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_59621, sys_modules_59621.module_type_store, module_type_store)
    else:
        from distutils.command.install_headers import install_headers as old_install_headers

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.command.install_headers', None, module_type_store, ['install_headers'], [old_install_headers])

else:
    # Assigning a type to the variable 'distutils.command.install_headers' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.command.install_headers', import_59620)

# Adding an alias
module_type_store.add_alias('old_install_headers', 'install_headers')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

# Declaration of the 'install_headers' class
# Getting the type of 'old_install_headers' (line 6)
old_install_headers_59622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 23), 'old_install_headers')

class install_headers(old_install_headers_59622, ):

    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 8, 4, False)
        # Assigning a type to the variable 'self' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_headers.run.__dict__.__setitem__('stypy_localization', localization)
        install_headers.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_headers.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_headers.run.__dict__.__setitem__('stypy_function_name', 'install_headers.run')
        install_headers.run.__dict__.__setitem__('stypy_param_names_list', [])
        install_headers.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_headers.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_headers.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_headers.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_headers.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_headers.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_headers.run', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Name (line 9):
        
        # Assigning a Attribute to a Name (line 9):
        # Getting the type of 'self' (line 9)
        self_59623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 18), 'self')
        # Obtaining the member 'distribution' of a type (line 9)
        distribution_59624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 18), self_59623, 'distribution')
        # Obtaining the member 'headers' of a type (line 9)
        headers_59625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 18), distribution_59624, 'headers')
        # Assigning a type to the variable 'headers' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'headers', headers_59625)
        
        
        # Getting the type of 'headers' (line 10)
        headers_59626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 15), 'headers')
        # Applying the 'not' unary operator (line 10)
        result_not__59627 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 11), 'not', headers_59626)
        
        # Testing the type of an if condition (line 10)
        if_condition_59628 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 10, 8), result_not__59627)
        # Assigning a type to the variable 'if_condition_59628' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'if_condition_59628', if_condition_59628)
        # SSA begins for if statement (line 10)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 10)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 13):
        
        # Assigning a Call to a Name (line 13):
        
        # Call to dirname(...): (line 13)
        # Processing the call arguments (line 13)
        # Getting the type of 'self' (line 13)
        self_59632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 33), 'self', False)
        # Obtaining the member 'install_dir' of a type (line 13)
        install_dir_59633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 33), self_59632, 'install_dir')
        # Processing the call keyword arguments (line 13)
        kwargs_59634 = {}
        # Getting the type of 'os' (line 13)
        os_59629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 13)
        path_59630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 17), os_59629, 'path')
        # Obtaining the member 'dirname' of a type (line 13)
        dirname_59631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 17), path_59630, 'dirname')
        # Calling dirname(args, kwargs) (line 13)
        dirname_call_result_59635 = invoke(stypy.reporting.localization.Localization(__file__, 13, 17), dirname_59631, *[install_dir_59633], **kwargs_59634)
        
        # Assigning a type to the variable 'prefix' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'prefix', dirname_call_result_59635)
        
        # Getting the type of 'headers' (line 14)
        headers_59636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 22), 'headers')
        # Testing the type of a for loop iterable (line 14)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 14, 8), headers_59636)
        # Getting the type of the for loop variable (line 14)
        for_loop_var_59637 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 14, 8), headers_59636)
        # Assigning a type to the variable 'header' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'header', for_loop_var_59637)
        # SSA begins for a for statement (line 14)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 15)
        # Getting the type of 'tuple' (line 15)
        tuple_59638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 34), 'tuple')
        # Getting the type of 'header' (line 15)
        header_59639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 26), 'header')
        
        (may_be_59640, more_types_in_union_59641) = may_be_subtype(tuple_59638, header_59639)

        if may_be_59640:

            if more_types_in_union_59641:
                # Runtime conditional SSA (line 15)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'header' (line 15)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'header', remove_not_subtype_from_union(header_59639, tuple))
            
            
            
            # Obtaining the type of the subscript
            int_59642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 26), 'int')
            # Getting the type of 'header' (line 17)
            header_59643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 19), 'header')
            # Obtaining the member '__getitem__' of a type (line 17)
            getitem___59644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 19), header_59643, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 17)
            subscript_call_result_59645 = invoke(stypy.reporting.localization.Localization(__file__, 17, 19), getitem___59644, int_59642)
            
            str_59646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 32), 'str', 'numpy.core')
            # Applying the binary operator '==' (line 17)
            result_eq_59647 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 19), '==', subscript_call_result_59645, str_59646)
            
            # Testing the type of an if condition (line 17)
            if_condition_59648 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 16), result_eq_59647)
            # Assigning a type to the variable 'if_condition_59648' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 16), 'if_condition_59648', if_condition_59648)
            # SSA begins for if statement (line 17)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Tuple to a Name (line 18):
            
            # Assigning a Tuple to a Name (line 18):
            
            # Obtaining an instance of the builtin type 'tuple' (line 18)
            tuple_59649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 18)
            # Adding element type (line 18)
            str_59650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 30), 'str', 'numpy')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 30), tuple_59649, str_59650)
            # Adding element type (line 18)
            
            # Obtaining the type of the subscript
            int_59651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 46), 'int')
            # Getting the type of 'header' (line 18)
            header_59652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 39), 'header')
            # Obtaining the member '__getitem__' of a type (line 18)
            getitem___59653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 39), header_59652, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 18)
            subscript_call_result_59654 = invoke(stypy.reporting.localization.Localization(__file__, 18, 39), getitem___59653, int_59651)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 30), tuple_59649, subscript_call_result_59654)
            
            # Assigning a type to the variable 'header' (line 18)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 20), 'header', tuple_59649)
            
            
            
            # Obtaining the type of the subscript
            int_59655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 51), 'int')
            
            # Call to splitext(...): (line 19)
            # Processing the call arguments (line 19)
            
            # Obtaining the type of the subscript
            int_59659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 47), 'int')
            # Getting the type of 'header' (line 19)
            header_59660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 40), 'header', False)
            # Obtaining the member '__getitem__' of a type (line 19)
            getitem___59661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 40), header_59660, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 19)
            subscript_call_result_59662 = invoke(stypy.reporting.localization.Localization(__file__, 19, 40), getitem___59661, int_59659)
            
            # Processing the call keyword arguments (line 19)
            kwargs_59663 = {}
            # Getting the type of 'os' (line 19)
            os_59656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'os', False)
            # Obtaining the member 'path' of a type (line 19)
            path_59657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 23), os_59656, 'path')
            # Obtaining the member 'splitext' of a type (line 19)
            splitext_59658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 23), path_59657, 'splitext')
            # Calling splitext(args, kwargs) (line 19)
            splitext_call_result_59664 = invoke(stypy.reporting.localization.Localization(__file__, 19, 23), splitext_59658, *[subscript_call_result_59662], **kwargs_59663)
            
            # Obtaining the member '__getitem__' of a type (line 19)
            getitem___59665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 23), splitext_call_result_59664, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 19)
            subscript_call_result_59666 = invoke(stypy.reporting.localization.Localization(__file__, 19, 23), getitem___59665, int_59655)
            
            str_59667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 57), 'str', '.inc')
            # Applying the binary operator '==' (line 19)
            result_eq_59668 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 23), '==', subscript_call_result_59666, str_59667)
            
            # Testing the type of an if condition (line 19)
            if_condition_59669 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 19, 20), result_eq_59668)
            # Assigning a type to the variable 'if_condition_59669' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'if_condition_59669', if_condition_59669)
            # SSA begins for if statement (line 19)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 19)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 17)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 21):
            
            # Assigning a Call to a Name (line 21):
            
            # Call to join(...): (line 21)
            
            # Obtaining an instance of the builtin type 'list' (line 21)
            list_59673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 35), 'list')
            # Adding type elements to the builtin type 'list' instance (line 21)
            # Adding element type (line 21)
            # Getting the type of 'prefix' (line 21)
            prefix_59674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 36), 'prefix', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 35), list_59673, prefix_59674)
            
            
            # Call to split(...): (line 21)
            # Processing the call arguments (line 21)
            str_59680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 60), 'str', '.')
            # Processing the call keyword arguments (line 21)
            kwargs_59681 = {}
            
            # Obtaining the type of the subscript
            int_59675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 51), 'int')
            # Getting the type of 'header' (line 21)
            header_59676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 44), 'header', False)
            # Obtaining the member '__getitem__' of a type (line 21)
            getitem___59677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 44), header_59676, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 21)
            subscript_call_result_59678 = invoke(stypy.reporting.localization.Localization(__file__, 21, 44), getitem___59677, int_59675)
            
            # Obtaining the member 'split' of a type (line 21)
            split_59679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 44), subscript_call_result_59678, 'split')
            # Calling split(args, kwargs) (line 21)
            split_call_result_59682 = invoke(stypy.reporting.localization.Localization(__file__, 21, 44), split_59679, *[str_59680], **kwargs_59681)
            
            # Applying the binary operator '+' (line 21)
            result_add_59683 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 35), '+', list_59673, split_call_result_59682)
            
            # Processing the call keyword arguments (line 21)
            kwargs_59684 = {}
            # Getting the type of 'os' (line 21)
            os_59670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'os', False)
            # Obtaining the member 'path' of a type (line 21)
            path_59671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 20), os_59670, 'path')
            # Obtaining the member 'join' of a type (line 21)
            join_59672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 20), path_59671, 'join')
            # Calling join(args, kwargs) (line 21)
            join_call_result_59685 = invoke(stypy.reporting.localization.Localization(__file__, 21, 20), join_59672, *[result_add_59683], **kwargs_59684)
            
            # Assigning a type to the variable 'd' (line 21)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'd', join_call_result_59685)
            
            # Assigning a Subscript to a Name (line 22):
            
            # Assigning a Subscript to a Name (line 22):
            
            # Obtaining the type of the subscript
            int_59686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 32), 'int')
            # Getting the type of 'header' (line 22)
            header_59687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), 'header')
            # Obtaining the member '__getitem__' of a type (line 22)
            getitem___59688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 25), header_59687, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 22)
            subscript_call_result_59689 = invoke(stypy.reporting.localization.Localization(__file__, 22, 25), getitem___59688, int_59686)
            
            # Assigning a type to the variable 'header' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'header', subscript_call_result_59689)

            if more_types_in_union_59641:
                # Runtime conditional SSA for else branch (line 15)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_59640) or more_types_in_union_59641):
            # Assigning a type to the variable 'header' (line 15)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'header', remove_subtype_from_union(header_59639, tuple))
            
            # Assigning a Attribute to a Name (line 24):
            
            # Assigning a Attribute to a Name (line 24):
            # Getting the type of 'self' (line 24)
            self_59690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), 'self')
            # Obtaining the member 'install_dir' of a type (line 24)
            install_dir_59691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 20), self_59690, 'install_dir')
            # Assigning a type to the variable 'd' (line 24)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'd', install_dir_59691)

            if (may_be_59640 and more_types_in_union_59641):
                # SSA join for if statement (line 15)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to mkpath(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'd' (line 25)
        d_59694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'd', False)
        # Processing the call keyword arguments (line 25)
        kwargs_59695 = {}
        # Getting the type of 'self' (line 25)
        self_59692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 25)
        mkpath_59693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 12), self_59692, 'mkpath')
        # Calling mkpath(args, kwargs) (line 25)
        mkpath_call_result_59696 = invoke(stypy.reporting.localization.Localization(__file__, 25, 12), mkpath_59693, *[d_59694], **kwargs_59695)
        
        
        # Assigning a Call to a Tuple (line 26):
        
        # Assigning a Call to a Name:
        
        # Call to copy_file(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'header' (line 26)
        header_59699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 38), 'header', False)
        # Getting the type of 'd' (line 26)
        d_59700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 46), 'd', False)
        # Processing the call keyword arguments (line 26)
        kwargs_59701 = {}
        # Getting the type of 'self' (line 26)
        self_59697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'self', False)
        # Obtaining the member 'copy_file' of a type (line 26)
        copy_file_59698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 23), self_59697, 'copy_file')
        # Calling copy_file(args, kwargs) (line 26)
        copy_file_call_result_59702 = invoke(stypy.reporting.localization.Localization(__file__, 26, 23), copy_file_59698, *[header_59699, d_59700], **kwargs_59701)
        
        # Assigning a type to the variable 'call_assignment_59617' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'call_assignment_59617', copy_file_call_result_59702)
        
        # Assigning a Call to a Name (line 26):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_59705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 12), 'int')
        # Processing the call keyword arguments
        kwargs_59706 = {}
        # Getting the type of 'call_assignment_59617' (line 26)
        call_assignment_59617_59703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'call_assignment_59617', False)
        # Obtaining the member '__getitem__' of a type (line 26)
        getitem___59704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 12), call_assignment_59617_59703, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_59707 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___59704, *[int_59705], **kwargs_59706)
        
        # Assigning a type to the variable 'call_assignment_59618' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'call_assignment_59618', getitem___call_result_59707)
        
        # Assigning a Name to a Name (line 26):
        # Getting the type of 'call_assignment_59618' (line 26)
        call_assignment_59618_59708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'call_assignment_59618')
        # Assigning a type to the variable 'out' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 13), 'out', call_assignment_59618_59708)
        
        # Assigning a Call to a Name (line 26):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_59711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 12), 'int')
        # Processing the call keyword arguments
        kwargs_59712 = {}
        # Getting the type of 'call_assignment_59617' (line 26)
        call_assignment_59617_59709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'call_assignment_59617', False)
        # Obtaining the member '__getitem__' of a type (line 26)
        getitem___59710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 12), call_assignment_59617_59709, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_59713 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___59710, *[int_59711], **kwargs_59712)
        
        # Assigning a type to the variable 'call_assignment_59619' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'call_assignment_59619', getitem___call_result_59713)
        
        # Assigning a Name to a Name (line 26):
        # Getting the type of 'call_assignment_59619' (line 26)
        call_assignment_59619_59714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'call_assignment_59619')
        # Assigning a type to the variable '_' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), '_', call_assignment_59619_59714)
        
        # Call to append(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'out' (line 27)
        out_59718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 33), 'out', False)
        # Processing the call keyword arguments (line 27)
        kwargs_59719 = {}
        # Getting the type of 'self' (line 27)
        self_59715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'self', False)
        # Obtaining the member 'outfiles' of a type (line 27)
        outfiles_59716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), self_59715, 'outfiles')
        # Obtaining the member 'append' of a type (line 27)
        append_59717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), outfiles_59716, 'append')
        # Calling append(args, kwargs) (line 27)
        append_call_result_59720 = invoke(stypy.reporting.localization.Localization(__file__, 27, 12), append_59717, *[out_59718], **kwargs_59719)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 8)
        stypy_return_type_59721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59721)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_59721


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 6, 0, False)
        # Assigning a type to the variable 'self' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_headers.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'install_headers' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'install_headers', install_headers)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
