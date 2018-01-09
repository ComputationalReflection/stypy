
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: class FooParent:
2:     def method(self):
3:         if True:
4:             return 3
5:         else:
6:             return list()
7: 
8: 
9: class FooChild(FooParent):
10:     def method(self):
11:         if True:
12:             return "a"
13:         else:
14:             return True
15: 
16: 
17: if True:
18:     o = FooParent()
19: else:
20:     o = FooChild()
21: 
22: x = o.method()
23: r1 = x.nothing()  # Detected (x is int | str)
24: l = len(x)  # Unreported (optimistic)
25: 
26: if True:
27:     o2 = FooParent()
28:     r2 = o2.method()
29: else:
30:     o2 = FooChild()
31:     r2 = o2.method()
32: 
33: r3 = r2.nothing()  # Detected (x is int | str)
34: l2 = len(x)  # Unreported (optimistic)
35: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'FooParent' class

class FooParent:

    @norecursion
    def method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'method'
        module_type_store = module_type_store.open_function_context('method', 2, 4, False)
        # Assigning a type to the variable 'self' (line 3)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FooParent.method.__dict__.__setitem__('stypy_localization', localization)
        FooParent.method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FooParent.method.__dict__.__setitem__('stypy_type_store', module_type_store)
        FooParent.method.__dict__.__setitem__('stypy_function_name', 'FooParent.method')
        FooParent.method.__dict__.__setitem__('stypy_param_names_list', [])
        FooParent.method.__dict__.__setitem__('stypy_varargs_param_name', None)
        FooParent.method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FooParent.method.__dict__.__setitem__('stypy_call_defaults', defaults)
        FooParent.method.__dict__.__setitem__('stypy_call_varargs', varargs)
        FooParent.method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FooParent.method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FooParent.method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'method(...)' code ##################

        
        # Getting the type of 'True' (line 3)
        True_6775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 11), 'True')
        # Testing the type of an if condition (line 3)
        if_condition_6776 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 3, 8), True_6775)
        # Assigning a type to the variable 'if_condition_6776' (line 3)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 8), 'if_condition_6776', if_condition_6776)
        # SSA begins for if statement (line 3)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_6777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 19), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 4)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 12), 'stypy_return_type', int_6777)
        # SSA branch for the else part of an if statement (line 3)
        module_type_store.open_ssa_branch('else')
        
        # Call to list(...): (line 6)
        # Processing the call keyword arguments (line 6)
        kwargs_6779 = {}
        # Getting the type of 'list' (line 6)
        list_6778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 19), 'list', False)
        # Calling list(args, kwargs) (line 6)
        list_call_result_6780 = invoke(stypy.reporting.localization.Localization(__file__, 6, 19), list_6778, *[], **kwargs_6779)
        
        # Assigning a type to the variable 'stypy_return_type' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 12), 'stypy_return_type', list_call_result_6780)
        # SSA join for if statement (line 3)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'method' in the type store
        # Getting the type of 'stypy_return_type' (line 2)
        stypy_return_type_6781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6781)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'method'
        return stypy_return_type_6781


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1, 0, False)
        # Assigning a type to the variable 'self' (line 2)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FooParent.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FooParent' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'FooParent', FooParent)
# Declaration of the 'FooChild' class
# Getting the type of 'FooParent' (line 9)
FooParent_6782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 15), 'FooParent')

class FooChild(FooParent_6782, ):

    @norecursion
    def method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'method'
        module_type_store = module_type_store.open_function_context('method', 10, 4, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FooChild.method.__dict__.__setitem__('stypy_localization', localization)
        FooChild.method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FooChild.method.__dict__.__setitem__('stypy_type_store', module_type_store)
        FooChild.method.__dict__.__setitem__('stypy_function_name', 'FooChild.method')
        FooChild.method.__dict__.__setitem__('stypy_param_names_list', [])
        FooChild.method.__dict__.__setitem__('stypy_varargs_param_name', None)
        FooChild.method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FooChild.method.__dict__.__setitem__('stypy_call_defaults', defaults)
        FooChild.method.__dict__.__setitem__('stypy_call_varargs', varargs)
        FooChild.method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FooChild.method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FooChild.method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'method(...)' code ##################

        
        # Getting the type of 'True' (line 11)
        True_6783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'True')
        # Testing the type of an if condition (line 11)
        if_condition_6784 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 11, 8), True_6783)
        # Assigning a type to the variable 'if_condition_6784' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'if_condition_6784', if_condition_6784)
        # SSA begins for if statement (line 11)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_6785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 19), 'str', 'a')
        # Assigning a type to the variable 'stypy_return_type' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'stypy_return_type', str_6785)
        # SSA branch for the else part of an if statement (line 11)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'True' (line 14)
        True_6786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), 'stypy_return_type', True_6786)
        # SSA join for if statement (line 11)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'method' in the type store
        # Getting the type of 'stypy_return_type' (line 10)
        stypy_return_type_6787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6787)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'method'
        return stypy_return_type_6787


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 9, 0, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FooChild.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FooChild' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'FooChild', FooChild)

# Getting the type of 'True' (line 17)
True_6788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 3), 'True')
# Testing the type of an if condition (line 17)
if_condition_6789 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 0), True_6788)
# Assigning a type to the variable 'if_condition_6789' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'if_condition_6789', if_condition_6789)
# SSA begins for if statement (line 17)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Name (line 18):

# Call to FooParent(...): (line 18)
# Processing the call keyword arguments (line 18)
kwargs_6791 = {}
# Getting the type of 'FooParent' (line 18)
FooParent_6790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'FooParent', False)
# Calling FooParent(args, kwargs) (line 18)
FooParent_call_result_6792 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), FooParent_6790, *[], **kwargs_6791)

# Assigning a type to the variable 'o' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'o', FooParent_call_result_6792)
# SSA branch for the else part of an if statement (line 17)
module_type_store.open_ssa_branch('else')

# Assigning a Call to a Name (line 20):

# Call to FooChild(...): (line 20)
# Processing the call keyword arguments (line 20)
kwargs_6794 = {}
# Getting the type of 'FooChild' (line 20)
FooChild_6793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'FooChild', False)
# Calling FooChild(args, kwargs) (line 20)
FooChild_call_result_6795 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), FooChild_6793, *[], **kwargs_6794)

# Assigning a type to the variable 'o' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'o', FooChild_call_result_6795)
# SSA join for if statement (line 17)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Call to a Name (line 22):

# Call to method(...): (line 22)
# Processing the call keyword arguments (line 22)
kwargs_6798 = {}
# Getting the type of 'o' (line 22)
o_6796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'o', False)
# Obtaining the member 'method' of a type (line 22)
method_6797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 4), o_6796, 'method')
# Calling method(args, kwargs) (line 22)
method_call_result_6799 = invoke(stypy.reporting.localization.Localization(__file__, 22, 4), method_6797, *[], **kwargs_6798)

# Assigning a type to the variable 'x' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'x', method_call_result_6799)

# Assigning a Call to a Name (line 23):

# Call to nothing(...): (line 23)
# Processing the call keyword arguments (line 23)
kwargs_6802 = {}
# Getting the type of 'x' (line 23)
x_6800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 5), 'x', False)
# Obtaining the member 'nothing' of a type (line 23)
nothing_6801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 5), x_6800, 'nothing')
# Calling nothing(args, kwargs) (line 23)
nothing_call_result_6803 = invoke(stypy.reporting.localization.Localization(__file__, 23, 5), nothing_6801, *[], **kwargs_6802)

# Assigning a type to the variable 'r1' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'r1', nothing_call_result_6803)

# Assigning a Call to a Name (line 24):

# Call to len(...): (line 24)
# Processing the call arguments (line 24)
# Getting the type of 'x' (line 24)
x_6805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'x', False)
# Processing the call keyword arguments (line 24)
kwargs_6806 = {}
# Getting the type of 'len' (line 24)
len_6804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'len', False)
# Calling len(args, kwargs) (line 24)
len_call_result_6807 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), len_6804, *[x_6805], **kwargs_6806)

# Assigning a type to the variable 'l' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'l', len_call_result_6807)

# Getting the type of 'True' (line 26)
True_6808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 3), 'True')
# Testing the type of an if condition (line 26)
if_condition_6809 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 0), True_6808)
# Assigning a type to the variable 'if_condition_6809' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'if_condition_6809', if_condition_6809)
# SSA begins for if statement (line 26)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Name (line 27):

# Call to FooParent(...): (line 27)
# Processing the call keyword arguments (line 27)
kwargs_6811 = {}
# Getting the type of 'FooParent' (line 27)
FooParent_6810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 9), 'FooParent', False)
# Calling FooParent(args, kwargs) (line 27)
FooParent_call_result_6812 = invoke(stypy.reporting.localization.Localization(__file__, 27, 9), FooParent_6810, *[], **kwargs_6811)

# Assigning a type to the variable 'o2' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'o2', FooParent_call_result_6812)

# Assigning a Call to a Name (line 28):

# Call to method(...): (line 28)
# Processing the call keyword arguments (line 28)
kwargs_6815 = {}
# Getting the type of 'o2' (line 28)
o2_6813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 9), 'o2', False)
# Obtaining the member 'method' of a type (line 28)
method_6814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 9), o2_6813, 'method')
# Calling method(args, kwargs) (line 28)
method_call_result_6816 = invoke(stypy.reporting.localization.Localization(__file__, 28, 9), method_6814, *[], **kwargs_6815)

# Assigning a type to the variable 'r2' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'r2', method_call_result_6816)
# SSA branch for the else part of an if statement (line 26)
module_type_store.open_ssa_branch('else')

# Assigning a Call to a Name (line 30):

# Call to FooChild(...): (line 30)
# Processing the call keyword arguments (line 30)
kwargs_6818 = {}
# Getting the type of 'FooChild' (line 30)
FooChild_6817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 9), 'FooChild', False)
# Calling FooChild(args, kwargs) (line 30)
FooChild_call_result_6819 = invoke(stypy.reporting.localization.Localization(__file__, 30, 9), FooChild_6817, *[], **kwargs_6818)

# Assigning a type to the variable 'o2' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'o2', FooChild_call_result_6819)

# Assigning a Call to a Name (line 31):

# Call to method(...): (line 31)
# Processing the call keyword arguments (line 31)
kwargs_6822 = {}
# Getting the type of 'o2' (line 31)
o2_6820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 9), 'o2', False)
# Obtaining the member 'method' of a type (line 31)
method_6821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 9), o2_6820, 'method')
# Calling method(args, kwargs) (line 31)
method_call_result_6823 = invoke(stypy.reporting.localization.Localization(__file__, 31, 9), method_6821, *[], **kwargs_6822)

# Assigning a type to the variable 'r2' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'r2', method_call_result_6823)
# SSA join for if statement (line 26)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Call to a Name (line 33):

# Call to nothing(...): (line 33)
# Processing the call keyword arguments (line 33)
kwargs_6826 = {}
# Getting the type of 'r2' (line 33)
r2_6824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 5), 'r2', False)
# Obtaining the member 'nothing' of a type (line 33)
nothing_6825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 5), r2_6824, 'nothing')
# Calling nothing(args, kwargs) (line 33)
nothing_call_result_6827 = invoke(stypy.reporting.localization.Localization(__file__, 33, 5), nothing_6825, *[], **kwargs_6826)

# Assigning a type to the variable 'r3' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'r3', nothing_call_result_6827)

# Assigning a Call to a Name (line 34):

# Call to len(...): (line 34)
# Processing the call arguments (line 34)
# Getting the type of 'x' (line 34)
x_6829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 9), 'x', False)
# Processing the call keyword arguments (line 34)
kwargs_6830 = {}
# Getting the type of 'len' (line 34)
len_6828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 5), 'len', False)
# Calling len(args, kwargs) (line 34)
len_call_result_6831 = invoke(stypy.reporting.localization.Localization(__file__, 34, 5), len_6828, *[x_6829], **kwargs_6830)

# Assigning a type to the variable 'l2' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'l2', len_call_result_6831)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
