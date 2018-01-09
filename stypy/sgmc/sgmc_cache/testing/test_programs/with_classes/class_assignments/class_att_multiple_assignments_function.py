
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: mapper = [(1,2,3), ('a', 'b', 'c'), (1.1, 2.2, 3.3)]
3: 
4: class Test:
5:     def my_zip(self, list_):
6:         return zip(list_)
7: 
8: class C:
9:     (m, n, o) = zip(*mapper)
10:     (p, q, r) = zip(Test().my_zip(mapper))
11: 
12: c = C()
13: 
14: r = c.m
15: r2 = c.n
16: r3 = c.o
17: 
18: r4 = c.p
19: r5 = c.q
20: r6 = c.r
21: 
22: 
23: 
24: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 2):

# Assigning a List to a Name (line 2):

# Obtaining an instance of the builtin type 'list' (line 2)
list_1668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 9), 'list')
# Adding type elements to the builtin type 'list' instance (line 2)
# Adding element type (line 2)

# Obtaining an instance of the builtin type 'tuple' (line 2)
tuple_1669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2)
# Adding element type (line 2)
int_1670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 11), tuple_1669, int_1670)
# Adding element type (line 2)
int_1671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 11), tuple_1669, int_1671)
# Adding element type (line 2)
int_1672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 11), tuple_1669, int_1672)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 9), list_1668, tuple_1669)
# Adding element type (line 2)

# Obtaining an instance of the builtin type 'tuple' (line 2)
tuple_1673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2)
# Adding element type (line 2)
str_1674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 20), 'str', 'a')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 20), tuple_1673, str_1674)
# Adding element type (line 2)
str_1675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 25), 'str', 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 20), tuple_1673, str_1675)
# Adding element type (line 2)
str_1676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 30), 'str', 'c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 20), tuple_1673, str_1676)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 9), list_1668, tuple_1673)
# Adding element type (line 2)

# Obtaining an instance of the builtin type 'tuple' (line 2)
tuple_1677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 37), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2)
# Adding element type (line 2)
float_1678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 37), tuple_1677, float_1678)
# Adding element type (line 2)
float_1679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 42), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 37), tuple_1677, float_1679)
# Adding element type (line 2)
float_1680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 47), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 37), tuple_1677, float_1680)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 9), list_1668, tuple_1677)

# Assigning a type to the variable 'mapper' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'mapper', list_1668)
# Declaration of the 'Test' class

class Test:

    @norecursion
    def my_zip(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'my_zip'
        module_type_store = module_type_store.open_function_context('my_zip', 5, 4, False)
        # Assigning a type to the variable 'self' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test.my_zip.__dict__.__setitem__('stypy_localization', localization)
        Test.my_zip.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test.my_zip.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test.my_zip.__dict__.__setitem__('stypy_function_name', 'Test.my_zip')
        Test.my_zip.__dict__.__setitem__('stypy_param_names_list', ['list_'])
        Test.my_zip.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test.my_zip.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test.my_zip.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test.my_zip.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test.my_zip.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test.my_zip.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.my_zip', ['list_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'my_zip', localization, ['list_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'my_zip(...)' code ##################

        
        # Call to zip(...): (line 6)
        # Processing the call arguments (line 6)
        # Getting the type of 'list_' (line 6)
        list__1682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 19), 'list_', False)
        # Processing the call keyword arguments (line 6)
        kwargs_1683 = {}
        # Getting the type of 'zip' (line 6)
        zip_1681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 15), 'zip', False)
        # Calling zip(args, kwargs) (line 6)
        zip_call_result_1684 = invoke(stypy.reporting.localization.Localization(__file__, 6, 15), zip_1681, *[list__1682], **kwargs_1683)
        
        # Assigning a type to the variable 'stypy_return_type' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type', zip_call_result_1684)
        
        # ################# End of 'my_zip(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'my_zip' in the type store
        # Getting the type of 'stypy_return_type' (line 5)
        stypy_return_type_1685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1685)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'my_zip'
        return stypy_return_type_1685


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 4, 0, False)
        # Assigning a type to the variable 'self' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Test' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'Test', Test)
# Declaration of the 'C' class

class C:
    
    # Assigning a Call to a Tuple (line 9):
    
    # Assigning a Call to a Tuple (line 10):
    pass

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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'C.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'C' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'C', C)

# Assigning a Subscript to a Name (line 9):

# Obtaining the type of the subscript
int_1686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 4), 'int')

# Call to zip(...): (line 9)
# Getting the type of 'mapper' (line 9)
mapper_1688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 21), 'mapper', False)
# Processing the call keyword arguments (line 9)
kwargs_1689 = {}
# Getting the type of 'zip' (line 9)
zip_1687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 16), 'zip', False)
# Calling zip(args, kwargs) (line 9)
zip_call_result_1690 = invoke(stypy.reporting.localization.Localization(__file__, 9, 16), zip_1687, *[mapper_1688], **kwargs_1689)

# Obtaining the member '__getitem__' of a type (line 9)
getitem___1691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), zip_call_result_1690, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 9)
subscript_call_result_1692 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), getitem___1691, int_1686)

# Getting the type of 'C'
C_1693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_var_assignment_1662' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1693, 'tuple_var_assignment_1662', subscript_call_result_1692)

# Assigning a Subscript to a Name (line 9):

# Obtaining the type of the subscript
int_1694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 4), 'int')

# Call to zip(...): (line 9)
# Getting the type of 'mapper' (line 9)
mapper_1696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 21), 'mapper', False)
# Processing the call keyword arguments (line 9)
kwargs_1697 = {}
# Getting the type of 'zip' (line 9)
zip_1695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 16), 'zip', False)
# Calling zip(args, kwargs) (line 9)
zip_call_result_1698 = invoke(stypy.reporting.localization.Localization(__file__, 9, 16), zip_1695, *[mapper_1696], **kwargs_1697)

# Obtaining the member '__getitem__' of a type (line 9)
getitem___1699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), zip_call_result_1698, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 9)
subscript_call_result_1700 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), getitem___1699, int_1694)

# Getting the type of 'C'
C_1701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_var_assignment_1663' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1701, 'tuple_var_assignment_1663', subscript_call_result_1700)

# Assigning a Subscript to a Name (line 9):

# Obtaining the type of the subscript
int_1702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 4), 'int')

# Call to zip(...): (line 9)
# Getting the type of 'mapper' (line 9)
mapper_1704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 21), 'mapper', False)
# Processing the call keyword arguments (line 9)
kwargs_1705 = {}
# Getting the type of 'zip' (line 9)
zip_1703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 16), 'zip', False)
# Calling zip(args, kwargs) (line 9)
zip_call_result_1706 = invoke(stypy.reporting.localization.Localization(__file__, 9, 16), zip_1703, *[mapper_1704], **kwargs_1705)

# Obtaining the member '__getitem__' of a type (line 9)
getitem___1707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), zip_call_result_1706, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 9)
subscript_call_result_1708 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), getitem___1707, int_1702)

# Getting the type of 'C'
C_1709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_var_assignment_1664' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1709, 'tuple_var_assignment_1664', subscript_call_result_1708)

# Assigning a Name to a Name (line 9):
# Getting the type of 'C'
C_1710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_var_assignment_1662' of a type
tuple_var_assignment_1662_1711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1710, 'tuple_var_assignment_1662')
# Getting the type of 'C'
C_1712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'm' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1712, 'm', tuple_var_assignment_1662_1711)

# Assigning a Name to a Name (line 9):
# Getting the type of 'C'
C_1713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_var_assignment_1663' of a type
tuple_var_assignment_1663_1714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1713, 'tuple_var_assignment_1663')
# Getting the type of 'C'
C_1715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'n' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1715, 'n', tuple_var_assignment_1663_1714)

# Assigning a Name to a Name (line 9):
# Getting the type of 'C'
C_1716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_var_assignment_1664' of a type
tuple_var_assignment_1664_1717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1716, 'tuple_var_assignment_1664')
# Getting the type of 'C'
C_1718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'o' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1718, 'o', tuple_var_assignment_1664_1717)

# Assigning a Subscript to a Name (line 10):

# Obtaining the type of the subscript
int_1719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 4), 'int')

# Call to zip(...): (line 10)
# Processing the call arguments (line 10)

# Call to my_zip(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'mapper' (line 10)
mapper_1725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 34), 'mapper', False)
# Processing the call keyword arguments (line 10)
kwargs_1726 = {}

# Call to Test(...): (line 10)
# Processing the call keyword arguments (line 10)
kwargs_1722 = {}
# Getting the type of 'Test' (line 10)
Test_1721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 20), 'Test', False)
# Calling Test(args, kwargs) (line 10)
Test_call_result_1723 = invoke(stypy.reporting.localization.Localization(__file__, 10, 20), Test_1721, *[], **kwargs_1722)

# Obtaining the member 'my_zip' of a type (line 10)
my_zip_1724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 20), Test_call_result_1723, 'my_zip')
# Calling my_zip(args, kwargs) (line 10)
my_zip_call_result_1727 = invoke(stypy.reporting.localization.Localization(__file__, 10, 20), my_zip_1724, *[mapper_1725], **kwargs_1726)

# Processing the call keyword arguments (line 10)
kwargs_1728 = {}
# Getting the type of 'zip' (line 10)
zip_1720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 16), 'zip', False)
# Calling zip(args, kwargs) (line 10)
zip_call_result_1729 = invoke(stypy.reporting.localization.Localization(__file__, 10, 16), zip_1720, *[my_zip_call_result_1727], **kwargs_1728)

# Obtaining the member '__getitem__' of a type (line 10)
getitem___1730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), zip_call_result_1729, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 10)
subscript_call_result_1731 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), getitem___1730, int_1719)

# Getting the type of 'C'
C_1732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_var_assignment_1665' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1732, 'tuple_var_assignment_1665', subscript_call_result_1731)

# Assigning a Subscript to a Name (line 10):

# Obtaining the type of the subscript
int_1733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 4), 'int')

# Call to zip(...): (line 10)
# Processing the call arguments (line 10)

# Call to my_zip(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'mapper' (line 10)
mapper_1739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 34), 'mapper', False)
# Processing the call keyword arguments (line 10)
kwargs_1740 = {}

# Call to Test(...): (line 10)
# Processing the call keyword arguments (line 10)
kwargs_1736 = {}
# Getting the type of 'Test' (line 10)
Test_1735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 20), 'Test', False)
# Calling Test(args, kwargs) (line 10)
Test_call_result_1737 = invoke(stypy.reporting.localization.Localization(__file__, 10, 20), Test_1735, *[], **kwargs_1736)

# Obtaining the member 'my_zip' of a type (line 10)
my_zip_1738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 20), Test_call_result_1737, 'my_zip')
# Calling my_zip(args, kwargs) (line 10)
my_zip_call_result_1741 = invoke(stypy.reporting.localization.Localization(__file__, 10, 20), my_zip_1738, *[mapper_1739], **kwargs_1740)

# Processing the call keyword arguments (line 10)
kwargs_1742 = {}
# Getting the type of 'zip' (line 10)
zip_1734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 16), 'zip', False)
# Calling zip(args, kwargs) (line 10)
zip_call_result_1743 = invoke(stypy.reporting.localization.Localization(__file__, 10, 16), zip_1734, *[my_zip_call_result_1741], **kwargs_1742)

# Obtaining the member '__getitem__' of a type (line 10)
getitem___1744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), zip_call_result_1743, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 10)
subscript_call_result_1745 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), getitem___1744, int_1733)

# Getting the type of 'C'
C_1746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_var_assignment_1666' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1746, 'tuple_var_assignment_1666', subscript_call_result_1745)

# Assigning a Subscript to a Name (line 10):

# Obtaining the type of the subscript
int_1747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 4), 'int')

# Call to zip(...): (line 10)
# Processing the call arguments (line 10)

# Call to my_zip(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'mapper' (line 10)
mapper_1753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 34), 'mapper', False)
# Processing the call keyword arguments (line 10)
kwargs_1754 = {}

# Call to Test(...): (line 10)
# Processing the call keyword arguments (line 10)
kwargs_1750 = {}
# Getting the type of 'Test' (line 10)
Test_1749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 20), 'Test', False)
# Calling Test(args, kwargs) (line 10)
Test_call_result_1751 = invoke(stypy.reporting.localization.Localization(__file__, 10, 20), Test_1749, *[], **kwargs_1750)

# Obtaining the member 'my_zip' of a type (line 10)
my_zip_1752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 20), Test_call_result_1751, 'my_zip')
# Calling my_zip(args, kwargs) (line 10)
my_zip_call_result_1755 = invoke(stypy.reporting.localization.Localization(__file__, 10, 20), my_zip_1752, *[mapper_1753], **kwargs_1754)

# Processing the call keyword arguments (line 10)
kwargs_1756 = {}
# Getting the type of 'zip' (line 10)
zip_1748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 16), 'zip', False)
# Calling zip(args, kwargs) (line 10)
zip_call_result_1757 = invoke(stypy.reporting.localization.Localization(__file__, 10, 16), zip_1748, *[my_zip_call_result_1755], **kwargs_1756)

# Obtaining the member '__getitem__' of a type (line 10)
getitem___1758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), zip_call_result_1757, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 10)
subscript_call_result_1759 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), getitem___1758, int_1747)

# Getting the type of 'C'
C_1760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'tuple_var_assignment_1667' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1760, 'tuple_var_assignment_1667', subscript_call_result_1759)

# Assigning a Name to a Name (line 10):
# Getting the type of 'C'
C_1761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_var_assignment_1665' of a type
tuple_var_assignment_1665_1762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1761, 'tuple_var_assignment_1665')
# Getting the type of 'C'
C_1763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'p' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1763, 'p', tuple_var_assignment_1665_1762)

# Assigning a Name to a Name (line 10):
# Getting the type of 'C'
C_1764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_var_assignment_1666' of a type
tuple_var_assignment_1666_1765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1764, 'tuple_var_assignment_1666')
# Getting the type of 'C'
C_1766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'q' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1766, 'q', tuple_var_assignment_1666_1765)

# Assigning a Name to a Name (line 10):
# Getting the type of 'C'
C_1767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Obtaining the member 'tuple_var_assignment_1667' of a type
tuple_var_assignment_1667_1768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1767, 'tuple_var_assignment_1667')
# Getting the type of 'C'
C_1769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'C')
# Setting the type of the member 'r' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), C_1769, 'r', tuple_var_assignment_1667_1768)

# Assigning a Call to a Name (line 12):

# Assigning a Call to a Name (line 12):

# Call to C(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_1771 = {}
# Getting the type of 'C' (line 12)
C_1770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'C', False)
# Calling C(args, kwargs) (line 12)
C_call_result_1772 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), C_1770, *[], **kwargs_1771)

# Assigning a type to the variable 'c' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'c', C_call_result_1772)

# Assigning a Attribute to a Name (line 14):

# Assigning a Attribute to a Name (line 14):
# Getting the type of 'c' (line 14)
c_1773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'c')
# Obtaining the member 'm' of a type (line 14)
m_1774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), c_1773, 'm')
# Assigning a type to the variable 'r' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'r', m_1774)

# Assigning a Attribute to a Name (line 15):

# Assigning a Attribute to a Name (line 15):
# Getting the type of 'c' (line 15)
c_1775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'c')
# Obtaining the member 'n' of a type (line 15)
n_1776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 5), c_1775, 'n')
# Assigning a type to the variable 'r2' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'r2', n_1776)

# Assigning a Attribute to a Name (line 16):

# Assigning a Attribute to a Name (line 16):
# Getting the type of 'c' (line 16)
c_1777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 5), 'c')
# Obtaining the member 'o' of a type (line 16)
o_1778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 5), c_1777, 'o')
# Assigning a type to the variable 'r3' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'r3', o_1778)

# Assigning a Attribute to a Name (line 18):

# Assigning a Attribute to a Name (line 18):
# Getting the type of 'c' (line 18)
c_1779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 5), 'c')
# Obtaining the member 'p' of a type (line 18)
p_1780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 5), c_1779, 'p')
# Assigning a type to the variable 'r4' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'r4', p_1780)

# Assigning a Attribute to a Name (line 19):

# Assigning a Attribute to a Name (line 19):
# Getting the type of 'c' (line 19)
c_1781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'c')
# Obtaining the member 'q' of a type (line 19)
q_1782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 5), c_1781, 'q')
# Assigning a type to the variable 'r5' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'r5', q_1782)

# Assigning a Attribute to a Name (line 20):

# Assigning a Attribute to a Name (line 20):
# Getting the type of 'c' (line 20)
c_1783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 5), 'c')
# Obtaining the member 'r' of a type (line 20)
r_1784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 5), c_1783, 'r')
# Assigning a type to the variable 'r6' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'r6', r_1784)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
