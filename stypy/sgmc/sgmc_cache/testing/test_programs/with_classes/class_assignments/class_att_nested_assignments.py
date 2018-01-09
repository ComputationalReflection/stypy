
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: class Inner:
2:     attInner = 3
3: 
4: 
5: class LessInner:
6:     attLessInner = Inner()
7: 
8: 
9: class Outer:
10:     attOuter = LessInner()
11: 
12: 
13: i1 = Inner()
14: r1 = i1.attInner
15: 
16: i2 = LessInner()
17: r2 = i2.attLessInner.attInner
18: 
19: i2.attLessInner.attInner = "3"
20: i2.attLessInner.attInner += "3"
21: r3 = i2.attLessInner.attInner

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'Inner' class

class Inner:
    pass

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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Inner.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Inner' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'Inner', Inner)

# Assigning a Num to a Name (line 2):
int_1785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 15), 'int')
# Getting the type of 'Inner'
Inner_1786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Inner')
# Setting the type of the member 'attInner' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Inner_1786, 'attInner', int_1785)
# Declaration of the 'LessInner' class

class LessInner:
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 5, 0, False)
        # Assigning a type to the variable 'self' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LessInner.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'LessInner' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'LessInner', LessInner)

# Assigning a Call to a Name (line 6):

# Call to Inner(...): (line 6)
# Processing the call keyword arguments (line 6)
kwargs_1788 = {}
# Getting the type of 'Inner' (line 6)
Inner_1787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 19), 'Inner', False)
# Calling Inner(args, kwargs) (line 6)
Inner_call_result_1789 = invoke(stypy.reporting.localization.Localization(__file__, 6, 19), Inner_1787, *[], **kwargs_1788)

# Getting the type of 'LessInner'
LessInner_1790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LessInner')
# Setting the type of the member 'attLessInner' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LessInner_1790, 'attLessInner', Inner_call_result_1789)
# Declaration of the 'Outer' class

class Outer:
    pass

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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Outer.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Outer' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'Outer', Outer)

# Assigning a Call to a Name (line 10):

# Call to LessInner(...): (line 10)
# Processing the call keyword arguments (line 10)
kwargs_1792 = {}
# Getting the type of 'LessInner' (line 10)
LessInner_1791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 15), 'LessInner', False)
# Calling LessInner(args, kwargs) (line 10)
LessInner_call_result_1793 = invoke(stypy.reporting.localization.Localization(__file__, 10, 15), LessInner_1791, *[], **kwargs_1792)

# Getting the type of 'Outer'
Outer_1794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Outer')
# Setting the type of the member 'attOuter' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Outer_1794, 'attOuter', LessInner_call_result_1793)

# Assigning a Call to a Name (line 13):

# Call to Inner(...): (line 13)
# Processing the call keyword arguments (line 13)
kwargs_1796 = {}
# Getting the type of 'Inner' (line 13)
Inner_1795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'Inner', False)
# Calling Inner(args, kwargs) (line 13)
Inner_call_result_1797 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), Inner_1795, *[], **kwargs_1796)

# Assigning a type to the variable 'i1' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'i1', Inner_call_result_1797)

# Assigning a Attribute to a Name (line 14):
# Getting the type of 'i1' (line 14)
i1_1798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 5), 'i1')
# Obtaining the member 'attInner' of a type (line 14)
attInner_1799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 5), i1_1798, 'attInner')
# Assigning a type to the variable 'r1' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'r1', attInner_1799)

# Assigning a Call to a Name (line 16):

# Call to LessInner(...): (line 16)
# Processing the call keyword arguments (line 16)
kwargs_1801 = {}
# Getting the type of 'LessInner' (line 16)
LessInner_1800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 5), 'LessInner', False)
# Calling LessInner(args, kwargs) (line 16)
LessInner_call_result_1802 = invoke(stypy.reporting.localization.Localization(__file__, 16, 5), LessInner_1800, *[], **kwargs_1801)

# Assigning a type to the variable 'i2' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'i2', LessInner_call_result_1802)

# Assigning a Attribute to a Name (line 17):
# Getting the type of 'i2' (line 17)
i2_1803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'i2')
# Obtaining the member 'attLessInner' of a type (line 17)
attLessInner_1804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), i2_1803, 'attLessInner')
# Obtaining the member 'attInner' of a type (line 17)
attInner_1805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), attLessInner_1804, 'attInner')
# Assigning a type to the variable 'r2' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r2', attInner_1805)

# Assigning a Str to a Attribute (line 19):
str_1806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 27), 'str', '3')
# Getting the type of 'i2' (line 19)
i2_1807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'i2')
# Obtaining the member 'attLessInner' of a type (line 19)
attLessInner_1808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 0), i2_1807, 'attLessInner')
# Setting the type of the member 'attInner' of a type (line 19)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 0), attLessInner_1808, 'attInner', str_1806)

# Getting the type of 'i2' (line 20)
i2_1809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'i2')
# Obtaining the member 'attLessInner' of a type (line 20)
attLessInner_1810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 0), i2_1809, 'attLessInner')
# Obtaining the member 'attInner' of a type (line 20)
attInner_1811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 0), attLessInner_1810, 'attInner')
str_1812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 28), 'str', '3')
# Applying the binary operator '+=' (line 20)
result_iadd_1813 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 0), '+=', attInner_1811, str_1812)
# Getting the type of 'i2' (line 20)
i2_1814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'i2')
# Obtaining the member 'attLessInner' of a type (line 20)
attLessInner_1815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 0), i2_1814, 'attLessInner')
# Setting the type of the member 'attInner' of a type (line 20)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 0), attLessInner_1815, 'attInner', result_iadd_1813)


# Assigning a Attribute to a Name (line 21):
# Getting the type of 'i2' (line 21)
i2_1816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 5), 'i2')
# Obtaining the member 'attLessInner' of a type (line 21)
attLessInner_1817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 5), i2_1816, 'attLessInner')
# Obtaining the member 'attInner' of a type (line 21)
attInner_1818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 5), attLessInner_1817, 'attInner')
# Assigning a type to the variable 'r3' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'r3', attInner_1818)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
