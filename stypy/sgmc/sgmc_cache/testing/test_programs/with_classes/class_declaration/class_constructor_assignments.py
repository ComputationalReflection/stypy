
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: class Record:
2:     def __init__(self, PtrComp=None, Discr=0, EnumComp=0,
3:                  IntComp=0, StringComp=0):
4:         self.PtrComp = PtrComp
5:         self.Discr = Discr
6:         self.EnumComp = EnumComp
7:         self.IntComp = IntComp
8:         self.StringComp = StringComp
9: 
10:     def copy(self):
11:         return Record(self.PtrComp, self.Discr, self.EnumComp,
12:                       self.IntComp, self.StringComp)
13: 
14: 
15: r = Record()
16: 
17: x1 = r.PtrComp
18: x2 = r.Discr
19: x3 = r.EnumComp
20: x4 = r.IntComp
21: x5 = r.StringComp
22: 
23: r2 = r.copy()
24: 
25: y1 = r2.PtrComp
26: y2 = r2.Discr
27: y3 = r2.EnumComp
28: y4 = r2.IntComp
29: y5 = r2.StringComp

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'Record' class

class Record:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 2)
        None_1819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 31), 'None')
        int_1820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 43), 'int')
        int_1821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 55), 'int')
        int_1822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 25), 'int')
        int_1823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 39), 'int')
        defaults = [None_1819, int_1820, int_1821, int_1822, int_1823]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 2, 4, False)
        # Assigning a type to the variable 'self' (line 3)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Record.__init__', ['PtrComp', 'Discr', 'EnumComp', 'IntComp', 'StringComp'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['PtrComp', 'Discr', 'EnumComp', 'IntComp', 'StringComp'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 4):
        # Getting the type of 'PtrComp' (line 4)
        PtrComp_1824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 23), 'PtrComp')
        # Getting the type of 'self' (line 4)
        self_1825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 8), 'self')
        # Setting the type of the member 'PtrComp' of a type (line 4)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 8), self_1825, 'PtrComp', PtrComp_1824)
        
        # Assigning a Name to a Attribute (line 5):
        # Getting the type of 'Discr' (line 5)
        Discr_1826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 21), 'Discr')
        # Getting the type of 'self' (line 5)
        self_1827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 8), 'self')
        # Setting the type of the member 'Discr' of a type (line 5)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 8), self_1827, 'Discr', Discr_1826)
        
        # Assigning a Name to a Attribute (line 6):
        # Getting the type of 'EnumComp' (line 6)
        EnumComp_1828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 24), 'EnumComp')
        # Getting the type of 'self' (line 6)
        self_1829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'self')
        # Setting the type of the member 'EnumComp' of a type (line 6)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 8), self_1829, 'EnumComp', EnumComp_1828)
        
        # Assigning a Name to a Attribute (line 7):
        # Getting the type of 'IntComp' (line 7)
        IntComp_1830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 23), 'IntComp')
        # Getting the type of 'self' (line 7)
        self_1831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self')
        # Setting the type of the member 'IntComp' of a type (line 7)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 8), self_1831, 'IntComp', IntComp_1830)
        
        # Assigning a Name to a Attribute (line 8):
        # Getting the type of 'StringComp' (line 8)
        StringComp_1832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 26), 'StringComp')
        # Getting the type of 'self' (line 8)
        self_1833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'self')
        # Setting the type of the member 'StringComp' of a type (line 8)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 8), self_1833, 'StringComp', StringComp_1832)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def copy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'copy'
        module_type_store = module_type_store.open_function_context('copy', 10, 4, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Record.copy.__dict__.__setitem__('stypy_localization', localization)
        Record.copy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Record.copy.__dict__.__setitem__('stypy_type_store', module_type_store)
        Record.copy.__dict__.__setitem__('stypy_function_name', 'Record.copy')
        Record.copy.__dict__.__setitem__('stypy_param_names_list', [])
        Record.copy.__dict__.__setitem__('stypy_varargs_param_name', None)
        Record.copy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Record.copy.__dict__.__setitem__('stypy_call_defaults', defaults)
        Record.copy.__dict__.__setitem__('stypy_call_varargs', varargs)
        Record.copy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Record.copy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Record.copy', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'copy', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'copy(...)' code ##################

        
        # Call to Record(...): (line 11)
        # Processing the call arguments (line 11)
        # Getting the type of 'self' (line 11)
        self_1835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 22), 'self', False)
        # Obtaining the member 'PtrComp' of a type (line 11)
        PtrComp_1836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 22), self_1835, 'PtrComp')
        # Getting the type of 'self' (line 11)
        self_1837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 36), 'self', False)
        # Obtaining the member 'Discr' of a type (line 11)
        Discr_1838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 36), self_1837, 'Discr')
        # Getting the type of 'self' (line 11)
        self_1839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 48), 'self', False)
        # Obtaining the member 'EnumComp' of a type (line 11)
        EnumComp_1840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 48), self_1839, 'EnumComp')
        # Getting the type of 'self' (line 12)
        self_1841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 22), 'self', False)
        # Obtaining the member 'IntComp' of a type (line 12)
        IntComp_1842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 22), self_1841, 'IntComp')
        # Getting the type of 'self' (line 12)
        self_1843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 36), 'self', False)
        # Obtaining the member 'StringComp' of a type (line 12)
        StringComp_1844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 36), self_1843, 'StringComp')
        # Processing the call keyword arguments (line 11)
        kwargs_1845 = {}
        # Getting the type of 'Record' (line 11)
        Record_1834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 15), 'Record', False)
        # Calling Record(args, kwargs) (line 11)
        Record_call_result_1846 = invoke(stypy.reporting.localization.Localization(__file__, 11, 15), Record_1834, *[PtrComp_1836, Discr_1838, EnumComp_1840, IntComp_1842, StringComp_1844], **kwargs_1845)
        
        # Assigning a type to the variable 'stypy_return_type' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type', Record_call_result_1846)
        
        # ################# End of 'copy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy' in the type store
        # Getting the type of 'stypy_return_type' (line 10)
        stypy_return_type_1847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1847)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy'
        return stypy_return_type_1847


# Assigning a type to the variable 'Record' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'Record', Record)

# Assigning a Call to a Name (line 15):

# Call to Record(...): (line 15)
# Processing the call keyword arguments (line 15)
kwargs_1849 = {}
# Getting the type of 'Record' (line 15)
Record_1848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'Record', False)
# Calling Record(args, kwargs) (line 15)
Record_call_result_1850 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), Record_1848, *[], **kwargs_1849)

# Assigning a type to the variable 'r' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'r', Record_call_result_1850)

# Assigning a Attribute to a Name (line 17):
# Getting the type of 'r' (line 17)
r_1851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'r')
# Obtaining the member 'PtrComp' of a type (line 17)
PtrComp_1852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), r_1851, 'PtrComp')
# Assigning a type to the variable 'x1' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'x1', PtrComp_1852)

# Assigning a Attribute to a Name (line 18):
# Getting the type of 'r' (line 18)
r_1853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 5), 'r')
# Obtaining the member 'Discr' of a type (line 18)
Discr_1854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 5), r_1853, 'Discr')
# Assigning a type to the variable 'x2' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'x2', Discr_1854)

# Assigning a Attribute to a Name (line 19):
# Getting the type of 'r' (line 19)
r_1855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'r')
# Obtaining the member 'EnumComp' of a type (line 19)
EnumComp_1856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 5), r_1855, 'EnumComp')
# Assigning a type to the variable 'x3' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'x3', EnumComp_1856)

# Assigning a Attribute to a Name (line 20):
# Getting the type of 'r' (line 20)
r_1857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 5), 'r')
# Obtaining the member 'IntComp' of a type (line 20)
IntComp_1858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 5), r_1857, 'IntComp')
# Assigning a type to the variable 'x4' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'x4', IntComp_1858)

# Assigning a Attribute to a Name (line 21):
# Getting the type of 'r' (line 21)
r_1859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 5), 'r')
# Obtaining the member 'StringComp' of a type (line 21)
StringComp_1860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 5), r_1859, 'StringComp')
# Assigning a type to the variable 'x5' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'x5', StringComp_1860)

# Assigning a Call to a Name (line 23):

# Call to copy(...): (line 23)
# Processing the call keyword arguments (line 23)
kwargs_1863 = {}
# Getting the type of 'r' (line 23)
r_1861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 5), 'r', False)
# Obtaining the member 'copy' of a type (line 23)
copy_1862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 5), r_1861, 'copy')
# Calling copy(args, kwargs) (line 23)
copy_call_result_1864 = invoke(stypy.reporting.localization.Localization(__file__, 23, 5), copy_1862, *[], **kwargs_1863)

# Assigning a type to the variable 'r2' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'r2', copy_call_result_1864)

# Assigning a Attribute to a Name (line 25):
# Getting the type of 'r2' (line 25)
r2_1865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 5), 'r2')
# Obtaining the member 'PtrComp' of a type (line 25)
PtrComp_1866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 5), r2_1865, 'PtrComp')
# Assigning a type to the variable 'y1' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'y1', PtrComp_1866)

# Assigning a Attribute to a Name (line 26):
# Getting the type of 'r2' (line 26)
r2_1867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 5), 'r2')
# Obtaining the member 'Discr' of a type (line 26)
Discr_1868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 5), r2_1867, 'Discr')
# Assigning a type to the variable 'y2' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'y2', Discr_1868)

# Assigning a Attribute to a Name (line 27):
# Getting the type of 'r2' (line 27)
r2_1869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 5), 'r2')
# Obtaining the member 'EnumComp' of a type (line 27)
EnumComp_1870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 5), r2_1869, 'EnumComp')
# Assigning a type to the variable 'y3' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'y3', EnumComp_1870)

# Assigning a Attribute to a Name (line 28):
# Getting the type of 'r2' (line 28)
r2_1871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 5), 'r2')
# Obtaining the member 'IntComp' of a type (line 28)
IntComp_1872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 5), r2_1871, 'IntComp')
# Assigning a type to the variable 'y4' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'y4', IntComp_1872)

# Assigning a Attribute to a Name (line 29):
# Getting the type of 'r2' (line 29)
r2_1873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 5), 'r2')
# Obtaining the member 'StringComp' of a type (line 29)
StringComp_1874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 5), r2_1873, 'StringComp')
# Assigning a type to the variable 'y5' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'y5', StringComp_1874)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
