
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: 
6: class NamedArray(np.ndarray):
7:     def __new__(cls, array, name="no name"):
8:         obj = np.asarray(array).view(cls)
9:         obj.name = name
10:         return obj
11: 
12:     def __array_finalize__(self, obj):
13:         if obj is None: return
14:         self.info = getattr(obj, 'name', "no name")
15: 
16: 
17: Z = NamedArray(np.arange(10), "range_10")
18: r = (Z.name)
19: 
20: # l = globals().copy()
21: # for v in l:
22: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
23: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_950 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_950) is not StypyTypeError):

    if (import_950 != 'pyd_module'):
        __import__(import_950)
        sys_modules_951 = sys.modules[import_950]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_951.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_950)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')

# Declaration of the 'NamedArray' class
# Getting the type of 'np' (line 6)
np_952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 17), 'np')
# Obtaining the member 'ndarray' of a type (line 6)
ndarray_953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 17), np_952, 'ndarray')

import numpy
#class NamedArray(ndarray_953, ):
class NamedArray(numpy.ndarray):

    @norecursion
    def __new__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 33), 'str', 'no name')
        defaults = [str_954]
        # Create a new context for function '__new__'
        module_type_store = module_type_store.open_function_context('__new__', 7, 4, False)
        # Assigning a type to the variable 'self' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NamedArray.__new__.__dict__.__setitem__('stypy_localization', localization)
        NamedArray.__new__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NamedArray.__new__.__dict__.__setitem__('stypy_type_store', module_type_store)
        NamedArray.__new__.__dict__.__setitem__('stypy_function_name', 'NamedArray.__new__')
        NamedArray.__new__.__dict__.__setitem__('stypy_param_names_list', ['array', 'name'])
        NamedArray.__new__.__dict__.__setitem__('stypy_varargs_param_name', None)
        NamedArray.__new__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NamedArray.__new__.__dict__.__setitem__('stypy_call_defaults', defaults)
        NamedArray.__new__.__dict__.__setitem__('stypy_call_varargs', varargs)
        NamedArray.__new__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NamedArray.__new__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NamedArray.__new__', ['array', 'name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__new__', localization, ['array', 'name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__new__(...)' code ##################

        
        # Assigning a Call to a Name (line 8):
        
        # Call to view(...): (line 8)
        # Processing the call arguments (line 8)
        # Getting the type of 'cls' (line 8)
        cls_961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 37), 'cls', False)
        # Processing the call keyword arguments (line 8)
        kwargs_962 = {}
        
        # Call to asarray(...): (line 8)
        # Processing the call arguments (line 8)
        # Getting the type of 'array' (line 8)
        array_957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 25), 'array', False)
        # Processing the call keyword arguments (line 8)
        kwargs_958 = {}
        # Getting the type of 'np' (line 8)
        np_955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 14), 'np', False)
        # Obtaining the member 'asarray' of a type (line 8)
        asarray_956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 14), np_955, 'asarray')
        # Calling asarray(args, kwargs) (line 8)
        asarray_call_result_959 = invoke(stypy.reporting.localization.Localization(__file__, 8, 14), asarray_956, *[array_957], **kwargs_958)
        
        # Obtaining the member 'view' of a type (line 8)
        view_960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 14), asarray_call_result_959, 'view')
        # Calling view(args, kwargs) (line 8)
        view_call_result_963 = invoke(stypy.reporting.localization.Localization(__file__, 8, 14), view_960, *[cls_961], **kwargs_962)
        
        # Assigning a type to the variable 'obj' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'obj', view_call_result_963)
        
        # Assigning a Name to a Attribute (line 9):
        # Getting the type of 'name' (line 9)
        name_964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 19), 'name')
        # Getting the type of 'obj' (line 9)
        obj_965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'obj')
        # Setting the type of the member 'name' of a type (line 9)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 8), obj_965, 'name', name_964)
        # Getting the type of 'obj' (line 10)
        obj_966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 15), 'obj')
        # Assigning a type to the variable 'stypy_return_type' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'stypy_return_type', obj_966)
        
        # ################# End of '__new__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__new__' in the type store
        # Getting the type of 'stypy_return_type' (line 7)
        stypy_return_type_967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_967)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__new__'
        return stypy_return_type_967


    @norecursion
    def __array_finalize__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__array_finalize__'
        module_type_store = module_type_store.open_function_context('__array_finalize__', 12, 4, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NamedArray.__array_finalize__.__dict__.__setitem__('stypy_localization', localization)
        NamedArray.__array_finalize__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NamedArray.__array_finalize__.__dict__.__setitem__('stypy_type_store', module_type_store)
        NamedArray.__array_finalize__.__dict__.__setitem__('stypy_function_name', 'NamedArray.__array_finalize__')
        NamedArray.__array_finalize__.__dict__.__setitem__('stypy_param_names_list', ['obj'])
        NamedArray.__array_finalize__.__dict__.__setitem__('stypy_varargs_param_name', None)
        NamedArray.__array_finalize__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NamedArray.__array_finalize__.__dict__.__setitem__('stypy_call_defaults', defaults)
        NamedArray.__array_finalize__.__dict__.__setitem__('stypy_call_varargs', varargs)
        NamedArray.__array_finalize__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NamedArray.__array_finalize__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NamedArray.__array_finalize__', ['obj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__array_finalize__', localization, ['obj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__array_finalize__(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 13)
        # Getting the type of 'obj' (line 13)
        obj_968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 'obj')
        # Getting the type of 'None' (line 13)
        None_969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 18), 'None')
        
        (may_be_970, more_types_in_union_971) = may_be_none(obj_968, None_969)

        if may_be_970:

            if more_types_in_union_971:
                # Runtime conditional SSA (line 13)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 13)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 24), 'stypy_return_type', types.NoneType)

            if more_types_in_union_971:
                # SSA join for if statement (line 13)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Attribute (line 14):
        
        # Call to getattr(...): (line 14)
        # Processing the call arguments (line 14)
        # Getting the type of 'obj' (line 14)
        obj_973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 28), 'obj', False)
        str_974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 33), 'str', 'name')
        str_975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 41), 'str', 'no name')
        # Processing the call keyword arguments (line 14)
        kwargs_976 = {}
        # Getting the type of 'getattr' (line 14)
        getattr_972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 14)
        getattr_call_result_977 = invoke(stypy.reporting.localization.Localization(__file__, 14, 20), getattr_972, *[obj_973, str_974, str_975], **kwargs_976)
        
        # Getting the type of 'self' (line 14)
        self_978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'self')
        # Setting the type of the member 'info' of a type (line 14)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), self_978, 'info', getattr_call_result_977)
        
        # ################# End of '__array_finalize__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__array_finalize__' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_979)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__array_finalize__'
        return stypy_return_type_979


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NamedArray.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'NamedArray' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'NamedArray', NamedArray)

# Assigning a Call to a Name (line 17):

# Call to NamedArray(...): (line 17)
# Processing the call arguments (line 17)

# Call to arange(...): (line 17)
# Processing the call arguments (line 17)
int_983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 25), 'int')
# Processing the call keyword arguments (line 17)
kwargs_984 = {}
# Getting the type of 'np' (line 17)
np_981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'np', False)
# Obtaining the member 'arange' of a type (line 17)
arange_982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 15), np_981, 'arange')
# Calling arange(args, kwargs) (line 17)
arange_call_result_985 = invoke(stypy.reporting.localization.Localization(__file__, 17, 15), arange_982, *[int_983], **kwargs_984)

str_986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 30), 'str', 'range_10')
# Processing the call keyword arguments (line 17)
kwargs_987 = {}
# Getting the type of 'NamedArray' (line 17)
NamedArray_980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'NamedArray', False)
# Calling NamedArray(args, kwargs) (line 17)
NamedArray_call_result_988 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), NamedArray_980, *[arange_call_result_985, str_986], **kwargs_987)

# Assigning a type to the variable 'Z' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'Z', NamedArray_call_result_988)

# Assigning a Attribute to a Name (line 18):
# Getting the type of 'Z' (line 18)
Z_989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 5), 'Z')
# Obtaining the member 'name' of a type (line 18)
name_990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 5), Z_989, 'name')
# Assigning a type to the variable 'r' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'r', name_990)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
