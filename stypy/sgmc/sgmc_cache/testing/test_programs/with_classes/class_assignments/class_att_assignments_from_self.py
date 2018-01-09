
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: glob = 0
2: 
3: class _ctypes(object):
4:     att1 = glob
5:     att2 = att1
6: 
7:     def data_as(self, obj):
8:         return 0
9: 
10:     def shape_as(self, obj):
11:         return 1
12: 
13:     def strides_as(self, obj):
14:         return 2
15: 
16:     def get_data(self):
17:         return 3
18: 
19:     def get_shape(self):
20:         return 4
21: 
22:     def get_strides(self):
23:         return 5
24: 
25:     def get_as_parameter(self):
26:         return 6
27: 
28:     data = property(get_data, None, doc="c-types data")
29:     shape = property(get_shape, None, doc="c-types shape")
30:     strides = property(get_strides, None, doc="c-types strides")
31:     _as_parameter_ = property(get_as_parameter, None, doc="_as parameter_")
32: 
33: ct = _ctypes()
34: 
35: r = ct.data
36: r2 = ct.shape
37: r3 = ct.strides
38: r4 = ct.get_data
39: 
40: ra = ct.att1
41: rb = ct.att2
42: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 1):
int_1262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 7), 'int')
# Assigning a type to the variable 'glob' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'glob', int_1262)
# Declaration of the '_ctypes' class

class _ctypes(object, ):

    @norecursion
    def data_as(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'data_as'
        module_type_store = module_type_store.open_function_context('data_as', 7, 4, False)
        # Assigning a type to the variable 'self' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ctypes.data_as.__dict__.__setitem__('stypy_localization', localization)
        _ctypes.data_as.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ctypes.data_as.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ctypes.data_as.__dict__.__setitem__('stypy_function_name', '_ctypes.data_as')
        _ctypes.data_as.__dict__.__setitem__('stypy_param_names_list', ['obj'])
        _ctypes.data_as.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ctypes.data_as.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ctypes.data_as.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ctypes.data_as.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ctypes.data_as.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ctypes.data_as.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ctypes.data_as', ['obj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'data_as', localization, ['obj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'data_as(...)' code ##################

        int_1263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type', int_1263)
        
        # ################# End of 'data_as(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'data_as' in the type store
        # Getting the type of 'stypy_return_type' (line 7)
        stypy_return_type_1264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1264)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'data_as'
        return stypy_return_type_1264


    @norecursion
    def shape_as(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'shape_as'
        module_type_store = module_type_store.open_function_context('shape_as', 10, 4, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ctypes.shape_as.__dict__.__setitem__('stypy_localization', localization)
        _ctypes.shape_as.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ctypes.shape_as.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ctypes.shape_as.__dict__.__setitem__('stypy_function_name', '_ctypes.shape_as')
        _ctypes.shape_as.__dict__.__setitem__('stypy_param_names_list', ['obj'])
        _ctypes.shape_as.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ctypes.shape_as.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ctypes.shape_as.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ctypes.shape_as.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ctypes.shape_as.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ctypes.shape_as.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ctypes.shape_as', ['obj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'shape_as', localization, ['obj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'shape_as(...)' code ##################

        int_1265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type', int_1265)
        
        # ################# End of 'shape_as(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'shape_as' in the type store
        # Getting the type of 'stypy_return_type' (line 10)
        stypy_return_type_1266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1266)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'shape_as'
        return stypy_return_type_1266


    @norecursion
    def strides_as(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'strides_as'
        module_type_store = module_type_store.open_function_context('strides_as', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ctypes.strides_as.__dict__.__setitem__('stypy_localization', localization)
        _ctypes.strides_as.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ctypes.strides_as.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ctypes.strides_as.__dict__.__setitem__('stypy_function_name', '_ctypes.strides_as')
        _ctypes.strides_as.__dict__.__setitem__('stypy_param_names_list', ['obj'])
        _ctypes.strides_as.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ctypes.strides_as.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ctypes.strides_as.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ctypes.strides_as.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ctypes.strides_as.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ctypes.strides_as.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ctypes.strides_as', ['obj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'strides_as', localization, ['obj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'strides_as(...)' code ##################

        int_1267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'stypy_return_type', int_1267)
        
        # ################# End of 'strides_as(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'strides_as' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_1268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1268)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'strides_as'
        return stypy_return_type_1268


    @norecursion
    def get_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_data'
        module_type_store = module_type_store.open_function_context('get_data', 16, 4, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ctypes.get_data.__dict__.__setitem__('stypy_localization', localization)
        _ctypes.get_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ctypes.get_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ctypes.get_data.__dict__.__setitem__('stypy_function_name', '_ctypes.get_data')
        _ctypes.get_data.__dict__.__setitem__('stypy_param_names_list', [])
        _ctypes.get_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ctypes.get_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ctypes.get_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ctypes.get_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ctypes.get_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ctypes.get_data.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ctypes.get_data', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_data', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_data(...)' code ##################

        int_1269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'stypy_return_type', int_1269)
        
        # ################# End of 'get_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_data' in the type store
        # Getting the type of 'stypy_return_type' (line 16)
        stypy_return_type_1270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1270)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_data'
        return stypy_return_type_1270


    @norecursion
    def get_shape(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_shape'
        module_type_store = module_type_store.open_function_context('get_shape', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ctypes.get_shape.__dict__.__setitem__('stypy_localization', localization)
        _ctypes.get_shape.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ctypes.get_shape.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ctypes.get_shape.__dict__.__setitem__('stypy_function_name', '_ctypes.get_shape')
        _ctypes.get_shape.__dict__.__setitem__('stypy_param_names_list', [])
        _ctypes.get_shape.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ctypes.get_shape.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ctypes.get_shape.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ctypes.get_shape.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ctypes.get_shape.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ctypes.get_shape.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ctypes.get_shape', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_shape', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_shape(...)' code ##################

        int_1271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'stypy_return_type', int_1271)
        
        # ################# End of 'get_shape(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_shape' in the type store
        # Getting the type of 'stypy_return_type' (line 19)
        stypy_return_type_1272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1272)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_shape'
        return stypy_return_type_1272


    @norecursion
    def get_strides(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_strides'
        module_type_store = module_type_store.open_function_context('get_strides', 22, 4, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ctypes.get_strides.__dict__.__setitem__('stypy_localization', localization)
        _ctypes.get_strides.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ctypes.get_strides.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ctypes.get_strides.__dict__.__setitem__('stypy_function_name', '_ctypes.get_strides')
        _ctypes.get_strides.__dict__.__setitem__('stypy_param_names_list', [])
        _ctypes.get_strides.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ctypes.get_strides.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ctypes.get_strides.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ctypes.get_strides.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ctypes.get_strides.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ctypes.get_strides.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ctypes.get_strides', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_strides', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_strides(...)' code ##################

        int_1273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'stypy_return_type', int_1273)
        
        # ################# End of 'get_strides(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_strides' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_1274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1274)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_strides'
        return stypy_return_type_1274


    @norecursion
    def get_as_parameter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_as_parameter'
        module_type_store = module_type_store.open_function_context('get_as_parameter', 25, 4, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_localization', localization)
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_function_name', '_ctypes.get_as_parameter')
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_param_names_list', [])
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ctypes.get_as_parameter.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ctypes.get_as_parameter', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_as_parameter', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_as_parameter(...)' code ##################

        int_1275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'stypy_return_type', int_1275)
        
        # ################# End of 'get_as_parameter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_as_parameter' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_1276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1276)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_as_parameter'
        return stypy_return_type_1276


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 3, 0, False)
        # Assigning a type to the variable 'self' (line 4)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ctypes.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_ctypes' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '_ctypes', _ctypes)

# Assigning a Name to a Name (line 4):
# Getting the type of 'glob' (line 4)
glob_1277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 11), 'glob')
# Getting the type of '_ctypes'
_ctypes_1278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ctypes')
# Setting the type of the member 'att1' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ctypes_1278, 'att1', glob_1277)

# Assigning a Name to a Name (line 5):
# Getting the type of '_ctypes'
_ctypes_1279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ctypes')
# Obtaining the member 'att1' of a type
att1_1280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ctypes_1279, 'att1')
# Getting the type of '_ctypes'
_ctypes_1281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ctypes')
# Setting the type of the member 'att2' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ctypes_1281, 'att2', att1_1280)

# Assigning a Call to a Name (line 28):

# Call to property(...): (line 28)
# Processing the call arguments (line 28)
# Getting the type of '_ctypes'
_ctypes_1283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ctypes', False)
# Obtaining the member 'get_data' of a type
get_data_1284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ctypes_1283, 'get_data')
# Getting the type of 'None' (line 28)
None_1285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 30), 'None', False)
# Processing the call keyword arguments (line 28)
str_1286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 40), 'str', 'c-types data')
keyword_1287 = str_1286
kwargs_1288 = {'doc': keyword_1287}
# Getting the type of 'property' (line 28)
property_1282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'property', False)
# Calling property(args, kwargs) (line 28)
property_call_result_1289 = invoke(stypy.reporting.localization.Localization(__file__, 28, 11), property_1282, *[get_data_1284, None_1285], **kwargs_1288)

# Getting the type of '_ctypes'
_ctypes_1290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ctypes')
# Setting the type of the member 'data' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ctypes_1290, 'data', property_call_result_1289)

# Assigning a Call to a Name (line 29):

# Call to property(...): (line 29)
# Processing the call arguments (line 29)
# Getting the type of '_ctypes'
_ctypes_1292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ctypes', False)
# Obtaining the member 'get_shape' of a type
get_shape_1293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ctypes_1292, 'get_shape')
# Getting the type of 'None' (line 29)
None_1294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 32), 'None', False)
# Processing the call keyword arguments (line 29)
str_1295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 42), 'str', 'c-types shape')
keyword_1296 = str_1295
kwargs_1297 = {'doc': keyword_1296}
# Getting the type of 'property' (line 29)
property_1291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'property', False)
# Calling property(args, kwargs) (line 29)
property_call_result_1298 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), property_1291, *[get_shape_1293, None_1294], **kwargs_1297)

# Getting the type of '_ctypes'
_ctypes_1299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ctypes')
# Setting the type of the member 'shape' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ctypes_1299, 'shape', property_call_result_1298)

# Assigning a Call to a Name (line 30):

# Call to property(...): (line 30)
# Processing the call arguments (line 30)
# Getting the type of '_ctypes'
_ctypes_1301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ctypes', False)
# Obtaining the member 'get_strides' of a type
get_strides_1302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ctypes_1301, 'get_strides')
# Getting the type of 'None' (line 30)
None_1303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 36), 'None', False)
# Processing the call keyword arguments (line 30)
str_1304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 46), 'str', 'c-types strides')
keyword_1305 = str_1304
kwargs_1306 = {'doc': keyword_1305}
# Getting the type of 'property' (line 30)
property_1300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 14), 'property', False)
# Calling property(args, kwargs) (line 30)
property_call_result_1307 = invoke(stypy.reporting.localization.Localization(__file__, 30, 14), property_1300, *[get_strides_1302, None_1303], **kwargs_1306)

# Getting the type of '_ctypes'
_ctypes_1308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ctypes')
# Setting the type of the member 'strides' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ctypes_1308, 'strides', property_call_result_1307)

# Assigning a Call to a Name (line 31):

# Call to property(...): (line 31)
# Processing the call arguments (line 31)
# Getting the type of '_ctypes'
_ctypes_1310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ctypes', False)
# Obtaining the member 'get_as_parameter' of a type
get_as_parameter_1311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ctypes_1310, 'get_as_parameter')
# Getting the type of 'None' (line 31)
None_1312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 48), 'None', False)
# Processing the call keyword arguments (line 31)
str_1313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 58), 'str', '_as parameter_')
keyword_1314 = str_1313
kwargs_1315 = {'doc': keyword_1314}
# Getting the type of 'property' (line 31)
property_1309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 21), 'property', False)
# Calling property(args, kwargs) (line 31)
property_call_result_1316 = invoke(stypy.reporting.localization.Localization(__file__, 31, 21), property_1309, *[get_as_parameter_1311, None_1312], **kwargs_1315)

# Getting the type of '_ctypes'
_ctypes_1317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ctypes')
# Setting the type of the member '_as_parameter_' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ctypes_1317, '_as_parameter_', property_call_result_1316)

# Assigning a Call to a Name (line 33):

# Call to _ctypes(...): (line 33)
# Processing the call keyword arguments (line 33)
kwargs_1319 = {}
# Getting the type of '_ctypes' (line 33)
_ctypes_1318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 5), '_ctypes', False)
# Calling _ctypes(args, kwargs) (line 33)
_ctypes_call_result_1320 = invoke(stypy.reporting.localization.Localization(__file__, 33, 5), _ctypes_1318, *[], **kwargs_1319)

# Assigning a type to the variable 'ct' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'ct', _ctypes_call_result_1320)

# Assigning a Attribute to a Name (line 35):
# Getting the type of 'ct' (line 35)
ct_1321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'ct')
# Obtaining the member 'data' of a type (line 35)
data_1322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 4), ct_1321, 'data')
# Assigning a type to the variable 'r' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'r', data_1322)

# Assigning a Attribute to a Name (line 36):
# Getting the type of 'ct' (line 36)
ct_1323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 5), 'ct')
# Obtaining the member 'shape' of a type (line 36)
shape_1324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 5), ct_1323, 'shape')
# Assigning a type to the variable 'r2' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'r2', shape_1324)

# Assigning a Attribute to a Name (line 37):
# Getting the type of 'ct' (line 37)
ct_1325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 5), 'ct')
# Obtaining the member 'strides' of a type (line 37)
strides_1326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 5), ct_1325, 'strides')
# Assigning a type to the variable 'r3' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'r3', strides_1326)

# Assigning a Attribute to a Name (line 38):
# Getting the type of 'ct' (line 38)
ct_1327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 5), 'ct')
# Obtaining the member 'get_data' of a type (line 38)
get_data_1328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 5), ct_1327, 'get_data')
# Assigning a type to the variable 'r4' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'r4', get_data_1328)

# Assigning a Attribute to a Name (line 40):
# Getting the type of 'ct' (line 40)
ct_1329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 5), 'ct')
# Obtaining the member 'att1' of a type (line 40)
att1_1330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 5), ct_1329, 'att1')
# Assigning a type to the variable 'ra' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'ra', att1_1330)

# Assigning a Attribute to a Name (line 41):
# Getting the type of 'ct' (line 41)
ct_1331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 5), 'ct')
# Obtaining the member 'att2' of a type (line 41)
att2_1332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 5), ct_1331, 'att2')
# Assigning a type to the variable 'rb' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'rb', att2_1332)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
