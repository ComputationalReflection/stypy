
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def asbytes(st):
2:     return 0
3: 
4: 
5: class StringConverter(object):
6:     _mapper = [(bool, str, False)]
7:     if True:
8:         _mapper.append((int, int, -1))
9: 
10:     _mapper.extend([(float, float, None), (str, bytes, asbytes('???'))])
11: 
12:     a, b, c = zip(*_mapper)
13: 
14: st = StringConverter()
15: r = st._mapper
16: r2 = st.a
17: r3 = st.b
18: r4 = st.c
19: 
20: print r
21: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def asbytes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'asbytes'
    module_type_store = module_type_store.open_function_context('asbytes', 1, 0, False)
    
    # Passed parameters checking function
    asbytes.stypy_localization = localization
    asbytes.stypy_type_of_self = None
    asbytes.stypy_type_store = module_type_store
    asbytes.stypy_function_name = 'asbytes'
    asbytes.stypy_param_names_list = ['st']
    asbytes.stypy_varargs_param_name = None
    asbytes.stypy_kwargs_param_name = None
    asbytes.stypy_call_defaults = defaults
    asbytes.stypy_call_varargs = varargs
    asbytes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'asbytes', ['st'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'asbytes', localization, ['st'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'asbytes(...)' code ##################

    int_1120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'stypy_return_type', int_1120)
    
    # ################# End of 'asbytes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'asbytes' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_1121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1121)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'asbytes'
    return stypy_return_type_1121

# Assigning a type to the variable 'asbytes' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'asbytes', asbytes)
# Declaration of the 'StringConverter' class

class StringConverter(object, ):
    
    # Assigning a Call to a Tuple (line 12):
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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StringConverter.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'StringConverter' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'StringConverter', StringConverter)

# Assigning a List to a Name (line 6):

# Obtaining an instance of the builtin type 'list' (line 6)
list_1122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)

# Obtaining an instance of the builtin type 'tuple' (line 6)
tuple_1123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6)
# Adding element type (line 6)
# Getting the type of 'bool' (line 6)
bool_1124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 16), 'bool')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 16), tuple_1123, bool_1124)
# Adding element type (line 6)
# Getting the type of 'str' (line 6)
str_1125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 22), 'str')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 16), tuple_1123, str_1125)
# Adding element type (line 6)
# Getting the type of 'False' (line 6)
False_1126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 27), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 16), tuple_1123, False_1126)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 14), list_1122, tuple_1123)

# Getting the type of 'StringConverter'
StringConverter_1127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Setting the type of the member '_mapper' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_1127, '_mapper', list_1122)

# Assigning a List to a Name (line 6):

# Getting the type of 'True' (line 7)
True_1128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 7), 'True')
# Testing the type of an if condition (line 7)
if_condition_1129 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 7, 4), True_1128)
# Assigning a type to the variable 'if_condition_1129' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'if_condition_1129', if_condition_1129)
# SSA begins for if statement (line 7)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to append(...): (line 8)
# Processing the call arguments (line 8)

# Obtaining an instance of the builtin type 'tuple' (line 8)
tuple_1133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 8)
# Adding element type (line 8)
# Getting the type of 'int' (line 8)
int_1134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 24), 'int', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 24), tuple_1133, int_1134)
# Adding element type (line 8)
# Getting the type of 'int' (line 8)
int_1135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 29), 'int', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 24), tuple_1133, int_1135)
# Adding element type (line 8)
int_1136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 24), tuple_1133, int_1136)

# Processing the call keyword arguments (line 8)
kwargs_1137 = {}
# Getting the type of 'StringConverter'
StringConverter_1130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter', False)
# Obtaining the member '_mapper' of a type
_mapper_1131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_1130, '_mapper')
# Obtaining the member 'append' of a type (line 8)
append_1132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 8), _mapper_1131, 'append')
# Calling append(args, kwargs) (line 8)
append_call_result_1138 = invoke(stypy.reporting.localization.Localization(__file__, 8, 8), append_1132, *[tuple_1133], **kwargs_1137)

# SSA join for if statement (line 7)
module_type_store = module_type_store.join_ssa_context()


# Call to extend(...): (line 10)
# Processing the call arguments (line 10)

# Obtaining an instance of the builtin type 'list' (line 10)
list_1142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)

# Obtaining an instance of the builtin type 'tuple' (line 10)
tuple_1143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 10)
# Adding element type (line 10)
# Getting the type of 'float' (line 10)
float_1144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 21), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 21), tuple_1143, float_1144)
# Adding element type (line 10)
# Getting the type of 'float' (line 10)
float_1145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 28), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 21), tuple_1143, float_1145)
# Adding element type (line 10)
# Getting the type of 'None' (line 10)
None_1146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 35), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 21), tuple_1143, None_1146)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 19), list_1142, tuple_1143)
# Adding element type (line 10)

# Obtaining an instance of the builtin type 'tuple' (line 10)
tuple_1147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 43), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 10)
# Adding element type (line 10)
# Getting the type of 'str' (line 10)
str_1148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 43), 'str', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 43), tuple_1147, str_1148)
# Adding element type (line 10)
# Getting the type of 'bytes' (line 10)
bytes_1149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 48), 'bytes', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 43), tuple_1147, bytes_1149)
# Adding element type (line 10)

# Call to asbytes(...): (line 10)
# Processing the call arguments (line 10)
str_1151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 63), 'str', '???')
# Processing the call keyword arguments (line 10)
kwargs_1152 = {}
# Getting the type of 'asbytes' (line 10)
asbytes_1150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 55), 'asbytes', False)
# Calling asbytes(args, kwargs) (line 10)
asbytes_call_result_1153 = invoke(stypy.reporting.localization.Localization(__file__, 10, 55), asbytes_1150, *[str_1151], **kwargs_1152)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 43), tuple_1147, asbytes_call_result_1153)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 19), list_1142, tuple_1147)

# Processing the call keyword arguments (line 10)
kwargs_1154 = {}
# Getting the type of 'StringConverter'
StringConverter_1139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter', False)
# Obtaining the member '_mapper' of a type
_mapper_1140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_1139, '_mapper')
# Obtaining the member 'extend' of a type (line 10)
extend_1141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), _mapper_1140, 'extend')
# Calling extend(args, kwargs) (line 10)
extend_call_result_1155 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), extend_1141, *[list_1142], **kwargs_1154)


# Assigning a Subscript to a Name (line 12):

# Obtaining the type of the subscript
int_1156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 4), 'int')

# Call to zip(...): (line 12)
# Getting the type of 'StringConverter'
StringConverter_1158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter', False)
# Obtaining the member '_mapper' of a type
_mapper_1159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_1158, '_mapper')
# Processing the call keyword arguments (line 12)
kwargs_1160 = {}
# Getting the type of 'zip' (line 12)
zip_1157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 14), 'zip', False)
# Calling zip(args, kwargs) (line 12)
zip_call_result_1161 = invoke(stypy.reporting.localization.Localization(__file__, 12, 14), zip_1157, *[_mapper_1159], **kwargs_1160)

# Obtaining the member '__getitem__' of a type (line 12)
getitem___1162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), zip_call_result_1161, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_1163 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), getitem___1162, int_1156)

# Getting the type of 'StringConverter'
StringConverter_1164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Setting the type of the member 'tuple_var_assignment_1117' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_1164, 'tuple_var_assignment_1117', subscript_call_result_1163)

# Assigning a Subscript to a Name (line 12):

# Obtaining the type of the subscript
int_1165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 4), 'int')

# Call to zip(...): (line 12)
# Getting the type of 'StringConverter'
StringConverter_1167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter', False)
# Obtaining the member '_mapper' of a type
_mapper_1168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_1167, '_mapper')
# Processing the call keyword arguments (line 12)
kwargs_1169 = {}
# Getting the type of 'zip' (line 12)
zip_1166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 14), 'zip', False)
# Calling zip(args, kwargs) (line 12)
zip_call_result_1170 = invoke(stypy.reporting.localization.Localization(__file__, 12, 14), zip_1166, *[_mapper_1168], **kwargs_1169)

# Obtaining the member '__getitem__' of a type (line 12)
getitem___1171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), zip_call_result_1170, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_1172 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), getitem___1171, int_1165)

# Getting the type of 'StringConverter'
StringConverter_1173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Setting the type of the member 'tuple_var_assignment_1118' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_1173, 'tuple_var_assignment_1118', subscript_call_result_1172)

# Assigning a Subscript to a Name (line 12):

# Obtaining the type of the subscript
int_1174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 4), 'int')

# Call to zip(...): (line 12)
# Getting the type of 'StringConverter'
StringConverter_1176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter', False)
# Obtaining the member '_mapper' of a type
_mapper_1177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_1176, '_mapper')
# Processing the call keyword arguments (line 12)
kwargs_1178 = {}
# Getting the type of 'zip' (line 12)
zip_1175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 14), 'zip', False)
# Calling zip(args, kwargs) (line 12)
zip_call_result_1179 = invoke(stypy.reporting.localization.Localization(__file__, 12, 14), zip_1175, *[_mapper_1177], **kwargs_1178)

# Obtaining the member '__getitem__' of a type (line 12)
getitem___1180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), zip_call_result_1179, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_1181 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), getitem___1180, int_1174)

# Getting the type of 'StringConverter'
StringConverter_1182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Setting the type of the member 'tuple_var_assignment_1119' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_1182, 'tuple_var_assignment_1119', subscript_call_result_1181)

# Assigning a Name to a Name (line 12):
# Getting the type of 'StringConverter'
StringConverter_1183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Obtaining the member 'tuple_var_assignment_1117' of a type
tuple_var_assignment_1117_1184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_1183, 'tuple_var_assignment_1117')
# Getting the type of 'StringConverter'
StringConverter_1185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Setting the type of the member 'a' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_1185, 'a', tuple_var_assignment_1117_1184)

# Assigning a Name to a Name (line 12):
# Getting the type of 'StringConverter'
StringConverter_1186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Obtaining the member 'tuple_var_assignment_1118' of a type
tuple_var_assignment_1118_1187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_1186, 'tuple_var_assignment_1118')
# Getting the type of 'StringConverter'
StringConverter_1188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Setting the type of the member 'b' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_1188, 'b', tuple_var_assignment_1118_1187)

# Assigning a Name to a Name (line 12):
# Getting the type of 'StringConverter'
StringConverter_1189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Obtaining the member 'tuple_var_assignment_1119' of a type
tuple_var_assignment_1119_1190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_1189, 'tuple_var_assignment_1119')
# Getting the type of 'StringConverter'
StringConverter_1191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'StringConverter')
# Setting the type of the member 'c' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), StringConverter_1191, 'c', tuple_var_assignment_1119_1190)

# Assigning a Call to a Name (line 14):

# Assigning a Call to a Name (line 14):

# Call to StringConverter(...): (line 14)
# Processing the call keyword arguments (line 14)
kwargs_1193 = {}
# Getting the type of 'StringConverter' (line 14)
StringConverter_1192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 5), 'StringConverter', False)
# Calling StringConverter(args, kwargs) (line 14)
StringConverter_call_result_1194 = invoke(stypy.reporting.localization.Localization(__file__, 14, 5), StringConverter_1192, *[], **kwargs_1193)

# Assigning a type to the variable 'st' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'st', StringConverter_call_result_1194)

# Assigning a Attribute to a Name (line 15):

# Assigning a Attribute to a Name (line 15):
# Getting the type of 'st' (line 15)
st_1195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'st')
# Obtaining the member '_mapper' of a type (line 15)
_mapper_1196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 4), st_1195, '_mapper')
# Assigning a type to the variable 'r' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'r', _mapper_1196)

# Assigning a Attribute to a Name (line 16):

# Assigning a Attribute to a Name (line 16):
# Getting the type of 'st' (line 16)
st_1197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 5), 'st')
# Obtaining the member 'a' of a type (line 16)
a_1198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 5), st_1197, 'a')
# Assigning a type to the variable 'r2' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'r2', a_1198)

# Assigning a Attribute to a Name (line 17):

# Assigning a Attribute to a Name (line 17):
# Getting the type of 'st' (line 17)
st_1199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'st')
# Obtaining the member 'b' of a type (line 17)
b_1200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), st_1199, 'b')
# Assigning a type to the variable 'r3' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r3', b_1200)

# Assigning a Attribute to a Name (line 18):

# Assigning a Attribute to a Name (line 18):
# Getting the type of 'st' (line 18)
st_1201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 5), 'st')
# Obtaining the member 'c' of a type (line 18)
c_1202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 5), st_1201, 'c')
# Assigning a type to the variable 'r4' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'r4', c_1202)
# Getting the type of 'r' (line 20)
r_1203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 6), 'r')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
