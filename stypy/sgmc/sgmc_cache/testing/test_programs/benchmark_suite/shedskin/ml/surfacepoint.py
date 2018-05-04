
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #  MiniLight Python : minimal global illumination renderer
2: #
3: #  Copyright (c) 2007-2008, Harrison Ainsworth / HXA7241 and Juraj Sukop.
4: #  http://www.hxa7241.org/
5: 
6: 
7: from math import cos, pi, sin, sqrt
8: from random import random
9: from vector3f import Vector3f, ONE, ZERO
10: 
11: class SurfacePoint(object):
12: 
13:     def __init__(self, triangle, position):
14:         self.triangle_ref = triangle
15:         self.position = position.copy() 
16: 
17:     def get_emission(self, to_position, out_direction, is_solid_angle):
18:         ray = to_position - self.position
19:         distance2 = ray.dot(ray)
20:         cos_area = out_direction.dot(self.triangle_ref.normal) * self.triangle_ref.area
21:         solid_angle = cos_area / max(distance2, 1e-6) if is_solid_angle else 1.0
22:         return self.triangle_ref.emitivity * solid_angle if cos_area > 0.0 else ZERO
23: 
24:     def get_reflection(self, in_direction, in_radiance, out_direction):
25:         in_dot = in_direction.dot(self.triangle_ref.normal)
26:         out_dot = out_direction.dot(self.triangle_ref.normal)
27:         return ZERO if (in_dot < 0.0) ^ (out_dot < 0.0) else in_radiance.mul(self.triangle_ref.reflectivity) * (abs(in_dot) / pi)
28: 
29:     def get_next_direction(self, in_direction):
30:         reflectivity_mean = self.triangle_ref.reflectivity.dot(ONE) / 3.0
31:         if random() < reflectivity_mean:
32:             color = self.triangle_ref.reflectivity * (1.0 / reflectivity_mean)
33:             _2pr1 = pi * 2.0 * random()
34:             sr2 = sqrt(random())
35:             x = (cos(_2pr1) * sr2)
36:             y = (sin(_2pr1) * sr2)
37:             z = sqrt(1.0 - (sr2 * sr2))
38:             normal = self.triangle_ref.normal
39:             tangent = self.triangle_ref.tangent
40:             if normal.dot(in_direction) < 0.0:
41:                 normal = -normal
42:             out_direction = (tangent * x) + (normal.cross(tangent) * y) + (normal * z)
43:             return out_direction, color
44:         else:
45:             return ZERO, ZERO
46: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from math import cos, pi, sin, sqrt' statement (line 7)
try:
    from math import cos, pi, sin, sqrt

except:
    cos = UndefinedType
    pi = UndefinedType
    sin = UndefinedType
    sqrt = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'math', None, module_type_store, ['cos', 'pi', 'sin', 'sqrt'], [cos, pi, sin, sqrt])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from random import random' statement (line 8)
try:
    from random import random

except:
    random = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'random', None, module_type_store, ['random'], [random])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from vector3f import Vector3f, ONE, ZERO' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')
import_1653 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'vector3f')

if (type(import_1653) is not StypyTypeError):

    if (import_1653 != 'pyd_module'):
        __import__(import_1653)
        sys_modules_1654 = sys.modules[import_1653]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'vector3f', sys_modules_1654.module_type_store, module_type_store, ['Vector3f', 'ONE', 'ZERO'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_1654, sys_modules_1654.module_type_store, module_type_store)
    else:
        from vector3f import Vector3f, ONE, ZERO

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'vector3f', None, module_type_store, ['Vector3f', 'ONE', 'ZERO'], [Vector3f, ONE, ZERO])

else:
    # Assigning a type to the variable 'vector3f' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'vector3f', import_1653)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')

# Declaration of the 'SurfacePoint' class

class SurfacePoint(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SurfacePoint.__init__', ['triangle', 'position'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['triangle', 'position'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 14):
        # Getting the type of 'triangle' (line 14)
        triangle_1655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 28), 'triangle')
        # Getting the type of 'self' (line 14)
        self_1656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'self')
        # Setting the type of the member 'triangle_ref' of a type (line 14)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), self_1656, 'triangle_ref', triangle_1655)
        
        # Assigning a Call to a Attribute (line 15):
        
        # Call to copy(...): (line 15)
        # Processing the call keyword arguments (line 15)
        kwargs_1659 = {}
        # Getting the type of 'position' (line 15)
        position_1657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 24), 'position', False)
        # Obtaining the member 'copy' of a type (line 15)
        copy_1658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 24), position_1657, 'copy')
        # Calling copy(args, kwargs) (line 15)
        copy_call_result_1660 = invoke(stypy.reporting.localization.Localization(__file__, 15, 24), copy_1658, *[], **kwargs_1659)
        
        # Getting the type of 'self' (line 15)
        self_1661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self')
        # Setting the type of the member 'position' of a type (line 15)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), self_1661, 'position', copy_call_result_1660)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_emission(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_emission'
        module_type_store = module_type_store.open_function_context('get_emission', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SurfacePoint.get_emission.__dict__.__setitem__('stypy_localization', localization)
        SurfacePoint.get_emission.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SurfacePoint.get_emission.__dict__.__setitem__('stypy_type_store', module_type_store)
        SurfacePoint.get_emission.__dict__.__setitem__('stypy_function_name', 'SurfacePoint.get_emission')
        SurfacePoint.get_emission.__dict__.__setitem__('stypy_param_names_list', ['to_position', 'out_direction', 'is_solid_angle'])
        SurfacePoint.get_emission.__dict__.__setitem__('stypy_varargs_param_name', None)
        SurfacePoint.get_emission.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SurfacePoint.get_emission.__dict__.__setitem__('stypy_call_defaults', defaults)
        SurfacePoint.get_emission.__dict__.__setitem__('stypy_call_varargs', varargs)
        SurfacePoint.get_emission.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SurfacePoint.get_emission.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SurfacePoint.get_emission', ['to_position', 'out_direction', 'is_solid_angle'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_emission', localization, ['to_position', 'out_direction', 'is_solid_angle'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_emission(...)' code ##################

        
        # Assigning a BinOp to a Name (line 18):
        # Getting the type of 'to_position' (line 18)
        to_position_1662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), 'to_position')
        # Getting the type of 'self' (line 18)
        self_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 28), 'self')
        # Obtaining the member 'position' of a type (line 18)
        position_1664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 28), self_1663, 'position')
        # Applying the binary operator '-' (line 18)
        result_sub_1665 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 14), '-', to_position_1662, position_1664)
        
        # Assigning a type to the variable 'ray' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'ray', result_sub_1665)
        
        # Assigning a Call to a Name (line 19):
        
        # Call to dot(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'ray' (line 19)
        ray_1668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 28), 'ray', False)
        # Processing the call keyword arguments (line 19)
        kwargs_1669 = {}
        # Getting the type of 'ray' (line 19)
        ray_1666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'ray', False)
        # Obtaining the member 'dot' of a type (line 19)
        dot_1667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 20), ray_1666, 'dot')
        # Calling dot(args, kwargs) (line 19)
        dot_call_result_1670 = invoke(stypy.reporting.localization.Localization(__file__, 19, 20), dot_1667, *[ray_1668], **kwargs_1669)
        
        # Assigning a type to the variable 'distance2' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'distance2', dot_call_result_1670)
        
        # Assigning a BinOp to a Name (line 20):
        
        # Call to dot(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'self' (line 20)
        self_1673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 37), 'self', False)
        # Obtaining the member 'triangle_ref' of a type (line 20)
        triangle_ref_1674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 37), self_1673, 'triangle_ref')
        # Obtaining the member 'normal' of a type (line 20)
        normal_1675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 37), triangle_ref_1674, 'normal')
        # Processing the call keyword arguments (line 20)
        kwargs_1676 = {}
        # Getting the type of 'out_direction' (line 20)
        out_direction_1671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 19), 'out_direction', False)
        # Obtaining the member 'dot' of a type (line 20)
        dot_1672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 19), out_direction_1671, 'dot')
        # Calling dot(args, kwargs) (line 20)
        dot_call_result_1677 = invoke(stypy.reporting.localization.Localization(__file__, 20, 19), dot_1672, *[normal_1675], **kwargs_1676)
        
        # Getting the type of 'self' (line 20)
        self_1678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 65), 'self')
        # Obtaining the member 'triangle_ref' of a type (line 20)
        triangle_ref_1679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 65), self_1678, 'triangle_ref')
        # Obtaining the member 'area' of a type (line 20)
        area_1680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 65), triangle_ref_1679, 'area')
        # Applying the binary operator '*' (line 20)
        result_mul_1681 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 19), '*', dot_call_result_1677, area_1680)
        
        # Assigning a type to the variable 'cos_area' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'cos_area', result_mul_1681)
        
        # Assigning a IfExp to a Name (line 21):
        
        # Getting the type of 'is_solid_angle' (line 21)
        is_solid_angle_1682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 57), 'is_solid_angle')
        # Testing the type of an if expression (line 21)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 21, 22), is_solid_angle_1682)
        # SSA begins for if expression (line 21)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'cos_area' (line 21)
        cos_area_1683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'cos_area')
        
        # Call to max(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'distance2' (line 21)
        distance2_1685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 37), 'distance2', False)
        float_1686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 48), 'float')
        # Processing the call keyword arguments (line 21)
        kwargs_1687 = {}
        # Getting the type of 'max' (line 21)
        max_1684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 33), 'max', False)
        # Calling max(args, kwargs) (line 21)
        max_call_result_1688 = invoke(stypy.reporting.localization.Localization(__file__, 21, 33), max_1684, *[distance2_1685, float_1686], **kwargs_1687)
        
        # Applying the binary operator 'div' (line 21)
        result_div_1689 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 22), 'div', cos_area_1683, max_call_result_1688)
        
        # SSA branch for the else part of an if expression (line 21)
        module_type_store.open_ssa_branch('if expression else')
        float_1690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 77), 'float')
        # SSA join for if expression (line 21)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_1691 = union_type.UnionType.add(result_div_1689, float_1690)
        
        # Assigning a type to the variable 'solid_angle' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'solid_angle', if_exp_1691)
        
        
        # Getting the type of 'cos_area' (line 22)
        cos_area_1692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 60), 'cos_area')
        float_1693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 71), 'float')
        # Applying the binary operator '>' (line 22)
        result_gt_1694 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 60), '>', cos_area_1692, float_1693)
        
        # Testing the type of an if expression (line 22)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 15), result_gt_1694)
        # SSA begins for if expression (line 22)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'self' (line 22)
        self_1695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 15), 'self')
        # Obtaining the member 'triangle_ref' of a type (line 22)
        triangle_ref_1696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 15), self_1695, 'triangle_ref')
        # Obtaining the member 'emitivity' of a type (line 22)
        emitivity_1697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 15), triangle_ref_1696, 'emitivity')
        # Getting the type of 'solid_angle' (line 22)
        solid_angle_1698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 45), 'solid_angle')
        # Applying the binary operator '*' (line 22)
        result_mul_1699 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 15), '*', emitivity_1697, solid_angle_1698)
        
        # SSA branch for the else part of an if expression (line 22)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'ZERO' (line 22)
        ZERO_1700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 80), 'ZERO')
        # SSA join for if expression (line 22)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_1701 = union_type.UnionType.add(result_mul_1699, ZERO_1700)
        
        # Assigning a type to the variable 'stypy_return_type' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'stypy_return_type', if_exp_1701)
        
        # ################# End of 'get_emission(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_emission' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_1702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1702)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_emission'
        return stypy_return_type_1702


    @norecursion
    def get_reflection(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_reflection'
        module_type_store = module_type_store.open_function_context('get_reflection', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SurfacePoint.get_reflection.__dict__.__setitem__('stypy_localization', localization)
        SurfacePoint.get_reflection.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SurfacePoint.get_reflection.__dict__.__setitem__('stypy_type_store', module_type_store)
        SurfacePoint.get_reflection.__dict__.__setitem__('stypy_function_name', 'SurfacePoint.get_reflection')
        SurfacePoint.get_reflection.__dict__.__setitem__('stypy_param_names_list', ['in_direction', 'in_radiance', 'out_direction'])
        SurfacePoint.get_reflection.__dict__.__setitem__('stypy_varargs_param_name', None)
        SurfacePoint.get_reflection.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SurfacePoint.get_reflection.__dict__.__setitem__('stypy_call_defaults', defaults)
        SurfacePoint.get_reflection.__dict__.__setitem__('stypy_call_varargs', varargs)
        SurfacePoint.get_reflection.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SurfacePoint.get_reflection.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SurfacePoint.get_reflection', ['in_direction', 'in_radiance', 'out_direction'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_reflection', localization, ['in_direction', 'in_radiance', 'out_direction'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_reflection(...)' code ##################

        
        # Assigning a Call to a Name (line 25):
        
        # Call to dot(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'self' (line 25)
        self_1705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 34), 'self', False)
        # Obtaining the member 'triangle_ref' of a type (line 25)
        triangle_ref_1706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 34), self_1705, 'triangle_ref')
        # Obtaining the member 'normal' of a type (line 25)
        normal_1707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 34), triangle_ref_1706, 'normal')
        # Processing the call keyword arguments (line 25)
        kwargs_1708 = {}
        # Getting the type of 'in_direction' (line 25)
        in_direction_1703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'in_direction', False)
        # Obtaining the member 'dot' of a type (line 25)
        dot_1704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 17), in_direction_1703, 'dot')
        # Calling dot(args, kwargs) (line 25)
        dot_call_result_1709 = invoke(stypy.reporting.localization.Localization(__file__, 25, 17), dot_1704, *[normal_1707], **kwargs_1708)
        
        # Assigning a type to the variable 'in_dot' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'in_dot', dot_call_result_1709)
        
        # Assigning a Call to a Name (line 26):
        
        # Call to dot(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'self' (line 26)
        self_1712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 36), 'self', False)
        # Obtaining the member 'triangle_ref' of a type (line 26)
        triangle_ref_1713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 36), self_1712, 'triangle_ref')
        # Obtaining the member 'normal' of a type (line 26)
        normal_1714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 36), triangle_ref_1713, 'normal')
        # Processing the call keyword arguments (line 26)
        kwargs_1715 = {}
        # Getting the type of 'out_direction' (line 26)
        out_direction_1710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), 'out_direction', False)
        # Obtaining the member 'dot' of a type (line 26)
        dot_1711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 18), out_direction_1710, 'dot')
        # Calling dot(args, kwargs) (line 26)
        dot_call_result_1716 = invoke(stypy.reporting.localization.Localization(__file__, 26, 18), dot_1711, *[normal_1714], **kwargs_1715)
        
        # Assigning a type to the variable 'out_dot' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'out_dot', dot_call_result_1716)
        
        
        # Getting the type of 'in_dot' (line 27)
        in_dot_1717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 24), 'in_dot')
        float_1718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 33), 'float')
        # Applying the binary operator '<' (line 27)
        result_lt_1719 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 24), '<', in_dot_1717, float_1718)
        
        
        # Getting the type of 'out_dot' (line 27)
        out_dot_1720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 41), 'out_dot')
        float_1721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 51), 'float')
        # Applying the binary operator '<' (line 27)
        result_lt_1722 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 41), '<', out_dot_1720, float_1721)
        
        # Applying the binary operator '^' (line 27)
        result_xor_1723 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 23), '^', result_lt_1719, result_lt_1722)
        
        # Testing the type of an if expression (line 27)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 15), result_xor_1723)
        # SSA begins for if expression (line 27)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'ZERO' (line 27)
        ZERO_1724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'ZERO')
        # SSA branch for the else part of an if expression (line 27)
        module_type_store.open_ssa_branch('if expression else')
        
        # Call to mul(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'self' (line 27)
        self_1727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 77), 'self', False)
        # Obtaining the member 'triangle_ref' of a type (line 27)
        triangle_ref_1728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 77), self_1727, 'triangle_ref')
        # Obtaining the member 'reflectivity' of a type (line 27)
        reflectivity_1729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 77), triangle_ref_1728, 'reflectivity')
        # Processing the call keyword arguments (line 27)
        kwargs_1730 = {}
        # Getting the type of 'in_radiance' (line 27)
        in_radiance_1725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 61), 'in_radiance', False)
        # Obtaining the member 'mul' of a type (line 27)
        mul_1726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 61), in_radiance_1725, 'mul')
        # Calling mul(args, kwargs) (line 27)
        mul_call_result_1731 = invoke(stypy.reporting.localization.Localization(__file__, 27, 61), mul_1726, *[reflectivity_1729], **kwargs_1730)
        
        
        # Call to abs(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'in_dot' (line 27)
        in_dot_1733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 116), 'in_dot', False)
        # Processing the call keyword arguments (line 27)
        kwargs_1734 = {}
        # Getting the type of 'abs' (line 27)
        abs_1732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 112), 'abs', False)
        # Calling abs(args, kwargs) (line 27)
        abs_call_result_1735 = invoke(stypy.reporting.localization.Localization(__file__, 27, 112), abs_1732, *[in_dot_1733], **kwargs_1734)
        
        # Getting the type of 'pi' (line 27)
        pi_1736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 126), 'pi')
        # Applying the binary operator 'div' (line 27)
        result_div_1737 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 112), 'div', abs_call_result_1735, pi_1736)
        
        # Applying the binary operator '*' (line 27)
        result_mul_1738 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 61), '*', mul_call_result_1731, result_div_1737)
        
        # SSA join for if expression (line 27)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_1739 = union_type.UnionType.add(ZERO_1724, result_mul_1738)
        
        # Assigning a type to the variable 'stypy_return_type' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type', if_exp_1739)
        
        # ################# End of 'get_reflection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_reflection' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_1740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1740)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_reflection'
        return stypy_return_type_1740


    @norecursion
    def get_next_direction(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_next_direction'
        module_type_store = module_type_store.open_function_context('get_next_direction', 29, 4, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SurfacePoint.get_next_direction.__dict__.__setitem__('stypy_localization', localization)
        SurfacePoint.get_next_direction.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SurfacePoint.get_next_direction.__dict__.__setitem__('stypy_type_store', module_type_store)
        SurfacePoint.get_next_direction.__dict__.__setitem__('stypy_function_name', 'SurfacePoint.get_next_direction')
        SurfacePoint.get_next_direction.__dict__.__setitem__('stypy_param_names_list', ['in_direction'])
        SurfacePoint.get_next_direction.__dict__.__setitem__('stypy_varargs_param_name', None)
        SurfacePoint.get_next_direction.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SurfacePoint.get_next_direction.__dict__.__setitem__('stypy_call_defaults', defaults)
        SurfacePoint.get_next_direction.__dict__.__setitem__('stypy_call_varargs', varargs)
        SurfacePoint.get_next_direction.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SurfacePoint.get_next_direction.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SurfacePoint.get_next_direction', ['in_direction'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_next_direction', localization, ['in_direction'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_next_direction(...)' code ##################

        
        # Assigning a BinOp to a Name (line 30):
        
        # Call to dot(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'ONE' (line 30)
        ONE_1745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 63), 'ONE', False)
        # Processing the call keyword arguments (line 30)
        kwargs_1746 = {}
        # Getting the type of 'self' (line 30)
        self_1741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 28), 'self', False)
        # Obtaining the member 'triangle_ref' of a type (line 30)
        triangle_ref_1742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 28), self_1741, 'triangle_ref')
        # Obtaining the member 'reflectivity' of a type (line 30)
        reflectivity_1743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 28), triangle_ref_1742, 'reflectivity')
        # Obtaining the member 'dot' of a type (line 30)
        dot_1744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 28), reflectivity_1743, 'dot')
        # Calling dot(args, kwargs) (line 30)
        dot_call_result_1747 = invoke(stypy.reporting.localization.Localization(__file__, 30, 28), dot_1744, *[ONE_1745], **kwargs_1746)
        
        float_1748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 70), 'float')
        # Applying the binary operator 'div' (line 30)
        result_div_1749 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 28), 'div', dot_call_result_1747, float_1748)
        
        # Assigning a type to the variable 'reflectivity_mean' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'reflectivity_mean', result_div_1749)
        
        
        # Call to random(...): (line 31)
        # Processing the call keyword arguments (line 31)
        kwargs_1751 = {}
        # Getting the type of 'random' (line 31)
        random_1750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'random', False)
        # Calling random(args, kwargs) (line 31)
        random_call_result_1752 = invoke(stypy.reporting.localization.Localization(__file__, 31, 11), random_1750, *[], **kwargs_1751)
        
        # Getting the type of 'reflectivity_mean' (line 31)
        reflectivity_mean_1753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 22), 'reflectivity_mean')
        # Applying the binary operator '<' (line 31)
        result_lt_1754 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 11), '<', random_call_result_1752, reflectivity_mean_1753)
        
        # Testing if the type of an if condition is none (line 31)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 31, 8), result_lt_1754):
            
            # Obtaining an instance of the builtin type 'tuple' (line 45)
            tuple_1830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 45)
            # Adding element type (line 45)
            # Getting the type of 'ZERO' (line 45)
            ZERO_1831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'ZERO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 19), tuple_1830, ZERO_1831)
            # Adding element type (line 45)
            # Getting the type of 'ZERO' (line 45)
            ZERO_1832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), 'ZERO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 19), tuple_1830, ZERO_1832)
            
            # Assigning a type to the variable 'stypy_return_type' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'stypy_return_type', tuple_1830)
        else:
            
            # Testing the type of an if condition (line 31)
            if_condition_1755 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 8), result_lt_1754)
            # Assigning a type to the variable 'if_condition_1755' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'if_condition_1755', if_condition_1755)
            # SSA begins for if statement (line 31)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 32):
            # Getting the type of 'self' (line 32)
            self_1756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 20), 'self')
            # Obtaining the member 'triangle_ref' of a type (line 32)
            triangle_ref_1757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 20), self_1756, 'triangle_ref')
            # Obtaining the member 'reflectivity' of a type (line 32)
            reflectivity_1758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 20), triangle_ref_1757, 'reflectivity')
            float_1759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 54), 'float')
            # Getting the type of 'reflectivity_mean' (line 32)
            reflectivity_mean_1760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 60), 'reflectivity_mean')
            # Applying the binary operator 'div' (line 32)
            result_div_1761 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 54), 'div', float_1759, reflectivity_mean_1760)
            
            # Applying the binary operator '*' (line 32)
            result_mul_1762 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 20), '*', reflectivity_1758, result_div_1761)
            
            # Assigning a type to the variable 'color' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'color', result_mul_1762)
            
            # Assigning a BinOp to a Name (line 33):
            # Getting the type of 'pi' (line 33)
            pi_1763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'pi')
            float_1764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 25), 'float')
            # Applying the binary operator '*' (line 33)
            result_mul_1765 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 20), '*', pi_1763, float_1764)
            
            
            # Call to random(...): (line 33)
            # Processing the call keyword arguments (line 33)
            kwargs_1767 = {}
            # Getting the type of 'random' (line 33)
            random_1766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 31), 'random', False)
            # Calling random(args, kwargs) (line 33)
            random_call_result_1768 = invoke(stypy.reporting.localization.Localization(__file__, 33, 31), random_1766, *[], **kwargs_1767)
            
            # Applying the binary operator '*' (line 33)
            result_mul_1769 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 29), '*', result_mul_1765, random_call_result_1768)
            
            # Assigning a type to the variable '_2pr1' (line 33)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), '_2pr1', result_mul_1769)
            
            # Assigning a Call to a Name (line 34):
            
            # Call to sqrt(...): (line 34)
            # Processing the call arguments (line 34)
            
            # Call to random(...): (line 34)
            # Processing the call keyword arguments (line 34)
            kwargs_1772 = {}
            # Getting the type of 'random' (line 34)
            random_1771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 23), 'random', False)
            # Calling random(args, kwargs) (line 34)
            random_call_result_1773 = invoke(stypy.reporting.localization.Localization(__file__, 34, 23), random_1771, *[], **kwargs_1772)
            
            # Processing the call keyword arguments (line 34)
            kwargs_1774 = {}
            # Getting the type of 'sqrt' (line 34)
            sqrt_1770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 18), 'sqrt', False)
            # Calling sqrt(args, kwargs) (line 34)
            sqrt_call_result_1775 = invoke(stypy.reporting.localization.Localization(__file__, 34, 18), sqrt_1770, *[random_call_result_1773], **kwargs_1774)
            
            # Assigning a type to the variable 'sr2' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'sr2', sqrt_call_result_1775)
            
            # Assigning a BinOp to a Name (line 35):
            
            # Call to cos(...): (line 35)
            # Processing the call arguments (line 35)
            # Getting the type of '_2pr1' (line 35)
            _2pr1_1777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 21), '_2pr1', False)
            # Processing the call keyword arguments (line 35)
            kwargs_1778 = {}
            # Getting the type of 'cos' (line 35)
            cos_1776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 17), 'cos', False)
            # Calling cos(args, kwargs) (line 35)
            cos_call_result_1779 = invoke(stypy.reporting.localization.Localization(__file__, 35, 17), cos_1776, *[_2pr1_1777], **kwargs_1778)
            
            # Getting the type of 'sr2' (line 35)
            sr2_1780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 30), 'sr2')
            # Applying the binary operator '*' (line 35)
            result_mul_1781 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 17), '*', cos_call_result_1779, sr2_1780)
            
            # Assigning a type to the variable 'x' (line 35)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'x', result_mul_1781)
            
            # Assigning a BinOp to a Name (line 36):
            
            # Call to sin(...): (line 36)
            # Processing the call arguments (line 36)
            # Getting the type of '_2pr1' (line 36)
            _2pr1_1783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 21), '_2pr1', False)
            # Processing the call keyword arguments (line 36)
            kwargs_1784 = {}
            # Getting the type of 'sin' (line 36)
            sin_1782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'sin', False)
            # Calling sin(args, kwargs) (line 36)
            sin_call_result_1785 = invoke(stypy.reporting.localization.Localization(__file__, 36, 17), sin_1782, *[_2pr1_1783], **kwargs_1784)
            
            # Getting the type of 'sr2' (line 36)
            sr2_1786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 30), 'sr2')
            # Applying the binary operator '*' (line 36)
            result_mul_1787 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 17), '*', sin_call_result_1785, sr2_1786)
            
            # Assigning a type to the variable 'y' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'y', result_mul_1787)
            
            # Assigning a Call to a Name (line 37):
            
            # Call to sqrt(...): (line 37)
            # Processing the call arguments (line 37)
            float_1789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 21), 'float')
            # Getting the type of 'sr2' (line 37)
            sr2_1790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 28), 'sr2', False)
            # Getting the type of 'sr2' (line 37)
            sr2_1791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 34), 'sr2', False)
            # Applying the binary operator '*' (line 37)
            result_mul_1792 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 28), '*', sr2_1790, sr2_1791)
            
            # Applying the binary operator '-' (line 37)
            result_sub_1793 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 21), '-', float_1789, result_mul_1792)
            
            # Processing the call keyword arguments (line 37)
            kwargs_1794 = {}
            # Getting the type of 'sqrt' (line 37)
            sqrt_1788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'sqrt', False)
            # Calling sqrt(args, kwargs) (line 37)
            sqrt_call_result_1795 = invoke(stypy.reporting.localization.Localization(__file__, 37, 16), sqrt_1788, *[result_sub_1793], **kwargs_1794)
            
            # Assigning a type to the variable 'z' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'z', sqrt_call_result_1795)
            
            # Assigning a Attribute to a Name (line 38):
            # Getting the type of 'self' (line 38)
            self_1796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 21), 'self')
            # Obtaining the member 'triangle_ref' of a type (line 38)
            triangle_ref_1797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 21), self_1796, 'triangle_ref')
            # Obtaining the member 'normal' of a type (line 38)
            normal_1798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 21), triangle_ref_1797, 'normal')
            # Assigning a type to the variable 'normal' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'normal', normal_1798)
            
            # Assigning a Attribute to a Name (line 39):
            # Getting the type of 'self' (line 39)
            self_1799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 22), 'self')
            # Obtaining the member 'triangle_ref' of a type (line 39)
            triangle_ref_1800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 22), self_1799, 'triangle_ref')
            # Obtaining the member 'tangent' of a type (line 39)
            tangent_1801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 22), triangle_ref_1800, 'tangent')
            # Assigning a type to the variable 'tangent' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'tangent', tangent_1801)
            
            
            # Call to dot(...): (line 40)
            # Processing the call arguments (line 40)
            # Getting the type of 'in_direction' (line 40)
            in_direction_1804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 26), 'in_direction', False)
            # Processing the call keyword arguments (line 40)
            kwargs_1805 = {}
            # Getting the type of 'normal' (line 40)
            normal_1802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'normal', False)
            # Obtaining the member 'dot' of a type (line 40)
            dot_1803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 15), normal_1802, 'dot')
            # Calling dot(args, kwargs) (line 40)
            dot_call_result_1806 = invoke(stypy.reporting.localization.Localization(__file__, 40, 15), dot_1803, *[in_direction_1804], **kwargs_1805)
            
            float_1807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 42), 'float')
            # Applying the binary operator '<' (line 40)
            result_lt_1808 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 15), '<', dot_call_result_1806, float_1807)
            
            # Testing if the type of an if condition is none (line 40)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 40, 12), result_lt_1808):
                pass
            else:
                
                # Testing the type of an if condition (line 40)
                if_condition_1809 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 12), result_lt_1808)
                # Assigning a type to the variable 'if_condition_1809' (line 40)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'if_condition_1809', if_condition_1809)
                # SSA begins for if statement (line 40)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a UnaryOp to a Name (line 41):
                
                # Getting the type of 'normal' (line 41)
                normal_1810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 26), 'normal')
                # Applying the 'usub' unary operator (line 41)
                result___neg___1811 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 25), 'usub', normal_1810)
                
                # Assigning a type to the variable 'normal' (line 41)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'normal', result___neg___1811)
                # SSA join for if statement (line 40)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a BinOp to a Name (line 42):
            # Getting the type of 'tangent' (line 42)
            tangent_1812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 29), 'tangent')
            # Getting the type of 'x' (line 42)
            x_1813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 39), 'x')
            # Applying the binary operator '*' (line 42)
            result_mul_1814 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 29), '*', tangent_1812, x_1813)
            
            
            # Call to cross(...): (line 42)
            # Processing the call arguments (line 42)
            # Getting the type of 'tangent' (line 42)
            tangent_1817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 58), 'tangent', False)
            # Processing the call keyword arguments (line 42)
            kwargs_1818 = {}
            # Getting the type of 'normal' (line 42)
            normal_1815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 45), 'normal', False)
            # Obtaining the member 'cross' of a type (line 42)
            cross_1816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 45), normal_1815, 'cross')
            # Calling cross(args, kwargs) (line 42)
            cross_call_result_1819 = invoke(stypy.reporting.localization.Localization(__file__, 42, 45), cross_1816, *[tangent_1817], **kwargs_1818)
            
            # Getting the type of 'y' (line 42)
            y_1820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 69), 'y')
            # Applying the binary operator '*' (line 42)
            result_mul_1821 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 45), '*', cross_call_result_1819, y_1820)
            
            # Applying the binary operator '+' (line 42)
            result_add_1822 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 28), '+', result_mul_1814, result_mul_1821)
            
            # Getting the type of 'normal' (line 42)
            normal_1823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 75), 'normal')
            # Getting the type of 'z' (line 42)
            z_1824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 84), 'z')
            # Applying the binary operator '*' (line 42)
            result_mul_1825 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 75), '*', normal_1823, z_1824)
            
            # Applying the binary operator '+' (line 42)
            result_add_1826 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 72), '+', result_add_1822, result_mul_1825)
            
            # Assigning a type to the variable 'out_direction' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'out_direction', result_add_1826)
            
            # Obtaining an instance of the builtin type 'tuple' (line 43)
            tuple_1827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 43)
            # Adding element type (line 43)
            # Getting the type of 'out_direction' (line 43)
            out_direction_1828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 19), 'out_direction')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 19), tuple_1827, out_direction_1828)
            # Adding element type (line 43)
            # Getting the type of 'color' (line 43)
            color_1829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 34), 'color')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 19), tuple_1827, color_1829)
            
            # Assigning a type to the variable 'stypy_return_type' (line 43)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'stypy_return_type', tuple_1827)
            # SSA branch for the else part of an if statement (line 31)
            module_type_store.open_ssa_branch('else')
            
            # Obtaining an instance of the builtin type 'tuple' (line 45)
            tuple_1830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 45)
            # Adding element type (line 45)
            # Getting the type of 'ZERO' (line 45)
            ZERO_1831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'ZERO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 19), tuple_1830, ZERO_1831)
            # Adding element type (line 45)
            # Getting the type of 'ZERO' (line 45)
            ZERO_1832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), 'ZERO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 19), tuple_1830, ZERO_1832)
            
            # Assigning a type to the variable 'stypy_return_type' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'stypy_return_type', tuple_1830)
            # SSA join for if statement (line 31)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'get_next_direction(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_next_direction' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_1833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1833)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_next_direction'
        return stypy_return_type_1833


# Assigning a type to the variable 'SurfacePoint' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'SurfacePoint', SurfacePoint)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
