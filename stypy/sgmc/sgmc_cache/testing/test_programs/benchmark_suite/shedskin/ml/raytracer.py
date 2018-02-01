
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #  MiniLight Python : minimal global illumination renderer
2: #
3: #  Copyright (c) 2007-2008, Harrison Ainsworth / HXA7241 and Juraj Sukop.
4: #  http://www.hxa7241.org/
5: 
6: 
7: from surfacepoint import SurfacePoint
8: from vector3f import ZERO
9: 
10: class RayTracer(object):
11: 
12:     def __init__(self, scene):
13:         self.scene_ref = scene
14: 
15:     def get_radiance(self, ray_origin, ray_direction, last_hit=None):
16:         hit_ref, hit_position = self.scene_ref.get_intersection(ray_origin, ray_direction, last_hit)
17:         if hit_ref:
18:             surface_point = SurfacePoint(hit_ref, hit_position)
19:             local_emission = ZERO if last_hit else surface_point.get_emission(ray_origin, -ray_direction, False)
20:             illumination = self.sample_emitters(ray_direction, surface_point)
21:             next_direction, color = surface_point.get_next_direction(-ray_direction)
22:             reflection = ZERO if next_direction.is_zero() else color.mul(self.get_radiance(surface_point.position, next_direction, surface_point.triangle_ref))
23:             return reflection + illumination + local_emission
24:         else:
25:             return self.scene_ref.get_default_emission(-ray_direction)
26: 
27:     def sample_emitters(self, ray_direction, surface_point):
28:         emitter_position, emitter_ref = self.scene_ref.get_emitter()
29:         if emitter_ref:
30:             emit_direction = (emitter_position - surface_point.position).unitize()
31:             hit_ref, p = self.scene_ref.get_intersection(surface_point.position, emit_direction, surface_point.triangle_ref)
32:             emission_in = SurfacePoint(emitter_ref, emitter_position).get_emission(surface_point.position, -emit_direction, True) if not hit_ref or emitter_ref == hit_ref else ZERO
33:             return surface_point.get_reflection(emit_direction, emission_in * self.scene_ref.emitters_count(), -ray_direction)
34:         else:
35:             return ZERO
36: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from surfacepoint import SurfacePoint' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')
import_737 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'surfacepoint')

if (type(import_737) is not StypyTypeError):

    if (import_737 != 'pyd_module'):
        __import__(import_737)
        sys_modules_738 = sys.modules[import_737]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'surfacepoint', sys_modules_738.module_type_store, module_type_store, ['SurfacePoint'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_738, sys_modules_738.module_type_store, module_type_store)
    else:
        from surfacepoint import SurfacePoint

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'surfacepoint', None, module_type_store, ['SurfacePoint'], [SurfacePoint])

else:
    # Assigning a type to the variable 'surfacepoint' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'surfacepoint', import_737)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from vector3f import ZERO' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')
import_739 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'vector3f')

if (type(import_739) is not StypyTypeError):

    if (import_739 != 'pyd_module'):
        __import__(import_739)
        sys_modules_740 = sys.modules[import_739]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'vector3f', sys_modules_740.module_type_store, module_type_store, ['ZERO'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_740, sys_modules_740.module_type_store, module_type_store)
    else:
        from vector3f import ZERO

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'vector3f', None, module_type_store, ['ZERO'], [ZERO])

else:
    # Assigning a type to the variable 'vector3f' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'vector3f', import_739)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')

# Declaration of the 'RayTracer' class

class RayTracer(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 12, 4, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RayTracer.__init__', ['scene'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['scene'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 13):
        
        # Assigning a Name to a Attribute (line 13):
        # Getting the type of 'scene' (line 13)
        scene_741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 25), 'scene')
        # Getting the type of 'self' (line 13)
        self_742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'self')
        # Setting the type of the member 'scene_ref' of a type (line 13)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), self_742, 'scene_ref', scene_741)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_radiance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 15)
        None_743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 63), 'None')
        defaults = [None_743]
        # Create a new context for function 'get_radiance'
        module_type_store = module_type_store.open_function_context('get_radiance', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RayTracer.get_radiance.__dict__.__setitem__('stypy_localization', localization)
        RayTracer.get_radiance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RayTracer.get_radiance.__dict__.__setitem__('stypy_type_store', module_type_store)
        RayTracer.get_radiance.__dict__.__setitem__('stypy_function_name', 'RayTracer.get_radiance')
        RayTracer.get_radiance.__dict__.__setitem__('stypy_param_names_list', ['ray_origin', 'ray_direction', 'last_hit'])
        RayTracer.get_radiance.__dict__.__setitem__('stypy_varargs_param_name', None)
        RayTracer.get_radiance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RayTracer.get_radiance.__dict__.__setitem__('stypy_call_defaults', defaults)
        RayTracer.get_radiance.__dict__.__setitem__('stypy_call_varargs', varargs)
        RayTracer.get_radiance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RayTracer.get_radiance.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RayTracer.get_radiance', ['ray_origin', 'ray_direction', 'last_hit'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_radiance', localization, ['ray_origin', 'ray_direction', 'last_hit'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_radiance(...)' code ##################

        
        # Assigning a Call to a Tuple (line 16):
        
        # Assigning a Call to a Name:
        
        # Call to get_intersection(...): (line 16)
        # Processing the call arguments (line 16)
        # Getting the type of 'ray_origin' (line 16)
        ray_origin_747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 64), 'ray_origin', False)
        # Getting the type of 'ray_direction' (line 16)
        ray_direction_748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 76), 'ray_direction', False)
        # Getting the type of 'last_hit' (line 16)
        last_hit_749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 91), 'last_hit', False)
        # Processing the call keyword arguments (line 16)
        kwargs_750 = {}
        # Getting the type of 'self' (line 16)
        self_744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 32), 'self', False)
        # Obtaining the member 'scene_ref' of a type (line 16)
        scene_ref_745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 32), self_744, 'scene_ref')
        # Obtaining the member 'get_intersection' of a type (line 16)
        get_intersection_746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 32), scene_ref_745, 'get_intersection')
        # Calling get_intersection(args, kwargs) (line 16)
        get_intersection_call_result_751 = invoke(stypy.reporting.localization.Localization(__file__, 16, 32), get_intersection_746, *[ray_origin_747, ray_direction_748, last_hit_749], **kwargs_750)
        
        # Assigning a type to the variable 'call_assignment_725' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'call_assignment_725', get_intersection_call_result_751)
        
        # Assigning a Call to a Name (line 16):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 8), 'int')
        # Processing the call keyword arguments
        kwargs_755 = {}
        # Getting the type of 'call_assignment_725' (line 16)
        call_assignment_725_752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'call_assignment_725', False)
        # Obtaining the member '__getitem__' of a type (line 16)
        getitem___753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), call_assignment_725_752, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_756 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___753, *[int_754], **kwargs_755)
        
        # Assigning a type to the variable 'call_assignment_726' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'call_assignment_726', getitem___call_result_756)
        
        # Assigning a Name to a Name (line 16):
        # Getting the type of 'call_assignment_726' (line 16)
        call_assignment_726_757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'call_assignment_726')
        # Assigning a type to the variable 'hit_ref' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'hit_ref', call_assignment_726_757)
        
        # Assigning a Call to a Name (line 16):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 8), 'int')
        # Processing the call keyword arguments
        kwargs_761 = {}
        # Getting the type of 'call_assignment_725' (line 16)
        call_assignment_725_758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'call_assignment_725', False)
        # Obtaining the member '__getitem__' of a type (line 16)
        getitem___759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), call_assignment_725_758, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_762 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___759, *[int_760], **kwargs_761)
        
        # Assigning a type to the variable 'call_assignment_727' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'call_assignment_727', getitem___call_result_762)
        
        # Assigning a Name to a Name (line 16):
        # Getting the type of 'call_assignment_727' (line 16)
        call_assignment_727_763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'call_assignment_727')
        # Assigning a type to the variable 'hit_position' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 17), 'hit_position', call_assignment_727_763)
        # Getting the type of 'hit_ref' (line 17)
        hit_ref_764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 11), 'hit_ref')
        # Testing if the type of an if condition is none (line 17)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 17, 8), hit_ref_764):
            
            # Call to get_default_emission(...): (line 25)
            # Processing the call arguments (line 25)
            
            # Getting the type of 'ray_direction' (line 25)
            ray_direction_833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 56), 'ray_direction', False)
            # Applying the 'usub' unary operator (line 25)
            result___neg___834 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 55), 'usub', ray_direction_833)
            
            # Processing the call keyword arguments (line 25)
            kwargs_835 = {}
            # Getting the type of 'self' (line 25)
            self_830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'self', False)
            # Obtaining the member 'scene_ref' of a type (line 25)
            scene_ref_831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 19), self_830, 'scene_ref')
            # Obtaining the member 'get_default_emission' of a type (line 25)
            get_default_emission_832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 19), scene_ref_831, 'get_default_emission')
            # Calling get_default_emission(args, kwargs) (line 25)
            get_default_emission_call_result_836 = invoke(stypy.reporting.localization.Localization(__file__, 25, 19), get_default_emission_832, *[result___neg___834], **kwargs_835)
            
            # Assigning a type to the variable 'stypy_return_type' (line 25)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'stypy_return_type', get_default_emission_call_result_836)
        else:
            
            # Testing the type of an if condition (line 17)
            if_condition_765 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 8), hit_ref_764)
            # Assigning a type to the variable 'if_condition_765' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'if_condition_765', if_condition_765)
            # SSA begins for if statement (line 17)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 18):
            
            # Assigning a Call to a Name (line 18):
            
            # Call to SurfacePoint(...): (line 18)
            # Processing the call arguments (line 18)
            # Getting the type of 'hit_ref' (line 18)
            hit_ref_767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 41), 'hit_ref', False)
            # Getting the type of 'hit_position' (line 18)
            hit_position_768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 50), 'hit_position', False)
            # Processing the call keyword arguments (line 18)
            kwargs_769 = {}
            # Getting the type of 'SurfacePoint' (line 18)
            SurfacePoint_766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 28), 'SurfacePoint', False)
            # Calling SurfacePoint(args, kwargs) (line 18)
            SurfacePoint_call_result_770 = invoke(stypy.reporting.localization.Localization(__file__, 18, 28), SurfacePoint_766, *[hit_ref_767, hit_position_768], **kwargs_769)
            
            # Assigning a type to the variable 'surface_point' (line 18)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'surface_point', SurfacePoint_call_result_770)
            
            # Assigning a IfExp to a Name (line 19):
            
            # Assigning a IfExp to a Name (line 19):
            
            # Getting the type of 'last_hit' (line 19)
            last_hit_771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 37), 'last_hit')
            # Testing the type of an if expression (line 19)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 19, 29), last_hit_771)
            # SSA begins for if expression (line 19)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
            # Getting the type of 'ZERO' (line 19)
            ZERO_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 29), 'ZERO')
            # SSA branch for the else part of an if expression (line 19)
            module_type_store.open_ssa_branch('if expression else')
            
            # Call to get_emission(...): (line 19)
            # Processing the call arguments (line 19)
            # Getting the type of 'ray_origin' (line 19)
            ray_origin_775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 78), 'ray_origin', False)
            
            # Getting the type of 'ray_direction' (line 19)
            ray_direction_776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 91), 'ray_direction', False)
            # Applying the 'usub' unary operator (line 19)
            result___neg___777 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 90), 'usub', ray_direction_776)
            
            # Getting the type of 'False' (line 19)
            False_778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 106), 'False', False)
            # Processing the call keyword arguments (line 19)
            kwargs_779 = {}
            # Getting the type of 'surface_point' (line 19)
            surface_point_773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 51), 'surface_point', False)
            # Obtaining the member 'get_emission' of a type (line 19)
            get_emission_774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 51), surface_point_773, 'get_emission')
            # Calling get_emission(args, kwargs) (line 19)
            get_emission_call_result_780 = invoke(stypy.reporting.localization.Localization(__file__, 19, 51), get_emission_774, *[ray_origin_775, result___neg___777, False_778], **kwargs_779)
            
            # SSA join for if expression (line 19)
            module_type_store = module_type_store.join_ssa_context()
            if_exp_781 = union_type.UnionType.add(ZERO_772, get_emission_call_result_780)
            
            # Assigning a type to the variable 'local_emission' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'local_emission', if_exp_781)
            
            # Assigning a Call to a Name (line 20):
            
            # Assigning a Call to a Name (line 20):
            
            # Call to sample_emitters(...): (line 20)
            # Processing the call arguments (line 20)
            # Getting the type of 'ray_direction' (line 20)
            ray_direction_784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 48), 'ray_direction', False)
            # Getting the type of 'surface_point' (line 20)
            surface_point_785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 63), 'surface_point', False)
            # Processing the call keyword arguments (line 20)
            kwargs_786 = {}
            # Getting the type of 'self' (line 20)
            self_782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 27), 'self', False)
            # Obtaining the member 'sample_emitters' of a type (line 20)
            sample_emitters_783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 27), self_782, 'sample_emitters')
            # Calling sample_emitters(args, kwargs) (line 20)
            sample_emitters_call_result_787 = invoke(stypy.reporting.localization.Localization(__file__, 20, 27), sample_emitters_783, *[ray_direction_784, surface_point_785], **kwargs_786)
            
            # Assigning a type to the variable 'illumination' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'illumination', sample_emitters_call_result_787)
            
            # Assigning a Call to a Tuple (line 21):
            
            # Assigning a Call to a Name:
            
            # Call to get_next_direction(...): (line 21)
            # Processing the call arguments (line 21)
            
            # Getting the type of 'ray_direction' (line 21)
            ray_direction_790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 70), 'ray_direction', False)
            # Applying the 'usub' unary operator (line 21)
            result___neg___791 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 69), 'usub', ray_direction_790)
            
            # Processing the call keyword arguments (line 21)
            kwargs_792 = {}
            # Getting the type of 'surface_point' (line 21)
            surface_point_788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 36), 'surface_point', False)
            # Obtaining the member 'get_next_direction' of a type (line 21)
            get_next_direction_789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 36), surface_point_788, 'get_next_direction')
            # Calling get_next_direction(args, kwargs) (line 21)
            get_next_direction_call_result_793 = invoke(stypy.reporting.localization.Localization(__file__, 21, 36), get_next_direction_789, *[result___neg___791], **kwargs_792)
            
            # Assigning a type to the variable 'call_assignment_728' (line 21)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'call_assignment_728', get_next_direction_call_result_793)
            
            # Assigning a Call to a Name (line 21):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 12), 'int')
            # Processing the call keyword arguments
            kwargs_797 = {}
            # Getting the type of 'call_assignment_728' (line 21)
            call_assignment_728_794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'call_assignment_728', False)
            # Obtaining the member '__getitem__' of a type (line 21)
            getitem___795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 12), call_assignment_728_794, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_798 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___795, *[int_796], **kwargs_797)
            
            # Assigning a type to the variable 'call_assignment_729' (line 21)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'call_assignment_729', getitem___call_result_798)
            
            # Assigning a Name to a Name (line 21):
            # Getting the type of 'call_assignment_729' (line 21)
            call_assignment_729_799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'call_assignment_729')
            # Assigning a type to the variable 'next_direction' (line 21)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'next_direction', call_assignment_729_799)
            
            # Assigning a Call to a Name (line 21):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 12), 'int')
            # Processing the call keyword arguments
            kwargs_803 = {}
            # Getting the type of 'call_assignment_728' (line 21)
            call_assignment_728_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'call_assignment_728', False)
            # Obtaining the member '__getitem__' of a type (line 21)
            getitem___801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 12), call_assignment_728_800, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_804 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___801, *[int_802], **kwargs_803)
            
            # Assigning a type to the variable 'call_assignment_730' (line 21)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'call_assignment_730', getitem___call_result_804)
            
            # Assigning a Name to a Name (line 21):
            # Getting the type of 'call_assignment_730' (line 21)
            call_assignment_730_805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'call_assignment_730')
            # Assigning a type to the variable 'color' (line 21)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 28), 'color', call_assignment_730_805)
            
            # Assigning a IfExp to a Name (line 22):
            
            # Assigning a IfExp to a Name (line 22):
            
            
            # Call to is_zero(...): (line 22)
            # Processing the call keyword arguments (line 22)
            kwargs_808 = {}
            # Getting the type of 'next_direction' (line 22)
            next_direction_806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 33), 'next_direction', False)
            # Obtaining the member 'is_zero' of a type (line 22)
            is_zero_807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 33), next_direction_806, 'is_zero')
            # Calling is_zero(args, kwargs) (line 22)
            is_zero_call_result_809 = invoke(stypy.reporting.localization.Localization(__file__, 22, 33), is_zero_807, *[], **kwargs_808)
            
            # Testing the type of an if expression (line 22)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 25), is_zero_call_result_809)
            # SSA begins for if expression (line 22)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
            # Getting the type of 'ZERO' (line 22)
            ZERO_810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), 'ZERO')
            # SSA branch for the else part of an if expression (line 22)
            module_type_store.open_ssa_branch('if expression else')
            
            # Call to mul(...): (line 22)
            # Processing the call arguments (line 22)
            
            # Call to get_radiance(...): (line 22)
            # Processing the call arguments (line 22)
            # Getting the type of 'surface_point' (line 22)
            surface_point_815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 91), 'surface_point', False)
            # Obtaining the member 'position' of a type (line 22)
            position_816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 91), surface_point_815, 'position')
            # Getting the type of 'next_direction' (line 22)
            next_direction_817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 115), 'next_direction', False)
            # Getting the type of 'surface_point' (line 22)
            surface_point_818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 131), 'surface_point', False)
            # Obtaining the member 'triangle_ref' of a type (line 22)
            triangle_ref_819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 131), surface_point_818, 'triangle_ref')
            # Processing the call keyword arguments (line 22)
            kwargs_820 = {}
            # Getting the type of 'self' (line 22)
            self_813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 73), 'self', False)
            # Obtaining the member 'get_radiance' of a type (line 22)
            get_radiance_814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 73), self_813, 'get_radiance')
            # Calling get_radiance(args, kwargs) (line 22)
            get_radiance_call_result_821 = invoke(stypy.reporting.localization.Localization(__file__, 22, 73), get_radiance_814, *[position_816, next_direction_817, triangle_ref_819], **kwargs_820)
            
            # Processing the call keyword arguments (line 22)
            kwargs_822 = {}
            # Getting the type of 'color' (line 22)
            color_811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 63), 'color', False)
            # Obtaining the member 'mul' of a type (line 22)
            mul_812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 63), color_811, 'mul')
            # Calling mul(args, kwargs) (line 22)
            mul_call_result_823 = invoke(stypy.reporting.localization.Localization(__file__, 22, 63), mul_812, *[get_radiance_call_result_821], **kwargs_822)
            
            # SSA join for if expression (line 22)
            module_type_store = module_type_store.join_ssa_context()
            if_exp_824 = union_type.UnionType.add(ZERO_810, mul_call_result_823)
            
            # Assigning a type to the variable 'reflection' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'reflection', if_exp_824)
            # Getting the type of 'reflection' (line 23)
            reflection_825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'reflection')
            # Getting the type of 'illumination' (line 23)
            illumination_826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 32), 'illumination')
            # Applying the binary operator '+' (line 23)
            result_add_827 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 19), '+', reflection_825, illumination_826)
            
            # Getting the type of 'local_emission' (line 23)
            local_emission_828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 47), 'local_emission')
            # Applying the binary operator '+' (line 23)
            result_add_829 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 45), '+', result_add_827, local_emission_828)
            
            # Assigning a type to the variable 'stypy_return_type' (line 23)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'stypy_return_type', result_add_829)
            # SSA branch for the else part of an if statement (line 17)
            module_type_store.open_ssa_branch('else')
            
            # Call to get_default_emission(...): (line 25)
            # Processing the call arguments (line 25)
            
            # Getting the type of 'ray_direction' (line 25)
            ray_direction_833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 56), 'ray_direction', False)
            # Applying the 'usub' unary operator (line 25)
            result___neg___834 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 55), 'usub', ray_direction_833)
            
            # Processing the call keyword arguments (line 25)
            kwargs_835 = {}
            # Getting the type of 'self' (line 25)
            self_830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'self', False)
            # Obtaining the member 'scene_ref' of a type (line 25)
            scene_ref_831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 19), self_830, 'scene_ref')
            # Obtaining the member 'get_default_emission' of a type (line 25)
            get_default_emission_832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 19), scene_ref_831, 'get_default_emission')
            # Calling get_default_emission(args, kwargs) (line 25)
            get_default_emission_call_result_836 = invoke(stypy.reporting.localization.Localization(__file__, 25, 19), get_default_emission_832, *[result___neg___834], **kwargs_835)
            
            # Assigning a type to the variable 'stypy_return_type' (line 25)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'stypy_return_type', get_default_emission_call_result_836)
            # SSA join for if statement (line 17)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'get_radiance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_radiance' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_837)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_radiance'
        return stypy_return_type_837


    @norecursion
    def sample_emitters(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sample_emitters'
        module_type_store = module_type_store.open_function_context('sample_emitters', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RayTracer.sample_emitters.__dict__.__setitem__('stypy_localization', localization)
        RayTracer.sample_emitters.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RayTracer.sample_emitters.__dict__.__setitem__('stypy_type_store', module_type_store)
        RayTracer.sample_emitters.__dict__.__setitem__('stypy_function_name', 'RayTracer.sample_emitters')
        RayTracer.sample_emitters.__dict__.__setitem__('stypy_param_names_list', ['ray_direction', 'surface_point'])
        RayTracer.sample_emitters.__dict__.__setitem__('stypy_varargs_param_name', None)
        RayTracer.sample_emitters.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RayTracer.sample_emitters.__dict__.__setitem__('stypy_call_defaults', defaults)
        RayTracer.sample_emitters.__dict__.__setitem__('stypy_call_varargs', varargs)
        RayTracer.sample_emitters.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RayTracer.sample_emitters.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RayTracer.sample_emitters', ['ray_direction', 'surface_point'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sample_emitters', localization, ['ray_direction', 'surface_point'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sample_emitters(...)' code ##################

        
        # Assigning a Call to a Tuple (line 28):
        
        # Assigning a Call to a Name:
        
        # Call to get_emitter(...): (line 28)
        # Processing the call keyword arguments (line 28)
        kwargs_841 = {}
        # Getting the type of 'self' (line 28)
        self_838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 40), 'self', False)
        # Obtaining the member 'scene_ref' of a type (line 28)
        scene_ref_839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 40), self_838, 'scene_ref')
        # Obtaining the member 'get_emitter' of a type (line 28)
        get_emitter_840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 40), scene_ref_839, 'get_emitter')
        # Calling get_emitter(args, kwargs) (line 28)
        get_emitter_call_result_842 = invoke(stypy.reporting.localization.Localization(__file__, 28, 40), get_emitter_840, *[], **kwargs_841)
        
        # Assigning a type to the variable 'call_assignment_731' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'call_assignment_731', get_emitter_call_result_842)
        
        # Assigning a Call to a Name (line 28):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'int')
        # Processing the call keyword arguments
        kwargs_846 = {}
        # Getting the type of 'call_assignment_731' (line 28)
        call_assignment_731_843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'call_assignment_731', False)
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), call_assignment_731_843, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_847 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___844, *[int_845], **kwargs_846)
        
        # Assigning a type to the variable 'call_assignment_732' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'call_assignment_732', getitem___call_result_847)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'call_assignment_732' (line 28)
        call_assignment_732_848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'call_assignment_732')
        # Assigning a type to the variable 'emitter_position' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'emitter_position', call_assignment_732_848)
        
        # Assigning a Call to a Name (line 28):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'int')
        # Processing the call keyword arguments
        kwargs_852 = {}
        # Getting the type of 'call_assignment_731' (line 28)
        call_assignment_731_849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'call_assignment_731', False)
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), call_assignment_731_849, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_853 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___850, *[int_851], **kwargs_852)
        
        # Assigning a type to the variable 'call_assignment_733' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'call_assignment_733', getitem___call_result_853)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'call_assignment_733' (line 28)
        call_assignment_733_854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'call_assignment_733')
        # Assigning a type to the variable 'emitter_ref' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 26), 'emitter_ref', call_assignment_733_854)
        # Getting the type of 'emitter_ref' (line 29)
        emitter_ref_855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 'emitter_ref')
        # Testing if the type of an if condition is none (line 29)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 29, 8), emitter_ref_855):
            # Getting the type of 'ZERO' (line 35)
            ZERO_921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), 'ZERO')
            # Assigning a type to the variable 'stypy_return_type' (line 35)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'stypy_return_type', ZERO_921)
        else:
            
            # Testing the type of an if condition (line 29)
            if_condition_856 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 8), emitter_ref_855)
            # Assigning a type to the variable 'if_condition_856' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'if_condition_856', if_condition_856)
            # SSA begins for if statement (line 29)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 30):
            
            # Assigning a Call to a Name (line 30):
            
            # Call to unitize(...): (line 30)
            # Processing the call keyword arguments (line 30)
            kwargs_862 = {}
            # Getting the type of 'emitter_position' (line 30)
            emitter_position_857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 30), 'emitter_position', False)
            # Getting the type of 'surface_point' (line 30)
            surface_point_858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 49), 'surface_point', False)
            # Obtaining the member 'position' of a type (line 30)
            position_859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 49), surface_point_858, 'position')
            # Applying the binary operator '-' (line 30)
            result_sub_860 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 30), '-', emitter_position_857, position_859)
            
            # Obtaining the member 'unitize' of a type (line 30)
            unitize_861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 30), result_sub_860, 'unitize')
            # Calling unitize(args, kwargs) (line 30)
            unitize_call_result_863 = invoke(stypy.reporting.localization.Localization(__file__, 30, 30), unitize_861, *[], **kwargs_862)
            
            # Assigning a type to the variable 'emit_direction' (line 30)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'emit_direction', unitize_call_result_863)
            
            # Assigning a Call to a Tuple (line 31):
            
            # Assigning a Call to a Name:
            
            # Call to get_intersection(...): (line 31)
            # Processing the call arguments (line 31)
            # Getting the type of 'surface_point' (line 31)
            surface_point_867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 57), 'surface_point', False)
            # Obtaining the member 'position' of a type (line 31)
            position_868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 57), surface_point_867, 'position')
            # Getting the type of 'emit_direction' (line 31)
            emit_direction_869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 81), 'emit_direction', False)
            # Getting the type of 'surface_point' (line 31)
            surface_point_870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 97), 'surface_point', False)
            # Obtaining the member 'triangle_ref' of a type (line 31)
            triangle_ref_871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 97), surface_point_870, 'triangle_ref')
            # Processing the call keyword arguments (line 31)
            kwargs_872 = {}
            # Getting the type of 'self' (line 31)
            self_864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 25), 'self', False)
            # Obtaining the member 'scene_ref' of a type (line 31)
            scene_ref_865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 25), self_864, 'scene_ref')
            # Obtaining the member 'get_intersection' of a type (line 31)
            get_intersection_866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 25), scene_ref_865, 'get_intersection')
            # Calling get_intersection(args, kwargs) (line 31)
            get_intersection_call_result_873 = invoke(stypy.reporting.localization.Localization(__file__, 31, 25), get_intersection_866, *[position_868, emit_direction_869, triangle_ref_871], **kwargs_872)
            
            # Assigning a type to the variable 'call_assignment_734' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'call_assignment_734', get_intersection_call_result_873)
            
            # Assigning a Call to a Name (line 31):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 12), 'int')
            # Processing the call keyword arguments
            kwargs_877 = {}
            # Getting the type of 'call_assignment_734' (line 31)
            call_assignment_734_874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'call_assignment_734', False)
            # Obtaining the member '__getitem__' of a type (line 31)
            getitem___875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), call_assignment_734_874, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_878 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___875, *[int_876], **kwargs_877)
            
            # Assigning a type to the variable 'call_assignment_735' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'call_assignment_735', getitem___call_result_878)
            
            # Assigning a Name to a Name (line 31):
            # Getting the type of 'call_assignment_735' (line 31)
            call_assignment_735_879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'call_assignment_735')
            # Assigning a type to the variable 'hit_ref' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'hit_ref', call_assignment_735_879)
            
            # Assigning a Call to a Name (line 31):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 12), 'int')
            # Processing the call keyword arguments
            kwargs_883 = {}
            # Getting the type of 'call_assignment_734' (line 31)
            call_assignment_734_880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'call_assignment_734', False)
            # Obtaining the member '__getitem__' of a type (line 31)
            getitem___881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), call_assignment_734_880, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_884 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___881, *[int_882], **kwargs_883)
            
            # Assigning a type to the variable 'call_assignment_736' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'call_assignment_736', getitem___call_result_884)
            
            # Assigning a Name to a Name (line 31):
            # Getting the type of 'call_assignment_736' (line 31)
            call_assignment_736_885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'call_assignment_736')
            # Assigning a type to the variable 'p' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 21), 'p', call_assignment_736_885)
            
            # Assigning a IfExp to a Name (line 32):
            
            # Assigning a IfExp to a Name (line 32):
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'hit_ref' (line 32)
            hit_ref_886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 137), 'hit_ref')
            # Applying the 'not' unary operator (line 32)
            result_not__887 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 133), 'not', hit_ref_886)
            
            
            # Getting the type of 'emitter_ref' (line 32)
            emitter_ref_888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 148), 'emitter_ref')
            # Getting the type of 'hit_ref' (line 32)
            hit_ref_889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 163), 'hit_ref')
            # Applying the binary operator '==' (line 32)
            result_eq_890 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 148), '==', emitter_ref_888, hit_ref_889)
            
            # Applying the binary operator 'or' (line 32)
            result_or_keyword_891 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 133), 'or', result_not__887, result_eq_890)
            
            # Testing the type of an if expression (line 32)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 26), result_or_keyword_891)
            # SSA begins for if expression (line 32)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
            
            # Call to get_emission(...): (line 32)
            # Processing the call arguments (line 32)
            # Getting the type of 'surface_point' (line 32)
            surface_point_898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 83), 'surface_point', False)
            # Obtaining the member 'position' of a type (line 32)
            position_899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 83), surface_point_898, 'position')
            
            # Getting the type of 'emit_direction' (line 32)
            emit_direction_900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 108), 'emit_direction', False)
            # Applying the 'usub' unary operator (line 32)
            result___neg___901 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 107), 'usub', emit_direction_900)
            
            # Getting the type of 'True' (line 32)
            True_902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 124), 'True', False)
            # Processing the call keyword arguments (line 32)
            kwargs_903 = {}
            
            # Call to SurfacePoint(...): (line 32)
            # Processing the call arguments (line 32)
            # Getting the type of 'emitter_ref' (line 32)
            emitter_ref_893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 39), 'emitter_ref', False)
            # Getting the type of 'emitter_position' (line 32)
            emitter_position_894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 52), 'emitter_position', False)
            # Processing the call keyword arguments (line 32)
            kwargs_895 = {}
            # Getting the type of 'SurfacePoint' (line 32)
            SurfacePoint_892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 26), 'SurfacePoint', False)
            # Calling SurfacePoint(args, kwargs) (line 32)
            SurfacePoint_call_result_896 = invoke(stypy.reporting.localization.Localization(__file__, 32, 26), SurfacePoint_892, *[emitter_ref_893, emitter_position_894], **kwargs_895)
            
            # Obtaining the member 'get_emission' of a type (line 32)
            get_emission_897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 26), SurfacePoint_call_result_896, 'get_emission')
            # Calling get_emission(args, kwargs) (line 32)
            get_emission_call_result_904 = invoke(stypy.reporting.localization.Localization(__file__, 32, 26), get_emission_897, *[position_899, result___neg___901, True_902], **kwargs_903)
            
            # SSA branch for the else part of an if expression (line 32)
            module_type_store.open_ssa_branch('if expression else')
            # Getting the type of 'ZERO' (line 32)
            ZERO_905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 176), 'ZERO')
            # SSA join for if expression (line 32)
            module_type_store = module_type_store.join_ssa_context()
            if_exp_906 = union_type.UnionType.add(get_emission_call_result_904, ZERO_905)
            
            # Assigning a type to the variable 'emission_in' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'emission_in', if_exp_906)
            
            # Call to get_reflection(...): (line 33)
            # Processing the call arguments (line 33)
            # Getting the type of 'emit_direction' (line 33)
            emit_direction_909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 48), 'emit_direction', False)
            # Getting the type of 'emission_in' (line 33)
            emission_in_910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 64), 'emission_in', False)
            
            # Call to emitters_count(...): (line 33)
            # Processing the call keyword arguments (line 33)
            kwargs_914 = {}
            # Getting the type of 'self' (line 33)
            self_911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 78), 'self', False)
            # Obtaining the member 'scene_ref' of a type (line 33)
            scene_ref_912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 78), self_911, 'scene_ref')
            # Obtaining the member 'emitters_count' of a type (line 33)
            emitters_count_913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 78), scene_ref_912, 'emitters_count')
            # Calling emitters_count(args, kwargs) (line 33)
            emitters_count_call_result_915 = invoke(stypy.reporting.localization.Localization(__file__, 33, 78), emitters_count_913, *[], **kwargs_914)
            
            # Applying the binary operator '*' (line 33)
            result_mul_916 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 64), '*', emission_in_910, emitters_count_call_result_915)
            
            
            # Getting the type of 'ray_direction' (line 33)
            ray_direction_917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 112), 'ray_direction', False)
            # Applying the 'usub' unary operator (line 33)
            result___neg___918 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 111), 'usub', ray_direction_917)
            
            # Processing the call keyword arguments (line 33)
            kwargs_919 = {}
            # Getting the type of 'surface_point' (line 33)
            surface_point_907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'surface_point', False)
            # Obtaining the member 'get_reflection' of a type (line 33)
            get_reflection_908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 19), surface_point_907, 'get_reflection')
            # Calling get_reflection(args, kwargs) (line 33)
            get_reflection_call_result_920 = invoke(stypy.reporting.localization.Localization(__file__, 33, 19), get_reflection_908, *[emit_direction_909, result_mul_916, result___neg___918], **kwargs_919)
            
            # Assigning a type to the variable 'stypy_return_type' (line 33)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'stypy_return_type', get_reflection_call_result_920)
            # SSA branch for the else part of an if statement (line 29)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'ZERO' (line 35)
            ZERO_921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), 'ZERO')
            # Assigning a type to the variable 'stypy_return_type' (line 35)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'stypy_return_type', ZERO_921)
            # SSA join for if statement (line 29)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'sample_emitters(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sample_emitters' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_922)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sample_emitters'
        return stypy_return_type_922


# Assigning a type to the variable 'RayTracer' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'RayTracer', RayTracer)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
