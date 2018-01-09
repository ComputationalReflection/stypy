
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: This module is to support *bbox_inches* option in savefig command.
3: '''
4: 
5: from __future__ import (absolute_import, division, print_function,
6:                         unicode_literals)
7: 
8: import six
9: 
10: import warnings
11: from matplotlib.transforms import Bbox, TransformedBbox, Affine2D
12: 
13: 
14: def adjust_bbox(fig, bbox_inches, fixed_dpi=None):
15:     '''
16:     Temporarily adjust the figure so that only the specified area
17:     (bbox_inches) is saved.
18: 
19:     It modifies fig.bbox, fig.bbox_inches,
20:     fig.transFigure._boxout, and fig.patch.  While the figure size
21:     changes, the scale of the original figure is conserved.  A
22:     function which restores the original values are returned.
23:     '''
24: 
25:     origBbox = fig.bbox
26:     origBboxInches = fig.bbox_inches
27:     _boxout = fig.transFigure._boxout
28: 
29:     asp_list = []
30:     locator_list = []
31:     for ax in fig.axes:
32:         pos = ax.get_position(original=False).frozen()
33:         locator_list.append(ax.get_axes_locator())
34:         asp_list.append(ax.get_aspect())
35: 
36:         def _l(a, r, pos=pos):
37:             return pos
38:         ax.set_axes_locator(_l)
39:         ax.set_aspect("auto")
40: 
41:     def restore_bbox():
42: 
43:         for ax, asp, loc in zip(fig.axes, asp_list, locator_list):
44:             ax.set_aspect(asp)
45:             ax.set_axes_locator(loc)
46: 
47:         fig.bbox = origBbox
48:         fig.bbox_inches = origBboxInches
49:         fig.transFigure._boxout = _boxout
50:         fig.transFigure.invalidate()
51:         fig.patch.set_bounds(0, 0, 1, 1)
52: 
53:     if fixed_dpi is not None:
54:         tr = Affine2D().scale(fixed_dpi)
55:         dpi_scale = fixed_dpi / fig.dpi
56:     else:
57:         tr = Affine2D().scale(fig.dpi)
58:         dpi_scale = 1.
59: 
60:     _bbox = TransformedBbox(bbox_inches, tr)
61: 
62:     fig.bbox_inches = Bbox.from_bounds(0, 0,
63:                                        bbox_inches.width, bbox_inches.height)
64:     x0, y0 = _bbox.x0, _bbox.y0
65:     w1, h1 = fig.bbox.width * dpi_scale, fig.bbox.height * dpi_scale
66:     fig.transFigure._boxout = Bbox.from_bounds(-x0, -y0, w1, h1)
67:     fig.transFigure.invalidate()
68: 
69:     fig.bbox = TransformedBbox(fig.bbox_inches, tr)
70: 
71:     fig.patch.set_bounds(x0 / w1, y0 / h1,
72:                          fig.bbox.width / w1, fig.bbox.height / h1)
73: 
74:     return restore_bbox
75: 
76: 
77: def process_figure_for_rasterizing(fig, bbox_inches_restore, fixed_dpi=None):
78:     '''
79:     This need to be called when figure dpi changes during the drawing
80:     (e.g., rasterizing). It recovers the bbox and re-adjust it with
81:     the new dpi.
82:     '''
83: 
84:     bbox_inches, restore_bbox = bbox_inches_restore
85:     restore_bbox()
86:     r = adjust_bbox(fig, bbox_inches, fixed_dpi)
87: 
88:     return bbox_inches, r
89: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_152776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'unicode', u'\nThis module is to support *bbox_inches* option in savefig command.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import six' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_152777 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six')

if (type(import_152777) is not StypyTypeError):

    if (import_152777 != 'pyd_module'):
        __import__(import_152777)
        sys_modules_152778 = sys.modules[import_152777]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', sys_modules_152778.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', import_152777)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import warnings' statement (line 10)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from matplotlib.transforms import Bbox, TransformedBbox, Affine2D' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_152779 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.transforms')

if (type(import_152779) is not StypyTypeError):

    if (import_152779 != 'pyd_module'):
        __import__(import_152779)
        sys_modules_152780 = sys.modules[import_152779]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.transforms', sys_modules_152780.module_type_store, module_type_store, ['Bbox', 'TransformedBbox', 'Affine2D'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_152780, sys_modules_152780.module_type_store, module_type_store)
    else:
        from matplotlib.transforms import Bbox, TransformedBbox, Affine2D

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.transforms', None, module_type_store, ['Bbox', 'TransformedBbox', 'Affine2D'], [Bbox, TransformedBbox, Affine2D])

else:
    # Assigning a type to the variable 'matplotlib.transforms' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.transforms', import_152779)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')


@norecursion
def adjust_bbox(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 14)
    None_152781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 44), 'None')
    defaults = [None_152781]
    # Create a new context for function 'adjust_bbox'
    module_type_store = module_type_store.open_function_context('adjust_bbox', 14, 0, False)
    
    # Passed parameters checking function
    adjust_bbox.stypy_localization = localization
    adjust_bbox.stypy_type_of_self = None
    adjust_bbox.stypy_type_store = module_type_store
    adjust_bbox.stypy_function_name = 'adjust_bbox'
    adjust_bbox.stypy_param_names_list = ['fig', 'bbox_inches', 'fixed_dpi']
    adjust_bbox.stypy_varargs_param_name = None
    adjust_bbox.stypy_kwargs_param_name = None
    adjust_bbox.stypy_call_defaults = defaults
    adjust_bbox.stypy_call_varargs = varargs
    adjust_bbox.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'adjust_bbox', ['fig', 'bbox_inches', 'fixed_dpi'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'adjust_bbox', localization, ['fig', 'bbox_inches', 'fixed_dpi'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'adjust_bbox(...)' code ##################

    unicode_152782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, (-1)), 'unicode', u'\n    Temporarily adjust the figure so that only the specified area\n    (bbox_inches) is saved.\n\n    It modifies fig.bbox, fig.bbox_inches,\n    fig.transFigure._boxout, and fig.patch.  While the figure size\n    changes, the scale of the original figure is conserved.  A\n    function which restores the original values are returned.\n    ')
    
    # Assigning a Attribute to a Name (line 25):
    
    # Assigning a Attribute to a Name (line 25):
    # Getting the type of 'fig' (line 25)
    fig_152783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'fig')
    # Obtaining the member 'bbox' of a type (line 25)
    bbox_152784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 15), fig_152783, 'bbox')
    # Assigning a type to the variable 'origBbox' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'origBbox', bbox_152784)
    
    # Assigning a Attribute to a Name (line 26):
    
    # Assigning a Attribute to a Name (line 26):
    # Getting the type of 'fig' (line 26)
    fig_152785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 21), 'fig')
    # Obtaining the member 'bbox_inches' of a type (line 26)
    bbox_inches_152786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 21), fig_152785, 'bbox_inches')
    # Assigning a type to the variable 'origBboxInches' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'origBboxInches', bbox_inches_152786)
    
    # Assigning a Attribute to a Name (line 27):
    
    # Assigning a Attribute to a Name (line 27):
    # Getting the type of 'fig' (line 27)
    fig_152787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 14), 'fig')
    # Obtaining the member 'transFigure' of a type (line 27)
    transFigure_152788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 14), fig_152787, 'transFigure')
    # Obtaining the member '_boxout' of a type (line 27)
    _boxout_152789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 14), transFigure_152788, '_boxout')
    # Assigning a type to the variable '_boxout' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), '_boxout', _boxout_152789)
    
    # Assigning a List to a Name (line 29):
    
    # Assigning a List to a Name (line 29):
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_152790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    
    # Assigning a type to the variable 'asp_list' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'asp_list', list_152790)
    
    # Assigning a List to a Name (line 30):
    
    # Assigning a List to a Name (line 30):
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_152791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    
    # Assigning a type to the variable 'locator_list' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'locator_list', list_152791)
    
    # Getting the type of 'fig' (line 31)
    fig_152792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'fig')
    # Obtaining the member 'axes' of a type (line 31)
    axes_152793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 14), fig_152792, 'axes')
    # Testing the type of a for loop iterable (line 31)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 31, 4), axes_152793)
    # Getting the type of the for loop variable (line 31)
    for_loop_var_152794 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 31, 4), axes_152793)
    # Assigning a type to the variable 'ax' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'ax', for_loop_var_152794)
    # SSA begins for a for statement (line 31)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 32):
    
    # Assigning a Call to a Name (line 32):
    
    # Call to frozen(...): (line 32)
    # Processing the call keyword arguments (line 32)
    kwargs_152802 = {}
    
    # Call to get_position(...): (line 32)
    # Processing the call keyword arguments (line 32)
    # Getting the type of 'False' (line 32)
    False_152797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 39), 'False', False)
    keyword_152798 = False_152797
    kwargs_152799 = {'original': keyword_152798}
    # Getting the type of 'ax' (line 32)
    ax_152795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 14), 'ax', False)
    # Obtaining the member 'get_position' of a type (line 32)
    get_position_152796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 14), ax_152795, 'get_position')
    # Calling get_position(args, kwargs) (line 32)
    get_position_call_result_152800 = invoke(stypy.reporting.localization.Localization(__file__, 32, 14), get_position_152796, *[], **kwargs_152799)
    
    # Obtaining the member 'frozen' of a type (line 32)
    frozen_152801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 14), get_position_call_result_152800, 'frozen')
    # Calling frozen(args, kwargs) (line 32)
    frozen_call_result_152803 = invoke(stypy.reporting.localization.Localization(__file__, 32, 14), frozen_152801, *[], **kwargs_152802)
    
    # Assigning a type to the variable 'pos' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'pos', frozen_call_result_152803)
    
    # Call to append(...): (line 33)
    # Processing the call arguments (line 33)
    
    # Call to get_axes_locator(...): (line 33)
    # Processing the call keyword arguments (line 33)
    kwargs_152808 = {}
    # Getting the type of 'ax' (line 33)
    ax_152806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 28), 'ax', False)
    # Obtaining the member 'get_axes_locator' of a type (line 33)
    get_axes_locator_152807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 28), ax_152806, 'get_axes_locator')
    # Calling get_axes_locator(args, kwargs) (line 33)
    get_axes_locator_call_result_152809 = invoke(stypy.reporting.localization.Localization(__file__, 33, 28), get_axes_locator_152807, *[], **kwargs_152808)
    
    # Processing the call keyword arguments (line 33)
    kwargs_152810 = {}
    # Getting the type of 'locator_list' (line 33)
    locator_list_152804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'locator_list', False)
    # Obtaining the member 'append' of a type (line 33)
    append_152805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), locator_list_152804, 'append')
    # Calling append(args, kwargs) (line 33)
    append_call_result_152811 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), append_152805, *[get_axes_locator_call_result_152809], **kwargs_152810)
    
    
    # Call to append(...): (line 34)
    # Processing the call arguments (line 34)
    
    # Call to get_aspect(...): (line 34)
    # Processing the call keyword arguments (line 34)
    kwargs_152816 = {}
    # Getting the type of 'ax' (line 34)
    ax_152814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 24), 'ax', False)
    # Obtaining the member 'get_aspect' of a type (line 34)
    get_aspect_152815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 24), ax_152814, 'get_aspect')
    # Calling get_aspect(args, kwargs) (line 34)
    get_aspect_call_result_152817 = invoke(stypy.reporting.localization.Localization(__file__, 34, 24), get_aspect_152815, *[], **kwargs_152816)
    
    # Processing the call keyword arguments (line 34)
    kwargs_152818 = {}
    # Getting the type of 'asp_list' (line 34)
    asp_list_152812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'asp_list', False)
    # Obtaining the member 'append' of a type (line 34)
    append_152813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), asp_list_152812, 'append')
    # Calling append(args, kwargs) (line 34)
    append_call_result_152819 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), append_152813, *[get_aspect_call_result_152817], **kwargs_152818)
    

    @norecursion
    def _l(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'pos' (line 36)
        pos_152820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 25), 'pos')
        defaults = [pos_152820]
        # Create a new context for function '_l'
        module_type_store = module_type_store.open_function_context('_l', 36, 8, False)
        
        # Passed parameters checking function
        _l.stypy_localization = localization
        _l.stypy_type_of_self = None
        _l.stypy_type_store = module_type_store
        _l.stypy_function_name = '_l'
        _l.stypy_param_names_list = ['a', 'r', 'pos']
        _l.stypy_varargs_param_name = None
        _l.stypy_kwargs_param_name = None
        _l.stypy_call_defaults = defaults
        _l.stypy_call_varargs = varargs
        _l.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_l', ['a', 'r', 'pos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_l', localization, ['a', 'r', 'pos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_l(...)' code ##################

        # Getting the type of 'pos' (line 37)
        pos_152821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 19), 'pos')
        # Assigning a type to the variable 'stypy_return_type' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'stypy_return_type', pos_152821)
        
        # ################# End of '_l(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_l' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_152822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_152822)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_l'
        return stypy_return_type_152822

    # Assigning a type to the variable '_l' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), '_l', _l)
    
    # Call to set_axes_locator(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of '_l' (line 38)
    _l_152825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 28), '_l', False)
    # Processing the call keyword arguments (line 38)
    kwargs_152826 = {}
    # Getting the type of 'ax' (line 38)
    ax_152823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'ax', False)
    # Obtaining the member 'set_axes_locator' of a type (line 38)
    set_axes_locator_152824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), ax_152823, 'set_axes_locator')
    # Calling set_axes_locator(args, kwargs) (line 38)
    set_axes_locator_call_result_152827 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), set_axes_locator_152824, *[_l_152825], **kwargs_152826)
    
    
    # Call to set_aspect(...): (line 39)
    # Processing the call arguments (line 39)
    unicode_152830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 22), 'unicode', u'auto')
    # Processing the call keyword arguments (line 39)
    kwargs_152831 = {}
    # Getting the type of 'ax' (line 39)
    ax_152828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'ax', False)
    # Obtaining the member 'set_aspect' of a type (line 39)
    set_aspect_152829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), ax_152828, 'set_aspect')
    # Calling set_aspect(args, kwargs) (line 39)
    set_aspect_call_result_152832 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), set_aspect_152829, *[unicode_152830], **kwargs_152831)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def restore_bbox(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'restore_bbox'
        module_type_store = module_type_store.open_function_context('restore_bbox', 41, 4, False)
        
        # Passed parameters checking function
        restore_bbox.stypy_localization = localization
        restore_bbox.stypy_type_of_self = None
        restore_bbox.stypy_type_store = module_type_store
        restore_bbox.stypy_function_name = 'restore_bbox'
        restore_bbox.stypy_param_names_list = []
        restore_bbox.stypy_varargs_param_name = None
        restore_bbox.stypy_kwargs_param_name = None
        restore_bbox.stypy_call_defaults = defaults
        restore_bbox.stypy_call_varargs = varargs
        restore_bbox.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'restore_bbox', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'restore_bbox', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'restore_bbox(...)' code ##################

        
        
        # Call to zip(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'fig' (line 43)
        fig_152834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 32), 'fig', False)
        # Obtaining the member 'axes' of a type (line 43)
        axes_152835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 32), fig_152834, 'axes')
        # Getting the type of 'asp_list' (line 43)
        asp_list_152836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 42), 'asp_list', False)
        # Getting the type of 'locator_list' (line 43)
        locator_list_152837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 52), 'locator_list', False)
        # Processing the call keyword arguments (line 43)
        kwargs_152838 = {}
        # Getting the type of 'zip' (line 43)
        zip_152833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 28), 'zip', False)
        # Calling zip(args, kwargs) (line 43)
        zip_call_result_152839 = invoke(stypy.reporting.localization.Localization(__file__, 43, 28), zip_152833, *[axes_152835, asp_list_152836, locator_list_152837], **kwargs_152838)
        
        # Testing the type of a for loop iterable (line 43)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 43, 8), zip_call_result_152839)
        # Getting the type of the for loop variable (line 43)
        for_loop_var_152840 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 43, 8), zip_call_result_152839)
        # Assigning a type to the variable 'ax' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'ax', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 8), for_loop_var_152840))
        # Assigning a type to the variable 'asp' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'asp', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 8), for_loop_var_152840))
        # Assigning a type to the variable 'loc' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'loc', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 8), for_loop_var_152840))
        # SSA begins for a for statement (line 43)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to set_aspect(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'asp' (line 44)
        asp_152843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 26), 'asp', False)
        # Processing the call keyword arguments (line 44)
        kwargs_152844 = {}
        # Getting the type of 'ax' (line 44)
        ax_152841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'ax', False)
        # Obtaining the member 'set_aspect' of a type (line 44)
        set_aspect_152842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), ax_152841, 'set_aspect')
        # Calling set_aspect(args, kwargs) (line 44)
        set_aspect_call_result_152845 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), set_aspect_152842, *[asp_152843], **kwargs_152844)
        
        
        # Call to set_axes_locator(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'loc' (line 45)
        loc_152848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 32), 'loc', False)
        # Processing the call keyword arguments (line 45)
        kwargs_152849 = {}
        # Getting the type of 'ax' (line 45)
        ax_152846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'ax', False)
        # Obtaining the member 'set_axes_locator' of a type (line 45)
        set_axes_locator_152847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), ax_152846, 'set_axes_locator')
        # Calling set_axes_locator(args, kwargs) (line 45)
        set_axes_locator_call_result_152850 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), set_axes_locator_152847, *[loc_152848], **kwargs_152849)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 47):
        
        # Assigning a Name to a Attribute (line 47):
        # Getting the type of 'origBbox' (line 47)
        origBbox_152851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'origBbox')
        # Getting the type of 'fig' (line 47)
        fig_152852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'fig')
        # Setting the type of the member 'bbox' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), fig_152852, 'bbox', origBbox_152851)
        
        # Assigning a Name to a Attribute (line 48):
        
        # Assigning a Name to a Attribute (line 48):
        # Getting the type of 'origBboxInches' (line 48)
        origBboxInches_152853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'origBboxInches')
        # Getting the type of 'fig' (line 48)
        fig_152854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'fig')
        # Setting the type of the member 'bbox_inches' of a type (line 48)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), fig_152854, 'bbox_inches', origBboxInches_152853)
        
        # Assigning a Name to a Attribute (line 49):
        
        # Assigning a Name to a Attribute (line 49):
        # Getting the type of '_boxout' (line 49)
        _boxout_152855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 34), '_boxout')
        # Getting the type of 'fig' (line 49)
        fig_152856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'fig')
        # Obtaining the member 'transFigure' of a type (line 49)
        transFigure_152857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), fig_152856, 'transFigure')
        # Setting the type of the member '_boxout' of a type (line 49)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), transFigure_152857, '_boxout', _boxout_152855)
        
        # Call to invalidate(...): (line 50)
        # Processing the call keyword arguments (line 50)
        kwargs_152861 = {}
        # Getting the type of 'fig' (line 50)
        fig_152858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'fig', False)
        # Obtaining the member 'transFigure' of a type (line 50)
        transFigure_152859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), fig_152858, 'transFigure')
        # Obtaining the member 'invalidate' of a type (line 50)
        invalidate_152860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), transFigure_152859, 'invalidate')
        # Calling invalidate(args, kwargs) (line 50)
        invalidate_call_result_152862 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), invalidate_152860, *[], **kwargs_152861)
        
        
        # Call to set_bounds(...): (line 51)
        # Processing the call arguments (line 51)
        int_152866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 29), 'int')
        int_152867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 32), 'int')
        int_152868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 35), 'int')
        int_152869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 38), 'int')
        # Processing the call keyword arguments (line 51)
        kwargs_152870 = {}
        # Getting the type of 'fig' (line 51)
        fig_152863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'fig', False)
        # Obtaining the member 'patch' of a type (line 51)
        patch_152864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), fig_152863, 'patch')
        # Obtaining the member 'set_bounds' of a type (line 51)
        set_bounds_152865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), patch_152864, 'set_bounds')
        # Calling set_bounds(args, kwargs) (line 51)
        set_bounds_call_result_152871 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), set_bounds_152865, *[int_152866, int_152867, int_152868, int_152869], **kwargs_152870)
        
        
        # ################# End of 'restore_bbox(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'restore_bbox' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_152872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_152872)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'restore_bbox'
        return stypy_return_type_152872

    # Assigning a type to the variable 'restore_bbox' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'restore_bbox', restore_bbox)
    
    # Type idiom detected: calculating its left and rigth part (line 53)
    # Getting the type of 'fixed_dpi' (line 53)
    fixed_dpi_152873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'fixed_dpi')
    # Getting the type of 'None' (line 53)
    None_152874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 24), 'None')
    
    (may_be_152875, more_types_in_union_152876) = may_not_be_none(fixed_dpi_152873, None_152874)

    if may_be_152875:

        if more_types_in_union_152876:
            # Runtime conditional SSA (line 53)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 54):
        
        # Assigning a Call to a Name (line 54):
        
        # Call to scale(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'fixed_dpi' (line 54)
        fixed_dpi_152881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 30), 'fixed_dpi', False)
        # Processing the call keyword arguments (line 54)
        kwargs_152882 = {}
        
        # Call to Affine2D(...): (line 54)
        # Processing the call keyword arguments (line 54)
        kwargs_152878 = {}
        # Getting the type of 'Affine2D' (line 54)
        Affine2D_152877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'Affine2D', False)
        # Calling Affine2D(args, kwargs) (line 54)
        Affine2D_call_result_152879 = invoke(stypy.reporting.localization.Localization(__file__, 54, 13), Affine2D_152877, *[], **kwargs_152878)
        
        # Obtaining the member 'scale' of a type (line 54)
        scale_152880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 13), Affine2D_call_result_152879, 'scale')
        # Calling scale(args, kwargs) (line 54)
        scale_call_result_152883 = invoke(stypy.reporting.localization.Localization(__file__, 54, 13), scale_152880, *[fixed_dpi_152881], **kwargs_152882)
        
        # Assigning a type to the variable 'tr' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tr', scale_call_result_152883)
        
        # Assigning a BinOp to a Name (line 55):
        
        # Assigning a BinOp to a Name (line 55):
        # Getting the type of 'fixed_dpi' (line 55)
        fixed_dpi_152884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 20), 'fixed_dpi')
        # Getting the type of 'fig' (line 55)
        fig_152885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 32), 'fig')
        # Obtaining the member 'dpi' of a type (line 55)
        dpi_152886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 32), fig_152885, 'dpi')
        # Applying the binary operator 'div' (line 55)
        result_div_152887 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 20), 'div', fixed_dpi_152884, dpi_152886)
        
        # Assigning a type to the variable 'dpi_scale' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'dpi_scale', result_div_152887)

        if more_types_in_union_152876:
            # Runtime conditional SSA for else branch (line 53)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_152875) or more_types_in_union_152876):
        
        # Assigning a Call to a Name (line 57):
        
        # Assigning a Call to a Name (line 57):
        
        # Call to scale(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'fig' (line 57)
        fig_152892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 30), 'fig', False)
        # Obtaining the member 'dpi' of a type (line 57)
        dpi_152893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 30), fig_152892, 'dpi')
        # Processing the call keyword arguments (line 57)
        kwargs_152894 = {}
        
        # Call to Affine2D(...): (line 57)
        # Processing the call keyword arguments (line 57)
        kwargs_152889 = {}
        # Getting the type of 'Affine2D' (line 57)
        Affine2D_152888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 13), 'Affine2D', False)
        # Calling Affine2D(args, kwargs) (line 57)
        Affine2D_call_result_152890 = invoke(stypy.reporting.localization.Localization(__file__, 57, 13), Affine2D_152888, *[], **kwargs_152889)
        
        # Obtaining the member 'scale' of a type (line 57)
        scale_152891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 13), Affine2D_call_result_152890, 'scale')
        # Calling scale(args, kwargs) (line 57)
        scale_call_result_152895 = invoke(stypy.reporting.localization.Localization(__file__, 57, 13), scale_152891, *[dpi_152893], **kwargs_152894)
        
        # Assigning a type to the variable 'tr' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'tr', scale_call_result_152895)
        
        # Assigning a Num to a Name (line 58):
        
        # Assigning a Num to a Name (line 58):
        float_152896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 20), 'float')
        # Assigning a type to the variable 'dpi_scale' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'dpi_scale', float_152896)

        if (may_be_152875 and more_types_in_union_152876):
            # SSA join for if statement (line 53)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 60):
    
    # Assigning a Call to a Name (line 60):
    
    # Call to TransformedBbox(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'bbox_inches' (line 60)
    bbox_inches_152898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 28), 'bbox_inches', False)
    # Getting the type of 'tr' (line 60)
    tr_152899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 41), 'tr', False)
    # Processing the call keyword arguments (line 60)
    kwargs_152900 = {}
    # Getting the type of 'TransformedBbox' (line 60)
    TransformedBbox_152897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'TransformedBbox', False)
    # Calling TransformedBbox(args, kwargs) (line 60)
    TransformedBbox_call_result_152901 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), TransformedBbox_152897, *[bbox_inches_152898, tr_152899], **kwargs_152900)
    
    # Assigning a type to the variable '_bbox' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), '_bbox', TransformedBbox_call_result_152901)
    
    # Assigning a Call to a Attribute (line 62):
    
    # Assigning a Call to a Attribute (line 62):
    
    # Call to from_bounds(...): (line 62)
    # Processing the call arguments (line 62)
    int_152904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 39), 'int')
    int_152905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 42), 'int')
    # Getting the type of 'bbox_inches' (line 63)
    bbox_inches_152906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 39), 'bbox_inches', False)
    # Obtaining the member 'width' of a type (line 63)
    width_152907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 39), bbox_inches_152906, 'width')
    # Getting the type of 'bbox_inches' (line 63)
    bbox_inches_152908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 58), 'bbox_inches', False)
    # Obtaining the member 'height' of a type (line 63)
    height_152909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 58), bbox_inches_152908, 'height')
    # Processing the call keyword arguments (line 62)
    kwargs_152910 = {}
    # Getting the type of 'Bbox' (line 62)
    Bbox_152902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 22), 'Bbox', False)
    # Obtaining the member 'from_bounds' of a type (line 62)
    from_bounds_152903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 22), Bbox_152902, 'from_bounds')
    # Calling from_bounds(args, kwargs) (line 62)
    from_bounds_call_result_152911 = invoke(stypy.reporting.localization.Localization(__file__, 62, 22), from_bounds_152903, *[int_152904, int_152905, width_152907, height_152909], **kwargs_152910)
    
    # Getting the type of 'fig' (line 62)
    fig_152912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'fig')
    # Setting the type of the member 'bbox_inches' of a type (line 62)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 4), fig_152912, 'bbox_inches', from_bounds_call_result_152911)
    
    # Assigning a Tuple to a Tuple (line 64):
    
    # Assigning a Attribute to a Name (line 64):
    # Getting the type of '_bbox' (line 64)
    _bbox_152913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 13), '_bbox')
    # Obtaining the member 'x0' of a type (line 64)
    x0_152914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 13), _bbox_152913, 'x0')
    # Assigning a type to the variable 'tuple_assignment_152770' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_assignment_152770', x0_152914)
    
    # Assigning a Attribute to a Name (line 64):
    # Getting the type of '_bbox' (line 64)
    _bbox_152915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 23), '_bbox')
    # Obtaining the member 'y0' of a type (line 64)
    y0_152916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 23), _bbox_152915, 'y0')
    # Assigning a type to the variable 'tuple_assignment_152771' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_assignment_152771', y0_152916)
    
    # Assigning a Name to a Name (line 64):
    # Getting the type of 'tuple_assignment_152770' (line 64)
    tuple_assignment_152770_152917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_assignment_152770')
    # Assigning a type to the variable 'x0' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'x0', tuple_assignment_152770_152917)
    
    # Assigning a Name to a Name (line 64):
    # Getting the type of 'tuple_assignment_152771' (line 64)
    tuple_assignment_152771_152918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'tuple_assignment_152771')
    # Assigning a type to the variable 'y0' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'y0', tuple_assignment_152771_152918)
    
    # Assigning a Tuple to a Tuple (line 65):
    
    # Assigning a BinOp to a Name (line 65):
    # Getting the type of 'fig' (line 65)
    fig_152919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 13), 'fig')
    # Obtaining the member 'bbox' of a type (line 65)
    bbox_152920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 13), fig_152919, 'bbox')
    # Obtaining the member 'width' of a type (line 65)
    width_152921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 13), bbox_152920, 'width')
    # Getting the type of 'dpi_scale' (line 65)
    dpi_scale_152922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'dpi_scale')
    # Applying the binary operator '*' (line 65)
    result_mul_152923 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 13), '*', width_152921, dpi_scale_152922)
    
    # Assigning a type to the variable 'tuple_assignment_152772' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_assignment_152772', result_mul_152923)
    
    # Assigning a BinOp to a Name (line 65):
    # Getting the type of 'fig' (line 65)
    fig_152924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 41), 'fig')
    # Obtaining the member 'bbox' of a type (line 65)
    bbox_152925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 41), fig_152924, 'bbox')
    # Obtaining the member 'height' of a type (line 65)
    height_152926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 41), bbox_152925, 'height')
    # Getting the type of 'dpi_scale' (line 65)
    dpi_scale_152927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 59), 'dpi_scale')
    # Applying the binary operator '*' (line 65)
    result_mul_152928 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 41), '*', height_152926, dpi_scale_152927)
    
    # Assigning a type to the variable 'tuple_assignment_152773' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_assignment_152773', result_mul_152928)
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'tuple_assignment_152772' (line 65)
    tuple_assignment_152772_152929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_assignment_152772')
    # Assigning a type to the variable 'w1' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'w1', tuple_assignment_152772_152929)
    
    # Assigning a Name to a Name (line 65):
    # Getting the type of 'tuple_assignment_152773' (line 65)
    tuple_assignment_152773_152930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'tuple_assignment_152773')
    # Assigning a type to the variable 'h1' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'h1', tuple_assignment_152773_152930)
    
    # Assigning a Call to a Attribute (line 66):
    
    # Assigning a Call to a Attribute (line 66):
    
    # Call to from_bounds(...): (line 66)
    # Processing the call arguments (line 66)
    
    # Getting the type of 'x0' (line 66)
    x0_152933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 48), 'x0', False)
    # Applying the 'usub' unary operator (line 66)
    result___neg___152934 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 47), 'usub', x0_152933)
    
    
    # Getting the type of 'y0' (line 66)
    y0_152935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 53), 'y0', False)
    # Applying the 'usub' unary operator (line 66)
    result___neg___152936 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 52), 'usub', y0_152935)
    
    # Getting the type of 'w1' (line 66)
    w1_152937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 57), 'w1', False)
    # Getting the type of 'h1' (line 66)
    h1_152938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 61), 'h1', False)
    # Processing the call keyword arguments (line 66)
    kwargs_152939 = {}
    # Getting the type of 'Bbox' (line 66)
    Bbox_152931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 30), 'Bbox', False)
    # Obtaining the member 'from_bounds' of a type (line 66)
    from_bounds_152932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 30), Bbox_152931, 'from_bounds')
    # Calling from_bounds(args, kwargs) (line 66)
    from_bounds_call_result_152940 = invoke(stypy.reporting.localization.Localization(__file__, 66, 30), from_bounds_152932, *[result___neg___152934, result___neg___152936, w1_152937, h1_152938], **kwargs_152939)
    
    # Getting the type of 'fig' (line 66)
    fig_152941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'fig')
    # Obtaining the member 'transFigure' of a type (line 66)
    transFigure_152942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 4), fig_152941, 'transFigure')
    # Setting the type of the member '_boxout' of a type (line 66)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 4), transFigure_152942, '_boxout', from_bounds_call_result_152940)
    
    # Call to invalidate(...): (line 67)
    # Processing the call keyword arguments (line 67)
    kwargs_152946 = {}
    # Getting the type of 'fig' (line 67)
    fig_152943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'fig', False)
    # Obtaining the member 'transFigure' of a type (line 67)
    transFigure_152944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 4), fig_152943, 'transFigure')
    # Obtaining the member 'invalidate' of a type (line 67)
    invalidate_152945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 4), transFigure_152944, 'invalidate')
    # Calling invalidate(args, kwargs) (line 67)
    invalidate_call_result_152947 = invoke(stypy.reporting.localization.Localization(__file__, 67, 4), invalidate_152945, *[], **kwargs_152946)
    
    
    # Assigning a Call to a Attribute (line 69):
    
    # Assigning a Call to a Attribute (line 69):
    
    # Call to TransformedBbox(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'fig' (line 69)
    fig_152949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 31), 'fig', False)
    # Obtaining the member 'bbox_inches' of a type (line 69)
    bbox_inches_152950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 31), fig_152949, 'bbox_inches')
    # Getting the type of 'tr' (line 69)
    tr_152951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 48), 'tr', False)
    # Processing the call keyword arguments (line 69)
    kwargs_152952 = {}
    # Getting the type of 'TransformedBbox' (line 69)
    TransformedBbox_152948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'TransformedBbox', False)
    # Calling TransformedBbox(args, kwargs) (line 69)
    TransformedBbox_call_result_152953 = invoke(stypy.reporting.localization.Localization(__file__, 69, 15), TransformedBbox_152948, *[bbox_inches_152950, tr_152951], **kwargs_152952)
    
    # Getting the type of 'fig' (line 69)
    fig_152954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'fig')
    # Setting the type of the member 'bbox' of a type (line 69)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 4), fig_152954, 'bbox', TransformedBbox_call_result_152953)
    
    # Call to set_bounds(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'x0' (line 71)
    x0_152958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 25), 'x0', False)
    # Getting the type of 'w1' (line 71)
    w1_152959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 30), 'w1', False)
    # Applying the binary operator 'div' (line 71)
    result_div_152960 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 25), 'div', x0_152958, w1_152959)
    
    # Getting the type of 'y0' (line 71)
    y0_152961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 34), 'y0', False)
    # Getting the type of 'h1' (line 71)
    h1_152962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 39), 'h1', False)
    # Applying the binary operator 'div' (line 71)
    result_div_152963 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 34), 'div', y0_152961, h1_152962)
    
    # Getting the type of 'fig' (line 72)
    fig_152964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 25), 'fig', False)
    # Obtaining the member 'bbox' of a type (line 72)
    bbox_152965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 25), fig_152964, 'bbox')
    # Obtaining the member 'width' of a type (line 72)
    width_152966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 25), bbox_152965, 'width')
    # Getting the type of 'w1' (line 72)
    w1_152967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 42), 'w1', False)
    # Applying the binary operator 'div' (line 72)
    result_div_152968 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 25), 'div', width_152966, w1_152967)
    
    # Getting the type of 'fig' (line 72)
    fig_152969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 46), 'fig', False)
    # Obtaining the member 'bbox' of a type (line 72)
    bbox_152970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 46), fig_152969, 'bbox')
    # Obtaining the member 'height' of a type (line 72)
    height_152971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 46), bbox_152970, 'height')
    # Getting the type of 'h1' (line 72)
    h1_152972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 64), 'h1', False)
    # Applying the binary operator 'div' (line 72)
    result_div_152973 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 46), 'div', height_152971, h1_152972)
    
    # Processing the call keyword arguments (line 71)
    kwargs_152974 = {}
    # Getting the type of 'fig' (line 71)
    fig_152955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'fig', False)
    # Obtaining the member 'patch' of a type (line 71)
    patch_152956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 4), fig_152955, 'patch')
    # Obtaining the member 'set_bounds' of a type (line 71)
    set_bounds_152957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 4), patch_152956, 'set_bounds')
    # Calling set_bounds(args, kwargs) (line 71)
    set_bounds_call_result_152975 = invoke(stypy.reporting.localization.Localization(__file__, 71, 4), set_bounds_152957, *[result_div_152960, result_div_152963, result_div_152968, result_div_152973], **kwargs_152974)
    
    # Getting the type of 'restore_bbox' (line 74)
    restore_bbox_152976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 11), 'restore_bbox')
    # Assigning a type to the variable 'stypy_return_type' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type', restore_bbox_152976)
    
    # ################# End of 'adjust_bbox(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'adjust_bbox' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_152977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_152977)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'adjust_bbox'
    return stypy_return_type_152977

# Assigning a type to the variable 'adjust_bbox' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'adjust_bbox', adjust_bbox)

@norecursion
def process_figure_for_rasterizing(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 77)
    None_152978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 71), 'None')
    defaults = [None_152978]
    # Create a new context for function 'process_figure_for_rasterizing'
    module_type_store = module_type_store.open_function_context('process_figure_for_rasterizing', 77, 0, False)
    
    # Passed parameters checking function
    process_figure_for_rasterizing.stypy_localization = localization
    process_figure_for_rasterizing.stypy_type_of_self = None
    process_figure_for_rasterizing.stypy_type_store = module_type_store
    process_figure_for_rasterizing.stypy_function_name = 'process_figure_for_rasterizing'
    process_figure_for_rasterizing.stypy_param_names_list = ['fig', 'bbox_inches_restore', 'fixed_dpi']
    process_figure_for_rasterizing.stypy_varargs_param_name = None
    process_figure_for_rasterizing.stypy_kwargs_param_name = None
    process_figure_for_rasterizing.stypy_call_defaults = defaults
    process_figure_for_rasterizing.stypy_call_varargs = varargs
    process_figure_for_rasterizing.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'process_figure_for_rasterizing', ['fig', 'bbox_inches_restore', 'fixed_dpi'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'process_figure_for_rasterizing', localization, ['fig', 'bbox_inches_restore', 'fixed_dpi'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'process_figure_for_rasterizing(...)' code ##################

    unicode_152979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, (-1)), 'unicode', u'\n    This need to be called when figure dpi changes during the drawing\n    (e.g., rasterizing). It recovers the bbox and re-adjust it with\n    the new dpi.\n    ')
    
    # Assigning a Name to a Tuple (line 84):
    
    # Assigning a Subscript to a Name (line 84):
    
    # Obtaining the type of the subscript
    int_152980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 4), 'int')
    # Getting the type of 'bbox_inches_restore' (line 84)
    bbox_inches_restore_152981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 32), 'bbox_inches_restore')
    # Obtaining the member '__getitem__' of a type (line 84)
    getitem___152982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 4), bbox_inches_restore_152981, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 84)
    subscript_call_result_152983 = invoke(stypy.reporting.localization.Localization(__file__, 84, 4), getitem___152982, int_152980)
    
    # Assigning a type to the variable 'tuple_var_assignment_152774' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'tuple_var_assignment_152774', subscript_call_result_152983)
    
    # Assigning a Subscript to a Name (line 84):
    
    # Obtaining the type of the subscript
    int_152984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 4), 'int')
    # Getting the type of 'bbox_inches_restore' (line 84)
    bbox_inches_restore_152985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 32), 'bbox_inches_restore')
    # Obtaining the member '__getitem__' of a type (line 84)
    getitem___152986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 4), bbox_inches_restore_152985, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 84)
    subscript_call_result_152987 = invoke(stypy.reporting.localization.Localization(__file__, 84, 4), getitem___152986, int_152984)
    
    # Assigning a type to the variable 'tuple_var_assignment_152775' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'tuple_var_assignment_152775', subscript_call_result_152987)
    
    # Assigning a Name to a Name (line 84):
    # Getting the type of 'tuple_var_assignment_152774' (line 84)
    tuple_var_assignment_152774_152988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'tuple_var_assignment_152774')
    # Assigning a type to the variable 'bbox_inches' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'bbox_inches', tuple_var_assignment_152774_152988)
    
    # Assigning a Name to a Name (line 84):
    # Getting the type of 'tuple_var_assignment_152775' (line 84)
    tuple_var_assignment_152775_152989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'tuple_var_assignment_152775')
    # Assigning a type to the variable 'restore_bbox' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 17), 'restore_bbox', tuple_var_assignment_152775_152989)
    
    # Call to restore_bbox(...): (line 85)
    # Processing the call keyword arguments (line 85)
    kwargs_152991 = {}
    # Getting the type of 'restore_bbox' (line 85)
    restore_bbox_152990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'restore_bbox', False)
    # Calling restore_bbox(args, kwargs) (line 85)
    restore_bbox_call_result_152992 = invoke(stypy.reporting.localization.Localization(__file__, 85, 4), restore_bbox_152990, *[], **kwargs_152991)
    
    
    # Assigning a Call to a Name (line 86):
    
    # Assigning a Call to a Name (line 86):
    
    # Call to adjust_bbox(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'fig' (line 86)
    fig_152994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'fig', False)
    # Getting the type of 'bbox_inches' (line 86)
    bbox_inches_152995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 25), 'bbox_inches', False)
    # Getting the type of 'fixed_dpi' (line 86)
    fixed_dpi_152996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 38), 'fixed_dpi', False)
    # Processing the call keyword arguments (line 86)
    kwargs_152997 = {}
    # Getting the type of 'adjust_bbox' (line 86)
    adjust_bbox_152993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'adjust_bbox', False)
    # Calling adjust_bbox(args, kwargs) (line 86)
    adjust_bbox_call_result_152998 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), adjust_bbox_152993, *[fig_152994, bbox_inches_152995, fixed_dpi_152996], **kwargs_152997)
    
    # Assigning a type to the variable 'r' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'r', adjust_bbox_call_result_152998)
    
    # Obtaining an instance of the builtin type 'tuple' (line 88)
    tuple_152999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 88)
    # Adding element type (line 88)
    # Getting the type of 'bbox_inches' (line 88)
    bbox_inches_153000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'bbox_inches')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 11), tuple_152999, bbox_inches_153000)
    # Adding element type (line 88)
    # Getting the type of 'r' (line 88)
    r_153001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'r')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 11), tuple_152999, r_153001)
    
    # Assigning a type to the variable 'stypy_return_type' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type', tuple_152999)
    
    # ################# End of 'process_figure_for_rasterizing(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'process_figure_for_rasterizing' in the type store
    # Getting the type of 'stypy_return_type' (line 77)
    stypy_return_type_153002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_153002)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'process_figure_for_rasterizing'
    return stypy_return_type_153002

# Assigning a type to the variable 'process_figure_for_rasterizing' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'process_figure_for_rasterizing', process_figure_for_rasterizing)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
