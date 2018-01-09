
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: from six.moves import tkinter as Tk
6: 
7: import numpy as np
8: 
9: from matplotlib.backends import _tkagg
10: 
11: def blit(photoimage, aggimage, bbox=None, colormode=1):
12:     tk = photoimage.tk
13: 
14:     if bbox is not None:
15:         bbox_array = bbox.__array__()
16:     else:
17:         bbox_array = None
18:     data = np.asarray(aggimage)
19:     try:
20:         tk.call(
21:             "PyAggImagePhoto", photoimage,
22:             id(data), colormode, id(bbox_array))
23:     except Tk.TclError:
24:         try:
25:             try:
26:                 _tkagg.tkinit(tk.interpaddr(), 1)
27:             except AttributeError:
28:                 _tkagg.tkinit(id(tk), 0)
29:             tk.call("PyAggImagePhoto", photoimage,
30:                     id(data), colormode, id(bbox_array))
31:         except (ImportError, AttributeError, Tk.TclError):
32:             raise
33: 
34: def test(aggimage):
35:     import time
36:     r = Tk.Tk()
37:     c = Tk.Canvas(r, width=aggimage.width, height=aggimage.height)
38:     c.pack()
39:     p = Tk.PhotoImage(width=aggimage.width, height=aggimage.height)
40:     blit(p, aggimage)
41:     c.create_image(aggimage.width,aggimage.height,image=p)
42:     blit(p, aggimage)
43:     while True: r.update_idletasks()
44: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_269337 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_269337) is not StypyTypeError):

    if (import_269337 != 'pyd_module'):
        __import__(import_269337)
        sys_modules_269338 = sys.modules[import_269337]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_269338.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_269337)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from six.moves import Tk' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_269339 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'six.moves')

if (type(import_269339) is not StypyTypeError):

    if (import_269339 != 'pyd_module'):
        __import__(import_269339)
        sys_modules_269340 = sys.modules[import_269339]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'six.moves', sys_modules_269340.module_type_store, module_type_store, ['tkinter'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_269340, sys_modules_269340.module_type_store, module_type_store)
    else:
        from six.moves import tkinter as Tk

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'six.moves', None, module_type_store, ['tkinter'], [Tk])

else:
    # Assigning a type to the variable 'six.moves' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'six.moves', import_269339)

# Adding an alias
module_type_store.add_alias('Tk', 'tkinter')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import numpy' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_269341 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_269341) is not StypyTypeError):

    if (import_269341 != 'pyd_module'):
        __import__(import_269341)
        sys_modules_269342 = sys.modules[import_269341]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', sys_modules_269342.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_269341)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from matplotlib.backends import _tkagg' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_269343 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backends')

if (type(import_269343) is not StypyTypeError):

    if (import_269343 != 'pyd_module'):
        __import__(import_269343)
        sys_modules_269344 = sys.modules[import_269343]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backends', sys_modules_269344.module_type_store, module_type_store, ['_tkagg'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_269344, sys_modules_269344.module_type_store, module_type_store)
    else:
        from matplotlib.backends import _tkagg

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backends', None, module_type_store, ['_tkagg'], [_tkagg])

else:
    # Assigning a type to the variable 'matplotlib.backends' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.backends', import_269343)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')


@norecursion
def blit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 11)
    None_269345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 36), 'None')
    int_269346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 52), 'int')
    defaults = [None_269345, int_269346]
    # Create a new context for function 'blit'
    module_type_store = module_type_store.open_function_context('blit', 11, 0, False)
    
    # Passed parameters checking function
    blit.stypy_localization = localization
    blit.stypy_type_of_self = None
    blit.stypy_type_store = module_type_store
    blit.stypy_function_name = 'blit'
    blit.stypy_param_names_list = ['photoimage', 'aggimage', 'bbox', 'colormode']
    blit.stypy_varargs_param_name = None
    blit.stypy_kwargs_param_name = None
    blit.stypy_call_defaults = defaults
    blit.stypy_call_varargs = varargs
    blit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'blit', ['photoimage', 'aggimage', 'bbox', 'colormode'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'blit', localization, ['photoimage', 'aggimage', 'bbox', 'colormode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'blit(...)' code ##################

    
    # Assigning a Attribute to a Name (line 12):
    # Getting the type of 'photoimage' (line 12)
    photoimage_269347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'photoimage')
    # Obtaining the member 'tk' of a type (line 12)
    tk_269348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 9), photoimage_269347, 'tk')
    # Assigning a type to the variable 'tk' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'tk', tk_269348)
    
    # Type idiom detected: calculating its left and rigth part (line 14)
    # Getting the type of 'bbox' (line 14)
    bbox_269349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'bbox')
    # Getting the type of 'None' (line 14)
    None_269350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'None')
    
    (may_be_269351, more_types_in_union_269352) = may_not_be_none(bbox_269349, None_269350)

    if may_be_269351:

        if more_types_in_union_269352:
            # Runtime conditional SSA (line 14)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 15):
        
        # Call to __array__(...): (line 15)
        # Processing the call keyword arguments (line 15)
        kwargs_269355 = {}
        # Getting the type of 'bbox' (line 15)
        bbox_269353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 21), 'bbox', False)
        # Obtaining the member '__array__' of a type (line 15)
        array___269354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 21), bbox_269353, '__array__')
        # Calling __array__(args, kwargs) (line 15)
        array___call_result_269356 = invoke(stypy.reporting.localization.Localization(__file__, 15, 21), array___269354, *[], **kwargs_269355)
        
        # Assigning a type to the variable 'bbox_array' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'bbox_array', array___call_result_269356)

        if more_types_in_union_269352:
            # Runtime conditional SSA for else branch (line 14)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_269351) or more_types_in_union_269352):
        
        # Assigning a Name to a Name (line 17):
        # Getting the type of 'None' (line 17)
        None_269357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 21), 'None')
        # Assigning a type to the variable 'bbox_array' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'bbox_array', None_269357)

        if (may_be_269351 and more_types_in_union_269352):
            # SSA join for if statement (line 14)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 18):
    
    # Call to asarray(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'aggimage' (line 18)
    aggimage_269360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 22), 'aggimage', False)
    # Processing the call keyword arguments (line 18)
    kwargs_269361 = {}
    # Getting the type of 'np' (line 18)
    np_269358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'np', False)
    # Obtaining the member 'asarray' of a type (line 18)
    asarray_269359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 11), np_269358, 'asarray')
    # Calling asarray(args, kwargs) (line 18)
    asarray_call_result_269362 = invoke(stypy.reporting.localization.Localization(__file__, 18, 11), asarray_269359, *[aggimage_269360], **kwargs_269361)
    
    # Assigning a type to the variable 'data' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'data', asarray_call_result_269362)
    
    
    # SSA begins for try-except statement (line 19)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to call(...): (line 20)
    # Processing the call arguments (line 20)
    unicode_269365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 12), 'unicode', u'PyAggImagePhoto')
    # Getting the type of 'photoimage' (line 21)
    photoimage_269366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 31), 'photoimage', False)
    
    # Call to id(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'data' (line 22)
    data_269368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 15), 'data', False)
    # Processing the call keyword arguments (line 22)
    kwargs_269369 = {}
    # Getting the type of 'id' (line 22)
    id_269367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'id', False)
    # Calling id(args, kwargs) (line 22)
    id_call_result_269370 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), id_269367, *[data_269368], **kwargs_269369)
    
    # Getting the type of 'colormode' (line 22)
    colormode_269371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 22), 'colormode', False)
    
    # Call to id(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'bbox_array' (line 22)
    bbox_array_269373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 36), 'bbox_array', False)
    # Processing the call keyword arguments (line 22)
    kwargs_269374 = {}
    # Getting the type of 'id' (line 22)
    id_269372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 33), 'id', False)
    # Calling id(args, kwargs) (line 22)
    id_call_result_269375 = invoke(stypy.reporting.localization.Localization(__file__, 22, 33), id_269372, *[bbox_array_269373], **kwargs_269374)
    
    # Processing the call keyword arguments (line 20)
    kwargs_269376 = {}
    # Getting the type of 'tk' (line 20)
    tk_269363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'tk', False)
    # Obtaining the member 'call' of a type (line 20)
    call_269364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), tk_269363, 'call')
    # Calling call(args, kwargs) (line 20)
    call_call_result_269377 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), call_269364, *[unicode_269365, photoimage_269366, id_call_result_269370, colormode_269371, id_call_result_269375], **kwargs_269376)
    
    # SSA branch for the except part of a try statement (line 19)
    # SSA branch for the except 'Attribute' branch of a try statement (line 19)
    module_type_store.open_ssa_branch('except')
    
    
    # SSA begins for try-except statement (line 24)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    
    # SSA begins for try-except statement (line 25)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to tkinit(...): (line 26)
    # Processing the call arguments (line 26)
    
    # Call to interpaddr(...): (line 26)
    # Processing the call keyword arguments (line 26)
    kwargs_269382 = {}
    # Getting the type of 'tk' (line 26)
    tk_269380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 30), 'tk', False)
    # Obtaining the member 'interpaddr' of a type (line 26)
    interpaddr_269381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 30), tk_269380, 'interpaddr')
    # Calling interpaddr(args, kwargs) (line 26)
    interpaddr_call_result_269383 = invoke(stypy.reporting.localization.Localization(__file__, 26, 30), interpaddr_269381, *[], **kwargs_269382)
    
    int_269384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 47), 'int')
    # Processing the call keyword arguments (line 26)
    kwargs_269385 = {}
    # Getting the type of '_tkagg' (line 26)
    _tkagg_269378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), '_tkagg', False)
    # Obtaining the member 'tkinit' of a type (line 26)
    tkinit_269379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 16), _tkagg_269378, 'tkinit')
    # Calling tkinit(args, kwargs) (line 26)
    tkinit_call_result_269386 = invoke(stypy.reporting.localization.Localization(__file__, 26, 16), tkinit_269379, *[interpaddr_call_result_269383, int_269384], **kwargs_269385)
    
    # SSA branch for the except part of a try statement (line 25)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 25)
    module_type_store.open_ssa_branch('except')
    
    # Call to tkinit(...): (line 28)
    # Processing the call arguments (line 28)
    
    # Call to id(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'tk' (line 28)
    tk_269390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 33), 'tk', False)
    # Processing the call keyword arguments (line 28)
    kwargs_269391 = {}
    # Getting the type of 'id' (line 28)
    id_269389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 30), 'id', False)
    # Calling id(args, kwargs) (line 28)
    id_call_result_269392 = invoke(stypy.reporting.localization.Localization(__file__, 28, 30), id_269389, *[tk_269390], **kwargs_269391)
    
    int_269393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 38), 'int')
    # Processing the call keyword arguments (line 28)
    kwargs_269394 = {}
    # Getting the type of '_tkagg' (line 28)
    _tkagg_269387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), '_tkagg', False)
    # Obtaining the member 'tkinit' of a type (line 28)
    tkinit_269388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 16), _tkagg_269387, 'tkinit')
    # Calling tkinit(args, kwargs) (line 28)
    tkinit_call_result_269395 = invoke(stypy.reporting.localization.Localization(__file__, 28, 16), tkinit_269388, *[id_call_result_269392, int_269393], **kwargs_269394)
    
    # SSA join for try-except statement (line 25)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to call(...): (line 29)
    # Processing the call arguments (line 29)
    unicode_269398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 20), 'unicode', u'PyAggImagePhoto')
    # Getting the type of 'photoimage' (line 29)
    photoimage_269399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 39), 'photoimage', False)
    
    # Call to id(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'data' (line 30)
    data_269401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 23), 'data', False)
    # Processing the call keyword arguments (line 30)
    kwargs_269402 = {}
    # Getting the type of 'id' (line 30)
    id_269400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 20), 'id', False)
    # Calling id(args, kwargs) (line 30)
    id_call_result_269403 = invoke(stypy.reporting.localization.Localization(__file__, 30, 20), id_269400, *[data_269401], **kwargs_269402)
    
    # Getting the type of 'colormode' (line 30)
    colormode_269404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 30), 'colormode', False)
    
    # Call to id(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'bbox_array' (line 30)
    bbox_array_269406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 44), 'bbox_array', False)
    # Processing the call keyword arguments (line 30)
    kwargs_269407 = {}
    # Getting the type of 'id' (line 30)
    id_269405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 41), 'id', False)
    # Calling id(args, kwargs) (line 30)
    id_call_result_269408 = invoke(stypy.reporting.localization.Localization(__file__, 30, 41), id_269405, *[bbox_array_269406], **kwargs_269407)
    
    # Processing the call keyword arguments (line 29)
    kwargs_269409 = {}
    # Getting the type of 'tk' (line 29)
    tk_269396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'tk', False)
    # Obtaining the member 'call' of a type (line 29)
    call_269397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 12), tk_269396, 'call')
    # Calling call(args, kwargs) (line 29)
    call_call_result_269410 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), call_269397, *[unicode_269398, photoimage_269399, id_call_result_269403, colormode_269404, id_call_result_269408], **kwargs_269409)
    
    # SSA branch for the except part of a try statement (line 24)
    # SSA branch for the except 'Tuple' branch of a try statement (line 24)
    module_type_store.open_ssa_branch('except')
    # SSA join for try-except statement (line 24)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 19)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'blit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'blit' in the type store
    # Getting the type of 'stypy_return_type' (line 11)
    stypy_return_type_269411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_269411)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'blit'
    return stypy_return_type_269411

# Assigning a type to the variable 'blit' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'blit', blit)

@norecursion
def test(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test'
    module_type_store = module_type_store.open_function_context('test', 34, 0, False)
    
    # Passed parameters checking function
    test.stypy_localization = localization
    test.stypy_type_of_self = None
    test.stypy_type_store = module_type_store
    test.stypy_function_name = 'test'
    test.stypy_param_names_list = ['aggimage']
    test.stypy_varargs_param_name = None
    test.stypy_kwargs_param_name = None
    test.stypy_call_defaults = defaults
    test.stypy_call_varargs = varargs
    test.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test', ['aggimage'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test', localization, ['aggimage'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 4))
    
    # 'import time' statement (line 35)
    import time

    import_module(stypy.reporting.localization.Localization(__file__, 35, 4), 'time', time, module_type_store)
    
    
    # Assigning a Call to a Name (line 36):
    
    # Call to Tk(...): (line 36)
    # Processing the call keyword arguments (line 36)
    kwargs_269414 = {}
    # Getting the type of 'Tk' (line 36)
    Tk_269412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'Tk', False)
    # Obtaining the member 'Tk' of a type (line 36)
    Tk_269413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), Tk_269412, 'Tk')
    # Calling Tk(args, kwargs) (line 36)
    Tk_call_result_269415 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), Tk_269413, *[], **kwargs_269414)
    
    # Assigning a type to the variable 'r' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'r', Tk_call_result_269415)
    
    # Assigning a Call to a Name (line 37):
    
    # Call to Canvas(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'r' (line 37)
    r_269418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 18), 'r', False)
    # Processing the call keyword arguments (line 37)
    # Getting the type of 'aggimage' (line 37)
    aggimage_269419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 27), 'aggimage', False)
    # Obtaining the member 'width' of a type (line 37)
    width_269420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 27), aggimage_269419, 'width')
    keyword_269421 = width_269420
    # Getting the type of 'aggimage' (line 37)
    aggimage_269422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 50), 'aggimage', False)
    # Obtaining the member 'height' of a type (line 37)
    height_269423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 50), aggimage_269422, 'height')
    keyword_269424 = height_269423
    kwargs_269425 = {'width': keyword_269421, 'height': keyword_269424}
    # Getting the type of 'Tk' (line 37)
    Tk_269416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'Tk', False)
    # Obtaining the member 'Canvas' of a type (line 37)
    Canvas_269417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), Tk_269416, 'Canvas')
    # Calling Canvas(args, kwargs) (line 37)
    Canvas_call_result_269426 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), Canvas_269417, *[r_269418], **kwargs_269425)
    
    # Assigning a type to the variable 'c' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'c', Canvas_call_result_269426)
    
    # Call to pack(...): (line 38)
    # Processing the call keyword arguments (line 38)
    kwargs_269429 = {}
    # Getting the type of 'c' (line 38)
    c_269427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'c', False)
    # Obtaining the member 'pack' of a type (line 38)
    pack_269428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 4), c_269427, 'pack')
    # Calling pack(args, kwargs) (line 38)
    pack_call_result_269430 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), pack_269428, *[], **kwargs_269429)
    
    
    # Assigning a Call to a Name (line 39):
    
    # Call to PhotoImage(...): (line 39)
    # Processing the call keyword arguments (line 39)
    # Getting the type of 'aggimage' (line 39)
    aggimage_269433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 28), 'aggimage', False)
    # Obtaining the member 'width' of a type (line 39)
    width_269434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 28), aggimage_269433, 'width')
    keyword_269435 = width_269434
    # Getting the type of 'aggimage' (line 39)
    aggimage_269436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 51), 'aggimage', False)
    # Obtaining the member 'height' of a type (line 39)
    height_269437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 51), aggimage_269436, 'height')
    keyword_269438 = height_269437
    kwargs_269439 = {'width': keyword_269435, 'height': keyword_269438}
    # Getting the type of 'Tk' (line 39)
    Tk_269431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'Tk', False)
    # Obtaining the member 'PhotoImage' of a type (line 39)
    PhotoImage_269432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), Tk_269431, 'PhotoImage')
    # Calling PhotoImage(args, kwargs) (line 39)
    PhotoImage_call_result_269440 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), PhotoImage_269432, *[], **kwargs_269439)
    
    # Assigning a type to the variable 'p' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'p', PhotoImage_call_result_269440)
    
    # Call to blit(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'p' (line 40)
    p_269442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 9), 'p', False)
    # Getting the type of 'aggimage' (line 40)
    aggimage_269443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'aggimage', False)
    # Processing the call keyword arguments (line 40)
    kwargs_269444 = {}
    # Getting the type of 'blit' (line 40)
    blit_269441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'blit', False)
    # Calling blit(args, kwargs) (line 40)
    blit_call_result_269445 = invoke(stypy.reporting.localization.Localization(__file__, 40, 4), blit_269441, *[p_269442, aggimage_269443], **kwargs_269444)
    
    
    # Call to create_image(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'aggimage' (line 41)
    aggimage_269448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 19), 'aggimage', False)
    # Obtaining the member 'width' of a type (line 41)
    width_269449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 19), aggimage_269448, 'width')
    # Getting the type of 'aggimage' (line 41)
    aggimage_269450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 34), 'aggimage', False)
    # Obtaining the member 'height' of a type (line 41)
    height_269451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 34), aggimage_269450, 'height')
    # Processing the call keyword arguments (line 41)
    # Getting the type of 'p' (line 41)
    p_269452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 56), 'p', False)
    keyword_269453 = p_269452
    kwargs_269454 = {'image': keyword_269453}
    # Getting the type of 'c' (line 41)
    c_269446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'c', False)
    # Obtaining the member 'create_image' of a type (line 41)
    create_image_269447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 4), c_269446, 'create_image')
    # Calling create_image(args, kwargs) (line 41)
    create_image_call_result_269455 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), create_image_269447, *[width_269449, height_269451], **kwargs_269454)
    
    
    # Call to blit(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'p' (line 42)
    p_269457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 9), 'p', False)
    # Getting the type of 'aggimage' (line 42)
    aggimage_269458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'aggimage', False)
    # Processing the call keyword arguments (line 42)
    kwargs_269459 = {}
    # Getting the type of 'blit' (line 42)
    blit_269456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'blit', False)
    # Calling blit(args, kwargs) (line 42)
    blit_call_result_269460 = invoke(stypy.reporting.localization.Localization(__file__, 42, 4), blit_269456, *[p_269457, aggimage_269458], **kwargs_269459)
    
    
    # Getting the type of 'True' (line 43)
    True_269461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 10), 'True')
    # Testing the type of an if condition (line 43)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 4), True_269461)
    # SSA begins for while statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Call to update_idletasks(...): (line 43)
    # Processing the call keyword arguments (line 43)
    kwargs_269464 = {}
    # Getting the type of 'r' (line 43)
    r_269462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'r', False)
    # Obtaining the member 'update_idletasks' of a type (line 43)
    update_idletasks_269463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 16), r_269462, 'update_idletasks')
    # Calling update_idletasks(args, kwargs) (line 43)
    update_idletasks_call_result_269465 = invoke(stypy.reporting.localization.Localization(__file__, 43, 16), update_idletasks_269463, *[], **kwargs_269464)
    
    # SSA join for while statement (line 43)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_269466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_269466)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test'
    return stypy_return_type_269466

# Assigning a type to the variable 'test' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'test', test)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
