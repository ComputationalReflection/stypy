
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: def add_docstring(doc_attr, txt):
4:     doc_attr.__doc__ = txt
5: 
6: 
7: def add_newdoc(place, obj, doc):
8:     try:
9:         new = getattr(__import__(place, globals(), {}, [obj]), obj)
10:         if isinstance(doc, str):
11:             add_docstring(new, doc.strip())
12:         elif isinstance(doc, tuple):
13:             add_docstring(getattr(new, doc[0]), doc[1].strip())
14:         elif isinstance(doc, list):
15:             for val in doc:
16:                 add_docstring(getattr(new, val[0]), val[1].strip())
17:     except:
18:         pass
19: 
20: 
21: add_newdoc('numpy.core.multiarray', 'can_cast', '''example''')
22: add_newdoc('numpy.core.multiarray', 'ndarray', ('__doc__', 'sample'))

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def add_docstring(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'add_docstring'
    module_type_store = module_type_store.open_function_context('add_docstring', 3, 0, False)
    
    # Passed parameters checking function
    add_docstring.stypy_localization = localization
    add_docstring.stypy_type_of_self = None
    add_docstring.stypy_type_store = module_type_store
    add_docstring.stypy_function_name = 'add_docstring'
    add_docstring.stypy_param_names_list = ['doc_attr', 'txt']
    add_docstring.stypy_varargs_param_name = None
    add_docstring.stypy_kwargs_param_name = None
    add_docstring.stypy_call_defaults = defaults
    add_docstring.stypy_call_varargs = varargs
    add_docstring.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'add_docstring', ['doc_attr', 'txt'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'add_docstring', localization, ['doc_attr', 'txt'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'add_docstring(...)' code ##################

    
    # Assigning a Name to a Attribute (line 4):
    # Getting the type of 'txt' (line 4)
    txt_5541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 23), 'txt')
    # Getting the type of 'doc_attr' (line 4)
    doc_attr_5542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'doc_attr')
    # Setting the type of the member '__doc__' of a type (line 4)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 4), doc_attr_5542, '__doc__', txt_5541)
    
    # ################# End of 'add_docstring(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'add_docstring' in the type store
    # Getting the type of 'stypy_return_type' (line 3)
    stypy_return_type_5543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5543)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'add_docstring'
    return stypy_return_type_5543

# Assigning a type to the variable 'add_docstring' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'add_docstring', add_docstring)

@norecursion
def add_newdoc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'add_newdoc'
    module_type_store = module_type_store.open_function_context('add_newdoc', 7, 0, False)
    
    # Passed parameters checking function
    add_newdoc.stypy_localization = localization
    add_newdoc.stypy_type_of_self = None
    add_newdoc.stypy_type_store = module_type_store
    add_newdoc.stypy_function_name = 'add_newdoc'
    add_newdoc.stypy_param_names_list = ['place', 'obj', 'doc']
    add_newdoc.stypy_varargs_param_name = None
    add_newdoc.stypy_kwargs_param_name = None
    add_newdoc.stypy_call_defaults = defaults
    add_newdoc.stypy_call_varargs = varargs
    add_newdoc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'add_newdoc', ['place', 'obj', 'doc'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'add_newdoc', localization, ['place', 'obj', 'doc'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'add_newdoc(...)' code ##################

    
    
    # SSA begins for try-except statement (line 8)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 9):
    
    # Call to getattr(...): (line 9)
    # Processing the call arguments (line 9)
    
    # Call to __import__(...): (line 9)
    # Processing the call arguments (line 9)
    # Getting the type of 'place' (line 9)
    place_5546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 33), 'place', False)
    
    # Call to globals(...): (line 9)
    # Processing the call keyword arguments (line 9)
    kwargs_5548 = {}
    # Getting the type of 'globals' (line 9)
    globals_5547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 40), 'globals', False)
    # Calling globals(args, kwargs) (line 9)
    globals_call_result_5549 = invoke(stypy.reporting.localization.Localization(__file__, 9, 40), globals_5547, *[], **kwargs_5548)
    
    
    # Obtaining an instance of the builtin type 'dict' (line 9)
    dict_5550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 51), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 9)
    
    
    # Obtaining an instance of the builtin type 'list' (line 9)
    list_5551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 55), 'list')
    # Adding type elements to the builtin type 'list' instance (line 9)
    # Adding element type (line 9)
    # Getting the type of 'obj' (line 9)
    obj_5552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 56), 'obj', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 55), list_5551, obj_5552)
    
    # Processing the call keyword arguments (line 9)
    kwargs_5553 = {}
    # Getting the type of '__import__' (line 9)
    import___5545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 22), '__import__', False)
    # Calling __import__(args, kwargs) (line 9)
    import___call_result_5554 = invoke(stypy.reporting.localization.Localization(__file__, 9, 22), import___5545, *[place_5546, globals_call_result_5549, dict_5550, list_5551], **kwargs_5553)
    
    # Getting the type of 'obj' (line 9)
    obj_5555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 63), 'obj', False)
    # Processing the call keyword arguments (line 9)
    kwargs_5556 = {}
    # Getting the type of 'getattr' (line 9)
    getattr_5544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 14), 'getattr', False)
    # Calling getattr(args, kwargs) (line 9)
    getattr_call_result_5557 = invoke(stypy.reporting.localization.Localization(__file__, 9, 14), getattr_5544, *[import___call_result_5554, obj_5555], **kwargs_5556)
    
    # Assigning a type to the variable 'new' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'new', getattr_call_result_5557)
    
    # Type idiom detected: calculating its left and rigth part (line 10)
    # Getting the type of 'str' (line 10)
    str_5558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 27), 'str')
    # Getting the type of 'doc' (line 10)
    doc_5559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 22), 'doc')
    
    (may_be_5560, more_types_in_union_5561) = may_be_subtype(str_5558, doc_5559)

    if may_be_5560:

        if more_types_in_union_5561:
            # Runtime conditional SSA (line 10)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'doc' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'doc', remove_not_subtype_from_union(doc_5559, str))
        
        # Call to add_docstring(...): (line 11)
        # Processing the call arguments (line 11)
        # Getting the type of 'new' (line 11)
        new_5563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 26), 'new', False)
        
        # Call to strip(...): (line 11)
        # Processing the call keyword arguments (line 11)
        kwargs_5566 = {}
        # Getting the type of 'doc' (line 11)
        doc_5564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 31), 'doc', False)
        # Obtaining the member 'strip' of a type (line 11)
        strip_5565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 31), doc_5564, 'strip')
        # Calling strip(args, kwargs) (line 11)
        strip_call_result_5567 = invoke(stypy.reporting.localization.Localization(__file__, 11, 31), strip_5565, *[], **kwargs_5566)
        
        # Processing the call keyword arguments (line 11)
        kwargs_5568 = {}
        # Getting the type of 'add_docstring' (line 11)
        add_docstring_5562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'add_docstring', False)
        # Calling add_docstring(args, kwargs) (line 11)
        add_docstring_call_result_5569 = invoke(stypy.reporting.localization.Localization(__file__, 11, 12), add_docstring_5562, *[new_5563, strip_call_result_5567], **kwargs_5568)
        

        if more_types_in_union_5561:
            # Runtime conditional SSA for else branch (line 10)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_5560) or more_types_in_union_5561):
        # Assigning a type to the variable 'doc' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'doc', remove_subtype_from_union(doc_5559, str))
        
        # Type idiom detected: calculating its left and rigth part (line 12)
        # Getting the type of 'tuple' (line 12)
        tuple_5570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 29), 'tuple')
        # Getting the type of 'doc' (line 12)
        doc_5571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 24), 'doc')
        
        (may_be_5572, more_types_in_union_5573) = may_be_subtype(tuple_5570, doc_5571)

        if may_be_5572:

            if more_types_in_union_5573:
                # Runtime conditional SSA (line 12)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'doc' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 13), 'doc', remove_not_subtype_from_union(doc_5571, tuple))
            
            # Call to add_docstring(...): (line 13)
            # Processing the call arguments (line 13)
            
            # Call to getattr(...): (line 13)
            # Processing the call arguments (line 13)
            # Getting the type of 'new' (line 13)
            new_5576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 34), 'new', False)
            
            # Obtaining the type of the subscript
            int_5577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 43), 'int')
            # Getting the type of 'doc' (line 13)
            doc_5578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 39), 'doc', False)
            # Obtaining the member '__getitem__' of a type (line 13)
            getitem___5579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 39), doc_5578, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 13)
            subscript_call_result_5580 = invoke(stypy.reporting.localization.Localization(__file__, 13, 39), getitem___5579, int_5577)
            
            # Processing the call keyword arguments (line 13)
            kwargs_5581 = {}
            # Getting the type of 'getattr' (line 13)
            getattr_5575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 26), 'getattr', False)
            # Calling getattr(args, kwargs) (line 13)
            getattr_call_result_5582 = invoke(stypy.reporting.localization.Localization(__file__, 13, 26), getattr_5575, *[new_5576, subscript_call_result_5580], **kwargs_5581)
            
            
            # Call to strip(...): (line 13)
            # Processing the call keyword arguments (line 13)
            kwargs_5588 = {}
            
            # Obtaining the type of the subscript
            int_5583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 52), 'int')
            # Getting the type of 'doc' (line 13)
            doc_5584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 48), 'doc', False)
            # Obtaining the member '__getitem__' of a type (line 13)
            getitem___5585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 48), doc_5584, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 13)
            subscript_call_result_5586 = invoke(stypy.reporting.localization.Localization(__file__, 13, 48), getitem___5585, int_5583)
            
            # Obtaining the member 'strip' of a type (line 13)
            strip_5587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 48), subscript_call_result_5586, 'strip')
            # Calling strip(args, kwargs) (line 13)
            strip_call_result_5589 = invoke(stypy.reporting.localization.Localization(__file__, 13, 48), strip_5587, *[], **kwargs_5588)
            
            # Processing the call keyword arguments (line 13)
            kwargs_5590 = {}
            # Getting the type of 'add_docstring' (line 13)
            add_docstring_5574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'add_docstring', False)
            # Calling add_docstring(args, kwargs) (line 13)
            add_docstring_call_result_5591 = invoke(stypy.reporting.localization.Localization(__file__, 13, 12), add_docstring_5574, *[getattr_call_result_5582, strip_call_result_5589], **kwargs_5590)
            

            if more_types_in_union_5573:
                # Runtime conditional SSA for else branch (line 12)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_5572) or more_types_in_union_5573):
            # Assigning a type to the variable 'doc' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 13), 'doc', remove_subtype_from_union(doc_5571, tuple))
            
            # Type idiom detected: calculating its left and rigth part (line 14)
            # Getting the type of 'list' (line 14)
            list_5592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 29), 'list')
            # Getting the type of 'doc' (line 14)
            doc_5593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 24), 'doc')
            
            (may_be_5594, more_types_in_union_5595) = may_be_subtype(list_5592, doc_5593)

            if may_be_5594:

                if more_types_in_union_5595:
                    # Runtime conditional SSA (line 14)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'doc' (line 14)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 13), 'doc', remove_not_subtype_from_union(doc_5593, list))
                
                # Getting the type of 'doc' (line 15)
                doc_5596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 23), 'doc')
                # Testing the type of a for loop iterable (line 15)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 15, 12), doc_5596)
                # Getting the type of the for loop variable (line 15)
                for_loop_var_5597 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 15, 12), doc_5596)
                # Assigning a type to the variable 'val' (line 15)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'val', for_loop_var_5597)
                # SSA begins for a for statement (line 15)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to add_docstring(...): (line 16)
                # Processing the call arguments (line 16)
                
                # Call to getattr(...): (line 16)
                # Processing the call arguments (line 16)
                # Getting the type of 'new' (line 16)
                new_5600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 38), 'new', False)
                
                # Obtaining the type of the subscript
                int_5601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 47), 'int')
                # Getting the type of 'val' (line 16)
                val_5602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 43), 'val', False)
                # Obtaining the member '__getitem__' of a type (line 16)
                getitem___5603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 43), val_5602, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 16)
                subscript_call_result_5604 = invoke(stypy.reporting.localization.Localization(__file__, 16, 43), getitem___5603, int_5601)
                
                # Processing the call keyword arguments (line 16)
                kwargs_5605 = {}
                # Getting the type of 'getattr' (line 16)
                getattr_5599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 30), 'getattr', False)
                # Calling getattr(args, kwargs) (line 16)
                getattr_call_result_5606 = invoke(stypy.reporting.localization.Localization(__file__, 16, 30), getattr_5599, *[new_5600, subscript_call_result_5604], **kwargs_5605)
                
                
                # Call to strip(...): (line 16)
                # Processing the call keyword arguments (line 16)
                kwargs_5612 = {}
                
                # Obtaining the type of the subscript
                int_5607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 56), 'int')
                # Getting the type of 'val' (line 16)
                val_5608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 52), 'val', False)
                # Obtaining the member '__getitem__' of a type (line 16)
                getitem___5609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 52), val_5608, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 16)
                subscript_call_result_5610 = invoke(stypy.reporting.localization.Localization(__file__, 16, 52), getitem___5609, int_5607)
                
                # Obtaining the member 'strip' of a type (line 16)
                strip_5611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 52), subscript_call_result_5610, 'strip')
                # Calling strip(args, kwargs) (line 16)
                strip_call_result_5613 = invoke(stypy.reporting.localization.Localization(__file__, 16, 52), strip_5611, *[], **kwargs_5612)
                
                # Processing the call keyword arguments (line 16)
                kwargs_5614 = {}
                # Getting the type of 'add_docstring' (line 16)
                add_docstring_5598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'add_docstring', False)
                # Calling add_docstring(args, kwargs) (line 16)
                add_docstring_call_result_5615 = invoke(stypy.reporting.localization.Localization(__file__, 16, 16), add_docstring_5598, *[getattr_call_result_5606, strip_call_result_5613], **kwargs_5614)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()
                

                if more_types_in_union_5595:
                    # SSA join for if statement (line 14)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_5572 and more_types_in_union_5573):
                # SSA join for if statement (line 12)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_5560 and more_types_in_union_5561):
            # SSA join for if statement (line 10)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the except part of a try statement (line 8)
    # SSA branch for the except '<any exception>' branch of a try statement (line 8)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 8)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'add_newdoc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'add_newdoc' in the type store
    # Getting the type of 'stypy_return_type' (line 7)
    stypy_return_type_5616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5616)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'add_newdoc'
    return stypy_return_type_5616

# Assigning a type to the variable 'add_newdoc' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'add_newdoc', add_newdoc)

# Call to add_newdoc(...): (line 21)
# Processing the call arguments (line 21)
str_5618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 11), 'str', 'numpy.core.multiarray')
str_5619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 36), 'str', 'can_cast')
str_5620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 48), 'str', 'example')
# Processing the call keyword arguments (line 21)
kwargs_5621 = {}
# Getting the type of 'add_newdoc' (line 21)
add_newdoc_5617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 21)
add_newdoc_call_result_5622 = invoke(stypy.reporting.localization.Localization(__file__, 21, 0), add_newdoc_5617, *[str_5618, str_5619, str_5620], **kwargs_5621)


# Call to add_newdoc(...): (line 22)
# Processing the call arguments (line 22)
str_5624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'str', 'numpy.core.multiarray')
str_5625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 36), 'str', 'ndarray')

# Obtaining an instance of the builtin type 'tuple' (line 22)
tuple_5626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 22)
# Adding element type (line 22)
str_5627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 48), 'str', '__doc__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 48), tuple_5626, str_5627)
# Adding element type (line 22)
str_5628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 59), 'str', 'sample')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 48), tuple_5626, str_5628)

# Processing the call keyword arguments (line 22)
kwargs_5629 = {}
# Getting the type of 'add_newdoc' (line 22)
add_newdoc_5623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 22)
add_newdoc_call_result_5630 = invoke(stypy.reporting.localization.Localization(__file__, 22, 0), add_newdoc_5623, *[str_5624, str_5625, tuple_5626], **kwargs_5629)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
