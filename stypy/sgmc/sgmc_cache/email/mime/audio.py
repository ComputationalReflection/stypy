
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2001-2006 Python Software Foundation
2: # Author: Anthony Baxter
3: # Contact: email-sig@python.org
4: 
5: '''Class representing audio/* type MIME documents.'''
6: 
7: __all__ = ['MIMEAudio']
8: 
9: import sndhdr
10: 
11: from cStringIO import StringIO
12: from email import encoders
13: from email.mime.nonmultipart import MIMENonMultipart
14: 
15: 
16: 
17: _sndhdr_MIMEmap = {'au'  : 'basic',
18:                    'wav' :'x-wav',
19:                    'aiff':'x-aiff',
20:                    'aifc':'x-aiff',
21:                    }
22: 
23: # There are others in sndhdr that don't have MIME types. :(
24: # Additional ones to be added to sndhdr? midi, mp3, realaudio, wma??
25: def _whatsnd(data):
26:     '''Try to identify a sound file type.
27: 
28:     sndhdr.what() has a pretty cruddy interface, unfortunately.  This is why
29:     we re-do it here.  It would be easier to reverse engineer the Unix 'file'
30:     command and use the standard 'magic' file, as shipped with a modern Unix.
31:     '''
32:     hdr = data[:512]
33:     fakefile = StringIO(hdr)
34:     for testfn in sndhdr.tests:
35:         res = testfn(hdr, fakefile)
36:         if res is not None:
37:             return _sndhdr_MIMEmap.get(res[0])
38:     return None
39: 
40: 
41: 
42: class MIMEAudio(MIMENonMultipart):
43:     '''Class for generating audio/* MIME documents.'''
44: 
45:     def __init__(self, _audiodata, _subtype=None,
46:                  _encoder=encoders.encode_base64, **_params):
47:         '''Create an audio/* type MIME document.
48: 
49:         _audiodata is a string containing the raw audio data.  If this data
50:         can be decoded by the standard Python `sndhdr' module, then the
51:         subtype will be automatically included in the Content-Type header.
52:         Otherwise, you can specify  the specific audio subtype via the
53:         _subtype parameter.  If _subtype is not given, and no subtype can be
54:         guessed, a TypeError is raised.
55: 
56:         _encoder is a function which will perform the actual encoding for
57:         transport of the image data.  It takes one argument, which is this
58:         Image instance.  It should use get_payload() and set_payload() to
59:         change the payload to the encoded form.  It should also add any
60:         Content-Transfer-Encoding or other headers to the message as
61:         necessary.  The default encoding is Base64.
62: 
63:         Any additional keyword arguments are passed to the base class
64:         constructor, which turns them into parameters on the Content-Type
65:         header.
66:         '''
67:         if _subtype is None:
68:             _subtype = _whatsnd(_audiodata)
69:         if _subtype is None:
70:             raise TypeError('Could not find audio MIME subtype')
71:         MIMENonMultipart.__init__(self, 'audio', _subtype, **_params)
72:         self.set_payload(_audiodata)
73:         _encoder(self)
74: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_20785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'str', 'Class representing audio/* type MIME documents.')

# Assigning a List to a Name (line 7):
__all__ = ['MIMEAudio']
module_type_store.set_exportable_members(['MIMEAudio'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_20786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_20787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'MIMEAudio')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_20786, str_20787)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_20786)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import sndhdr' statement (line 9)
import sndhdr

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'sndhdr', sndhdr, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from cStringIO import StringIO' statement (line 11)
try:
    from cStringIO import StringIO

except:
    StringIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'cStringIO', None, module_type_store, ['StringIO'], [StringIO])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from email import encoders' statement (line 12)
try:
    from email import encoders

except:
    encoders = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'email', None, module_type_store, ['encoders'], [encoders])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from email.mime.nonmultipart import MIMENonMultipart' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/email/mime/')
import_20788 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'email.mime.nonmultipart')

if (type(import_20788) is not StypyTypeError):

    if (import_20788 != 'pyd_module'):
        __import__(import_20788)
        sys_modules_20789 = sys.modules[import_20788]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'email.mime.nonmultipart', sys_modules_20789.module_type_store, module_type_store, ['MIMENonMultipart'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_20789, sys_modules_20789.module_type_store, module_type_store)
    else:
        from email.mime.nonmultipart import MIMENonMultipart

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'email.mime.nonmultipart', None, module_type_store, ['MIMENonMultipart'], [MIMENonMultipart])

else:
    # Assigning a type to the variable 'email.mime.nonmultipart' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'email.mime.nonmultipart', import_20788)

remove_current_file_folder_from_path('C:/Python27/lib/email/mime/')


# Assigning a Dict to a Name (line 17):

# Obtaining an instance of the builtin type 'dict' (line 17)
dict_20790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 17)
# Adding element type (key, value) (line 17)
str_20791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 19), 'str', 'au')
str_20792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 27), 'str', 'basic')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 18), dict_20790, (str_20791, str_20792))
# Adding element type (key, value) (line 17)
str_20793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 19), 'str', 'wav')
str_20794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 26), 'str', 'x-wav')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 18), dict_20790, (str_20793, str_20794))
# Adding element type (key, value) (line 17)
str_20795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 19), 'str', 'aiff')
str_20796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'str', 'x-aiff')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 18), dict_20790, (str_20795, str_20796))
# Adding element type (key, value) (line 17)
str_20797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 19), 'str', 'aifc')
str_20798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'str', 'x-aiff')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 18), dict_20790, (str_20797, str_20798))

# Assigning a type to the variable '_sndhdr_MIMEmap' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), '_sndhdr_MIMEmap', dict_20790)

@norecursion
def _whatsnd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_whatsnd'
    module_type_store = module_type_store.open_function_context('_whatsnd', 25, 0, False)
    
    # Passed parameters checking function
    _whatsnd.stypy_localization = localization
    _whatsnd.stypy_type_of_self = None
    _whatsnd.stypy_type_store = module_type_store
    _whatsnd.stypy_function_name = '_whatsnd'
    _whatsnd.stypy_param_names_list = ['data']
    _whatsnd.stypy_varargs_param_name = None
    _whatsnd.stypy_kwargs_param_name = None
    _whatsnd.stypy_call_defaults = defaults
    _whatsnd.stypy_call_varargs = varargs
    _whatsnd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_whatsnd', ['data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_whatsnd', localization, ['data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_whatsnd(...)' code ##################

    str_20799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, (-1)), 'str', "Try to identify a sound file type.\n\n    sndhdr.what() has a pretty cruddy interface, unfortunately.  This is why\n    we re-do it here.  It would be easier to reverse engineer the Unix 'file'\n    command and use the standard 'magic' file, as shipped with a modern Unix.\n    ")
    
    # Assigning a Subscript to a Name (line 32):
    
    # Obtaining the type of the subscript
    int_20800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 16), 'int')
    slice_20801 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 32, 10), None, int_20800, None)
    # Getting the type of 'data' (line 32)
    data_20802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 10), 'data')
    # Obtaining the member '__getitem__' of a type (line 32)
    getitem___20803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 10), data_20802, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 32)
    subscript_call_result_20804 = invoke(stypy.reporting.localization.Localization(__file__, 32, 10), getitem___20803, slice_20801)
    
    # Assigning a type to the variable 'hdr' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'hdr', subscript_call_result_20804)
    
    # Assigning a Call to a Name (line 33):
    
    # Call to StringIO(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'hdr' (line 33)
    hdr_20806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 24), 'hdr', False)
    # Processing the call keyword arguments (line 33)
    kwargs_20807 = {}
    # Getting the type of 'StringIO' (line 33)
    StringIO_20805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'StringIO', False)
    # Calling StringIO(args, kwargs) (line 33)
    StringIO_call_result_20808 = invoke(stypy.reporting.localization.Localization(__file__, 33, 15), StringIO_20805, *[hdr_20806], **kwargs_20807)
    
    # Assigning a type to the variable 'fakefile' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'fakefile', StringIO_call_result_20808)
    
    # Getting the type of 'sndhdr' (line 34)
    sndhdr_20809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 18), 'sndhdr')
    # Obtaining the member 'tests' of a type (line 34)
    tests_20810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 18), sndhdr_20809, 'tests')
    # Assigning a type to the variable 'tests_20810' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'tests_20810', tests_20810)
    # Testing if the for loop is going to be iterated (line 34)
    # Testing the type of a for loop iterable (line 34)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 34, 4), tests_20810)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 34, 4), tests_20810):
        # Getting the type of the for loop variable (line 34)
        for_loop_var_20811 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 34, 4), tests_20810)
        # Assigning a type to the variable 'testfn' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'testfn', for_loop_var_20811)
        # SSA begins for a for statement (line 34)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 35):
        
        # Call to testfn(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'hdr' (line 35)
        hdr_20813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 21), 'hdr', False)
        # Getting the type of 'fakefile' (line 35)
        fakefile_20814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 26), 'fakefile', False)
        # Processing the call keyword arguments (line 35)
        kwargs_20815 = {}
        # Getting the type of 'testfn' (line 35)
        testfn_20812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 14), 'testfn', False)
        # Calling testfn(args, kwargs) (line 35)
        testfn_call_result_20816 = invoke(stypy.reporting.localization.Localization(__file__, 35, 14), testfn_20812, *[hdr_20813, fakefile_20814], **kwargs_20815)
        
        # Assigning a type to the variable 'res' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'res', testfn_call_result_20816)
        
        # Type idiom detected: calculating its left and rigth part (line 36)
        # Getting the type of 'res' (line 36)
        res_20817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'res')
        # Getting the type of 'None' (line 36)
        None_20818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 22), 'None')
        
        (may_be_20819, more_types_in_union_20820) = may_not_be_none(res_20817, None_20818)

        if may_be_20819:

            if more_types_in_union_20820:
                # Runtime conditional SSA (line 36)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to get(...): (line 37)
            # Processing the call arguments (line 37)
            
            # Obtaining the type of the subscript
            int_20823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 43), 'int')
            # Getting the type of 'res' (line 37)
            res_20824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 39), 'res', False)
            # Obtaining the member '__getitem__' of a type (line 37)
            getitem___20825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 39), res_20824, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 37)
            subscript_call_result_20826 = invoke(stypy.reporting.localization.Localization(__file__, 37, 39), getitem___20825, int_20823)
            
            # Processing the call keyword arguments (line 37)
            kwargs_20827 = {}
            # Getting the type of '_sndhdr_MIMEmap' (line 37)
            _sndhdr_MIMEmap_20821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 19), '_sndhdr_MIMEmap', False)
            # Obtaining the member 'get' of a type (line 37)
            get_20822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 19), _sndhdr_MIMEmap_20821, 'get')
            # Calling get(args, kwargs) (line 37)
            get_call_result_20828 = invoke(stypy.reporting.localization.Localization(__file__, 37, 19), get_20822, *[subscript_call_result_20826], **kwargs_20827)
            
            # Assigning a type to the variable 'stypy_return_type' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'stypy_return_type', get_call_result_20828)

            if more_types_in_union_20820:
                # SSA join for if statement (line 36)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'None' (line 38)
    None_20829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type', None_20829)
    
    # ################# End of '_whatsnd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_whatsnd' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_20830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20830)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_whatsnd'
    return stypy_return_type_20830

# Assigning a type to the variable '_whatsnd' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), '_whatsnd', _whatsnd)
# Declaration of the 'MIMEAudio' class
# Getting the type of 'MIMENonMultipart' (line 42)
MIMENonMultipart_20831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'MIMENonMultipart')

class MIMEAudio(MIMENonMultipart_20831, ):
    str_20832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 4), 'str', 'Class for generating audio/* MIME documents.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 45)
        None_20833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 44), 'None')
        # Getting the type of 'encoders' (line 46)
        encoders_20834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'encoders')
        # Obtaining the member 'encode_base64' of a type (line 46)
        encode_base64_20835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 26), encoders_20834, 'encode_base64')
        defaults = [None_20833, encode_base64_20835]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MIMEAudio.__init__', ['_audiodata', '_subtype', '_encoder'], None, '_params', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['_audiodata', '_subtype', '_encoder'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_20836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, (-1)), 'str', "Create an audio/* type MIME document.\n\n        _audiodata is a string containing the raw audio data.  If this data\n        can be decoded by the standard Python `sndhdr' module, then the\n        subtype will be automatically included in the Content-Type header.\n        Otherwise, you can specify  the specific audio subtype via the\n        _subtype parameter.  If _subtype is not given, and no subtype can be\n        guessed, a TypeError is raised.\n\n        _encoder is a function which will perform the actual encoding for\n        transport of the image data.  It takes one argument, which is this\n        Image instance.  It should use get_payload() and set_payload() to\n        change the payload to the encoded form.  It should also add any\n        Content-Transfer-Encoding or other headers to the message as\n        necessary.  The default encoding is Base64.\n\n        Any additional keyword arguments are passed to the base class\n        constructor, which turns them into parameters on the Content-Type\n        header.\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 67)
        # Getting the type of '_subtype' (line 67)
        _subtype_20837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), '_subtype')
        # Getting the type of 'None' (line 67)
        None_20838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 23), 'None')
        
        (may_be_20839, more_types_in_union_20840) = may_be_none(_subtype_20837, None_20838)

        if may_be_20839:

            if more_types_in_union_20840:
                # Runtime conditional SSA (line 67)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 68):
            
            # Call to _whatsnd(...): (line 68)
            # Processing the call arguments (line 68)
            # Getting the type of '_audiodata' (line 68)
            _audiodata_20842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 32), '_audiodata', False)
            # Processing the call keyword arguments (line 68)
            kwargs_20843 = {}
            # Getting the type of '_whatsnd' (line 68)
            _whatsnd_20841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 23), '_whatsnd', False)
            # Calling _whatsnd(args, kwargs) (line 68)
            _whatsnd_call_result_20844 = invoke(stypy.reporting.localization.Localization(__file__, 68, 23), _whatsnd_20841, *[_audiodata_20842], **kwargs_20843)
            
            # Assigning a type to the variable '_subtype' (line 68)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), '_subtype', _whatsnd_call_result_20844)

            if more_types_in_union_20840:
                # SSA join for if statement (line 67)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 69)
        # Getting the type of '_subtype' (line 69)
        _subtype_20845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), '_subtype')
        # Getting the type of 'None' (line 69)
        None_20846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 23), 'None')
        
        (may_be_20847, more_types_in_union_20848) = may_be_none(_subtype_20845, None_20846)

        if may_be_20847:

            if more_types_in_union_20848:
                # Runtime conditional SSA (line 69)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to TypeError(...): (line 70)
            # Processing the call arguments (line 70)
            str_20850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 28), 'str', 'Could not find audio MIME subtype')
            # Processing the call keyword arguments (line 70)
            kwargs_20851 = {}
            # Getting the type of 'TypeError' (line 70)
            TypeError_20849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 18), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 70)
            TypeError_call_result_20852 = invoke(stypy.reporting.localization.Localization(__file__, 70, 18), TypeError_20849, *[str_20850], **kwargs_20851)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 70, 12), TypeError_call_result_20852, 'raise parameter', BaseException)

            if more_types_in_union_20848:
                # SSA join for if statement (line 69)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of '_subtype' (line 69)
        _subtype_20853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), '_subtype')
        # Assigning a type to the variable '_subtype' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), '_subtype', remove_type_from_union(_subtype_20853, types.NoneType))
        
        # Call to __init__(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'self' (line 71)
        self_20856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 34), 'self', False)
        str_20857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 40), 'str', 'audio')
        # Getting the type of '_subtype' (line 71)
        _subtype_20858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 49), '_subtype', False)
        # Processing the call keyword arguments (line 71)
        # Getting the type of '_params' (line 71)
        _params_20859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 61), '_params', False)
        kwargs_20860 = {'_params_20859': _params_20859}
        # Getting the type of 'MIMENonMultipart' (line 71)
        MIMENonMultipart_20854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'MIMENonMultipart', False)
        # Obtaining the member '__init__' of a type (line 71)
        init___20855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), MIMENonMultipart_20854, '__init__')
        # Calling __init__(args, kwargs) (line 71)
        init___call_result_20861 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), init___20855, *[self_20856, str_20857, _subtype_20858], **kwargs_20860)
        
        
        # Call to set_payload(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of '_audiodata' (line 72)
        _audiodata_20864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 25), '_audiodata', False)
        # Processing the call keyword arguments (line 72)
        kwargs_20865 = {}
        # Getting the type of 'self' (line 72)
        self_20862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self', False)
        # Obtaining the member 'set_payload' of a type (line 72)
        set_payload_20863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_20862, 'set_payload')
        # Calling set_payload(args, kwargs) (line 72)
        set_payload_call_result_20866 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), set_payload_20863, *[_audiodata_20864], **kwargs_20865)
        
        
        # Call to _encoder(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'self' (line 73)
        self_20868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 17), 'self', False)
        # Processing the call keyword arguments (line 73)
        kwargs_20869 = {}
        # Getting the type of '_encoder' (line 73)
        _encoder_20867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), '_encoder', False)
        # Calling _encoder(args, kwargs) (line 73)
        _encoder_call_result_20870 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), _encoder_20867, *[self_20868], **kwargs_20869)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'MIMEAudio' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'MIMEAudio', MIMEAudio)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
