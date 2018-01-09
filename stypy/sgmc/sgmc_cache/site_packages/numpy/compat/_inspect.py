
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Subset of inspect module from upstream python
2: 
3: We use this instead of upstream because upstream inspect is slow to import, and
4: significanly contributes to numpy import times. Importing this copy has almost
5: no overhead.
6: 
7: '''
8: from __future__ import division, absolute_import, print_function
9: 
10: import types
11: 
12: __all__ = ['getargspec', 'formatargspec']
13: 
14: # ----------------------------------------------------------- type-checking
15: def ismethod(object):
16:     '''Return true if the object is an instance method.
17: 
18:     Instance method objects provide these attributes:
19:         __doc__         documentation string
20:         __name__        name with which this method was defined
21:         im_class        class object in which this method belongs
22:         im_func         function object containing implementation of method
23:         im_self         instance to which this method is bound, or None
24: 
25:     '''
26:     return isinstance(object, types.MethodType)
27: 
28: def isfunction(object):
29:     '''Return true if the object is a user-defined function.
30: 
31:     Function objects provide these attributes:
32:         __doc__         documentation string
33:         __name__        name with which this function was defined
34:         func_code       code object containing compiled function bytecode
35:         func_defaults   tuple of any default values for arguments
36:         func_doc        (same as __doc__)
37:         func_globals    global namespace in which this function was defined
38:         func_name       (same as __name__)
39: 
40:     '''
41:     return isinstance(object, types.FunctionType)
42: 
43: def iscode(object):
44:     '''Return true if the object is a code object.
45: 
46:     Code objects provide these attributes:
47:         co_argcount     number of arguments (not including * or ** args)
48:         co_code         string of raw compiled bytecode
49:         co_consts       tuple of constants used in the bytecode
50:         co_filename     name of file in which this code object was created
51:         co_firstlineno  number of first line in Python source code
52:         co_flags        bitmap: 1=optimized | 2=newlocals | 4=*arg | 8=**arg
53:         co_lnotab       encoded mapping of line numbers to bytecode indices
54:         co_name         name with which this code object was defined
55:         co_names        tuple of names of local variables
56:         co_nlocals      number of local variables
57:         co_stacksize    virtual machine stack space required
58:         co_varnames     tuple of names of arguments and local variables
59:         
60:     '''
61:     return isinstance(object, types.CodeType)
62: 
63: # ------------------------------------------------ argument list extraction
64: # These constants are from Python's compile.h.
65: CO_OPTIMIZED, CO_NEWLOCALS, CO_VARARGS, CO_VARKEYWORDS = 1, 2, 4, 8
66: 
67: def getargs(co):
68:     '''Get information about the arguments accepted by a code object.
69: 
70:     Three things are returned: (args, varargs, varkw), where 'args' is
71:     a list of argument names (possibly containing nested lists), and
72:     'varargs' and 'varkw' are the names of the * and ** arguments or None.
73: 
74:     '''
75: 
76:     if not iscode(co):
77:         raise TypeError('arg is not a code object')
78: 
79:     nargs = co.co_argcount
80:     names = co.co_varnames
81:     args = list(names[:nargs])
82: 
83:     # The following acrobatics are for anonymous (tuple) arguments.
84:     # Which we do not need to support, so remove to avoid importing
85:     # the dis module.
86:     for i in range(nargs):
87:         if args[i][:1] in ['', '.']:
88:             raise TypeError("tuple function arguments are not supported")
89:     varargs = None
90:     if co.co_flags & CO_VARARGS:
91:         varargs = co.co_varnames[nargs]
92:         nargs = nargs + 1
93:     varkw = None
94:     if co.co_flags & CO_VARKEYWORDS:
95:         varkw = co.co_varnames[nargs]
96:     return args, varargs, varkw
97: 
98: def getargspec(func):
99:     '''Get the names and default values of a function's arguments.
100: 
101:     A tuple of four things is returned: (args, varargs, varkw, defaults).
102:     'args' is a list of the argument names (it may contain nested lists).
103:     'varargs' and 'varkw' are the names of the * and ** arguments or None.
104:     'defaults' is an n-tuple of the default values of the last n arguments.
105: 
106:     '''
107: 
108:     if ismethod(func):
109:         func = func.__func__
110:     if not isfunction(func):
111:         raise TypeError('arg is not a Python function')
112:     args, varargs, varkw = getargs(func.__code__)
113:     return args, varargs, varkw, func.__defaults__
114: 
115: def getargvalues(frame):
116:     '''Get information about arguments passed into a particular frame.
117: 
118:     A tuple of four things is returned: (args, varargs, varkw, locals).
119:     'args' is a list of the argument names (it may contain nested lists).
120:     'varargs' and 'varkw' are the names of the * and ** arguments or None.
121:     'locals' is the locals dictionary of the given frame.
122:     
123:     '''
124:     args, varargs, varkw = getargs(frame.f_code)
125:     return args, varargs, varkw, frame.f_locals
126: 
127: def joinseq(seq):
128:     if len(seq) == 1:
129:         return '(' + seq[0] + ',)'
130:     else:
131:         return '(' + ', '.join(seq) + ')'
132: 
133: def strseq(object, convert, join=joinseq):
134:     '''Recursively walk a sequence, stringifying each element.
135: 
136:     '''
137:     if type(object) in [list, tuple]:
138:         return join([strseq(_o, convert, join) for _o in object])
139:     else:
140:         return convert(object)
141: 
142: def formatargspec(args, varargs=None, varkw=None, defaults=None,
143:                   formatarg=str,
144:                   formatvarargs=lambda name: '*' + name,
145:                   formatvarkw=lambda name: '**' + name,
146:                   formatvalue=lambda value: '=' + repr(value),
147:                   join=joinseq):
148:     '''Format an argument spec from the 4 values returned by getargspec.
149: 
150:     The first four arguments are (args, varargs, varkw, defaults).  The
151:     other four arguments are the corresponding optional formatting functions
152:     that are called to turn names and values into strings.  The ninth
153:     argument is an optional function to format the sequence of arguments.
154: 
155:     '''
156:     specs = []
157:     if defaults:
158:         firstdefault = len(args) - len(defaults)
159:     for i in range(len(args)):
160:         spec = strseq(args[i], formatarg, join)
161:         if defaults and i >= firstdefault:
162:             spec = spec + formatvalue(defaults[i - firstdefault])
163:         specs.append(spec)
164:     if varargs is not None:
165:         specs.append(formatvarargs(varargs))
166:     if varkw is not None:
167:         specs.append(formatvarkw(varkw))
168:     return '(' + ', '.join(specs) + ')'
169: 
170: def formatargvalues(args, varargs, varkw, locals,
171:                     formatarg=str,
172:                     formatvarargs=lambda name: '*' + name,
173:                     formatvarkw=lambda name: '**' + name,
174:                     formatvalue=lambda value: '=' + repr(value),
175:                     join=joinseq):
176:     '''Format an argument spec from the 4 values returned by getargvalues.
177: 
178:     The first four arguments are (args, varargs, varkw, locals).  The
179:     next four arguments are the corresponding optional formatting functions
180:     that are called to turn names and values into strings.  The ninth
181:     argument is an optional function to format the sequence of arguments.
182: 
183:     '''
184:     def convert(name, locals=locals,
185:                 formatarg=formatarg, formatvalue=formatvalue):
186:         return formatarg(name) + formatvalue(locals[name])
187:     specs = []
188:     for i in range(len(args)):
189:         specs.append(strseq(args[i], convert, join))
190:     if varargs:
191:         specs.append(formatvarargs(varargs) + formatvalue(locals[varargs]))
192:     if varkw:
193:         specs.append(formatvarkw(varkw) + formatvalue(locals[varkw]))
194:     return '(' + ', '.join(specs) + ')'
195: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_25856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', 'Subset of inspect module from upstream python\n\nWe use this instead of upstream because upstream inspect is slow to import, and\nsignificanly contributes to numpy import times. Importing this copy has almost\nno overhead.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import types' statement (line 10)
import types

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'types', types, module_type_store)


# Assigning a List to a Name (line 12):

# Assigning a List to a Name (line 12):
__all__ = ['getargspec', 'formatargspec']
module_type_store.set_exportable_members(['getargspec', 'formatargspec'])

# Obtaining an instance of the builtin type 'list' (line 12)
list_25857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
str_25858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'str', 'getargspec')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_25857, str_25858)
# Adding element type (line 12)
str_25859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 25), 'str', 'formatargspec')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_25857, str_25859)

# Assigning a type to the variable '__all__' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '__all__', list_25857)

@norecursion
def ismethod(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ismethod'
    module_type_store = module_type_store.open_function_context('ismethod', 15, 0, False)
    
    # Passed parameters checking function
    ismethod.stypy_localization = localization
    ismethod.stypy_type_of_self = None
    ismethod.stypy_type_store = module_type_store
    ismethod.stypy_function_name = 'ismethod'
    ismethod.stypy_param_names_list = ['object']
    ismethod.stypy_varargs_param_name = None
    ismethod.stypy_kwargs_param_name = None
    ismethod.stypy_call_defaults = defaults
    ismethod.stypy_call_varargs = varargs
    ismethod.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ismethod', ['object'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ismethod', localization, ['object'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ismethod(...)' code ##################

    str_25860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, (-1)), 'str', 'Return true if the object is an instance method.\n\n    Instance method objects provide these attributes:\n        __doc__         documentation string\n        __name__        name with which this method was defined\n        im_class        class object in which this method belongs\n        im_func         function object containing implementation of method\n        im_self         instance to which this method is bound, or None\n\n    ')
    
    # Call to isinstance(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'object' (line 26)
    object_25862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 22), 'object', False)
    # Getting the type of 'types' (line 26)
    types_25863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 30), 'types', False)
    # Obtaining the member 'MethodType' of a type (line 26)
    MethodType_25864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 30), types_25863, 'MethodType')
    # Processing the call keyword arguments (line 26)
    kwargs_25865 = {}
    # Getting the type of 'isinstance' (line 26)
    isinstance_25861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 26)
    isinstance_call_result_25866 = invoke(stypy.reporting.localization.Localization(__file__, 26, 11), isinstance_25861, *[object_25862, MethodType_25864], **kwargs_25865)
    
    # Assigning a type to the variable 'stypy_return_type' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type', isinstance_call_result_25866)
    
    # ################# End of 'ismethod(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ismethod' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_25867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25867)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ismethod'
    return stypy_return_type_25867

# Assigning a type to the variable 'ismethod' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'ismethod', ismethod)

@norecursion
def isfunction(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isfunction'
    module_type_store = module_type_store.open_function_context('isfunction', 28, 0, False)
    
    # Passed parameters checking function
    isfunction.stypy_localization = localization
    isfunction.stypy_type_of_self = None
    isfunction.stypy_type_store = module_type_store
    isfunction.stypy_function_name = 'isfunction'
    isfunction.stypy_param_names_list = ['object']
    isfunction.stypy_varargs_param_name = None
    isfunction.stypy_kwargs_param_name = None
    isfunction.stypy_call_defaults = defaults
    isfunction.stypy_call_varargs = varargs
    isfunction.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isfunction', ['object'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isfunction', localization, ['object'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isfunction(...)' code ##################

    str_25868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, (-1)), 'str', 'Return true if the object is a user-defined function.\n\n    Function objects provide these attributes:\n        __doc__         documentation string\n        __name__        name with which this function was defined\n        func_code       code object containing compiled function bytecode\n        func_defaults   tuple of any default values for arguments\n        func_doc        (same as __doc__)\n        func_globals    global namespace in which this function was defined\n        func_name       (same as __name__)\n\n    ')
    
    # Call to isinstance(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'object' (line 41)
    object_25870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'object', False)
    # Getting the type of 'types' (line 41)
    types_25871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 30), 'types', False)
    # Obtaining the member 'FunctionType' of a type (line 41)
    FunctionType_25872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 30), types_25871, 'FunctionType')
    # Processing the call keyword arguments (line 41)
    kwargs_25873 = {}
    # Getting the type of 'isinstance' (line 41)
    isinstance_25869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 41)
    isinstance_call_result_25874 = invoke(stypy.reporting.localization.Localization(__file__, 41, 11), isinstance_25869, *[object_25870, FunctionType_25872], **kwargs_25873)
    
    # Assigning a type to the variable 'stypy_return_type' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type', isinstance_call_result_25874)
    
    # ################# End of 'isfunction(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isfunction' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_25875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25875)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isfunction'
    return stypy_return_type_25875

# Assigning a type to the variable 'isfunction' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'isfunction', isfunction)

@norecursion
def iscode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iscode'
    module_type_store = module_type_store.open_function_context('iscode', 43, 0, False)
    
    # Passed parameters checking function
    iscode.stypy_localization = localization
    iscode.stypy_type_of_self = None
    iscode.stypy_type_store = module_type_store
    iscode.stypy_function_name = 'iscode'
    iscode.stypy_param_names_list = ['object']
    iscode.stypy_varargs_param_name = None
    iscode.stypy_kwargs_param_name = None
    iscode.stypy_call_defaults = defaults
    iscode.stypy_call_varargs = varargs
    iscode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iscode', ['object'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iscode', localization, ['object'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iscode(...)' code ##################

    str_25876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, (-1)), 'str', 'Return true if the object is a code object.\n\n    Code objects provide these attributes:\n        co_argcount     number of arguments (not including * or ** args)\n        co_code         string of raw compiled bytecode\n        co_consts       tuple of constants used in the bytecode\n        co_filename     name of file in which this code object was created\n        co_firstlineno  number of first line in Python source code\n        co_flags        bitmap: 1=optimized | 2=newlocals | 4=*arg | 8=**arg\n        co_lnotab       encoded mapping of line numbers to bytecode indices\n        co_name         name with which this code object was defined\n        co_names        tuple of names of local variables\n        co_nlocals      number of local variables\n        co_stacksize    virtual machine stack space required\n        co_varnames     tuple of names of arguments and local variables\n        \n    ')
    
    # Call to isinstance(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'object' (line 61)
    object_25878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 22), 'object', False)
    # Getting the type of 'types' (line 61)
    types_25879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'types', False)
    # Obtaining the member 'CodeType' of a type (line 61)
    CodeType_25880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 30), types_25879, 'CodeType')
    # Processing the call keyword arguments (line 61)
    kwargs_25881 = {}
    # Getting the type of 'isinstance' (line 61)
    isinstance_25877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 61)
    isinstance_call_result_25882 = invoke(stypy.reporting.localization.Localization(__file__, 61, 11), isinstance_25877, *[object_25878, CodeType_25880], **kwargs_25881)
    
    # Assigning a type to the variable 'stypy_return_type' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type', isinstance_call_result_25882)
    
    # ################# End of 'iscode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iscode' in the type store
    # Getting the type of 'stypy_return_type' (line 43)
    stypy_return_type_25883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25883)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iscode'
    return stypy_return_type_25883

# Assigning a type to the variable 'iscode' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'iscode', iscode)

# Assigning a Tuple to a Tuple (line 65):

# Assigning a Num to a Name (line 65):
int_25884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 57), 'int')
# Assigning a type to the variable 'tuple_assignment_25844' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'tuple_assignment_25844', int_25884)

# Assigning a Num to a Name (line 65):
int_25885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 60), 'int')
# Assigning a type to the variable 'tuple_assignment_25845' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'tuple_assignment_25845', int_25885)

# Assigning a Num to a Name (line 65):
int_25886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 63), 'int')
# Assigning a type to the variable 'tuple_assignment_25846' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'tuple_assignment_25846', int_25886)

# Assigning a Num to a Name (line 65):
int_25887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 66), 'int')
# Assigning a type to the variable 'tuple_assignment_25847' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'tuple_assignment_25847', int_25887)

# Assigning a Name to a Name (line 65):
# Getting the type of 'tuple_assignment_25844' (line 65)
tuple_assignment_25844_25888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'tuple_assignment_25844')
# Assigning a type to the variable 'CO_OPTIMIZED' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'CO_OPTIMIZED', tuple_assignment_25844_25888)

# Assigning a Name to a Name (line 65):
# Getting the type of 'tuple_assignment_25845' (line 65)
tuple_assignment_25845_25889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'tuple_assignment_25845')
# Assigning a type to the variable 'CO_NEWLOCALS' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 14), 'CO_NEWLOCALS', tuple_assignment_25845_25889)

# Assigning a Name to a Name (line 65):
# Getting the type of 'tuple_assignment_25846' (line 65)
tuple_assignment_25846_25890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'tuple_assignment_25846')
# Assigning a type to the variable 'CO_VARARGS' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 28), 'CO_VARARGS', tuple_assignment_25846_25890)

# Assigning a Name to a Name (line 65):
# Getting the type of 'tuple_assignment_25847' (line 65)
tuple_assignment_25847_25891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'tuple_assignment_25847')
# Assigning a type to the variable 'CO_VARKEYWORDS' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 40), 'CO_VARKEYWORDS', tuple_assignment_25847_25891)

@norecursion
def getargs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getargs'
    module_type_store = module_type_store.open_function_context('getargs', 67, 0, False)
    
    # Passed parameters checking function
    getargs.stypy_localization = localization
    getargs.stypy_type_of_self = None
    getargs.stypy_type_store = module_type_store
    getargs.stypy_function_name = 'getargs'
    getargs.stypy_param_names_list = ['co']
    getargs.stypy_varargs_param_name = None
    getargs.stypy_kwargs_param_name = None
    getargs.stypy_call_defaults = defaults
    getargs.stypy_call_varargs = varargs
    getargs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getargs', ['co'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getargs', localization, ['co'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getargs(...)' code ##################

    str_25892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, (-1)), 'str', "Get information about the arguments accepted by a code object.\n\n    Three things are returned: (args, varargs, varkw), where 'args' is\n    a list of argument names (possibly containing nested lists), and\n    'varargs' and 'varkw' are the names of the * and ** arguments or None.\n\n    ")
    
    
    
    # Call to iscode(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'co' (line 76)
    co_25894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 18), 'co', False)
    # Processing the call keyword arguments (line 76)
    kwargs_25895 = {}
    # Getting the type of 'iscode' (line 76)
    iscode_25893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'iscode', False)
    # Calling iscode(args, kwargs) (line 76)
    iscode_call_result_25896 = invoke(stypy.reporting.localization.Localization(__file__, 76, 11), iscode_25893, *[co_25894], **kwargs_25895)
    
    # Applying the 'not' unary operator (line 76)
    result_not__25897 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 7), 'not', iscode_call_result_25896)
    
    # Testing the type of an if condition (line 76)
    if_condition_25898 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 4), result_not__25897)
    # Assigning a type to the variable 'if_condition_25898' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'if_condition_25898', if_condition_25898)
    # SSA begins for if statement (line 76)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 77)
    # Processing the call arguments (line 77)
    str_25900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 24), 'str', 'arg is not a code object')
    # Processing the call keyword arguments (line 77)
    kwargs_25901 = {}
    # Getting the type of 'TypeError' (line 77)
    TypeError_25899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 77)
    TypeError_call_result_25902 = invoke(stypy.reporting.localization.Localization(__file__, 77, 14), TypeError_25899, *[str_25900], **kwargs_25901)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 77, 8), TypeError_call_result_25902, 'raise parameter', BaseException)
    # SSA join for if statement (line 76)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 79):
    
    # Assigning a Attribute to a Name (line 79):
    # Getting the type of 'co' (line 79)
    co_25903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'co')
    # Obtaining the member 'co_argcount' of a type (line 79)
    co_argcount_25904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), co_25903, 'co_argcount')
    # Assigning a type to the variable 'nargs' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'nargs', co_argcount_25904)
    
    # Assigning a Attribute to a Name (line 80):
    
    # Assigning a Attribute to a Name (line 80):
    # Getting the type of 'co' (line 80)
    co_25905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'co')
    # Obtaining the member 'co_varnames' of a type (line 80)
    co_varnames_25906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), co_25905, 'co_varnames')
    # Assigning a type to the variable 'names' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'names', co_varnames_25906)
    
    # Assigning a Call to a Name (line 81):
    
    # Assigning a Call to a Name (line 81):
    
    # Call to list(...): (line 81)
    # Processing the call arguments (line 81)
    
    # Obtaining the type of the subscript
    # Getting the type of 'nargs' (line 81)
    nargs_25908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 23), 'nargs', False)
    slice_25909 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 81, 16), None, nargs_25908, None)
    # Getting the type of 'names' (line 81)
    names_25910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'names', False)
    # Obtaining the member '__getitem__' of a type (line 81)
    getitem___25911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 16), names_25910, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 81)
    subscript_call_result_25912 = invoke(stypy.reporting.localization.Localization(__file__, 81, 16), getitem___25911, slice_25909)
    
    # Processing the call keyword arguments (line 81)
    kwargs_25913 = {}
    # Getting the type of 'list' (line 81)
    list_25907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'list', False)
    # Calling list(args, kwargs) (line 81)
    list_call_result_25914 = invoke(stypy.reporting.localization.Localization(__file__, 81, 11), list_25907, *[subscript_call_result_25912], **kwargs_25913)
    
    # Assigning a type to the variable 'args' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'args', list_call_result_25914)
    
    
    # Call to range(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'nargs' (line 86)
    nargs_25916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'nargs', False)
    # Processing the call keyword arguments (line 86)
    kwargs_25917 = {}
    # Getting the type of 'range' (line 86)
    range_25915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 13), 'range', False)
    # Calling range(args, kwargs) (line 86)
    range_call_result_25918 = invoke(stypy.reporting.localization.Localization(__file__, 86, 13), range_25915, *[nargs_25916], **kwargs_25917)
    
    # Testing the type of a for loop iterable (line 86)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 86, 4), range_call_result_25918)
    # Getting the type of the for loop variable (line 86)
    for_loop_var_25919 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 86, 4), range_call_result_25918)
    # Assigning a type to the variable 'i' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'i', for_loop_var_25919)
    # SSA begins for a for statement (line 86)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    int_25920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 20), 'int')
    slice_25921 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 87, 11), None, int_25920, None)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 87)
    i_25922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'i')
    # Getting the type of 'args' (line 87)
    args_25923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 11), 'args')
    # Obtaining the member '__getitem__' of a type (line 87)
    getitem___25924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 11), args_25923, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 87)
    subscript_call_result_25925 = invoke(stypy.reporting.localization.Localization(__file__, 87, 11), getitem___25924, i_25922)
    
    # Obtaining the member '__getitem__' of a type (line 87)
    getitem___25926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 11), subscript_call_result_25925, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 87)
    subscript_call_result_25927 = invoke(stypy.reporting.localization.Localization(__file__, 87, 11), getitem___25926, slice_25921)
    
    
    # Obtaining an instance of the builtin type 'list' (line 87)
    list_25928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 87)
    # Adding element type (line 87)
    str_25929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 27), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 26), list_25928, str_25929)
    # Adding element type (line 87)
    str_25930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 31), 'str', '.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 26), list_25928, str_25930)
    
    # Applying the binary operator 'in' (line 87)
    result_contains_25931 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 11), 'in', subscript_call_result_25927, list_25928)
    
    # Testing the type of an if condition (line 87)
    if_condition_25932 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 8), result_contains_25931)
    # Assigning a type to the variable 'if_condition_25932' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'if_condition_25932', if_condition_25932)
    # SSA begins for if statement (line 87)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 88)
    # Processing the call arguments (line 88)
    str_25934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 28), 'str', 'tuple function arguments are not supported')
    # Processing the call keyword arguments (line 88)
    kwargs_25935 = {}
    # Getting the type of 'TypeError' (line 88)
    TypeError_25933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 88)
    TypeError_call_result_25936 = invoke(stypy.reporting.localization.Localization(__file__, 88, 18), TypeError_25933, *[str_25934], **kwargs_25935)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 88, 12), TypeError_call_result_25936, 'raise parameter', BaseException)
    # SSA join for if statement (line 87)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 89):
    
    # Assigning a Name to a Name (line 89):
    # Getting the type of 'None' (line 89)
    None_25937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 14), 'None')
    # Assigning a type to the variable 'varargs' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'varargs', None_25937)
    
    # Getting the type of 'co' (line 90)
    co_25938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 7), 'co')
    # Obtaining the member 'co_flags' of a type (line 90)
    co_flags_25939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 7), co_25938, 'co_flags')
    # Getting the type of 'CO_VARARGS' (line 90)
    CO_VARARGS_25940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 21), 'CO_VARARGS')
    # Applying the binary operator '&' (line 90)
    result_and__25941 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 7), '&', co_flags_25939, CO_VARARGS_25940)
    
    # Testing the type of an if condition (line 90)
    if_condition_25942 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 4), result_and__25941)
    # Assigning a type to the variable 'if_condition_25942' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'if_condition_25942', if_condition_25942)
    # SSA begins for if statement (line 90)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 91):
    
    # Assigning a Subscript to a Name (line 91):
    
    # Obtaining the type of the subscript
    # Getting the type of 'nargs' (line 91)
    nargs_25943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 33), 'nargs')
    # Getting the type of 'co' (line 91)
    co_25944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 18), 'co')
    # Obtaining the member 'co_varnames' of a type (line 91)
    co_varnames_25945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 18), co_25944, 'co_varnames')
    # Obtaining the member '__getitem__' of a type (line 91)
    getitem___25946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 18), co_varnames_25945, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 91)
    subscript_call_result_25947 = invoke(stypy.reporting.localization.Localization(__file__, 91, 18), getitem___25946, nargs_25943)
    
    # Assigning a type to the variable 'varargs' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'varargs', subscript_call_result_25947)
    
    # Assigning a BinOp to a Name (line 92):
    
    # Assigning a BinOp to a Name (line 92):
    # Getting the type of 'nargs' (line 92)
    nargs_25948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'nargs')
    int_25949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 24), 'int')
    # Applying the binary operator '+' (line 92)
    result_add_25950 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 16), '+', nargs_25948, int_25949)
    
    # Assigning a type to the variable 'nargs' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'nargs', result_add_25950)
    # SSA join for if statement (line 90)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 93):
    
    # Assigning a Name to a Name (line 93):
    # Getting the type of 'None' (line 93)
    None_25951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'None')
    # Assigning a type to the variable 'varkw' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'varkw', None_25951)
    
    # Getting the type of 'co' (line 94)
    co_25952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 7), 'co')
    # Obtaining the member 'co_flags' of a type (line 94)
    co_flags_25953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 7), co_25952, 'co_flags')
    # Getting the type of 'CO_VARKEYWORDS' (line 94)
    CO_VARKEYWORDS_25954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 21), 'CO_VARKEYWORDS')
    # Applying the binary operator '&' (line 94)
    result_and__25955 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 7), '&', co_flags_25953, CO_VARKEYWORDS_25954)
    
    # Testing the type of an if condition (line 94)
    if_condition_25956 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 4), result_and__25955)
    # Assigning a type to the variable 'if_condition_25956' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'if_condition_25956', if_condition_25956)
    # SSA begins for if statement (line 94)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 95):
    
    # Assigning a Subscript to a Name (line 95):
    
    # Obtaining the type of the subscript
    # Getting the type of 'nargs' (line 95)
    nargs_25957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 31), 'nargs')
    # Getting the type of 'co' (line 95)
    co_25958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'co')
    # Obtaining the member 'co_varnames' of a type (line 95)
    co_varnames_25959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 16), co_25958, 'co_varnames')
    # Obtaining the member '__getitem__' of a type (line 95)
    getitem___25960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 16), co_varnames_25959, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 95)
    subscript_call_result_25961 = invoke(stypy.reporting.localization.Localization(__file__, 95, 16), getitem___25960, nargs_25957)
    
    # Assigning a type to the variable 'varkw' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'varkw', subscript_call_result_25961)
    # SSA join for if statement (line 94)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 96)
    tuple_25962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 96)
    # Adding element type (line 96)
    # Getting the type of 'args' (line 96)
    args_25963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'args')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 11), tuple_25962, args_25963)
    # Adding element type (line 96)
    # Getting the type of 'varargs' (line 96)
    varargs_25964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 17), 'varargs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 11), tuple_25962, varargs_25964)
    # Adding element type (line 96)
    # Getting the type of 'varkw' (line 96)
    varkw_25965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'varkw')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 11), tuple_25962, varkw_25965)
    
    # Assigning a type to the variable 'stypy_return_type' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type', tuple_25962)
    
    # ################# End of 'getargs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getargs' in the type store
    # Getting the type of 'stypy_return_type' (line 67)
    stypy_return_type_25966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25966)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getargs'
    return stypy_return_type_25966

# Assigning a type to the variable 'getargs' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'getargs', getargs)

@norecursion
def getargspec(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getargspec'
    module_type_store = module_type_store.open_function_context('getargspec', 98, 0, False)
    
    # Passed parameters checking function
    getargspec.stypy_localization = localization
    getargspec.stypy_type_of_self = None
    getargspec.stypy_type_store = module_type_store
    getargspec.stypy_function_name = 'getargspec'
    getargspec.stypy_param_names_list = ['func']
    getargspec.stypy_varargs_param_name = None
    getargspec.stypy_kwargs_param_name = None
    getargspec.stypy_call_defaults = defaults
    getargspec.stypy_call_varargs = varargs
    getargspec.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getargspec', ['func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getargspec', localization, ['func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getargspec(...)' code ##################

    str_25967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, (-1)), 'str', "Get the names and default values of a function's arguments.\n\n    A tuple of four things is returned: (args, varargs, varkw, defaults).\n    'args' is a list of the argument names (it may contain nested lists).\n    'varargs' and 'varkw' are the names of the * and ** arguments or None.\n    'defaults' is an n-tuple of the default values of the last n arguments.\n\n    ")
    
    
    # Call to ismethod(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'func' (line 108)
    func_25969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'func', False)
    # Processing the call keyword arguments (line 108)
    kwargs_25970 = {}
    # Getting the type of 'ismethod' (line 108)
    ismethod_25968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 7), 'ismethod', False)
    # Calling ismethod(args, kwargs) (line 108)
    ismethod_call_result_25971 = invoke(stypy.reporting.localization.Localization(__file__, 108, 7), ismethod_25968, *[func_25969], **kwargs_25970)
    
    # Testing the type of an if condition (line 108)
    if_condition_25972 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 4), ismethod_call_result_25971)
    # Assigning a type to the variable 'if_condition_25972' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'if_condition_25972', if_condition_25972)
    # SSA begins for if statement (line 108)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 109):
    
    # Assigning a Attribute to a Name (line 109):
    # Getting the type of 'func' (line 109)
    func_25973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 15), 'func')
    # Obtaining the member '__func__' of a type (line 109)
    func___25974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 15), func_25973, '__func__')
    # Assigning a type to the variable 'func' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'func', func___25974)
    # SSA join for if statement (line 108)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to isfunction(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'func' (line 110)
    func_25976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'func', False)
    # Processing the call keyword arguments (line 110)
    kwargs_25977 = {}
    # Getting the type of 'isfunction' (line 110)
    isfunction_25975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'isfunction', False)
    # Calling isfunction(args, kwargs) (line 110)
    isfunction_call_result_25978 = invoke(stypy.reporting.localization.Localization(__file__, 110, 11), isfunction_25975, *[func_25976], **kwargs_25977)
    
    # Applying the 'not' unary operator (line 110)
    result_not__25979 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 7), 'not', isfunction_call_result_25978)
    
    # Testing the type of an if condition (line 110)
    if_condition_25980 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 4), result_not__25979)
    # Assigning a type to the variable 'if_condition_25980' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'if_condition_25980', if_condition_25980)
    # SSA begins for if statement (line 110)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 111)
    # Processing the call arguments (line 111)
    str_25982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 24), 'str', 'arg is not a Python function')
    # Processing the call keyword arguments (line 111)
    kwargs_25983 = {}
    # Getting the type of 'TypeError' (line 111)
    TypeError_25981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 111)
    TypeError_call_result_25984 = invoke(stypy.reporting.localization.Localization(__file__, 111, 14), TypeError_25981, *[str_25982], **kwargs_25983)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 111, 8), TypeError_call_result_25984, 'raise parameter', BaseException)
    # SSA join for if statement (line 110)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 112):
    
    # Assigning a Call to a Name:
    
    # Call to getargs(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'func' (line 112)
    func_25986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 35), 'func', False)
    # Obtaining the member '__code__' of a type (line 112)
    code___25987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 35), func_25986, '__code__')
    # Processing the call keyword arguments (line 112)
    kwargs_25988 = {}
    # Getting the type of 'getargs' (line 112)
    getargs_25985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 27), 'getargs', False)
    # Calling getargs(args, kwargs) (line 112)
    getargs_call_result_25989 = invoke(stypy.reporting.localization.Localization(__file__, 112, 27), getargs_25985, *[code___25987], **kwargs_25988)
    
    # Assigning a type to the variable 'call_assignment_25848' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'call_assignment_25848', getargs_call_result_25989)
    
    # Assigning a Call to a Name (line 112):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_25992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 4), 'int')
    # Processing the call keyword arguments
    kwargs_25993 = {}
    # Getting the type of 'call_assignment_25848' (line 112)
    call_assignment_25848_25990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'call_assignment_25848', False)
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___25991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 4), call_assignment_25848_25990, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_25994 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___25991, *[int_25992], **kwargs_25993)
    
    # Assigning a type to the variable 'call_assignment_25849' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'call_assignment_25849', getitem___call_result_25994)
    
    # Assigning a Name to a Name (line 112):
    # Getting the type of 'call_assignment_25849' (line 112)
    call_assignment_25849_25995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'call_assignment_25849')
    # Assigning a type to the variable 'args' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'args', call_assignment_25849_25995)
    
    # Assigning a Call to a Name (line 112):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_25998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 4), 'int')
    # Processing the call keyword arguments
    kwargs_25999 = {}
    # Getting the type of 'call_assignment_25848' (line 112)
    call_assignment_25848_25996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'call_assignment_25848', False)
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___25997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 4), call_assignment_25848_25996, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_26000 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___25997, *[int_25998], **kwargs_25999)
    
    # Assigning a type to the variable 'call_assignment_25850' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'call_assignment_25850', getitem___call_result_26000)
    
    # Assigning a Name to a Name (line 112):
    # Getting the type of 'call_assignment_25850' (line 112)
    call_assignment_25850_26001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'call_assignment_25850')
    # Assigning a type to the variable 'varargs' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 10), 'varargs', call_assignment_25850_26001)
    
    # Assigning a Call to a Name (line 112):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_26004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 4), 'int')
    # Processing the call keyword arguments
    kwargs_26005 = {}
    # Getting the type of 'call_assignment_25848' (line 112)
    call_assignment_25848_26002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'call_assignment_25848', False)
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___26003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 4), call_assignment_25848_26002, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_26006 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___26003, *[int_26004], **kwargs_26005)
    
    # Assigning a type to the variable 'call_assignment_25851' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'call_assignment_25851', getitem___call_result_26006)
    
    # Assigning a Name to a Name (line 112):
    # Getting the type of 'call_assignment_25851' (line 112)
    call_assignment_25851_26007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'call_assignment_25851')
    # Assigning a type to the variable 'varkw' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'varkw', call_assignment_25851_26007)
    
    # Obtaining an instance of the builtin type 'tuple' (line 113)
    tuple_26008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 113)
    # Adding element type (line 113)
    # Getting the type of 'args' (line 113)
    args_26009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 11), 'args')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 11), tuple_26008, args_26009)
    # Adding element type (line 113)
    # Getting the type of 'varargs' (line 113)
    varargs_26010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 17), 'varargs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 11), tuple_26008, varargs_26010)
    # Adding element type (line 113)
    # Getting the type of 'varkw' (line 113)
    varkw_26011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 26), 'varkw')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 11), tuple_26008, varkw_26011)
    # Adding element type (line 113)
    # Getting the type of 'func' (line 113)
    func_26012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 33), 'func')
    # Obtaining the member '__defaults__' of a type (line 113)
    defaults___26013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 33), func_26012, '__defaults__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 11), tuple_26008, defaults___26013)
    
    # Assigning a type to the variable 'stypy_return_type' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'stypy_return_type', tuple_26008)
    
    # ################# End of 'getargspec(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getargspec' in the type store
    # Getting the type of 'stypy_return_type' (line 98)
    stypy_return_type_26014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_26014)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getargspec'
    return stypy_return_type_26014

# Assigning a type to the variable 'getargspec' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'getargspec', getargspec)

@norecursion
def getargvalues(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getargvalues'
    module_type_store = module_type_store.open_function_context('getargvalues', 115, 0, False)
    
    # Passed parameters checking function
    getargvalues.stypy_localization = localization
    getargvalues.stypy_type_of_self = None
    getargvalues.stypy_type_store = module_type_store
    getargvalues.stypy_function_name = 'getargvalues'
    getargvalues.stypy_param_names_list = ['frame']
    getargvalues.stypy_varargs_param_name = None
    getargvalues.stypy_kwargs_param_name = None
    getargvalues.stypy_call_defaults = defaults
    getargvalues.stypy_call_varargs = varargs
    getargvalues.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getargvalues', ['frame'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getargvalues', localization, ['frame'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getargvalues(...)' code ##################

    str_26015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, (-1)), 'str', "Get information about arguments passed into a particular frame.\n\n    A tuple of four things is returned: (args, varargs, varkw, locals).\n    'args' is a list of the argument names (it may contain nested lists).\n    'varargs' and 'varkw' are the names of the * and ** arguments or None.\n    'locals' is the locals dictionary of the given frame.\n    \n    ")
    
    # Assigning a Call to a Tuple (line 124):
    
    # Assigning a Call to a Name:
    
    # Call to getargs(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'frame' (line 124)
    frame_26017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 35), 'frame', False)
    # Obtaining the member 'f_code' of a type (line 124)
    f_code_26018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 35), frame_26017, 'f_code')
    # Processing the call keyword arguments (line 124)
    kwargs_26019 = {}
    # Getting the type of 'getargs' (line 124)
    getargs_26016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'getargs', False)
    # Calling getargs(args, kwargs) (line 124)
    getargs_call_result_26020 = invoke(stypy.reporting.localization.Localization(__file__, 124, 27), getargs_26016, *[f_code_26018], **kwargs_26019)
    
    # Assigning a type to the variable 'call_assignment_25852' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'call_assignment_25852', getargs_call_result_26020)
    
    # Assigning a Call to a Name (line 124):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_26023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 4), 'int')
    # Processing the call keyword arguments
    kwargs_26024 = {}
    # Getting the type of 'call_assignment_25852' (line 124)
    call_assignment_25852_26021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'call_assignment_25852', False)
    # Obtaining the member '__getitem__' of a type (line 124)
    getitem___26022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 4), call_assignment_25852_26021, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_26025 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___26022, *[int_26023], **kwargs_26024)
    
    # Assigning a type to the variable 'call_assignment_25853' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'call_assignment_25853', getitem___call_result_26025)
    
    # Assigning a Name to a Name (line 124):
    # Getting the type of 'call_assignment_25853' (line 124)
    call_assignment_25853_26026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'call_assignment_25853')
    # Assigning a type to the variable 'args' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'args', call_assignment_25853_26026)
    
    # Assigning a Call to a Name (line 124):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_26029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 4), 'int')
    # Processing the call keyword arguments
    kwargs_26030 = {}
    # Getting the type of 'call_assignment_25852' (line 124)
    call_assignment_25852_26027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'call_assignment_25852', False)
    # Obtaining the member '__getitem__' of a type (line 124)
    getitem___26028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 4), call_assignment_25852_26027, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_26031 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___26028, *[int_26029], **kwargs_26030)
    
    # Assigning a type to the variable 'call_assignment_25854' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'call_assignment_25854', getitem___call_result_26031)
    
    # Assigning a Name to a Name (line 124):
    # Getting the type of 'call_assignment_25854' (line 124)
    call_assignment_25854_26032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'call_assignment_25854')
    # Assigning a type to the variable 'varargs' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 10), 'varargs', call_assignment_25854_26032)
    
    # Assigning a Call to a Name (line 124):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_26035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 4), 'int')
    # Processing the call keyword arguments
    kwargs_26036 = {}
    # Getting the type of 'call_assignment_25852' (line 124)
    call_assignment_25852_26033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'call_assignment_25852', False)
    # Obtaining the member '__getitem__' of a type (line 124)
    getitem___26034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 4), call_assignment_25852_26033, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_26037 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___26034, *[int_26035], **kwargs_26036)
    
    # Assigning a type to the variable 'call_assignment_25855' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'call_assignment_25855', getitem___call_result_26037)
    
    # Assigning a Name to a Name (line 124):
    # Getting the type of 'call_assignment_25855' (line 124)
    call_assignment_25855_26038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'call_assignment_25855')
    # Assigning a type to the variable 'varkw' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'varkw', call_assignment_25855_26038)
    
    # Obtaining an instance of the builtin type 'tuple' (line 125)
    tuple_26039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 125)
    # Adding element type (line 125)
    # Getting the type of 'args' (line 125)
    args_26040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), 'args')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 11), tuple_26039, args_26040)
    # Adding element type (line 125)
    # Getting the type of 'varargs' (line 125)
    varargs_26041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 17), 'varargs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 11), tuple_26039, varargs_26041)
    # Adding element type (line 125)
    # Getting the type of 'varkw' (line 125)
    varkw_26042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 26), 'varkw')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 11), tuple_26039, varkw_26042)
    # Adding element type (line 125)
    # Getting the type of 'frame' (line 125)
    frame_26043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 33), 'frame')
    # Obtaining the member 'f_locals' of a type (line 125)
    f_locals_26044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 33), frame_26043, 'f_locals')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 11), tuple_26039, f_locals_26044)
    
    # Assigning a type to the variable 'stypy_return_type' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type', tuple_26039)
    
    # ################# End of 'getargvalues(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getargvalues' in the type store
    # Getting the type of 'stypy_return_type' (line 115)
    stypy_return_type_26045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_26045)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getargvalues'
    return stypy_return_type_26045

# Assigning a type to the variable 'getargvalues' (line 115)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'getargvalues', getargvalues)

@norecursion
def joinseq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'joinseq'
    module_type_store = module_type_store.open_function_context('joinseq', 127, 0, False)
    
    # Passed parameters checking function
    joinseq.stypy_localization = localization
    joinseq.stypy_type_of_self = None
    joinseq.stypy_type_store = module_type_store
    joinseq.stypy_function_name = 'joinseq'
    joinseq.stypy_param_names_list = ['seq']
    joinseq.stypy_varargs_param_name = None
    joinseq.stypy_kwargs_param_name = None
    joinseq.stypy_call_defaults = defaults
    joinseq.stypy_call_varargs = varargs
    joinseq.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'joinseq', ['seq'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'joinseq', localization, ['seq'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'joinseq(...)' code ##################

    
    
    
    # Call to len(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'seq' (line 128)
    seq_26047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'seq', False)
    # Processing the call keyword arguments (line 128)
    kwargs_26048 = {}
    # Getting the type of 'len' (line 128)
    len_26046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 7), 'len', False)
    # Calling len(args, kwargs) (line 128)
    len_call_result_26049 = invoke(stypy.reporting.localization.Localization(__file__, 128, 7), len_26046, *[seq_26047], **kwargs_26048)
    
    int_26050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 19), 'int')
    # Applying the binary operator '==' (line 128)
    result_eq_26051 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 7), '==', len_call_result_26049, int_26050)
    
    # Testing the type of an if condition (line 128)
    if_condition_26052 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 128, 4), result_eq_26051)
    # Assigning a type to the variable 'if_condition_26052' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'if_condition_26052', if_condition_26052)
    # SSA begins for if statement (line 128)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_26053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 15), 'str', '(')
    
    # Obtaining the type of the subscript
    int_26054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 25), 'int')
    # Getting the type of 'seq' (line 129)
    seq_26055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'seq')
    # Obtaining the member '__getitem__' of a type (line 129)
    getitem___26056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 21), seq_26055, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 129)
    subscript_call_result_26057 = invoke(stypy.reporting.localization.Localization(__file__, 129, 21), getitem___26056, int_26054)
    
    # Applying the binary operator '+' (line 129)
    result_add_26058 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 15), '+', str_26053, subscript_call_result_26057)
    
    str_26059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 30), 'str', ',)')
    # Applying the binary operator '+' (line 129)
    result_add_26060 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 28), '+', result_add_26058, str_26059)
    
    # Assigning a type to the variable 'stypy_return_type' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'stypy_return_type', result_add_26060)
    # SSA branch for the else part of an if statement (line 128)
    module_type_store.open_ssa_branch('else')
    str_26061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 15), 'str', '(')
    
    # Call to join(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'seq' (line 131)
    seq_26064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 31), 'seq', False)
    # Processing the call keyword arguments (line 131)
    kwargs_26065 = {}
    str_26062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 21), 'str', ', ')
    # Obtaining the member 'join' of a type (line 131)
    join_26063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 21), str_26062, 'join')
    # Calling join(args, kwargs) (line 131)
    join_call_result_26066 = invoke(stypy.reporting.localization.Localization(__file__, 131, 21), join_26063, *[seq_26064], **kwargs_26065)
    
    # Applying the binary operator '+' (line 131)
    result_add_26067 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 15), '+', str_26061, join_call_result_26066)
    
    str_26068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 38), 'str', ')')
    # Applying the binary operator '+' (line 131)
    result_add_26069 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 36), '+', result_add_26067, str_26068)
    
    # Assigning a type to the variable 'stypy_return_type' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'stypy_return_type', result_add_26069)
    # SSA join for if statement (line 128)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'joinseq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'joinseq' in the type store
    # Getting the type of 'stypy_return_type' (line 127)
    stypy_return_type_26070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_26070)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'joinseq'
    return stypy_return_type_26070

# Assigning a type to the variable 'joinseq' (line 127)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'joinseq', joinseq)

@norecursion
def strseq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'joinseq' (line 133)
    joinseq_26071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 33), 'joinseq')
    defaults = [joinseq_26071]
    # Create a new context for function 'strseq'
    module_type_store = module_type_store.open_function_context('strseq', 133, 0, False)
    
    # Passed parameters checking function
    strseq.stypy_localization = localization
    strseq.stypy_type_of_self = None
    strseq.stypy_type_store = module_type_store
    strseq.stypy_function_name = 'strseq'
    strseq.stypy_param_names_list = ['object', 'convert', 'join']
    strseq.stypy_varargs_param_name = None
    strseq.stypy_kwargs_param_name = None
    strseq.stypy_call_defaults = defaults
    strseq.stypy_call_varargs = varargs
    strseq.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'strseq', ['object', 'convert', 'join'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'strseq', localization, ['object', 'convert', 'join'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'strseq(...)' code ##################

    str_26072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, (-1)), 'str', 'Recursively walk a sequence, stringifying each element.\n\n    ')
    
    
    
    # Call to type(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'object' (line 137)
    object_26074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'object', False)
    # Processing the call keyword arguments (line 137)
    kwargs_26075 = {}
    # Getting the type of 'type' (line 137)
    type_26073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 7), 'type', False)
    # Calling type(args, kwargs) (line 137)
    type_call_result_26076 = invoke(stypy.reporting.localization.Localization(__file__, 137, 7), type_26073, *[object_26074], **kwargs_26075)
    
    
    # Obtaining an instance of the builtin type 'list' (line 137)
    list_26077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 137)
    # Adding element type (line 137)
    # Getting the type of 'list' (line 137)
    list_26078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 24), 'list')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 23), list_26077, list_26078)
    # Adding element type (line 137)
    # Getting the type of 'tuple' (line 137)
    tuple_26079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 30), 'tuple')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 23), list_26077, tuple_26079)
    
    # Applying the binary operator 'in' (line 137)
    result_contains_26080 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 7), 'in', type_call_result_26076, list_26077)
    
    # Testing the type of an if condition (line 137)
    if_condition_26081 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 4), result_contains_26080)
    # Assigning a type to the variable 'if_condition_26081' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'if_condition_26081', if_condition_26081)
    # SSA begins for if statement (line 137)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to join(...): (line 138)
    # Processing the call arguments (line 138)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'object' (line 138)
    object_26089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 57), 'object', False)
    comprehension_26090 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 21), object_26089)
    # Assigning a type to the variable '_o' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 21), '_o', comprehension_26090)
    
    # Call to strseq(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of '_o' (line 138)
    _o_26084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 28), '_o', False)
    # Getting the type of 'convert' (line 138)
    convert_26085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 32), 'convert', False)
    # Getting the type of 'join' (line 138)
    join_26086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 41), 'join', False)
    # Processing the call keyword arguments (line 138)
    kwargs_26087 = {}
    # Getting the type of 'strseq' (line 138)
    strseq_26083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 21), 'strseq', False)
    # Calling strseq(args, kwargs) (line 138)
    strseq_call_result_26088 = invoke(stypy.reporting.localization.Localization(__file__, 138, 21), strseq_26083, *[_o_26084, convert_26085, join_26086], **kwargs_26087)
    
    list_26091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 21), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 21), list_26091, strseq_call_result_26088)
    # Processing the call keyword arguments (line 138)
    kwargs_26092 = {}
    # Getting the type of 'join' (line 138)
    join_26082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'join', False)
    # Calling join(args, kwargs) (line 138)
    join_call_result_26093 = invoke(stypy.reporting.localization.Localization(__file__, 138, 15), join_26082, *[list_26091], **kwargs_26092)
    
    # Assigning a type to the variable 'stypy_return_type' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'stypy_return_type', join_call_result_26093)
    # SSA branch for the else part of an if statement (line 137)
    module_type_store.open_ssa_branch('else')
    
    # Call to convert(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'object' (line 140)
    object_26095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 23), 'object', False)
    # Processing the call keyword arguments (line 140)
    kwargs_26096 = {}
    # Getting the type of 'convert' (line 140)
    convert_26094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 15), 'convert', False)
    # Calling convert(args, kwargs) (line 140)
    convert_call_result_26097 = invoke(stypy.reporting.localization.Localization(__file__, 140, 15), convert_26094, *[object_26095], **kwargs_26096)
    
    # Assigning a type to the variable 'stypy_return_type' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'stypy_return_type', convert_call_result_26097)
    # SSA join for if statement (line 137)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'strseq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'strseq' in the type store
    # Getting the type of 'stypy_return_type' (line 133)
    stypy_return_type_26098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_26098)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'strseq'
    return stypy_return_type_26098

# Assigning a type to the variable 'strseq' (line 133)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'strseq', strseq)

@norecursion
def formatargspec(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 142)
    None_26099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 32), 'None')
    # Getting the type of 'None' (line 142)
    None_26100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 44), 'None')
    # Getting the type of 'None' (line 142)
    None_26101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 59), 'None')
    # Getting the type of 'str' (line 143)
    str_26102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 28), 'str')

    @norecursion
    def _stypy_temp_lambda_8(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_8'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_8', 144, 32, True)
        # Passed parameters checking function
        _stypy_temp_lambda_8.stypy_localization = localization
        _stypy_temp_lambda_8.stypy_type_of_self = None
        _stypy_temp_lambda_8.stypy_type_store = module_type_store
        _stypy_temp_lambda_8.stypy_function_name = '_stypy_temp_lambda_8'
        _stypy_temp_lambda_8.stypy_param_names_list = ['name']
        _stypy_temp_lambda_8.stypy_varargs_param_name = None
        _stypy_temp_lambda_8.stypy_kwargs_param_name = None
        _stypy_temp_lambda_8.stypy_call_defaults = defaults
        _stypy_temp_lambda_8.stypy_call_varargs = varargs
        _stypy_temp_lambda_8.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_8', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_8', ['name'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        str_26103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 45), 'str', '*')
        # Getting the type of 'name' (line 144)
        name_26104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 51), 'name')
        # Applying the binary operator '+' (line 144)
        result_add_26105 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 45), '+', str_26103, name_26104)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 32), 'stypy_return_type', result_add_26105)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_8' in the type store
        # Getting the type of 'stypy_return_type' (line 144)
        stypy_return_type_26106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 32), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26106)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_8'
        return stypy_return_type_26106

    # Assigning a type to the variable '_stypy_temp_lambda_8' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 32), '_stypy_temp_lambda_8', _stypy_temp_lambda_8)
    # Getting the type of '_stypy_temp_lambda_8' (line 144)
    _stypy_temp_lambda_8_26107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 32), '_stypy_temp_lambda_8')

    @norecursion
    def _stypy_temp_lambda_9(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_9'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_9', 145, 30, True)
        # Passed parameters checking function
        _stypy_temp_lambda_9.stypy_localization = localization
        _stypy_temp_lambda_9.stypy_type_of_self = None
        _stypy_temp_lambda_9.stypy_type_store = module_type_store
        _stypy_temp_lambda_9.stypy_function_name = '_stypy_temp_lambda_9'
        _stypy_temp_lambda_9.stypy_param_names_list = ['name']
        _stypy_temp_lambda_9.stypy_varargs_param_name = None
        _stypy_temp_lambda_9.stypy_kwargs_param_name = None
        _stypy_temp_lambda_9.stypy_call_defaults = defaults
        _stypy_temp_lambda_9.stypy_call_varargs = varargs
        _stypy_temp_lambda_9.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_9', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_9', ['name'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        str_26108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 43), 'str', '**')
        # Getting the type of 'name' (line 145)
        name_26109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 50), 'name')
        # Applying the binary operator '+' (line 145)
        result_add_26110 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 43), '+', str_26108, name_26109)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 30), 'stypy_return_type', result_add_26110)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_9' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_26111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 30), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26111)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_9'
        return stypy_return_type_26111

    # Assigning a type to the variable '_stypy_temp_lambda_9' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 30), '_stypy_temp_lambda_9', _stypy_temp_lambda_9)
    # Getting the type of '_stypy_temp_lambda_9' (line 145)
    _stypy_temp_lambda_9_26112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 30), '_stypy_temp_lambda_9')

    @norecursion
    def _stypy_temp_lambda_10(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_10'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_10', 146, 30, True)
        # Passed parameters checking function
        _stypy_temp_lambda_10.stypy_localization = localization
        _stypy_temp_lambda_10.stypy_type_of_self = None
        _stypy_temp_lambda_10.stypy_type_store = module_type_store
        _stypy_temp_lambda_10.stypy_function_name = '_stypy_temp_lambda_10'
        _stypy_temp_lambda_10.stypy_param_names_list = ['value']
        _stypy_temp_lambda_10.stypy_varargs_param_name = None
        _stypy_temp_lambda_10.stypy_kwargs_param_name = None
        _stypy_temp_lambda_10.stypy_call_defaults = defaults
        _stypy_temp_lambda_10.stypy_call_varargs = varargs
        _stypy_temp_lambda_10.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_10', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_10', ['value'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        str_26113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 44), 'str', '=')
        
        # Call to repr(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'value' (line 146)
        value_26115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 55), 'value', False)
        # Processing the call keyword arguments (line 146)
        kwargs_26116 = {}
        # Getting the type of 'repr' (line 146)
        repr_26114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 50), 'repr', False)
        # Calling repr(args, kwargs) (line 146)
        repr_call_result_26117 = invoke(stypy.reporting.localization.Localization(__file__, 146, 50), repr_26114, *[value_26115], **kwargs_26116)
        
        # Applying the binary operator '+' (line 146)
        result_add_26118 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 44), '+', str_26113, repr_call_result_26117)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 30), 'stypy_return_type', result_add_26118)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_10' in the type store
        # Getting the type of 'stypy_return_type' (line 146)
        stypy_return_type_26119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 30), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26119)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_10'
        return stypy_return_type_26119

    # Assigning a type to the variable '_stypy_temp_lambda_10' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 30), '_stypy_temp_lambda_10', _stypy_temp_lambda_10)
    # Getting the type of '_stypy_temp_lambda_10' (line 146)
    _stypy_temp_lambda_10_26120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 30), '_stypy_temp_lambda_10')
    # Getting the type of 'joinseq' (line 147)
    joinseq_26121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 23), 'joinseq')
    defaults = [None_26099, None_26100, None_26101, str_26102, _stypy_temp_lambda_8_26107, _stypy_temp_lambda_9_26112, _stypy_temp_lambda_10_26120, joinseq_26121]
    # Create a new context for function 'formatargspec'
    module_type_store = module_type_store.open_function_context('formatargspec', 142, 0, False)
    
    # Passed parameters checking function
    formatargspec.stypy_localization = localization
    formatargspec.stypy_type_of_self = None
    formatargspec.stypy_type_store = module_type_store
    formatargspec.stypy_function_name = 'formatargspec'
    formatargspec.stypy_param_names_list = ['args', 'varargs', 'varkw', 'defaults', 'formatarg', 'formatvarargs', 'formatvarkw', 'formatvalue', 'join']
    formatargspec.stypy_varargs_param_name = None
    formatargspec.stypy_kwargs_param_name = None
    formatargspec.stypy_call_defaults = defaults
    formatargspec.stypy_call_varargs = varargs
    formatargspec.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'formatargspec', ['args', 'varargs', 'varkw', 'defaults', 'formatarg', 'formatvarargs', 'formatvarkw', 'formatvalue', 'join'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'formatargspec', localization, ['args', 'varargs', 'varkw', 'defaults', 'formatarg', 'formatvarargs', 'formatvarkw', 'formatvalue', 'join'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'formatargspec(...)' code ##################

    str_26122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, (-1)), 'str', 'Format an argument spec from the 4 values returned by getargspec.\n\n    The first four arguments are (args, varargs, varkw, defaults).  The\n    other four arguments are the corresponding optional formatting functions\n    that are called to turn names and values into strings.  The ninth\n    argument is an optional function to format the sequence of arguments.\n\n    ')
    
    # Assigning a List to a Name (line 156):
    
    # Assigning a List to a Name (line 156):
    
    # Obtaining an instance of the builtin type 'list' (line 156)
    list_26123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 156)
    
    # Assigning a type to the variable 'specs' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'specs', list_26123)
    
    # Getting the type of 'defaults' (line 157)
    defaults_26124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 7), 'defaults')
    # Testing the type of an if condition (line 157)
    if_condition_26125 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 4), defaults_26124)
    # Assigning a type to the variable 'if_condition_26125' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'if_condition_26125', if_condition_26125)
    # SSA begins for if statement (line 157)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 158):
    
    # Assigning a BinOp to a Name (line 158):
    
    # Call to len(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'args' (line 158)
    args_26127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 27), 'args', False)
    # Processing the call keyword arguments (line 158)
    kwargs_26128 = {}
    # Getting the type of 'len' (line 158)
    len_26126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 23), 'len', False)
    # Calling len(args, kwargs) (line 158)
    len_call_result_26129 = invoke(stypy.reporting.localization.Localization(__file__, 158, 23), len_26126, *[args_26127], **kwargs_26128)
    
    
    # Call to len(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'defaults' (line 158)
    defaults_26131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 39), 'defaults', False)
    # Processing the call keyword arguments (line 158)
    kwargs_26132 = {}
    # Getting the type of 'len' (line 158)
    len_26130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 35), 'len', False)
    # Calling len(args, kwargs) (line 158)
    len_call_result_26133 = invoke(stypy.reporting.localization.Localization(__file__, 158, 35), len_26130, *[defaults_26131], **kwargs_26132)
    
    # Applying the binary operator '-' (line 158)
    result_sub_26134 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 23), '-', len_call_result_26129, len_call_result_26133)
    
    # Assigning a type to the variable 'firstdefault' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'firstdefault', result_sub_26134)
    # SSA join for if statement (line 157)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 159)
    # Processing the call arguments (line 159)
    
    # Call to len(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'args' (line 159)
    args_26137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 'args', False)
    # Processing the call keyword arguments (line 159)
    kwargs_26138 = {}
    # Getting the type of 'len' (line 159)
    len_26136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 19), 'len', False)
    # Calling len(args, kwargs) (line 159)
    len_call_result_26139 = invoke(stypy.reporting.localization.Localization(__file__, 159, 19), len_26136, *[args_26137], **kwargs_26138)
    
    # Processing the call keyword arguments (line 159)
    kwargs_26140 = {}
    # Getting the type of 'range' (line 159)
    range_26135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 13), 'range', False)
    # Calling range(args, kwargs) (line 159)
    range_call_result_26141 = invoke(stypy.reporting.localization.Localization(__file__, 159, 13), range_26135, *[len_call_result_26139], **kwargs_26140)
    
    # Testing the type of a for loop iterable (line 159)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 159, 4), range_call_result_26141)
    # Getting the type of the for loop variable (line 159)
    for_loop_var_26142 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 159, 4), range_call_result_26141)
    # Assigning a type to the variable 'i' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'i', for_loop_var_26142)
    # SSA begins for a for statement (line 159)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 160):
    
    # Assigning a Call to a Name (line 160):
    
    # Call to strseq(...): (line 160)
    # Processing the call arguments (line 160)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 160)
    i_26144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 27), 'i', False)
    # Getting the type of 'args' (line 160)
    args_26145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 22), 'args', False)
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___26146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 22), args_26145, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
    subscript_call_result_26147 = invoke(stypy.reporting.localization.Localization(__file__, 160, 22), getitem___26146, i_26144)
    
    # Getting the type of 'formatarg' (line 160)
    formatarg_26148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 31), 'formatarg', False)
    # Getting the type of 'join' (line 160)
    join_26149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 42), 'join', False)
    # Processing the call keyword arguments (line 160)
    kwargs_26150 = {}
    # Getting the type of 'strseq' (line 160)
    strseq_26143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'strseq', False)
    # Calling strseq(args, kwargs) (line 160)
    strseq_call_result_26151 = invoke(stypy.reporting.localization.Localization(__file__, 160, 15), strseq_26143, *[subscript_call_result_26147, formatarg_26148, join_26149], **kwargs_26150)
    
    # Assigning a type to the variable 'spec' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'spec', strseq_call_result_26151)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'defaults' (line 161)
    defaults_26152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 11), 'defaults')
    
    # Getting the type of 'i' (line 161)
    i_26153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 24), 'i')
    # Getting the type of 'firstdefault' (line 161)
    firstdefault_26154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 29), 'firstdefault')
    # Applying the binary operator '>=' (line 161)
    result_ge_26155 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 24), '>=', i_26153, firstdefault_26154)
    
    # Applying the binary operator 'and' (line 161)
    result_and_keyword_26156 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 11), 'and', defaults_26152, result_ge_26155)
    
    # Testing the type of an if condition (line 161)
    if_condition_26157 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 8), result_and_keyword_26156)
    # Assigning a type to the variable 'if_condition_26157' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'if_condition_26157', if_condition_26157)
    # SSA begins for if statement (line 161)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 162):
    
    # Assigning a BinOp to a Name (line 162):
    # Getting the type of 'spec' (line 162)
    spec_26158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), 'spec')
    
    # Call to formatvalue(...): (line 162)
    # Processing the call arguments (line 162)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 162)
    i_26160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 47), 'i', False)
    # Getting the type of 'firstdefault' (line 162)
    firstdefault_26161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 51), 'firstdefault', False)
    # Applying the binary operator '-' (line 162)
    result_sub_26162 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 47), '-', i_26160, firstdefault_26161)
    
    # Getting the type of 'defaults' (line 162)
    defaults_26163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 38), 'defaults', False)
    # Obtaining the member '__getitem__' of a type (line 162)
    getitem___26164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 38), defaults_26163, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 162)
    subscript_call_result_26165 = invoke(stypy.reporting.localization.Localization(__file__, 162, 38), getitem___26164, result_sub_26162)
    
    # Processing the call keyword arguments (line 162)
    kwargs_26166 = {}
    # Getting the type of 'formatvalue' (line 162)
    formatvalue_26159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 26), 'formatvalue', False)
    # Calling formatvalue(args, kwargs) (line 162)
    formatvalue_call_result_26167 = invoke(stypy.reporting.localization.Localization(__file__, 162, 26), formatvalue_26159, *[subscript_call_result_26165], **kwargs_26166)
    
    # Applying the binary operator '+' (line 162)
    result_add_26168 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 19), '+', spec_26158, formatvalue_call_result_26167)
    
    # Assigning a type to the variable 'spec' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'spec', result_add_26168)
    # SSA join for if statement (line 161)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'spec' (line 163)
    spec_26171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 21), 'spec', False)
    # Processing the call keyword arguments (line 163)
    kwargs_26172 = {}
    # Getting the type of 'specs' (line 163)
    specs_26169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'specs', False)
    # Obtaining the member 'append' of a type (line 163)
    append_26170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), specs_26169, 'append')
    # Calling append(args, kwargs) (line 163)
    append_call_result_26173 = invoke(stypy.reporting.localization.Localization(__file__, 163, 8), append_26170, *[spec_26171], **kwargs_26172)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 164)
    # Getting the type of 'varargs' (line 164)
    varargs_26174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'varargs')
    # Getting the type of 'None' (line 164)
    None_26175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 22), 'None')
    
    (may_be_26176, more_types_in_union_26177) = may_not_be_none(varargs_26174, None_26175)

    if may_be_26176:

        if more_types_in_union_26177:
            # Runtime conditional SSA (line 164)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to append(...): (line 165)
        # Processing the call arguments (line 165)
        
        # Call to formatvarargs(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'varargs' (line 165)
        varargs_26181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 35), 'varargs', False)
        # Processing the call keyword arguments (line 165)
        kwargs_26182 = {}
        # Getting the type of 'formatvarargs' (line 165)
        formatvarargs_26180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 21), 'formatvarargs', False)
        # Calling formatvarargs(args, kwargs) (line 165)
        formatvarargs_call_result_26183 = invoke(stypy.reporting.localization.Localization(__file__, 165, 21), formatvarargs_26180, *[varargs_26181], **kwargs_26182)
        
        # Processing the call keyword arguments (line 165)
        kwargs_26184 = {}
        # Getting the type of 'specs' (line 165)
        specs_26178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'specs', False)
        # Obtaining the member 'append' of a type (line 165)
        append_26179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), specs_26178, 'append')
        # Calling append(args, kwargs) (line 165)
        append_call_result_26185 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), append_26179, *[formatvarargs_call_result_26183], **kwargs_26184)
        

        if more_types_in_union_26177:
            # SSA join for if statement (line 164)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 166)
    # Getting the type of 'varkw' (line 166)
    varkw_26186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'varkw')
    # Getting the type of 'None' (line 166)
    None_26187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 20), 'None')
    
    (may_be_26188, more_types_in_union_26189) = may_not_be_none(varkw_26186, None_26187)

    if may_be_26188:

        if more_types_in_union_26189:
            # Runtime conditional SSA (line 166)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to append(...): (line 167)
        # Processing the call arguments (line 167)
        
        # Call to formatvarkw(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'varkw' (line 167)
        varkw_26193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 33), 'varkw', False)
        # Processing the call keyword arguments (line 167)
        kwargs_26194 = {}
        # Getting the type of 'formatvarkw' (line 167)
        formatvarkw_26192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'formatvarkw', False)
        # Calling formatvarkw(args, kwargs) (line 167)
        formatvarkw_call_result_26195 = invoke(stypy.reporting.localization.Localization(__file__, 167, 21), formatvarkw_26192, *[varkw_26193], **kwargs_26194)
        
        # Processing the call keyword arguments (line 167)
        kwargs_26196 = {}
        # Getting the type of 'specs' (line 167)
        specs_26190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'specs', False)
        # Obtaining the member 'append' of a type (line 167)
        append_26191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), specs_26190, 'append')
        # Calling append(args, kwargs) (line 167)
        append_call_result_26197 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), append_26191, *[formatvarkw_call_result_26195], **kwargs_26196)
        

        if more_types_in_union_26189:
            # SSA join for if statement (line 166)
            module_type_store = module_type_store.join_ssa_context()


    
    str_26198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 11), 'str', '(')
    
    # Call to join(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'specs' (line 168)
    specs_26201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'specs', False)
    # Processing the call keyword arguments (line 168)
    kwargs_26202 = {}
    str_26199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 17), 'str', ', ')
    # Obtaining the member 'join' of a type (line 168)
    join_26200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 17), str_26199, 'join')
    # Calling join(args, kwargs) (line 168)
    join_call_result_26203 = invoke(stypy.reporting.localization.Localization(__file__, 168, 17), join_26200, *[specs_26201], **kwargs_26202)
    
    # Applying the binary operator '+' (line 168)
    result_add_26204 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 11), '+', str_26198, join_call_result_26203)
    
    str_26205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 36), 'str', ')')
    # Applying the binary operator '+' (line 168)
    result_add_26206 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 34), '+', result_add_26204, str_26205)
    
    # Assigning a type to the variable 'stypy_return_type' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type', result_add_26206)
    
    # ################# End of 'formatargspec(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'formatargspec' in the type store
    # Getting the type of 'stypy_return_type' (line 142)
    stypy_return_type_26207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_26207)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'formatargspec'
    return stypy_return_type_26207

# Assigning a type to the variable 'formatargspec' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'formatargspec', formatargspec)

@norecursion
def formatargvalues(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'str' (line 171)
    str_26208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 30), 'str')

    @norecursion
    def _stypy_temp_lambda_11(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_11'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_11', 172, 34, True)
        # Passed parameters checking function
        _stypy_temp_lambda_11.stypy_localization = localization
        _stypy_temp_lambda_11.stypy_type_of_self = None
        _stypy_temp_lambda_11.stypy_type_store = module_type_store
        _stypy_temp_lambda_11.stypy_function_name = '_stypy_temp_lambda_11'
        _stypy_temp_lambda_11.stypy_param_names_list = ['name']
        _stypy_temp_lambda_11.stypy_varargs_param_name = None
        _stypy_temp_lambda_11.stypy_kwargs_param_name = None
        _stypy_temp_lambda_11.stypy_call_defaults = defaults
        _stypy_temp_lambda_11.stypy_call_varargs = varargs
        _stypy_temp_lambda_11.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_11', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_11', ['name'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        str_26209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 47), 'str', '*')
        # Getting the type of 'name' (line 172)
        name_26210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 53), 'name')
        # Applying the binary operator '+' (line 172)
        result_add_26211 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 47), '+', str_26209, name_26210)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 34), 'stypy_return_type', result_add_26211)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_11' in the type store
        # Getting the type of 'stypy_return_type' (line 172)
        stypy_return_type_26212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 34), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26212)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_11'
        return stypy_return_type_26212

    # Assigning a type to the variable '_stypy_temp_lambda_11' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 34), '_stypy_temp_lambda_11', _stypy_temp_lambda_11)
    # Getting the type of '_stypy_temp_lambda_11' (line 172)
    _stypy_temp_lambda_11_26213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 34), '_stypy_temp_lambda_11')

    @norecursion
    def _stypy_temp_lambda_12(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_12'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_12', 173, 32, True)
        # Passed parameters checking function
        _stypy_temp_lambda_12.stypy_localization = localization
        _stypy_temp_lambda_12.stypy_type_of_self = None
        _stypy_temp_lambda_12.stypy_type_store = module_type_store
        _stypy_temp_lambda_12.stypy_function_name = '_stypy_temp_lambda_12'
        _stypy_temp_lambda_12.stypy_param_names_list = ['name']
        _stypy_temp_lambda_12.stypy_varargs_param_name = None
        _stypy_temp_lambda_12.stypy_kwargs_param_name = None
        _stypy_temp_lambda_12.stypy_call_defaults = defaults
        _stypy_temp_lambda_12.stypy_call_varargs = varargs
        _stypy_temp_lambda_12.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_12', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_12', ['name'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        str_26214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 45), 'str', '**')
        # Getting the type of 'name' (line 173)
        name_26215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 52), 'name')
        # Applying the binary operator '+' (line 173)
        result_add_26216 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 45), '+', str_26214, name_26215)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 32), 'stypy_return_type', result_add_26216)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_12' in the type store
        # Getting the type of 'stypy_return_type' (line 173)
        stypy_return_type_26217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 32), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26217)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_12'
        return stypy_return_type_26217

    # Assigning a type to the variable '_stypy_temp_lambda_12' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 32), '_stypy_temp_lambda_12', _stypy_temp_lambda_12)
    # Getting the type of '_stypy_temp_lambda_12' (line 173)
    _stypy_temp_lambda_12_26218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 32), '_stypy_temp_lambda_12')

    @norecursion
    def _stypy_temp_lambda_13(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_13'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_13', 174, 32, True)
        # Passed parameters checking function
        _stypy_temp_lambda_13.stypy_localization = localization
        _stypy_temp_lambda_13.stypy_type_of_self = None
        _stypy_temp_lambda_13.stypy_type_store = module_type_store
        _stypy_temp_lambda_13.stypy_function_name = '_stypy_temp_lambda_13'
        _stypy_temp_lambda_13.stypy_param_names_list = ['value']
        _stypy_temp_lambda_13.stypy_varargs_param_name = None
        _stypy_temp_lambda_13.stypy_kwargs_param_name = None
        _stypy_temp_lambda_13.stypy_call_defaults = defaults
        _stypy_temp_lambda_13.stypy_call_varargs = varargs
        _stypy_temp_lambda_13.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_13', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_13', ['value'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        str_26219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 46), 'str', '=')
        
        # Call to repr(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'value' (line 174)
        value_26221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 57), 'value', False)
        # Processing the call keyword arguments (line 174)
        kwargs_26222 = {}
        # Getting the type of 'repr' (line 174)
        repr_26220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 52), 'repr', False)
        # Calling repr(args, kwargs) (line 174)
        repr_call_result_26223 = invoke(stypy.reporting.localization.Localization(__file__, 174, 52), repr_26220, *[value_26221], **kwargs_26222)
        
        # Applying the binary operator '+' (line 174)
        result_add_26224 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 46), '+', str_26219, repr_call_result_26223)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 32), 'stypy_return_type', result_add_26224)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_13' in the type store
        # Getting the type of 'stypy_return_type' (line 174)
        stypy_return_type_26225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 32), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26225)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_13'
        return stypy_return_type_26225

    # Assigning a type to the variable '_stypy_temp_lambda_13' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 32), '_stypy_temp_lambda_13', _stypy_temp_lambda_13)
    # Getting the type of '_stypy_temp_lambda_13' (line 174)
    _stypy_temp_lambda_13_26226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 32), '_stypy_temp_lambda_13')
    # Getting the type of 'joinseq' (line 175)
    joinseq_26227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 'joinseq')
    defaults = [str_26208, _stypy_temp_lambda_11_26213, _stypy_temp_lambda_12_26218, _stypy_temp_lambda_13_26226, joinseq_26227]
    # Create a new context for function 'formatargvalues'
    module_type_store = module_type_store.open_function_context('formatargvalues', 170, 0, False)
    
    # Passed parameters checking function
    formatargvalues.stypy_localization = localization
    formatargvalues.stypy_type_of_self = None
    formatargvalues.stypy_type_store = module_type_store
    formatargvalues.stypy_function_name = 'formatargvalues'
    formatargvalues.stypy_param_names_list = ['args', 'varargs', 'varkw', 'locals', 'formatarg', 'formatvarargs', 'formatvarkw', 'formatvalue', 'join']
    formatargvalues.stypy_varargs_param_name = None
    formatargvalues.stypy_kwargs_param_name = None
    formatargvalues.stypy_call_defaults = defaults
    formatargvalues.stypy_call_varargs = varargs
    formatargvalues.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'formatargvalues', ['args', 'varargs', 'varkw', 'locals', 'formatarg', 'formatvarargs', 'formatvarkw', 'formatvalue', 'join'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'formatargvalues', localization, ['args', 'varargs', 'varkw', 'locals', 'formatarg', 'formatvarargs', 'formatvarkw', 'formatvalue', 'join'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'formatargvalues(...)' code ##################

    str_26228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, (-1)), 'str', 'Format an argument spec from the 4 values returned by getargvalues.\n\n    The first four arguments are (args, varargs, varkw, locals).  The\n    next four arguments are the corresponding optional formatting functions\n    that are called to turn names and values into strings.  The ninth\n    argument is an optional function to format the sequence of arguments.\n\n    ')

    @norecursion
    def convert(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'locals' (line 184)
        locals_26229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 29), 'locals')
        # Getting the type of 'formatarg' (line 185)
        formatarg_26230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 26), 'formatarg')
        # Getting the type of 'formatvalue' (line 185)
        formatvalue_26231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 49), 'formatvalue')
        defaults = [locals_26229, formatarg_26230, formatvalue_26231]
        # Create a new context for function 'convert'
        module_type_store = module_type_store.open_function_context('convert', 184, 4, False)
        
        # Passed parameters checking function
        convert.stypy_localization = localization
        convert.stypy_type_of_self = None
        convert.stypy_type_store = module_type_store
        convert.stypy_function_name = 'convert'
        convert.stypy_param_names_list = ['name', 'locals', 'formatarg', 'formatvalue']
        convert.stypy_varargs_param_name = None
        convert.stypy_kwargs_param_name = None
        convert.stypy_call_defaults = defaults
        convert.stypy_call_varargs = varargs
        convert.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'convert', ['name', 'locals', 'formatarg', 'formatvalue'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'convert', localization, ['name', 'locals', 'formatarg', 'formatvalue'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'convert(...)' code ##################

        
        # Call to formatarg(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'name' (line 186)
        name_26233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), 'name', False)
        # Processing the call keyword arguments (line 186)
        kwargs_26234 = {}
        # Getting the type of 'formatarg' (line 186)
        formatarg_26232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 15), 'formatarg', False)
        # Calling formatarg(args, kwargs) (line 186)
        formatarg_call_result_26235 = invoke(stypy.reporting.localization.Localization(__file__, 186, 15), formatarg_26232, *[name_26233], **kwargs_26234)
        
        
        # Call to formatvalue(...): (line 186)
        # Processing the call arguments (line 186)
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 186)
        name_26237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 52), 'name', False)
        # Getting the type of 'locals' (line 186)
        locals_26238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 45), 'locals', False)
        # Obtaining the member '__getitem__' of a type (line 186)
        getitem___26239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 45), locals_26238, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 186)
        subscript_call_result_26240 = invoke(stypy.reporting.localization.Localization(__file__, 186, 45), getitem___26239, name_26237)
        
        # Processing the call keyword arguments (line 186)
        kwargs_26241 = {}
        # Getting the type of 'formatvalue' (line 186)
        formatvalue_26236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 33), 'formatvalue', False)
        # Calling formatvalue(args, kwargs) (line 186)
        formatvalue_call_result_26242 = invoke(stypy.reporting.localization.Localization(__file__, 186, 33), formatvalue_26236, *[subscript_call_result_26240], **kwargs_26241)
        
        # Applying the binary operator '+' (line 186)
        result_add_26243 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 15), '+', formatarg_call_result_26235, formatvalue_call_result_26242)
        
        # Assigning a type to the variable 'stypy_return_type' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'stypy_return_type', result_add_26243)
        
        # ################# End of 'convert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'convert' in the type store
        # Getting the type of 'stypy_return_type' (line 184)
        stypy_return_type_26244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26244)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'convert'
        return stypy_return_type_26244

    # Assigning a type to the variable 'convert' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'convert', convert)
    
    # Assigning a List to a Name (line 187):
    
    # Assigning a List to a Name (line 187):
    
    # Obtaining an instance of the builtin type 'list' (line 187)
    list_26245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 187)
    
    # Assigning a type to the variable 'specs' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'specs', list_26245)
    
    
    # Call to range(...): (line 188)
    # Processing the call arguments (line 188)
    
    # Call to len(...): (line 188)
    # Processing the call arguments (line 188)
    # Getting the type of 'args' (line 188)
    args_26248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 23), 'args', False)
    # Processing the call keyword arguments (line 188)
    kwargs_26249 = {}
    # Getting the type of 'len' (line 188)
    len_26247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 19), 'len', False)
    # Calling len(args, kwargs) (line 188)
    len_call_result_26250 = invoke(stypy.reporting.localization.Localization(__file__, 188, 19), len_26247, *[args_26248], **kwargs_26249)
    
    # Processing the call keyword arguments (line 188)
    kwargs_26251 = {}
    # Getting the type of 'range' (line 188)
    range_26246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 13), 'range', False)
    # Calling range(args, kwargs) (line 188)
    range_call_result_26252 = invoke(stypy.reporting.localization.Localization(__file__, 188, 13), range_26246, *[len_call_result_26250], **kwargs_26251)
    
    # Testing the type of a for loop iterable (line 188)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 188, 4), range_call_result_26252)
    # Getting the type of the for loop variable (line 188)
    for_loop_var_26253 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 188, 4), range_call_result_26252)
    # Assigning a type to the variable 'i' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'i', for_loop_var_26253)
    # SSA begins for a for statement (line 188)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 189)
    # Processing the call arguments (line 189)
    
    # Call to strseq(...): (line 189)
    # Processing the call arguments (line 189)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 189)
    i_26257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 33), 'i', False)
    # Getting the type of 'args' (line 189)
    args_26258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 28), 'args', False)
    # Obtaining the member '__getitem__' of a type (line 189)
    getitem___26259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 28), args_26258, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 189)
    subscript_call_result_26260 = invoke(stypy.reporting.localization.Localization(__file__, 189, 28), getitem___26259, i_26257)
    
    # Getting the type of 'convert' (line 189)
    convert_26261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 37), 'convert', False)
    # Getting the type of 'join' (line 189)
    join_26262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 46), 'join', False)
    # Processing the call keyword arguments (line 189)
    kwargs_26263 = {}
    # Getting the type of 'strseq' (line 189)
    strseq_26256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 21), 'strseq', False)
    # Calling strseq(args, kwargs) (line 189)
    strseq_call_result_26264 = invoke(stypy.reporting.localization.Localization(__file__, 189, 21), strseq_26256, *[subscript_call_result_26260, convert_26261, join_26262], **kwargs_26263)
    
    # Processing the call keyword arguments (line 189)
    kwargs_26265 = {}
    # Getting the type of 'specs' (line 189)
    specs_26254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'specs', False)
    # Obtaining the member 'append' of a type (line 189)
    append_26255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), specs_26254, 'append')
    # Calling append(args, kwargs) (line 189)
    append_call_result_26266 = invoke(stypy.reporting.localization.Localization(__file__, 189, 8), append_26255, *[strseq_call_result_26264], **kwargs_26265)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'varargs' (line 190)
    varargs_26267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 7), 'varargs')
    # Testing the type of an if condition (line 190)
    if_condition_26268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 4), varargs_26267)
    # Assigning a type to the variable 'if_condition_26268' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'if_condition_26268', if_condition_26268)
    # SSA begins for if statement (line 190)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 191)
    # Processing the call arguments (line 191)
    
    # Call to formatvarargs(...): (line 191)
    # Processing the call arguments (line 191)
    # Getting the type of 'varargs' (line 191)
    varargs_26272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 35), 'varargs', False)
    # Processing the call keyword arguments (line 191)
    kwargs_26273 = {}
    # Getting the type of 'formatvarargs' (line 191)
    formatvarargs_26271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 21), 'formatvarargs', False)
    # Calling formatvarargs(args, kwargs) (line 191)
    formatvarargs_call_result_26274 = invoke(stypy.reporting.localization.Localization(__file__, 191, 21), formatvarargs_26271, *[varargs_26272], **kwargs_26273)
    
    
    # Call to formatvalue(...): (line 191)
    # Processing the call arguments (line 191)
    
    # Obtaining the type of the subscript
    # Getting the type of 'varargs' (line 191)
    varargs_26276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 65), 'varargs', False)
    # Getting the type of 'locals' (line 191)
    locals_26277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 58), 'locals', False)
    # Obtaining the member '__getitem__' of a type (line 191)
    getitem___26278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 58), locals_26277, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 191)
    subscript_call_result_26279 = invoke(stypy.reporting.localization.Localization(__file__, 191, 58), getitem___26278, varargs_26276)
    
    # Processing the call keyword arguments (line 191)
    kwargs_26280 = {}
    # Getting the type of 'formatvalue' (line 191)
    formatvalue_26275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 46), 'formatvalue', False)
    # Calling formatvalue(args, kwargs) (line 191)
    formatvalue_call_result_26281 = invoke(stypy.reporting.localization.Localization(__file__, 191, 46), formatvalue_26275, *[subscript_call_result_26279], **kwargs_26280)
    
    # Applying the binary operator '+' (line 191)
    result_add_26282 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 21), '+', formatvarargs_call_result_26274, formatvalue_call_result_26281)
    
    # Processing the call keyword arguments (line 191)
    kwargs_26283 = {}
    # Getting the type of 'specs' (line 191)
    specs_26269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'specs', False)
    # Obtaining the member 'append' of a type (line 191)
    append_26270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), specs_26269, 'append')
    # Calling append(args, kwargs) (line 191)
    append_call_result_26284 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), append_26270, *[result_add_26282], **kwargs_26283)
    
    # SSA join for if statement (line 190)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'varkw' (line 192)
    varkw_26285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 7), 'varkw')
    # Testing the type of an if condition (line 192)
    if_condition_26286 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 4), varkw_26285)
    # Assigning a type to the variable 'if_condition_26286' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'if_condition_26286', if_condition_26286)
    # SSA begins for if statement (line 192)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 193)
    # Processing the call arguments (line 193)
    
    # Call to formatvarkw(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'varkw' (line 193)
    varkw_26290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 33), 'varkw', False)
    # Processing the call keyword arguments (line 193)
    kwargs_26291 = {}
    # Getting the type of 'formatvarkw' (line 193)
    formatvarkw_26289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 21), 'formatvarkw', False)
    # Calling formatvarkw(args, kwargs) (line 193)
    formatvarkw_call_result_26292 = invoke(stypy.reporting.localization.Localization(__file__, 193, 21), formatvarkw_26289, *[varkw_26290], **kwargs_26291)
    
    
    # Call to formatvalue(...): (line 193)
    # Processing the call arguments (line 193)
    
    # Obtaining the type of the subscript
    # Getting the type of 'varkw' (line 193)
    varkw_26294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 61), 'varkw', False)
    # Getting the type of 'locals' (line 193)
    locals_26295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 54), 'locals', False)
    # Obtaining the member '__getitem__' of a type (line 193)
    getitem___26296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 54), locals_26295, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 193)
    subscript_call_result_26297 = invoke(stypy.reporting.localization.Localization(__file__, 193, 54), getitem___26296, varkw_26294)
    
    # Processing the call keyword arguments (line 193)
    kwargs_26298 = {}
    # Getting the type of 'formatvalue' (line 193)
    formatvalue_26293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 42), 'formatvalue', False)
    # Calling formatvalue(args, kwargs) (line 193)
    formatvalue_call_result_26299 = invoke(stypy.reporting.localization.Localization(__file__, 193, 42), formatvalue_26293, *[subscript_call_result_26297], **kwargs_26298)
    
    # Applying the binary operator '+' (line 193)
    result_add_26300 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 21), '+', formatvarkw_call_result_26292, formatvalue_call_result_26299)
    
    # Processing the call keyword arguments (line 193)
    kwargs_26301 = {}
    # Getting the type of 'specs' (line 193)
    specs_26287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'specs', False)
    # Obtaining the member 'append' of a type (line 193)
    append_26288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), specs_26287, 'append')
    # Calling append(args, kwargs) (line 193)
    append_call_result_26302 = invoke(stypy.reporting.localization.Localization(__file__, 193, 8), append_26288, *[result_add_26300], **kwargs_26301)
    
    # SSA join for if statement (line 192)
    module_type_store = module_type_store.join_ssa_context()
    
    str_26303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 11), 'str', '(')
    
    # Call to join(...): (line 194)
    # Processing the call arguments (line 194)
    # Getting the type of 'specs' (line 194)
    specs_26306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 27), 'specs', False)
    # Processing the call keyword arguments (line 194)
    kwargs_26307 = {}
    str_26304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 17), 'str', ', ')
    # Obtaining the member 'join' of a type (line 194)
    join_26305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 17), str_26304, 'join')
    # Calling join(args, kwargs) (line 194)
    join_call_result_26308 = invoke(stypy.reporting.localization.Localization(__file__, 194, 17), join_26305, *[specs_26306], **kwargs_26307)
    
    # Applying the binary operator '+' (line 194)
    result_add_26309 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 11), '+', str_26303, join_call_result_26308)
    
    str_26310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 36), 'str', ')')
    # Applying the binary operator '+' (line 194)
    result_add_26311 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 34), '+', result_add_26309, str_26310)
    
    # Assigning a type to the variable 'stypy_return_type' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'stypy_return_type', result_add_26311)
    
    # ################# End of 'formatargvalues(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'formatargvalues' in the type store
    # Getting the type of 'stypy_return_type' (line 170)
    stypy_return_type_26312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_26312)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'formatargvalues'
    return stypy_return_type_26312

# Assigning a type to the variable 'formatargvalues' (line 170)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'formatargvalues', formatargvalues)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
