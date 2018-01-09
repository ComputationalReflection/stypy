
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Manage figures for pyplot interface.
3: '''
4: from __future__ import (absolute_import, division, print_function,
5:                         unicode_literals)
6: 
7: import six
8: import sys
9: import gc
10: import atexit
11: 
12: 
13: def error_msg(msg):
14:     print(msg, file=sys.stderr)
15: 
16: 
17: class Gcf(object):
18:     '''
19:     Singleton to manage a set of integer-numbered figures.
20: 
21:     This class is never instantiated; it consists of two class
22:     attributes (a list and a dictionary), and a set of static
23:     methods that operate on those attributes, accessing them
24:     directly as class attributes.
25: 
26:     Attributes:
27: 
28:         *figs*:
29:           dictionary of the form {*num*: *manager*, ...}
30: 
31:         *_activeQue*:
32:           list of *managers*, with active one at the end
33: 
34:     '''
35:     _activeQue = []
36:     figs = {}
37: 
38:     @classmethod
39:     def get_fig_manager(cls, num):
40:         '''
41:         If figure manager *num* exists, make it the active
42:         figure and return the manager; otherwise return *None*.
43:         '''
44:         manager = cls.figs.get(num, None)
45:         if manager is not None:
46:             cls.set_active(manager)
47:         return manager
48: 
49:     @classmethod
50:     def destroy(cls, num):
51:         '''
52:         Try to remove all traces of figure *num*.
53: 
54:         In the interactive backends, this is bound to the
55:         window "destroy" and "delete" events.
56:         '''
57:         if not cls.has_fignum(num):
58:             return
59:         manager = cls.figs[num]
60:         manager.canvas.mpl_disconnect(manager._cidgcf)
61: 
62:         # There must be a good reason for the following careful
63:         # rebuilding of the activeQue; what is it?
64:         oldQue = cls._activeQue[:]
65:         cls._activeQue = []
66:         for f in oldQue:
67:             if f != manager:
68:                 cls._activeQue.append(f)
69: 
70:         del cls.figs[num]
71:         manager.destroy()
72:         gc.collect(1)
73: 
74:     @classmethod
75:     def destroy_fig(cls, fig):
76:         "*fig* is a Figure instance"
77:         num = None
78:         for manager in six.itervalues(cls.figs):
79:             if manager.canvas.figure == fig:
80:                 num = manager.num
81:                 break
82:         if num is not None:
83:             cls.destroy(num)
84: 
85:     @classmethod
86:     def destroy_all(cls):
87:         # this is need to ensure that gc is available in corner cases
88:         # where modules are being torn down after install with easy_install
89:         import gc  # noqa
90:         for manager in list(cls.figs.values()):
91:             manager.canvas.mpl_disconnect(manager._cidgcf)
92:             manager.destroy()
93: 
94:         cls._activeQue = []
95:         cls.figs.clear()
96:         gc.collect(1)
97: 
98:     @classmethod
99:     def has_fignum(cls, num):
100:         '''
101:         Return *True* if figure *num* exists.
102:         '''
103:         return num in cls.figs
104: 
105:     @classmethod
106:     def get_all_fig_managers(cls):
107:         '''
108:         Return a list of figure managers.
109:         '''
110:         return list(cls.figs.values())
111: 
112:     @classmethod
113:     def get_num_fig_managers(cls):
114:         '''
115:         Return the number of figures being managed.
116:         '''
117:         return len(cls.figs)
118: 
119:     @classmethod
120:     def get_active(cls):
121:         '''
122:         Return the manager of the active figure, or *None*.
123:         '''
124:         if len(cls._activeQue) == 0:
125:             return None
126:         else:
127:             return cls._activeQue[-1]
128: 
129:     @classmethod
130:     def set_active(cls, manager):
131:         '''
132:         Make the figure corresponding to *manager* the active one.
133:         '''
134:         oldQue = cls._activeQue[:]
135:         cls._activeQue = []
136:         for m in oldQue:
137:             if m != manager:
138:                 cls._activeQue.append(m)
139:         cls._activeQue.append(manager)
140:         cls.figs[manager.num] = manager
141: 
142:     @classmethod
143:     def draw_all(cls, force=False):
144:         '''
145:         Redraw all figures registered with the pyplot
146:         state machine.
147:         '''
148:         for f_mgr in cls.get_all_fig_managers():
149:             if force or f_mgr.canvas.figure.stale:
150:                 f_mgr.canvas.draw_idle()
151: 
152: atexit.register(Gcf.destroy_all)
153: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_187925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'unicode', u'\nManage figures for pyplot interface.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import six' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_187926 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six')

if (type(import_187926) is not StypyTypeError):

    if (import_187926 != 'pyd_module'):
        __import__(import_187926)
        sys_modules_187927 = sys.modules[import_187926]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', sys_modules_187927.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', import_187926)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import sys' statement (line 8)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import gc' statement (line 9)
import gc

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'gc', gc, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import atexit' statement (line 10)
import atexit

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'atexit', atexit, module_type_store)


@norecursion
def error_msg(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'error_msg'
    module_type_store = module_type_store.open_function_context('error_msg', 13, 0, False)
    
    # Passed parameters checking function
    error_msg.stypy_localization = localization
    error_msg.stypy_type_of_self = None
    error_msg.stypy_type_store = module_type_store
    error_msg.stypy_function_name = 'error_msg'
    error_msg.stypy_param_names_list = ['msg']
    error_msg.stypy_varargs_param_name = None
    error_msg.stypy_kwargs_param_name = None
    error_msg.stypy_call_defaults = defaults
    error_msg.stypy_call_varargs = varargs
    error_msg.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'error_msg', ['msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'error_msg', localization, ['msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'error_msg(...)' code ##################

    
    # Call to print(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'msg' (line 14)
    msg_187929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'msg', False)
    # Processing the call keyword arguments (line 14)
    # Getting the type of 'sys' (line 14)
    sys_187930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 20), 'sys', False)
    # Obtaining the member 'stderr' of a type (line 14)
    stderr_187931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 20), sys_187930, 'stderr')
    keyword_187932 = stderr_187931
    kwargs_187933 = {'file': keyword_187932}
    # Getting the type of 'print' (line 14)
    print_187928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'print', False)
    # Calling print(args, kwargs) (line 14)
    print_call_result_187934 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), print_187928, *[msg_187929], **kwargs_187933)
    
    
    # ################# End of 'error_msg(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'error_msg' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_187935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_187935)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'error_msg'
    return stypy_return_type_187935

# Assigning a type to the variable 'error_msg' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'error_msg', error_msg)
# Declaration of the 'Gcf' class

class Gcf(object, ):
    unicode_187936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, (-1)), 'unicode', u'\n    Singleton to manage a set of integer-numbered figures.\n\n    This class is never instantiated; it consists of two class\n    attributes (a list and a dictionary), and a set of static\n    methods that operate on those attributes, accessing them\n    directly as class attributes.\n\n    Attributes:\n\n        *figs*:\n          dictionary of the form {*num*: *manager*, ...}\n\n        *_activeQue*:\n          list of *managers*, with active one at the end\n\n    ')

    @norecursion
    def get_fig_manager(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_fig_manager'
        module_type_store = module_type_store.open_function_context('get_fig_manager', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Gcf.get_fig_manager.__dict__.__setitem__('stypy_localization', localization)
        Gcf.get_fig_manager.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Gcf.get_fig_manager.__dict__.__setitem__('stypy_type_store', module_type_store)
        Gcf.get_fig_manager.__dict__.__setitem__('stypy_function_name', 'Gcf.get_fig_manager')
        Gcf.get_fig_manager.__dict__.__setitem__('stypy_param_names_list', ['num'])
        Gcf.get_fig_manager.__dict__.__setitem__('stypy_varargs_param_name', None)
        Gcf.get_fig_manager.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Gcf.get_fig_manager.__dict__.__setitem__('stypy_call_defaults', defaults)
        Gcf.get_fig_manager.__dict__.__setitem__('stypy_call_varargs', varargs)
        Gcf.get_fig_manager.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Gcf.get_fig_manager.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Gcf.get_fig_manager', ['num'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_fig_manager', localization, ['num'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_fig_manager(...)' code ##################

        unicode_187937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'unicode', u'\n        If figure manager *num* exists, make it the active\n        figure and return the manager; otherwise return *None*.\n        ')
        
        # Assigning a Call to a Name (line 44):
        
        # Call to get(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'num' (line 44)
        num_187941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 31), 'num', False)
        # Getting the type of 'None' (line 44)
        None_187942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 36), 'None', False)
        # Processing the call keyword arguments (line 44)
        kwargs_187943 = {}
        # Getting the type of 'cls' (line 44)
        cls_187938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 18), 'cls', False)
        # Obtaining the member 'figs' of a type (line 44)
        figs_187939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 18), cls_187938, 'figs')
        # Obtaining the member 'get' of a type (line 44)
        get_187940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 18), figs_187939, 'get')
        # Calling get(args, kwargs) (line 44)
        get_call_result_187944 = invoke(stypy.reporting.localization.Localization(__file__, 44, 18), get_187940, *[num_187941, None_187942], **kwargs_187943)
        
        # Assigning a type to the variable 'manager' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'manager', get_call_result_187944)
        
        # Type idiom detected: calculating its left and rigth part (line 45)
        # Getting the type of 'manager' (line 45)
        manager_187945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'manager')
        # Getting the type of 'None' (line 45)
        None_187946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 26), 'None')
        
        (may_be_187947, more_types_in_union_187948) = may_not_be_none(manager_187945, None_187946)

        if may_be_187947:

            if more_types_in_union_187948:
                # Runtime conditional SSA (line 45)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to set_active(...): (line 46)
            # Processing the call arguments (line 46)
            # Getting the type of 'manager' (line 46)
            manager_187951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 27), 'manager', False)
            # Processing the call keyword arguments (line 46)
            kwargs_187952 = {}
            # Getting the type of 'cls' (line 46)
            cls_187949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'cls', False)
            # Obtaining the member 'set_active' of a type (line 46)
            set_active_187950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 12), cls_187949, 'set_active')
            # Calling set_active(args, kwargs) (line 46)
            set_active_call_result_187953 = invoke(stypy.reporting.localization.Localization(__file__, 46, 12), set_active_187950, *[manager_187951], **kwargs_187952)
            

            if more_types_in_union_187948:
                # SSA join for if statement (line 45)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'manager' (line 47)
        manager_187954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'manager')
        # Assigning a type to the variable 'stypy_return_type' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'stypy_return_type', manager_187954)
        
        # ################# End of 'get_fig_manager(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_fig_manager' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_187955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187955)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_fig_manager'
        return stypy_return_type_187955


    @norecursion
    def destroy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'destroy'
        module_type_store = module_type_store.open_function_context('destroy', 49, 4, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Gcf.destroy.__dict__.__setitem__('stypy_localization', localization)
        Gcf.destroy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Gcf.destroy.__dict__.__setitem__('stypy_type_store', module_type_store)
        Gcf.destroy.__dict__.__setitem__('stypy_function_name', 'Gcf.destroy')
        Gcf.destroy.__dict__.__setitem__('stypy_param_names_list', ['num'])
        Gcf.destroy.__dict__.__setitem__('stypy_varargs_param_name', None)
        Gcf.destroy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Gcf.destroy.__dict__.__setitem__('stypy_call_defaults', defaults)
        Gcf.destroy.__dict__.__setitem__('stypy_call_varargs', varargs)
        Gcf.destroy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Gcf.destroy.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Gcf.destroy', ['num'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'destroy', localization, ['num'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'destroy(...)' code ##################

        unicode_187956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'unicode', u'\n        Try to remove all traces of figure *num*.\n\n        In the interactive backends, this is bound to the\n        window "destroy" and "delete" events.\n        ')
        
        
        
        # Call to has_fignum(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'num' (line 57)
        num_187959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 30), 'num', False)
        # Processing the call keyword arguments (line 57)
        kwargs_187960 = {}
        # Getting the type of 'cls' (line 57)
        cls_187957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'cls', False)
        # Obtaining the member 'has_fignum' of a type (line 57)
        has_fignum_187958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 15), cls_187957, 'has_fignum')
        # Calling has_fignum(args, kwargs) (line 57)
        has_fignum_call_result_187961 = invoke(stypy.reporting.localization.Localization(__file__, 57, 15), has_fignum_187958, *[num_187959], **kwargs_187960)
        
        # Applying the 'not' unary operator (line 57)
        result_not__187962 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 11), 'not', has_fignum_call_result_187961)
        
        # Testing the type of an if condition (line 57)
        if_condition_187963 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 8), result_not__187962)
        # Assigning a type to the variable 'if_condition_187963' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'if_condition_187963', if_condition_187963)
        # SSA begins for if statement (line 57)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 57)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 59):
        
        # Obtaining the type of the subscript
        # Getting the type of 'num' (line 59)
        num_187964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 27), 'num')
        # Getting the type of 'cls' (line 59)
        cls_187965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 18), 'cls')
        # Obtaining the member 'figs' of a type (line 59)
        figs_187966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 18), cls_187965, 'figs')
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___187967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 18), figs_187966, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_187968 = invoke(stypy.reporting.localization.Localization(__file__, 59, 18), getitem___187967, num_187964)
        
        # Assigning a type to the variable 'manager' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'manager', subscript_call_result_187968)
        
        # Call to mpl_disconnect(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'manager' (line 60)
        manager_187972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 38), 'manager', False)
        # Obtaining the member '_cidgcf' of a type (line 60)
        _cidgcf_187973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 38), manager_187972, '_cidgcf')
        # Processing the call keyword arguments (line 60)
        kwargs_187974 = {}
        # Getting the type of 'manager' (line 60)
        manager_187969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'manager', False)
        # Obtaining the member 'canvas' of a type (line 60)
        canvas_187970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), manager_187969, 'canvas')
        # Obtaining the member 'mpl_disconnect' of a type (line 60)
        mpl_disconnect_187971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), canvas_187970, 'mpl_disconnect')
        # Calling mpl_disconnect(args, kwargs) (line 60)
        mpl_disconnect_call_result_187975 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), mpl_disconnect_187971, *[_cidgcf_187973], **kwargs_187974)
        
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        slice_187976 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 64, 17), None, None, None)
        # Getting the type of 'cls' (line 64)
        cls_187977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 17), 'cls')
        # Obtaining the member '_activeQue' of a type (line 64)
        _activeQue_187978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 17), cls_187977, '_activeQue')
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___187979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 17), _activeQue_187978, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_187980 = invoke(stypy.reporting.localization.Localization(__file__, 64, 17), getitem___187979, slice_187976)
        
        # Assigning a type to the variable 'oldQue' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'oldQue', subscript_call_result_187980)
        
        # Assigning a List to a Attribute (line 65):
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_187981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        
        # Getting the type of 'cls' (line 65)
        cls_187982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'cls')
        # Setting the type of the member '_activeQue' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), cls_187982, '_activeQue', list_187981)
        
        # Getting the type of 'oldQue' (line 66)
        oldQue_187983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'oldQue')
        # Testing the type of a for loop iterable (line 66)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 66, 8), oldQue_187983)
        # Getting the type of the for loop variable (line 66)
        for_loop_var_187984 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 66, 8), oldQue_187983)
        # Assigning a type to the variable 'f' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'f', for_loop_var_187984)
        # SSA begins for a for statement (line 66)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'f' (line 67)
        f_187985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'f')
        # Getting the type of 'manager' (line 67)
        manager_187986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), 'manager')
        # Applying the binary operator '!=' (line 67)
        result_ne_187987 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 15), '!=', f_187985, manager_187986)
        
        # Testing the type of an if condition (line 67)
        if_condition_187988 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 12), result_ne_187987)
        # Assigning a type to the variable 'if_condition_187988' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'if_condition_187988', if_condition_187988)
        # SSA begins for if statement (line 67)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'f' (line 68)
        f_187992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 38), 'f', False)
        # Processing the call keyword arguments (line 68)
        kwargs_187993 = {}
        # Getting the type of 'cls' (line 68)
        cls_187989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'cls', False)
        # Obtaining the member '_activeQue' of a type (line 68)
        _activeQue_187990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), cls_187989, '_activeQue')
        # Obtaining the member 'append' of a type (line 68)
        append_187991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), _activeQue_187990, 'append')
        # Calling append(args, kwargs) (line 68)
        append_call_result_187994 = invoke(stypy.reporting.localization.Localization(__file__, 68, 16), append_187991, *[f_187992], **kwargs_187993)
        
        # SSA join for if statement (line 67)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Deleting a member
        # Getting the type of 'cls' (line 70)
        cls_187995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'cls')
        # Obtaining the member 'figs' of a type (line 70)
        figs_187996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), cls_187995, 'figs')
        
        # Obtaining the type of the subscript
        # Getting the type of 'num' (line 70)
        num_187997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 21), 'num')
        # Getting the type of 'cls' (line 70)
        cls_187998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'cls')
        # Obtaining the member 'figs' of a type (line 70)
        figs_187999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), cls_187998, 'figs')
        # Obtaining the member '__getitem__' of a type (line 70)
        getitem___188000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), figs_187999, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 70)
        subscript_call_result_188001 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), getitem___188000, num_187997)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 8), figs_187996, subscript_call_result_188001)
        
        # Call to destroy(...): (line 71)
        # Processing the call keyword arguments (line 71)
        kwargs_188004 = {}
        # Getting the type of 'manager' (line 71)
        manager_188002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'manager', False)
        # Obtaining the member 'destroy' of a type (line 71)
        destroy_188003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), manager_188002, 'destroy')
        # Calling destroy(args, kwargs) (line 71)
        destroy_call_result_188005 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), destroy_188003, *[], **kwargs_188004)
        
        
        # Call to collect(...): (line 72)
        # Processing the call arguments (line 72)
        int_188008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 19), 'int')
        # Processing the call keyword arguments (line 72)
        kwargs_188009 = {}
        # Getting the type of 'gc' (line 72)
        gc_188006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'gc', False)
        # Obtaining the member 'collect' of a type (line 72)
        collect_188007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), gc_188006, 'collect')
        # Calling collect(args, kwargs) (line 72)
        collect_call_result_188010 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), collect_188007, *[int_188008], **kwargs_188009)
        
        
        # ################# End of 'destroy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'destroy' in the type store
        # Getting the type of 'stypy_return_type' (line 49)
        stypy_return_type_188011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188011)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'destroy'
        return stypy_return_type_188011


    @norecursion
    def destroy_fig(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'destroy_fig'
        module_type_store = module_type_store.open_function_context('destroy_fig', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Gcf.destroy_fig.__dict__.__setitem__('stypy_localization', localization)
        Gcf.destroy_fig.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Gcf.destroy_fig.__dict__.__setitem__('stypy_type_store', module_type_store)
        Gcf.destroy_fig.__dict__.__setitem__('stypy_function_name', 'Gcf.destroy_fig')
        Gcf.destroy_fig.__dict__.__setitem__('stypy_param_names_list', ['fig'])
        Gcf.destroy_fig.__dict__.__setitem__('stypy_varargs_param_name', None)
        Gcf.destroy_fig.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Gcf.destroy_fig.__dict__.__setitem__('stypy_call_defaults', defaults)
        Gcf.destroy_fig.__dict__.__setitem__('stypy_call_varargs', varargs)
        Gcf.destroy_fig.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Gcf.destroy_fig.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Gcf.destroy_fig', ['fig'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'destroy_fig', localization, ['fig'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'destroy_fig(...)' code ##################

        unicode_188012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 8), 'unicode', u'*fig* is a Figure instance')
        
        # Assigning a Name to a Name (line 77):
        # Getting the type of 'None' (line 77)
        None_188013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 14), 'None')
        # Assigning a type to the variable 'num' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'num', None_188013)
        
        
        # Call to itervalues(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'cls' (line 78)
        cls_188016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 38), 'cls', False)
        # Obtaining the member 'figs' of a type (line 78)
        figs_188017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 38), cls_188016, 'figs')
        # Processing the call keyword arguments (line 78)
        kwargs_188018 = {}
        # Getting the type of 'six' (line 78)
        six_188014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'six', False)
        # Obtaining the member 'itervalues' of a type (line 78)
        itervalues_188015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), six_188014, 'itervalues')
        # Calling itervalues(args, kwargs) (line 78)
        itervalues_call_result_188019 = invoke(stypy.reporting.localization.Localization(__file__, 78, 23), itervalues_188015, *[figs_188017], **kwargs_188018)
        
        # Testing the type of a for loop iterable (line 78)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 78, 8), itervalues_call_result_188019)
        # Getting the type of the for loop variable (line 78)
        for_loop_var_188020 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 78, 8), itervalues_call_result_188019)
        # Assigning a type to the variable 'manager' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'manager', for_loop_var_188020)
        # SSA begins for a for statement (line 78)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'manager' (line 79)
        manager_188021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'manager')
        # Obtaining the member 'canvas' of a type (line 79)
        canvas_188022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 15), manager_188021, 'canvas')
        # Obtaining the member 'figure' of a type (line 79)
        figure_188023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 15), canvas_188022, 'figure')
        # Getting the type of 'fig' (line 79)
        fig_188024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 40), 'fig')
        # Applying the binary operator '==' (line 79)
        result_eq_188025 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 15), '==', figure_188023, fig_188024)
        
        # Testing the type of an if condition (line 79)
        if_condition_188026 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 12), result_eq_188025)
        # Assigning a type to the variable 'if_condition_188026' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'if_condition_188026', if_condition_188026)
        # SSA begins for if statement (line 79)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 80):
        # Getting the type of 'manager' (line 80)
        manager_188027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 22), 'manager')
        # Obtaining the member 'num' of a type (line 80)
        num_188028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 22), manager_188027, 'num')
        # Assigning a type to the variable 'num' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'num', num_188028)
        # SSA join for if statement (line 79)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 82)
        # Getting the type of 'num' (line 82)
        num_188029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'num')
        # Getting the type of 'None' (line 82)
        None_188030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 22), 'None')
        
        (may_be_188031, more_types_in_union_188032) = may_not_be_none(num_188029, None_188030)

        if may_be_188031:

            if more_types_in_union_188032:
                # Runtime conditional SSA (line 82)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to destroy(...): (line 83)
            # Processing the call arguments (line 83)
            # Getting the type of 'num' (line 83)
            num_188035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'num', False)
            # Processing the call keyword arguments (line 83)
            kwargs_188036 = {}
            # Getting the type of 'cls' (line 83)
            cls_188033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'cls', False)
            # Obtaining the member 'destroy' of a type (line 83)
            destroy_188034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), cls_188033, 'destroy')
            # Calling destroy(args, kwargs) (line 83)
            destroy_call_result_188037 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), destroy_188034, *[num_188035], **kwargs_188036)
            

            if more_types_in_union_188032:
                # SSA join for if statement (line 82)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'destroy_fig(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'destroy_fig' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_188038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188038)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'destroy_fig'
        return stypy_return_type_188038


    @norecursion
    def destroy_all(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'destroy_all'
        module_type_store = module_type_store.open_function_context('destroy_all', 85, 4, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Gcf.destroy_all.__dict__.__setitem__('stypy_localization', localization)
        Gcf.destroy_all.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Gcf.destroy_all.__dict__.__setitem__('stypy_type_store', module_type_store)
        Gcf.destroy_all.__dict__.__setitem__('stypy_function_name', 'Gcf.destroy_all')
        Gcf.destroy_all.__dict__.__setitem__('stypy_param_names_list', [])
        Gcf.destroy_all.__dict__.__setitem__('stypy_varargs_param_name', None)
        Gcf.destroy_all.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Gcf.destroy_all.__dict__.__setitem__('stypy_call_defaults', defaults)
        Gcf.destroy_all.__dict__.__setitem__('stypy_call_varargs', varargs)
        Gcf.destroy_all.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Gcf.destroy_all.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Gcf.destroy_all', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'destroy_all', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'destroy_all(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 89, 8))
        
        # 'import gc' statement (line 89)
        import gc

        import_module(stypy.reporting.localization.Localization(__file__, 89, 8), 'gc', gc, module_type_store)
        
        
        
        # Call to list(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Call to values(...): (line 90)
        # Processing the call keyword arguments (line 90)
        kwargs_188043 = {}
        # Getting the type of 'cls' (line 90)
        cls_188040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 28), 'cls', False)
        # Obtaining the member 'figs' of a type (line 90)
        figs_188041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 28), cls_188040, 'figs')
        # Obtaining the member 'values' of a type (line 90)
        values_188042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 28), figs_188041, 'values')
        # Calling values(args, kwargs) (line 90)
        values_call_result_188044 = invoke(stypy.reporting.localization.Localization(__file__, 90, 28), values_188042, *[], **kwargs_188043)
        
        # Processing the call keyword arguments (line 90)
        kwargs_188045 = {}
        # Getting the type of 'list' (line 90)
        list_188039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 23), 'list', False)
        # Calling list(args, kwargs) (line 90)
        list_call_result_188046 = invoke(stypy.reporting.localization.Localization(__file__, 90, 23), list_188039, *[values_call_result_188044], **kwargs_188045)
        
        # Testing the type of a for loop iterable (line 90)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 90, 8), list_call_result_188046)
        # Getting the type of the for loop variable (line 90)
        for_loop_var_188047 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 90, 8), list_call_result_188046)
        # Assigning a type to the variable 'manager' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'manager', for_loop_var_188047)
        # SSA begins for a for statement (line 90)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to mpl_disconnect(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'manager' (line 91)
        manager_188051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 42), 'manager', False)
        # Obtaining the member '_cidgcf' of a type (line 91)
        _cidgcf_188052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 42), manager_188051, '_cidgcf')
        # Processing the call keyword arguments (line 91)
        kwargs_188053 = {}
        # Getting the type of 'manager' (line 91)
        manager_188048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'manager', False)
        # Obtaining the member 'canvas' of a type (line 91)
        canvas_188049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), manager_188048, 'canvas')
        # Obtaining the member 'mpl_disconnect' of a type (line 91)
        mpl_disconnect_188050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), canvas_188049, 'mpl_disconnect')
        # Calling mpl_disconnect(args, kwargs) (line 91)
        mpl_disconnect_call_result_188054 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), mpl_disconnect_188050, *[_cidgcf_188052], **kwargs_188053)
        
        
        # Call to destroy(...): (line 92)
        # Processing the call keyword arguments (line 92)
        kwargs_188057 = {}
        # Getting the type of 'manager' (line 92)
        manager_188055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'manager', False)
        # Obtaining the member 'destroy' of a type (line 92)
        destroy_188056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), manager_188055, 'destroy')
        # Calling destroy(args, kwargs) (line 92)
        destroy_call_result_188058 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), destroy_188056, *[], **kwargs_188057)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Attribute (line 94):
        
        # Obtaining an instance of the builtin type 'list' (line 94)
        list_188059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 94)
        
        # Getting the type of 'cls' (line 94)
        cls_188060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'cls')
        # Setting the type of the member '_activeQue' of a type (line 94)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), cls_188060, '_activeQue', list_188059)
        
        # Call to clear(...): (line 95)
        # Processing the call keyword arguments (line 95)
        kwargs_188064 = {}
        # Getting the type of 'cls' (line 95)
        cls_188061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'cls', False)
        # Obtaining the member 'figs' of a type (line 95)
        figs_188062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), cls_188061, 'figs')
        # Obtaining the member 'clear' of a type (line 95)
        clear_188063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), figs_188062, 'clear')
        # Calling clear(args, kwargs) (line 95)
        clear_call_result_188065 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), clear_188063, *[], **kwargs_188064)
        
        
        # Call to collect(...): (line 96)
        # Processing the call arguments (line 96)
        int_188068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 19), 'int')
        # Processing the call keyword arguments (line 96)
        kwargs_188069 = {}
        # Getting the type of 'gc' (line 96)
        gc_188066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'gc', False)
        # Obtaining the member 'collect' of a type (line 96)
        collect_188067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), gc_188066, 'collect')
        # Calling collect(args, kwargs) (line 96)
        collect_call_result_188070 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), collect_188067, *[int_188068], **kwargs_188069)
        
        
        # ################# End of 'destroy_all(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'destroy_all' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_188071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188071)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'destroy_all'
        return stypy_return_type_188071


    @norecursion
    def has_fignum(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_fignum'
        module_type_store = module_type_store.open_function_context('has_fignum', 98, 4, False)
        # Assigning a type to the variable 'self' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Gcf.has_fignum.__dict__.__setitem__('stypy_localization', localization)
        Gcf.has_fignum.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Gcf.has_fignum.__dict__.__setitem__('stypy_type_store', module_type_store)
        Gcf.has_fignum.__dict__.__setitem__('stypy_function_name', 'Gcf.has_fignum')
        Gcf.has_fignum.__dict__.__setitem__('stypy_param_names_list', ['num'])
        Gcf.has_fignum.__dict__.__setitem__('stypy_varargs_param_name', None)
        Gcf.has_fignum.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Gcf.has_fignum.__dict__.__setitem__('stypy_call_defaults', defaults)
        Gcf.has_fignum.__dict__.__setitem__('stypy_call_varargs', varargs)
        Gcf.has_fignum.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Gcf.has_fignum.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Gcf.has_fignum', ['num'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_fignum', localization, ['num'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_fignum(...)' code ##################

        unicode_188072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, (-1)), 'unicode', u'\n        Return *True* if figure *num* exists.\n        ')
        
        # Getting the type of 'num' (line 103)
        num_188073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'num')
        # Getting the type of 'cls' (line 103)
        cls_188074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 22), 'cls')
        # Obtaining the member 'figs' of a type (line 103)
        figs_188075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 22), cls_188074, 'figs')
        # Applying the binary operator 'in' (line 103)
        result_contains_188076 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 15), 'in', num_188073, figs_188075)
        
        # Assigning a type to the variable 'stypy_return_type' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'stypy_return_type', result_contains_188076)
        
        # ################# End of 'has_fignum(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_fignum' in the type store
        # Getting the type of 'stypy_return_type' (line 98)
        stypy_return_type_188077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188077)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_fignum'
        return stypy_return_type_188077


    @norecursion
    def get_all_fig_managers(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_all_fig_managers'
        module_type_store = module_type_store.open_function_context('get_all_fig_managers', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Gcf.get_all_fig_managers.__dict__.__setitem__('stypy_localization', localization)
        Gcf.get_all_fig_managers.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Gcf.get_all_fig_managers.__dict__.__setitem__('stypy_type_store', module_type_store)
        Gcf.get_all_fig_managers.__dict__.__setitem__('stypy_function_name', 'Gcf.get_all_fig_managers')
        Gcf.get_all_fig_managers.__dict__.__setitem__('stypy_param_names_list', [])
        Gcf.get_all_fig_managers.__dict__.__setitem__('stypy_varargs_param_name', None)
        Gcf.get_all_fig_managers.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Gcf.get_all_fig_managers.__dict__.__setitem__('stypy_call_defaults', defaults)
        Gcf.get_all_fig_managers.__dict__.__setitem__('stypy_call_varargs', varargs)
        Gcf.get_all_fig_managers.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Gcf.get_all_fig_managers.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Gcf.get_all_fig_managers', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_all_fig_managers', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_all_fig_managers(...)' code ##################

        unicode_188078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, (-1)), 'unicode', u'\n        Return a list of figure managers.\n        ')
        
        # Call to list(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Call to values(...): (line 110)
        # Processing the call keyword arguments (line 110)
        kwargs_188083 = {}
        # Getting the type of 'cls' (line 110)
        cls_188080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 20), 'cls', False)
        # Obtaining the member 'figs' of a type (line 110)
        figs_188081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 20), cls_188080, 'figs')
        # Obtaining the member 'values' of a type (line 110)
        values_188082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 20), figs_188081, 'values')
        # Calling values(args, kwargs) (line 110)
        values_call_result_188084 = invoke(stypy.reporting.localization.Localization(__file__, 110, 20), values_188082, *[], **kwargs_188083)
        
        # Processing the call keyword arguments (line 110)
        kwargs_188085 = {}
        # Getting the type of 'list' (line 110)
        list_188079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 15), 'list', False)
        # Calling list(args, kwargs) (line 110)
        list_call_result_188086 = invoke(stypy.reporting.localization.Localization(__file__, 110, 15), list_188079, *[values_call_result_188084], **kwargs_188085)
        
        # Assigning a type to the variable 'stypy_return_type' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'stypy_return_type', list_call_result_188086)
        
        # ################# End of 'get_all_fig_managers(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_all_fig_managers' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_188087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188087)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_all_fig_managers'
        return stypy_return_type_188087


    @norecursion
    def get_num_fig_managers(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_num_fig_managers'
        module_type_store = module_type_store.open_function_context('get_num_fig_managers', 112, 4, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Gcf.get_num_fig_managers.__dict__.__setitem__('stypy_localization', localization)
        Gcf.get_num_fig_managers.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Gcf.get_num_fig_managers.__dict__.__setitem__('stypy_type_store', module_type_store)
        Gcf.get_num_fig_managers.__dict__.__setitem__('stypy_function_name', 'Gcf.get_num_fig_managers')
        Gcf.get_num_fig_managers.__dict__.__setitem__('stypy_param_names_list', [])
        Gcf.get_num_fig_managers.__dict__.__setitem__('stypy_varargs_param_name', None)
        Gcf.get_num_fig_managers.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Gcf.get_num_fig_managers.__dict__.__setitem__('stypy_call_defaults', defaults)
        Gcf.get_num_fig_managers.__dict__.__setitem__('stypy_call_varargs', varargs)
        Gcf.get_num_fig_managers.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Gcf.get_num_fig_managers.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Gcf.get_num_fig_managers', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_num_fig_managers', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_num_fig_managers(...)' code ##################

        unicode_188088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, (-1)), 'unicode', u'\n        Return the number of figures being managed.\n        ')
        
        # Call to len(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'cls' (line 117)
        cls_188090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 19), 'cls', False)
        # Obtaining the member 'figs' of a type (line 117)
        figs_188091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 19), cls_188090, 'figs')
        # Processing the call keyword arguments (line 117)
        kwargs_188092 = {}
        # Getting the type of 'len' (line 117)
        len_188089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'len', False)
        # Calling len(args, kwargs) (line 117)
        len_call_result_188093 = invoke(stypy.reporting.localization.Localization(__file__, 117, 15), len_188089, *[figs_188091], **kwargs_188092)
        
        # Assigning a type to the variable 'stypy_return_type' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'stypy_return_type', len_call_result_188093)
        
        # ################# End of 'get_num_fig_managers(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_num_fig_managers' in the type store
        # Getting the type of 'stypy_return_type' (line 112)
        stypy_return_type_188094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188094)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_num_fig_managers'
        return stypy_return_type_188094


    @norecursion
    def get_active(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_active'
        module_type_store = module_type_store.open_function_context('get_active', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Gcf.get_active.__dict__.__setitem__('stypy_localization', localization)
        Gcf.get_active.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Gcf.get_active.__dict__.__setitem__('stypy_type_store', module_type_store)
        Gcf.get_active.__dict__.__setitem__('stypy_function_name', 'Gcf.get_active')
        Gcf.get_active.__dict__.__setitem__('stypy_param_names_list', [])
        Gcf.get_active.__dict__.__setitem__('stypy_varargs_param_name', None)
        Gcf.get_active.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Gcf.get_active.__dict__.__setitem__('stypy_call_defaults', defaults)
        Gcf.get_active.__dict__.__setitem__('stypy_call_varargs', varargs)
        Gcf.get_active.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Gcf.get_active.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Gcf.get_active', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_active', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_active(...)' code ##################

        unicode_188095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, (-1)), 'unicode', u'\n        Return the manager of the active figure, or *None*.\n        ')
        
        
        
        # Call to len(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'cls' (line 124)
        cls_188097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 15), 'cls', False)
        # Obtaining the member '_activeQue' of a type (line 124)
        _activeQue_188098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 15), cls_188097, '_activeQue')
        # Processing the call keyword arguments (line 124)
        kwargs_188099 = {}
        # Getting the type of 'len' (line 124)
        len_188096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 11), 'len', False)
        # Calling len(args, kwargs) (line 124)
        len_call_result_188100 = invoke(stypy.reporting.localization.Localization(__file__, 124, 11), len_188096, *[_activeQue_188098], **kwargs_188099)
        
        int_188101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 34), 'int')
        # Applying the binary operator '==' (line 124)
        result_eq_188102 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 11), '==', len_call_result_188100, int_188101)
        
        # Testing the type of an if condition (line 124)
        if_condition_188103 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 8), result_eq_188102)
        # Assigning a type to the variable 'if_condition_188103' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'if_condition_188103', if_condition_188103)
        # SSA begins for if statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 125)
        None_188104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'stypy_return_type', None_188104)
        # SSA branch for the else part of an if statement (line 124)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining the type of the subscript
        int_188105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 34), 'int')
        # Getting the type of 'cls' (line 127)
        cls_188106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 19), 'cls')
        # Obtaining the member '_activeQue' of a type (line 127)
        _activeQue_188107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 19), cls_188106, '_activeQue')
        # Obtaining the member '__getitem__' of a type (line 127)
        getitem___188108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 19), _activeQue_188107, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 127)
        subscript_call_result_188109 = invoke(stypy.reporting.localization.Localization(__file__, 127, 19), getitem___188108, int_188105)
        
        # Assigning a type to the variable 'stypy_return_type' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'stypy_return_type', subscript_call_result_188109)
        # SSA join for if statement (line 124)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_active(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_active' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_188110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188110)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_active'
        return stypy_return_type_188110


    @norecursion
    def set_active(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_active'
        module_type_store = module_type_store.open_function_context('set_active', 129, 4, False)
        # Assigning a type to the variable 'self' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Gcf.set_active.__dict__.__setitem__('stypy_localization', localization)
        Gcf.set_active.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Gcf.set_active.__dict__.__setitem__('stypy_type_store', module_type_store)
        Gcf.set_active.__dict__.__setitem__('stypy_function_name', 'Gcf.set_active')
        Gcf.set_active.__dict__.__setitem__('stypy_param_names_list', ['manager'])
        Gcf.set_active.__dict__.__setitem__('stypy_varargs_param_name', None)
        Gcf.set_active.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Gcf.set_active.__dict__.__setitem__('stypy_call_defaults', defaults)
        Gcf.set_active.__dict__.__setitem__('stypy_call_varargs', varargs)
        Gcf.set_active.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Gcf.set_active.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Gcf.set_active', ['manager'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_active', localization, ['manager'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_active(...)' code ##################

        unicode_188111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, (-1)), 'unicode', u'\n        Make the figure corresponding to *manager* the active one.\n        ')
        
        # Assigning a Subscript to a Name (line 134):
        
        # Obtaining the type of the subscript
        slice_188112 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 134, 17), None, None, None)
        # Getting the type of 'cls' (line 134)
        cls_188113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 17), 'cls')
        # Obtaining the member '_activeQue' of a type (line 134)
        _activeQue_188114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 17), cls_188113, '_activeQue')
        # Obtaining the member '__getitem__' of a type (line 134)
        getitem___188115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 17), _activeQue_188114, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 134)
        subscript_call_result_188116 = invoke(stypy.reporting.localization.Localization(__file__, 134, 17), getitem___188115, slice_188112)
        
        # Assigning a type to the variable 'oldQue' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'oldQue', subscript_call_result_188116)
        
        # Assigning a List to a Attribute (line 135):
        
        # Obtaining an instance of the builtin type 'list' (line 135)
        list_188117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 135)
        
        # Getting the type of 'cls' (line 135)
        cls_188118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'cls')
        # Setting the type of the member '_activeQue' of a type (line 135)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), cls_188118, '_activeQue', list_188117)
        
        # Getting the type of 'oldQue' (line 136)
        oldQue_188119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 17), 'oldQue')
        # Testing the type of a for loop iterable (line 136)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 136, 8), oldQue_188119)
        # Getting the type of the for loop variable (line 136)
        for_loop_var_188120 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 136, 8), oldQue_188119)
        # Assigning a type to the variable 'm' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'm', for_loop_var_188120)
        # SSA begins for a for statement (line 136)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'm' (line 137)
        m_188121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'm')
        # Getting the type of 'manager' (line 137)
        manager_188122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 20), 'manager')
        # Applying the binary operator '!=' (line 137)
        result_ne_188123 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 15), '!=', m_188121, manager_188122)
        
        # Testing the type of an if condition (line 137)
        if_condition_188124 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 12), result_ne_188123)
        # Assigning a type to the variable 'if_condition_188124' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'if_condition_188124', if_condition_188124)
        # SSA begins for if statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'm' (line 138)
        m_188128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 38), 'm', False)
        # Processing the call keyword arguments (line 138)
        kwargs_188129 = {}
        # Getting the type of 'cls' (line 138)
        cls_188125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'cls', False)
        # Obtaining the member '_activeQue' of a type (line 138)
        _activeQue_188126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 16), cls_188125, '_activeQue')
        # Obtaining the member 'append' of a type (line 138)
        append_188127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 16), _activeQue_188126, 'append')
        # Calling append(args, kwargs) (line 138)
        append_call_result_188130 = invoke(stypy.reporting.localization.Localization(__file__, 138, 16), append_188127, *[m_188128], **kwargs_188129)
        
        # SSA join for if statement (line 137)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'manager' (line 139)
        manager_188134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 30), 'manager', False)
        # Processing the call keyword arguments (line 139)
        kwargs_188135 = {}
        # Getting the type of 'cls' (line 139)
        cls_188131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'cls', False)
        # Obtaining the member '_activeQue' of a type (line 139)
        _activeQue_188132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), cls_188131, '_activeQue')
        # Obtaining the member 'append' of a type (line 139)
        append_188133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), _activeQue_188132, 'append')
        # Calling append(args, kwargs) (line 139)
        append_call_result_188136 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), append_188133, *[manager_188134], **kwargs_188135)
        
        
        # Assigning a Name to a Subscript (line 140):
        # Getting the type of 'manager' (line 140)
        manager_188137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 32), 'manager')
        # Getting the type of 'cls' (line 140)
        cls_188138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'cls')
        # Obtaining the member 'figs' of a type (line 140)
        figs_188139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), cls_188138, 'figs')
        # Getting the type of 'manager' (line 140)
        manager_188140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 17), 'manager')
        # Obtaining the member 'num' of a type (line 140)
        num_188141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 17), manager_188140, 'num')
        # Storing an element on a container (line 140)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 8), figs_188139, (num_188141, manager_188137))
        
        # ################# End of 'set_active(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_active' in the type store
        # Getting the type of 'stypy_return_type' (line 129)
        stypy_return_type_188142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188142)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_active'
        return stypy_return_type_188142


    @norecursion
    def draw_all(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 143)
        False_188143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 28), 'False')
        defaults = [False_188143]
        # Create a new context for function 'draw_all'
        module_type_store = module_type_store.open_function_context('draw_all', 142, 4, False)
        # Assigning a type to the variable 'self' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Gcf.draw_all.__dict__.__setitem__('stypy_localization', localization)
        Gcf.draw_all.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Gcf.draw_all.__dict__.__setitem__('stypy_type_store', module_type_store)
        Gcf.draw_all.__dict__.__setitem__('stypy_function_name', 'Gcf.draw_all')
        Gcf.draw_all.__dict__.__setitem__('stypy_param_names_list', ['force'])
        Gcf.draw_all.__dict__.__setitem__('stypy_varargs_param_name', None)
        Gcf.draw_all.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Gcf.draw_all.__dict__.__setitem__('stypy_call_defaults', defaults)
        Gcf.draw_all.__dict__.__setitem__('stypy_call_varargs', varargs)
        Gcf.draw_all.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Gcf.draw_all.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Gcf.draw_all', ['force'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_all', localization, ['force'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_all(...)' code ##################

        unicode_188144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, (-1)), 'unicode', u'\n        Redraw all figures registered with the pyplot\n        state machine.\n        ')
        
        
        # Call to get_all_fig_managers(...): (line 148)
        # Processing the call keyword arguments (line 148)
        kwargs_188147 = {}
        # Getting the type of 'cls' (line 148)
        cls_188145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 21), 'cls', False)
        # Obtaining the member 'get_all_fig_managers' of a type (line 148)
        get_all_fig_managers_188146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 21), cls_188145, 'get_all_fig_managers')
        # Calling get_all_fig_managers(args, kwargs) (line 148)
        get_all_fig_managers_call_result_188148 = invoke(stypy.reporting.localization.Localization(__file__, 148, 21), get_all_fig_managers_188146, *[], **kwargs_188147)
        
        # Testing the type of a for loop iterable (line 148)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 148, 8), get_all_fig_managers_call_result_188148)
        # Getting the type of the for loop variable (line 148)
        for_loop_var_188149 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 148, 8), get_all_fig_managers_call_result_188148)
        # Assigning a type to the variable 'f_mgr' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'f_mgr', for_loop_var_188149)
        # SSA begins for a for statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        # Getting the type of 'force' (line 149)
        force_188150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'force')
        # Getting the type of 'f_mgr' (line 149)
        f_mgr_188151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'f_mgr')
        # Obtaining the member 'canvas' of a type (line 149)
        canvas_188152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 24), f_mgr_188151, 'canvas')
        # Obtaining the member 'figure' of a type (line 149)
        figure_188153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 24), canvas_188152, 'figure')
        # Obtaining the member 'stale' of a type (line 149)
        stale_188154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 24), figure_188153, 'stale')
        # Applying the binary operator 'or' (line 149)
        result_or_keyword_188155 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 15), 'or', force_188150, stale_188154)
        
        # Testing the type of an if condition (line 149)
        if_condition_188156 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 12), result_or_keyword_188155)
        # Assigning a type to the variable 'if_condition_188156' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'if_condition_188156', if_condition_188156)
        # SSA begins for if statement (line 149)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to draw_idle(...): (line 150)
        # Processing the call keyword arguments (line 150)
        kwargs_188160 = {}
        # Getting the type of 'f_mgr' (line 150)
        f_mgr_188157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'f_mgr', False)
        # Obtaining the member 'canvas' of a type (line 150)
        canvas_188158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 16), f_mgr_188157, 'canvas')
        # Obtaining the member 'draw_idle' of a type (line 150)
        draw_idle_188159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 16), canvas_188158, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 150)
        draw_idle_call_result_188161 = invoke(stypy.reporting.localization.Localization(__file__, 150, 16), draw_idle_188159, *[], **kwargs_188160)
        
        # SSA join for if statement (line 149)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'draw_all(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_all' in the type store
        # Getting the type of 'stypy_return_type' (line 142)
        stypy_return_type_188162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188162)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_all'
        return stypy_return_type_188162


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 17, 0, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Gcf.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Gcf' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'Gcf', Gcf)

# Assigning a List to a Name (line 35):

# Obtaining an instance of the builtin type 'list' (line 35)
list_188163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 35)

# Getting the type of 'Gcf'
Gcf_188164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Gcf')
# Setting the type of the member '_activeQue' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Gcf_188164, '_activeQue', list_188163)

# Assigning a Dict to a Name (line 36):

# Obtaining an instance of the builtin type 'dict' (line 36)
dict_188165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 36)

# Getting the type of 'Gcf'
Gcf_188166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Gcf')
# Setting the type of the member 'figs' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Gcf_188166, 'figs', dict_188165)

# Call to register(...): (line 152)
# Processing the call arguments (line 152)
# Getting the type of 'Gcf' (line 152)
Gcf_188169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'Gcf', False)
# Obtaining the member 'destroy_all' of a type (line 152)
destroy_all_188170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 16), Gcf_188169, 'destroy_all')
# Processing the call keyword arguments (line 152)
kwargs_188171 = {}
# Getting the type of 'atexit' (line 152)
atexit_188167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'atexit', False)
# Obtaining the member 'register' of a type (line 152)
register_188168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 0), atexit_188167, 'register')
# Calling register(args, kwargs) (line 152)
register_call_result_188172 = invoke(stypy.reporting.localization.Localization(__file__, 152, 0), register_188168, *[destroy_all_188170], **kwargs_188171)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
