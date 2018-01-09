
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets
2: 
3: 
4: class UiSubplotTool(QtWidgets.QDialog):
5: 
6:     def __init__(self, *args, **kwargs):
7:         super(UiSubplotTool, self).__init__(*args, **kwargs)
8:         self.setObjectName("SubplotTool")
9:         self._widgets = {}
10: 
11:         layout = QtWidgets.QHBoxLayout()
12:         self.setLayout(layout)
13: 
14:         left = QtWidgets.QVBoxLayout()
15:         layout.addLayout(left)
16:         right = QtWidgets.QVBoxLayout()
17:         layout.addLayout(right)
18: 
19:         box = QtWidgets.QGroupBox("Borders")
20:         left.addWidget(box)
21:         inner = QtWidgets.QFormLayout(box)
22:         for side in ["top", "bottom", "left", "right"]:
23:             self._widgets[side] = widget = QtWidgets.QDoubleSpinBox()
24:             widget.setMinimum(0)
25:             widget.setMaximum(1)
26:             widget.setDecimals(3)
27:             widget.setSingleStep(.005)
28:             widget.setKeyboardTracking(False)
29:             inner.addRow(side, widget)
30:         left.addStretch(1)
31: 
32:         box = QtWidgets.QGroupBox("Spacings")
33:         right.addWidget(box)
34:         inner = QtWidgets.QFormLayout(box)
35:         for side in ["hspace", "wspace"]:
36:             self._widgets[side] = widget = QtWidgets.QDoubleSpinBox()
37:             widget.setMinimum(0)
38:             widget.setMaximum(1)
39:             widget.setDecimals(3)
40:             widget.setSingleStep(.005)
41:             widget.setKeyboardTracking(False)
42:             inner.addRow(side, widget)
43:         right.addStretch(1)
44: 
45:         widget = QtWidgets.QPushButton("Export values")
46:         self._widgets["Export values"] = widget
47:         # Don't trigger on <enter>, which is used to input values.
48:         widget.setAutoDefault(False)
49:         left.addWidget(widget)
50: 
51:         for action in ["Tight layout", "Reset", "Close"]:
52:             self._widgets[action] = widget = QtWidgets.QPushButton(action)
53:             widget.setAutoDefault(False)
54:             right.addWidget(widget)
55: 
56:         self._widgets["Close"].setFocus()
57: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets' statement (line 1)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/qt_editor/')
import_272926 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'matplotlib.backends.qt_compat')

if (type(import_272926) is not StypyTypeError):

    if (import_272926 != 'pyd_module'):
        __import__(import_272926)
        sys_modules_272927 = sys.modules[import_272926]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'matplotlib.backends.qt_compat', sys_modules_272927.module_type_store, module_type_store, ['QtCore', 'QtGui', 'QtWidgets'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_272927, sys_modules_272927.module_type_store, module_type_store)
    else:
        from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'matplotlib.backends.qt_compat', None, module_type_store, ['QtCore', 'QtGui', 'QtWidgets'], [QtCore, QtGui, QtWidgets])

else:
    # Assigning a type to the variable 'matplotlib.backends.qt_compat' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'matplotlib.backends.qt_compat', import_272926)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/qt_editor/')

# Declaration of the 'UiSubplotTool' class
# Getting the type of 'QtWidgets' (line 4)
QtWidgets_272928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 20), 'QtWidgets')
# Obtaining the member 'QDialog' of a type (line 4)
QDialog_272929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 20), QtWidgets_272928, 'QDialog')

class UiSubplotTool(QDialog_272929, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 6, 4, False)
        # Assigning a type to the variable 'self' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UiSubplotTool.__init__', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 7)
        # Getting the type of 'args' (line 7)
        args_272936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 45), 'args', False)
        # Processing the call keyword arguments (line 7)
        # Getting the type of 'kwargs' (line 7)
        kwargs_272937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 53), 'kwargs', False)
        kwargs_272938 = {'kwargs_272937': kwargs_272937}
        
        # Call to super(...): (line 7)
        # Processing the call arguments (line 7)
        # Getting the type of 'UiSubplotTool' (line 7)
        UiSubplotTool_272931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 14), 'UiSubplotTool', False)
        # Getting the type of 'self' (line 7)
        self_272932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 29), 'self', False)
        # Processing the call keyword arguments (line 7)
        kwargs_272933 = {}
        # Getting the type of 'super' (line 7)
        super_272930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'super', False)
        # Calling super(args, kwargs) (line 7)
        super_call_result_272934 = invoke(stypy.reporting.localization.Localization(__file__, 7, 8), super_272930, *[UiSubplotTool_272931, self_272932], **kwargs_272933)
        
        # Obtaining the member '__init__' of a type (line 7)
        init___272935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 8), super_call_result_272934, '__init__')
        # Calling __init__(args, kwargs) (line 7)
        init___call_result_272939 = invoke(stypy.reporting.localization.Localization(__file__, 7, 8), init___272935, *[args_272936], **kwargs_272938)
        
        
        # Call to setObjectName(...): (line 8)
        # Processing the call arguments (line 8)
        str_272942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 27), 'str', 'SubplotTool')
        # Processing the call keyword arguments (line 8)
        kwargs_272943 = {}
        # Getting the type of 'self' (line 8)
        self_272940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'self', False)
        # Obtaining the member 'setObjectName' of a type (line 8)
        setObjectName_272941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 8), self_272940, 'setObjectName')
        # Calling setObjectName(args, kwargs) (line 8)
        setObjectName_call_result_272944 = invoke(stypy.reporting.localization.Localization(__file__, 8, 8), setObjectName_272941, *[str_272942], **kwargs_272943)
        
        
        # Assigning a Dict to a Attribute (line 9):
        
        # Obtaining an instance of the builtin type 'dict' (line 9)
        dict_272945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 24), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 9)
        
        # Getting the type of 'self' (line 9)
        self_272946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'self')
        # Setting the type of the member '_widgets' of a type (line 9)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 8), self_272946, '_widgets', dict_272945)
        
        # Assigning a Call to a Name (line 11):
        
        # Call to QHBoxLayout(...): (line 11)
        # Processing the call keyword arguments (line 11)
        kwargs_272949 = {}
        # Getting the type of 'QtWidgets' (line 11)
        QtWidgets_272947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 17), 'QtWidgets', False)
        # Obtaining the member 'QHBoxLayout' of a type (line 11)
        QHBoxLayout_272948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 17), QtWidgets_272947, 'QHBoxLayout')
        # Calling QHBoxLayout(args, kwargs) (line 11)
        QHBoxLayout_call_result_272950 = invoke(stypy.reporting.localization.Localization(__file__, 11, 17), QHBoxLayout_272948, *[], **kwargs_272949)
        
        # Assigning a type to the variable 'layout' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'layout', QHBoxLayout_call_result_272950)
        
        # Call to setLayout(...): (line 12)
        # Processing the call arguments (line 12)
        # Getting the type of 'layout' (line 12)
        layout_272953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 23), 'layout', False)
        # Processing the call keyword arguments (line 12)
        kwargs_272954 = {}
        # Getting the type of 'self' (line 12)
        self_272951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'self', False)
        # Obtaining the member 'setLayout' of a type (line 12)
        setLayout_272952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), self_272951, 'setLayout')
        # Calling setLayout(args, kwargs) (line 12)
        setLayout_call_result_272955 = invoke(stypy.reporting.localization.Localization(__file__, 12, 8), setLayout_272952, *[layout_272953], **kwargs_272954)
        
        
        # Assigning a Call to a Name (line 14):
        
        # Call to QVBoxLayout(...): (line 14)
        # Processing the call keyword arguments (line 14)
        kwargs_272958 = {}
        # Getting the type of 'QtWidgets' (line 14)
        QtWidgets_272956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'QtWidgets', False)
        # Obtaining the member 'QVBoxLayout' of a type (line 14)
        QVBoxLayout_272957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 15), QtWidgets_272956, 'QVBoxLayout')
        # Calling QVBoxLayout(args, kwargs) (line 14)
        QVBoxLayout_call_result_272959 = invoke(stypy.reporting.localization.Localization(__file__, 14, 15), QVBoxLayout_272957, *[], **kwargs_272958)
        
        # Assigning a type to the variable 'left' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'left', QVBoxLayout_call_result_272959)
        
        # Call to addLayout(...): (line 15)
        # Processing the call arguments (line 15)
        # Getting the type of 'left' (line 15)
        left_272962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 25), 'left', False)
        # Processing the call keyword arguments (line 15)
        kwargs_272963 = {}
        # Getting the type of 'layout' (line 15)
        layout_272960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'layout', False)
        # Obtaining the member 'addLayout' of a type (line 15)
        addLayout_272961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), layout_272960, 'addLayout')
        # Calling addLayout(args, kwargs) (line 15)
        addLayout_call_result_272964 = invoke(stypy.reporting.localization.Localization(__file__, 15, 8), addLayout_272961, *[left_272962], **kwargs_272963)
        
        
        # Assigning a Call to a Name (line 16):
        
        # Call to QVBoxLayout(...): (line 16)
        # Processing the call keyword arguments (line 16)
        kwargs_272967 = {}
        # Getting the type of 'QtWidgets' (line 16)
        QtWidgets_272965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'QtWidgets', False)
        # Obtaining the member 'QVBoxLayout' of a type (line 16)
        QVBoxLayout_272966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 16), QtWidgets_272965, 'QVBoxLayout')
        # Calling QVBoxLayout(args, kwargs) (line 16)
        QVBoxLayout_call_result_272968 = invoke(stypy.reporting.localization.Localization(__file__, 16, 16), QVBoxLayout_272966, *[], **kwargs_272967)
        
        # Assigning a type to the variable 'right' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'right', QVBoxLayout_call_result_272968)
        
        # Call to addLayout(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'right' (line 17)
        right_272971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 25), 'right', False)
        # Processing the call keyword arguments (line 17)
        kwargs_272972 = {}
        # Getting the type of 'layout' (line 17)
        layout_272969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'layout', False)
        # Obtaining the member 'addLayout' of a type (line 17)
        addLayout_272970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), layout_272969, 'addLayout')
        # Calling addLayout(args, kwargs) (line 17)
        addLayout_call_result_272973 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), addLayout_272970, *[right_272971], **kwargs_272972)
        
        
        # Assigning a Call to a Name (line 19):
        
        # Call to QGroupBox(...): (line 19)
        # Processing the call arguments (line 19)
        str_272976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 34), 'str', 'Borders')
        # Processing the call keyword arguments (line 19)
        kwargs_272977 = {}
        # Getting the type of 'QtWidgets' (line 19)
        QtWidgets_272974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 14), 'QtWidgets', False)
        # Obtaining the member 'QGroupBox' of a type (line 19)
        QGroupBox_272975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 14), QtWidgets_272974, 'QGroupBox')
        # Calling QGroupBox(args, kwargs) (line 19)
        QGroupBox_call_result_272978 = invoke(stypy.reporting.localization.Localization(__file__, 19, 14), QGroupBox_272975, *[str_272976], **kwargs_272977)
        
        # Assigning a type to the variable 'box' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'box', QGroupBox_call_result_272978)
        
        # Call to addWidget(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'box' (line 20)
        box_272981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 23), 'box', False)
        # Processing the call keyword arguments (line 20)
        kwargs_272982 = {}
        # Getting the type of 'left' (line 20)
        left_272979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'left', False)
        # Obtaining the member 'addWidget' of a type (line 20)
        addWidget_272980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), left_272979, 'addWidget')
        # Calling addWidget(args, kwargs) (line 20)
        addWidget_call_result_272983 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), addWidget_272980, *[box_272981], **kwargs_272982)
        
        
        # Assigning a Call to a Name (line 21):
        
        # Call to QFormLayout(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'box' (line 21)
        box_272986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 38), 'box', False)
        # Processing the call keyword arguments (line 21)
        kwargs_272987 = {}
        # Getting the type of 'QtWidgets' (line 21)
        QtWidgets_272984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'QtWidgets', False)
        # Obtaining the member 'QFormLayout' of a type (line 21)
        QFormLayout_272985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 16), QtWidgets_272984, 'QFormLayout')
        # Calling QFormLayout(args, kwargs) (line 21)
        QFormLayout_call_result_272988 = invoke(stypy.reporting.localization.Localization(__file__, 21, 16), QFormLayout_272985, *[box_272986], **kwargs_272987)
        
        # Assigning a type to the variable 'inner' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'inner', QFormLayout_call_result_272988)
        
        
        # Obtaining an instance of the builtin type 'list' (line 22)
        list_272989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 22)
        # Adding element type (line 22)
        str_272990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'str', 'top')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 20), list_272989, str_272990)
        # Adding element type (line 22)
        str_272991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 28), 'str', 'bottom')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 20), list_272989, str_272991)
        # Adding element type (line 22)
        str_272992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 38), 'str', 'left')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 20), list_272989, str_272992)
        # Adding element type (line 22)
        str_272993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 46), 'str', 'right')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 20), list_272989, str_272993)
        
        # Testing the type of a for loop iterable (line 22)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 22, 8), list_272989)
        # Getting the type of the for loop variable (line 22)
        for_loop_var_272994 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 22, 8), list_272989)
        # Assigning a type to the variable 'side' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'side', for_loop_var_272994)
        # SSA begins for a for statement (line 22)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Multiple assignment of 2 elements.
        
        # Call to QDoubleSpinBox(...): (line 23)
        # Processing the call keyword arguments (line 23)
        kwargs_272997 = {}
        # Getting the type of 'QtWidgets' (line 23)
        QtWidgets_272995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 43), 'QtWidgets', False)
        # Obtaining the member 'QDoubleSpinBox' of a type (line 23)
        QDoubleSpinBox_272996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 43), QtWidgets_272995, 'QDoubleSpinBox')
        # Calling QDoubleSpinBox(args, kwargs) (line 23)
        QDoubleSpinBox_call_result_272998 = invoke(stypy.reporting.localization.Localization(__file__, 23, 43), QDoubleSpinBox_272996, *[], **kwargs_272997)
        
        # Assigning a type to the variable 'widget' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 34), 'widget', QDoubleSpinBox_call_result_272998)
        # Getting the type of 'widget' (line 23)
        widget_272999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 34), 'widget')
        # Getting the type of 'self' (line 23)
        self_273000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'self')
        # Obtaining the member '_widgets' of a type (line 23)
        _widgets_273001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 12), self_273000, '_widgets')
        # Getting the type of 'side' (line 23)
        side_273002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 26), 'side')
        # Storing an element on a container (line 23)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 12), _widgets_273001, (side_273002, widget_272999))
        
        # Call to setMinimum(...): (line 24)
        # Processing the call arguments (line 24)
        int_273005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 30), 'int')
        # Processing the call keyword arguments (line 24)
        kwargs_273006 = {}
        # Getting the type of 'widget' (line 24)
        widget_273003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'widget', False)
        # Obtaining the member 'setMinimum' of a type (line 24)
        setMinimum_273004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 12), widget_273003, 'setMinimum')
        # Calling setMinimum(args, kwargs) (line 24)
        setMinimum_call_result_273007 = invoke(stypy.reporting.localization.Localization(__file__, 24, 12), setMinimum_273004, *[int_273005], **kwargs_273006)
        
        
        # Call to setMaximum(...): (line 25)
        # Processing the call arguments (line 25)
        int_273010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 30), 'int')
        # Processing the call keyword arguments (line 25)
        kwargs_273011 = {}
        # Getting the type of 'widget' (line 25)
        widget_273008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'widget', False)
        # Obtaining the member 'setMaximum' of a type (line 25)
        setMaximum_273009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 12), widget_273008, 'setMaximum')
        # Calling setMaximum(args, kwargs) (line 25)
        setMaximum_call_result_273012 = invoke(stypy.reporting.localization.Localization(__file__, 25, 12), setMaximum_273009, *[int_273010], **kwargs_273011)
        
        
        # Call to setDecimals(...): (line 26)
        # Processing the call arguments (line 26)
        int_273015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 31), 'int')
        # Processing the call keyword arguments (line 26)
        kwargs_273016 = {}
        # Getting the type of 'widget' (line 26)
        widget_273013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'widget', False)
        # Obtaining the member 'setDecimals' of a type (line 26)
        setDecimals_273014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 12), widget_273013, 'setDecimals')
        # Calling setDecimals(args, kwargs) (line 26)
        setDecimals_call_result_273017 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), setDecimals_273014, *[int_273015], **kwargs_273016)
        
        
        # Call to setSingleStep(...): (line 27)
        # Processing the call arguments (line 27)
        float_273020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 33), 'float')
        # Processing the call keyword arguments (line 27)
        kwargs_273021 = {}
        # Getting the type of 'widget' (line 27)
        widget_273018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'widget', False)
        # Obtaining the member 'setSingleStep' of a type (line 27)
        setSingleStep_273019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), widget_273018, 'setSingleStep')
        # Calling setSingleStep(args, kwargs) (line 27)
        setSingleStep_call_result_273022 = invoke(stypy.reporting.localization.Localization(__file__, 27, 12), setSingleStep_273019, *[float_273020], **kwargs_273021)
        
        
        # Call to setKeyboardTracking(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'False' (line 28)
        False_273025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 39), 'False', False)
        # Processing the call keyword arguments (line 28)
        kwargs_273026 = {}
        # Getting the type of 'widget' (line 28)
        widget_273023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'widget', False)
        # Obtaining the member 'setKeyboardTracking' of a type (line 28)
        setKeyboardTracking_273024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 12), widget_273023, 'setKeyboardTracking')
        # Calling setKeyboardTracking(args, kwargs) (line 28)
        setKeyboardTracking_call_result_273027 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), setKeyboardTracking_273024, *[False_273025], **kwargs_273026)
        
        
        # Call to addRow(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'side' (line 29)
        side_273030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 25), 'side', False)
        # Getting the type of 'widget' (line 29)
        widget_273031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 31), 'widget', False)
        # Processing the call keyword arguments (line 29)
        kwargs_273032 = {}
        # Getting the type of 'inner' (line 29)
        inner_273028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'inner', False)
        # Obtaining the member 'addRow' of a type (line 29)
        addRow_273029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 12), inner_273028, 'addRow')
        # Calling addRow(args, kwargs) (line 29)
        addRow_call_result_273033 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), addRow_273029, *[side_273030, widget_273031], **kwargs_273032)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to addStretch(...): (line 30)
        # Processing the call arguments (line 30)
        int_273036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 24), 'int')
        # Processing the call keyword arguments (line 30)
        kwargs_273037 = {}
        # Getting the type of 'left' (line 30)
        left_273034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'left', False)
        # Obtaining the member 'addStretch' of a type (line 30)
        addStretch_273035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), left_273034, 'addStretch')
        # Calling addStretch(args, kwargs) (line 30)
        addStretch_call_result_273038 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), addStretch_273035, *[int_273036], **kwargs_273037)
        
        
        # Assigning a Call to a Name (line 32):
        
        # Call to QGroupBox(...): (line 32)
        # Processing the call arguments (line 32)
        str_273041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 34), 'str', 'Spacings')
        # Processing the call keyword arguments (line 32)
        kwargs_273042 = {}
        # Getting the type of 'QtWidgets' (line 32)
        QtWidgets_273039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 14), 'QtWidgets', False)
        # Obtaining the member 'QGroupBox' of a type (line 32)
        QGroupBox_273040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 14), QtWidgets_273039, 'QGroupBox')
        # Calling QGroupBox(args, kwargs) (line 32)
        QGroupBox_call_result_273043 = invoke(stypy.reporting.localization.Localization(__file__, 32, 14), QGroupBox_273040, *[str_273041], **kwargs_273042)
        
        # Assigning a type to the variable 'box' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'box', QGroupBox_call_result_273043)
        
        # Call to addWidget(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'box' (line 33)
        box_273046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 24), 'box', False)
        # Processing the call keyword arguments (line 33)
        kwargs_273047 = {}
        # Getting the type of 'right' (line 33)
        right_273044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'right', False)
        # Obtaining the member 'addWidget' of a type (line 33)
        addWidget_273045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), right_273044, 'addWidget')
        # Calling addWidget(args, kwargs) (line 33)
        addWidget_call_result_273048 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), addWidget_273045, *[box_273046], **kwargs_273047)
        
        
        # Assigning a Call to a Name (line 34):
        
        # Call to QFormLayout(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'box' (line 34)
        box_273051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 38), 'box', False)
        # Processing the call keyword arguments (line 34)
        kwargs_273052 = {}
        # Getting the type of 'QtWidgets' (line 34)
        QtWidgets_273049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'QtWidgets', False)
        # Obtaining the member 'QFormLayout' of a type (line 34)
        QFormLayout_273050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 16), QtWidgets_273049, 'QFormLayout')
        # Calling QFormLayout(args, kwargs) (line 34)
        QFormLayout_call_result_273053 = invoke(stypy.reporting.localization.Localization(__file__, 34, 16), QFormLayout_273050, *[box_273051], **kwargs_273052)
        
        # Assigning a type to the variable 'inner' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'inner', QFormLayout_call_result_273053)
        
        
        # Obtaining an instance of the builtin type 'list' (line 35)
        list_273054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 35)
        # Adding element type (line 35)
        str_273055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 21), 'str', 'hspace')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 20), list_273054, str_273055)
        # Adding element type (line 35)
        str_273056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 31), 'str', 'wspace')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 20), list_273054, str_273056)
        
        # Testing the type of a for loop iterable (line 35)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 35, 8), list_273054)
        # Getting the type of the for loop variable (line 35)
        for_loop_var_273057 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 35, 8), list_273054)
        # Assigning a type to the variable 'side' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'side', for_loop_var_273057)
        # SSA begins for a for statement (line 35)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Multiple assignment of 2 elements.
        
        # Call to QDoubleSpinBox(...): (line 36)
        # Processing the call keyword arguments (line 36)
        kwargs_273060 = {}
        # Getting the type of 'QtWidgets' (line 36)
        QtWidgets_273058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 43), 'QtWidgets', False)
        # Obtaining the member 'QDoubleSpinBox' of a type (line 36)
        QDoubleSpinBox_273059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 43), QtWidgets_273058, 'QDoubleSpinBox')
        # Calling QDoubleSpinBox(args, kwargs) (line 36)
        QDoubleSpinBox_call_result_273061 = invoke(stypy.reporting.localization.Localization(__file__, 36, 43), QDoubleSpinBox_273059, *[], **kwargs_273060)
        
        # Assigning a type to the variable 'widget' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 34), 'widget', QDoubleSpinBox_call_result_273061)
        # Getting the type of 'widget' (line 36)
        widget_273062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 34), 'widget')
        # Getting the type of 'self' (line 36)
        self_273063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'self')
        # Obtaining the member '_widgets' of a type (line 36)
        _widgets_273064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), self_273063, '_widgets')
        # Getting the type of 'side' (line 36)
        side_273065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 26), 'side')
        # Storing an element on a container (line 36)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 12), _widgets_273064, (side_273065, widget_273062))
        
        # Call to setMinimum(...): (line 37)
        # Processing the call arguments (line 37)
        int_273068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 30), 'int')
        # Processing the call keyword arguments (line 37)
        kwargs_273069 = {}
        # Getting the type of 'widget' (line 37)
        widget_273066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'widget', False)
        # Obtaining the member 'setMinimum' of a type (line 37)
        setMinimum_273067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 12), widget_273066, 'setMinimum')
        # Calling setMinimum(args, kwargs) (line 37)
        setMinimum_call_result_273070 = invoke(stypy.reporting.localization.Localization(__file__, 37, 12), setMinimum_273067, *[int_273068], **kwargs_273069)
        
        
        # Call to setMaximum(...): (line 38)
        # Processing the call arguments (line 38)
        int_273073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 30), 'int')
        # Processing the call keyword arguments (line 38)
        kwargs_273074 = {}
        # Getting the type of 'widget' (line 38)
        widget_273071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'widget', False)
        # Obtaining the member 'setMaximum' of a type (line 38)
        setMaximum_273072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 12), widget_273071, 'setMaximum')
        # Calling setMaximum(args, kwargs) (line 38)
        setMaximum_call_result_273075 = invoke(stypy.reporting.localization.Localization(__file__, 38, 12), setMaximum_273072, *[int_273073], **kwargs_273074)
        
        
        # Call to setDecimals(...): (line 39)
        # Processing the call arguments (line 39)
        int_273078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 31), 'int')
        # Processing the call keyword arguments (line 39)
        kwargs_273079 = {}
        # Getting the type of 'widget' (line 39)
        widget_273076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'widget', False)
        # Obtaining the member 'setDecimals' of a type (line 39)
        setDecimals_273077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 12), widget_273076, 'setDecimals')
        # Calling setDecimals(args, kwargs) (line 39)
        setDecimals_call_result_273080 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), setDecimals_273077, *[int_273078], **kwargs_273079)
        
        
        # Call to setSingleStep(...): (line 40)
        # Processing the call arguments (line 40)
        float_273083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 33), 'float')
        # Processing the call keyword arguments (line 40)
        kwargs_273084 = {}
        # Getting the type of 'widget' (line 40)
        widget_273081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'widget', False)
        # Obtaining the member 'setSingleStep' of a type (line 40)
        setSingleStep_273082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 12), widget_273081, 'setSingleStep')
        # Calling setSingleStep(args, kwargs) (line 40)
        setSingleStep_call_result_273085 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), setSingleStep_273082, *[float_273083], **kwargs_273084)
        
        
        # Call to setKeyboardTracking(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'False' (line 41)
        False_273088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 39), 'False', False)
        # Processing the call keyword arguments (line 41)
        kwargs_273089 = {}
        # Getting the type of 'widget' (line 41)
        widget_273086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'widget', False)
        # Obtaining the member 'setKeyboardTracking' of a type (line 41)
        setKeyboardTracking_273087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), widget_273086, 'setKeyboardTracking')
        # Calling setKeyboardTracking(args, kwargs) (line 41)
        setKeyboardTracking_call_result_273090 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), setKeyboardTracking_273087, *[False_273088], **kwargs_273089)
        
        
        # Call to addRow(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'side' (line 42)
        side_273093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 25), 'side', False)
        # Getting the type of 'widget' (line 42)
        widget_273094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 31), 'widget', False)
        # Processing the call keyword arguments (line 42)
        kwargs_273095 = {}
        # Getting the type of 'inner' (line 42)
        inner_273091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'inner', False)
        # Obtaining the member 'addRow' of a type (line 42)
        addRow_273092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 12), inner_273091, 'addRow')
        # Calling addRow(args, kwargs) (line 42)
        addRow_call_result_273096 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), addRow_273092, *[side_273093, widget_273094], **kwargs_273095)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to addStretch(...): (line 43)
        # Processing the call arguments (line 43)
        int_273099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 25), 'int')
        # Processing the call keyword arguments (line 43)
        kwargs_273100 = {}
        # Getting the type of 'right' (line 43)
        right_273097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'right', False)
        # Obtaining the member 'addStretch' of a type (line 43)
        addStretch_273098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), right_273097, 'addStretch')
        # Calling addStretch(args, kwargs) (line 43)
        addStretch_call_result_273101 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), addStretch_273098, *[int_273099], **kwargs_273100)
        
        
        # Assigning a Call to a Name (line 45):
        
        # Call to QPushButton(...): (line 45)
        # Processing the call arguments (line 45)
        str_273104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 39), 'str', 'Export values')
        # Processing the call keyword arguments (line 45)
        kwargs_273105 = {}
        # Getting the type of 'QtWidgets' (line 45)
        QtWidgets_273102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 17), 'QtWidgets', False)
        # Obtaining the member 'QPushButton' of a type (line 45)
        QPushButton_273103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 17), QtWidgets_273102, 'QPushButton')
        # Calling QPushButton(args, kwargs) (line 45)
        QPushButton_call_result_273106 = invoke(stypy.reporting.localization.Localization(__file__, 45, 17), QPushButton_273103, *[str_273104], **kwargs_273105)
        
        # Assigning a type to the variable 'widget' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'widget', QPushButton_call_result_273106)
        
        # Assigning a Name to a Subscript (line 46):
        # Getting the type of 'widget' (line 46)
        widget_273107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 41), 'widget')
        # Getting the type of 'self' (line 46)
        self_273108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self')
        # Obtaining the member '_widgets' of a type (line 46)
        _widgets_273109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_273108, '_widgets')
        str_273110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'str', 'Export values')
        # Storing an element on a container (line 46)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 8), _widgets_273109, (str_273110, widget_273107))
        
        # Call to setAutoDefault(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'False' (line 48)
        False_273113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 30), 'False', False)
        # Processing the call keyword arguments (line 48)
        kwargs_273114 = {}
        # Getting the type of 'widget' (line 48)
        widget_273111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'widget', False)
        # Obtaining the member 'setAutoDefault' of a type (line 48)
        setAutoDefault_273112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), widget_273111, 'setAutoDefault')
        # Calling setAutoDefault(args, kwargs) (line 48)
        setAutoDefault_call_result_273115 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), setAutoDefault_273112, *[False_273113], **kwargs_273114)
        
        
        # Call to addWidget(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'widget' (line 49)
        widget_273118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 23), 'widget', False)
        # Processing the call keyword arguments (line 49)
        kwargs_273119 = {}
        # Getting the type of 'left' (line 49)
        left_273116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'left', False)
        # Obtaining the member 'addWidget' of a type (line 49)
        addWidget_273117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), left_273116, 'addWidget')
        # Calling addWidget(args, kwargs) (line 49)
        addWidget_call_result_273120 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), addWidget_273117, *[widget_273118], **kwargs_273119)
        
        
        
        # Obtaining an instance of the builtin type 'list' (line 51)
        list_273121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 51)
        # Adding element type (line 51)
        str_273122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 23), 'str', 'Tight layout')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 22), list_273121, str_273122)
        # Adding element type (line 51)
        str_273123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 39), 'str', 'Reset')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 22), list_273121, str_273123)
        # Adding element type (line 51)
        str_273124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 48), 'str', 'Close')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 22), list_273121, str_273124)
        
        # Testing the type of a for loop iterable (line 51)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 51, 8), list_273121)
        # Getting the type of the for loop variable (line 51)
        for_loop_var_273125 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 51, 8), list_273121)
        # Assigning a type to the variable 'action' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'action', for_loop_var_273125)
        # SSA begins for a for statement (line 51)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Multiple assignment of 2 elements.
        
        # Call to QPushButton(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'action' (line 52)
        action_273128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 67), 'action', False)
        # Processing the call keyword arguments (line 52)
        kwargs_273129 = {}
        # Getting the type of 'QtWidgets' (line 52)
        QtWidgets_273126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 45), 'QtWidgets', False)
        # Obtaining the member 'QPushButton' of a type (line 52)
        QPushButton_273127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 45), QtWidgets_273126, 'QPushButton')
        # Calling QPushButton(args, kwargs) (line 52)
        QPushButton_call_result_273130 = invoke(stypy.reporting.localization.Localization(__file__, 52, 45), QPushButton_273127, *[action_273128], **kwargs_273129)
        
        # Assigning a type to the variable 'widget' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 36), 'widget', QPushButton_call_result_273130)
        # Getting the type of 'widget' (line 52)
        widget_273131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 36), 'widget')
        # Getting the type of 'self' (line 52)
        self_273132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'self')
        # Obtaining the member '_widgets' of a type (line 52)
        _widgets_273133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), self_273132, '_widgets')
        # Getting the type of 'action' (line 52)
        action_273134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 26), 'action')
        # Storing an element on a container (line 52)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 12), _widgets_273133, (action_273134, widget_273131))
        
        # Call to setAutoDefault(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'False' (line 53)
        False_273137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 34), 'False', False)
        # Processing the call keyword arguments (line 53)
        kwargs_273138 = {}
        # Getting the type of 'widget' (line 53)
        widget_273135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'widget', False)
        # Obtaining the member 'setAutoDefault' of a type (line 53)
        setAutoDefault_273136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 12), widget_273135, 'setAutoDefault')
        # Calling setAutoDefault(args, kwargs) (line 53)
        setAutoDefault_call_result_273139 = invoke(stypy.reporting.localization.Localization(__file__, 53, 12), setAutoDefault_273136, *[False_273137], **kwargs_273138)
        
        
        # Call to addWidget(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'widget' (line 54)
        widget_273142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'widget', False)
        # Processing the call keyword arguments (line 54)
        kwargs_273143 = {}
        # Getting the type of 'right' (line 54)
        right_273140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'right', False)
        # Obtaining the member 'addWidget' of a type (line 54)
        addWidget_273141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), right_273140, 'addWidget')
        # Calling addWidget(args, kwargs) (line 54)
        addWidget_call_result_273144 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), addWidget_273141, *[widget_273142], **kwargs_273143)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to setFocus(...): (line 56)
        # Processing the call keyword arguments (line 56)
        kwargs_273151 = {}
        
        # Obtaining the type of the subscript
        str_273145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 22), 'str', 'Close')
        # Getting the type of 'self' (line 56)
        self_273146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self', False)
        # Obtaining the member '_widgets' of a type (line 56)
        _widgets_273147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_273146, '_widgets')
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___273148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), _widgets_273147, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_273149 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), getitem___273148, str_273145)
        
        # Obtaining the member 'setFocus' of a type (line 56)
        setFocus_273150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), subscript_call_result_273149, 'setFocus')
        # Calling setFocus(args, kwargs) (line 56)
        setFocus_call_result_273152 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), setFocus_273150, *[], **kwargs_273151)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'UiSubplotTool' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'UiSubplotTool', UiSubplotTool)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
