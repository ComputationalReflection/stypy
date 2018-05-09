
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import datetime
2: import inspect
3: 
4: from stypy_copy import stypy_parameters_copy
5: 
6: output_to_console = True
7: 
8: '''
9:  Multiplatform terminal color messages to improve visual quality of the output
10:  Also handles message logging for stypy.
11:  This code has been adapted from tcaswell snippet, found in:
12:  http://stackoverflow.com/questions/2654113/python-how-to-get-the-callers-method-name-in-the-called-method
13: '''
14: 
15: 
16: def get_caller_data(skip=2):
17:     '''Get a name of a caller in the format module.class.method
18: 
19:        `skip` specifies how many levels of stack to skip while getting caller
20:        name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.
21: 
22:        An empty string is returned if skipped levels exceed stack height
23:     '''
24:     stack = inspect.stack()
25:     start = 0 + skip
26:     if len(stack) < start + 1:
27:         return ''
28:     parentframe = stack[start][0]
29: 
30:     name = []
31:     module = inspect.getmodule(parentframe)
32: 
33:     if module:
34:         # Name of the file (at the end of the path, removing the c of the .pyc extension
35:         name.append(module.__file__.split("\\")[-1])
36: 
37:     # detect classname
38:     if 'self' in parentframe.f_locals:
39:         name.append(parentframe.f_locals['self'].__class__.__name__)
40: 
41:     codename = parentframe.f_code.co_name
42: 
43:     if codename != '<module>':  # top level usually
44:         name.append(codename)  # function or a method
45:     del parentframe
46: 
47:     # Strip full file path
48:     name[0] = name[0].split("/")[-1]
49:     return str(name)
50: 
51: 
52: class ColorType:
53:     ANSIColors = False
54: 
55: 
56: try:
57:     import ctypes
58: 
59: 
60:     def setup_handles():
61:         '''
62:         Determines if it is possible to have colored output
63:         :return:
64:         '''
65:         # Constants from the Windows API
66:         STD_OUTPUT_HANDLE = -11
67: 
68:         def get_csbi_attributes(handle):
69:             # Based on IPython's winconsole.py, written by Alexander Belchenko
70:             import struct
71: 
72:             csbi = ctypes.create_string_buffer(22)
73:             res = ctypes.windll.kernel32.GetConsoleScreenBufferInfo(handle, csbi)
74:             # assert res
75: 
76:             (bufx, bufy, curx, cury, wattr,
77:              left, top, right, bottom, maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
78:             return wattr
79: 
80:         handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
81:         reset = get_csbi_attributes(handle)
82:         return handle, reset
83: 
84: 
85:     ColorType.ANSIColors = False  # Windows do not support ANSI terminals
86: 
87: except Exception as e:
88:     ColorType.ANSIColors = True  # ANSI escape sequences work with other terminals
89: 
90: 
91: class Colors:
92:     ANSI_BLUE = '\033[94m'
93:     ANSI_GREEN = '\033[92m'
94:     ANSI_WARNING = '\033[93m'
95:     ANSI_FAIL = '\033[91m'
96:     ANSI_ENDC = '\033[0m'
97: 
98:     WIN_BLUE = 0x0009
99:     WIN_WHITE = 0x000F
100:     WIN_GREEN = 0x000A
101:     WIN_WARNING = 0x000E
102:     WIN_FAIL = 0x000C
103: 
104: 
105: def get_date_time():
106:     '''
107:     Obtains current date and time
108:     :return:
109:     '''
110:     return str(datetime.datetime.now())[:-7]
111: 
112: 
113: def log(msg):
114:     '''
115:     Logs information messages to the corresponding log file
116:     :param msg:
117:     :return:
118:     '''
119:     try:
120:         file_ = open(stypy_parameters_copy.LOG_PATH + "/" + stypy_parameters_copy.INFO_LOG_FILE, "a")
121:     except:
122:         return  # No log is not a critical error condition
123: 
124:     if (msg == "\n") or (msg == ""):
125:         if msg == "":
126:             file_.write(msg + "\n")
127:         else:
128:             file_.write(msg)
129:     else:
130:         if not (file_ is None):
131:             file_.write("[" + get_date_time() + "] " + msg + "\n")
132: 
133:     file_.close()
134: 
135: 
136: def ok(msg):
137:     '''
138:     Handles green log information messages
139:     :param msg:
140:     :return:
141:     '''
142:     txt = get_caller_data() + ": " + msg
143:     if ColorType.ANSIColors:
144:         if output_to_console:
145:             print(Colors.ANSI_GREEN + msg + Colors.ANSI_ENDC)
146:     else:
147:         handle, reset = setup_handles()
148:         ctypes.windll.kernel32.SetConsoleTextAttribute(handle, Colors.WIN_GREEN)
149: 
150:         if output_to_console:
151:             print (msg)
152: 
153:         ctypes.windll.kernel32.SetConsoleTextAttribute(handle, reset)
154: 
155:     log(txt)
156: 
157: 
158: def info(msg):
159:     '''
160:     Handles white log information messages
161:     :param msg:
162:     :return:
163:     '''
164:     txt = get_caller_data() + ": " + msg
165:     if ColorType.ANSIColors:
166:         if output_to_console:
167:             print(txt)
168:     else:
169:         handle, reset = setup_handles()
170:         ctypes.windll.kernel32.SetConsoleTextAttribute(handle, Colors.WIN_WHITE)
171: 
172:         if output_to_console:
173:             print (txt)
174: 
175:         ctypes.windll.kernel32.SetConsoleTextAttribute(handle, reset)
176: 
177:     log(txt)
178: 
179: 
180: def __aux_warning_and_error_write(msg, call_data, ansi_console_color, win_console_color, file_name, msg_type):
181:     '''
182:     Helper function to output warning or error messages, depending on its parameters.
183:     :param msg: Message to print
184:     :param call_data: Caller information
185:     :param ansi_console_color: ANSI terminals color to use
186:     :param win_console_color: Windows terminals color to use
187:     :param file_name: File to write to
188:     :param msg_type: Type of message to write (WARNING/ERROR)
189:     :return:
190:     '''
191:     if ColorType.ANSIColors:
192:         if output_to_console:
193:             txt = str(call_data) + ". " + msg_type + ": " + str(msg)
194:             print(ansi_console_color + txt + Colors.ANSI_ENDC)
195:     else:
196:         handle, reset = setup_handles()
197:         ctypes.windll.kernel32.SetConsoleTextAttribute(handle, win_console_color)
198: 
199:         if output_to_console:
200:             print(msg_type + ": " + msg)
201: 
202:         ctypes.windll.kernel32.SetConsoleTextAttribute(handle, reset)
203: 
204:     try:
205:         file_ = open(stypy_parameters_copy.LOG_PATH + "/" + file_name, "a")
206:     except:
207:         return  # No log is not a critical error condition
208: 
209:     txt = str(call_data) + " (" + get_date_time() + "). " + msg_type + ": " + msg
210:     file_.write(txt + "\n")
211:     file_.close()
212: 
213: 
214: def warning(msg):
215:     '''
216:     Proxy for __aux_warning_and_error_write, supplying parameters to write warning messages
217:     :param msg:
218:     :return:
219:     '''
220:     call_data = get_caller_data()
221:     __aux_warning_and_error_write(msg, call_data, Colors.ANSI_WARNING, Colors.WIN_WARNING,
222:                                   stypy_parameters_copy.WARNING_LOG_FILE, "WARNING")
223: 
224: 
225: def error(msg):
226:     '''
227:     Proxy for __aux_warning_and_error_write, supplying parameters to write error messages
228:     :param msg:
229:     :return:
230:     '''
231:     call_data = get_caller_data()
232:     __aux_warning_and_error_write(msg, call_data, Colors.ANSI_FAIL, Colors.WIN_FAIL, stypy_parameters_copy.ERROR_LOG_FILE,
233:                                   "ERROR")
234: 
235: 
236: def new_logging_session():
237:     '''
238:     Put a header to the log files indicating that log messages below that header belong to a new execution
239:     :return:
240:     '''
241:     try:
242:         file_ = open(stypy_parameters_copy.LOG_PATH + "/" + stypy_parameters_copy.ERROR_LOG_FILE, "a")
243:         file_.write("\n\n")
244:         file_.write("NEW LOGGING SESSION BEGIN AT: " + get_date_time())
245:         file_.write("\n\n")
246:         file_.close()
247: 
248:         file_ = open(stypy_parameters_copy.LOG_PATH + "/" + stypy_parameters_copy.INFO_LOG_FILE, "a")
249:         file_.write("\n\n")
250:         file_.write("NEW LOGGING SESSION BEGIN AT: " + get_date_time())
251:         file_.write("\n\n")
252:         file_.close()
253: 
254:         file_ = open(stypy_parameters_copy.LOG_PATH + "/" + stypy_parameters_copy.WARNING_LOG_FILE, "a")
255:         file_.write("\n\n")
256:         file_.write("NEW LOGGING SESSION BEGIN AT: " + get_date_time())
257:         file_.write("\n\n")
258:         file_.close()
259:     except:
260:         return
261: 
262: 
263: def reset_logs():
264:     '''
265:     Erases log files
266:     :return:
267:     '''
268:     try:
269:         file_ = open(stypy_parameters_copy.LOG_PATH + "/" + stypy_parameters_copy.ERROR_LOG_FILE, "w")
270:         file_.write("")
271:         file_.close()
272: 
273:         file_ = open(stypy_parameters_copy.LOG_PATH + "/" + stypy_parameters_copy.WARNING_LOG_FILE, "w")
274:         file_.write("")
275:         file_.close()
276: 
277:         file_ = open(stypy_parameters_copy.LOG_PATH + "/" + stypy_parameters_copy.INFO_LOG_FILE, "w")
278:         file_.write("")
279:         file_.close()
280:     except:
281:         return
282: 
283: 
284: def reset_colors():
285:     '''
286:     Reset Windows colors to leave the console with the default ones
287:     :return:
288:     '''
289:     if ColorType.ANSIColors:
290:         pass  # ANSI consoles do not need resets
291:     else:
292:         handle, reset = setup_handles()
293:         ctypes.windll.kernel32.SetConsoleTextAttribute(handle, reset)
294: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import datetime' statement (line 1)
import datetime

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'datetime', datetime, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import inspect' statement (line 2)
import inspect

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'inspect', inspect, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from stypy_copy import stypy_parameters_copy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/log_copy/')
import_3941 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy')

if (type(import_3941) is not StypyTypeError):

    if (import_3941 != 'pyd_module'):
        __import__(import_3941)
        sys_modules_3942 = sys.modules[import_3941]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy', sys_modules_3942.module_type_store, module_type_store, ['stypy_parameters_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_3942, sys_modules_3942.module_type_store, module_type_store)
    else:
        from stypy_copy import stypy_parameters_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy', None, module_type_store, ['stypy_parameters_copy'], [stypy_parameters_copy])

else:
    # Assigning a type to the variable 'stypy_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy', import_3941)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/log_copy/')


# Assigning a Name to a Name (line 6):

# Assigning a Name to a Name (line 6):
# Getting the type of 'True' (line 6)
True_3943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 20), 'True')
# Assigning a type to the variable 'output_to_console' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'output_to_console', True_3943)
str_3944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, (-1)), 'str', '\n Multiplatform terminal color messages to improve visual quality of the output\n Also handles message logging for stypy.\n This code has been adapted from tcaswell snippet, found in:\n http://stackoverflow.com/questions/2654113/python-how-to-get-the-callers-method-name-in-the-called-method\n')

@norecursion
def get_caller_data(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_3945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'int')
    defaults = [int_3945]
    # Create a new context for function 'get_caller_data'
    module_type_store = module_type_store.open_function_context('get_caller_data', 16, 0, False)
    
    # Passed parameters checking function
    get_caller_data.stypy_localization = localization
    get_caller_data.stypy_type_of_self = None
    get_caller_data.stypy_type_store = module_type_store
    get_caller_data.stypy_function_name = 'get_caller_data'
    get_caller_data.stypy_param_names_list = ['skip']
    get_caller_data.stypy_varargs_param_name = None
    get_caller_data.stypy_kwargs_param_name = None
    get_caller_data.stypy_call_defaults = defaults
    get_caller_data.stypy_call_varargs = varargs
    get_caller_data.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_caller_data', ['skip'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_caller_data', localization, ['skip'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_caller_data(...)' code ##################

    str_3946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, (-1)), 'str', 'Get a name of a caller in the format module.class.method\n\n       `skip` specifies how many levels of stack to skip while getting caller\n       name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.\n\n       An empty string is returned if skipped levels exceed stack height\n    ')
    
    # Assigning a Call to a Name (line 24):
    
    # Assigning a Call to a Name (line 24):
    
    # Call to stack(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_3949 = {}
    # Getting the type of 'inspect' (line 24)
    inspect_3947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'inspect', False)
    # Obtaining the member 'stack' of a type (line 24)
    stack_3948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 12), inspect_3947, 'stack')
    # Calling stack(args, kwargs) (line 24)
    stack_call_result_3950 = invoke(stypy.reporting.localization.Localization(__file__, 24, 12), stack_3948, *[], **kwargs_3949)
    
    # Assigning a type to the variable 'stack' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stack', stack_call_result_3950)
    
    # Assigning a BinOp to a Name (line 25):
    
    # Assigning a BinOp to a Name (line 25):
    int_3951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 12), 'int')
    # Getting the type of 'skip' (line 25)
    skip_3952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'skip')
    # Applying the binary operator '+' (line 25)
    result_add_3953 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 12), '+', int_3951, skip_3952)
    
    # Assigning a type to the variable 'start' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'start', result_add_3953)
    
    
    # Call to len(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'stack' (line 26)
    stack_3955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'stack', False)
    # Processing the call keyword arguments (line 26)
    kwargs_3956 = {}
    # Getting the type of 'len' (line 26)
    len_3954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 7), 'len', False)
    # Calling len(args, kwargs) (line 26)
    len_call_result_3957 = invoke(stypy.reporting.localization.Localization(__file__, 26, 7), len_3954, *[stack_3955], **kwargs_3956)
    
    # Getting the type of 'start' (line 26)
    start_3958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 20), 'start')
    int_3959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 28), 'int')
    # Applying the binary operator '+' (line 26)
    result_add_3960 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 20), '+', start_3958, int_3959)
    
    # Applying the binary operator '<' (line 26)
    result_lt_3961 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 7), '<', len_call_result_3957, result_add_3960)
    
    # Testing if the type of an if condition is none (line 26)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 26, 4), result_lt_3961):
        pass
    else:
        
        # Testing the type of an if condition (line 26)
        if_condition_3962 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 4), result_lt_3961)
        # Assigning a type to the variable 'if_condition_3962' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'if_condition_3962', if_condition_3962)
        # SSA begins for if statement (line 26)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_3963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'str', '')
        # Assigning a type to the variable 'stypy_return_type' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type', str_3963)
        # SSA join for if statement (line 26)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Subscript to a Name (line 28):
    
    # Assigning a Subscript to a Name (line 28):
    
    # Obtaining the type of the subscript
    int_3964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 31), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'start' (line 28)
    start_3965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'start')
    # Getting the type of 'stack' (line 28)
    stack_3966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), 'stack')
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___3967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 18), stack_3966, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_3968 = invoke(stypy.reporting.localization.Localization(__file__, 28, 18), getitem___3967, start_3965)
    
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___3969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 18), subscript_call_result_3968, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_3970 = invoke(stypy.reporting.localization.Localization(__file__, 28, 18), getitem___3969, int_3964)
    
    # Assigning a type to the variable 'parentframe' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'parentframe', subscript_call_result_3970)
    
    # Assigning a List to a Name (line 30):
    
    # Assigning a List to a Name (line 30):
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_3971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    
    # Assigning a type to the variable 'name' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'name', list_3971)
    
    # Assigning a Call to a Name (line 31):
    
    # Assigning a Call to a Name (line 31):
    
    # Call to getmodule(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'parentframe' (line 31)
    parentframe_3974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 31), 'parentframe', False)
    # Processing the call keyword arguments (line 31)
    kwargs_3975 = {}
    # Getting the type of 'inspect' (line 31)
    inspect_3972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 13), 'inspect', False)
    # Obtaining the member 'getmodule' of a type (line 31)
    getmodule_3973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 13), inspect_3972, 'getmodule')
    # Calling getmodule(args, kwargs) (line 31)
    getmodule_call_result_3976 = invoke(stypy.reporting.localization.Localization(__file__, 31, 13), getmodule_3973, *[parentframe_3974], **kwargs_3975)
    
    # Assigning a type to the variable 'module' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'module', getmodule_call_result_3976)
    # Getting the type of 'module' (line 33)
    module_3977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 7), 'module')
    # Testing if the type of an if condition is none (line 33)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 33, 4), module_3977):
        pass
    else:
        
        # Testing the type of an if condition (line 33)
        if_condition_3978 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 4), module_3977)
        # Assigning a type to the variable 'if_condition_3978' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'if_condition_3978', if_condition_3978)
        # SSA begins for if statement (line 33)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 35)
        # Processing the call arguments (line 35)
        
        # Obtaining the type of the subscript
        int_3981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 48), 'int')
        
        # Call to split(...): (line 35)
        # Processing the call arguments (line 35)
        str_3985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 42), 'str', '\\')
        # Processing the call keyword arguments (line 35)
        kwargs_3986 = {}
        # Getting the type of 'module' (line 35)
        module_3982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'module', False)
        # Obtaining the member '__file__' of a type (line 35)
        file___3983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 20), module_3982, '__file__')
        # Obtaining the member 'split' of a type (line 35)
        split_3984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 20), file___3983, 'split')
        # Calling split(args, kwargs) (line 35)
        split_call_result_3987 = invoke(stypy.reporting.localization.Localization(__file__, 35, 20), split_3984, *[str_3985], **kwargs_3986)
        
        # Obtaining the member '__getitem__' of a type (line 35)
        getitem___3988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 20), split_call_result_3987, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 35)
        subscript_call_result_3989 = invoke(stypy.reporting.localization.Localization(__file__, 35, 20), getitem___3988, int_3981)
        
        # Processing the call keyword arguments (line 35)
        kwargs_3990 = {}
        # Getting the type of 'name' (line 35)
        name_3979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'name', False)
        # Obtaining the member 'append' of a type (line 35)
        append_3980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), name_3979, 'append')
        # Calling append(args, kwargs) (line 35)
        append_call_result_3991 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), append_3980, *[subscript_call_result_3989], **kwargs_3990)
        
        # SSA join for if statement (line 33)
        module_type_store = module_type_store.join_ssa_context()
        

    
    str_3992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 7), 'str', 'self')
    # Getting the type of 'parentframe' (line 38)
    parentframe_3993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 17), 'parentframe')
    # Obtaining the member 'f_locals' of a type (line 38)
    f_locals_3994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 17), parentframe_3993, 'f_locals')
    # Applying the binary operator 'in' (line 38)
    result_contains_3995 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 7), 'in', str_3992, f_locals_3994)
    
    # Testing if the type of an if condition is none (line 38)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 38, 4), result_contains_3995):
        pass
    else:
        
        # Testing the type of an if condition (line 38)
        if_condition_3996 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 4), result_contains_3995)
        # Assigning a type to the variable 'if_condition_3996' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'if_condition_3996', if_condition_3996)
        # SSA begins for if statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Obtaining the type of the subscript
        str_3999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 41), 'str', 'self')
        # Getting the type of 'parentframe' (line 39)
        parentframe_4000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'parentframe', False)
        # Obtaining the member 'f_locals' of a type (line 39)
        f_locals_4001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 20), parentframe_4000, 'f_locals')
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___4002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 20), f_locals_4001, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_4003 = invoke(stypy.reporting.localization.Localization(__file__, 39, 20), getitem___4002, str_3999)
        
        # Obtaining the member '__class__' of a type (line 39)
        class___4004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 20), subscript_call_result_4003, '__class__')
        # Obtaining the member '__name__' of a type (line 39)
        name___4005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 20), class___4004, '__name__')
        # Processing the call keyword arguments (line 39)
        kwargs_4006 = {}
        # Getting the type of 'name' (line 39)
        name_3997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'name', False)
        # Obtaining the member 'append' of a type (line 39)
        append_3998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), name_3997, 'append')
        # Calling append(args, kwargs) (line 39)
        append_call_result_4007 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), append_3998, *[name___4005], **kwargs_4006)
        
        # SSA join for if statement (line 38)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Attribute to a Name (line 41):
    
    # Assigning a Attribute to a Name (line 41):
    # Getting the type of 'parentframe' (line 41)
    parentframe_4008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'parentframe')
    # Obtaining the member 'f_code' of a type (line 41)
    f_code_4009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 15), parentframe_4008, 'f_code')
    # Obtaining the member 'co_name' of a type (line 41)
    co_name_4010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 15), f_code_4009, 'co_name')
    # Assigning a type to the variable 'codename' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'codename', co_name_4010)
    
    # Getting the type of 'codename' (line 43)
    codename_4011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 7), 'codename')
    str_4012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'str', '<module>')
    # Applying the binary operator '!=' (line 43)
    result_ne_4013 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 7), '!=', codename_4011, str_4012)
    
    # Testing if the type of an if condition is none (line 43)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 43, 4), result_ne_4013):
        pass
    else:
        
        # Testing the type of an if condition (line 43)
        if_condition_4014 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 4), result_ne_4013)
        # Assigning a type to the variable 'if_condition_4014' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'if_condition_4014', if_condition_4014)
        # SSA begins for if statement (line 43)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'codename' (line 44)
        codename_4017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'codename', False)
        # Processing the call keyword arguments (line 44)
        kwargs_4018 = {}
        # Getting the type of 'name' (line 44)
        name_4015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'name', False)
        # Obtaining the member 'append' of a type (line 44)
        append_4016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), name_4015, 'append')
        # Calling append(args, kwargs) (line 44)
        append_call_result_4019 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), append_4016, *[codename_4017], **kwargs_4018)
        
        # SSA join for if statement (line 43)
        module_type_store = module_type_store.join_ssa_context()
        

    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 45, 4), module_type_store, 'parentframe')
    
    # Assigning a Subscript to a Subscript (line 48):
    
    # Assigning a Subscript to a Subscript (line 48):
    
    # Obtaining the type of the subscript
    int_4020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 33), 'int')
    
    # Call to split(...): (line 48)
    # Processing the call arguments (line 48)
    str_4026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 28), 'str', '/')
    # Processing the call keyword arguments (line 48)
    kwargs_4027 = {}
    
    # Obtaining the type of the subscript
    int_4021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 19), 'int')
    # Getting the type of 'name' (line 48)
    name_4022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 14), 'name', False)
    # Obtaining the member '__getitem__' of a type (line 48)
    getitem___4023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 14), name_4022, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 48)
    subscript_call_result_4024 = invoke(stypy.reporting.localization.Localization(__file__, 48, 14), getitem___4023, int_4021)
    
    # Obtaining the member 'split' of a type (line 48)
    split_4025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 14), subscript_call_result_4024, 'split')
    # Calling split(args, kwargs) (line 48)
    split_call_result_4028 = invoke(stypy.reporting.localization.Localization(__file__, 48, 14), split_4025, *[str_4026], **kwargs_4027)
    
    # Obtaining the member '__getitem__' of a type (line 48)
    getitem___4029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 14), split_call_result_4028, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 48)
    subscript_call_result_4030 = invoke(stypy.reporting.localization.Localization(__file__, 48, 14), getitem___4029, int_4020)
    
    # Getting the type of 'name' (line 48)
    name_4031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'name')
    int_4032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 9), 'int')
    # Storing an element on a container (line 48)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 4), name_4031, (int_4032, subscript_call_result_4030))
    
    # Call to str(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'name' (line 49)
    name_4034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'name', False)
    # Processing the call keyword arguments (line 49)
    kwargs_4035 = {}
    # Getting the type of 'str' (line 49)
    str_4033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'str', False)
    # Calling str(args, kwargs) (line 49)
    str_call_result_4036 = invoke(stypy.reporting.localization.Localization(__file__, 49, 11), str_4033, *[name_4034], **kwargs_4035)
    
    # Assigning a type to the variable 'stypy_return_type' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type', str_call_result_4036)
    
    # ################# End of 'get_caller_data(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_caller_data' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_4037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4037)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_caller_data'
    return stypy_return_type_4037

# Assigning a type to the variable 'get_caller_data' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'get_caller_data', get_caller_data)
# Declaration of the 'ColorType' class

class ColorType:
    
    # Assigning a Name to a Name (line 53):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 52, 0, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ColorType.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ColorType' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'ColorType', ColorType)

# Assigning a Name to a Name (line 53):
# Getting the type of 'False' (line 53)
False_4038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'False')
# Getting the type of 'ColorType'
ColorType_4039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ColorType')
# Setting the type of the member 'ANSIColors' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ColorType_4039, 'ANSIColors', False_4038)


# SSA begins for try-except statement (line 56)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 57, 4))

# 'import ctypes' statement (line 57)
import ctypes

import_module(stypy.reporting.localization.Localization(__file__, 57, 4), 'ctypes', ctypes, module_type_store)


@norecursion
def setup_handles(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'setup_handles'
    module_type_store = module_type_store.open_function_context('setup_handles', 60, 4, False)
    
    # Passed parameters checking function
    setup_handles.stypy_localization = localization
    setup_handles.stypy_type_of_self = None
    setup_handles.stypy_type_store = module_type_store
    setup_handles.stypy_function_name = 'setup_handles'
    setup_handles.stypy_param_names_list = []
    setup_handles.stypy_varargs_param_name = None
    setup_handles.stypy_kwargs_param_name = None
    setup_handles.stypy_call_defaults = defaults
    setup_handles.stypy_call_varargs = varargs
    setup_handles.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'setup_handles', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'setup_handles', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'setup_handles(...)' code ##################

    str_4040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, (-1)), 'str', '\n        Determines if it is possible to have colored output\n        :return:\n        ')
    
    # Assigning a Num to a Name (line 66):
    
    # Assigning a Num to a Name (line 66):
    int_4041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 28), 'int')
    # Assigning a type to the variable 'STD_OUTPUT_HANDLE' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'STD_OUTPUT_HANDLE', int_4041)

    @norecursion
    def get_csbi_attributes(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_csbi_attributes'
        module_type_store = module_type_store.open_function_context('get_csbi_attributes', 68, 8, False)
        
        # Passed parameters checking function
        get_csbi_attributes.stypy_localization = localization
        get_csbi_attributes.stypy_type_of_self = None
        get_csbi_attributes.stypy_type_store = module_type_store
        get_csbi_attributes.stypy_function_name = 'get_csbi_attributes'
        get_csbi_attributes.stypy_param_names_list = ['handle']
        get_csbi_attributes.stypy_varargs_param_name = None
        get_csbi_attributes.stypy_kwargs_param_name = None
        get_csbi_attributes.stypy_call_defaults = defaults
        get_csbi_attributes.stypy_call_varargs = varargs
        get_csbi_attributes.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'get_csbi_attributes', ['handle'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_csbi_attributes', localization, ['handle'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_csbi_attributes(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 70, 12))
        
        # 'import struct' statement (line 70)
        import struct

        import_module(stypy.reporting.localization.Localization(__file__, 70, 12), 'struct', struct, module_type_store)
        
        
        # Assigning a Call to a Name (line 72):
        
        # Assigning a Call to a Name (line 72):
        
        # Call to create_string_buffer(...): (line 72)
        # Processing the call arguments (line 72)
        int_4044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 47), 'int')
        # Processing the call keyword arguments (line 72)
        kwargs_4045 = {}
        # Getting the type of 'ctypes' (line 72)
        ctypes_4042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'ctypes', False)
        # Obtaining the member 'create_string_buffer' of a type (line 72)
        create_string_buffer_4043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 19), ctypes_4042, 'create_string_buffer')
        # Calling create_string_buffer(args, kwargs) (line 72)
        create_string_buffer_call_result_4046 = invoke(stypy.reporting.localization.Localization(__file__, 72, 19), create_string_buffer_4043, *[int_4044], **kwargs_4045)
        
        # Assigning a type to the variable 'csbi' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'csbi', create_string_buffer_call_result_4046)
        
        # Assigning a Call to a Name (line 73):
        
        # Assigning a Call to a Name (line 73):
        
        # Call to GetConsoleScreenBufferInfo(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'handle' (line 73)
        handle_4051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 68), 'handle', False)
        # Getting the type of 'csbi' (line 73)
        csbi_4052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 76), 'csbi', False)
        # Processing the call keyword arguments (line 73)
        kwargs_4053 = {}
        # Getting the type of 'ctypes' (line 73)
        ctypes_4047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 18), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 73)
        windll_4048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 18), ctypes_4047, 'windll')
        # Obtaining the member 'kernel32' of a type (line 73)
        kernel32_4049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 18), windll_4048, 'kernel32')
        # Obtaining the member 'GetConsoleScreenBufferInfo' of a type (line 73)
        GetConsoleScreenBufferInfo_4050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 18), kernel32_4049, 'GetConsoleScreenBufferInfo')
        # Calling GetConsoleScreenBufferInfo(args, kwargs) (line 73)
        GetConsoleScreenBufferInfo_call_result_4054 = invoke(stypy.reporting.localization.Localization(__file__, 73, 18), GetConsoleScreenBufferInfo_4050, *[handle_4051, csbi_4052], **kwargs_4053)
        
        # Assigning a type to the variable 'res' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'res', GetConsoleScreenBufferInfo_call_result_4054)
        
        # Assigning a Call to a Tuple (line 76):
        
        # Assigning a Call to a Name:
        
        # Call to unpack(...): (line 77)
        # Processing the call arguments (line 77)
        str_4057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 67), 'str', 'hhhhHhhhhhh')
        # Getting the type of 'csbi' (line 77)
        csbi_4058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 82), 'csbi', False)
        # Obtaining the member 'raw' of a type (line 77)
        raw_4059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 82), csbi_4058, 'raw')
        # Processing the call keyword arguments (line 77)
        kwargs_4060 = {}
        # Getting the type of 'struct' (line 77)
        struct_4055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 53), 'struct', False)
        # Obtaining the member 'unpack' of a type (line 77)
        unpack_4056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 53), struct_4055, 'unpack')
        # Calling unpack(args, kwargs) (line 77)
        unpack_call_result_4061 = invoke(stypy.reporting.localization.Localization(__file__, 77, 53), unpack_4056, *[str_4057, raw_4059], **kwargs_4060)
        
        # Assigning a type to the variable 'call_assignment_3917' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3917', unpack_call_result_4061)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3917' (line 76)
        call_assignment_3917_4062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3917', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4063 = stypy_get_value_from_tuple(call_assignment_3917_4062, 11, 0)
        
        # Assigning a type to the variable 'call_assignment_3918' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3918', stypy_get_value_from_tuple_call_result_4063)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_3918' (line 76)
        call_assignment_3918_4064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3918')
        # Assigning a type to the variable 'bufx' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 13), 'bufx', call_assignment_3918_4064)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3917' (line 76)
        call_assignment_3917_4065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3917', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4066 = stypy_get_value_from_tuple(call_assignment_3917_4065, 11, 1)
        
        # Assigning a type to the variable 'call_assignment_3919' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3919', stypy_get_value_from_tuple_call_result_4066)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_3919' (line 76)
        call_assignment_3919_4067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3919')
        # Assigning a type to the variable 'bufy' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 19), 'bufy', call_assignment_3919_4067)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3917' (line 76)
        call_assignment_3917_4068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3917', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4069 = stypy_get_value_from_tuple(call_assignment_3917_4068, 11, 2)
        
        # Assigning a type to the variable 'call_assignment_3920' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3920', stypy_get_value_from_tuple_call_result_4069)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_3920' (line 76)
        call_assignment_3920_4070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3920')
        # Assigning a type to the variable 'curx' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 25), 'curx', call_assignment_3920_4070)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3917' (line 76)
        call_assignment_3917_4071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3917', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4072 = stypy_get_value_from_tuple(call_assignment_3917_4071, 11, 3)
        
        # Assigning a type to the variable 'call_assignment_3921' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3921', stypy_get_value_from_tuple_call_result_4072)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_3921' (line 76)
        call_assignment_3921_4073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3921')
        # Assigning a type to the variable 'cury' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 31), 'cury', call_assignment_3921_4073)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3917' (line 76)
        call_assignment_3917_4074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3917', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4075 = stypy_get_value_from_tuple(call_assignment_3917_4074, 11, 4)
        
        # Assigning a type to the variable 'call_assignment_3922' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3922', stypy_get_value_from_tuple_call_result_4075)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_3922' (line 76)
        call_assignment_3922_4076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3922')
        # Assigning a type to the variable 'wattr' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 37), 'wattr', call_assignment_3922_4076)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3917' (line 76)
        call_assignment_3917_4077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3917', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4078 = stypy_get_value_from_tuple(call_assignment_3917_4077, 11, 5)
        
        # Assigning a type to the variable 'call_assignment_3923' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3923', stypy_get_value_from_tuple_call_result_4078)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_3923' (line 76)
        call_assignment_3923_4079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3923')
        # Assigning a type to the variable 'left' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 13), 'left', call_assignment_3923_4079)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3917' (line 76)
        call_assignment_3917_4080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3917', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4081 = stypy_get_value_from_tuple(call_assignment_3917_4080, 11, 6)
        
        # Assigning a type to the variable 'call_assignment_3924' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3924', stypy_get_value_from_tuple_call_result_4081)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_3924' (line 76)
        call_assignment_3924_4082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3924')
        # Assigning a type to the variable 'top' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'top', call_assignment_3924_4082)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3917' (line 76)
        call_assignment_3917_4083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3917', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4084 = stypy_get_value_from_tuple(call_assignment_3917_4083, 11, 7)
        
        # Assigning a type to the variable 'call_assignment_3925' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3925', stypy_get_value_from_tuple_call_result_4084)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_3925' (line 76)
        call_assignment_3925_4085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3925')
        # Assigning a type to the variable 'right' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 24), 'right', call_assignment_3925_4085)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3917' (line 76)
        call_assignment_3917_4086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3917', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4087 = stypy_get_value_from_tuple(call_assignment_3917_4086, 11, 8)
        
        # Assigning a type to the variable 'call_assignment_3926' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3926', stypy_get_value_from_tuple_call_result_4087)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_3926' (line 76)
        call_assignment_3926_4088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3926')
        # Assigning a type to the variable 'bottom' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 31), 'bottom', call_assignment_3926_4088)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3917' (line 76)
        call_assignment_3917_4089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3917', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4090 = stypy_get_value_from_tuple(call_assignment_3917_4089, 11, 9)
        
        # Assigning a type to the variable 'call_assignment_3927' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3927', stypy_get_value_from_tuple_call_result_4090)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_3927' (line 76)
        call_assignment_3927_4091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3927')
        # Assigning a type to the variable 'maxx' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 39), 'maxx', call_assignment_3927_4091)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3917' (line 76)
        call_assignment_3917_4092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3917', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4093 = stypy_get_value_from_tuple(call_assignment_3917_4092, 11, 10)
        
        # Assigning a type to the variable 'call_assignment_3928' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3928', stypy_get_value_from_tuple_call_result_4093)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_3928' (line 76)
        call_assignment_3928_4094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_3928')
        # Assigning a type to the variable 'maxy' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 45), 'maxy', call_assignment_3928_4094)
        # Getting the type of 'wattr' (line 78)
        wattr_4095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'wattr')
        # Assigning a type to the variable 'stypy_return_type' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'stypy_return_type', wattr_4095)
        
        # ################# End of 'get_csbi_attributes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_csbi_attributes' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_4096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4096)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_csbi_attributes'
        return stypy_return_type_4096

    # Assigning a type to the variable 'get_csbi_attributes' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'get_csbi_attributes', get_csbi_attributes)
    
    # Assigning a Call to a Name (line 80):
    
    # Assigning a Call to a Name (line 80):
    
    # Call to GetStdHandle(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'STD_OUTPUT_HANDLE' (line 80)
    STD_OUTPUT_HANDLE_4101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 53), 'STD_OUTPUT_HANDLE', False)
    # Processing the call keyword arguments (line 80)
    kwargs_4102 = {}
    # Getting the type of 'ctypes' (line 80)
    ctypes_4097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'ctypes', False)
    # Obtaining the member 'windll' of a type (line 80)
    windll_4098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 17), ctypes_4097, 'windll')
    # Obtaining the member 'kernel32' of a type (line 80)
    kernel32_4099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 17), windll_4098, 'kernel32')
    # Obtaining the member 'GetStdHandle' of a type (line 80)
    GetStdHandle_4100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 17), kernel32_4099, 'GetStdHandle')
    # Calling GetStdHandle(args, kwargs) (line 80)
    GetStdHandle_call_result_4103 = invoke(stypy.reporting.localization.Localization(__file__, 80, 17), GetStdHandle_4100, *[STD_OUTPUT_HANDLE_4101], **kwargs_4102)
    
    # Assigning a type to the variable 'handle' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'handle', GetStdHandle_call_result_4103)
    
    # Assigning a Call to a Name (line 81):
    
    # Assigning a Call to a Name (line 81):
    
    # Call to get_csbi_attributes(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'handle' (line 81)
    handle_4105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 36), 'handle', False)
    # Processing the call keyword arguments (line 81)
    kwargs_4106 = {}
    # Getting the type of 'get_csbi_attributes' (line 81)
    get_csbi_attributes_4104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'get_csbi_attributes', False)
    # Calling get_csbi_attributes(args, kwargs) (line 81)
    get_csbi_attributes_call_result_4107 = invoke(stypy.reporting.localization.Localization(__file__, 81, 16), get_csbi_attributes_4104, *[handle_4105], **kwargs_4106)
    
    # Assigning a type to the variable 'reset' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'reset', get_csbi_attributes_call_result_4107)
    
    # Obtaining an instance of the builtin type 'tuple' (line 82)
    tuple_4108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 82)
    # Adding element type (line 82)
    # Getting the type of 'handle' (line 82)
    handle_4109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'handle')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 15), tuple_4108, handle_4109)
    # Adding element type (line 82)
    # Getting the type of 'reset' (line 82)
    reset_4110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 23), 'reset')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 15), tuple_4108, reset_4110)
    
    # Assigning a type to the variable 'stypy_return_type' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stypy_return_type', tuple_4108)
    
    # ################# End of 'setup_handles(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'setup_handles' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_4111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4111)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'setup_handles'
    return stypy_return_type_4111

# Assigning a type to the variable 'setup_handles' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'setup_handles', setup_handles)

# Assigning a Name to a Attribute (line 85):

# Assigning a Name to a Attribute (line 85):
# Getting the type of 'False' (line 85)
False_4112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 27), 'False')
# Getting the type of 'ColorType' (line 85)
ColorType_4113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'ColorType')
# Setting the type of the member 'ANSIColors' of a type (line 85)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 4), ColorType_4113, 'ANSIColors', False_4112)
# SSA branch for the except part of a try statement (line 56)
# SSA branch for the except 'Exception' branch of a try statement (line 56)
# Storing handler type
module_type_store.open_ssa_branch('except')
# Getting the type of 'Exception' (line 87)
Exception_4114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 7), 'Exception')
# Assigning a type to the variable 'e' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'e', Exception_4114)

# Assigning a Name to a Attribute (line 88):

# Assigning a Name to a Attribute (line 88):
# Getting the type of 'True' (line 88)
True_4115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 27), 'True')
# Getting the type of 'ColorType' (line 88)
ColorType_4116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'ColorType')
# Setting the type of the member 'ANSIColors' of a type (line 88)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 4), ColorType_4116, 'ANSIColors', True_4115)
# SSA join for try-except statement (line 56)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'Colors' class

class Colors:
    
    # Assigning a Str to a Name (line 92):
    
    # Assigning a Str to a Name (line 93):
    
    # Assigning a Str to a Name (line 94):
    
    # Assigning a Str to a Name (line 95):
    
    # Assigning a Str to a Name (line 96):
    
    # Assigning a Num to a Name (line 98):
    
    # Assigning a Num to a Name (line 99):
    
    # Assigning a Num to a Name (line 100):
    
    # Assigning a Num to a Name (line 101):
    
    # Assigning a Num to a Name (line 102):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 91, 0, False)
        # Assigning a type to the variable 'self' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Colors.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Colors' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'Colors', Colors)

# Assigning a Str to a Name (line 92):
str_4117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 16), 'str', '\x1b[94m')
# Getting the type of 'Colors'
Colors_4118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colors')
# Setting the type of the member 'ANSI_BLUE' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colors_4118, 'ANSI_BLUE', str_4117)

# Assigning a Str to a Name (line 93):
str_4119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 17), 'str', '\x1b[92m')
# Getting the type of 'Colors'
Colors_4120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colors')
# Setting the type of the member 'ANSI_GREEN' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colors_4120, 'ANSI_GREEN', str_4119)

# Assigning a Str to a Name (line 94):
str_4121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 19), 'str', '\x1b[93m')
# Getting the type of 'Colors'
Colors_4122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colors')
# Setting the type of the member 'ANSI_WARNING' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colors_4122, 'ANSI_WARNING', str_4121)

# Assigning a Str to a Name (line 95):
str_4123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 16), 'str', '\x1b[91m')
# Getting the type of 'Colors'
Colors_4124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colors')
# Setting the type of the member 'ANSI_FAIL' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colors_4124, 'ANSI_FAIL', str_4123)

# Assigning a Str to a Name (line 96):
str_4125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 16), 'str', '\x1b[0m')
# Getting the type of 'Colors'
Colors_4126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colors')
# Setting the type of the member 'ANSI_ENDC' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colors_4126, 'ANSI_ENDC', str_4125)

# Assigning a Num to a Name (line 98):
int_4127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 15), 'int')
# Getting the type of 'Colors'
Colors_4128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colors')
# Setting the type of the member 'WIN_BLUE' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colors_4128, 'WIN_BLUE', int_4127)

# Assigning a Num to a Name (line 99):
int_4129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 16), 'int')
# Getting the type of 'Colors'
Colors_4130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colors')
# Setting the type of the member 'WIN_WHITE' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colors_4130, 'WIN_WHITE', int_4129)

# Assigning a Num to a Name (line 100):
int_4131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 16), 'int')
# Getting the type of 'Colors'
Colors_4132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colors')
# Setting the type of the member 'WIN_GREEN' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colors_4132, 'WIN_GREEN', int_4131)

# Assigning a Num to a Name (line 101):
int_4133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 18), 'int')
# Getting the type of 'Colors'
Colors_4134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colors')
# Setting the type of the member 'WIN_WARNING' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colors_4134, 'WIN_WARNING', int_4133)

# Assigning a Num to a Name (line 102):
int_4135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 15), 'int')
# Getting the type of 'Colors'
Colors_4136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colors')
# Setting the type of the member 'WIN_FAIL' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colors_4136, 'WIN_FAIL', int_4135)

@norecursion
def get_date_time(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_date_time'
    module_type_store = module_type_store.open_function_context('get_date_time', 105, 0, False)
    
    # Passed parameters checking function
    get_date_time.stypy_localization = localization
    get_date_time.stypy_type_of_self = None
    get_date_time.stypy_type_store = module_type_store
    get_date_time.stypy_function_name = 'get_date_time'
    get_date_time.stypy_param_names_list = []
    get_date_time.stypy_varargs_param_name = None
    get_date_time.stypy_kwargs_param_name = None
    get_date_time.stypy_call_defaults = defaults
    get_date_time.stypy_call_varargs = varargs
    get_date_time.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_date_time', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_date_time', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_date_time(...)' code ##################

    str_4137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, (-1)), 'str', '\n    Obtains current date and time\n    :return:\n    ')
    
    # Obtaining the type of the subscript
    int_4138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 41), 'int')
    slice_4139 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 110, 11), None, int_4138, None)
    
    # Call to str(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Call to now(...): (line 110)
    # Processing the call keyword arguments (line 110)
    kwargs_4144 = {}
    # Getting the type of 'datetime' (line 110)
    datetime_4141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 15), 'datetime', False)
    # Obtaining the member 'datetime' of a type (line 110)
    datetime_4142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 15), datetime_4141, 'datetime')
    # Obtaining the member 'now' of a type (line 110)
    now_4143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 15), datetime_4142, 'now')
    # Calling now(args, kwargs) (line 110)
    now_call_result_4145 = invoke(stypy.reporting.localization.Localization(__file__, 110, 15), now_4143, *[], **kwargs_4144)
    
    # Processing the call keyword arguments (line 110)
    kwargs_4146 = {}
    # Getting the type of 'str' (line 110)
    str_4140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'str', False)
    # Calling str(args, kwargs) (line 110)
    str_call_result_4147 = invoke(stypy.reporting.localization.Localization(__file__, 110, 11), str_4140, *[now_call_result_4145], **kwargs_4146)
    
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___4148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 11), str_call_result_4147, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_4149 = invoke(stypy.reporting.localization.Localization(__file__, 110, 11), getitem___4148, slice_4139)
    
    # Assigning a type to the variable 'stypy_return_type' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'stypy_return_type', subscript_call_result_4149)
    
    # ################# End of 'get_date_time(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_date_time' in the type store
    # Getting the type of 'stypy_return_type' (line 105)
    stypy_return_type_4150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4150)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_date_time'
    return stypy_return_type_4150

# Assigning a type to the variable 'get_date_time' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'get_date_time', get_date_time)

@norecursion
def log(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'log'
    module_type_store = module_type_store.open_function_context('log', 113, 0, False)
    
    # Passed parameters checking function
    log.stypy_localization = localization
    log.stypy_type_of_self = None
    log.stypy_type_store = module_type_store
    log.stypy_function_name = 'log'
    log.stypy_param_names_list = ['msg']
    log.stypy_varargs_param_name = None
    log.stypy_kwargs_param_name = None
    log.stypy_call_defaults = defaults
    log.stypy_call_varargs = varargs
    log.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'log', ['msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'log', localization, ['msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'log(...)' code ##################

    str_4151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, (-1)), 'str', '\n    Logs information messages to the corresponding log file\n    :param msg:\n    :return:\n    ')
    
    
    # SSA begins for try-except statement (line 119)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 120):
    
    # Assigning a Call to a Name (line 120):
    
    # Call to open(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'stypy_parameters_copy' (line 120)
    stypy_parameters_copy_4153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 21), 'stypy_parameters_copy', False)
    # Obtaining the member 'LOG_PATH' of a type (line 120)
    LOG_PATH_4154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 21), stypy_parameters_copy_4153, 'LOG_PATH')
    str_4155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 54), 'str', '/')
    # Applying the binary operator '+' (line 120)
    result_add_4156 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 21), '+', LOG_PATH_4154, str_4155)
    
    # Getting the type of 'stypy_parameters_copy' (line 120)
    stypy_parameters_copy_4157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 60), 'stypy_parameters_copy', False)
    # Obtaining the member 'INFO_LOG_FILE' of a type (line 120)
    INFO_LOG_FILE_4158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 60), stypy_parameters_copy_4157, 'INFO_LOG_FILE')
    # Applying the binary operator '+' (line 120)
    result_add_4159 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 58), '+', result_add_4156, INFO_LOG_FILE_4158)
    
    str_4160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 97), 'str', 'a')
    # Processing the call keyword arguments (line 120)
    kwargs_4161 = {}
    # Getting the type of 'open' (line 120)
    open_4152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'open', False)
    # Calling open(args, kwargs) (line 120)
    open_call_result_4162 = invoke(stypy.reporting.localization.Localization(__file__, 120, 16), open_4152, *[result_add_4159, str_4160], **kwargs_4161)
    
    # Assigning a type to the variable 'file_' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'file_', open_call_result_4162)
    # SSA branch for the except part of a try statement (line 119)
    # SSA branch for the except '<any exception>' branch of a try statement (line 119)
    module_type_store.open_ssa_branch('except')
    # Assigning a type to the variable 'stypy_return_type' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'stypy_return_type', types.NoneType)
    # SSA join for try-except statement (line 119)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'msg' (line 124)
    msg_4163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'msg')
    str_4164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 15), 'str', '\n')
    # Applying the binary operator '==' (line 124)
    result_eq_4165 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 8), '==', msg_4163, str_4164)
    
    
    # Getting the type of 'msg' (line 124)
    msg_4166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 25), 'msg')
    str_4167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 32), 'str', '')
    # Applying the binary operator '==' (line 124)
    result_eq_4168 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 25), '==', msg_4166, str_4167)
    
    # Applying the binary operator 'or' (line 124)
    result_or_keyword_4169 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 7), 'or', result_eq_4165, result_eq_4168)
    
    # Testing if the type of an if condition is none (line 124)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 124, 4), result_or_keyword_4169):
        
        # Type idiom detected: calculating its left and rigth part (line 130)
        # Getting the type of 'file_' (line 130)
        file__4187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'file_')
        # Getting the type of 'None' (line 130)
        None_4188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'None')
        
        (may_be_4189, more_types_in_union_4190) = may_not_be_none(file__4187, None_4188)

        if may_be_4189:

            if more_types_in_union_4190:
                # Runtime conditional SSA (line 130)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to write(...): (line 131)
            # Processing the call arguments (line 131)
            str_4193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 24), 'str', '[')
            
            # Call to get_date_time(...): (line 131)
            # Processing the call keyword arguments (line 131)
            kwargs_4195 = {}
            # Getting the type of 'get_date_time' (line 131)
            get_date_time_4194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 30), 'get_date_time', False)
            # Calling get_date_time(args, kwargs) (line 131)
            get_date_time_call_result_4196 = invoke(stypy.reporting.localization.Localization(__file__, 131, 30), get_date_time_4194, *[], **kwargs_4195)
            
            # Applying the binary operator '+' (line 131)
            result_add_4197 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 24), '+', str_4193, get_date_time_call_result_4196)
            
            str_4198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 48), 'str', '] ')
            # Applying the binary operator '+' (line 131)
            result_add_4199 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 46), '+', result_add_4197, str_4198)
            
            # Getting the type of 'msg' (line 131)
            msg_4200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 55), 'msg', False)
            # Applying the binary operator '+' (line 131)
            result_add_4201 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 53), '+', result_add_4199, msg_4200)
            
            str_4202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 61), 'str', '\n')
            # Applying the binary operator '+' (line 131)
            result_add_4203 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 59), '+', result_add_4201, str_4202)
            
            # Processing the call keyword arguments (line 131)
            kwargs_4204 = {}
            # Getting the type of 'file_' (line 131)
            file__4191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'file_', False)
            # Obtaining the member 'write' of a type (line 131)
            write_4192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), file__4191, 'write')
            # Calling write(args, kwargs) (line 131)
            write_call_result_4205 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), write_4192, *[result_add_4203], **kwargs_4204)
            

            if more_types_in_union_4190:
                # SSA join for if statement (line 130)
                module_type_store = module_type_store.join_ssa_context()


        
    else:
        
        # Testing the type of an if condition (line 124)
        if_condition_4170 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 4), result_or_keyword_4169)
        # Assigning a type to the variable 'if_condition_4170' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'if_condition_4170', if_condition_4170)
        # SSA begins for if statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'msg' (line 125)
        msg_4171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), 'msg')
        str_4172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 18), 'str', '')
        # Applying the binary operator '==' (line 125)
        result_eq_4173 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 11), '==', msg_4171, str_4172)
        
        # Testing if the type of an if condition is none (line 125)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 8), result_eq_4173):
            
            # Call to write(...): (line 128)
            # Processing the call arguments (line 128)
            # Getting the type of 'msg' (line 128)
            msg_4184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'msg', False)
            # Processing the call keyword arguments (line 128)
            kwargs_4185 = {}
            # Getting the type of 'file_' (line 128)
            file__4182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'file_', False)
            # Obtaining the member 'write' of a type (line 128)
            write_4183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), file__4182, 'write')
            # Calling write(args, kwargs) (line 128)
            write_call_result_4186 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), write_4183, *[msg_4184], **kwargs_4185)
            
        else:
            
            # Testing the type of an if condition (line 125)
            if_condition_4174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 8), result_eq_4173)
            # Assigning a type to the variable 'if_condition_4174' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'if_condition_4174', if_condition_4174)
            # SSA begins for if statement (line 125)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 126)
            # Processing the call arguments (line 126)
            # Getting the type of 'msg' (line 126)
            msg_4177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'msg', False)
            str_4178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 30), 'str', '\n')
            # Applying the binary operator '+' (line 126)
            result_add_4179 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 24), '+', msg_4177, str_4178)
            
            # Processing the call keyword arguments (line 126)
            kwargs_4180 = {}
            # Getting the type of 'file_' (line 126)
            file__4175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'file_', False)
            # Obtaining the member 'write' of a type (line 126)
            write_4176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 12), file__4175, 'write')
            # Calling write(args, kwargs) (line 126)
            write_call_result_4181 = invoke(stypy.reporting.localization.Localization(__file__, 126, 12), write_4176, *[result_add_4179], **kwargs_4180)
            
            # SSA branch for the else part of an if statement (line 125)
            module_type_store.open_ssa_branch('else')
            
            # Call to write(...): (line 128)
            # Processing the call arguments (line 128)
            # Getting the type of 'msg' (line 128)
            msg_4184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'msg', False)
            # Processing the call keyword arguments (line 128)
            kwargs_4185 = {}
            # Getting the type of 'file_' (line 128)
            file__4182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'file_', False)
            # Obtaining the member 'write' of a type (line 128)
            write_4183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), file__4182, 'write')
            # Calling write(args, kwargs) (line 128)
            write_call_result_4186 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), write_4183, *[msg_4184], **kwargs_4185)
            
            # SSA join for if statement (line 125)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA branch for the else part of an if statement (line 124)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 130)
        # Getting the type of 'file_' (line 130)
        file__4187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'file_')
        # Getting the type of 'None' (line 130)
        None_4188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'None')
        
        (may_be_4189, more_types_in_union_4190) = may_not_be_none(file__4187, None_4188)

        if may_be_4189:

            if more_types_in_union_4190:
                # Runtime conditional SSA (line 130)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to write(...): (line 131)
            # Processing the call arguments (line 131)
            str_4193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 24), 'str', '[')
            
            # Call to get_date_time(...): (line 131)
            # Processing the call keyword arguments (line 131)
            kwargs_4195 = {}
            # Getting the type of 'get_date_time' (line 131)
            get_date_time_4194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 30), 'get_date_time', False)
            # Calling get_date_time(args, kwargs) (line 131)
            get_date_time_call_result_4196 = invoke(stypy.reporting.localization.Localization(__file__, 131, 30), get_date_time_4194, *[], **kwargs_4195)
            
            # Applying the binary operator '+' (line 131)
            result_add_4197 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 24), '+', str_4193, get_date_time_call_result_4196)
            
            str_4198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 48), 'str', '] ')
            # Applying the binary operator '+' (line 131)
            result_add_4199 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 46), '+', result_add_4197, str_4198)
            
            # Getting the type of 'msg' (line 131)
            msg_4200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 55), 'msg', False)
            # Applying the binary operator '+' (line 131)
            result_add_4201 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 53), '+', result_add_4199, msg_4200)
            
            str_4202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 61), 'str', '\n')
            # Applying the binary operator '+' (line 131)
            result_add_4203 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 59), '+', result_add_4201, str_4202)
            
            # Processing the call keyword arguments (line 131)
            kwargs_4204 = {}
            # Getting the type of 'file_' (line 131)
            file__4191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'file_', False)
            # Obtaining the member 'write' of a type (line 131)
            write_4192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), file__4191, 'write')
            # Calling write(args, kwargs) (line 131)
            write_call_result_4205 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), write_4192, *[result_add_4203], **kwargs_4204)
            

            if more_types_in_union_4190:
                # SSA join for if statement (line 130)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 124)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to close(...): (line 133)
    # Processing the call keyword arguments (line 133)
    kwargs_4208 = {}
    # Getting the type of 'file_' (line 133)
    file__4206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'file_', False)
    # Obtaining the member 'close' of a type (line 133)
    close_4207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 4), file__4206, 'close')
    # Calling close(args, kwargs) (line 133)
    close_call_result_4209 = invoke(stypy.reporting.localization.Localization(__file__, 133, 4), close_4207, *[], **kwargs_4208)
    
    
    # ################# End of 'log(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'log' in the type store
    # Getting the type of 'stypy_return_type' (line 113)
    stypy_return_type_4210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4210)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'log'
    return stypy_return_type_4210

# Assigning a type to the variable 'log' (line 113)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), 'log', log)

@norecursion
def ok(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ok'
    module_type_store = module_type_store.open_function_context('ok', 136, 0, False)
    
    # Passed parameters checking function
    ok.stypy_localization = localization
    ok.stypy_type_of_self = None
    ok.stypy_type_store = module_type_store
    ok.stypy_function_name = 'ok'
    ok.stypy_param_names_list = ['msg']
    ok.stypy_varargs_param_name = None
    ok.stypy_kwargs_param_name = None
    ok.stypy_call_defaults = defaults
    ok.stypy_call_varargs = varargs
    ok.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ok', ['msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ok', localization, ['msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ok(...)' code ##################

    str_4211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, (-1)), 'str', '\n    Handles green log information messages\n    :param msg:\n    :return:\n    ')
    
    # Assigning a BinOp to a Name (line 142):
    
    # Assigning a BinOp to a Name (line 142):
    
    # Call to get_caller_data(...): (line 142)
    # Processing the call keyword arguments (line 142)
    kwargs_4213 = {}
    # Getting the type of 'get_caller_data' (line 142)
    get_caller_data_4212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 10), 'get_caller_data', False)
    # Calling get_caller_data(args, kwargs) (line 142)
    get_caller_data_call_result_4214 = invoke(stypy.reporting.localization.Localization(__file__, 142, 10), get_caller_data_4212, *[], **kwargs_4213)
    
    str_4215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 30), 'str', ': ')
    # Applying the binary operator '+' (line 142)
    result_add_4216 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 10), '+', get_caller_data_call_result_4214, str_4215)
    
    # Getting the type of 'msg' (line 142)
    msg_4217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 37), 'msg')
    # Applying the binary operator '+' (line 142)
    result_add_4218 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 35), '+', result_add_4216, msg_4217)
    
    # Assigning a type to the variable 'txt' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'txt', result_add_4218)
    # Getting the type of 'ColorType' (line 143)
    ColorType_4219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 7), 'ColorType')
    # Obtaining the member 'ANSIColors' of a type (line 143)
    ANSIColors_4220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 7), ColorType_4219, 'ANSIColors')
    # Testing if the type of an if condition is none (line 143)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 143, 4), ANSIColors_4220):
        
        # Assigning a Call to a Tuple (line 147):
        
        # Assigning a Call to a Name:
        
        # Call to setup_handles(...): (line 147)
        # Processing the call keyword arguments (line 147)
        kwargs_4232 = {}
        # Getting the type of 'setup_handles' (line 147)
        setup_handles_4231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 24), 'setup_handles', False)
        # Calling setup_handles(args, kwargs) (line 147)
        setup_handles_call_result_4233 = invoke(stypy.reporting.localization.Localization(__file__, 147, 24), setup_handles_4231, *[], **kwargs_4232)
        
        # Assigning a type to the variable 'call_assignment_3929' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_3929', setup_handles_call_result_4233)
        
        # Assigning a Call to a Name (line 147):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3929' (line 147)
        call_assignment_3929_4234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_3929', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4235 = stypy_get_value_from_tuple(call_assignment_3929_4234, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_3930' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_3930', stypy_get_value_from_tuple_call_result_4235)
        
        # Assigning a Name to a Name (line 147):
        # Getting the type of 'call_assignment_3930' (line 147)
        call_assignment_3930_4236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_3930')
        # Assigning a type to the variable 'handle' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'handle', call_assignment_3930_4236)
        
        # Assigning a Call to a Name (line 147):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3929' (line 147)
        call_assignment_3929_4237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_3929', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4238 = stypy_get_value_from_tuple(call_assignment_3929_4237, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_3931' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_3931', stypy_get_value_from_tuple_call_result_4238)
        
        # Assigning a Name to a Name (line 147):
        # Getting the type of 'call_assignment_3931' (line 147)
        call_assignment_3931_4239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_3931')
        # Assigning a type to the variable 'reset' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'reset', call_assignment_3931_4239)
        
        # Call to SetConsoleTextAttribute(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'handle' (line 148)
        handle_4244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 55), 'handle', False)
        # Getting the type of 'Colors' (line 148)
        Colors_4245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 63), 'Colors', False)
        # Obtaining the member 'WIN_GREEN' of a type (line 148)
        WIN_GREEN_4246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 63), Colors_4245, 'WIN_GREEN')
        # Processing the call keyword arguments (line 148)
        kwargs_4247 = {}
        # Getting the type of 'ctypes' (line 148)
        ctypes_4240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 148)
        windll_4241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), ctypes_4240, 'windll')
        # Obtaining the member 'kernel32' of a type (line 148)
        kernel32_4242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), windll_4241, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 148)
        SetConsoleTextAttribute_4243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), kernel32_4242, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 148)
        SetConsoleTextAttribute_call_result_4248 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), SetConsoleTextAttribute_4243, *[handle_4244, WIN_GREEN_4246], **kwargs_4247)
        
        # Getting the type of 'output_to_console' (line 150)
        output_to_console_4249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'output_to_console')
        # Testing if the type of an if condition is none (line 150)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 150, 8), output_to_console_4249):
            pass
        else:
            
            # Testing the type of an if condition (line 150)
            if_condition_4250 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 8), output_to_console_4249)
            # Assigning a type to the variable 'if_condition_4250' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'if_condition_4250', if_condition_4250)
            # SSA begins for if statement (line 150)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'msg' (line 151)
            msg_4251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 19), 'msg')
            # SSA join for if statement (line 150)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to SetConsoleTextAttribute(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'handle' (line 153)
        handle_4256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 55), 'handle', False)
        # Getting the type of 'reset' (line 153)
        reset_4257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 63), 'reset', False)
        # Processing the call keyword arguments (line 153)
        kwargs_4258 = {}
        # Getting the type of 'ctypes' (line 153)
        ctypes_4252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 153)
        windll_4253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), ctypes_4252, 'windll')
        # Obtaining the member 'kernel32' of a type (line 153)
        kernel32_4254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), windll_4253, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 153)
        SetConsoleTextAttribute_4255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), kernel32_4254, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 153)
        SetConsoleTextAttribute_call_result_4259 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), SetConsoleTextAttribute_4255, *[handle_4256, reset_4257], **kwargs_4258)
        
    else:
        
        # Testing the type of an if condition (line 143)
        if_condition_4221 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 4), ANSIColors_4220)
        # Assigning a type to the variable 'if_condition_4221' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'if_condition_4221', if_condition_4221)
        # SSA begins for if statement (line 143)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'output_to_console' (line 144)
        output_to_console_4222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'output_to_console')
        # Testing if the type of an if condition is none (line 144)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 144, 8), output_to_console_4222):
            pass
        else:
            
            # Testing the type of an if condition (line 144)
            if_condition_4223 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 8), output_to_console_4222)
            # Assigning a type to the variable 'if_condition_4223' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'if_condition_4223', if_condition_4223)
            # SSA begins for if statement (line 144)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'Colors' (line 145)
            Colors_4224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 18), 'Colors')
            # Obtaining the member 'ANSI_GREEN' of a type (line 145)
            ANSI_GREEN_4225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 18), Colors_4224, 'ANSI_GREEN')
            # Getting the type of 'msg' (line 145)
            msg_4226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 38), 'msg')
            # Applying the binary operator '+' (line 145)
            result_add_4227 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 18), '+', ANSI_GREEN_4225, msg_4226)
            
            # Getting the type of 'Colors' (line 145)
            Colors_4228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 44), 'Colors')
            # Obtaining the member 'ANSI_ENDC' of a type (line 145)
            ANSI_ENDC_4229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 44), Colors_4228, 'ANSI_ENDC')
            # Applying the binary operator '+' (line 145)
            result_add_4230 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 42), '+', result_add_4227, ANSI_ENDC_4229)
            
            # SSA join for if statement (line 144)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA branch for the else part of an if statement (line 143)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Tuple (line 147):
        
        # Assigning a Call to a Name:
        
        # Call to setup_handles(...): (line 147)
        # Processing the call keyword arguments (line 147)
        kwargs_4232 = {}
        # Getting the type of 'setup_handles' (line 147)
        setup_handles_4231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 24), 'setup_handles', False)
        # Calling setup_handles(args, kwargs) (line 147)
        setup_handles_call_result_4233 = invoke(stypy.reporting.localization.Localization(__file__, 147, 24), setup_handles_4231, *[], **kwargs_4232)
        
        # Assigning a type to the variable 'call_assignment_3929' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_3929', setup_handles_call_result_4233)
        
        # Assigning a Call to a Name (line 147):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3929' (line 147)
        call_assignment_3929_4234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_3929', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4235 = stypy_get_value_from_tuple(call_assignment_3929_4234, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_3930' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_3930', stypy_get_value_from_tuple_call_result_4235)
        
        # Assigning a Name to a Name (line 147):
        # Getting the type of 'call_assignment_3930' (line 147)
        call_assignment_3930_4236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_3930')
        # Assigning a type to the variable 'handle' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'handle', call_assignment_3930_4236)
        
        # Assigning a Call to a Name (line 147):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3929' (line 147)
        call_assignment_3929_4237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_3929', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4238 = stypy_get_value_from_tuple(call_assignment_3929_4237, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_3931' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_3931', stypy_get_value_from_tuple_call_result_4238)
        
        # Assigning a Name to a Name (line 147):
        # Getting the type of 'call_assignment_3931' (line 147)
        call_assignment_3931_4239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_3931')
        # Assigning a type to the variable 'reset' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'reset', call_assignment_3931_4239)
        
        # Call to SetConsoleTextAttribute(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'handle' (line 148)
        handle_4244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 55), 'handle', False)
        # Getting the type of 'Colors' (line 148)
        Colors_4245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 63), 'Colors', False)
        # Obtaining the member 'WIN_GREEN' of a type (line 148)
        WIN_GREEN_4246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 63), Colors_4245, 'WIN_GREEN')
        # Processing the call keyword arguments (line 148)
        kwargs_4247 = {}
        # Getting the type of 'ctypes' (line 148)
        ctypes_4240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 148)
        windll_4241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), ctypes_4240, 'windll')
        # Obtaining the member 'kernel32' of a type (line 148)
        kernel32_4242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), windll_4241, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 148)
        SetConsoleTextAttribute_4243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), kernel32_4242, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 148)
        SetConsoleTextAttribute_call_result_4248 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), SetConsoleTextAttribute_4243, *[handle_4244, WIN_GREEN_4246], **kwargs_4247)
        
        # Getting the type of 'output_to_console' (line 150)
        output_to_console_4249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'output_to_console')
        # Testing if the type of an if condition is none (line 150)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 150, 8), output_to_console_4249):
            pass
        else:
            
            # Testing the type of an if condition (line 150)
            if_condition_4250 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 8), output_to_console_4249)
            # Assigning a type to the variable 'if_condition_4250' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'if_condition_4250', if_condition_4250)
            # SSA begins for if statement (line 150)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'msg' (line 151)
            msg_4251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 19), 'msg')
            # SSA join for if statement (line 150)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to SetConsoleTextAttribute(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'handle' (line 153)
        handle_4256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 55), 'handle', False)
        # Getting the type of 'reset' (line 153)
        reset_4257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 63), 'reset', False)
        # Processing the call keyword arguments (line 153)
        kwargs_4258 = {}
        # Getting the type of 'ctypes' (line 153)
        ctypes_4252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 153)
        windll_4253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), ctypes_4252, 'windll')
        # Obtaining the member 'kernel32' of a type (line 153)
        kernel32_4254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), windll_4253, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 153)
        SetConsoleTextAttribute_4255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), kernel32_4254, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 153)
        SetConsoleTextAttribute_call_result_4259 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), SetConsoleTextAttribute_4255, *[handle_4256, reset_4257], **kwargs_4258)
        
        # SSA join for if statement (line 143)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to log(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'txt' (line 155)
    txt_4261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'txt', False)
    # Processing the call keyword arguments (line 155)
    kwargs_4262 = {}
    # Getting the type of 'log' (line 155)
    log_4260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'log', False)
    # Calling log(args, kwargs) (line 155)
    log_call_result_4263 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), log_4260, *[txt_4261], **kwargs_4262)
    
    
    # ################# End of 'ok(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ok' in the type store
    # Getting the type of 'stypy_return_type' (line 136)
    stypy_return_type_4264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4264)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ok'
    return stypy_return_type_4264

# Assigning a type to the variable 'ok' (line 136)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'ok', ok)

@norecursion
def info(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'info'
    module_type_store = module_type_store.open_function_context('info', 158, 0, False)
    
    # Passed parameters checking function
    info.stypy_localization = localization
    info.stypy_type_of_self = None
    info.stypy_type_store = module_type_store
    info.stypy_function_name = 'info'
    info.stypy_param_names_list = ['msg']
    info.stypy_varargs_param_name = None
    info.stypy_kwargs_param_name = None
    info.stypy_call_defaults = defaults
    info.stypy_call_varargs = varargs
    info.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'info', ['msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'info', localization, ['msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'info(...)' code ##################

    str_4265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, (-1)), 'str', '\n    Handles white log information messages\n    :param msg:\n    :return:\n    ')
    
    # Assigning a BinOp to a Name (line 164):
    
    # Assigning a BinOp to a Name (line 164):
    
    # Call to get_caller_data(...): (line 164)
    # Processing the call keyword arguments (line 164)
    kwargs_4267 = {}
    # Getting the type of 'get_caller_data' (line 164)
    get_caller_data_4266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 10), 'get_caller_data', False)
    # Calling get_caller_data(args, kwargs) (line 164)
    get_caller_data_call_result_4268 = invoke(stypy.reporting.localization.Localization(__file__, 164, 10), get_caller_data_4266, *[], **kwargs_4267)
    
    str_4269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 30), 'str', ': ')
    # Applying the binary operator '+' (line 164)
    result_add_4270 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 10), '+', get_caller_data_call_result_4268, str_4269)
    
    # Getting the type of 'msg' (line 164)
    msg_4271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 37), 'msg')
    # Applying the binary operator '+' (line 164)
    result_add_4272 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 35), '+', result_add_4270, msg_4271)
    
    # Assigning a type to the variable 'txt' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'txt', result_add_4272)
    # Getting the type of 'ColorType' (line 165)
    ColorType_4273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 7), 'ColorType')
    # Obtaining the member 'ANSIColors' of a type (line 165)
    ANSIColors_4274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 7), ColorType_4273, 'ANSIColors')
    # Testing if the type of an if condition is none (line 165)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 165, 4), ANSIColors_4274):
        
        # Assigning a Call to a Tuple (line 169):
        
        # Assigning a Call to a Name:
        
        # Call to setup_handles(...): (line 169)
        # Processing the call keyword arguments (line 169)
        kwargs_4280 = {}
        # Getting the type of 'setup_handles' (line 169)
        setup_handles_4279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 24), 'setup_handles', False)
        # Calling setup_handles(args, kwargs) (line 169)
        setup_handles_call_result_4281 = invoke(stypy.reporting.localization.Localization(__file__, 169, 24), setup_handles_4279, *[], **kwargs_4280)
        
        # Assigning a type to the variable 'call_assignment_3932' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_3932', setup_handles_call_result_4281)
        
        # Assigning a Call to a Name (line 169):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3932' (line 169)
        call_assignment_3932_4282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_3932', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4283 = stypy_get_value_from_tuple(call_assignment_3932_4282, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_3933' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_3933', stypy_get_value_from_tuple_call_result_4283)
        
        # Assigning a Name to a Name (line 169):
        # Getting the type of 'call_assignment_3933' (line 169)
        call_assignment_3933_4284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_3933')
        # Assigning a type to the variable 'handle' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'handle', call_assignment_3933_4284)
        
        # Assigning a Call to a Name (line 169):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3932' (line 169)
        call_assignment_3932_4285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_3932', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4286 = stypy_get_value_from_tuple(call_assignment_3932_4285, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_3934' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_3934', stypy_get_value_from_tuple_call_result_4286)
        
        # Assigning a Name to a Name (line 169):
        # Getting the type of 'call_assignment_3934' (line 169)
        call_assignment_3934_4287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_3934')
        # Assigning a type to the variable 'reset' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'reset', call_assignment_3934_4287)
        
        # Call to SetConsoleTextAttribute(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'handle' (line 170)
        handle_4292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 55), 'handle', False)
        # Getting the type of 'Colors' (line 170)
        Colors_4293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 63), 'Colors', False)
        # Obtaining the member 'WIN_WHITE' of a type (line 170)
        WIN_WHITE_4294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 63), Colors_4293, 'WIN_WHITE')
        # Processing the call keyword arguments (line 170)
        kwargs_4295 = {}
        # Getting the type of 'ctypes' (line 170)
        ctypes_4288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 170)
        windll_4289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), ctypes_4288, 'windll')
        # Obtaining the member 'kernel32' of a type (line 170)
        kernel32_4290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), windll_4289, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 170)
        SetConsoleTextAttribute_4291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), kernel32_4290, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 170)
        SetConsoleTextAttribute_call_result_4296 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), SetConsoleTextAttribute_4291, *[handle_4292, WIN_WHITE_4294], **kwargs_4295)
        
        # Getting the type of 'output_to_console' (line 172)
        output_to_console_4297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 11), 'output_to_console')
        # Testing if the type of an if condition is none (line 172)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 172, 8), output_to_console_4297):
            pass
        else:
            
            # Testing the type of an if condition (line 172)
            if_condition_4298 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 8), output_to_console_4297)
            # Assigning a type to the variable 'if_condition_4298' (line 172)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'if_condition_4298', if_condition_4298)
            # SSA begins for if statement (line 172)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'txt' (line 173)
            txt_4299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 'txt')
            # SSA join for if statement (line 172)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to SetConsoleTextAttribute(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'handle' (line 175)
        handle_4304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 55), 'handle', False)
        # Getting the type of 'reset' (line 175)
        reset_4305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 63), 'reset', False)
        # Processing the call keyword arguments (line 175)
        kwargs_4306 = {}
        # Getting the type of 'ctypes' (line 175)
        ctypes_4300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 175)
        windll_4301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), ctypes_4300, 'windll')
        # Obtaining the member 'kernel32' of a type (line 175)
        kernel32_4302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), windll_4301, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 175)
        SetConsoleTextAttribute_4303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), kernel32_4302, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 175)
        SetConsoleTextAttribute_call_result_4307 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), SetConsoleTextAttribute_4303, *[handle_4304, reset_4305], **kwargs_4306)
        
    else:
        
        # Testing the type of an if condition (line 165)
        if_condition_4275 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 4), ANSIColors_4274)
        # Assigning a type to the variable 'if_condition_4275' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'if_condition_4275', if_condition_4275)
        # SSA begins for if statement (line 165)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'output_to_console' (line 166)
        output_to_console_4276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 11), 'output_to_console')
        # Testing if the type of an if condition is none (line 166)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 166, 8), output_to_console_4276):
            pass
        else:
            
            # Testing the type of an if condition (line 166)
            if_condition_4277 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 8), output_to_console_4276)
            # Assigning a type to the variable 'if_condition_4277' (line 166)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'if_condition_4277', if_condition_4277)
            # SSA begins for if statement (line 166)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'txt' (line 167)
            txt_4278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'txt')
            # SSA join for if statement (line 166)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA branch for the else part of an if statement (line 165)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Tuple (line 169):
        
        # Assigning a Call to a Name:
        
        # Call to setup_handles(...): (line 169)
        # Processing the call keyword arguments (line 169)
        kwargs_4280 = {}
        # Getting the type of 'setup_handles' (line 169)
        setup_handles_4279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 24), 'setup_handles', False)
        # Calling setup_handles(args, kwargs) (line 169)
        setup_handles_call_result_4281 = invoke(stypy.reporting.localization.Localization(__file__, 169, 24), setup_handles_4279, *[], **kwargs_4280)
        
        # Assigning a type to the variable 'call_assignment_3932' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_3932', setup_handles_call_result_4281)
        
        # Assigning a Call to a Name (line 169):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3932' (line 169)
        call_assignment_3932_4282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_3932', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4283 = stypy_get_value_from_tuple(call_assignment_3932_4282, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_3933' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_3933', stypy_get_value_from_tuple_call_result_4283)
        
        # Assigning a Name to a Name (line 169):
        # Getting the type of 'call_assignment_3933' (line 169)
        call_assignment_3933_4284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_3933')
        # Assigning a type to the variable 'handle' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'handle', call_assignment_3933_4284)
        
        # Assigning a Call to a Name (line 169):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3932' (line 169)
        call_assignment_3932_4285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_3932', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4286 = stypy_get_value_from_tuple(call_assignment_3932_4285, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_3934' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_3934', stypy_get_value_from_tuple_call_result_4286)
        
        # Assigning a Name to a Name (line 169):
        # Getting the type of 'call_assignment_3934' (line 169)
        call_assignment_3934_4287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_3934')
        # Assigning a type to the variable 'reset' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'reset', call_assignment_3934_4287)
        
        # Call to SetConsoleTextAttribute(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'handle' (line 170)
        handle_4292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 55), 'handle', False)
        # Getting the type of 'Colors' (line 170)
        Colors_4293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 63), 'Colors', False)
        # Obtaining the member 'WIN_WHITE' of a type (line 170)
        WIN_WHITE_4294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 63), Colors_4293, 'WIN_WHITE')
        # Processing the call keyword arguments (line 170)
        kwargs_4295 = {}
        # Getting the type of 'ctypes' (line 170)
        ctypes_4288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 170)
        windll_4289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), ctypes_4288, 'windll')
        # Obtaining the member 'kernel32' of a type (line 170)
        kernel32_4290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), windll_4289, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 170)
        SetConsoleTextAttribute_4291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), kernel32_4290, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 170)
        SetConsoleTextAttribute_call_result_4296 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), SetConsoleTextAttribute_4291, *[handle_4292, WIN_WHITE_4294], **kwargs_4295)
        
        # Getting the type of 'output_to_console' (line 172)
        output_to_console_4297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 11), 'output_to_console')
        # Testing if the type of an if condition is none (line 172)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 172, 8), output_to_console_4297):
            pass
        else:
            
            # Testing the type of an if condition (line 172)
            if_condition_4298 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 8), output_to_console_4297)
            # Assigning a type to the variable 'if_condition_4298' (line 172)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'if_condition_4298', if_condition_4298)
            # SSA begins for if statement (line 172)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'txt' (line 173)
            txt_4299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 'txt')
            # SSA join for if statement (line 172)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to SetConsoleTextAttribute(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'handle' (line 175)
        handle_4304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 55), 'handle', False)
        # Getting the type of 'reset' (line 175)
        reset_4305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 63), 'reset', False)
        # Processing the call keyword arguments (line 175)
        kwargs_4306 = {}
        # Getting the type of 'ctypes' (line 175)
        ctypes_4300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 175)
        windll_4301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), ctypes_4300, 'windll')
        # Obtaining the member 'kernel32' of a type (line 175)
        kernel32_4302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), windll_4301, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 175)
        SetConsoleTextAttribute_4303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), kernel32_4302, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 175)
        SetConsoleTextAttribute_call_result_4307 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), SetConsoleTextAttribute_4303, *[handle_4304, reset_4305], **kwargs_4306)
        
        # SSA join for if statement (line 165)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to log(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'txt' (line 177)
    txt_4309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'txt', False)
    # Processing the call keyword arguments (line 177)
    kwargs_4310 = {}
    # Getting the type of 'log' (line 177)
    log_4308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'log', False)
    # Calling log(args, kwargs) (line 177)
    log_call_result_4311 = invoke(stypy.reporting.localization.Localization(__file__, 177, 4), log_4308, *[txt_4309], **kwargs_4310)
    
    
    # ################# End of 'info(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'info' in the type store
    # Getting the type of 'stypy_return_type' (line 158)
    stypy_return_type_4312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4312)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'info'
    return stypy_return_type_4312

# Assigning a type to the variable 'info' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'info', info)

@norecursion
def __aux_warning_and_error_write(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__aux_warning_and_error_write'
    module_type_store = module_type_store.open_function_context('__aux_warning_and_error_write', 180, 0, False)
    
    # Passed parameters checking function
    __aux_warning_and_error_write.stypy_localization = localization
    __aux_warning_and_error_write.stypy_type_of_self = None
    __aux_warning_and_error_write.stypy_type_store = module_type_store
    __aux_warning_and_error_write.stypy_function_name = '__aux_warning_and_error_write'
    __aux_warning_and_error_write.stypy_param_names_list = ['msg', 'call_data', 'ansi_console_color', 'win_console_color', 'file_name', 'msg_type']
    __aux_warning_and_error_write.stypy_varargs_param_name = None
    __aux_warning_and_error_write.stypy_kwargs_param_name = None
    __aux_warning_and_error_write.stypy_call_defaults = defaults
    __aux_warning_and_error_write.stypy_call_varargs = varargs
    __aux_warning_and_error_write.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__aux_warning_and_error_write', ['msg', 'call_data', 'ansi_console_color', 'win_console_color', 'file_name', 'msg_type'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__aux_warning_and_error_write', localization, ['msg', 'call_data', 'ansi_console_color', 'win_console_color', 'file_name', 'msg_type'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__aux_warning_and_error_write(...)' code ##################

    str_4313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, (-1)), 'str', '\n    Helper function to output warning or error messages, depending on its parameters.\n    :param msg: Message to print\n    :param call_data: Caller information\n    :param ansi_console_color: ANSI terminals color to use\n    :param win_console_color: Windows terminals color to use\n    :param file_name: File to write to\n    :param msg_type: Type of message to write (WARNING/ERROR)\n    :return:\n    ')
    # Getting the type of 'ColorType' (line 191)
    ColorType_4314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 7), 'ColorType')
    # Obtaining the member 'ANSIColors' of a type (line 191)
    ANSIColors_4315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 7), ColorType_4314, 'ANSIColors')
    # Testing if the type of an if condition is none (line 191)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 191, 4), ANSIColors_4315):
        
        # Assigning a Call to a Tuple (line 196):
        
        # Assigning a Call to a Name:
        
        # Call to setup_handles(...): (line 196)
        # Processing the call keyword arguments (line 196)
        kwargs_4341 = {}
        # Getting the type of 'setup_handles' (line 196)
        setup_handles_4340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 24), 'setup_handles', False)
        # Calling setup_handles(args, kwargs) (line 196)
        setup_handles_call_result_4342 = invoke(stypy.reporting.localization.Localization(__file__, 196, 24), setup_handles_4340, *[], **kwargs_4341)
        
        # Assigning a type to the variable 'call_assignment_3935' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_3935', setup_handles_call_result_4342)
        
        # Assigning a Call to a Name (line 196):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3935' (line 196)
        call_assignment_3935_4343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_3935', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4344 = stypy_get_value_from_tuple(call_assignment_3935_4343, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_3936' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_3936', stypy_get_value_from_tuple_call_result_4344)
        
        # Assigning a Name to a Name (line 196):
        # Getting the type of 'call_assignment_3936' (line 196)
        call_assignment_3936_4345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_3936')
        # Assigning a type to the variable 'handle' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'handle', call_assignment_3936_4345)
        
        # Assigning a Call to a Name (line 196):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3935' (line 196)
        call_assignment_3935_4346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_3935', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4347 = stypy_get_value_from_tuple(call_assignment_3935_4346, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_3937' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_3937', stypy_get_value_from_tuple_call_result_4347)
        
        # Assigning a Name to a Name (line 196):
        # Getting the type of 'call_assignment_3937' (line 196)
        call_assignment_3937_4348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_3937')
        # Assigning a type to the variable 'reset' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'reset', call_assignment_3937_4348)
        
        # Call to SetConsoleTextAttribute(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'handle' (line 197)
        handle_4353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 55), 'handle', False)
        # Getting the type of 'win_console_color' (line 197)
        win_console_color_4354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 63), 'win_console_color', False)
        # Processing the call keyword arguments (line 197)
        kwargs_4355 = {}
        # Getting the type of 'ctypes' (line 197)
        ctypes_4349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 197)
        windll_4350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), ctypes_4349, 'windll')
        # Obtaining the member 'kernel32' of a type (line 197)
        kernel32_4351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), windll_4350, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 197)
        SetConsoleTextAttribute_4352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), kernel32_4351, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 197)
        SetConsoleTextAttribute_call_result_4356 = invoke(stypy.reporting.localization.Localization(__file__, 197, 8), SetConsoleTextAttribute_4352, *[handle_4353, win_console_color_4354], **kwargs_4355)
        
        # Getting the type of 'output_to_console' (line 199)
        output_to_console_4357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 11), 'output_to_console')
        # Testing if the type of an if condition is none (line 199)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 199, 8), output_to_console_4357):
            pass
        else:
            
            # Testing the type of an if condition (line 199)
            if_condition_4358 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 8), output_to_console_4357)
            # Assigning a type to the variable 'if_condition_4358' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'if_condition_4358', if_condition_4358)
            # SSA begins for if statement (line 199)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'msg_type' (line 200)
            msg_type_4359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 18), 'msg_type')
            str_4360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 29), 'str', ': ')
            # Applying the binary operator '+' (line 200)
            result_add_4361 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 18), '+', msg_type_4359, str_4360)
            
            # Getting the type of 'msg' (line 200)
            msg_4362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 36), 'msg')
            # Applying the binary operator '+' (line 200)
            result_add_4363 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 34), '+', result_add_4361, msg_4362)
            
            # SSA join for if statement (line 199)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to SetConsoleTextAttribute(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'handle' (line 202)
        handle_4368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 55), 'handle', False)
        # Getting the type of 'reset' (line 202)
        reset_4369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 63), 'reset', False)
        # Processing the call keyword arguments (line 202)
        kwargs_4370 = {}
        # Getting the type of 'ctypes' (line 202)
        ctypes_4364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 202)
        windll_4365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), ctypes_4364, 'windll')
        # Obtaining the member 'kernel32' of a type (line 202)
        kernel32_4366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), windll_4365, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 202)
        SetConsoleTextAttribute_4367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), kernel32_4366, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 202)
        SetConsoleTextAttribute_call_result_4371 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), SetConsoleTextAttribute_4367, *[handle_4368, reset_4369], **kwargs_4370)
        
    else:
        
        # Testing the type of an if condition (line 191)
        if_condition_4316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 4), ANSIColors_4315)
        # Assigning a type to the variable 'if_condition_4316' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'if_condition_4316', if_condition_4316)
        # SSA begins for if statement (line 191)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'output_to_console' (line 192)
        output_to_console_4317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 11), 'output_to_console')
        # Testing if the type of an if condition is none (line 192)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 192, 8), output_to_console_4317):
            pass
        else:
            
            # Testing the type of an if condition (line 192)
            if_condition_4318 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 8), output_to_console_4317)
            # Assigning a type to the variable 'if_condition_4318' (line 192)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'if_condition_4318', if_condition_4318)
            # SSA begins for if statement (line 192)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 193):
            
            # Assigning a BinOp to a Name (line 193):
            
            # Call to str(...): (line 193)
            # Processing the call arguments (line 193)
            # Getting the type of 'call_data' (line 193)
            call_data_4320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 22), 'call_data', False)
            # Processing the call keyword arguments (line 193)
            kwargs_4321 = {}
            # Getting the type of 'str' (line 193)
            str_4319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 18), 'str', False)
            # Calling str(args, kwargs) (line 193)
            str_call_result_4322 = invoke(stypy.reporting.localization.Localization(__file__, 193, 18), str_4319, *[call_data_4320], **kwargs_4321)
            
            str_4323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 35), 'str', '. ')
            # Applying the binary operator '+' (line 193)
            result_add_4324 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 18), '+', str_call_result_4322, str_4323)
            
            # Getting the type of 'msg_type' (line 193)
            msg_type_4325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 42), 'msg_type')
            # Applying the binary operator '+' (line 193)
            result_add_4326 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 40), '+', result_add_4324, msg_type_4325)
            
            str_4327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 53), 'str', ': ')
            # Applying the binary operator '+' (line 193)
            result_add_4328 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 51), '+', result_add_4326, str_4327)
            
            
            # Call to str(...): (line 193)
            # Processing the call arguments (line 193)
            # Getting the type of 'msg' (line 193)
            msg_4330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 64), 'msg', False)
            # Processing the call keyword arguments (line 193)
            kwargs_4331 = {}
            # Getting the type of 'str' (line 193)
            str_4329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 60), 'str', False)
            # Calling str(args, kwargs) (line 193)
            str_call_result_4332 = invoke(stypy.reporting.localization.Localization(__file__, 193, 60), str_4329, *[msg_4330], **kwargs_4331)
            
            # Applying the binary operator '+' (line 193)
            result_add_4333 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 58), '+', result_add_4328, str_call_result_4332)
            
            # Assigning a type to the variable 'txt' (line 193)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'txt', result_add_4333)
            # Getting the type of 'ansi_console_color' (line 194)
            ansi_console_color_4334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 18), 'ansi_console_color')
            # Getting the type of 'txt' (line 194)
            txt_4335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 39), 'txt')
            # Applying the binary operator '+' (line 194)
            result_add_4336 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 18), '+', ansi_console_color_4334, txt_4335)
            
            # Getting the type of 'Colors' (line 194)
            Colors_4337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 45), 'Colors')
            # Obtaining the member 'ANSI_ENDC' of a type (line 194)
            ANSI_ENDC_4338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 45), Colors_4337, 'ANSI_ENDC')
            # Applying the binary operator '+' (line 194)
            result_add_4339 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 43), '+', result_add_4336, ANSI_ENDC_4338)
            
            # SSA join for if statement (line 192)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA branch for the else part of an if statement (line 191)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Tuple (line 196):
        
        # Assigning a Call to a Name:
        
        # Call to setup_handles(...): (line 196)
        # Processing the call keyword arguments (line 196)
        kwargs_4341 = {}
        # Getting the type of 'setup_handles' (line 196)
        setup_handles_4340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 24), 'setup_handles', False)
        # Calling setup_handles(args, kwargs) (line 196)
        setup_handles_call_result_4342 = invoke(stypy.reporting.localization.Localization(__file__, 196, 24), setup_handles_4340, *[], **kwargs_4341)
        
        # Assigning a type to the variable 'call_assignment_3935' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_3935', setup_handles_call_result_4342)
        
        # Assigning a Call to a Name (line 196):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3935' (line 196)
        call_assignment_3935_4343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_3935', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4344 = stypy_get_value_from_tuple(call_assignment_3935_4343, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_3936' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_3936', stypy_get_value_from_tuple_call_result_4344)
        
        # Assigning a Name to a Name (line 196):
        # Getting the type of 'call_assignment_3936' (line 196)
        call_assignment_3936_4345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_3936')
        # Assigning a type to the variable 'handle' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'handle', call_assignment_3936_4345)
        
        # Assigning a Call to a Name (line 196):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3935' (line 196)
        call_assignment_3935_4346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_3935', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4347 = stypy_get_value_from_tuple(call_assignment_3935_4346, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_3937' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_3937', stypy_get_value_from_tuple_call_result_4347)
        
        # Assigning a Name to a Name (line 196):
        # Getting the type of 'call_assignment_3937' (line 196)
        call_assignment_3937_4348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_3937')
        # Assigning a type to the variable 'reset' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'reset', call_assignment_3937_4348)
        
        # Call to SetConsoleTextAttribute(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'handle' (line 197)
        handle_4353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 55), 'handle', False)
        # Getting the type of 'win_console_color' (line 197)
        win_console_color_4354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 63), 'win_console_color', False)
        # Processing the call keyword arguments (line 197)
        kwargs_4355 = {}
        # Getting the type of 'ctypes' (line 197)
        ctypes_4349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 197)
        windll_4350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), ctypes_4349, 'windll')
        # Obtaining the member 'kernel32' of a type (line 197)
        kernel32_4351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), windll_4350, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 197)
        SetConsoleTextAttribute_4352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), kernel32_4351, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 197)
        SetConsoleTextAttribute_call_result_4356 = invoke(stypy.reporting.localization.Localization(__file__, 197, 8), SetConsoleTextAttribute_4352, *[handle_4353, win_console_color_4354], **kwargs_4355)
        
        # Getting the type of 'output_to_console' (line 199)
        output_to_console_4357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 11), 'output_to_console')
        # Testing if the type of an if condition is none (line 199)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 199, 8), output_to_console_4357):
            pass
        else:
            
            # Testing the type of an if condition (line 199)
            if_condition_4358 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 8), output_to_console_4357)
            # Assigning a type to the variable 'if_condition_4358' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'if_condition_4358', if_condition_4358)
            # SSA begins for if statement (line 199)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'msg_type' (line 200)
            msg_type_4359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 18), 'msg_type')
            str_4360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 29), 'str', ': ')
            # Applying the binary operator '+' (line 200)
            result_add_4361 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 18), '+', msg_type_4359, str_4360)
            
            # Getting the type of 'msg' (line 200)
            msg_4362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 36), 'msg')
            # Applying the binary operator '+' (line 200)
            result_add_4363 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 34), '+', result_add_4361, msg_4362)
            
            # SSA join for if statement (line 199)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to SetConsoleTextAttribute(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'handle' (line 202)
        handle_4368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 55), 'handle', False)
        # Getting the type of 'reset' (line 202)
        reset_4369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 63), 'reset', False)
        # Processing the call keyword arguments (line 202)
        kwargs_4370 = {}
        # Getting the type of 'ctypes' (line 202)
        ctypes_4364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 202)
        windll_4365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), ctypes_4364, 'windll')
        # Obtaining the member 'kernel32' of a type (line 202)
        kernel32_4366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), windll_4365, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 202)
        SetConsoleTextAttribute_4367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), kernel32_4366, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 202)
        SetConsoleTextAttribute_call_result_4371 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), SetConsoleTextAttribute_4367, *[handle_4368, reset_4369], **kwargs_4370)
        
        # SSA join for if statement (line 191)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # SSA begins for try-except statement (line 204)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 205):
    
    # Assigning a Call to a Name (line 205):
    
    # Call to open(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'stypy_parameters_copy' (line 205)
    stypy_parameters_copy_4373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 21), 'stypy_parameters_copy', False)
    # Obtaining the member 'LOG_PATH' of a type (line 205)
    LOG_PATH_4374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 21), stypy_parameters_copy_4373, 'LOG_PATH')
    str_4375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 54), 'str', '/')
    # Applying the binary operator '+' (line 205)
    result_add_4376 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 21), '+', LOG_PATH_4374, str_4375)
    
    # Getting the type of 'file_name' (line 205)
    file_name_4377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 60), 'file_name', False)
    # Applying the binary operator '+' (line 205)
    result_add_4378 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 58), '+', result_add_4376, file_name_4377)
    
    str_4379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 71), 'str', 'a')
    # Processing the call keyword arguments (line 205)
    kwargs_4380 = {}
    # Getting the type of 'open' (line 205)
    open_4372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 16), 'open', False)
    # Calling open(args, kwargs) (line 205)
    open_call_result_4381 = invoke(stypy.reporting.localization.Localization(__file__, 205, 16), open_4372, *[result_add_4378, str_4379], **kwargs_4380)
    
    # Assigning a type to the variable 'file_' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'file_', open_call_result_4381)
    # SSA branch for the except part of a try statement (line 204)
    # SSA branch for the except '<any exception>' branch of a try statement (line 204)
    module_type_store.open_ssa_branch('except')
    # Assigning a type to the variable 'stypy_return_type' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'stypy_return_type', types.NoneType)
    # SSA join for try-except statement (line 204)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 209):
    
    # Assigning a BinOp to a Name (line 209):
    
    # Call to str(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'call_data' (line 209)
    call_data_4383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 14), 'call_data', False)
    # Processing the call keyword arguments (line 209)
    kwargs_4384 = {}
    # Getting the type of 'str' (line 209)
    str_4382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 10), 'str', False)
    # Calling str(args, kwargs) (line 209)
    str_call_result_4385 = invoke(stypy.reporting.localization.Localization(__file__, 209, 10), str_4382, *[call_data_4383], **kwargs_4384)
    
    str_4386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 27), 'str', ' (')
    # Applying the binary operator '+' (line 209)
    result_add_4387 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 10), '+', str_call_result_4385, str_4386)
    
    
    # Call to get_date_time(...): (line 209)
    # Processing the call keyword arguments (line 209)
    kwargs_4389 = {}
    # Getting the type of 'get_date_time' (line 209)
    get_date_time_4388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 34), 'get_date_time', False)
    # Calling get_date_time(args, kwargs) (line 209)
    get_date_time_call_result_4390 = invoke(stypy.reporting.localization.Localization(__file__, 209, 34), get_date_time_4388, *[], **kwargs_4389)
    
    # Applying the binary operator '+' (line 209)
    result_add_4391 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 32), '+', result_add_4387, get_date_time_call_result_4390)
    
    str_4392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 52), 'str', '). ')
    # Applying the binary operator '+' (line 209)
    result_add_4393 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 50), '+', result_add_4391, str_4392)
    
    # Getting the type of 'msg_type' (line 209)
    msg_type_4394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 60), 'msg_type')
    # Applying the binary operator '+' (line 209)
    result_add_4395 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 58), '+', result_add_4393, msg_type_4394)
    
    str_4396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 71), 'str', ': ')
    # Applying the binary operator '+' (line 209)
    result_add_4397 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 69), '+', result_add_4395, str_4396)
    
    # Getting the type of 'msg' (line 209)
    msg_4398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 78), 'msg')
    # Applying the binary operator '+' (line 209)
    result_add_4399 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 76), '+', result_add_4397, msg_4398)
    
    # Assigning a type to the variable 'txt' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'txt', result_add_4399)
    
    # Call to write(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'txt' (line 210)
    txt_4402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'txt', False)
    str_4403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 22), 'str', '\n')
    # Applying the binary operator '+' (line 210)
    result_add_4404 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 16), '+', txt_4402, str_4403)
    
    # Processing the call keyword arguments (line 210)
    kwargs_4405 = {}
    # Getting the type of 'file_' (line 210)
    file__4400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'file_', False)
    # Obtaining the member 'write' of a type (line 210)
    write_4401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 4), file__4400, 'write')
    # Calling write(args, kwargs) (line 210)
    write_call_result_4406 = invoke(stypy.reporting.localization.Localization(__file__, 210, 4), write_4401, *[result_add_4404], **kwargs_4405)
    
    
    # Call to close(...): (line 211)
    # Processing the call keyword arguments (line 211)
    kwargs_4409 = {}
    # Getting the type of 'file_' (line 211)
    file__4407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'file_', False)
    # Obtaining the member 'close' of a type (line 211)
    close_4408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 4), file__4407, 'close')
    # Calling close(args, kwargs) (line 211)
    close_call_result_4410 = invoke(stypy.reporting.localization.Localization(__file__, 211, 4), close_4408, *[], **kwargs_4409)
    
    
    # ################# End of '__aux_warning_and_error_write(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__aux_warning_and_error_write' in the type store
    # Getting the type of 'stypy_return_type' (line 180)
    stypy_return_type_4411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4411)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__aux_warning_and_error_write'
    return stypy_return_type_4411

# Assigning a type to the variable '__aux_warning_and_error_write' (line 180)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), '__aux_warning_and_error_write', __aux_warning_and_error_write)

@norecursion
def warning(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'warning'
    module_type_store = module_type_store.open_function_context('warning', 214, 0, False)
    
    # Passed parameters checking function
    warning.stypy_localization = localization
    warning.stypy_type_of_self = None
    warning.stypy_type_store = module_type_store
    warning.stypy_function_name = 'warning'
    warning.stypy_param_names_list = ['msg']
    warning.stypy_varargs_param_name = None
    warning.stypy_kwargs_param_name = None
    warning.stypy_call_defaults = defaults
    warning.stypy_call_varargs = varargs
    warning.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'warning', ['msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'warning', localization, ['msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'warning(...)' code ##################

    str_4412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, (-1)), 'str', '\n    Proxy for __aux_warning_and_error_write, supplying parameters to write warning messages\n    :param msg:\n    :return:\n    ')
    
    # Assigning a Call to a Name (line 220):
    
    # Assigning a Call to a Name (line 220):
    
    # Call to get_caller_data(...): (line 220)
    # Processing the call keyword arguments (line 220)
    kwargs_4414 = {}
    # Getting the type of 'get_caller_data' (line 220)
    get_caller_data_4413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'get_caller_data', False)
    # Calling get_caller_data(args, kwargs) (line 220)
    get_caller_data_call_result_4415 = invoke(stypy.reporting.localization.Localization(__file__, 220, 16), get_caller_data_4413, *[], **kwargs_4414)
    
    # Assigning a type to the variable 'call_data' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'call_data', get_caller_data_call_result_4415)
    
    # Call to __aux_warning_and_error_write(...): (line 221)
    # Processing the call arguments (line 221)
    # Getting the type of 'msg' (line 221)
    msg_4417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 34), 'msg', False)
    # Getting the type of 'call_data' (line 221)
    call_data_4418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 39), 'call_data', False)
    # Getting the type of 'Colors' (line 221)
    Colors_4419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 50), 'Colors', False)
    # Obtaining the member 'ANSI_WARNING' of a type (line 221)
    ANSI_WARNING_4420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 50), Colors_4419, 'ANSI_WARNING')
    # Getting the type of 'Colors' (line 221)
    Colors_4421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 71), 'Colors', False)
    # Obtaining the member 'WIN_WARNING' of a type (line 221)
    WIN_WARNING_4422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 71), Colors_4421, 'WIN_WARNING')
    # Getting the type of 'stypy_parameters_copy' (line 222)
    stypy_parameters_copy_4423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 34), 'stypy_parameters_copy', False)
    # Obtaining the member 'WARNING_LOG_FILE' of a type (line 222)
    WARNING_LOG_FILE_4424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 34), stypy_parameters_copy_4423, 'WARNING_LOG_FILE')
    str_4425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 74), 'str', 'WARNING')
    # Processing the call keyword arguments (line 221)
    kwargs_4426 = {}
    # Getting the type of '__aux_warning_and_error_write' (line 221)
    aux_warning_and_error_write_4416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), '__aux_warning_and_error_write', False)
    # Calling __aux_warning_and_error_write(args, kwargs) (line 221)
    aux_warning_and_error_write_call_result_4427 = invoke(stypy.reporting.localization.Localization(__file__, 221, 4), aux_warning_and_error_write_4416, *[msg_4417, call_data_4418, ANSI_WARNING_4420, WIN_WARNING_4422, WARNING_LOG_FILE_4424, str_4425], **kwargs_4426)
    
    
    # ################# End of 'warning(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'warning' in the type store
    # Getting the type of 'stypy_return_type' (line 214)
    stypy_return_type_4428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4428)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'warning'
    return stypy_return_type_4428

# Assigning a type to the variable 'warning' (line 214)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 0), 'warning', warning)

@norecursion
def error(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'error'
    module_type_store = module_type_store.open_function_context('error', 225, 0, False)
    
    # Passed parameters checking function
    error.stypy_localization = localization
    error.stypy_type_of_self = None
    error.stypy_type_store = module_type_store
    error.stypy_function_name = 'error'
    error.stypy_param_names_list = ['msg']
    error.stypy_varargs_param_name = None
    error.stypy_kwargs_param_name = None
    error.stypy_call_defaults = defaults
    error.stypy_call_varargs = varargs
    error.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'error', ['msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'error', localization, ['msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'error(...)' code ##################

    str_4429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, (-1)), 'str', '\n    Proxy for __aux_warning_and_error_write, supplying parameters to write error messages\n    :param msg:\n    :return:\n    ')
    
    # Assigning a Call to a Name (line 231):
    
    # Assigning a Call to a Name (line 231):
    
    # Call to get_caller_data(...): (line 231)
    # Processing the call keyword arguments (line 231)
    kwargs_4431 = {}
    # Getting the type of 'get_caller_data' (line 231)
    get_caller_data_4430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'get_caller_data', False)
    # Calling get_caller_data(args, kwargs) (line 231)
    get_caller_data_call_result_4432 = invoke(stypy.reporting.localization.Localization(__file__, 231, 16), get_caller_data_4430, *[], **kwargs_4431)
    
    # Assigning a type to the variable 'call_data' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'call_data', get_caller_data_call_result_4432)
    
    # Call to __aux_warning_and_error_write(...): (line 232)
    # Processing the call arguments (line 232)
    # Getting the type of 'msg' (line 232)
    msg_4434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 34), 'msg', False)
    # Getting the type of 'call_data' (line 232)
    call_data_4435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 39), 'call_data', False)
    # Getting the type of 'Colors' (line 232)
    Colors_4436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 50), 'Colors', False)
    # Obtaining the member 'ANSI_FAIL' of a type (line 232)
    ANSI_FAIL_4437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 50), Colors_4436, 'ANSI_FAIL')
    # Getting the type of 'Colors' (line 232)
    Colors_4438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 68), 'Colors', False)
    # Obtaining the member 'WIN_FAIL' of a type (line 232)
    WIN_FAIL_4439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 68), Colors_4438, 'WIN_FAIL')
    # Getting the type of 'stypy_parameters_copy' (line 232)
    stypy_parameters_copy_4440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 85), 'stypy_parameters_copy', False)
    # Obtaining the member 'ERROR_LOG_FILE' of a type (line 232)
    ERROR_LOG_FILE_4441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 85), stypy_parameters_copy_4440, 'ERROR_LOG_FILE')
    str_4442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 34), 'str', 'ERROR')
    # Processing the call keyword arguments (line 232)
    kwargs_4443 = {}
    # Getting the type of '__aux_warning_and_error_write' (line 232)
    aux_warning_and_error_write_4433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), '__aux_warning_and_error_write', False)
    # Calling __aux_warning_and_error_write(args, kwargs) (line 232)
    aux_warning_and_error_write_call_result_4444 = invoke(stypy.reporting.localization.Localization(__file__, 232, 4), aux_warning_and_error_write_4433, *[msg_4434, call_data_4435, ANSI_FAIL_4437, WIN_FAIL_4439, ERROR_LOG_FILE_4441, str_4442], **kwargs_4443)
    
    
    # ################# End of 'error(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'error' in the type store
    # Getting the type of 'stypy_return_type' (line 225)
    stypy_return_type_4445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4445)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'error'
    return stypy_return_type_4445

# Assigning a type to the variable 'error' (line 225)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'error', error)

@norecursion
def new_logging_session(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'new_logging_session'
    module_type_store = module_type_store.open_function_context('new_logging_session', 236, 0, False)
    
    # Passed parameters checking function
    new_logging_session.stypy_localization = localization
    new_logging_session.stypy_type_of_self = None
    new_logging_session.stypy_type_store = module_type_store
    new_logging_session.stypy_function_name = 'new_logging_session'
    new_logging_session.stypy_param_names_list = []
    new_logging_session.stypy_varargs_param_name = None
    new_logging_session.stypy_kwargs_param_name = None
    new_logging_session.stypy_call_defaults = defaults
    new_logging_session.stypy_call_varargs = varargs
    new_logging_session.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'new_logging_session', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'new_logging_session', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'new_logging_session(...)' code ##################

    str_4446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, (-1)), 'str', '\n    Put a header to the log files indicating that log messages below that header belong to a new execution\n    :return:\n    ')
    
    
    # SSA begins for try-except statement (line 241)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 242):
    
    # Assigning a Call to a Name (line 242):
    
    # Call to open(...): (line 242)
    # Processing the call arguments (line 242)
    # Getting the type of 'stypy_parameters_copy' (line 242)
    stypy_parameters_copy_4448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 21), 'stypy_parameters_copy', False)
    # Obtaining the member 'LOG_PATH' of a type (line 242)
    LOG_PATH_4449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 21), stypy_parameters_copy_4448, 'LOG_PATH')
    str_4450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 54), 'str', '/')
    # Applying the binary operator '+' (line 242)
    result_add_4451 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 21), '+', LOG_PATH_4449, str_4450)
    
    # Getting the type of 'stypy_parameters_copy' (line 242)
    stypy_parameters_copy_4452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 60), 'stypy_parameters_copy', False)
    # Obtaining the member 'ERROR_LOG_FILE' of a type (line 242)
    ERROR_LOG_FILE_4453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 60), stypy_parameters_copy_4452, 'ERROR_LOG_FILE')
    # Applying the binary operator '+' (line 242)
    result_add_4454 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 58), '+', result_add_4451, ERROR_LOG_FILE_4453)
    
    str_4455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 98), 'str', 'a')
    # Processing the call keyword arguments (line 242)
    kwargs_4456 = {}
    # Getting the type of 'open' (line 242)
    open_4447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'open', False)
    # Calling open(args, kwargs) (line 242)
    open_call_result_4457 = invoke(stypy.reporting.localization.Localization(__file__, 242, 16), open_4447, *[result_add_4454, str_4455], **kwargs_4456)
    
    # Assigning a type to the variable 'file_' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'file_', open_call_result_4457)
    
    # Call to write(...): (line 243)
    # Processing the call arguments (line 243)
    str_4460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 20), 'str', '\n\n')
    # Processing the call keyword arguments (line 243)
    kwargs_4461 = {}
    # Getting the type of 'file_' (line 243)
    file__4458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 243)
    write_4459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), file__4458, 'write')
    # Calling write(args, kwargs) (line 243)
    write_call_result_4462 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), write_4459, *[str_4460], **kwargs_4461)
    
    
    # Call to write(...): (line 244)
    # Processing the call arguments (line 244)
    str_4465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 20), 'str', 'NEW LOGGING SESSION BEGIN AT: ')
    
    # Call to get_date_time(...): (line 244)
    # Processing the call keyword arguments (line 244)
    kwargs_4467 = {}
    # Getting the type of 'get_date_time' (line 244)
    get_date_time_4466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 55), 'get_date_time', False)
    # Calling get_date_time(args, kwargs) (line 244)
    get_date_time_call_result_4468 = invoke(stypy.reporting.localization.Localization(__file__, 244, 55), get_date_time_4466, *[], **kwargs_4467)
    
    # Applying the binary operator '+' (line 244)
    result_add_4469 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 20), '+', str_4465, get_date_time_call_result_4468)
    
    # Processing the call keyword arguments (line 244)
    kwargs_4470 = {}
    # Getting the type of 'file_' (line 244)
    file__4463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 244)
    write_4464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), file__4463, 'write')
    # Calling write(args, kwargs) (line 244)
    write_call_result_4471 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), write_4464, *[result_add_4469], **kwargs_4470)
    
    
    # Call to write(...): (line 245)
    # Processing the call arguments (line 245)
    str_4474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 20), 'str', '\n\n')
    # Processing the call keyword arguments (line 245)
    kwargs_4475 = {}
    # Getting the type of 'file_' (line 245)
    file__4472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 245)
    write_4473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 8), file__4472, 'write')
    # Calling write(args, kwargs) (line 245)
    write_call_result_4476 = invoke(stypy.reporting.localization.Localization(__file__, 245, 8), write_4473, *[str_4474], **kwargs_4475)
    
    
    # Call to close(...): (line 246)
    # Processing the call keyword arguments (line 246)
    kwargs_4479 = {}
    # Getting the type of 'file_' (line 246)
    file__4477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'file_', False)
    # Obtaining the member 'close' of a type (line 246)
    close_4478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), file__4477, 'close')
    # Calling close(args, kwargs) (line 246)
    close_call_result_4480 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), close_4478, *[], **kwargs_4479)
    
    
    # Assigning a Call to a Name (line 248):
    
    # Assigning a Call to a Name (line 248):
    
    # Call to open(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'stypy_parameters_copy' (line 248)
    stypy_parameters_copy_4482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 21), 'stypy_parameters_copy', False)
    # Obtaining the member 'LOG_PATH' of a type (line 248)
    LOG_PATH_4483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 21), stypy_parameters_copy_4482, 'LOG_PATH')
    str_4484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 54), 'str', '/')
    # Applying the binary operator '+' (line 248)
    result_add_4485 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 21), '+', LOG_PATH_4483, str_4484)
    
    # Getting the type of 'stypy_parameters_copy' (line 248)
    stypy_parameters_copy_4486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 60), 'stypy_parameters_copy', False)
    # Obtaining the member 'INFO_LOG_FILE' of a type (line 248)
    INFO_LOG_FILE_4487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 60), stypy_parameters_copy_4486, 'INFO_LOG_FILE')
    # Applying the binary operator '+' (line 248)
    result_add_4488 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 58), '+', result_add_4485, INFO_LOG_FILE_4487)
    
    str_4489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 97), 'str', 'a')
    # Processing the call keyword arguments (line 248)
    kwargs_4490 = {}
    # Getting the type of 'open' (line 248)
    open_4481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'open', False)
    # Calling open(args, kwargs) (line 248)
    open_call_result_4491 = invoke(stypy.reporting.localization.Localization(__file__, 248, 16), open_4481, *[result_add_4488, str_4489], **kwargs_4490)
    
    # Assigning a type to the variable 'file_' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'file_', open_call_result_4491)
    
    # Call to write(...): (line 249)
    # Processing the call arguments (line 249)
    str_4494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 20), 'str', '\n\n')
    # Processing the call keyword arguments (line 249)
    kwargs_4495 = {}
    # Getting the type of 'file_' (line 249)
    file__4492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 249)
    write_4493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), file__4492, 'write')
    # Calling write(args, kwargs) (line 249)
    write_call_result_4496 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), write_4493, *[str_4494], **kwargs_4495)
    
    
    # Call to write(...): (line 250)
    # Processing the call arguments (line 250)
    str_4499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 20), 'str', 'NEW LOGGING SESSION BEGIN AT: ')
    
    # Call to get_date_time(...): (line 250)
    # Processing the call keyword arguments (line 250)
    kwargs_4501 = {}
    # Getting the type of 'get_date_time' (line 250)
    get_date_time_4500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 55), 'get_date_time', False)
    # Calling get_date_time(args, kwargs) (line 250)
    get_date_time_call_result_4502 = invoke(stypy.reporting.localization.Localization(__file__, 250, 55), get_date_time_4500, *[], **kwargs_4501)
    
    # Applying the binary operator '+' (line 250)
    result_add_4503 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 20), '+', str_4499, get_date_time_call_result_4502)
    
    # Processing the call keyword arguments (line 250)
    kwargs_4504 = {}
    # Getting the type of 'file_' (line 250)
    file__4497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 250)
    write_4498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), file__4497, 'write')
    # Calling write(args, kwargs) (line 250)
    write_call_result_4505 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), write_4498, *[result_add_4503], **kwargs_4504)
    
    
    # Call to write(...): (line 251)
    # Processing the call arguments (line 251)
    str_4508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 20), 'str', '\n\n')
    # Processing the call keyword arguments (line 251)
    kwargs_4509 = {}
    # Getting the type of 'file_' (line 251)
    file__4506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 251)
    write_4507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), file__4506, 'write')
    # Calling write(args, kwargs) (line 251)
    write_call_result_4510 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), write_4507, *[str_4508], **kwargs_4509)
    
    
    # Call to close(...): (line 252)
    # Processing the call keyword arguments (line 252)
    kwargs_4513 = {}
    # Getting the type of 'file_' (line 252)
    file__4511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'file_', False)
    # Obtaining the member 'close' of a type (line 252)
    close_4512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 8), file__4511, 'close')
    # Calling close(args, kwargs) (line 252)
    close_call_result_4514 = invoke(stypy.reporting.localization.Localization(__file__, 252, 8), close_4512, *[], **kwargs_4513)
    
    
    # Assigning a Call to a Name (line 254):
    
    # Assigning a Call to a Name (line 254):
    
    # Call to open(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'stypy_parameters_copy' (line 254)
    stypy_parameters_copy_4516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 21), 'stypy_parameters_copy', False)
    # Obtaining the member 'LOG_PATH' of a type (line 254)
    LOG_PATH_4517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 21), stypy_parameters_copy_4516, 'LOG_PATH')
    str_4518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 54), 'str', '/')
    # Applying the binary operator '+' (line 254)
    result_add_4519 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 21), '+', LOG_PATH_4517, str_4518)
    
    # Getting the type of 'stypy_parameters_copy' (line 254)
    stypy_parameters_copy_4520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 60), 'stypy_parameters_copy', False)
    # Obtaining the member 'WARNING_LOG_FILE' of a type (line 254)
    WARNING_LOG_FILE_4521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 60), stypy_parameters_copy_4520, 'WARNING_LOG_FILE')
    # Applying the binary operator '+' (line 254)
    result_add_4522 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 58), '+', result_add_4519, WARNING_LOG_FILE_4521)
    
    str_4523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 100), 'str', 'a')
    # Processing the call keyword arguments (line 254)
    kwargs_4524 = {}
    # Getting the type of 'open' (line 254)
    open_4515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'open', False)
    # Calling open(args, kwargs) (line 254)
    open_call_result_4525 = invoke(stypy.reporting.localization.Localization(__file__, 254, 16), open_4515, *[result_add_4522, str_4523], **kwargs_4524)
    
    # Assigning a type to the variable 'file_' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'file_', open_call_result_4525)
    
    # Call to write(...): (line 255)
    # Processing the call arguments (line 255)
    str_4528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 20), 'str', '\n\n')
    # Processing the call keyword arguments (line 255)
    kwargs_4529 = {}
    # Getting the type of 'file_' (line 255)
    file__4526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 255)
    write_4527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 8), file__4526, 'write')
    # Calling write(args, kwargs) (line 255)
    write_call_result_4530 = invoke(stypy.reporting.localization.Localization(__file__, 255, 8), write_4527, *[str_4528], **kwargs_4529)
    
    
    # Call to write(...): (line 256)
    # Processing the call arguments (line 256)
    str_4533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 20), 'str', 'NEW LOGGING SESSION BEGIN AT: ')
    
    # Call to get_date_time(...): (line 256)
    # Processing the call keyword arguments (line 256)
    kwargs_4535 = {}
    # Getting the type of 'get_date_time' (line 256)
    get_date_time_4534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 55), 'get_date_time', False)
    # Calling get_date_time(args, kwargs) (line 256)
    get_date_time_call_result_4536 = invoke(stypy.reporting.localization.Localization(__file__, 256, 55), get_date_time_4534, *[], **kwargs_4535)
    
    # Applying the binary operator '+' (line 256)
    result_add_4537 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 20), '+', str_4533, get_date_time_call_result_4536)
    
    # Processing the call keyword arguments (line 256)
    kwargs_4538 = {}
    # Getting the type of 'file_' (line 256)
    file__4531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 256)
    write_4532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 8), file__4531, 'write')
    # Calling write(args, kwargs) (line 256)
    write_call_result_4539 = invoke(stypy.reporting.localization.Localization(__file__, 256, 8), write_4532, *[result_add_4537], **kwargs_4538)
    
    
    # Call to write(...): (line 257)
    # Processing the call arguments (line 257)
    str_4542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 20), 'str', '\n\n')
    # Processing the call keyword arguments (line 257)
    kwargs_4543 = {}
    # Getting the type of 'file_' (line 257)
    file__4540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 257)
    write_4541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), file__4540, 'write')
    # Calling write(args, kwargs) (line 257)
    write_call_result_4544 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), write_4541, *[str_4542], **kwargs_4543)
    
    
    # Call to close(...): (line 258)
    # Processing the call keyword arguments (line 258)
    kwargs_4547 = {}
    # Getting the type of 'file_' (line 258)
    file__4545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'file_', False)
    # Obtaining the member 'close' of a type (line 258)
    close_4546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 8), file__4545, 'close')
    # Calling close(args, kwargs) (line 258)
    close_call_result_4548 = invoke(stypy.reporting.localization.Localization(__file__, 258, 8), close_4546, *[], **kwargs_4547)
    
    # SSA branch for the except part of a try statement (line 241)
    # SSA branch for the except '<any exception>' branch of a try statement (line 241)
    module_type_store.open_ssa_branch('except')
    # Assigning a type to the variable 'stypy_return_type' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'stypy_return_type', types.NoneType)
    # SSA join for try-except statement (line 241)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'new_logging_session(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'new_logging_session' in the type store
    # Getting the type of 'stypy_return_type' (line 236)
    stypy_return_type_4549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4549)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'new_logging_session'
    return stypy_return_type_4549

# Assigning a type to the variable 'new_logging_session' (line 236)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 0), 'new_logging_session', new_logging_session)

@norecursion
def reset_logs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'reset_logs'
    module_type_store = module_type_store.open_function_context('reset_logs', 263, 0, False)
    
    # Passed parameters checking function
    reset_logs.stypy_localization = localization
    reset_logs.stypy_type_of_self = None
    reset_logs.stypy_type_store = module_type_store
    reset_logs.stypy_function_name = 'reset_logs'
    reset_logs.stypy_param_names_list = []
    reset_logs.stypy_varargs_param_name = None
    reset_logs.stypy_kwargs_param_name = None
    reset_logs.stypy_call_defaults = defaults
    reset_logs.stypy_call_varargs = varargs
    reset_logs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'reset_logs', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'reset_logs', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'reset_logs(...)' code ##################

    str_4550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, (-1)), 'str', '\n    Erases log files\n    :return:\n    ')
    
    
    # SSA begins for try-except statement (line 268)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 269):
    
    # Assigning a Call to a Name (line 269):
    
    # Call to open(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'stypy_parameters_copy' (line 269)
    stypy_parameters_copy_4552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 21), 'stypy_parameters_copy', False)
    # Obtaining the member 'LOG_PATH' of a type (line 269)
    LOG_PATH_4553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 21), stypy_parameters_copy_4552, 'LOG_PATH')
    str_4554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 54), 'str', '/')
    # Applying the binary operator '+' (line 269)
    result_add_4555 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 21), '+', LOG_PATH_4553, str_4554)
    
    # Getting the type of 'stypy_parameters_copy' (line 269)
    stypy_parameters_copy_4556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 60), 'stypy_parameters_copy', False)
    # Obtaining the member 'ERROR_LOG_FILE' of a type (line 269)
    ERROR_LOG_FILE_4557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 60), stypy_parameters_copy_4556, 'ERROR_LOG_FILE')
    # Applying the binary operator '+' (line 269)
    result_add_4558 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 58), '+', result_add_4555, ERROR_LOG_FILE_4557)
    
    str_4559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 98), 'str', 'w')
    # Processing the call keyword arguments (line 269)
    kwargs_4560 = {}
    # Getting the type of 'open' (line 269)
    open_4551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 16), 'open', False)
    # Calling open(args, kwargs) (line 269)
    open_call_result_4561 = invoke(stypy.reporting.localization.Localization(__file__, 269, 16), open_4551, *[result_add_4558, str_4559], **kwargs_4560)
    
    # Assigning a type to the variable 'file_' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'file_', open_call_result_4561)
    
    # Call to write(...): (line 270)
    # Processing the call arguments (line 270)
    str_4564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 20), 'str', '')
    # Processing the call keyword arguments (line 270)
    kwargs_4565 = {}
    # Getting the type of 'file_' (line 270)
    file__4562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 270)
    write_4563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), file__4562, 'write')
    # Calling write(args, kwargs) (line 270)
    write_call_result_4566 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), write_4563, *[str_4564], **kwargs_4565)
    
    
    # Call to close(...): (line 271)
    # Processing the call keyword arguments (line 271)
    kwargs_4569 = {}
    # Getting the type of 'file_' (line 271)
    file__4567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'file_', False)
    # Obtaining the member 'close' of a type (line 271)
    close_4568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 8), file__4567, 'close')
    # Calling close(args, kwargs) (line 271)
    close_call_result_4570 = invoke(stypy.reporting.localization.Localization(__file__, 271, 8), close_4568, *[], **kwargs_4569)
    
    
    # Assigning a Call to a Name (line 273):
    
    # Assigning a Call to a Name (line 273):
    
    # Call to open(...): (line 273)
    # Processing the call arguments (line 273)
    # Getting the type of 'stypy_parameters_copy' (line 273)
    stypy_parameters_copy_4572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 21), 'stypy_parameters_copy', False)
    # Obtaining the member 'LOG_PATH' of a type (line 273)
    LOG_PATH_4573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 21), stypy_parameters_copy_4572, 'LOG_PATH')
    str_4574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 54), 'str', '/')
    # Applying the binary operator '+' (line 273)
    result_add_4575 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 21), '+', LOG_PATH_4573, str_4574)
    
    # Getting the type of 'stypy_parameters_copy' (line 273)
    stypy_parameters_copy_4576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 60), 'stypy_parameters_copy', False)
    # Obtaining the member 'WARNING_LOG_FILE' of a type (line 273)
    WARNING_LOG_FILE_4577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 60), stypy_parameters_copy_4576, 'WARNING_LOG_FILE')
    # Applying the binary operator '+' (line 273)
    result_add_4578 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 58), '+', result_add_4575, WARNING_LOG_FILE_4577)
    
    str_4579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 100), 'str', 'w')
    # Processing the call keyword arguments (line 273)
    kwargs_4580 = {}
    # Getting the type of 'open' (line 273)
    open_4571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'open', False)
    # Calling open(args, kwargs) (line 273)
    open_call_result_4581 = invoke(stypy.reporting.localization.Localization(__file__, 273, 16), open_4571, *[result_add_4578, str_4579], **kwargs_4580)
    
    # Assigning a type to the variable 'file_' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'file_', open_call_result_4581)
    
    # Call to write(...): (line 274)
    # Processing the call arguments (line 274)
    str_4584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 20), 'str', '')
    # Processing the call keyword arguments (line 274)
    kwargs_4585 = {}
    # Getting the type of 'file_' (line 274)
    file__4582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 274)
    write_4583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), file__4582, 'write')
    # Calling write(args, kwargs) (line 274)
    write_call_result_4586 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), write_4583, *[str_4584], **kwargs_4585)
    
    
    # Call to close(...): (line 275)
    # Processing the call keyword arguments (line 275)
    kwargs_4589 = {}
    # Getting the type of 'file_' (line 275)
    file__4587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'file_', False)
    # Obtaining the member 'close' of a type (line 275)
    close_4588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), file__4587, 'close')
    # Calling close(args, kwargs) (line 275)
    close_call_result_4590 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), close_4588, *[], **kwargs_4589)
    
    
    # Assigning a Call to a Name (line 277):
    
    # Assigning a Call to a Name (line 277):
    
    # Call to open(...): (line 277)
    # Processing the call arguments (line 277)
    # Getting the type of 'stypy_parameters_copy' (line 277)
    stypy_parameters_copy_4592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 21), 'stypy_parameters_copy', False)
    # Obtaining the member 'LOG_PATH' of a type (line 277)
    LOG_PATH_4593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 21), stypy_parameters_copy_4592, 'LOG_PATH')
    str_4594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 54), 'str', '/')
    # Applying the binary operator '+' (line 277)
    result_add_4595 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 21), '+', LOG_PATH_4593, str_4594)
    
    # Getting the type of 'stypy_parameters_copy' (line 277)
    stypy_parameters_copy_4596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 60), 'stypy_parameters_copy', False)
    # Obtaining the member 'INFO_LOG_FILE' of a type (line 277)
    INFO_LOG_FILE_4597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 60), stypy_parameters_copy_4596, 'INFO_LOG_FILE')
    # Applying the binary operator '+' (line 277)
    result_add_4598 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 58), '+', result_add_4595, INFO_LOG_FILE_4597)
    
    str_4599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 97), 'str', 'w')
    # Processing the call keyword arguments (line 277)
    kwargs_4600 = {}
    # Getting the type of 'open' (line 277)
    open_4591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 16), 'open', False)
    # Calling open(args, kwargs) (line 277)
    open_call_result_4601 = invoke(stypy.reporting.localization.Localization(__file__, 277, 16), open_4591, *[result_add_4598, str_4599], **kwargs_4600)
    
    # Assigning a type to the variable 'file_' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'file_', open_call_result_4601)
    
    # Call to write(...): (line 278)
    # Processing the call arguments (line 278)
    str_4604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 20), 'str', '')
    # Processing the call keyword arguments (line 278)
    kwargs_4605 = {}
    # Getting the type of 'file_' (line 278)
    file__4602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 278)
    write_4603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), file__4602, 'write')
    # Calling write(args, kwargs) (line 278)
    write_call_result_4606 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), write_4603, *[str_4604], **kwargs_4605)
    
    
    # Call to close(...): (line 279)
    # Processing the call keyword arguments (line 279)
    kwargs_4609 = {}
    # Getting the type of 'file_' (line 279)
    file__4607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'file_', False)
    # Obtaining the member 'close' of a type (line 279)
    close_4608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 8), file__4607, 'close')
    # Calling close(args, kwargs) (line 279)
    close_call_result_4610 = invoke(stypy.reporting.localization.Localization(__file__, 279, 8), close_4608, *[], **kwargs_4609)
    
    # SSA branch for the except part of a try statement (line 268)
    # SSA branch for the except '<any exception>' branch of a try statement (line 268)
    module_type_store.open_ssa_branch('except')
    # Assigning a type to the variable 'stypy_return_type' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'stypy_return_type', types.NoneType)
    # SSA join for try-except statement (line 268)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'reset_logs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'reset_logs' in the type store
    # Getting the type of 'stypy_return_type' (line 263)
    stypy_return_type_4611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4611)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'reset_logs'
    return stypy_return_type_4611

# Assigning a type to the variable 'reset_logs' (line 263)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'reset_logs', reset_logs)

@norecursion
def reset_colors(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'reset_colors'
    module_type_store = module_type_store.open_function_context('reset_colors', 284, 0, False)
    
    # Passed parameters checking function
    reset_colors.stypy_localization = localization
    reset_colors.stypy_type_of_self = None
    reset_colors.stypy_type_store = module_type_store
    reset_colors.stypy_function_name = 'reset_colors'
    reset_colors.stypy_param_names_list = []
    reset_colors.stypy_varargs_param_name = None
    reset_colors.stypy_kwargs_param_name = None
    reset_colors.stypy_call_defaults = defaults
    reset_colors.stypy_call_varargs = varargs
    reset_colors.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'reset_colors', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'reset_colors', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'reset_colors(...)' code ##################

    str_4612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, (-1)), 'str', '\n    Reset Windows colors to leave the console with the default ones\n    :return:\n    ')
    # Getting the type of 'ColorType' (line 289)
    ColorType_4613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 7), 'ColorType')
    # Obtaining the member 'ANSIColors' of a type (line 289)
    ANSIColors_4614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 7), ColorType_4613, 'ANSIColors')
    # Testing if the type of an if condition is none (line 289)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 289, 4), ANSIColors_4614):
        
        # Assigning a Call to a Tuple (line 292):
        
        # Assigning a Call to a Name:
        
        # Call to setup_handles(...): (line 292)
        # Processing the call keyword arguments (line 292)
        kwargs_4617 = {}
        # Getting the type of 'setup_handles' (line 292)
        setup_handles_4616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 24), 'setup_handles', False)
        # Calling setup_handles(args, kwargs) (line 292)
        setup_handles_call_result_4618 = invoke(stypy.reporting.localization.Localization(__file__, 292, 24), setup_handles_4616, *[], **kwargs_4617)
        
        # Assigning a type to the variable 'call_assignment_3938' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_3938', setup_handles_call_result_4618)
        
        # Assigning a Call to a Name (line 292):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3938' (line 292)
        call_assignment_3938_4619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_3938', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4620 = stypy_get_value_from_tuple(call_assignment_3938_4619, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_3939' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_3939', stypy_get_value_from_tuple_call_result_4620)
        
        # Assigning a Name to a Name (line 292):
        # Getting the type of 'call_assignment_3939' (line 292)
        call_assignment_3939_4621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_3939')
        # Assigning a type to the variable 'handle' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'handle', call_assignment_3939_4621)
        
        # Assigning a Call to a Name (line 292):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3938' (line 292)
        call_assignment_3938_4622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_3938', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4623 = stypy_get_value_from_tuple(call_assignment_3938_4622, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_3940' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_3940', stypy_get_value_from_tuple_call_result_4623)
        
        # Assigning a Name to a Name (line 292):
        # Getting the type of 'call_assignment_3940' (line 292)
        call_assignment_3940_4624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_3940')
        # Assigning a type to the variable 'reset' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 16), 'reset', call_assignment_3940_4624)
        
        # Call to SetConsoleTextAttribute(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'handle' (line 293)
        handle_4629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 55), 'handle', False)
        # Getting the type of 'reset' (line 293)
        reset_4630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 63), 'reset', False)
        # Processing the call keyword arguments (line 293)
        kwargs_4631 = {}
        # Getting the type of 'ctypes' (line 293)
        ctypes_4625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 293)
        windll_4626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), ctypes_4625, 'windll')
        # Obtaining the member 'kernel32' of a type (line 293)
        kernel32_4627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), windll_4626, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 293)
        SetConsoleTextAttribute_4628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), kernel32_4627, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 293)
        SetConsoleTextAttribute_call_result_4632 = invoke(stypy.reporting.localization.Localization(__file__, 293, 8), SetConsoleTextAttribute_4628, *[handle_4629, reset_4630], **kwargs_4631)
        
    else:
        
        # Testing the type of an if condition (line 289)
        if_condition_4615 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 289, 4), ANSIColors_4614)
        # Assigning a type to the variable 'if_condition_4615' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'if_condition_4615', if_condition_4615)
        # SSA begins for if statement (line 289)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 289)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Tuple (line 292):
        
        # Assigning a Call to a Name:
        
        # Call to setup_handles(...): (line 292)
        # Processing the call keyword arguments (line 292)
        kwargs_4617 = {}
        # Getting the type of 'setup_handles' (line 292)
        setup_handles_4616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 24), 'setup_handles', False)
        # Calling setup_handles(args, kwargs) (line 292)
        setup_handles_call_result_4618 = invoke(stypy.reporting.localization.Localization(__file__, 292, 24), setup_handles_4616, *[], **kwargs_4617)
        
        # Assigning a type to the variable 'call_assignment_3938' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_3938', setup_handles_call_result_4618)
        
        # Assigning a Call to a Name (line 292):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3938' (line 292)
        call_assignment_3938_4619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_3938', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4620 = stypy_get_value_from_tuple(call_assignment_3938_4619, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_3939' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_3939', stypy_get_value_from_tuple_call_result_4620)
        
        # Assigning a Name to a Name (line 292):
        # Getting the type of 'call_assignment_3939' (line 292)
        call_assignment_3939_4621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_3939')
        # Assigning a type to the variable 'handle' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'handle', call_assignment_3939_4621)
        
        # Assigning a Call to a Name (line 292):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_3938' (line 292)
        call_assignment_3938_4622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_3938', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4623 = stypy_get_value_from_tuple(call_assignment_3938_4622, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_3940' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_3940', stypy_get_value_from_tuple_call_result_4623)
        
        # Assigning a Name to a Name (line 292):
        # Getting the type of 'call_assignment_3940' (line 292)
        call_assignment_3940_4624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_3940')
        # Assigning a type to the variable 'reset' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 16), 'reset', call_assignment_3940_4624)
        
        # Call to SetConsoleTextAttribute(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'handle' (line 293)
        handle_4629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 55), 'handle', False)
        # Getting the type of 'reset' (line 293)
        reset_4630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 63), 'reset', False)
        # Processing the call keyword arguments (line 293)
        kwargs_4631 = {}
        # Getting the type of 'ctypes' (line 293)
        ctypes_4625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 293)
        windll_4626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), ctypes_4625, 'windll')
        # Obtaining the member 'kernel32' of a type (line 293)
        kernel32_4627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), windll_4626, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 293)
        SetConsoleTextAttribute_4628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), kernel32_4627, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 293)
        SetConsoleTextAttribute_call_result_4632 = invoke(stypy.reporting.localization.Localization(__file__, 293, 8), SetConsoleTextAttribute_4628, *[handle_4629, reset_4630], **kwargs_4631)
        
        # SSA join for if statement (line 289)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'reset_colors(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'reset_colors' in the type store
    # Getting the type of 'stypy_return_type' (line 284)
    stypy_return_type_4633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4633)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'reset_colors'
    return stypy_return_type_4633

# Assigning a type to the variable 'reset_colors' (line 284)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 0), 'reset_colors', reset_colors)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
