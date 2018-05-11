
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import datetime
2: import inspect
3: 
4: from ...stypy_copy import stypy_parameters_copy
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

# 'from testing.test_programs.stypy_code_copy.stypy_copy import stypy_parameters_copy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/log_copy/')
import_4227 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy')

if (type(import_4227) is not StypyTypeError):

    if (import_4227 != 'pyd_module'):
        __import__(import_4227)
        sys_modules_4228 = sys.modules[import_4227]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', sys_modules_4228.module_type_store, module_type_store, ['stypy_parameters_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_4228, sys_modules_4228.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy import stypy_parameters_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', None, module_type_store, ['stypy_parameters_copy'], [stypy_parameters_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', import_4227)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/log_copy/')


# Assigning a Name to a Name (line 6):

# Assigning a Name to a Name (line 6):
# Getting the type of 'True' (line 6)
True_4229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 20), 'True')
# Assigning a type to the variable 'output_to_console' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'output_to_console', True_4229)
str_4230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, (-1)), 'str', '\n Multiplatform terminal color messages to improve visual quality of the output\n Also handles message logging for stypy.\n This code has been adapted from tcaswell snippet, found in:\n http://stackoverflow.com/questions/2654113/python-how-to-get-the-callers-method-name-in-the-called-method\n')

@norecursion
def get_caller_data(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_4231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'int')
    defaults = [int_4231]
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

    str_4232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, (-1)), 'str', 'Get a name of a caller in the format module.class.method\n\n       `skip` specifies how many levels of stack to skip while getting caller\n       name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.\n\n       An empty string is returned if skipped levels exceed stack height\n    ')
    
    # Assigning a Call to a Name (line 24):
    
    # Assigning a Call to a Name (line 24):
    
    # Call to stack(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_4235 = {}
    # Getting the type of 'inspect' (line 24)
    inspect_4233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'inspect', False)
    # Obtaining the member 'stack' of a type (line 24)
    stack_4234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 12), inspect_4233, 'stack')
    # Calling stack(args, kwargs) (line 24)
    stack_call_result_4236 = invoke(stypy.reporting.localization.Localization(__file__, 24, 12), stack_4234, *[], **kwargs_4235)
    
    # Assigning a type to the variable 'stack' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stack', stack_call_result_4236)
    
    # Assigning a BinOp to a Name (line 25):
    
    # Assigning a BinOp to a Name (line 25):
    int_4237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 12), 'int')
    # Getting the type of 'skip' (line 25)
    skip_4238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'skip')
    # Applying the binary operator '+' (line 25)
    result_add_4239 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 12), '+', int_4237, skip_4238)
    
    # Assigning a type to the variable 'start' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'start', result_add_4239)
    
    
    # Call to len(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'stack' (line 26)
    stack_4241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'stack', False)
    # Processing the call keyword arguments (line 26)
    kwargs_4242 = {}
    # Getting the type of 'len' (line 26)
    len_4240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 7), 'len', False)
    # Calling len(args, kwargs) (line 26)
    len_call_result_4243 = invoke(stypy.reporting.localization.Localization(__file__, 26, 7), len_4240, *[stack_4241], **kwargs_4242)
    
    # Getting the type of 'start' (line 26)
    start_4244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 20), 'start')
    int_4245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 28), 'int')
    # Applying the binary operator '+' (line 26)
    result_add_4246 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 20), '+', start_4244, int_4245)
    
    # Applying the binary operator '<' (line 26)
    result_lt_4247 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 7), '<', len_call_result_4243, result_add_4246)
    
    # Testing if the type of an if condition is none (line 26)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 26, 4), result_lt_4247):
        pass
    else:
        
        # Testing the type of an if condition (line 26)
        if_condition_4248 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 4), result_lt_4247)
        # Assigning a type to the variable 'if_condition_4248' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'if_condition_4248', if_condition_4248)
        # SSA begins for if statement (line 26)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_4249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'str', '')
        # Assigning a type to the variable 'stypy_return_type' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type', str_4249)
        # SSA join for if statement (line 26)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Subscript to a Name (line 28):
    
    # Assigning a Subscript to a Name (line 28):
    
    # Obtaining the type of the subscript
    int_4250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 31), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'start' (line 28)
    start_4251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'start')
    # Getting the type of 'stack' (line 28)
    stack_4252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), 'stack')
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___4253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 18), stack_4252, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_4254 = invoke(stypy.reporting.localization.Localization(__file__, 28, 18), getitem___4253, start_4251)
    
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___4255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 18), subscript_call_result_4254, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_4256 = invoke(stypy.reporting.localization.Localization(__file__, 28, 18), getitem___4255, int_4250)
    
    # Assigning a type to the variable 'parentframe' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'parentframe', subscript_call_result_4256)
    
    # Assigning a List to a Name (line 30):
    
    # Assigning a List to a Name (line 30):
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_4257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    
    # Assigning a type to the variable 'name' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'name', list_4257)
    
    # Assigning a Call to a Name (line 31):
    
    # Assigning a Call to a Name (line 31):
    
    # Call to getmodule(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'parentframe' (line 31)
    parentframe_4260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 31), 'parentframe', False)
    # Processing the call keyword arguments (line 31)
    kwargs_4261 = {}
    # Getting the type of 'inspect' (line 31)
    inspect_4258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 13), 'inspect', False)
    # Obtaining the member 'getmodule' of a type (line 31)
    getmodule_4259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 13), inspect_4258, 'getmodule')
    # Calling getmodule(args, kwargs) (line 31)
    getmodule_call_result_4262 = invoke(stypy.reporting.localization.Localization(__file__, 31, 13), getmodule_4259, *[parentframe_4260], **kwargs_4261)
    
    # Assigning a type to the variable 'module' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'module', getmodule_call_result_4262)
    # Getting the type of 'module' (line 33)
    module_4263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 7), 'module')
    # Testing if the type of an if condition is none (line 33)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 33, 4), module_4263):
        pass
    else:
        
        # Testing the type of an if condition (line 33)
        if_condition_4264 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 4), module_4263)
        # Assigning a type to the variable 'if_condition_4264' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'if_condition_4264', if_condition_4264)
        # SSA begins for if statement (line 33)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 35)
        # Processing the call arguments (line 35)
        
        # Obtaining the type of the subscript
        int_4267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 48), 'int')
        
        # Call to split(...): (line 35)
        # Processing the call arguments (line 35)
        str_4271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 42), 'str', '\\')
        # Processing the call keyword arguments (line 35)
        kwargs_4272 = {}
        # Getting the type of 'module' (line 35)
        module_4268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'module', False)
        # Obtaining the member '__file__' of a type (line 35)
        file___4269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 20), module_4268, '__file__')
        # Obtaining the member 'split' of a type (line 35)
        split_4270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 20), file___4269, 'split')
        # Calling split(args, kwargs) (line 35)
        split_call_result_4273 = invoke(stypy.reporting.localization.Localization(__file__, 35, 20), split_4270, *[str_4271], **kwargs_4272)
        
        # Obtaining the member '__getitem__' of a type (line 35)
        getitem___4274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 20), split_call_result_4273, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 35)
        subscript_call_result_4275 = invoke(stypy.reporting.localization.Localization(__file__, 35, 20), getitem___4274, int_4267)
        
        # Processing the call keyword arguments (line 35)
        kwargs_4276 = {}
        # Getting the type of 'name' (line 35)
        name_4265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'name', False)
        # Obtaining the member 'append' of a type (line 35)
        append_4266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), name_4265, 'append')
        # Calling append(args, kwargs) (line 35)
        append_call_result_4277 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), append_4266, *[subscript_call_result_4275], **kwargs_4276)
        
        # SSA join for if statement (line 33)
        module_type_store = module_type_store.join_ssa_context()
        

    
    str_4278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 7), 'str', 'self')
    # Getting the type of 'parentframe' (line 38)
    parentframe_4279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 17), 'parentframe')
    # Obtaining the member 'f_locals' of a type (line 38)
    f_locals_4280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 17), parentframe_4279, 'f_locals')
    # Applying the binary operator 'in' (line 38)
    result_contains_4281 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 7), 'in', str_4278, f_locals_4280)
    
    # Testing if the type of an if condition is none (line 38)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 38, 4), result_contains_4281):
        pass
    else:
        
        # Testing the type of an if condition (line 38)
        if_condition_4282 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 4), result_contains_4281)
        # Assigning a type to the variable 'if_condition_4282' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'if_condition_4282', if_condition_4282)
        # SSA begins for if statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Obtaining the type of the subscript
        str_4285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 41), 'str', 'self')
        # Getting the type of 'parentframe' (line 39)
        parentframe_4286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'parentframe', False)
        # Obtaining the member 'f_locals' of a type (line 39)
        f_locals_4287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 20), parentframe_4286, 'f_locals')
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___4288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 20), f_locals_4287, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_4289 = invoke(stypy.reporting.localization.Localization(__file__, 39, 20), getitem___4288, str_4285)
        
        # Obtaining the member '__class__' of a type (line 39)
        class___4290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 20), subscript_call_result_4289, '__class__')
        # Obtaining the member '__name__' of a type (line 39)
        name___4291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 20), class___4290, '__name__')
        # Processing the call keyword arguments (line 39)
        kwargs_4292 = {}
        # Getting the type of 'name' (line 39)
        name_4283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'name', False)
        # Obtaining the member 'append' of a type (line 39)
        append_4284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), name_4283, 'append')
        # Calling append(args, kwargs) (line 39)
        append_call_result_4293 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), append_4284, *[name___4291], **kwargs_4292)
        
        # SSA join for if statement (line 38)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Attribute to a Name (line 41):
    
    # Assigning a Attribute to a Name (line 41):
    # Getting the type of 'parentframe' (line 41)
    parentframe_4294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'parentframe')
    # Obtaining the member 'f_code' of a type (line 41)
    f_code_4295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 15), parentframe_4294, 'f_code')
    # Obtaining the member 'co_name' of a type (line 41)
    co_name_4296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 15), f_code_4295, 'co_name')
    # Assigning a type to the variable 'codename' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'codename', co_name_4296)
    
    # Getting the type of 'codename' (line 43)
    codename_4297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 7), 'codename')
    str_4298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'str', '<module>')
    # Applying the binary operator '!=' (line 43)
    result_ne_4299 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 7), '!=', codename_4297, str_4298)
    
    # Testing if the type of an if condition is none (line 43)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 43, 4), result_ne_4299):
        pass
    else:
        
        # Testing the type of an if condition (line 43)
        if_condition_4300 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 4), result_ne_4299)
        # Assigning a type to the variable 'if_condition_4300' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'if_condition_4300', if_condition_4300)
        # SSA begins for if statement (line 43)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'codename' (line 44)
        codename_4303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'codename', False)
        # Processing the call keyword arguments (line 44)
        kwargs_4304 = {}
        # Getting the type of 'name' (line 44)
        name_4301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'name', False)
        # Obtaining the member 'append' of a type (line 44)
        append_4302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), name_4301, 'append')
        # Calling append(args, kwargs) (line 44)
        append_call_result_4305 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), append_4302, *[codename_4303], **kwargs_4304)
        
        # SSA join for if statement (line 43)
        module_type_store = module_type_store.join_ssa_context()
        

    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 45, 4), module_type_store, 'parentframe')
    
    # Assigning a Subscript to a Subscript (line 48):
    
    # Assigning a Subscript to a Subscript (line 48):
    
    # Obtaining the type of the subscript
    int_4306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 33), 'int')
    
    # Call to split(...): (line 48)
    # Processing the call arguments (line 48)
    str_4312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 28), 'str', '/')
    # Processing the call keyword arguments (line 48)
    kwargs_4313 = {}
    
    # Obtaining the type of the subscript
    int_4307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 19), 'int')
    # Getting the type of 'name' (line 48)
    name_4308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 14), 'name', False)
    # Obtaining the member '__getitem__' of a type (line 48)
    getitem___4309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 14), name_4308, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 48)
    subscript_call_result_4310 = invoke(stypy.reporting.localization.Localization(__file__, 48, 14), getitem___4309, int_4307)
    
    # Obtaining the member 'split' of a type (line 48)
    split_4311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 14), subscript_call_result_4310, 'split')
    # Calling split(args, kwargs) (line 48)
    split_call_result_4314 = invoke(stypy.reporting.localization.Localization(__file__, 48, 14), split_4311, *[str_4312], **kwargs_4313)
    
    # Obtaining the member '__getitem__' of a type (line 48)
    getitem___4315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 14), split_call_result_4314, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 48)
    subscript_call_result_4316 = invoke(stypy.reporting.localization.Localization(__file__, 48, 14), getitem___4315, int_4306)
    
    # Getting the type of 'name' (line 48)
    name_4317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'name')
    int_4318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 9), 'int')
    # Storing an element on a container (line 48)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 4), name_4317, (int_4318, subscript_call_result_4316))
    
    # Call to str(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'name' (line 49)
    name_4320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'name', False)
    # Processing the call keyword arguments (line 49)
    kwargs_4321 = {}
    # Getting the type of 'str' (line 49)
    str_4319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'str', False)
    # Calling str(args, kwargs) (line 49)
    str_call_result_4322 = invoke(stypy.reporting.localization.Localization(__file__, 49, 11), str_4319, *[name_4320], **kwargs_4321)
    
    # Assigning a type to the variable 'stypy_return_type' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type', str_call_result_4322)
    
    # ################# End of 'get_caller_data(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_caller_data' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_4323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4323)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_caller_data'
    return stypy_return_type_4323

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
False_4324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'False')
# Getting the type of 'ColorType'
ColorType_4325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ColorType')
# Setting the type of the member 'ANSIColors' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ColorType_4325, 'ANSIColors', False_4324)


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

    str_4326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, (-1)), 'str', '\n        Determines if it is possible to have colored output\n        :return:\n        ')
    
    # Assigning a Num to a Name (line 66):
    
    # Assigning a Num to a Name (line 66):
    int_4327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 28), 'int')
    # Assigning a type to the variable 'STD_OUTPUT_HANDLE' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'STD_OUTPUT_HANDLE', int_4327)

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
        int_4330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 47), 'int')
        # Processing the call keyword arguments (line 72)
        kwargs_4331 = {}
        # Getting the type of 'ctypes' (line 72)
        ctypes_4328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'ctypes', False)
        # Obtaining the member 'create_string_buffer' of a type (line 72)
        create_string_buffer_4329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 19), ctypes_4328, 'create_string_buffer')
        # Calling create_string_buffer(args, kwargs) (line 72)
        create_string_buffer_call_result_4332 = invoke(stypy.reporting.localization.Localization(__file__, 72, 19), create_string_buffer_4329, *[int_4330], **kwargs_4331)
        
        # Assigning a type to the variable 'csbi' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'csbi', create_string_buffer_call_result_4332)
        
        # Assigning a Call to a Name (line 73):
        
        # Assigning a Call to a Name (line 73):
        
        # Call to GetConsoleScreenBufferInfo(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'handle' (line 73)
        handle_4337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 68), 'handle', False)
        # Getting the type of 'csbi' (line 73)
        csbi_4338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 76), 'csbi', False)
        # Processing the call keyword arguments (line 73)
        kwargs_4339 = {}
        # Getting the type of 'ctypes' (line 73)
        ctypes_4333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 18), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 73)
        windll_4334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 18), ctypes_4333, 'windll')
        # Obtaining the member 'kernel32' of a type (line 73)
        kernel32_4335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 18), windll_4334, 'kernel32')
        # Obtaining the member 'GetConsoleScreenBufferInfo' of a type (line 73)
        GetConsoleScreenBufferInfo_4336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 18), kernel32_4335, 'GetConsoleScreenBufferInfo')
        # Calling GetConsoleScreenBufferInfo(args, kwargs) (line 73)
        GetConsoleScreenBufferInfo_call_result_4340 = invoke(stypy.reporting.localization.Localization(__file__, 73, 18), GetConsoleScreenBufferInfo_4336, *[handle_4337, csbi_4338], **kwargs_4339)
        
        # Assigning a type to the variable 'res' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'res', GetConsoleScreenBufferInfo_call_result_4340)
        
        # Assigning a Call to a Tuple (line 76):
        
        # Assigning a Call to a Name:
        
        # Call to unpack(...): (line 77)
        # Processing the call arguments (line 77)
        str_4343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 67), 'str', 'hhhhHhhhhhh')
        # Getting the type of 'csbi' (line 77)
        csbi_4344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 82), 'csbi', False)
        # Obtaining the member 'raw' of a type (line 77)
        raw_4345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 82), csbi_4344, 'raw')
        # Processing the call keyword arguments (line 77)
        kwargs_4346 = {}
        # Getting the type of 'struct' (line 77)
        struct_4341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 53), 'struct', False)
        # Obtaining the member 'unpack' of a type (line 77)
        unpack_4342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 53), struct_4341, 'unpack')
        # Calling unpack(args, kwargs) (line 77)
        unpack_call_result_4347 = invoke(stypy.reporting.localization.Localization(__file__, 77, 53), unpack_4342, *[str_4343, raw_4345], **kwargs_4346)
        
        # Assigning a type to the variable 'call_assignment_4203' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4203', unpack_call_result_4347)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4203' (line 76)
        call_assignment_4203_4348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4203', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4349 = stypy_get_value_from_tuple(call_assignment_4203_4348, 11, 0)
        
        # Assigning a type to the variable 'call_assignment_4204' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4204', stypy_get_value_from_tuple_call_result_4349)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_4204' (line 76)
        call_assignment_4204_4350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4204')
        # Assigning a type to the variable 'bufx' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 13), 'bufx', call_assignment_4204_4350)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4203' (line 76)
        call_assignment_4203_4351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4203', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4352 = stypy_get_value_from_tuple(call_assignment_4203_4351, 11, 1)
        
        # Assigning a type to the variable 'call_assignment_4205' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4205', stypy_get_value_from_tuple_call_result_4352)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_4205' (line 76)
        call_assignment_4205_4353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4205')
        # Assigning a type to the variable 'bufy' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 19), 'bufy', call_assignment_4205_4353)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4203' (line 76)
        call_assignment_4203_4354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4203', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4355 = stypy_get_value_from_tuple(call_assignment_4203_4354, 11, 2)
        
        # Assigning a type to the variable 'call_assignment_4206' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4206', stypy_get_value_from_tuple_call_result_4355)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_4206' (line 76)
        call_assignment_4206_4356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4206')
        # Assigning a type to the variable 'curx' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 25), 'curx', call_assignment_4206_4356)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4203' (line 76)
        call_assignment_4203_4357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4203', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4358 = stypy_get_value_from_tuple(call_assignment_4203_4357, 11, 3)
        
        # Assigning a type to the variable 'call_assignment_4207' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4207', stypy_get_value_from_tuple_call_result_4358)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_4207' (line 76)
        call_assignment_4207_4359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4207')
        # Assigning a type to the variable 'cury' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 31), 'cury', call_assignment_4207_4359)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4203' (line 76)
        call_assignment_4203_4360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4203', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4361 = stypy_get_value_from_tuple(call_assignment_4203_4360, 11, 4)
        
        # Assigning a type to the variable 'call_assignment_4208' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4208', stypy_get_value_from_tuple_call_result_4361)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_4208' (line 76)
        call_assignment_4208_4362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4208')
        # Assigning a type to the variable 'wattr' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 37), 'wattr', call_assignment_4208_4362)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4203' (line 76)
        call_assignment_4203_4363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4203', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4364 = stypy_get_value_from_tuple(call_assignment_4203_4363, 11, 5)
        
        # Assigning a type to the variable 'call_assignment_4209' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4209', stypy_get_value_from_tuple_call_result_4364)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_4209' (line 76)
        call_assignment_4209_4365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4209')
        # Assigning a type to the variable 'left' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 13), 'left', call_assignment_4209_4365)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4203' (line 76)
        call_assignment_4203_4366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4203', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4367 = stypy_get_value_from_tuple(call_assignment_4203_4366, 11, 6)
        
        # Assigning a type to the variable 'call_assignment_4210' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4210', stypy_get_value_from_tuple_call_result_4367)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_4210' (line 76)
        call_assignment_4210_4368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4210')
        # Assigning a type to the variable 'top' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'top', call_assignment_4210_4368)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4203' (line 76)
        call_assignment_4203_4369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4203', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4370 = stypy_get_value_from_tuple(call_assignment_4203_4369, 11, 7)
        
        # Assigning a type to the variable 'call_assignment_4211' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4211', stypy_get_value_from_tuple_call_result_4370)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_4211' (line 76)
        call_assignment_4211_4371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4211')
        # Assigning a type to the variable 'right' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 24), 'right', call_assignment_4211_4371)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4203' (line 76)
        call_assignment_4203_4372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4203', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4373 = stypy_get_value_from_tuple(call_assignment_4203_4372, 11, 8)
        
        # Assigning a type to the variable 'call_assignment_4212' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4212', stypy_get_value_from_tuple_call_result_4373)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_4212' (line 76)
        call_assignment_4212_4374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4212')
        # Assigning a type to the variable 'bottom' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 31), 'bottom', call_assignment_4212_4374)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4203' (line 76)
        call_assignment_4203_4375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4203', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4376 = stypy_get_value_from_tuple(call_assignment_4203_4375, 11, 9)
        
        # Assigning a type to the variable 'call_assignment_4213' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4213', stypy_get_value_from_tuple_call_result_4376)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_4213' (line 76)
        call_assignment_4213_4377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4213')
        # Assigning a type to the variable 'maxx' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 39), 'maxx', call_assignment_4213_4377)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4203' (line 76)
        call_assignment_4203_4378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4203', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4379 = stypy_get_value_from_tuple(call_assignment_4203_4378, 11, 10)
        
        # Assigning a type to the variable 'call_assignment_4214' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4214', stypy_get_value_from_tuple_call_result_4379)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'call_assignment_4214' (line 76)
        call_assignment_4214_4380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'call_assignment_4214')
        # Assigning a type to the variable 'maxy' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 45), 'maxy', call_assignment_4214_4380)
        # Getting the type of 'wattr' (line 78)
        wattr_4381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'wattr')
        # Assigning a type to the variable 'stypy_return_type' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'stypy_return_type', wattr_4381)
        
        # ################# End of 'get_csbi_attributes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_csbi_attributes' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_4382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4382)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_csbi_attributes'
        return stypy_return_type_4382

    # Assigning a type to the variable 'get_csbi_attributes' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'get_csbi_attributes', get_csbi_attributes)
    
    # Assigning a Call to a Name (line 80):
    
    # Assigning a Call to a Name (line 80):
    
    # Call to GetStdHandle(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'STD_OUTPUT_HANDLE' (line 80)
    STD_OUTPUT_HANDLE_4387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 53), 'STD_OUTPUT_HANDLE', False)
    # Processing the call keyword arguments (line 80)
    kwargs_4388 = {}
    # Getting the type of 'ctypes' (line 80)
    ctypes_4383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'ctypes', False)
    # Obtaining the member 'windll' of a type (line 80)
    windll_4384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 17), ctypes_4383, 'windll')
    # Obtaining the member 'kernel32' of a type (line 80)
    kernel32_4385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 17), windll_4384, 'kernel32')
    # Obtaining the member 'GetStdHandle' of a type (line 80)
    GetStdHandle_4386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 17), kernel32_4385, 'GetStdHandle')
    # Calling GetStdHandle(args, kwargs) (line 80)
    GetStdHandle_call_result_4389 = invoke(stypy.reporting.localization.Localization(__file__, 80, 17), GetStdHandle_4386, *[STD_OUTPUT_HANDLE_4387], **kwargs_4388)
    
    # Assigning a type to the variable 'handle' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'handle', GetStdHandle_call_result_4389)
    
    # Assigning a Call to a Name (line 81):
    
    # Assigning a Call to a Name (line 81):
    
    # Call to get_csbi_attributes(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'handle' (line 81)
    handle_4391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 36), 'handle', False)
    # Processing the call keyword arguments (line 81)
    kwargs_4392 = {}
    # Getting the type of 'get_csbi_attributes' (line 81)
    get_csbi_attributes_4390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'get_csbi_attributes', False)
    # Calling get_csbi_attributes(args, kwargs) (line 81)
    get_csbi_attributes_call_result_4393 = invoke(stypy.reporting.localization.Localization(__file__, 81, 16), get_csbi_attributes_4390, *[handle_4391], **kwargs_4392)
    
    # Assigning a type to the variable 'reset' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'reset', get_csbi_attributes_call_result_4393)
    
    # Obtaining an instance of the builtin type 'tuple' (line 82)
    tuple_4394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 82)
    # Adding element type (line 82)
    # Getting the type of 'handle' (line 82)
    handle_4395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'handle')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 15), tuple_4394, handle_4395)
    # Adding element type (line 82)
    # Getting the type of 'reset' (line 82)
    reset_4396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 23), 'reset')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 15), tuple_4394, reset_4396)
    
    # Assigning a type to the variable 'stypy_return_type' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stypy_return_type', tuple_4394)
    
    # ################# End of 'setup_handles(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'setup_handles' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_4397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4397)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'setup_handles'
    return stypy_return_type_4397

# Assigning a type to the variable 'setup_handles' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'setup_handles', setup_handles)

# Assigning a Name to a Attribute (line 85):

# Assigning a Name to a Attribute (line 85):
# Getting the type of 'False' (line 85)
False_4398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 27), 'False')
# Getting the type of 'ColorType' (line 85)
ColorType_4399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'ColorType')
# Setting the type of the member 'ANSIColors' of a type (line 85)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 4), ColorType_4399, 'ANSIColors', False_4398)
# SSA branch for the except part of a try statement (line 56)
# SSA branch for the except 'Exception' branch of a try statement (line 56)
# Storing handler type
module_type_store.open_ssa_branch('except')
# Getting the type of 'Exception' (line 87)
Exception_4400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 7), 'Exception')
# Assigning a type to the variable 'e' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'e', Exception_4400)

# Assigning a Name to a Attribute (line 88):

# Assigning a Name to a Attribute (line 88):
# Getting the type of 'True' (line 88)
True_4401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 27), 'True')
# Getting the type of 'ColorType' (line 88)
ColorType_4402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'ColorType')
# Setting the type of the member 'ANSIColors' of a type (line 88)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 4), ColorType_4402, 'ANSIColors', True_4401)
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
str_4403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 16), 'str', '\x1b[94m')
# Getting the type of 'Colors'
Colors_4404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colors')
# Setting the type of the member 'ANSI_BLUE' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colors_4404, 'ANSI_BLUE', str_4403)

# Assigning a Str to a Name (line 93):
str_4405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 17), 'str', '\x1b[92m')
# Getting the type of 'Colors'
Colors_4406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colors')
# Setting the type of the member 'ANSI_GREEN' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colors_4406, 'ANSI_GREEN', str_4405)

# Assigning a Str to a Name (line 94):
str_4407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 19), 'str', '\x1b[93m')
# Getting the type of 'Colors'
Colors_4408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colors')
# Setting the type of the member 'ANSI_WARNING' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colors_4408, 'ANSI_WARNING', str_4407)

# Assigning a Str to a Name (line 95):
str_4409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 16), 'str', '\x1b[91m')
# Getting the type of 'Colors'
Colors_4410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colors')
# Setting the type of the member 'ANSI_FAIL' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colors_4410, 'ANSI_FAIL', str_4409)

# Assigning a Str to a Name (line 96):
str_4411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 16), 'str', '\x1b[0m')
# Getting the type of 'Colors'
Colors_4412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colors')
# Setting the type of the member 'ANSI_ENDC' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colors_4412, 'ANSI_ENDC', str_4411)

# Assigning a Num to a Name (line 98):
int_4413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 15), 'int')
# Getting the type of 'Colors'
Colors_4414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colors')
# Setting the type of the member 'WIN_BLUE' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colors_4414, 'WIN_BLUE', int_4413)

# Assigning a Num to a Name (line 99):
int_4415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 16), 'int')
# Getting the type of 'Colors'
Colors_4416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colors')
# Setting the type of the member 'WIN_WHITE' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colors_4416, 'WIN_WHITE', int_4415)

# Assigning a Num to a Name (line 100):
int_4417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 16), 'int')
# Getting the type of 'Colors'
Colors_4418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colors')
# Setting the type of the member 'WIN_GREEN' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colors_4418, 'WIN_GREEN', int_4417)

# Assigning a Num to a Name (line 101):
int_4419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 18), 'int')
# Getting the type of 'Colors'
Colors_4420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colors')
# Setting the type of the member 'WIN_WARNING' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colors_4420, 'WIN_WARNING', int_4419)

# Assigning a Num to a Name (line 102):
int_4421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 15), 'int')
# Getting the type of 'Colors'
Colors_4422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Colors')
# Setting the type of the member 'WIN_FAIL' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Colors_4422, 'WIN_FAIL', int_4421)

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

    str_4423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, (-1)), 'str', '\n    Obtains current date and time\n    :return:\n    ')
    
    # Obtaining the type of the subscript
    int_4424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 41), 'int')
    slice_4425 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 110, 11), None, int_4424, None)
    
    # Call to str(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Call to now(...): (line 110)
    # Processing the call keyword arguments (line 110)
    kwargs_4430 = {}
    # Getting the type of 'datetime' (line 110)
    datetime_4427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 15), 'datetime', False)
    # Obtaining the member 'datetime' of a type (line 110)
    datetime_4428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 15), datetime_4427, 'datetime')
    # Obtaining the member 'now' of a type (line 110)
    now_4429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 15), datetime_4428, 'now')
    # Calling now(args, kwargs) (line 110)
    now_call_result_4431 = invoke(stypy.reporting.localization.Localization(__file__, 110, 15), now_4429, *[], **kwargs_4430)
    
    # Processing the call keyword arguments (line 110)
    kwargs_4432 = {}
    # Getting the type of 'str' (line 110)
    str_4426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'str', False)
    # Calling str(args, kwargs) (line 110)
    str_call_result_4433 = invoke(stypy.reporting.localization.Localization(__file__, 110, 11), str_4426, *[now_call_result_4431], **kwargs_4432)
    
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___4434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 11), str_call_result_4433, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_4435 = invoke(stypy.reporting.localization.Localization(__file__, 110, 11), getitem___4434, slice_4425)
    
    # Assigning a type to the variable 'stypy_return_type' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'stypy_return_type', subscript_call_result_4435)
    
    # ################# End of 'get_date_time(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_date_time' in the type store
    # Getting the type of 'stypy_return_type' (line 105)
    stypy_return_type_4436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4436)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_date_time'
    return stypy_return_type_4436

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

    str_4437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, (-1)), 'str', '\n    Logs information messages to the corresponding log file\n    :param msg:\n    :return:\n    ')
    
    
    # SSA begins for try-except statement (line 119)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 120):
    
    # Assigning a Call to a Name (line 120):
    
    # Call to open(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'stypy_parameters_copy' (line 120)
    stypy_parameters_copy_4439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 21), 'stypy_parameters_copy', False)
    # Obtaining the member 'LOG_PATH' of a type (line 120)
    LOG_PATH_4440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 21), stypy_parameters_copy_4439, 'LOG_PATH')
    str_4441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 54), 'str', '/')
    # Applying the binary operator '+' (line 120)
    result_add_4442 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 21), '+', LOG_PATH_4440, str_4441)
    
    # Getting the type of 'stypy_parameters_copy' (line 120)
    stypy_parameters_copy_4443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 60), 'stypy_parameters_copy', False)
    # Obtaining the member 'INFO_LOG_FILE' of a type (line 120)
    INFO_LOG_FILE_4444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 60), stypy_parameters_copy_4443, 'INFO_LOG_FILE')
    # Applying the binary operator '+' (line 120)
    result_add_4445 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 58), '+', result_add_4442, INFO_LOG_FILE_4444)
    
    str_4446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 97), 'str', 'a')
    # Processing the call keyword arguments (line 120)
    kwargs_4447 = {}
    # Getting the type of 'open' (line 120)
    open_4438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'open', False)
    # Calling open(args, kwargs) (line 120)
    open_call_result_4448 = invoke(stypy.reporting.localization.Localization(__file__, 120, 16), open_4438, *[result_add_4445, str_4446], **kwargs_4447)
    
    # Assigning a type to the variable 'file_' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'file_', open_call_result_4448)
    # SSA branch for the except part of a try statement (line 119)
    # SSA branch for the except '<any exception>' branch of a try statement (line 119)
    module_type_store.open_ssa_branch('except')
    # Assigning a type to the variable 'stypy_return_type' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'stypy_return_type', types.NoneType)
    # SSA join for try-except statement (line 119)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'msg' (line 124)
    msg_4449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'msg')
    str_4450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 15), 'str', '\n')
    # Applying the binary operator '==' (line 124)
    result_eq_4451 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 8), '==', msg_4449, str_4450)
    
    
    # Getting the type of 'msg' (line 124)
    msg_4452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 25), 'msg')
    str_4453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 32), 'str', '')
    # Applying the binary operator '==' (line 124)
    result_eq_4454 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 25), '==', msg_4452, str_4453)
    
    # Applying the binary operator 'or' (line 124)
    result_or_keyword_4455 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 7), 'or', result_eq_4451, result_eq_4454)
    
    # Testing if the type of an if condition is none (line 124)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 124, 4), result_or_keyword_4455):
        
        # Type idiom detected: calculating its left and rigth part (line 130)
        # Getting the type of 'file_' (line 130)
        file__4473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'file_')
        # Getting the type of 'None' (line 130)
        None_4474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'None')
        
        (may_be_4475, more_types_in_union_4476) = may_not_be_none(file__4473, None_4474)

        if may_be_4475:

            if more_types_in_union_4476:
                # Runtime conditional SSA (line 130)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to write(...): (line 131)
            # Processing the call arguments (line 131)
            str_4479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 24), 'str', '[')
            
            # Call to get_date_time(...): (line 131)
            # Processing the call keyword arguments (line 131)
            kwargs_4481 = {}
            # Getting the type of 'get_date_time' (line 131)
            get_date_time_4480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 30), 'get_date_time', False)
            # Calling get_date_time(args, kwargs) (line 131)
            get_date_time_call_result_4482 = invoke(stypy.reporting.localization.Localization(__file__, 131, 30), get_date_time_4480, *[], **kwargs_4481)
            
            # Applying the binary operator '+' (line 131)
            result_add_4483 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 24), '+', str_4479, get_date_time_call_result_4482)
            
            str_4484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 48), 'str', '] ')
            # Applying the binary operator '+' (line 131)
            result_add_4485 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 46), '+', result_add_4483, str_4484)
            
            # Getting the type of 'msg' (line 131)
            msg_4486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 55), 'msg', False)
            # Applying the binary operator '+' (line 131)
            result_add_4487 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 53), '+', result_add_4485, msg_4486)
            
            str_4488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 61), 'str', '\n')
            # Applying the binary operator '+' (line 131)
            result_add_4489 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 59), '+', result_add_4487, str_4488)
            
            # Processing the call keyword arguments (line 131)
            kwargs_4490 = {}
            # Getting the type of 'file_' (line 131)
            file__4477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'file_', False)
            # Obtaining the member 'write' of a type (line 131)
            write_4478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), file__4477, 'write')
            # Calling write(args, kwargs) (line 131)
            write_call_result_4491 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), write_4478, *[result_add_4489], **kwargs_4490)
            

            if more_types_in_union_4476:
                # SSA join for if statement (line 130)
                module_type_store = module_type_store.join_ssa_context()


        
    else:
        
        # Testing the type of an if condition (line 124)
        if_condition_4456 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 4), result_or_keyword_4455)
        # Assigning a type to the variable 'if_condition_4456' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'if_condition_4456', if_condition_4456)
        # SSA begins for if statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'msg' (line 125)
        msg_4457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), 'msg')
        str_4458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 18), 'str', '')
        # Applying the binary operator '==' (line 125)
        result_eq_4459 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 11), '==', msg_4457, str_4458)
        
        # Testing if the type of an if condition is none (line 125)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 8), result_eq_4459):
            
            # Call to write(...): (line 128)
            # Processing the call arguments (line 128)
            # Getting the type of 'msg' (line 128)
            msg_4470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'msg', False)
            # Processing the call keyword arguments (line 128)
            kwargs_4471 = {}
            # Getting the type of 'file_' (line 128)
            file__4468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'file_', False)
            # Obtaining the member 'write' of a type (line 128)
            write_4469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), file__4468, 'write')
            # Calling write(args, kwargs) (line 128)
            write_call_result_4472 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), write_4469, *[msg_4470], **kwargs_4471)
            
        else:
            
            # Testing the type of an if condition (line 125)
            if_condition_4460 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 8), result_eq_4459)
            # Assigning a type to the variable 'if_condition_4460' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'if_condition_4460', if_condition_4460)
            # SSA begins for if statement (line 125)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 126)
            # Processing the call arguments (line 126)
            # Getting the type of 'msg' (line 126)
            msg_4463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'msg', False)
            str_4464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 30), 'str', '\n')
            # Applying the binary operator '+' (line 126)
            result_add_4465 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 24), '+', msg_4463, str_4464)
            
            # Processing the call keyword arguments (line 126)
            kwargs_4466 = {}
            # Getting the type of 'file_' (line 126)
            file__4461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'file_', False)
            # Obtaining the member 'write' of a type (line 126)
            write_4462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 12), file__4461, 'write')
            # Calling write(args, kwargs) (line 126)
            write_call_result_4467 = invoke(stypy.reporting.localization.Localization(__file__, 126, 12), write_4462, *[result_add_4465], **kwargs_4466)
            
            # SSA branch for the else part of an if statement (line 125)
            module_type_store.open_ssa_branch('else')
            
            # Call to write(...): (line 128)
            # Processing the call arguments (line 128)
            # Getting the type of 'msg' (line 128)
            msg_4470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'msg', False)
            # Processing the call keyword arguments (line 128)
            kwargs_4471 = {}
            # Getting the type of 'file_' (line 128)
            file__4468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'file_', False)
            # Obtaining the member 'write' of a type (line 128)
            write_4469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), file__4468, 'write')
            # Calling write(args, kwargs) (line 128)
            write_call_result_4472 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), write_4469, *[msg_4470], **kwargs_4471)
            
            # SSA join for if statement (line 125)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA branch for the else part of an if statement (line 124)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 130)
        # Getting the type of 'file_' (line 130)
        file__4473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'file_')
        # Getting the type of 'None' (line 130)
        None_4474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'None')
        
        (may_be_4475, more_types_in_union_4476) = may_not_be_none(file__4473, None_4474)

        if may_be_4475:

            if more_types_in_union_4476:
                # Runtime conditional SSA (line 130)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to write(...): (line 131)
            # Processing the call arguments (line 131)
            str_4479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 24), 'str', '[')
            
            # Call to get_date_time(...): (line 131)
            # Processing the call keyword arguments (line 131)
            kwargs_4481 = {}
            # Getting the type of 'get_date_time' (line 131)
            get_date_time_4480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 30), 'get_date_time', False)
            # Calling get_date_time(args, kwargs) (line 131)
            get_date_time_call_result_4482 = invoke(stypy.reporting.localization.Localization(__file__, 131, 30), get_date_time_4480, *[], **kwargs_4481)
            
            # Applying the binary operator '+' (line 131)
            result_add_4483 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 24), '+', str_4479, get_date_time_call_result_4482)
            
            str_4484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 48), 'str', '] ')
            # Applying the binary operator '+' (line 131)
            result_add_4485 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 46), '+', result_add_4483, str_4484)
            
            # Getting the type of 'msg' (line 131)
            msg_4486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 55), 'msg', False)
            # Applying the binary operator '+' (line 131)
            result_add_4487 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 53), '+', result_add_4485, msg_4486)
            
            str_4488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 61), 'str', '\n')
            # Applying the binary operator '+' (line 131)
            result_add_4489 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 59), '+', result_add_4487, str_4488)
            
            # Processing the call keyword arguments (line 131)
            kwargs_4490 = {}
            # Getting the type of 'file_' (line 131)
            file__4477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'file_', False)
            # Obtaining the member 'write' of a type (line 131)
            write_4478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), file__4477, 'write')
            # Calling write(args, kwargs) (line 131)
            write_call_result_4491 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), write_4478, *[result_add_4489], **kwargs_4490)
            

            if more_types_in_union_4476:
                # SSA join for if statement (line 130)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 124)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to close(...): (line 133)
    # Processing the call keyword arguments (line 133)
    kwargs_4494 = {}
    # Getting the type of 'file_' (line 133)
    file__4492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'file_', False)
    # Obtaining the member 'close' of a type (line 133)
    close_4493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 4), file__4492, 'close')
    # Calling close(args, kwargs) (line 133)
    close_call_result_4495 = invoke(stypy.reporting.localization.Localization(__file__, 133, 4), close_4493, *[], **kwargs_4494)
    
    
    # ################# End of 'log(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'log' in the type store
    # Getting the type of 'stypy_return_type' (line 113)
    stypy_return_type_4496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4496)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'log'
    return stypy_return_type_4496

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

    str_4497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, (-1)), 'str', '\n    Handles green log information messages\n    :param msg:\n    :return:\n    ')
    
    # Assigning a BinOp to a Name (line 142):
    
    # Assigning a BinOp to a Name (line 142):
    
    # Call to get_caller_data(...): (line 142)
    # Processing the call keyword arguments (line 142)
    kwargs_4499 = {}
    # Getting the type of 'get_caller_data' (line 142)
    get_caller_data_4498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 10), 'get_caller_data', False)
    # Calling get_caller_data(args, kwargs) (line 142)
    get_caller_data_call_result_4500 = invoke(stypy.reporting.localization.Localization(__file__, 142, 10), get_caller_data_4498, *[], **kwargs_4499)
    
    str_4501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 30), 'str', ': ')
    # Applying the binary operator '+' (line 142)
    result_add_4502 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 10), '+', get_caller_data_call_result_4500, str_4501)
    
    # Getting the type of 'msg' (line 142)
    msg_4503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 37), 'msg')
    # Applying the binary operator '+' (line 142)
    result_add_4504 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 35), '+', result_add_4502, msg_4503)
    
    # Assigning a type to the variable 'txt' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'txt', result_add_4504)
    # Getting the type of 'ColorType' (line 143)
    ColorType_4505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 7), 'ColorType')
    # Obtaining the member 'ANSIColors' of a type (line 143)
    ANSIColors_4506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 7), ColorType_4505, 'ANSIColors')
    # Testing if the type of an if condition is none (line 143)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 143, 4), ANSIColors_4506):
        
        # Assigning a Call to a Tuple (line 147):
        
        # Assigning a Call to a Name:
        
        # Call to setup_handles(...): (line 147)
        # Processing the call keyword arguments (line 147)
        kwargs_4518 = {}
        # Getting the type of 'setup_handles' (line 147)
        setup_handles_4517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 24), 'setup_handles', False)
        # Calling setup_handles(args, kwargs) (line 147)
        setup_handles_call_result_4519 = invoke(stypy.reporting.localization.Localization(__file__, 147, 24), setup_handles_4517, *[], **kwargs_4518)
        
        # Assigning a type to the variable 'call_assignment_4215' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_4215', setup_handles_call_result_4519)
        
        # Assigning a Call to a Name (line 147):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4215' (line 147)
        call_assignment_4215_4520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_4215', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4521 = stypy_get_value_from_tuple(call_assignment_4215_4520, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_4216' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_4216', stypy_get_value_from_tuple_call_result_4521)
        
        # Assigning a Name to a Name (line 147):
        # Getting the type of 'call_assignment_4216' (line 147)
        call_assignment_4216_4522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_4216')
        # Assigning a type to the variable 'handle' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'handle', call_assignment_4216_4522)
        
        # Assigning a Call to a Name (line 147):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4215' (line 147)
        call_assignment_4215_4523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_4215', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4524 = stypy_get_value_from_tuple(call_assignment_4215_4523, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_4217' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_4217', stypy_get_value_from_tuple_call_result_4524)
        
        # Assigning a Name to a Name (line 147):
        # Getting the type of 'call_assignment_4217' (line 147)
        call_assignment_4217_4525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_4217')
        # Assigning a type to the variable 'reset' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'reset', call_assignment_4217_4525)
        
        # Call to SetConsoleTextAttribute(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'handle' (line 148)
        handle_4530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 55), 'handle', False)
        # Getting the type of 'Colors' (line 148)
        Colors_4531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 63), 'Colors', False)
        # Obtaining the member 'WIN_GREEN' of a type (line 148)
        WIN_GREEN_4532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 63), Colors_4531, 'WIN_GREEN')
        # Processing the call keyword arguments (line 148)
        kwargs_4533 = {}
        # Getting the type of 'ctypes' (line 148)
        ctypes_4526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 148)
        windll_4527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), ctypes_4526, 'windll')
        # Obtaining the member 'kernel32' of a type (line 148)
        kernel32_4528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), windll_4527, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 148)
        SetConsoleTextAttribute_4529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), kernel32_4528, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 148)
        SetConsoleTextAttribute_call_result_4534 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), SetConsoleTextAttribute_4529, *[handle_4530, WIN_GREEN_4532], **kwargs_4533)
        
        # Getting the type of 'output_to_console' (line 150)
        output_to_console_4535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'output_to_console')
        # Testing if the type of an if condition is none (line 150)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 150, 8), output_to_console_4535):
            pass
        else:
            
            # Testing the type of an if condition (line 150)
            if_condition_4536 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 8), output_to_console_4535)
            # Assigning a type to the variable 'if_condition_4536' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'if_condition_4536', if_condition_4536)
            # SSA begins for if statement (line 150)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'msg' (line 151)
            msg_4537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 19), 'msg')
            # SSA join for if statement (line 150)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to SetConsoleTextAttribute(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'handle' (line 153)
        handle_4542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 55), 'handle', False)
        # Getting the type of 'reset' (line 153)
        reset_4543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 63), 'reset', False)
        # Processing the call keyword arguments (line 153)
        kwargs_4544 = {}
        # Getting the type of 'ctypes' (line 153)
        ctypes_4538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 153)
        windll_4539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), ctypes_4538, 'windll')
        # Obtaining the member 'kernel32' of a type (line 153)
        kernel32_4540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), windll_4539, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 153)
        SetConsoleTextAttribute_4541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), kernel32_4540, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 153)
        SetConsoleTextAttribute_call_result_4545 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), SetConsoleTextAttribute_4541, *[handle_4542, reset_4543], **kwargs_4544)
        
    else:
        
        # Testing the type of an if condition (line 143)
        if_condition_4507 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 4), ANSIColors_4506)
        # Assigning a type to the variable 'if_condition_4507' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'if_condition_4507', if_condition_4507)
        # SSA begins for if statement (line 143)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'output_to_console' (line 144)
        output_to_console_4508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'output_to_console')
        # Testing if the type of an if condition is none (line 144)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 144, 8), output_to_console_4508):
            pass
        else:
            
            # Testing the type of an if condition (line 144)
            if_condition_4509 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 8), output_to_console_4508)
            # Assigning a type to the variable 'if_condition_4509' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'if_condition_4509', if_condition_4509)
            # SSA begins for if statement (line 144)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'Colors' (line 145)
            Colors_4510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 18), 'Colors')
            # Obtaining the member 'ANSI_GREEN' of a type (line 145)
            ANSI_GREEN_4511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 18), Colors_4510, 'ANSI_GREEN')
            # Getting the type of 'msg' (line 145)
            msg_4512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 38), 'msg')
            # Applying the binary operator '+' (line 145)
            result_add_4513 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 18), '+', ANSI_GREEN_4511, msg_4512)
            
            # Getting the type of 'Colors' (line 145)
            Colors_4514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 44), 'Colors')
            # Obtaining the member 'ANSI_ENDC' of a type (line 145)
            ANSI_ENDC_4515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 44), Colors_4514, 'ANSI_ENDC')
            # Applying the binary operator '+' (line 145)
            result_add_4516 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 42), '+', result_add_4513, ANSI_ENDC_4515)
            
            # SSA join for if statement (line 144)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA branch for the else part of an if statement (line 143)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Tuple (line 147):
        
        # Assigning a Call to a Name:
        
        # Call to setup_handles(...): (line 147)
        # Processing the call keyword arguments (line 147)
        kwargs_4518 = {}
        # Getting the type of 'setup_handles' (line 147)
        setup_handles_4517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 24), 'setup_handles', False)
        # Calling setup_handles(args, kwargs) (line 147)
        setup_handles_call_result_4519 = invoke(stypy.reporting.localization.Localization(__file__, 147, 24), setup_handles_4517, *[], **kwargs_4518)
        
        # Assigning a type to the variable 'call_assignment_4215' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_4215', setup_handles_call_result_4519)
        
        # Assigning a Call to a Name (line 147):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4215' (line 147)
        call_assignment_4215_4520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_4215', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4521 = stypy_get_value_from_tuple(call_assignment_4215_4520, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_4216' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_4216', stypy_get_value_from_tuple_call_result_4521)
        
        # Assigning a Name to a Name (line 147):
        # Getting the type of 'call_assignment_4216' (line 147)
        call_assignment_4216_4522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_4216')
        # Assigning a type to the variable 'handle' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'handle', call_assignment_4216_4522)
        
        # Assigning a Call to a Name (line 147):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4215' (line 147)
        call_assignment_4215_4523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_4215', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4524 = stypy_get_value_from_tuple(call_assignment_4215_4523, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_4217' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_4217', stypy_get_value_from_tuple_call_result_4524)
        
        # Assigning a Name to a Name (line 147):
        # Getting the type of 'call_assignment_4217' (line 147)
        call_assignment_4217_4525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_4217')
        # Assigning a type to the variable 'reset' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'reset', call_assignment_4217_4525)
        
        # Call to SetConsoleTextAttribute(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'handle' (line 148)
        handle_4530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 55), 'handle', False)
        # Getting the type of 'Colors' (line 148)
        Colors_4531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 63), 'Colors', False)
        # Obtaining the member 'WIN_GREEN' of a type (line 148)
        WIN_GREEN_4532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 63), Colors_4531, 'WIN_GREEN')
        # Processing the call keyword arguments (line 148)
        kwargs_4533 = {}
        # Getting the type of 'ctypes' (line 148)
        ctypes_4526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 148)
        windll_4527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), ctypes_4526, 'windll')
        # Obtaining the member 'kernel32' of a type (line 148)
        kernel32_4528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), windll_4527, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 148)
        SetConsoleTextAttribute_4529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), kernel32_4528, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 148)
        SetConsoleTextAttribute_call_result_4534 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), SetConsoleTextAttribute_4529, *[handle_4530, WIN_GREEN_4532], **kwargs_4533)
        
        # Getting the type of 'output_to_console' (line 150)
        output_to_console_4535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'output_to_console')
        # Testing if the type of an if condition is none (line 150)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 150, 8), output_to_console_4535):
            pass
        else:
            
            # Testing the type of an if condition (line 150)
            if_condition_4536 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 8), output_to_console_4535)
            # Assigning a type to the variable 'if_condition_4536' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'if_condition_4536', if_condition_4536)
            # SSA begins for if statement (line 150)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'msg' (line 151)
            msg_4537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 19), 'msg')
            # SSA join for if statement (line 150)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to SetConsoleTextAttribute(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'handle' (line 153)
        handle_4542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 55), 'handle', False)
        # Getting the type of 'reset' (line 153)
        reset_4543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 63), 'reset', False)
        # Processing the call keyword arguments (line 153)
        kwargs_4544 = {}
        # Getting the type of 'ctypes' (line 153)
        ctypes_4538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 153)
        windll_4539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), ctypes_4538, 'windll')
        # Obtaining the member 'kernel32' of a type (line 153)
        kernel32_4540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), windll_4539, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 153)
        SetConsoleTextAttribute_4541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), kernel32_4540, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 153)
        SetConsoleTextAttribute_call_result_4545 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), SetConsoleTextAttribute_4541, *[handle_4542, reset_4543], **kwargs_4544)
        
        # SSA join for if statement (line 143)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to log(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'txt' (line 155)
    txt_4547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'txt', False)
    # Processing the call keyword arguments (line 155)
    kwargs_4548 = {}
    # Getting the type of 'log' (line 155)
    log_4546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'log', False)
    # Calling log(args, kwargs) (line 155)
    log_call_result_4549 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), log_4546, *[txt_4547], **kwargs_4548)
    
    
    # ################# End of 'ok(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ok' in the type store
    # Getting the type of 'stypy_return_type' (line 136)
    stypy_return_type_4550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4550)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ok'
    return stypy_return_type_4550

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

    str_4551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, (-1)), 'str', '\n    Handles white log information messages\n    :param msg:\n    :return:\n    ')
    
    # Assigning a BinOp to a Name (line 164):
    
    # Assigning a BinOp to a Name (line 164):
    
    # Call to get_caller_data(...): (line 164)
    # Processing the call keyword arguments (line 164)
    kwargs_4553 = {}
    # Getting the type of 'get_caller_data' (line 164)
    get_caller_data_4552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 10), 'get_caller_data', False)
    # Calling get_caller_data(args, kwargs) (line 164)
    get_caller_data_call_result_4554 = invoke(stypy.reporting.localization.Localization(__file__, 164, 10), get_caller_data_4552, *[], **kwargs_4553)
    
    str_4555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 30), 'str', ': ')
    # Applying the binary operator '+' (line 164)
    result_add_4556 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 10), '+', get_caller_data_call_result_4554, str_4555)
    
    # Getting the type of 'msg' (line 164)
    msg_4557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 37), 'msg')
    # Applying the binary operator '+' (line 164)
    result_add_4558 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 35), '+', result_add_4556, msg_4557)
    
    # Assigning a type to the variable 'txt' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'txt', result_add_4558)
    # Getting the type of 'ColorType' (line 165)
    ColorType_4559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 7), 'ColorType')
    # Obtaining the member 'ANSIColors' of a type (line 165)
    ANSIColors_4560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 7), ColorType_4559, 'ANSIColors')
    # Testing if the type of an if condition is none (line 165)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 165, 4), ANSIColors_4560):
        
        # Assigning a Call to a Tuple (line 169):
        
        # Assigning a Call to a Name:
        
        # Call to setup_handles(...): (line 169)
        # Processing the call keyword arguments (line 169)
        kwargs_4566 = {}
        # Getting the type of 'setup_handles' (line 169)
        setup_handles_4565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 24), 'setup_handles', False)
        # Calling setup_handles(args, kwargs) (line 169)
        setup_handles_call_result_4567 = invoke(stypy.reporting.localization.Localization(__file__, 169, 24), setup_handles_4565, *[], **kwargs_4566)
        
        # Assigning a type to the variable 'call_assignment_4218' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_4218', setup_handles_call_result_4567)
        
        # Assigning a Call to a Name (line 169):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4218' (line 169)
        call_assignment_4218_4568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_4218', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4569 = stypy_get_value_from_tuple(call_assignment_4218_4568, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_4219' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_4219', stypy_get_value_from_tuple_call_result_4569)
        
        # Assigning a Name to a Name (line 169):
        # Getting the type of 'call_assignment_4219' (line 169)
        call_assignment_4219_4570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_4219')
        # Assigning a type to the variable 'handle' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'handle', call_assignment_4219_4570)
        
        # Assigning a Call to a Name (line 169):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4218' (line 169)
        call_assignment_4218_4571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_4218', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4572 = stypy_get_value_from_tuple(call_assignment_4218_4571, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_4220' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_4220', stypy_get_value_from_tuple_call_result_4572)
        
        # Assigning a Name to a Name (line 169):
        # Getting the type of 'call_assignment_4220' (line 169)
        call_assignment_4220_4573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_4220')
        # Assigning a type to the variable 'reset' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'reset', call_assignment_4220_4573)
        
        # Call to SetConsoleTextAttribute(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'handle' (line 170)
        handle_4578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 55), 'handle', False)
        # Getting the type of 'Colors' (line 170)
        Colors_4579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 63), 'Colors', False)
        # Obtaining the member 'WIN_WHITE' of a type (line 170)
        WIN_WHITE_4580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 63), Colors_4579, 'WIN_WHITE')
        # Processing the call keyword arguments (line 170)
        kwargs_4581 = {}
        # Getting the type of 'ctypes' (line 170)
        ctypes_4574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 170)
        windll_4575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), ctypes_4574, 'windll')
        # Obtaining the member 'kernel32' of a type (line 170)
        kernel32_4576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), windll_4575, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 170)
        SetConsoleTextAttribute_4577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), kernel32_4576, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 170)
        SetConsoleTextAttribute_call_result_4582 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), SetConsoleTextAttribute_4577, *[handle_4578, WIN_WHITE_4580], **kwargs_4581)
        
        # Getting the type of 'output_to_console' (line 172)
        output_to_console_4583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 11), 'output_to_console')
        # Testing if the type of an if condition is none (line 172)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 172, 8), output_to_console_4583):
            pass
        else:
            
            # Testing the type of an if condition (line 172)
            if_condition_4584 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 8), output_to_console_4583)
            # Assigning a type to the variable 'if_condition_4584' (line 172)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'if_condition_4584', if_condition_4584)
            # SSA begins for if statement (line 172)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'txt' (line 173)
            txt_4585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 'txt')
            # SSA join for if statement (line 172)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to SetConsoleTextAttribute(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'handle' (line 175)
        handle_4590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 55), 'handle', False)
        # Getting the type of 'reset' (line 175)
        reset_4591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 63), 'reset', False)
        # Processing the call keyword arguments (line 175)
        kwargs_4592 = {}
        # Getting the type of 'ctypes' (line 175)
        ctypes_4586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 175)
        windll_4587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), ctypes_4586, 'windll')
        # Obtaining the member 'kernel32' of a type (line 175)
        kernel32_4588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), windll_4587, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 175)
        SetConsoleTextAttribute_4589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), kernel32_4588, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 175)
        SetConsoleTextAttribute_call_result_4593 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), SetConsoleTextAttribute_4589, *[handle_4590, reset_4591], **kwargs_4592)
        
    else:
        
        # Testing the type of an if condition (line 165)
        if_condition_4561 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 4), ANSIColors_4560)
        # Assigning a type to the variable 'if_condition_4561' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'if_condition_4561', if_condition_4561)
        # SSA begins for if statement (line 165)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'output_to_console' (line 166)
        output_to_console_4562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 11), 'output_to_console')
        # Testing if the type of an if condition is none (line 166)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 166, 8), output_to_console_4562):
            pass
        else:
            
            # Testing the type of an if condition (line 166)
            if_condition_4563 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 8), output_to_console_4562)
            # Assigning a type to the variable 'if_condition_4563' (line 166)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'if_condition_4563', if_condition_4563)
            # SSA begins for if statement (line 166)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'txt' (line 167)
            txt_4564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'txt')
            # SSA join for if statement (line 166)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA branch for the else part of an if statement (line 165)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Tuple (line 169):
        
        # Assigning a Call to a Name:
        
        # Call to setup_handles(...): (line 169)
        # Processing the call keyword arguments (line 169)
        kwargs_4566 = {}
        # Getting the type of 'setup_handles' (line 169)
        setup_handles_4565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 24), 'setup_handles', False)
        # Calling setup_handles(args, kwargs) (line 169)
        setup_handles_call_result_4567 = invoke(stypy.reporting.localization.Localization(__file__, 169, 24), setup_handles_4565, *[], **kwargs_4566)
        
        # Assigning a type to the variable 'call_assignment_4218' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_4218', setup_handles_call_result_4567)
        
        # Assigning a Call to a Name (line 169):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4218' (line 169)
        call_assignment_4218_4568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_4218', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4569 = stypy_get_value_from_tuple(call_assignment_4218_4568, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_4219' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_4219', stypy_get_value_from_tuple_call_result_4569)
        
        # Assigning a Name to a Name (line 169):
        # Getting the type of 'call_assignment_4219' (line 169)
        call_assignment_4219_4570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_4219')
        # Assigning a type to the variable 'handle' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'handle', call_assignment_4219_4570)
        
        # Assigning a Call to a Name (line 169):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4218' (line 169)
        call_assignment_4218_4571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_4218', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4572 = stypy_get_value_from_tuple(call_assignment_4218_4571, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_4220' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_4220', stypy_get_value_from_tuple_call_result_4572)
        
        # Assigning a Name to a Name (line 169):
        # Getting the type of 'call_assignment_4220' (line 169)
        call_assignment_4220_4573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'call_assignment_4220')
        # Assigning a type to the variable 'reset' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'reset', call_assignment_4220_4573)
        
        # Call to SetConsoleTextAttribute(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'handle' (line 170)
        handle_4578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 55), 'handle', False)
        # Getting the type of 'Colors' (line 170)
        Colors_4579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 63), 'Colors', False)
        # Obtaining the member 'WIN_WHITE' of a type (line 170)
        WIN_WHITE_4580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 63), Colors_4579, 'WIN_WHITE')
        # Processing the call keyword arguments (line 170)
        kwargs_4581 = {}
        # Getting the type of 'ctypes' (line 170)
        ctypes_4574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 170)
        windll_4575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), ctypes_4574, 'windll')
        # Obtaining the member 'kernel32' of a type (line 170)
        kernel32_4576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), windll_4575, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 170)
        SetConsoleTextAttribute_4577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), kernel32_4576, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 170)
        SetConsoleTextAttribute_call_result_4582 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), SetConsoleTextAttribute_4577, *[handle_4578, WIN_WHITE_4580], **kwargs_4581)
        
        # Getting the type of 'output_to_console' (line 172)
        output_to_console_4583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 11), 'output_to_console')
        # Testing if the type of an if condition is none (line 172)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 172, 8), output_to_console_4583):
            pass
        else:
            
            # Testing the type of an if condition (line 172)
            if_condition_4584 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 8), output_to_console_4583)
            # Assigning a type to the variable 'if_condition_4584' (line 172)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'if_condition_4584', if_condition_4584)
            # SSA begins for if statement (line 172)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'txt' (line 173)
            txt_4585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 'txt')
            # SSA join for if statement (line 172)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to SetConsoleTextAttribute(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'handle' (line 175)
        handle_4590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 55), 'handle', False)
        # Getting the type of 'reset' (line 175)
        reset_4591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 63), 'reset', False)
        # Processing the call keyword arguments (line 175)
        kwargs_4592 = {}
        # Getting the type of 'ctypes' (line 175)
        ctypes_4586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 175)
        windll_4587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), ctypes_4586, 'windll')
        # Obtaining the member 'kernel32' of a type (line 175)
        kernel32_4588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), windll_4587, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 175)
        SetConsoleTextAttribute_4589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), kernel32_4588, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 175)
        SetConsoleTextAttribute_call_result_4593 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), SetConsoleTextAttribute_4589, *[handle_4590, reset_4591], **kwargs_4592)
        
        # SSA join for if statement (line 165)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to log(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'txt' (line 177)
    txt_4595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'txt', False)
    # Processing the call keyword arguments (line 177)
    kwargs_4596 = {}
    # Getting the type of 'log' (line 177)
    log_4594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'log', False)
    # Calling log(args, kwargs) (line 177)
    log_call_result_4597 = invoke(stypy.reporting.localization.Localization(__file__, 177, 4), log_4594, *[txt_4595], **kwargs_4596)
    
    
    # ################# End of 'info(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'info' in the type store
    # Getting the type of 'stypy_return_type' (line 158)
    stypy_return_type_4598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4598)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'info'
    return stypy_return_type_4598

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

    str_4599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, (-1)), 'str', '\n    Helper function to output warning or error messages, depending on its parameters.\n    :param msg: Message to print\n    :param call_data: Caller information\n    :param ansi_console_color: ANSI terminals color to use\n    :param win_console_color: Windows terminals color to use\n    :param file_name: File to write to\n    :param msg_type: Type of message to write (WARNING/ERROR)\n    :return:\n    ')
    # Getting the type of 'ColorType' (line 191)
    ColorType_4600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 7), 'ColorType')
    # Obtaining the member 'ANSIColors' of a type (line 191)
    ANSIColors_4601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 7), ColorType_4600, 'ANSIColors')
    # Testing if the type of an if condition is none (line 191)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 191, 4), ANSIColors_4601):
        
        # Assigning a Call to a Tuple (line 196):
        
        # Assigning a Call to a Name:
        
        # Call to setup_handles(...): (line 196)
        # Processing the call keyword arguments (line 196)
        kwargs_4627 = {}
        # Getting the type of 'setup_handles' (line 196)
        setup_handles_4626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 24), 'setup_handles', False)
        # Calling setup_handles(args, kwargs) (line 196)
        setup_handles_call_result_4628 = invoke(stypy.reporting.localization.Localization(__file__, 196, 24), setup_handles_4626, *[], **kwargs_4627)
        
        # Assigning a type to the variable 'call_assignment_4221' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_4221', setup_handles_call_result_4628)
        
        # Assigning a Call to a Name (line 196):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4221' (line 196)
        call_assignment_4221_4629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_4221', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4630 = stypy_get_value_from_tuple(call_assignment_4221_4629, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_4222' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_4222', stypy_get_value_from_tuple_call_result_4630)
        
        # Assigning a Name to a Name (line 196):
        # Getting the type of 'call_assignment_4222' (line 196)
        call_assignment_4222_4631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_4222')
        # Assigning a type to the variable 'handle' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'handle', call_assignment_4222_4631)
        
        # Assigning a Call to a Name (line 196):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4221' (line 196)
        call_assignment_4221_4632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_4221', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4633 = stypy_get_value_from_tuple(call_assignment_4221_4632, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_4223' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_4223', stypy_get_value_from_tuple_call_result_4633)
        
        # Assigning a Name to a Name (line 196):
        # Getting the type of 'call_assignment_4223' (line 196)
        call_assignment_4223_4634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_4223')
        # Assigning a type to the variable 'reset' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'reset', call_assignment_4223_4634)
        
        # Call to SetConsoleTextAttribute(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'handle' (line 197)
        handle_4639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 55), 'handle', False)
        # Getting the type of 'win_console_color' (line 197)
        win_console_color_4640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 63), 'win_console_color', False)
        # Processing the call keyword arguments (line 197)
        kwargs_4641 = {}
        # Getting the type of 'ctypes' (line 197)
        ctypes_4635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 197)
        windll_4636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), ctypes_4635, 'windll')
        # Obtaining the member 'kernel32' of a type (line 197)
        kernel32_4637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), windll_4636, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 197)
        SetConsoleTextAttribute_4638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), kernel32_4637, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 197)
        SetConsoleTextAttribute_call_result_4642 = invoke(stypy.reporting.localization.Localization(__file__, 197, 8), SetConsoleTextAttribute_4638, *[handle_4639, win_console_color_4640], **kwargs_4641)
        
        # Getting the type of 'output_to_console' (line 199)
        output_to_console_4643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 11), 'output_to_console')
        # Testing if the type of an if condition is none (line 199)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 199, 8), output_to_console_4643):
            pass
        else:
            
            # Testing the type of an if condition (line 199)
            if_condition_4644 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 8), output_to_console_4643)
            # Assigning a type to the variable 'if_condition_4644' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'if_condition_4644', if_condition_4644)
            # SSA begins for if statement (line 199)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'msg_type' (line 200)
            msg_type_4645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 18), 'msg_type')
            str_4646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 29), 'str', ': ')
            # Applying the binary operator '+' (line 200)
            result_add_4647 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 18), '+', msg_type_4645, str_4646)
            
            # Getting the type of 'msg' (line 200)
            msg_4648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 36), 'msg')
            # Applying the binary operator '+' (line 200)
            result_add_4649 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 34), '+', result_add_4647, msg_4648)
            
            # SSA join for if statement (line 199)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to SetConsoleTextAttribute(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'handle' (line 202)
        handle_4654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 55), 'handle', False)
        # Getting the type of 'reset' (line 202)
        reset_4655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 63), 'reset', False)
        # Processing the call keyword arguments (line 202)
        kwargs_4656 = {}
        # Getting the type of 'ctypes' (line 202)
        ctypes_4650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 202)
        windll_4651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), ctypes_4650, 'windll')
        # Obtaining the member 'kernel32' of a type (line 202)
        kernel32_4652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), windll_4651, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 202)
        SetConsoleTextAttribute_4653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), kernel32_4652, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 202)
        SetConsoleTextAttribute_call_result_4657 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), SetConsoleTextAttribute_4653, *[handle_4654, reset_4655], **kwargs_4656)
        
    else:
        
        # Testing the type of an if condition (line 191)
        if_condition_4602 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 4), ANSIColors_4601)
        # Assigning a type to the variable 'if_condition_4602' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'if_condition_4602', if_condition_4602)
        # SSA begins for if statement (line 191)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'output_to_console' (line 192)
        output_to_console_4603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 11), 'output_to_console')
        # Testing if the type of an if condition is none (line 192)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 192, 8), output_to_console_4603):
            pass
        else:
            
            # Testing the type of an if condition (line 192)
            if_condition_4604 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 8), output_to_console_4603)
            # Assigning a type to the variable 'if_condition_4604' (line 192)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'if_condition_4604', if_condition_4604)
            # SSA begins for if statement (line 192)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 193):
            
            # Assigning a BinOp to a Name (line 193):
            
            # Call to str(...): (line 193)
            # Processing the call arguments (line 193)
            # Getting the type of 'call_data' (line 193)
            call_data_4606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 22), 'call_data', False)
            # Processing the call keyword arguments (line 193)
            kwargs_4607 = {}
            # Getting the type of 'str' (line 193)
            str_4605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 18), 'str', False)
            # Calling str(args, kwargs) (line 193)
            str_call_result_4608 = invoke(stypy.reporting.localization.Localization(__file__, 193, 18), str_4605, *[call_data_4606], **kwargs_4607)
            
            str_4609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 35), 'str', '. ')
            # Applying the binary operator '+' (line 193)
            result_add_4610 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 18), '+', str_call_result_4608, str_4609)
            
            # Getting the type of 'msg_type' (line 193)
            msg_type_4611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 42), 'msg_type')
            # Applying the binary operator '+' (line 193)
            result_add_4612 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 40), '+', result_add_4610, msg_type_4611)
            
            str_4613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 53), 'str', ': ')
            # Applying the binary operator '+' (line 193)
            result_add_4614 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 51), '+', result_add_4612, str_4613)
            
            
            # Call to str(...): (line 193)
            # Processing the call arguments (line 193)
            # Getting the type of 'msg' (line 193)
            msg_4616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 64), 'msg', False)
            # Processing the call keyword arguments (line 193)
            kwargs_4617 = {}
            # Getting the type of 'str' (line 193)
            str_4615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 60), 'str', False)
            # Calling str(args, kwargs) (line 193)
            str_call_result_4618 = invoke(stypy.reporting.localization.Localization(__file__, 193, 60), str_4615, *[msg_4616], **kwargs_4617)
            
            # Applying the binary operator '+' (line 193)
            result_add_4619 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 58), '+', result_add_4614, str_call_result_4618)
            
            # Assigning a type to the variable 'txt' (line 193)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'txt', result_add_4619)
            # Getting the type of 'ansi_console_color' (line 194)
            ansi_console_color_4620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 18), 'ansi_console_color')
            # Getting the type of 'txt' (line 194)
            txt_4621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 39), 'txt')
            # Applying the binary operator '+' (line 194)
            result_add_4622 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 18), '+', ansi_console_color_4620, txt_4621)
            
            # Getting the type of 'Colors' (line 194)
            Colors_4623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 45), 'Colors')
            # Obtaining the member 'ANSI_ENDC' of a type (line 194)
            ANSI_ENDC_4624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 45), Colors_4623, 'ANSI_ENDC')
            # Applying the binary operator '+' (line 194)
            result_add_4625 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 43), '+', result_add_4622, ANSI_ENDC_4624)
            
            # SSA join for if statement (line 192)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA branch for the else part of an if statement (line 191)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Tuple (line 196):
        
        # Assigning a Call to a Name:
        
        # Call to setup_handles(...): (line 196)
        # Processing the call keyword arguments (line 196)
        kwargs_4627 = {}
        # Getting the type of 'setup_handles' (line 196)
        setup_handles_4626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 24), 'setup_handles', False)
        # Calling setup_handles(args, kwargs) (line 196)
        setup_handles_call_result_4628 = invoke(stypy.reporting.localization.Localization(__file__, 196, 24), setup_handles_4626, *[], **kwargs_4627)
        
        # Assigning a type to the variable 'call_assignment_4221' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_4221', setup_handles_call_result_4628)
        
        # Assigning a Call to a Name (line 196):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4221' (line 196)
        call_assignment_4221_4629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_4221', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4630 = stypy_get_value_from_tuple(call_assignment_4221_4629, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_4222' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_4222', stypy_get_value_from_tuple_call_result_4630)
        
        # Assigning a Name to a Name (line 196):
        # Getting the type of 'call_assignment_4222' (line 196)
        call_assignment_4222_4631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_4222')
        # Assigning a type to the variable 'handle' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'handle', call_assignment_4222_4631)
        
        # Assigning a Call to a Name (line 196):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4221' (line 196)
        call_assignment_4221_4632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_4221', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4633 = stypy_get_value_from_tuple(call_assignment_4221_4632, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_4223' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_4223', stypy_get_value_from_tuple_call_result_4633)
        
        # Assigning a Name to a Name (line 196):
        # Getting the type of 'call_assignment_4223' (line 196)
        call_assignment_4223_4634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'call_assignment_4223')
        # Assigning a type to the variable 'reset' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'reset', call_assignment_4223_4634)
        
        # Call to SetConsoleTextAttribute(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'handle' (line 197)
        handle_4639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 55), 'handle', False)
        # Getting the type of 'win_console_color' (line 197)
        win_console_color_4640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 63), 'win_console_color', False)
        # Processing the call keyword arguments (line 197)
        kwargs_4641 = {}
        # Getting the type of 'ctypes' (line 197)
        ctypes_4635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 197)
        windll_4636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), ctypes_4635, 'windll')
        # Obtaining the member 'kernel32' of a type (line 197)
        kernel32_4637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), windll_4636, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 197)
        SetConsoleTextAttribute_4638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), kernel32_4637, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 197)
        SetConsoleTextAttribute_call_result_4642 = invoke(stypy.reporting.localization.Localization(__file__, 197, 8), SetConsoleTextAttribute_4638, *[handle_4639, win_console_color_4640], **kwargs_4641)
        
        # Getting the type of 'output_to_console' (line 199)
        output_to_console_4643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 11), 'output_to_console')
        # Testing if the type of an if condition is none (line 199)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 199, 8), output_to_console_4643):
            pass
        else:
            
            # Testing the type of an if condition (line 199)
            if_condition_4644 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 8), output_to_console_4643)
            # Assigning a type to the variable 'if_condition_4644' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'if_condition_4644', if_condition_4644)
            # SSA begins for if statement (line 199)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'msg_type' (line 200)
            msg_type_4645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 18), 'msg_type')
            str_4646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 29), 'str', ': ')
            # Applying the binary operator '+' (line 200)
            result_add_4647 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 18), '+', msg_type_4645, str_4646)
            
            # Getting the type of 'msg' (line 200)
            msg_4648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 36), 'msg')
            # Applying the binary operator '+' (line 200)
            result_add_4649 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 34), '+', result_add_4647, msg_4648)
            
            # SSA join for if statement (line 199)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to SetConsoleTextAttribute(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'handle' (line 202)
        handle_4654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 55), 'handle', False)
        # Getting the type of 'reset' (line 202)
        reset_4655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 63), 'reset', False)
        # Processing the call keyword arguments (line 202)
        kwargs_4656 = {}
        # Getting the type of 'ctypes' (line 202)
        ctypes_4650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 202)
        windll_4651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), ctypes_4650, 'windll')
        # Obtaining the member 'kernel32' of a type (line 202)
        kernel32_4652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), windll_4651, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 202)
        SetConsoleTextAttribute_4653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), kernel32_4652, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 202)
        SetConsoleTextAttribute_call_result_4657 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), SetConsoleTextAttribute_4653, *[handle_4654, reset_4655], **kwargs_4656)
        
        # SSA join for if statement (line 191)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # SSA begins for try-except statement (line 204)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 205):
    
    # Assigning a Call to a Name (line 205):
    
    # Call to open(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'stypy_parameters_copy' (line 205)
    stypy_parameters_copy_4659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 21), 'stypy_parameters_copy', False)
    # Obtaining the member 'LOG_PATH' of a type (line 205)
    LOG_PATH_4660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 21), stypy_parameters_copy_4659, 'LOG_PATH')
    str_4661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 54), 'str', '/')
    # Applying the binary operator '+' (line 205)
    result_add_4662 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 21), '+', LOG_PATH_4660, str_4661)
    
    # Getting the type of 'file_name' (line 205)
    file_name_4663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 60), 'file_name', False)
    # Applying the binary operator '+' (line 205)
    result_add_4664 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 58), '+', result_add_4662, file_name_4663)
    
    str_4665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 71), 'str', 'a')
    # Processing the call keyword arguments (line 205)
    kwargs_4666 = {}
    # Getting the type of 'open' (line 205)
    open_4658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 16), 'open', False)
    # Calling open(args, kwargs) (line 205)
    open_call_result_4667 = invoke(stypy.reporting.localization.Localization(__file__, 205, 16), open_4658, *[result_add_4664, str_4665], **kwargs_4666)
    
    # Assigning a type to the variable 'file_' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'file_', open_call_result_4667)
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
    call_data_4669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 14), 'call_data', False)
    # Processing the call keyword arguments (line 209)
    kwargs_4670 = {}
    # Getting the type of 'str' (line 209)
    str_4668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 10), 'str', False)
    # Calling str(args, kwargs) (line 209)
    str_call_result_4671 = invoke(stypy.reporting.localization.Localization(__file__, 209, 10), str_4668, *[call_data_4669], **kwargs_4670)
    
    str_4672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 27), 'str', ' (')
    # Applying the binary operator '+' (line 209)
    result_add_4673 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 10), '+', str_call_result_4671, str_4672)
    
    
    # Call to get_date_time(...): (line 209)
    # Processing the call keyword arguments (line 209)
    kwargs_4675 = {}
    # Getting the type of 'get_date_time' (line 209)
    get_date_time_4674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 34), 'get_date_time', False)
    # Calling get_date_time(args, kwargs) (line 209)
    get_date_time_call_result_4676 = invoke(stypy.reporting.localization.Localization(__file__, 209, 34), get_date_time_4674, *[], **kwargs_4675)
    
    # Applying the binary operator '+' (line 209)
    result_add_4677 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 32), '+', result_add_4673, get_date_time_call_result_4676)
    
    str_4678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 52), 'str', '). ')
    # Applying the binary operator '+' (line 209)
    result_add_4679 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 50), '+', result_add_4677, str_4678)
    
    # Getting the type of 'msg_type' (line 209)
    msg_type_4680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 60), 'msg_type')
    # Applying the binary operator '+' (line 209)
    result_add_4681 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 58), '+', result_add_4679, msg_type_4680)
    
    str_4682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 71), 'str', ': ')
    # Applying the binary operator '+' (line 209)
    result_add_4683 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 69), '+', result_add_4681, str_4682)
    
    # Getting the type of 'msg' (line 209)
    msg_4684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 78), 'msg')
    # Applying the binary operator '+' (line 209)
    result_add_4685 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 76), '+', result_add_4683, msg_4684)
    
    # Assigning a type to the variable 'txt' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'txt', result_add_4685)
    
    # Call to write(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'txt' (line 210)
    txt_4688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'txt', False)
    str_4689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 22), 'str', '\n')
    # Applying the binary operator '+' (line 210)
    result_add_4690 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 16), '+', txt_4688, str_4689)
    
    # Processing the call keyword arguments (line 210)
    kwargs_4691 = {}
    # Getting the type of 'file_' (line 210)
    file__4686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'file_', False)
    # Obtaining the member 'write' of a type (line 210)
    write_4687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 4), file__4686, 'write')
    # Calling write(args, kwargs) (line 210)
    write_call_result_4692 = invoke(stypy.reporting.localization.Localization(__file__, 210, 4), write_4687, *[result_add_4690], **kwargs_4691)
    
    
    # Call to close(...): (line 211)
    # Processing the call keyword arguments (line 211)
    kwargs_4695 = {}
    # Getting the type of 'file_' (line 211)
    file__4693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'file_', False)
    # Obtaining the member 'close' of a type (line 211)
    close_4694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 4), file__4693, 'close')
    # Calling close(args, kwargs) (line 211)
    close_call_result_4696 = invoke(stypy.reporting.localization.Localization(__file__, 211, 4), close_4694, *[], **kwargs_4695)
    
    
    # ################# End of '__aux_warning_and_error_write(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__aux_warning_and_error_write' in the type store
    # Getting the type of 'stypy_return_type' (line 180)
    stypy_return_type_4697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4697)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__aux_warning_and_error_write'
    return stypy_return_type_4697

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

    str_4698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, (-1)), 'str', '\n    Proxy for __aux_warning_and_error_write, supplying parameters to write warning messages\n    :param msg:\n    :return:\n    ')
    
    # Assigning a Call to a Name (line 220):
    
    # Assigning a Call to a Name (line 220):
    
    # Call to get_caller_data(...): (line 220)
    # Processing the call keyword arguments (line 220)
    kwargs_4700 = {}
    # Getting the type of 'get_caller_data' (line 220)
    get_caller_data_4699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'get_caller_data', False)
    # Calling get_caller_data(args, kwargs) (line 220)
    get_caller_data_call_result_4701 = invoke(stypy.reporting.localization.Localization(__file__, 220, 16), get_caller_data_4699, *[], **kwargs_4700)
    
    # Assigning a type to the variable 'call_data' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'call_data', get_caller_data_call_result_4701)
    
    # Call to __aux_warning_and_error_write(...): (line 221)
    # Processing the call arguments (line 221)
    # Getting the type of 'msg' (line 221)
    msg_4703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 34), 'msg', False)
    # Getting the type of 'call_data' (line 221)
    call_data_4704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 39), 'call_data', False)
    # Getting the type of 'Colors' (line 221)
    Colors_4705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 50), 'Colors', False)
    # Obtaining the member 'ANSI_WARNING' of a type (line 221)
    ANSI_WARNING_4706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 50), Colors_4705, 'ANSI_WARNING')
    # Getting the type of 'Colors' (line 221)
    Colors_4707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 71), 'Colors', False)
    # Obtaining the member 'WIN_WARNING' of a type (line 221)
    WIN_WARNING_4708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 71), Colors_4707, 'WIN_WARNING')
    # Getting the type of 'stypy_parameters_copy' (line 222)
    stypy_parameters_copy_4709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 34), 'stypy_parameters_copy', False)
    # Obtaining the member 'WARNING_LOG_FILE' of a type (line 222)
    WARNING_LOG_FILE_4710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 34), stypy_parameters_copy_4709, 'WARNING_LOG_FILE')
    str_4711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 74), 'str', 'WARNING')
    # Processing the call keyword arguments (line 221)
    kwargs_4712 = {}
    # Getting the type of '__aux_warning_and_error_write' (line 221)
    aux_warning_and_error_write_4702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), '__aux_warning_and_error_write', False)
    # Calling __aux_warning_and_error_write(args, kwargs) (line 221)
    aux_warning_and_error_write_call_result_4713 = invoke(stypy.reporting.localization.Localization(__file__, 221, 4), aux_warning_and_error_write_4702, *[msg_4703, call_data_4704, ANSI_WARNING_4706, WIN_WARNING_4708, WARNING_LOG_FILE_4710, str_4711], **kwargs_4712)
    
    
    # ################# End of 'warning(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'warning' in the type store
    # Getting the type of 'stypy_return_type' (line 214)
    stypy_return_type_4714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4714)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'warning'
    return stypy_return_type_4714

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

    str_4715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, (-1)), 'str', '\n    Proxy for __aux_warning_and_error_write, supplying parameters to write error messages\n    :param msg:\n    :return:\n    ')
    
    # Assigning a Call to a Name (line 231):
    
    # Assigning a Call to a Name (line 231):
    
    # Call to get_caller_data(...): (line 231)
    # Processing the call keyword arguments (line 231)
    kwargs_4717 = {}
    # Getting the type of 'get_caller_data' (line 231)
    get_caller_data_4716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'get_caller_data', False)
    # Calling get_caller_data(args, kwargs) (line 231)
    get_caller_data_call_result_4718 = invoke(stypy.reporting.localization.Localization(__file__, 231, 16), get_caller_data_4716, *[], **kwargs_4717)
    
    # Assigning a type to the variable 'call_data' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'call_data', get_caller_data_call_result_4718)
    
    # Call to __aux_warning_and_error_write(...): (line 232)
    # Processing the call arguments (line 232)
    # Getting the type of 'msg' (line 232)
    msg_4720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 34), 'msg', False)
    # Getting the type of 'call_data' (line 232)
    call_data_4721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 39), 'call_data', False)
    # Getting the type of 'Colors' (line 232)
    Colors_4722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 50), 'Colors', False)
    # Obtaining the member 'ANSI_FAIL' of a type (line 232)
    ANSI_FAIL_4723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 50), Colors_4722, 'ANSI_FAIL')
    # Getting the type of 'Colors' (line 232)
    Colors_4724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 68), 'Colors', False)
    # Obtaining the member 'WIN_FAIL' of a type (line 232)
    WIN_FAIL_4725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 68), Colors_4724, 'WIN_FAIL')
    # Getting the type of 'stypy_parameters_copy' (line 232)
    stypy_parameters_copy_4726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 85), 'stypy_parameters_copy', False)
    # Obtaining the member 'ERROR_LOG_FILE' of a type (line 232)
    ERROR_LOG_FILE_4727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 85), stypy_parameters_copy_4726, 'ERROR_LOG_FILE')
    str_4728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 34), 'str', 'ERROR')
    # Processing the call keyword arguments (line 232)
    kwargs_4729 = {}
    # Getting the type of '__aux_warning_and_error_write' (line 232)
    aux_warning_and_error_write_4719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), '__aux_warning_and_error_write', False)
    # Calling __aux_warning_and_error_write(args, kwargs) (line 232)
    aux_warning_and_error_write_call_result_4730 = invoke(stypy.reporting.localization.Localization(__file__, 232, 4), aux_warning_and_error_write_4719, *[msg_4720, call_data_4721, ANSI_FAIL_4723, WIN_FAIL_4725, ERROR_LOG_FILE_4727, str_4728], **kwargs_4729)
    
    
    # ################# End of 'error(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'error' in the type store
    # Getting the type of 'stypy_return_type' (line 225)
    stypy_return_type_4731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4731)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'error'
    return stypy_return_type_4731

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

    str_4732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, (-1)), 'str', '\n    Put a header to the log files indicating that log messages below that header belong to a new execution\n    :return:\n    ')
    
    
    # SSA begins for try-except statement (line 241)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 242):
    
    # Assigning a Call to a Name (line 242):
    
    # Call to open(...): (line 242)
    # Processing the call arguments (line 242)
    # Getting the type of 'stypy_parameters_copy' (line 242)
    stypy_parameters_copy_4734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 21), 'stypy_parameters_copy', False)
    # Obtaining the member 'LOG_PATH' of a type (line 242)
    LOG_PATH_4735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 21), stypy_parameters_copy_4734, 'LOG_PATH')
    str_4736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 54), 'str', '/')
    # Applying the binary operator '+' (line 242)
    result_add_4737 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 21), '+', LOG_PATH_4735, str_4736)
    
    # Getting the type of 'stypy_parameters_copy' (line 242)
    stypy_parameters_copy_4738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 60), 'stypy_parameters_copy', False)
    # Obtaining the member 'ERROR_LOG_FILE' of a type (line 242)
    ERROR_LOG_FILE_4739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 60), stypy_parameters_copy_4738, 'ERROR_LOG_FILE')
    # Applying the binary operator '+' (line 242)
    result_add_4740 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 58), '+', result_add_4737, ERROR_LOG_FILE_4739)
    
    str_4741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 98), 'str', 'a')
    # Processing the call keyword arguments (line 242)
    kwargs_4742 = {}
    # Getting the type of 'open' (line 242)
    open_4733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'open', False)
    # Calling open(args, kwargs) (line 242)
    open_call_result_4743 = invoke(stypy.reporting.localization.Localization(__file__, 242, 16), open_4733, *[result_add_4740, str_4741], **kwargs_4742)
    
    # Assigning a type to the variable 'file_' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'file_', open_call_result_4743)
    
    # Call to write(...): (line 243)
    # Processing the call arguments (line 243)
    str_4746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 20), 'str', '\n\n')
    # Processing the call keyword arguments (line 243)
    kwargs_4747 = {}
    # Getting the type of 'file_' (line 243)
    file__4744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 243)
    write_4745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), file__4744, 'write')
    # Calling write(args, kwargs) (line 243)
    write_call_result_4748 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), write_4745, *[str_4746], **kwargs_4747)
    
    
    # Call to write(...): (line 244)
    # Processing the call arguments (line 244)
    str_4751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 20), 'str', 'NEW LOGGING SESSION BEGIN AT: ')
    
    # Call to get_date_time(...): (line 244)
    # Processing the call keyword arguments (line 244)
    kwargs_4753 = {}
    # Getting the type of 'get_date_time' (line 244)
    get_date_time_4752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 55), 'get_date_time', False)
    # Calling get_date_time(args, kwargs) (line 244)
    get_date_time_call_result_4754 = invoke(stypy.reporting.localization.Localization(__file__, 244, 55), get_date_time_4752, *[], **kwargs_4753)
    
    # Applying the binary operator '+' (line 244)
    result_add_4755 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 20), '+', str_4751, get_date_time_call_result_4754)
    
    # Processing the call keyword arguments (line 244)
    kwargs_4756 = {}
    # Getting the type of 'file_' (line 244)
    file__4749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 244)
    write_4750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), file__4749, 'write')
    # Calling write(args, kwargs) (line 244)
    write_call_result_4757 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), write_4750, *[result_add_4755], **kwargs_4756)
    
    
    # Call to write(...): (line 245)
    # Processing the call arguments (line 245)
    str_4760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 20), 'str', '\n\n')
    # Processing the call keyword arguments (line 245)
    kwargs_4761 = {}
    # Getting the type of 'file_' (line 245)
    file__4758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 245)
    write_4759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 8), file__4758, 'write')
    # Calling write(args, kwargs) (line 245)
    write_call_result_4762 = invoke(stypy.reporting.localization.Localization(__file__, 245, 8), write_4759, *[str_4760], **kwargs_4761)
    
    
    # Call to close(...): (line 246)
    # Processing the call keyword arguments (line 246)
    kwargs_4765 = {}
    # Getting the type of 'file_' (line 246)
    file__4763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'file_', False)
    # Obtaining the member 'close' of a type (line 246)
    close_4764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), file__4763, 'close')
    # Calling close(args, kwargs) (line 246)
    close_call_result_4766 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), close_4764, *[], **kwargs_4765)
    
    
    # Assigning a Call to a Name (line 248):
    
    # Assigning a Call to a Name (line 248):
    
    # Call to open(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'stypy_parameters_copy' (line 248)
    stypy_parameters_copy_4768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 21), 'stypy_parameters_copy', False)
    # Obtaining the member 'LOG_PATH' of a type (line 248)
    LOG_PATH_4769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 21), stypy_parameters_copy_4768, 'LOG_PATH')
    str_4770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 54), 'str', '/')
    # Applying the binary operator '+' (line 248)
    result_add_4771 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 21), '+', LOG_PATH_4769, str_4770)
    
    # Getting the type of 'stypy_parameters_copy' (line 248)
    stypy_parameters_copy_4772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 60), 'stypy_parameters_copy', False)
    # Obtaining the member 'INFO_LOG_FILE' of a type (line 248)
    INFO_LOG_FILE_4773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 60), stypy_parameters_copy_4772, 'INFO_LOG_FILE')
    # Applying the binary operator '+' (line 248)
    result_add_4774 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 58), '+', result_add_4771, INFO_LOG_FILE_4773)
    
    str_4775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 97), 'str', 'a')
    # Processing the call keyword arguments (line 248)
    kwargs_4776 = {}
    # Getting the type of 'open' (line 248)
    open_4767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'open', False)
    # Calling open(args, kwargs) (line 248)
    open_call_result_4777 = invoke(stypy.reporting.localization.Localization(__file__, 248, 16), open_4767, *[result_add_4774, str_4775], **kwargs_4776)
    
    # Assigning a type to the variable 'file_' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'file_', open_call_result_4777)
    
    # Call to write(...): (line 249)
    # Processing the call arguments (line 249)
    str_4780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 20), 'str', '\n\n')
    # Processing the call keyword arguments (line 249)
    kwargs_4781 = {}
    # Getting the type of 'file_' (line 249)
    file__4778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 249)
    write_4779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), file__4778, 'write')
    # Calling write(args, kwargs) (line 249)
    write_call_result_4782 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), write_4779, *[str_4780], **kwargs_4781)
    
    
    # Call to write(...): (line 250)
    # Processing the call arguments (line 250)
    str_4785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 20), 'str', 'NEW LOGGING SESSION BEGIN AT: ')
    
    # Call to get_date_time(...): (line 250)
    # Processing the call keyword arguments (line 250)
    kwargs_4787 = {}
    # Getting the type of 'get_date_time' (line 250)
    get_date_time_4786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 55), 'get_date_time', False)
    # Calling get_date_time(args, kwargs) (line 250)
    get_date_time_call_result_4788 = invoke(stypy.reporting.localization.Localization(__file__, 250, 55), get_date_time_4786, *[], **kwargs_4787)
    
    # Applying the binary operator '+' (line 250)
    result_add_4789 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 20), '+', str_4785, get_date_time_call_result_4788)
    
    # Processing the call keyword arguments (line 250)
    kwargs_4790 = {}
    # Getting the type of 'file_' (line 250)
    file__4783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 250)
    write_4784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), file__4783, 'write')
    # Calling write(args, kwargs) (line 250)
    write_call_result_4791 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), write_4784, *[result_add_4789], **kwargs_4790)
    
    
    # Call to write(...): (line 251)
    # Processing the call arguments (line 251)
    str_4794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 20), 'str', '\n\n')
    # Processing the call keyword arguments (line 251)
    kwargs_4795 = {}
    # Getting the type of 'file_' (line 251)
    file__4792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 251)
    write_4793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), file__4792, 'write')
    # Calling write(args, kwargs) (line 251)
    write_call_result_4796 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), write_4793, *[str_4794], **kwargs_4795)
    
    
    # Call to close(...): (line 252)
    # Processing the call keyword arguments (line 252)
    kwargs_4799 = {}
    # Getting the type of 'file_' (line 252)
    file__4797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'file_', False)
    # Obtaining the member 'close' of a type (line 252)
    close_4798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 8), file__4797, 'close')
    # Calling close(args, kwargs) (line 252)
    close_call_result_4800 = invoke(stypy.reporting.localization.Localization(__file__, 252, 8), close_4798, *[], **kwargs_4799)
    
    
    # Assigning a Call to a Name (line 254):
    
    # Assigning a Call to a Name (line 254):
    
    # Call to open(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'stypy_parameters_copy' (line 254)
    stypy_parameters_copy_4802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 21), 'stypy_parameters_copy', False)
    # Obtaining the member 'LOG_PATH' of a type (line 254)
    LOG_PATH_4803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 21), stypy_parameters_copy_4802, 'LOG_PATH')
    str_4804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 54), 'str', '/')
    # Applying the binary operator '+' (line 254)
    result_add_4805 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 21), '+', LOG_PATH_4803, str_4804)
    
    # Getting the type of 'stypy_parameters_copy' (line 254)
    stypy_parameters_copy_4806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 60), 'stypy_parameters_copy', False)
    # Obtaining the member 'WARNING_LOG_FILE' of a type (line 254)
    WARNING_LOG_FILE_4807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 60), stypy_parameters_copy_4806, 'WARNING_LOG_FILE')
    # Applying the binary operator '+' (line 254)
    result_add_4808 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 58), '+', result_add_4805, WARNING_LOG_FILE_4807)
    
    str_4809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 100), 'str', 'a')
    # Processing the call keyword arguments (line 254)
    kwargs_4810 = {}
    # Getting the type of 'open' (line 254)
    open_4801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'open', False)
    # Calling open(args, kwargs) (line 254)
    open_call_result_4811 = invoke(stypy.reporting.localization.Localization(__file__, 254, 16), open_4801, *[result_add_4808, str_4809], **kwargs_4810)
    
    # Assigning a type to the variable 'file_' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'file_', open_call_result_4811)
    
    # Call to write(...): (line 255)
    # Processing the call arguments (line 255)
    str_4814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 20), 'str', '\n\n')
    # Processing the call keyword arguments (line 255)
    kwargs_4815 = {}
    # Getting the type of 'file_' (line 255)
    file__4812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 255)
    write_4813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 8), file__4812, 'write')
    # Calling write(args, kwargs) (line 255)
    write_call_result_4816 = invoke(stypy.reporting.localization.Localization(__file__, 255, 8), write_4813, *[str_4814], **kwargs_4815)
    
    
    # Call to write(...): (line 256)
    # Processing the call arguments (line 256)
    str_4819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 20), 'str', 'NEW LOGGING SESSION BEGIN AT: ')
    
    # Call to get_date_time(...): (line 256)
    # Processing the call keyword arguments (line 256)
    kwargs_4821 = {}
    # Getting the type of 'get_date_time' (line 256)
    get_date_time_4820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 55), 'get_date_time', False)
    # Calling get_date_time(args, kwargs) (line 256)
    get_date_time_call_result_4822 = invoke(stypy.reporting.localization.Localization(__file__, 256, 55), get_date_time_4820, *[], **kwargs_4821)
    
    # Applying the binary operator '+' (line 256)
    result_add_4823 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 20), '+', str_4819, get_date_time_call_result_4822)
    
    # Processing the call keyword arguments (line 256)
    kwargs_4824 = {}
    # Getting the type of 'file_' (line 256)
    file__4817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 256)
    write_4818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 8), file__4817, 'write')
    # Calling write(args, kwargs) (line 256)
    write_call_result_4825 = invoke(stypy.reporting.localization.Localization(__file__, 256, 8), write_4818, *[result_add_4823], **kwargs_4824)
    
    
    # Call to write(...): (line 257)
    # Processing the call arguments (line 257)
    str_4828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 20), 'str', '\n\n')
    # Processing the call keyword arguments (line 257)
    kwargs_4829 = {}
    # Getting the type of 'file_' (line 257)
    file__4826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 257)
    write_4827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), file__4826, 'write')
    # Calling write(args, kwargs) (line 257)
    write_call_result_4830 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), write_4827, *[str_4828], **kwargs_4829)
    
    
    # Call to close(...): (line 258)
    # Processing the call keyword arguments (line 258)
    kwargs_4833 = {}
    # Getting the type of 'file_' (line 258)
    file__4831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'file_', False)
    # Obtaining the member 'close' of a type (line 258)
    close_4832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 8), file__4831, 'close')
    # Calling close(args, kwargs) (line 258)
    close_call_result_4834 = invoke(stypy.reporting.localization.Localization(__file__, 258, 8), close_4832, *[], **kwargs_4833)
    
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
    stypy_return_type_4835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4835)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'new_logging_session'
    return stypy_return_type_4835

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

    str_4836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, (-1)), 'str', '\n    Erases log files\n    :return:\n    ')
    
    
    # SSA begins for try-except statement (line 268)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 269):
    
    # Assigning a Call to a Name (line 269):
    
    # Call to open(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'stypy_parameters_copy' (line 269)
    stypy_parameters_copy_4838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 21), 'stypy_parameters_copy', False)
    # Obtaining the member 'LOG_PATH' of a type (line 269)
    LOG_PATH_4839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 21), stypy_parameters_copy_4838, 'LOG_PATH')
    str_4840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 54), 'str', '/')
    # Applying the binary operator '+' (line 269)
    result_add_4841 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 21), '+', LOG_PATH_4839, str_4840)
    
    # Getting the type of 'stypy_parameters_copy' (line 269)
    stypy_parameters_copy_4842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 60), 'stypy_parameters_copy', False)
    # Obtaining the member 'ERROR_LOG_FILE' of a type (line 269)
    ERROR_LOG_FILE_4843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 60), stypy_parameters_copy_4842, 'ERROR_LOG_FILE')
    # Applying the binary operator '+' (line 269)
    result_add_4844 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 58), '+', result_add_4841, ERROR_LOG_FILE_4843)
    
    str_4845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 98), 'str', 'w')
    # Processing the call keyword arguments (line 269)
    kwargs_4846 = {}
    # Getting the type of 'open' (line 269)
    open_4837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 16), 'open', False)
    # Calling open(args, kwargs) (line 269)
    open_call_result_4847 = invoke(stypy.reporting.localization.Localization(__file__, 269, 16), open_4837, *[result_add_4844, str_4845], **kwargs_4846)
    
    # Assigning a type to the variable 'file_' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'file_', open_call_result_4847)
    
    # Call to write(...): (line 270)
    # Processing the call arguments (line 270)
    str_4850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 20), 'str', '')
    # Processing the call keyword arguments (line 270)
    kwargs_4851 = {}
    # Getting the type of 'file_' (line 270)
    file__4848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 270)
    write_4849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), file__4848, 'write')
    # Calling write(args, kwargs) (line 270)
    write_call_result_4852 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), write_4849, *[str_4850], **kwargs_4851)
    
    
    # Call to close(...): (line 271)
    # Processing the call keyword arguments (line 271)
    kwargs_4855 = {}
    # Getting the type of 'file_' (line 271)
    file__4853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'file_', False)
    # Obtaining the member 'close' of a type (line 271)
    close_4854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 8), file__4853, 'close')
    # Calling close(args, kwargs) (line 271)
    close_call_result_4856 = invoke(stypy.reporting.localization.Localization(__file__, 271, 8), close_4854, *[], **kwargs_4855)
    
    
    # Assigning a Call to a Name (line 273):
    
    # Assigning a Call to a Name (line 273):
    
    # Call to open(...): (line 273)
    # Processing the call arguments (line 273)
    # Getting the type of 'stypy_parameters_copy' (line 273)
    stypy_parameters_copy_4858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 21), 'stypy_parameters_copy', False)
    # Obtaining the member 'LOG_PATH' of a type (line 273)
    LOG_PATH_4859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 21), stypy_parameters_copy_4858, 'LOG_PATH')
    str_4860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 54), 'str', '/')
    # Applying the binary operator '+' (line 273)
    result_add_4861 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 21), '+', LOG_PATH_4859, str_4860)
    
    # Getting the type of 'stypy_parameters_copy' (line 273)
    stypy_parameters_copy_4862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 60), 'stypy_parameters_copy', False)
    # Obtaining the member 'WARNING_LOG_FILE' of a type (line 273)
    WARNING_LOG_FILE_4863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 60), stypy_parameters_copy_4862, 'WARNING_LOG_FILE')
    # Applying the binary operator '+' (line 273)
    result_add_4864 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 58), '+', result_add_4861, WARNING_LOG_FILE_4863)
    
    str_4865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 100), 'str', 'w')
    # Processing the call keyword arguments (line 273)
    kwargs_4866 = {}
    # Getting the type of 'open' (line 273)
    open_4857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'open', False)
    # Calling open(args, kwargs) (line 273)
    open_call_result_4867 = invoke(stypy.reporting.localization.Localization(__file__, 273, 16), open_4857, *[result_add_4864, str_4865], **kwargs_4866)
    
    # Assigning a type to the variable 'file_' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'file_', open_call_result_4867)
    
    # Call to write(...): (line 274)
    # Processing the call arguments (line 274)
    str_4870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 20), 'str', '')
    # Processing the call keyword arguments (line 274)
    kwargs_4871 = {}
    # Getting the type of 'file_' (line 274)
    file__4868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 274)
    write_4869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), file__4868, 'write')
    # Calling write(args, kwargs) (line 274)
    write_call_result_4872 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), write_4869, *[str_4870], **kwargs_4871)
    
    
    # Call to close(...): (line 275)
    # Processing the call keyword arguments (line 275)
    kwargs_4875 = {}
    # Getting the type of 'file_' (line 275)
    file__4873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'file_', False)
    # Obtaining the member 'close' of a type (line 275)
    close_4874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), file__4873, 'close')
    # Calling close(args, kwargs) (line 275)
    close_call_result_4876 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), close_4874, *[], **kwargs_4875)
    
    
    # Assigning a Call to a Name (line 277):
    
    # Assigning a Call to a Name (line 277):
    
    # Call to open(...): (line 277)
    # Processing the call arguments (line 277)
    # Getting the type of 'stypy_parameters_copy' (line 277)
    stypy_parameters_copy_4878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 21), 'stypy_parameters_copy', False)
    # Obtaining the member 'LOG_PATH' of a type (line 277)
    LOG_PATH_4879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 21), stypy_parameters_copy_4878, 'LOG_PATH')
    str_4880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 54), 'str', '/')
    # Applying the binary operator '+' (line 277)
    result_add_4881 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 21), '+', LOG_PATH_4879, str_4880)
    
    # Getting the type of 'stypy_parameters_copy' (line 277)
    stypy_parameters_copy_4882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 60), 'stypy_parameters_copy', False)
    # Obtaining the member 'INFO_LOG_FILE' of a type (line 277)
    INFO_LOG_FILE_4883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 60), stypy_parameters_copy_4882, 'INFO_LOG_FILE')
    # Applying the binary operator '+' (line 277)
    result_add_4884 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 58), '+', result_add_4881, INFO_LOG_FILE_4883)
    
    str_4885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 97), 'str', 'w')
    # Processing the call keyword arguments (line 277)
    kwargs_4886 = {}
    # Getting the type of 'open' (line 277)
    open_4877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 16), 'open', False)
    # Calling open(args, kwargs) (line 277)
    open_call_result_4887 = invoke(stypy.reporting.localization.Localization(__file__, 277, 16), open_4877, *[result_add_4884, str_4885], **kwargs_4886)
    
    # Assigning a type to the variable 'file_' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'file_', open_call_result_4887)
    
    # Call to write(...): (line 278)
    # Processing the call arguments (line 278)
    str_4890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 20), 'str', '')
    # Processing the call keyword arguments (line 278)
    kwargs_4891 = {}
    # Getting the type of 'file_' (line 278)
    file__4888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'file_', False)
    # Obtaining the member 'write' of a type (line 278)
    write_4889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), file__4888, 'write')
    # Calling write(args, kwargs) (line 278)
    write_call_result_4892 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), write_4889, *[str_4890], **kwargs_4891)
    
    
    # Call to close(...): (line 279)
    # Processing the call keyword arguments (line 279)
    kwargs_4895 = {}
    # Getting the type of 'file_' (line 279)
    file__4893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'file_', False)
    # Obtaining the member 'close' of a type (line 279)
    close_4894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 8), file__4893, 'close')
    # Calling close(args, kwargs) (line 279)
    close_call_result_4896 = invoke(stypy.reporting.localization.Localization(__file__, 279, 8), close_4894, *[], **kwargs_4895)
    
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
    stypy_return_type_4897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4897)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'reset_logs'
    return stypy_return_type_4897

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

    str_4898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, (-1)), 'str', '\n    Reset Windows colors to leave the console with the default ones\n    :return:\n    ')
    # Getting the type of 'ColorType' (line 289)
    ColorType_4899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 7), 'ColorType')
    # Obtaining the member 'ANSIColors' of a type (line 289)
    ANSIColors_4900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 7), ColorType_4899, 'ANSIColors')
    # Testing if the type of an if condition is none (line 289)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 289, 4), ANSIColors_4900):
        
        # Assigning a Call to a Tuple (line 292):
        
        # Assigning a Call to a Name:
        
        # Call to setup_handles(...): (line 292)
        # Processing the call keyword arguments (line 292)
        kwargs_4903 = {}
        # Getting the type of 'setup_handles' (line 292)
        setup_handles_4902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 24), 'setup_handles', False)
        # Calling setup_handles(args, kwargs) (line 292)
        setup_handles_call_result_4904 = invoke(stypy.reporting.localization.Localization(__file__, 292, 24), setup_handles_4902, *[], **kwargs_4903)
        
        # Assigning a type to the variable 'call_assignment_4224' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_4224', setup_handles_call_result_4904)
        
        # Assigning a Call to a Name (line 292):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4224' (line 292)
        call_assignment_4224_4905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_4224', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4906 = stypy_get_value_from_tuple(call_assignment_4224_4905, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_4225' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_4225', stypy_get_value_from_tuple_call_result_4906)
        
        # Assigning a Name to a Name (line 292):
        # Getting the type of 'call_assignment_4225' (line 292)
        call_assignment_4225_4907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_4225')
        # Assigning a type to the variable 'handle' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'handle', call_assignment_4225_4907)
        
        # Assigning a Call to a Name (line 292):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4224' (line 292)
        call_assignment_4224_4908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_4224', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4909 = stypy_get_value_from_tuple(call_assignment_4224_4908, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_4226' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_4226', stypy_get_value_from_tuple_call_result_4909)
        
        # Assigning a Name to a Name (line 292):
        # Getting the type of 'call_assignment_4226' (line 292)
        call_assignment_4226_4910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_4226')
        # Assigning a type to the variable 'reset' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 16), 'reset', call_assignment_4226_4910)
        
        # Call to SetConsoleTextAttribute(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'handle' (line 293)
        handle_4915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 55), 'handle', False)
        # Getting the type of 'reset' (line 293)
        reset_4916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 63), 'reset', False)
        # Processing the call keyword arguments (line 293)
        kwargs_4917 = {}
        # Getting the type of 'ctypes' (line 293)
        ctypes_4911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 293)
        windll_4912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), ctypes_4911, 'windll')
        # Obtaining the member 'kernel32' of a type (line 293)
        kernel32_4913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), windll_4912, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 293)
        SetConsoleTextAttribute_4914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), kernel32_4913, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 293)
        SetConsoleTextAttribute_call_result_4918 = invoke(stypy.reporting.localization.Localization(__file__, 293, 8), SetConsoleTextAttribute_4914, *[handle_4915, reset_4916], **kwargs_4917)
        
    else:
        
        # Testing the type of an if condition (line 289)
        if_condition_4901 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 289, 4), ANSIColors_4900)
        # Assigning a type to the variable 'if_condition_4901' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'if_condition_4901', if_condition_4901)
        # SSA begins for if statement (line 289)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 289)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Tuple (line 292):
        
        # Assigning a Call to a Name:
        
        # Call to setup_handles(...): (line 292)
        # Processing the call keyword arguments (line 292)
        kwargs_4903 = {}
        # Getting the type of 'setup_handles' (line 292)
        setup_handles_4902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 24), 'setup_handles', False)
        # Calling setup_handles(args, kwargs) (line 292)
        setup_handles_call_result_4904 = invoke(stypy.reporting.localization.Localization(__file__, 292, 24), setup_handles_4902, *[], **kwargs_4903)
        
        # Assigning a type to the variable 'call_assignment_4224' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_4224', setup_handles_call_result_4904)
        
        # Assigning a Call to a Name (line 292):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4224' (line 292)
        call_assignment_4224_4905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_4224', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4906 = stypy_get_value_from_tuple(call_assignment_4224_4905, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_4225' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_4225', stypy_get_value_from_tuple_call_result_4906)
        
        # Assigning a Name to a Name (line 292):
        # Getting the type of 'call_assignment_4225' (line 292)
        call_assignment_4225_4907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_4225')
        # Assigning a type to the variable 'handle' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'handle', call_assignment_4225_4907)
        
        # Assigning a Call to a Name (line 292):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_4224' (line 292)
        call_assignment_4224_4908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_4224', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_4909 = stypy_get_value_from_tuple(call_assignment_4224_4908, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_4226' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_4226', stypy_get_value_from_tuple_call_result_4909)
        
        # Assigning a Name to a Name (line 292):
        # Getting the type of 'call_assignment_4226' (line 292)
        call_assignment_4226_4910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'call_assignment_4226')
        # Assigning a type to the variable 'reset' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 16), 'reset', call_assignment_4226_4910)
        
        # Call to SetConsoleTextAttribute(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'handle' (line 293)
        handle_4915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 55), 'handle', False)
        # Getting the type of 'reset' (line 293)
        reset_4916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 63), 'reset', False)
        # Processing the call keyword arguments (line 293)
        kwargs_4917 = {}
        # Getting the type of 'ctypes' (line 293)
        ctypes_4911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'ctypes', False)
        # Obtaining the member 'windll' of a type (line 293)
        windll_4912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), ctypes_4911, 'windll')
        # Obtaining the member 'kernel32' of a type (line 293)
        kernel32_4913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), windll_4912, 'kernel32')
        # Obtaining the member 'SetConsoleTextAttribute' of a type (line 293)
        SetConsoleTextAttribute_4914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), kernel32_4913, 'SetConsoleTextAttribute')
        # Calling SetConsoleTextAttribute(args, kwargs) (line 293)
        SetConsoleTextAttribute_call_result_4918 = invoke(stypy.reporting.localization.Localization(__file__, 293, 8), SetConsoleTextAttribute_4914, *[handle_4915, reset_4916], **kwargs_4917)
        
        # SSA join for if statement (line 289)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'reset_colors(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'reset_colors' in the type store
    # Getting the type of 'stypy_return_type' (line 284)
    stypy_return_type_4919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4919)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'reset_colors'
    return stypy_return_type_4919

# Assigning a type to the variable 'reset_colors' (line 284)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 0), 'reset_colors', reset_colors)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
