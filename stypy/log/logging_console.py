import datetime
import inspect

from stypy import stypy_parameters

output_to_console = True

"""
 Multi-platform terminal color messages to improve visual quality of the output
 Also handles message logging for stypy.
 This code has been adapted from tcaswell snippet, found in:
 http://stackoverflow.com/questions/2654113/python-how-to-get-the-callers-method-name-in-the-called-method
"""


def get_caller_data(skip=2):
    """Get a name of a caller in the format module.class.method

       `skip` specifies how many levels of stack to skip while getting caller
       name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.

       An empty string is returned if skipped levels exceed stack height
    """
    stack = inspect.stack()
    start = 0 + skip
    if len(stack) < start + 1:
        return ''
    parentframe = stack[start][0]

    name = []
    module = inspect.getmodule(parentframe)

    if module:
        # Name of the file (at the end of the path, removing the c of the .pyc extension
        name.append(module.__file__.split("\\")[-1])

    # detect classname
    if 'self' in parentframe.f_locals:
        name.append(parentframe.f_locals['self'].__class__.__name__)

    codename = parentframe.f_code.co_name

    if codename != '<module>':  # top level usually
        name.append(codename)  # function or a method
    del parentframe

    # Strip full file path
    name[0] = name[0].split("/")[-1]
    return str(name)


class ColorType:
    ANSIColors = False


try:
    import ctypes


    def setup_handles():
        """
        Determines if it is possible to have colored output
        :return:
        """
        # Constants from the Windows API
        STD_OUTPUT_HANDLE = -11

        def get_csbi_attributes(handle):
            # Based on IPython's winconsole.py, written by Alexander Belchenko
            import struct

            csbi = ctypes.create_string_buffer(22)
            res = ctypes.windll.kernel32.GetConsoleScreenBufferInfo(handle, csbi)
            # assert res

            (bufx, bufy, curx, cury, wattr,
             left, top, right, bottom, maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
            return wattr

        handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        reset = get_csbi_attributes(handle)
        return handle, reset


    ColorType.ANSIColors = False  # Windows do not support ANSI terminals

except Exception as e:
    ColorType.ANSIColors = True  # ANSI escape sequences work with other terminals


class Colors:
    ANSI_BLUE = '\033[94m'
    ANSI_GREEN = '\033[92m'
    ANSI_WARNING = '\033[93m'
    ANSI_FAIL = '\033[91m'
    ANSI_ENDC = '\033[0m'

    WIN_BLUE = 0x0009
    WIN_WHITE = 0x000F
    WIN_GREEN = 0x000A
    WIN_WARNING = 0x000E
    WIN_FAIL = 0x000C


def get_date_time():
    """
    Obtains current date and time
    :return:
    """
    return str(datetime.datetime.now())[:-7]


def log(msg):
    """
    Logs information messages to the corresponding log file
    :param msg:
    :return:
    """
    try:
        file_ = open(stypy_parameters.LOG_PATH + "/" + stypy_parameters.INFO_LOG_FILE, "a")
    except:
        return  # No log is not a critical error condition

    if (msg == "\n") or (msg == ""):
        if msg == "":
            file_.write(msg + "\n")
        else:
            file_.write(msg)
    else:
        if not (file_ is None):
            file_.write("[" + get_date_time() + "] " + msg + "\n")

    file_.close()


def ok(msg):
    """
    Handles green log information messages
    :param msg:
    :return:
    """
    txt = get_caller_data() + ": " + msg
    if ColorType.ANSIColors:
        if output_to_console:
            print(Colors.ANSI_GREEN + msg + Colors.ANSI_ENDC)
    else:
        handle, reset = setup_handles()
        ctypes.windll.kernel32.SetConsoleTextAttribute(handle, Colors.WIN_GREEN)

        if output_to_console:
            print (msg)

        ctypes.windll.kernel32.SetConsoleTextAttribute(handle, reset)

    log(txt)


def info(msg):
    """
    Handles white log information messages
    :param msg:
    :return:
    """
    txt = get_caller_data() + ": " + msg
    if ColorType.ANSIColors:
        if output_to_console:
            print(txt)
    else:
        handle, reset = setup_handles()
        ctypes.windll.kernel32.SetConsoleTextAttribute(handle, Colors.WIN_WHITE)

        if output_to_console:
            print (txt)

        ctypes.windll.kernel32.SetConsoleTextAttribute(handle, reset)

    log(txt)


def __aux_warning_and_error_write(msg, call_data, ansi_console_color, win_console_color, file_name, msg_type):
    """
    Helper function to output warning or error messages, depending on its parameters.
    :param msg: Message to print
    :param call_data: Caller information
    :param ansi_console_color: ANSI terminals color to use
    :param win_console_color: Windows terminals color to use
    :param file_name: File to write to
    :param msg_type: Type of message to write (WARNING/ERROR)
    :return:
    """
    if ColorType.ANSIColors:
        if output_to_console:
            txt = str(call_data) + ". " + msg_type + ": " + str(msg)
            print(ansi_console_color + txt + Colors.ANSI_ENDC)
    else:
        handle, reset = setup_handles()
        ctypes.windll.kernel32.SetConsoleTextAttribute(handle, win_console_color)

        if output_to_console:
            print(msg_type + ": " + msg)

        ctypes.windll.kernel32.SetConsoleTextAttribute(handle, reset)

    try:
        file_ = open(stypy_parameters.LOG_PATH + "/" + file_name, "a")
    except:
        return  # No log is not a critical error condition

    txt = str(call_data) + " (" + get_date_time() + "). " + msg_type + ": " + msg
    file_.write(txt + "\n")
    file_.close()


def warning(msg):
    """
    Proxy for __aux_warning_and_error_write, supplying parameters to write warning messages
    :param msg:
    :return:
    """
    call_data = get_caller_data()
    __aux_warning_and_error_write(msg, call_data, Colors.ANSI_WARNING, Colors.WIN_WARNING,
                                  stypy_parameters.WARNING_LOG_FILE, "WARNING")


def error(msg):
    """
    Proxy for __aux_warning_and_error_write, supplying parameters to write error messages
    :param msg:
    :return:
    """
    call_data = get_caller_data()
    __aux_warning_and_error_write(msg, call_data, Colors.ANSI_FAIL, Colors.WIN_FAIL, stypy_parameters.ERROR_LOG_FILE,
                                  "ERROR")


def new_logging_session():
    """
    Put a header to the log files indicating that log messages below that header belong to a new execution
    :return:
    """
    try:
        file_ = open(stypy_parameters.LOG_PATH + "/" + stypy_parameters.ERROR_LOG_FILE, "a")
        file_.write("\n\n")
        file_.write("NEW LOGGING SESSION BEGIN AT: " + get_date_time())
        file_.write("\n\n")
        file_.close()

        file_ = open(stypy_parameters.LOG_PATH + "/" + stypy_parameters.INFO_LOG_FILE, "a")
        file_.write("\n\n")
        file_.write("NEW LOGGING SESSION BEGIN AT: " + get_date_time())
        file_.write("\n\n")
        file_.close()

        file_ = open(stypy_parameters.LOG_PATH + "/" + stypy_parameters.WARNING_LOG_FILE, "a")
        file_.write("\n\n")
        file_.write("NEW LOGGING SESSION BEGIN AT: " + get_date_time())
        file_.write("\n\n")
        file_.close()
    except:
        return


def reset_logs():
    """
    Erases log files
    :return:
    """
    try:
        file_ = open(stypy_parameters.LOG_PATH + "/" + stypy_parameters.ERROR_LOG_FILE, "w")
        file_.write("")
        file_.close()

        file_ = open(stypy_parameters.LOG_PATH + "/" + stypy_parameters.WARNING_LOG_FILE, "w")
        file_.write("")
        file_.close()

        file_ = open(stypy_parameters.LOG_PATH + "/" + stypy_parameters.INFO_LOG_FILE, "w")
        file_.write("")
        file_.close()
    except:
        return


def reset_colors():
    """
    Reset Windows colors to leave the console with the default ones
    :return:
    """
    if ColorType.ANSIColors:
        pass  # ANSI consoles do not need resets
    else:
        handle, reset = setup_handles()
        ctypes.windll.kernel32.SetConsoleTextAttribute(handle, reset)
