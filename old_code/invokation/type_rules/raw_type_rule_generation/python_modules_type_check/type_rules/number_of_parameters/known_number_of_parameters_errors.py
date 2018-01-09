"""
Identifier functions of known "invalid number of parameters" errors. These are used to filter type
rule errors and keep only those that can be potentially recoverable when using the "type polling" method
to guess call parameter number(s).
"""
__known_num_of_parameters_errors = [
    lambda msg: ("expected" in msg and "argument" in msg),
    lambda msg: ("equired argument" in msg),
    lambda msg: ("equires at least" in msg),
    lambda msg: ("takes at least" in msg),
    lambda msg: ("takes at most" in msg),
    lambda msg: ("takes exactly" in msg),
    lambda msg: ("takes" in msg and "arguments" in msg),
    lambda msg: ("needs" in msg and "argument" in msg),
    lambda msg: ("needs" in msg and "args" in msg),
    lambda msg: ("expected" in msg and "arguments" in msg),
    lambda msg: ("requires" in msg and "but receives" in msg),
    lambda msg: ("requires" in msg and "but received" in msg),
    lambda msg: ("takes no parameters" in msg),
    lambda msg: ("takes no arguments" in msg),
]

def is_known_num_of_parameters_error(msg):
    """
    Determines if an error message belongs to a known "number of parameters" error.
    :param msg: Message (str)
    :return: Bool value
    """
    try:
        for error_func in __known_num_of_parameters_errors:
            if error_func(msg):
                return True
        return False
    except:
        return False
