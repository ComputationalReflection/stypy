def function_1(x):
    return x / 2

def function_2(x):
    return x / 2

def function_3(x):
    return x / 2

r1 = function_1("a")  # Reported on call site (not on the function code, no stack trace)
r2 = function_2(range(5))  # Reported on call site (not on the function code, no stack trace)
r3 = function_3(4)  # Nothing is reported, as the call is valid
