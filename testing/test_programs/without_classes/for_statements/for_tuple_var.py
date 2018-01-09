
arguments = {'a': 1, 'b': 2}
ret_str = ""

for key, arg in arguments.items():
    ret_str += str(key) + ": " + str(arg)

print ret_str
