
# functionargs: function
# functionargs(*args: tuple[str]) -> str 

def functionargs(*args):
    return args[0]

# args: tuple[str]; r1: str
r1 = functionargs('hi')
# x1: TypeError
x1 = r1.thisdonotexist()
# functionkw: function
# functionkw(**kwargs: dict[{str: str}]) -> TypeError 

def functionkw(**kwargs):
    return kwargs[0]

# functionkw2: function
# functionkw2(**kwargs: dict[{str: str}]) -> str 

def functionkw2(**kwargs):
    return kwargs['val']

# r2: TypeError; kwargs: dict[{str: str}]
r2 = functionkw(val='hi')
# x2: TypeError
x2 = r2.thisdonotexist()
# r3: str; kwargs: dict[{str: str}]
r3 = functionkw2(val='hi')
# x3: TypeError
x3 = r2.thisdonotexist()
# r4: str; kwargs: dict[{str: str}]
r4 = functionkw2(not_exist='hi')