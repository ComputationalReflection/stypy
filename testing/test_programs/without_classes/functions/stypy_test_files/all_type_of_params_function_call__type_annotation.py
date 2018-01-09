
# f: function
# f(x: int, y: int, z: int, *arguments: tuple[int], **kwarguments: dict{}) -> None 

def f(x, y, z, *arguments, **kwarguments):
    pass

# f2: function
# f2(x: int, y: int, z: int, *arguments: tuple[int], **kwarguments: dict[{str: int}]) -> None 

def f2(x, y, z, *arguments, **kwarguments):
    pass

# f3: function
# f3(x: int, y: bool, z: str, *args: tuple[], **kwargs: dict[{str: str \/ int}]) -> None 

def f3(x=5, y=6, z=4, *args, **kwargs):
    pass

# y: int; x: int; z: int; arguments: tuple[int]; kwarguments: dict{}
f(2, 3, 4, 5, 6, 7)
# y: int; x: int; z: int; arguments: tuple[int]; kwarguments: dict[{str: int}]
f2(1, 2, 8, 6, 4, r=23)
# y: bool; x: int; z: str; args: tuple[]; kwargs: dict[{str: str \/ int}]
f3(z='1', x=4, y=True, r=11, s='12')