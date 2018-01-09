
# identity: function
# identity(x: int) -> int /\ identity(x: str) -> str /\ identity(x: float) -> float 

def identity(x):
    return x

# y: int; x: int
y = identity(3)
# x: str; z: str
z = identity('3')
# x: float; w: float
w = identity(3.4)