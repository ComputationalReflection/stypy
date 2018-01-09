class Parent1:
    pass

class Child(Parent1):
    pass

class Parent2:
    pass


class HybridChild(Parent1, Parent2):
    pass


instance = HybridChild()
