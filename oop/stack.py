# DataStructure LIFO

class Stack:
    """Stack!
    """

    def __init__(self):
        self.elements = []

    def get_element(self):
        if len(self.elements) > 0:
            return self.elements.pop()
        else:
            return

    def add_element(self, val):
        self.elements.append(val)

    def __str__(self):
        result = "["
        for i, elm in enumerate(self.elements):
            result += f"{elm}"
            if i != len(self.elements) - 1:
                result += "->"
        result += "]"
        return result


s1 = Stack()
s1.add_element(2)
print(s1)
s1.add_element(25)
print(s1)
print(s1.get_element())
print(s1)
print(s1.get_element())
print(s1)
