# DataStructure FIFO

class Queue:
    """Queue!
    """

    def __init__(self):
        self.elements = []

    def peek(self):
        if len(self.elements) > 0:
            return self.elements.pop(0)
        else:
            return

    def push(self, val):
        self.elements.append(val)

    def __str__(self):
        result = "["
        for i, elm in enumerate(self.elements):
            result += f"{elm}"
            if i != len(self.elements) - 1:
                result += "->"
        result += "]"
        return result


q1 = Queue()
q1.push(2)
print(q1)
q1.push(25)
print(q1)
print(q1.peek())
print(q1)
print(q1.peek())
print(q1)
