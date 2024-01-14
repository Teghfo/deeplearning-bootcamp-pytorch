import math


class Circle:

    count: int = 0  # class attribute

    def __init__(self, radius) -> None:
        self.radius = radius  # instance attribute
        Circle.count += 1

    def area(self):  # instance method
        return self.radius * self.radius * math.pi

    def perimeter(self):
        return 2 * self.radius * math.pi


circle_obj1 = Circle(10)
circle_obj1.area()

circle_obj2 = Circle(15)
circle_obj2.area()

print(circle_obj1.count)
print(circle_obj2.count)
