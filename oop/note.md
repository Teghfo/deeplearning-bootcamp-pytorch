# OOP in python

An object is any entity that has attributes and behaviors. For example, a car is an object. It has:

* attributes - model, year, color, etc.
* behaviors - accelerating, braking, etc. 

A class is a blueprint for that object.

```python
class Car:
    def __init__(self):
        pass

    def accelerating(self):
        pass
```


# Inheritance
Inheritance allows a class to inherit attributes and methods from another class. Inheritance is a way of creating a new class for using details of an existing class without modifying it.

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

my_dog = Dog("Buddy")
my_cat = Cat("Whiskers")

print(my_dog.speak())  # Output: Buddy says Woof!
print(my_cat.speak())  # Output: Whiskers says Meow!

```

# Polymorphism
Polymorphism allows objects of different classes to be treated as objects of a common base class.

```python
class Animal:
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        return "Woof!"

class Cat(Animal):
    def make_sound(self):
        return "Meow!"

def animal_sound(animal: Animal):
    return animal.make_sound()

my_dog = Dog()
my_cat = Cat()

print(animal_sound(my_dog))  # Output: Woof!
print(animal_sound(my_cat))  # Output: Meow!
```

# Abstraction
Abstraction allows you to hide the complex implementation details and show only the necessary features of an object. Abstraction is a fundamental concept in object-oriented programming that involves simplifying complex systems by modeling classes based on the essential properties and behaviors they exhibit, while hiding unnecessary details. In simpler terms, abstraction allows you to focus on what an object does rather than how it achieves its functionality. 

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius * self.radius

class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side * self.side

my_circle = Circle(5)
my_square = Square(4)

print(my_circle.area())  # Output: 78.5
print(my_square.area())  # Output: 16
```

# Encapsulation
Encapsulation involves bundling the data (attributes) and the methods that operate on the data into a single unit (a class). The primary purpose of encapsulation is to restrict access to the internal details of an object and to provide a controlled interface for interacting with it.

```python
class Car:
    def __init__(self, make, model):
        self._make = make  # Protected attribute
        self._model = model  # Protected attribute
        self.__fuel = 100  # Private attribute

    def drive(self):
        self.__fuel -= 10
        return f"{self._make} {self._model} is driving."

    def get_fuel_level(self):
        return self.__fuel

my_car = Car("Toyota", "Camry")

print(my_car.drive())  # Output: Toyota Camry is driving.
print(my_car.get_fuel_level())  # Output: 90
```

# Dunder methods
Dunder" is a colloquial term used in the Python community to refer to special methods that have double underscores on both sides of their name, hence the term "dunder" (double underscore). These methods are also known as magic methods or special methods. They play a crucial role in defining how objects behave in various situations and enable customization of the behavior of classes.

Here are some important dunder methods and their significance:

1. `__init__`:

   * Purpose: Constructor method.
   * Usage: Called when an object is created. Used to initialize the attributes of an object.

    ```python
    class MyClass:
        def __init__(self, value):
            self.value = value

    ```

2. `__str__` and `__repr__`

    * Purpose: String representation of the object.
    * Usage: `__str__` is used by the str() function and print() function. `__repr__` is used by the repr() function and is meant to provide an unambiguous representation.
    ```python
    class MyClass:
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return f"MyClass instance with value: {self.value}"

        def __repr__(self):
            return f"MyClass({self.value})"
    ```

3. `__len__`
   * Purpose: Length of the object.
   * Usage: Used by the len() function.

    ```python
    class MyList:
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)
    ```
4. `__getitem__` and `__setitem__`
   * Purpose: Accessing and setting values using square bracket notation.
   * Usage: Used for index-based access.
    ```python
    class MyList:
    def __init__(self, items):
        self.items = items

    def __getitem__(self, index):
        return self.items[index]

    def __setitem__(self, index, value):
        self.items[index] = value
    ```
5. `__call__`
    * Purpose: Make an instance of a class callable.
    * Usage: Allows an object to be called as if it were a function.
    ```python
    class CallableClass:
    def __call__(self, *args, **kwargs):
        return f"Called with arguments: {args}, {kwargs}"

    obj = CallableClass()
    result = obj(1, 2, keyword='value')
    ```