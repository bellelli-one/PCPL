from lab_python_oop.shape import Shape
from lab_python_oop.color import Color
import math

class Circle(Shape):

    figType = 'Круг'

    def getname(self):
        return self.figType

    def __init__(self, radius, color):
        self.radius = radius
        self.color = Color()
        self.color.figcolor = color

    def square(self):
        return round(math.pi * self.radius ** 2, 2)
    
    def repr(self):
        return f"Круг с радиусом {self.radius},"\
               f"цвет - {self.color.figcolor},"\
               f"площадь - {self.square()}"