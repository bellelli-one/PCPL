from lab_python_oop.shape import Shape
from lab_python_oop.color import Color

class Rectangle(Shape):

    figType = "Прямоугольник"

    def getname(self):
        return self.figType
    
    def __init__(self, width, height, color):
        self.width = width
        self.height = height
        self.color = Color()
        self.color.figcolor = color

    def square(self):
        return self.width * self.height
    
    def repr(self):
        return f"Прямоугольник с сторонами {self.width} и {self.height},"\
               f"цвет - {self.color.figcolor},"\
               f"площадь - {self.square()}"