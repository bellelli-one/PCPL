from lab_python_oop.rectangle import Rectangle

class Square(Rectangle):

    figType = "Квадрат"

    def getname(self):
        return self.figType
    
    def __init__(self, side, color):
        self.side = side
        super().__init__(self.side, self.side, color)

    def square(self):
        return super().square()
    
    def repr(self):
        return f"{self.figType} со стороной {self.side},"\
               f"цвет - {self.color.figcolor},"\
               f"площадь - {self.square()}"
