from lab_python_oop.rectangle import Rectangle
from lab_python_oop.circle import Circle
from lab_python_oop.square import Square

def main():
    rec = Rectangle(1, 2, "синий")
    cir = Circle(5, "зеленый")
    sq = Square(4, "красный")
    print(rec.repr())
    print(cir.repr())
    print(sq.repr())

if __name__ == "__main__":
    main()