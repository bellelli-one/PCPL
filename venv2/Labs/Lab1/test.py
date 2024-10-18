import unittest
from lab_python_oop.rectangle import Rectangle
from lab_python_oop.circle import Circle
from lab_python_oop.square import Square

class Test(unittest.TestCase):
    def setUp(self):
        self.rec = Rectangle(1, 2, "синий")
        self.cir = Circle(5, "зеленый")
        self.sq = Square(4, "красный")
    
    def test_square(self):
        self.assertEqual(self.rec.square(), 2)
        self.assertEqual(self.cir.square(), 78.54)
        self.assertEqual(self.sq.square(), 16)

if __name__ == "__main__":
    unittest.main()
    