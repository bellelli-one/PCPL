class Color:
    def __init__(self):
        self.color = None

    @property
    def figcolor(self):
        return self.color
    
    @figcolor.setter
    def figcolor(self, color):
        self.color = color  