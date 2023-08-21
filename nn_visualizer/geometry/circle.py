class Circle():
    def __init__(self, x: float = 0, y: float = 0, radius: float = 0):
        self.x = x
        self.y = y
        self.radius = radius

    def center(self):
        return (self.x, self.y)
