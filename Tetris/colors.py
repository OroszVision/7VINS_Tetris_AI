class Colors:
    dark_grey = (26,31,40)
    red = (255,0,0)
    green = (0,255,0)
    blue = (0,0,255)
    cyan = (0,255,255)
    yellow = (255,255,0)
    orange = (255,165,0)
    purple = (128,0,128)
    white = (255,255,255)
    dark_blue = (44,44,127)
    light_blue = (59,85,162) 

    @classmethod
    def get_cell_colors(cls):
        return [cls.dark_grey,cls.green,cls.red,
                cls.orange,cls.yellow,cls.purple,
                cls.cyan,cls.blue]
