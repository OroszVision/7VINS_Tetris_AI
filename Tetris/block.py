from colors import Colors
from position import Position
import pygame

class Block:
    def __init__(self, id) -> None:
        self.id = id
        self.cells = {}
        self.cell_size = 30
        self.rotation_state = 0
        self.colors = Colors.get_cell_colors()
        self.row_offset = 0
        self.column_offset = 0

    def move(self,rows,cols):
        self.row_offset += rows
        self.column_offset += cols

    def get_cell_positions(self):
        tiles = self.cells[self.rotation_state]
        moved_tiles = []
        for position in tiles:
            position = Position(position.row + self.row_offset,
                                position.column + self.column_offset)
            moved_tiles.append(position)
        return moved_tiles

    def draw(self,screen, offet_x,offet_y):
        tiles = self.get_cell_positions()
        for tile in tiles:
            tile_rect = pygame.Rect(offet_x + tile.column * self.cell_size,
                                    offet_y + tile.row * self.cell_size,
                                    self.cell_size - 1,
                                    self.cell_size - 1)
            pygame.draw.rect(screen, self.colors[self.id],tile_rect)

    def rotate(self):
        self.rotation_state += 1
        if (self.rotation_state) == len(self.cells):
            self.rotation_state = 0

    def undo_rotate(self):
        self.rotation_state -= 1
        if self.rotation_state == 0:
            self.rotation_state = len(self.cells - 1)