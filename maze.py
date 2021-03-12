import pygame
import math

WIDTH = 1000
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("ML Maze")


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

maze = [[0,1,0,1,0,1,0,1,0,1],[0,1,0,1,0,1,0,1,0,1],[0,1,0,1,0,1,0,1,0,1],[0,1,0,1,0,1,0,1,0,1],[0,1,0,1,0,1,0,1,0,1]]

class Spot:
	def __init__(self, row, col, width, total_rows):
		self.row = row
		self.col = col
		self.x = row * width
		self.y = col * width
		self.color = WHITE
		self.neighbors = []
		self.width = width
		self.total_rows = total_rows

	def get_pos(self):
		return self.row, self.col


	def is_barrier(self):
		return self.color == BLACK


	def open(self):
		self.color = WHITE


	def make_barrier(self):
		self.color = BLACK



	def draw(self, win):
		pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

	def update_neighbors(self, grid):
		self.neighbors = []
		if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # DOWN
			self.neighbors.append(grid[self.row + 1][self.col])

		if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # UP
			self.neighbors.append(grid[self.row - 1][self.col])

		if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # RIGHT
			self.neighbors.append(grid[self.row][self.col + 1])

		if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # LEFT
			self.neighbors.append(grid[self.row][self.col - 1])

		if self.col > 0 and self.row >0 and not grid[self.row-1][self.col - 1].is_barrier(): # TopLeft
			self.neighbors.append(grid[self.row-1][self.col - 1])

		if self.col > 0 and self.row>0 and not grid[self.row+1][self.col +1].is_barrier(): # TopRight
			self.neighbors.append(grid[self.row+1][self.col + 1])
			
		if self.col > 0 and self.row>0 and not grid[self.row-1][self.col +1].is_barrier(): # BottomRight
			self.neighbors.append(grid[self.row-1][self.col + 1])

		if self.col > 0 and self.row>0 and not grid[self.row + 1][self.col -1].is_barrier(): # BottomLeft
			self.neighbors.append(grid[self.row+1][self.col - 1])


	def __lt__(self, other):
		return False










def make_grid(rows, width):
	grid = []
	gap = width // rows
	for i in range(rows):
		grid.append([])
		for j in range(rows):
			spot = Spot(i, j, gap, rows)
			grid[i].append(spot)

	return grid




def draw_grid(win, rows, width):
	gap = width // rows
	for i in range(rows):
		pygame.draw.line(win, BLACK, (0, i * gap), (width, i * gap))
		for j in range(rows):
			pygame.draw.line(win, BLACK, (j * gap, 0), (j * gap, width))


def draw(win, grid, rows, width):
	win.fill(WHITE)

	for row in grid:
		for spot in row:
			spot.draw(win)

	draw_grid(win, rows, width)
	pygame.display.update()





def main(win, width, maze):
	ROWS = 50
	grid = make_grid(ROWS, width)

	start = None
	end = None

	run = True
	while run:
		draw(win, grid, ROWS, width)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False

			for row in range(len(maze)):
				for col in range(len(maze[row])):
					if(maze[row][col] == 1):
						spot = grid[row][col]
						spot.make_barrier()
						#spot.update_neighbors(grid)



	pygame.quit()



main(WIN, WIDTH, maze)