import cv2
import numpy as np
import math
import heapq
img =cv2.imread("image.png") #loads image
edges = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts image to gray
edges=cv2.Canny(img,100,150) #converts regular image to edges image


edges = cv2.dilate(edges, np.ones((3,3), dtype=np.int8))  # this is to make the edges thicker
#in case it is required

start=-1 #init start and end values
end=-1


#A* path finding algo
# Define the Cell class


class Cell:
    def __init__(self):
      # Parent cell's row index
        self.parent_i = 0
    # Parent cell's column index
        self.parent_j = 0
 # Total cost of the cell (g + h)
        self.f = float('inf')
    # Cost from start to this cell
        self.g = float('inf')
    # Heuristic cost from this cell to destination
        self.h = 0


# Define the size of the grid
ROW = len(edges)-1
COL = len(edges[0])-1
print(len(edges),len(edges[0])) 
# Check if a cell is valid (within the grid)


def is_valid(row, col):
  return 0 <= row < ROW and 0 <= col < COL and edges[row][col] >= 0

    
        
# Check if a cell is unblocked


def is_unblocked(grid, row, col):
    return grid[row][col] == 0

# Check if a cell is the destination


def is_destination(row, col, dest):
    return row == dest[0] and col == dest[1]

# Calculate the heuristic value of a cell (Euclidean distance to destination)


def calculate_h_value(row, col, dest):
    return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5

# Trace the path from source to destination


def trace_path(cell_details, dest):
    print("The Path is ")
    path = []
    row = dest[0]
    col = dest[1]

    # Trace the path from destination to source using parent cells
    while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
        path.append((row, col))
        temp_row = cell_details[row][col].parent_i
        temp_col = cell_details[row][col].parent_j
        row = temp_row
        col = temp_col

    # Add the source cell to the path
    path.append((row, col))
    # Reverse the path to get the path from source to destination
    path.reverse()

    # Print the path
    for i in path:
        img[i[0]:i[0]+2,i[1]:i[1]+2]=(0,255,0)
        img[i[0]:i[0]-2,i[1]:i[1]-2]=(0,255,0)
        
    

# Implement the A* search algorithm


def a_star_search(grid, src, dest):
    
    # Check if the source and destination are valid
    if not is_valid(src[0], src[1]) or not is_valid(dest[0], dest[1]):
        print("Source or destination is invalid")
        return

    # Check if the source and destination are unblocked
    if not is_unblocked(grid, src[0], src[1]) or not is_unblocked(grid, dest[0], dest[1]):
        print("Source or the destination is blocked")
        return

    # Check if we are already at the destination
    if is_destination(src[0], src[1], dest):
        print("We are already at the destination")
        return

    # Initialize the closed list (visited cells)
    closed_list = [[False for _ in range(COL)] for _ in range(ROW)]
    # Initialize the details of each cell
    cell_details = [[Cell() for _ in range(COL)] for _ in range(ROW)]

    # Initialize the start cell details
    i = src[0]
    j = src[1]
    cell_details[i][j].f = 0
    cell_details[i][j].g = 0
    cell_details[i][j].h = 0
    cell_details[i][j].parent_i = i
    cell_details[i][j].parent_j = j

    # Initialize the open list (cells to be visited) with the start cell
    open_list = []
    heapq.heappush(open_list, (0.0, i, j))

    # Initialize the flag for whether destination is found
    found_dest = False

    # Main loop of A* search algorithm
    while len(open_list) > 0:
        # Pop the cell with the smallest f value from the open list
        p = heapq.heappop(open_list)

        # Mark the cell as visited
        i = p[1]
        j = p[2]
        closed_list[i][j] = True

        # For each direction, check the successors
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dir in directions:

            new_i = i + dir[0]
            new_j = j + dir[1]
            

            # If the successor is valid, unblocked, and not visited
            if is_valid(new_i, new_j) and is_unblocked(grid, new_i, new_j) and not closed_list[new_i][new_j]:
                # If the successor is the destination
                if is_destination(new_i, new_j, dest):
                    # Set the parent of the destination cell
                    cell_details[new_i][new_j].parent_i = i
                    cell_details[new_i][new_j].parent_j = j
                    print("The destination cell is found")
                    # Trace and print the path from source to destination
                    trace_path(cell_details, dest)
                    found_dest = True
                    return
                else:
                    # Calculate the new f, g, and h values
                    g_new = cell_details[i][j].g + 1.0
                    h_new = calculate_h_value(new_i, new_j, dest)
                    f_new = g_new + h_new

                    # If the cell is not in the open list or the new f value is smaller
                    if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                        # Add the cell to the open list
                        heapq.heappush(open_list, (f_new, new_i, new_j))
                        # Update the cell details
                        cell_details[new_i][new_j].f = f_new
                        cell_details[new_i][new_j].g = g_new
                        cell_details[new_i][new_j].h = h_new
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j

    # If the destination is not found after visiting all cells
    if not found_dest:
        print("Failed to find the destination cell")


def solve():
    a_star_search(edges,start,end) #call A* pass in the start and the end as well as the img array




def mouse_callback(event, x, y, flags, param): #function for clicking start and end position
    global start,end
    
    
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        print(f"Left button clicked at ({x}, {y})")
        if start==-1:
            start=y,x
        elif end==-1:
            end=y,x
            solve()
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right mouse button click
        print(f"Right button clicked at ({x}, {y})")
    elif event == cv2.EVENT_MBUTTONDOWN:  # Middle mouse button click
        print(f"Middle button clicked at ({x}, {y})")


# Create a named window
cv2.namedWindow("Image")








# Set the mouse callback function to capture mouse events
cv2.setMouseCallback("Image", mouse_callback) 

    



while True:
    cv2.imshow("Image", img)  #print the img we will work with
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()
