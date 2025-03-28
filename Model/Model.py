import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Model:
    """
    Class for a cellular automata, modeling tumor growth

    Parameters:
        side_length (int): side length of the field (10um)
        cycles (int): duration of the model given in hours
        pmax (int): maximum proliferation potential of RTC
        PA (int): chance for apoptosis of RTC (in percent)
        CCT (int): cell cycle time of cells given in hours
        Dt (float): time step of the model given in days
        PS (int): STC-STC division chance (in percent)
        mu (int): migration capacity of the cells
    """
    
    def __init__(self, side_length, cycles, pmax, PA, CCT, Dt, PS, mu):
        # Parameters
        self.side_length = side_length
        self.cycles = cycles
        self.pmax = pmax
        
        # containers
        self.stc_number = []
        self.rtc_number = []
        self.images = []
        self.field = []
        
        # Chances
        self.PP = int(CCT*Dt/24*100)
        self.PM = 100*mu/24
        self.PA = PA
        self.PS = PS
    
    def init_state(self):
        # Create the field
        self.field = np.zeros((self.side_length, self.side_length))

        # Add an STC to the middle
        self.add_cell(self.side_length//2, self.side_length//2, self.pmax+1)
        
    def plot_state(self):
        # Plot the current state of growth
        self.fig, self.axs = plt.subplots(1, 2, figsize=(11,4))

        self.axs[0].imshow(self.field)
        self.axs[1].plot(self.stc_number, 'C1', label='STC')
        self.axs[1].plot(self.rtc_number, 'C2', label='RTC')

        # Titles/labels of the plots
        titles = [str(self.cycles)+ " hour cell growth", "Tumor cell count"]
        labs_x = [str(self.side_length*10) + " micrometers", "Time (hours)"]
        labs_y = [str(self.side_length*10) + " micrometers", "Cell numbers"]

        for i, ax in enumerate(self.axs):
            ax.set_title(titles[i])
            ax.set_xlabel(labs_x[i])
            ax.set_ylabel(labs_y[i])

        # Color bar and legend
        self.fig.colorbar(self.axs[0].imshow(self.field))
        self.axs[1].legend()
    
    def find_tumor_cells(self):
        # Where are tumor cells?
        coords = np.nonzero(self.field)
        coords = np.transpose(coords)
        
        # Shuffle to randomize direction
        np.random.shuffle(coords)
        self.tumorcells = coords

    def count_tumor_cells(self):
        # Count RTC and STC
        stc_count = np.count_nonzero(self.field == self.pmax + 1)
        rtc_count = len(self.tumorcells) - stc_count
        
        # Save the current number
        self.stc_number.append(stc_count)
        self.rtc_number.append(rtc_count)

    def get_free_neighbours(self, x, y):
        """
        Get the neighboring coordinates of a given cell in a 2D NumPy matrix.

        Parameters:
            x, y (int): representing the coordinates of the cell.
        """
    
        directions = [
            (-1, 0), (1, 0),
            (0, -1), (0, 1),
            (-1,-1), (-1,1),
            (1, -1), (1, 1)]
    
        neighbours = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < self.side_length-1 and 0 < ny < self.side_length-1:
                if self.field[nx][ny] == 0: neighbours.append([nx, ny])
        return neighbours
    
    def __apop_true__(self, x, y):
        # Apoptosis
        if self.field[x][y] == self.pmax+1:
            return False
        else:
            return self.PA >= random.randint(1,100)

    def __action_true__(self, x, y, ch):
        # Migration or proliferation
        free_nb = self.get_free_neighbours(x, y)
        if len(free_nb) == 0:
            return False
        else:
            return ch >= random.randint(1,100)
    
    def cell_step(self, x, y, step_type):
        """
        The function that makes a single cell do one of the following actions:
        prolif STC - STC, prolif STC - RTC, prolif RTC - RTC, migration (1-4)

        Parameters:
            x (int): x coordinate of the field
            y (int): y coordinate of the field
            step_type (int): type of division or migration (1-4)
        """
        
        # Choose random target position
        free_nb = self.get_free_neighbours(x, y)
        target = free_nb[random.randint(1,len(free_nb)) - 1]
        
        match step_type:
            case 1:
                # Proliferation STC -> STC + STC
                self.field[target[0]][target[1]] = self.pmax+1
            case 2:
                # Proliferation STC -> STC + RTC
                self.field[target[0]][target[1]] = self.pmax
            case 3:
                # Proliferation RTC -> RTC + RTC
                self.field[x][y] -= 1
                self.field[target[0]][target[1]] = self.field[x][y]
            case 4:
                # Migration
                self.field[target[0]][target[1]] = self.field[x][y]
                self.field[x][y] = 0
        
    def cell_action(self):
        """
        This is the function that decides what action a cell will do
        After the decision, it kills the cell or calls the 'cell_step' function
        
        This function goes through every single cell in the field

        """
        for cell in self.tumorcells:
            # Apoptosis
            if self.__apop_true__(cell[0], cell[1]):
                self.field[cell[0]][cell[1]] = 0
            # Proliferation
            elif self.__action_true__(cell[0], cell[1], self.PP):
                # If STC
                if self.field[cell[0]][cell[1]] == self.pmax+1:
                    if self.PS >= random.randint(1,100):
                        self.cell_step(cell[0], cell[1], 1)
                    else:
                        self.cell_step(cell[0], cell[1], 2)
                # If RTC
                else:
                    self.cell_step(cell[0], cell[1], 3)
            # Migration
            elif self.__action_true__(cell[0], cell[1], self.PM):
                self.cell_step(cell[0], cell[1], 4)
        
    def save_to_img(self):
        # Save current state of field to image
        growth = self.axs[0].imshow(self.field, animated=True)
        cell_count_stc, = self.axs[1].plot(self.stc_number, 'C1', label='STC')
        cell_count_rtc, = self.axs[1].plot(self.rtc_number, 'C2', label='RTC')
        self.images.append([growth, cell_count_stc, cell_count_rtc])

    def animate_growth(self):
        # Animate the progress of growth
        return animation.ArtistAnimation(self.fig, self.images, interval=50, blit=True, repeat_delay=100)

    def save_to_excel(self):
        # Save current state of field to file
        df = pd.DataFrame(self.field)
        df.to_excel('Tumor growth.xlsx', index=False)

    def add_cell(self, x, y, value):
        """
        The function that adds (or removes) cells at coordinates
        Create init state before calling this function

        Parameters:
            x (int): x coordinate of the field
            y (int): y coordinate of the field
            value (int): the new value at the given position
        """
        self.field[x][y] = value

    def measure_runtime(func):
        # Decorator to measure completion time
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            runtime = end_time - start_time
            print("Model completion time (s): " + str(runtime))
            return result
        return wrapper

    @measure_runtime
    def run_model(self, unique_init_state, animated, save_to_excel):
        """
        The function that runs the entire model

        Parameters:
            animated (bool): set to true for animation, false for static plot
            save_to_excel (bool): set to true for saving results to excel file
            unique_init_state (bool): set to true if you don't want new init state

        For animation change matplotlib backend from inline to auto
        """

        # Create initial state
        if not unique_init_state: self.init_state()
        self.find_tumor_cells()
        if animated: self.plot_state()

        # Growth loop
        for c in range(self.cycles):
            self.cell_action()
            self.find_tumor_cells()
            self.count_tumor_cells()
            if animated: self.save_to_img()

        # Output settings
        if save_to_excel: self.save_to_excel()
        if animated: self.ani = self.animate_growth()
        else:
            self.plot_state()

# Create Model class
M = Model(75, 500, 10, 1, 24, 1/24, 15, 4)
M.run_model(unique_init_state = False, animated = False, save_to_excel = False)
