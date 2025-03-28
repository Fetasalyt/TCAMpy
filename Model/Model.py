import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import skew, kurtosis

class Model_3D:
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
        # Create initial field
        slen = self.side_length
        self.field = np.zeros((slen, slen, slen))

        # Add STC to the middle
        self.add_cell(slen//2, slen//2, slen//2, self.pmax+1)

    def add_cell(self, x, y, z, value):
        """
        Adds (or removes) cells at the given coordinates
        Create initial state before calling this function

        Parameters:
            x, y, z (int): representing the cell coordinates
            value (int): the new value at the given position
        """
        self.field[x][y][z] = value

    def plot_state(self):
        # Get indices where values are greater than 0
        x, y, z = np.where(self.field > 0)
        values = self.field[x, y, z]
        
        # Plot the current state of growth
        fig = plt.figure(figsize=(18, 5))

        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        axs = [ax1, ax2, ax3]
        field_for_histogram = self.field[self.field > 0]

        axs[0].scatter(x, y, z, c = values, vmin=1, vmax=self.pmax+1)
        axs[1].plot(self.stc_number, 'C1', label='STC')
        axs[1].plot(self.rtc_number, 'C2', label='RTC')
        axs[2].hist(field_for_histogram.ravel(), edgecolor='black')

        # Color bar
        fig.colorbar(axs[0].scatter(x, y, z, c = values, vmin=1, vmax=self.pmax+1), ax=axs[0], shrink=0.6)

        # Titles/labels of the plots
        titles = [str(self.cycles)+ " hour cell growth", "Tumor cell count", "Proliferation potential destribution"]
        labs_x = [str(self.side_length*10) + " micrometers", "Time (hours)", "Proliferation potential values"]
        labs_y = [str(self.side_length*10) + " micrometers", "Cell numbers", "Number of appearance"]

        axs[0].set_zlabel(str(self.side_length*10) + " micrometers")
        axs[2].set_xticks(range(1, self.pmax + 1))

        for i, ax in enumerate(axs):
            ax.set_title(titles[i])
            ax.set_xlabel(labs_x[i])
            ax.set_ylabel(labs_y[i])
            
        plt.tight_layout()

    def find_tumor_cells(self):
        # Find the tumor cells
        coords = np.nonzero(self.field)
        coords = np.transpose(coords)
        
        # Shuffle to randomize direction
        np.random.shuffle(coords)
        self.tumor_cells = coords

    def count_tumor_cells(self):
        # Current number of tumor cells
        stc_count = np.count_nonzero(self.field == self.pmax+1)
        rtc_count = len(self.tumor_cells) - stc_count
        
        # Save the current number
        self.stc_number.append(stc_count)
        self.rtc_number.append(rtc_count)

    def get_free_neighbours(self, x, y, z):
        """
        Get the neighboring cells of a given cell on the field.

        Parameters:
            x, y, z (int): representing the cell coordinates
        """
    
        directions = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
            (-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0),
            (-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1),
            (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
            (-1,-1, -1), (-1,-1, 1), (-1,1, -1), (-1,1, 1),
            (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]
    
        free_neighbours = []
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 < nx < self.side_length-1 and 0 < ny < self.side_length-1 and 0 < nz < self.side_length-1:
                if self.field[nx][ny][nz] == 0: free_neighbours.append([nx, ny, nz])
        return free_neighbours
    
    def __apop_true__(self, x, y, z):
        # Apoptosis
        if self.field[x][y][z] == self.pmax+1:
            return False
        else:
            return self.PA >= random.randint(1,100)

    def __action_true__(self, x, y, z, ch):
        # Migration or proliferation
        free_nb = self.get_free_neighbours(x, y, z)
        if len(free_nb) == 0:
            return False
        else:
            return ch >= random.randint(1,100)

    def cell_step(self, x, y, z, step_type):
        """
        The function that makes a single cell do one of the following actions:
        prolif STC - STC, prolif STC - RTC, prolif RTC - RTC, migration (1-4)

        Parameters:
            x (int): x coordinate of the field
            y (int): y coordinate of the field
            step_type (int): type of division or migration (1-4)
        """
        
        # Choose random target position
        free_nb = self.get_free_neighbours(x, y, z)
        target = free_nb[random.randint(1,len(free_nb)) - 1]
        
        match step_type:
            case 1:
                # Proliferation STC -> STC + STC
                self.field[target[0]][target[1]][target[2]] = self.pmax+1
            case 2:
                # Proliferation STC -> STC + RTC
                self.field[target[0]][target[1]][target[2]] = self.pmax
            case 3:
                # Proliferation RTC -> RTC + RTC
                self.field[x][y][z] -= 1
                self.field[target[0]][target[1]][target[2]] = self.field[x][y][z]
            case 4:
                # Migration
                self.field[target[0]][target[1]][target[2]] = self.field[x][y][z]
                self.field[x][y][z] = 0
                
    def cell_action(self):
        """
        This is the function that decides what action a cell will do
        After the decision, it kills the cell or calls the 'cell_step' function
        
        This function goes through every single cell in the field

        """
        for cell in self.tumor_cells:
            # Apoptosis
            if self.__apop_true__(cell[0], cell[1], cell[2]):
                self.field[cell[0]][cell[1]][cell[2]] = 0
            # Proliferation
            elif self.__action_true__(cell[0], cell[1], cell[2], self.PP):
                # If STC
                if self.field[cell[0]][cell[1]][cell[2]] == self.pmax+1:
                    if self.PS >= random.randint(1,100):
                        self.cell_step(cell[0], cell[1], cell[2], 1)
                    else:
                        self.cell_step(cell[0], cell[1], cell[2], 2)
                # If RTC
                else:
                    self.cell_step(cell[0], cell[1], cell[2], 3)
            # Migration
            elif self.__action_true__(cell[0], cell[1], cell[2], self.PM):
                self.cell_step(cell[0], cell[1], cell[2], 4)
    
    def save_to_img(self):
        # Save current state of field to image
        x, y, z = np.where(self.field > 0)
        values = self.field[x, y, z]
    
        growth = self.ax.scatter(x, y, z, c=values, cmap='viridis', alpha=0.7, animated=True)
        self.images.append([growth])
    
    def animate_growth(self):
        # Animate the progress of growth
        return animation.ArtistAnimation(self.fig, self.images, interval=50, blit=True, repeat_delay=100)
    
    def get_statistics(self):
        """
        Returns various statistical properties of the field.
        """
        
        # Only condsider cells for statistics
        nonzero_field = self.field[self.field > 0]

        texts = ["Minimum Proliferation Potential: ","Maximum Proliferation Potential: ",
                 "Mean Proliferation Potential: ", "Standard Deviation: ", "Variance: ",
                 "Median Proliferation Potential: ", "Skewness: ", "Kurtosis: ", "\nValues: ",
                 ]
        
        value = [nonzero_field.min(), nonzero_field.max(), nonzero_field.mean(),
                 nonzero_field.std(), nonzero_field.var(), np.median(nonzero_field),
                 skew(nonzero_field.ravel()), kurtosis(nonzero_field.ravel()), "",
                 ]
        
        # Value Distribution
        unique, counts = np.unique(nonzero_field, return_counts=True)
        for val, count in zip(unique, counts):
            texts.append(str(val) + ": ")
            value.append(count)
        
        texts.append("\nPercentiles: ")
        value.append("")
        
        # Percentiles
        percentiles = [0, 25, 50, 75, 100]  # Min, Q1, Median, Q3, Max
        percentile_values = np.percentile(nonzero_field, percentiles)
        for p, v in zip(percentiles, percentile_values):
            texts.append(str(p) + ": ")
            value.append(v)

        return texts, value
    
    def print_statistics(self):
        text, value = self.get_statistics()
        for i in range(len(text)):
            print(text[i] + str(value[i]))
    
    def save_stats_to_excel(self):
        # Saves the statistics to an excel file
        text, value = self.get_statistics()
        df = pd.DataFrame([text, value])
        df.to_excel('Model statistics.xlsx', index=False)
    
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
    def run_model(self, animated, stats):
        """
        The function that runs the entire model

        Parameters:
            animated (bool): set to true for animation, false for static plot
            histogram (bool): set to true to see histogram of proliferation potentials
            stats (bool): set to true to print statistics of the field

        For animation change matplotlib backend from inline to auto
        """

        # Create initial state
        if len(self.field) == 0: self.init_state()
        self.find_tumor_cells()
        if animated:
            x, y, z = np.where(self.field > 0)
            values = self.field[x, y, z]

            self.fig = plt.figure(figsize=(8, 8))
            self.ax = self.fig.add_subplot(projection='3d')
            self.ax.scatter(x, y, z, c = values, vmin=1, vmax=self.pmax+1)
            self.ax.set_title(str(self.cycles)+ " hour cell growth")
            self.ax.set_xlabel(str(self.side_length*10) + " micrometers")
            self.ax.set_ylabel(str(self.side_length*10) + " micrometers")
            self.ax.set_zlabel(str(self.side_length*10) + " micrometers")

        # Growth loop
        for c in range(self.cycles):
            self.cell_action()
            self.find_tumor_cells()
            self.count_tumor_cells()
            if animated: self.save_to_img()

        # Output settings
        self.plot_state()
        if animated: self.ani = self.animate_growth()
        if stats: self.print_statistics()

M = Model_3D(50, 500, 10, 1, 24, 1/24, 15, 4)
M.run_model(animated = False, stats = True)
