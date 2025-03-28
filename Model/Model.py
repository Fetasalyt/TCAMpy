import time
import random
import imageio
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import skew, kurtosis

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
        self.CCT = CCT
        self.Dt = Dt
        self.mu = mu
        
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
        fig, axs = plt.subplots(1, 3, figsize=(17,4))
        field_for_histogram = self.field[self.field > 0]

        axs[0].imshow(self.field)
        axs[1].plot(self.stc_number, 'C1', label='STC')
        axs[1].plot(self.rtc_number, 'C2', label='RTC')
        axs[2].hist(field_for_histogram.ravel(), edgecolor='black')

        # Titles/labels of the plots
        titles = [str(self.cycles)+ " hour cell growth", "Tumor cell count", "Proliferation potential destribution"]
        labs_x = [str(self.side_length*10) + " micrometers", "Time (hours)", "Proliferation potential values"]
        labs_y = [str(self.side_length*10) + " micrometers", "Cell numbers", "Number of appearance"]

        axs[2].set_xticks(range(1, self.pmax + 1))

        for i, ax in enumerate(axs):
            ax.set_title(titles[i])
            ax.set_xlabel(labs_x[i])
            ax.set_ylabel(labs_y[i])

        # Color bar and legend
        fig.colorbar(axs[0].imshow(self.field))
        axs[1].legend()
    
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
        growth = self.ax.imshow(self.field, animated=True)
        self.images.append([growth])

    def animate_growth(self):
        # Animate the progress of growth
        return animation.ArtistAnimation(self.fig, self.images, interval=50, blit=True, repeat_delay=100)

    def save_field_to_excel(self):
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
        self.field[y][x] = value

    def get_statistics(self):
        """
        Returns various statistical properties of the field.
        """
        
        # Only condsider cells for statistics
        nonzero_field = self.field[self.field > 0]

        texts = ["Minimum Proliferation Potential: ","Maximum Proliferation Potential: ",
                 "Mean Proliferation Potential: ", "Standard Deviation: ", "Variance: ",
                 "Median Proliferation Potential: ", "Skewness: ", "Kurtosis: ", "\nValues: ",]
        
        value = [nonzero_field.min(), nonzero_field.max(), nonzero_field.mean(),
                 nonzero_field.std(), nonzero_field.var(), np.median(nonzero_field),
                 skew(nonzero_field.ravel()), kurtosis(nonzero_field.ravel()), "",]
        
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
            stats (bool): set to true to print statistics of the field

        For animation: matplotlib backend cannot be inline
        """

        # Create initial state
        if len(self.field) == 0: self.init_state()
        self.find_tumor_cells()
        
        if animated:
            self.fig, self.ax = plt.subplots()
            self.ax.imshow(self.field)
            self.ax.set_title(str(self.cycles)+ " hour cell growth")
            self.ax.set_xlabel(str(self.side_length*10) + " micrometers")
            self.ax.set_ylabel(str(self.side_length*10) + " micrometers")

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

    def create_dashboard(self):
        """
        Creates a graphical user interface. The GUI is a streamlit dashboard
        
        On the GUI you can change the value of parameters, modify initial state,
        activate animation, run and plot the model, and save the results to a file
        """
        
        st.title("Tumor Growth Cellular Automata Model")
        
        # User inputs for model parameters
        self.side_length = st.slider("Side Length (10um)", min_value=10, max_value=200, value=self.side_length)
        self.cycles = st.slider("Duration (hours)", min_value=50, max_value=1000, value=self.cycles)
        self.pmax = st.slider("Max Proliferation Potential", min_value=1, max_value=20, value=self.pmax)
        self.PA = st.slider("Apoptosis Chance (%)", min_value=0, max_value=100, value=self.PA)
        self.CCT = st.slider("Cell Cycle Time (hours)", min_value=1, max_value=48, value=self.CCT)
        self.Dt = st.slider("Time Step (days)", min_value=0.01, max_value=1.0, value=self.Dt, step=0.01)
        self.PS = st.slider("STC-STC Division Chance (%)", min_value=0, max_value=100, value=self.PS)
        self.mu = st.slider("Migration Capacity", min_value=0, max_value=10, value=self.mu)
        
        # Checkbox for animation and init state
        animated = st.checkbox("Enable growth animation")
        
        # Ensure field is stored persistently
        if "field" not in st.session_state:
            self.init_state()
            st.session_state.field = self.field.copy()
        else:
            self.field = st.session_state.field.copy()
        
        # Section to add cells manually
        st.subheader("Modify Initial State")
        x_coord = st.number_input("X Coordinate", min_value=0, max_value=self.side_length-1, value=self.side_length//2)
        y_coord = st.number_input("Y Coordinate", min_value=0, max_value=self.side_length-1, value=self.side_length//2)
        cell_value = st.number_input("Cell Value", min_value=0, max_value=self.pmax+1, value=self.pmax+1)
        
        if st.button("Add Cell"):
            self.add_cell(x_coord, y_coord, cell_value)
            st.session_state.field = self.field.copy()
            st.success(f"Cell added at ({x_coord}, {y_coord}) with value {cell_value}")
            
        # Display updated field
        fig, ax = plt.subplots()
        ax.imshow(self.field, cmap='viridis')
        ax.set_title("Updated Initial State")
        st.pyplot(fig)
        
        # Run Model Button
        if st.button("Run Model"):
            self.PP = int(self.CCT*self.Dt/24*100)
            self.PM = 100*self.mu/24
            M.run_model(animated = animated, histogram = False, stats = False)
            
            # Display results
            fig, axs = plt.subplots(1, 2, figsize=(11, 4))
            axs[0].imshow(self.field)
            axs[1].plot(self.stc_number, 'C1', label='STC')
            axs[1].plot(self.rtc_number, 'C2', label='RTC')
            axs[1].legend()
            st.pyplot(fig)
            
            # Animation
            if animated:
                st.write("Generating animation...")
                
                frames = []
                for img in self.images:
                    fig, ax = plt.subplots()
                    ax.imshow(img[0].get_array(), cmap='viridis')
                    
                    # Convert plot to image
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    fig.savefig(temp_file.name, format='png', bbox_inches='tight')
                    plt.close(fig)
                    frames.append(imageio.imread(temp_file.name))
                    
                # Save frames as a GIF
                with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmpfile:
                    imageio.mimsave(tmpfile.name, frames, duration=0.1)
                    st.image(tmpfile.name, caption="Tumor Growth Animation", use_container_width=True)
        
            # Histogram of proliferation potential values
            fig, ax = plt.subplots()
            field_for_histogram = self.field[self.field > 0]
            ax.hist(field_for_histogram.ravel(), edgecolor='black')
            
            # Titles/labels of the plots
            ax.set_title("Proliferation potential destribution")
            ax.set_xlabel("Proliferation potential values")
            ax.set_ylabel("Number of appearance")
            ax.set_xticks(range(1, self.pmax + 1))
            st.pyplot(fig)
    
            # Statistics of the model
            text, value = self.get_statistics()
            for i in range(len(text)):
                st.write(text[i] + str(value[i]))

# Example usage
M = Model(75, 500, 10, 1, 24, 1/24, 15, 4)
M.run_model(animated = False, stats = True)

# M.create_dashboard()
