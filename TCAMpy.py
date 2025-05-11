import io
import time
import random
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import skew, kurtosis

class TModel:
    """
    Class for a cellular automata, modeling tumor growth.

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
        """
        Creates the initial state with one STC in the middle.
        """

        self.field = np.zeros((self.side_length, self.side_length))
        self.mod_cell(self.side_length//2, self.side_length//2, self.pmax+1)
        
    def plot_state(self):
        """
        Plots the field and the growth with cell numbers.
        Creates a histogram of proliferation potentials.
        """
        
        # Create the figue and axis
        fig, axs  = plt.subplots(1, 3, figsize=(17,4))
        pp_values = self.get_prolif_potentials().values()

        axs[0].imshow(self.field)
        axs[1].plot(self.stc_number, 'C1', label='STC')
        axs[1].plot(self.rtc_number, 'C2', label='RTC')
        axs[2].bar(range(1, self.pmax + 2), list(pp_values), edgecolor='black')

        # Titles/labels of the plots
        titles = [str(self.cycles)+ " hour cell growth", "Tumor cell count", "Value destribution"]
        labs_x = [str(self.side_length*10) + " micrometers", "Time (hours)", "Proliferation potential"]
        labs_y = [str(self.side_length*10) + " micrometers", "Cell numbers", "Number of appearance"]

        for i, ax in enumerate(axs):
            ax.set_title(titles[i])
            ax.set_xlabel(labs_x[i])
            ax.set_ylabel(labs_y[i])

        # Color bar and legend
        fig.colorbar(axs[0].imshow(self.field))
        axs[1].legend()
    
    def find_tumor_cells(self):
        """
        Saves the coordinates of tumor cells to self.tumor_cells.
        """
     
        # Where are tumor cells?
        coords = np.nonzero(self.field)
        coords = np.transpose(coords)
        
        # Shuffle to randomize direction
        np.random.shuffle(coords)
        self.tumor_cells = coords

    def count_tumor_cells(self):
        """
        Saves the number of STCs/RTCs to self.stc_number/self.rtc_number.
        """
        
        # Count RTC and STC
        stc_count = np.count_nonzero(self.field == self.pmax + 1)
        rtc_count = len(self.tumor_cells) - stc_count
        
        # Save the current number
        self.stc_number.append(stc_count)
        self.rtc_number.append(rtc_count)

    def get_free_neighbours(self, x, y):
        """
        Returns the neighboring coordinates of a given cell in a 2D NumPy matrix.

        Parameters:
            x, y (int): representing the coordinates of the cell
            
        Returns:
            list: a list with the coords of the neighbouring cells
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
        if self.field[x][y]  == self.pmax+1: return False
        else: return self.PA >= random.randint(1,100)

    def __action_true__(self, x, y, ch):
        free_nb = self.get_free_neighbours(x, y)
        
        if len(free_nb) == 0: return False
        else: return ch >= random.randint(1,100)      
    
    def cell_step(self, x, y, step_type):
        """
        The function that makes a single cell do one of the following actions:
        prolif STC - STC, prolif STC - RTC, prolif RTC - RTC, migration (1-4).

        Parameters:
            x, y (int): representing the coordinates of the cell
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
        This is the function that decides what action a cell will do.
        Either kills the cell or calls the 'cell_step' function.
        This function goes through every single cell in the field.
        """
        
        for cell in self.tumor_cells:
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

    def animate_growth(self):
        """
        Creates and returns animation of the growth.
        
        Returns:
            ArtistAnimation: the animation of the growth
        """
        
        return animation.ArtistAnimation(self.fig, self.images, interval=50, blit=True)

    def save_field_to_excel(self, file_name):
        """
        Saves the current state of self.field to an excel file.
        
        Parameters:
            file_name (str): name of the excel file
        """

        pd.DataFrame(self.field).to_excel(file_name, index=False)

    def mod_cell(self, x, y, value):
        """
        Modifies cell value. (Create initial state before this!)

        Parameters:
            x, y (int): representing coordinates of the cell
            value (int): the new value at the given position
        """
        
        self.field[y][x] = value

    def get_prolif_potentials(self):
        """
        Returns a dictionary of proliferation potential numbers.
        
        Returns:
            dict: a dictionary of the proliferation potentials
        """
        
        nonzero_field  = np.array(self.field)[np.array(self.field) > 0]
        unique, counts = np.unique(nonzero_field, return_counts=True)
        prolif_potents = {}
        
        for i in range(1, self.pmax + 2):
            prolif_potents[i] = 0
            
        for val, count in zip(unique, counts):
            prolif_potents[int(val)] = count
            
        return prolif_potents

    def get_statistics(self):
        """
        Returns various statistical properties of the model.
        
        Returns:
            dict: a dictionary of the statistical properties
        """
        
        nonzero_field = self.field[self.field > 0]

        # Statistics
        stats = {
            "Min":       nonzero_field.min(),
            "Max":       nonzero_field.max(),
            "Mean":      nonzero_field.mean(),
            "Std":       nonzero_field.std(),
            "Var":       nonzero_field.var(),
            "Median":    np.median(nonzero_field),
            "Skew":      skew(nonzero_field.ravel()),
            "Kurtosis":  kurtosis(nonzero_field.ravel()),
            "Final STC": self.stc_number[self.cycles-1],
            "Final RTC": self.rtc_number[self.cycles-1],
        }
        
        # Proliferation potentials
        stats.update(self.get_prolif_potentials())
            
        # Cell Numbers
        checkpoints = np.linspace(0, self.cycles - 1, 11, dtype=int)
        for idx in checkpoints:
            hour = (idx + 1)
            stats[f"{hour}h_STC"] = self.stc_number[idx]
        for idx in checkpoints:
            hour = (idx + 1)
            stats[f"{hour}h_RTC"] = self.rtc_number[idx]
    
        return stats
    
    def save_statistics(self, file_name):
        """
        Saves various statistical properties of the model to an excel file.
        
        Parameters:
            file_name (str): name of the excel file
        """
        
        stats_dict = self.get_statistics()
        df = pd.DataFrame([stats_dict])
        df.to_excel(file_name, index=False)

    def measure_runtime(func):
        # Decorator to measure completion time
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result     = func(*args, **kwargs)
            end_time   = time.time()
            runtime    = end_time - start_time
            print("Model completion time (s): " + str(runtime))
            return result
        return wrapper

    @measure_runtime
    def run_model(self, plot, animate, stats):
        """
        The function that runs the entire model.
        For animation: matplotlib backend cannot be inline!

        Parameters:
            animate (bool): set to true for animation, false for static plot
            stats (bool): set to true to print statistics of the field
        """

        # Create initial state
        if len(self.field) == 0: self.init_state()
        self.find_tumor_cells()
        
        if animate:
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
            if animate:
                growth = self.ax.imshow(self.field, animated=True)
                self.images.append([growth])

        # Output settings
        if plot: self.plot_state()
        if animate: self.ani = self.animate_growth()
        if stats: print(self.get_statistics())
        
    @measure_runtime
    def run_multimodel(self, count, init_field):
        """
        Runs the model multiple times and returns a DataFrame of statistics.
    
        Parameters:
            count (int): number of times to run the model
            init_field (np.array): initial state of the field for each run
    
        Returns:
            pd.DataFrame: collected statistics from each run
        """
        
        stats = []
        
        for i in range(count):
            self.field = init_field
            self.stc_number = []
            self.rtc_number = []
            self.run_model(plot = False, animate = False, stats = False)
            stats.append(self.get_statistics())
        all_stats = pd.DataFrame(stats)
        
        return all_stats

    def plot_averages(self, data):
        """
        The function that plots the averages of multiple model results.
        Works with the results of the 'run_multimodel' function.
        
        Parameters:
            data (pd.DataFrame): Your data in a pandas dataframe format
            
        Returns:
            fig: The plots of the averages with SD values
        """
        
        stc_cols = sorted([col for col in data.columns if "_STC" in str(col)], key=lambda x: int(str(x).split("h")[0]))
        rtc_cols = sorted([col for col in data.columns if "_RTC" in str(col)], key=lambda x: int(str(x).split("h")[0]))
        pp_cols  = sorted([col for col in data.columns if isinstance(col, int)])
        
        avg_stc = data[stc_cols].mean()
        std_stc = data[stc_cols].std()
        avg_rtc = data[rtc_cols].mean()
        std_rtc = data[rtc_cols].std()
        avg_pp  = data[pp_cols].mean()
        std_pp  = data[pp_cols].std()
        
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(14, 5))
        timepoints      = np.linspace(0, self.cycles - 1, 11)
        
        ax1.plot(timepoints, avg_stc, label='STC', color='C1')
        ax1.fill_between(timepoints, avg_stc - std_stc, avg_stc + std_stc, color='C1', alpha=0.3)

        ax1.plot(timepoints, avg_rtc, label='RTC', color='C2')
        ax1.fill_between(timepoints, avg_rtc - std_rtc, avg_rtc + std_rtc, color='C2', alpha=0.3)
        
        ax1.set_title("Average Tumor Cell Count")
        ax1.set_xlabel("Model Time (hours)")
        ax1.set_ylabel("Number of Cells")
        ax1.legend()

        ax2.bar(pp_cols, avg_pp, yerr=std_pp, capsize=5, edgecolor='black')
        ax2.set_title("Average Proliferation Potential Distribution")
        ax2.set_xlabel("Proliferation Potential")
        ax2.set_ylabel("Average Count")
        
        plt.tight_layout()
        return fig


class TDashboard:
    """
    Class for a Streamlit dashboard providing a GUI for the model.
    
    Parameters:
        model (TModel): The created model you want a dashboard for
    """
    
    def __init__(self, model):
        self.model = model

    def run_dashboard(self):
        """
        The function that creates the entire streamlit dashboard for your model.
        """
        
        st.set_page_config(layout="wide")
        st.markdown("<h1 style='text-align: center;'>TCAMpy</h1>", unsafe_allow_html=True)
        self.col1, _, self.col3 = st.columns([4, 1, 12])

        with self.col1:
            self._initialize()
            self._modify_cell()
            self._execute_model()

        with self.col3:
            self._visualize_results()
            self._show_statistics()
            self._reset_save_stats()

    def _initialize(self):
        """
        The function that sets the parameters and initializes the model.
        """
        
        st.markdown("<h2 style='text-align: center;'>Model Parameters</h2>", unsafe_allow_html=True)

        self.model.side_length = st.slider("Side Length (10um)", 10, 200, value=self.model.side_length)
        self.model.cycles      = st.slider("Model Duration (hours)", 50, 1000, value=self.model.cycles)
        self.model.pmax        = st.slider("Max Proliferation Potential", 1, 20, value=self.model.pmax)
        self.model.PA          = st.slider("Apoptosis Chance (RTC) (%)", 0, 100, value=self.model.PA)
        self.model.CCT         = st.slider("Cell Cycle Time (hours)", 1, 48, value=self.model.CCT)
        self.model.Dt          = st.slider("Time Step (days)", 0.01, 1.0, value=self.model.Dt, step=0.01)
        self.model.PS          = st.slider("STC-STC Division Chance (%)", 0, 100, value=self.model.PS)
        self.model.mu          = st.slider("Migration Capacity", 0, 10, value=self.model.mu)

        self.model.PP = int(self.model.CCT * self.model.Dt / 24 * 100)
        self.model.PM = 100 * self.model.mu / 24

        init_config = (
            self.model.side_length, self.model.cycles, self.model.pmax,
            self.model.PA, self.model.CCT, self.model.Dt, self.model.PS, self.model.mu
        )
        config_hash = hashlib.md5(str(init_config).encode()).hexdigest()

        if (
            "initialized" not in st.session_state
            or "init_config_hash" not in st.session_state
            or st.session_state.init_config_hash != config_hash
        ):
            self.model.init_state()
            st.session_state.field = self.model.field.copy()
            st.session_state.initialized = True
            st.session_state.init_config_hash = config_hash

    def _modify_cell(self):
        """
        The function for initial state modification logic.
        """
        
        st.markdown("<h2 style='text-align: center;'>Initial State</h2>", unsafe_allow_html=True)

        x_coord = st.number_input("X Coordinate", 0, self.model.side_length - 1, value=self.model.side_length // 2)
        y_coord = st.number_input("Y Coordinate", 0, self.model.side_length - 1, value=self.model.side_length // 2)
        cell_value = st.number_input("Cell Value", 0, self.model.pmax + 1, value=self.model.pmax + 1)

        if st.button("Modify Cell"):
            self.model.field = st.session_state.field.copy()
            self.model.mod_cell(x_coord, y_coord, cell_value)
            st.session_state.field = self.model.field.copy()
            st.success(f"Cell modified at ({x_coord}, {y_coord}) to {cell_value}")

        fig, ax = plt.subplots()
        ax.imshow(st.session_state.field, cmap='viridis')
        ax.set_title("Updated Initial State")
        st.pyplot(fig)

    def _execute_model(self):
        """
        The function for model running logic.
        """
        
        if st.button("Run Model"):
            self.model.field = st.session_state.field.copy()
            self.model.stc_number = []
            self.model.rtc_number = []
            self.model.run_model(plot=False, animate=False, stats=False)

            stats_dict = self.model.get_statistics()
            new_stats_df = pd.DataFrame([stats_dict])

            if "stats_df" not in st.session_state:
                st.session_state.stats_df = new_stats_df
            else:
                st.session_state.stats_df = pd.concat([st.session_state.stats_df, new_stats_df], ignore_index=True)

            st.session_state.model_result = {
                "field": self.model.field.copy(),
                "stc_number": self.model.stc_number.copy(),
                "rtc_number": self.model.rtc_number.copy(),
                "params": {
                    "cycles": self.model.cycles,
                    "side_length": self.model.side_length,
                    "pmax": self.model.pmax
                }
            }

    def _visualize_results(self):
        """
        The function for the result visualization logic.
        """
        
        if "model_result" not in st.session_state:
            return

        result = st.session_state.model_result
        field = result["field"]
        stc = result["stc_number"]
        rtc = result["rtc_number"]
        pp_values = self.model.get_prolif_potentials().values()

        st.markdown("<h2 style='text-align: center;'>Visualization</h2>", unsafe_allow_html=True)
        fig, axs = plt.subplots(1, 3, figsize=(16, 4))
        axs[0].imshow(field)
        axs[1].plot(stc, 'C1', label='STC')
        axs[1].plot(rtc, 'C2', label='RTC')
        axs[2].bar(range(1, self.model.pmax + 2), list(pp_values), edgecolor='black')

        titles = [f"{result['params']['cycles']} hour cell growth", "Tumor cell count", "Value distribution"]
        labs_x = [f"{result['params']['side_length']*10} micrometers", "Time (hours)", "Proliferation potential"]
        labs_y = [f"{result['params']['side_length']*10} micrometers", "Cell numbers", "Number of appearance"]

        for i, ax in enumerate(axs):
            ax.set_title(titles[i])
            ax.set_xlabel(labs_x[i])
            ax.set_ylabel(labs_y[i])

        fig.colorbar(axs[0].imshow(field))
        axs[1].legend()
        st.pyplot(fig)

    def _show_statistics(self):
        """
        The function for the statistics printing logic.
        """
        
        if "stats_df" not in st.session_state:
            return

        df = st.session_state.stats_df
        base_cols = [col for col in df.columns if not str(col).isdigit() and "_STC" not in str(col) and "_RTC" not in str(col)]
        stc_cols  = sorted([col for col in df.columns if "_STC" in str(col)], key=lambda x: int(str(x).split("h")[0]))
        rtc_cols  = sorted([col for col in df.columns if "_RTC" in str(col)], key=lambda x: int(str(x).split("h")[0]))
        pp_cols   = sorted([col for col in df.columns if isinstance(col, int)])
        df.index  = df.index + 1

        st.markdown("<h2 style='text-align: center;'>Statistics</h2>", unsafe_allow_html=True)
        st.dataframe(df[base_cols])
        st.markdown("<h2 style='text-align: center;'>Proliferation Potentials</h2>", unsafe_allow_html=True)
        st.dataframe(df[pp_cols])
        st.markdown("<h2 style='text-align: center;'>Cell Numbers (STC)</h2>", unsafe_allow_html=True)
        st.dataframe(df[stc_cols])
        st.markdown("<h2 style='text-align: center;'>Cell Numbers (RTC)</h2>", unsafe_allow_html=True)
        st.dataframe(df[rtc_cols])
        st.markdown("<h2 style='text-align: center;'>Model Averages</h2>", unsafe_allow_html=True)

        st.pyplot(self.model.plot_averages(st.session_state.stats_df))

    def _reset_save_stats(self):
        """
        The function for the reset/download statistics logic.
        """
        
        if "stats_df" in st.session_state:
            if st.button("Reset Statistics Table"):
                del st.session_state.stats_df
                st.success("Statistics have been reset.")

            buffer = io.BytesIO()
            st.session_state.stats_df.to_excel(buffer, index=False)
            buffer.seek(0)

            st.download_button(
                label="Download Statistics (xlsx)",
                data=buffer,
                file_name="tumor_model_statistics.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
