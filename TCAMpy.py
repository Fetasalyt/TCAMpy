import io
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
        fig, axs = plt.subplots(1, 3, figsize=(17,4))
        field_for_histogram = self.field[self.field > 0]

        axs[0].imshow(self.field)
        axs[1].plot(self.stc_number, 'C1', label='STC')
        axs[1].plot(self.rtc_number, 'C2', label='RTC')
        axs[2].hist(field_for_histogram.ravel(), edgecolor='black')

        # Titles/labels of the plots
        titles = [str(self.cycles)+ " hour cell growth", "Tumor cell count", "Value destribution"]
        labs_x = [str(self.side_length*10) + " micrometers", "Time (hours)", "Proliferation potential"]
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
        if self.field[x][y] == self.pmax+1:
            return False
        else:
            return self.PA >= random.randint(1,100)

    def __action_true__(self, x, y, ch):
        free_nb = self.get_free_neighbours(x, y)
        if len(free_nb) == 0:
            return False
        else:
            return ch >= random.randint(1,100)
    
    def cell_step(self, x, y, step_type):
        """
        The function that makes a single cell do one of the following actions:
        prolif STC - STC, prolif STC - RTC, prolif RTC - RTC, migration (1-4).

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
        Save the return to a self.var_name variable!
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

    def get_statistics(self):
        """
        Returns various statistical properties of the model.
        """
        
        # Statistics
        nonzero_field = self.field[self.field > 0]
        stats = {
            "Min": nonzero_field.min(),
            "Max": nonzero_field.max(),
            "Mean": nonzero_field.mean(),
            "Std": nonzero_field.std(),
            "Var": nonzero_field.var(),
            "Median": np.median(nonzero_field),
            "Skew": skew(nonzero_field.ravel()),
            "Kurtosis": kurtosis(nonzero_field.ravel()),
            "Final STC": self.stc_number[self.cycles-1],
            "Final RTC": self.rtc_number[self.cycles-1],
        }
        
        # Proliferation potentials
        unique, counts = np.unique(nonzero_field, return_counts=True)
        for val, count in zip(unique, counts):
            stats[f"PP_{int(val)}"] = count
    
        return stats
    
    def print_statistics(self):
        """
        Prints various statistical properties of the model.
        """
        
        print(self.get_statistics())
    
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
        if stats: self.print_statistics()

    def create_dashboard(self):
        """
        Creates a graphical user interface. The GUI is a streamlit dashboard.
        
        On the GUI you can change the value of parameters, modify initial state,
        activate animation, run and plot the model, and save the results to a file.
        """
        
        st.set_page_config(layout="wide")
        
        st.markdown("<h1 style='text-align: center;'>TCAMpy</h1>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([4, 1, 12])
        
        # --- INITIALIZATION ---
        if "initialized" not in st.session_state:
            self.init_state()
            st.session_state.field = self.field.copy()
            st.session_state.run_model = True
            st.session_state.initialized = True
    
        # --- COLUMN 1: PARAMETERS + RUN BUTTON ---
        with col1:
            st.markdown("<h2 style='text-align: center;'>Model Parameters</h2>", unsafe_allow_html=True)
    
            # UI Controls – only update the instance, not the stored state yet
            side_length = st.slider("Side Length (10um)", 10, 200, value=self.side_length)
            cycles      = st.slider("Model Duration (hours)", 50, 1000, value=self.cycles)
            pmax        = st.slider("Max Proliferation Potential", 1, 20, value=self.pmax)
            PA          = st.slider("Apoptosis Chance (RTC) (%)", 0, 100, value=self.PA)
            CCT         = st.slider("Cell Cycle Time (hours)", 1, 48, value=self.CCT)
            Dt          = st.slider("Time Step (days)", 0.01, 1.0, value=self.Dt, step=0.01)
            PS          = st.slider("STC-STC Division Chance (%)", 0, 100, value=self.PS)
            mu          = st.slider("Migration Capacity", 0, 10, value=self.mu)
    
            st.markdown("<h2 style='text-align: center;'>Initial State</h2>", unsafe_allow_html=True)
    
            x_coord    = st.number_input("X Coordinate", 0, side_length-1, value=side_length//2)
            y_coord    = st.number_input("Y Coordinate", 0, side_length-1, value=side_length//2)
            cell_value = st.number_input("Cell Value", 0, pmax+1, value=pmax+1)
    
            if st.button("Add Cell"):
                self.field = st.session_state.field.copy()
                self.mod_cell(x_coord, y_coord, cell_value)
                st.session_state.field = self.field.copy()
                st.success(f"Cell added at ({x_coord}, {y_coord}) with value {cell_value}")
    
            # Show current field (pre-run)
            fig, ax = plt.subplots()
            ax.imshow(st.session_state.field, cmap='viridis')
            ax.set_title("Updated Initial State")
            st.pyplot(fig)
    
            if st.button("Run Model"):
                self.side_length = side_length
                self.cycles = cycles
                self.pmax = pmax
                self.PA = PA
                self.CCT = CCT
                self.Dt = Dt
                self.PS = PS
                self.mu = mu
                self.PP = int(CCT * Dt / 24 * 100)
                self.PM = 100 * mu / 24
    
                self.field = st.session_state.field.copy()
                self.stc_number = []
                self.rtc_number = []
    
                self.run_model(plot=False, animate=False, stats=False)
    
                # Get statistics as a dict and convert to one-row DataFrame
                stats_dict = self.get_statistics()
                new_df = pd.DataFrame([stats_dict])
                
                # If not already in session state, create the cumulative stats DataFrame
                if "stats_df" not in st.session_state:
                    st.session_state.stats_df = new_df
                else:
                    st.session_state.stats_df = pd.concat([st.session_state.stats_df, new_df], ignore_index=True)
    
                # Store model state after running
                st.session_state.model_result = {
                    "field": self.field.copy(),
                    "stc_number": self.stc_number.copy(),
                    "rtc_number": self.rtc_number.copy(),
                    "params": {
                        "cycles": self.cycles,
                        "side_length": self.side_length,
                        "pmax": self.pmax
                    }
                }

                # Freeze until next click
                st.session_state.run_model = False
    
        # --- COLUMN 3: RESULTS ONLY WHEN MODEL HAS BEEN RUN ---
        with col3:
            if "model_result" in st.session_state:
                result = st.session_state.model_result
                field = result["field"]
                stc = result["stc_number"]
                rtc = result["rtc_number"]
    
                st.markdown("<h2 style='text-align: center;'>Visualization</h2>", unsafe_allow_html=True)
                field_for_histogram = np.array(field)[np.array(field) > 0]
                fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    
                axs[0].imshow(field)
                axs[1].plot(stc, 'C1', label='STC')
                axs[1].plot(rtc, 'C2', label='RTC')
                axs[2].hist(field_for_histogram.ravel(), edgecolor='black')
    
                titles = [f"{result['params']['cycles']} hour cell growth", "Tumor cell count", "Value distribution"]
                labs_x = [f"{result['params']['side_length']*10} micrometers", "Time (hours)", "Proliferation potential"]
                labs_y = [f"{result['params']['side_length']*10} micrometers", "Cell numbers", "Number of appearance"]
    
                axs[2].set_xticks(range(1, result['params']['pmax'] + 1))
                for i, ax in enumerate(axs):
                    ax.set_title(titles[i])
                    ax.set_xlabel(labs_x[i])
                    ax.set_ylabel(labs_y[i])
    
                fig.colorbar(axs[0].imshow(field))
                axs[1].legend()
                st.pyplot(fig)
    
                self.field = field
                self.stc_number = stc
                self.rtc_number = rtc
                
                # Show Statistics
                if "stats_df" in st.session_state:
                    df = st.session_state.stats_df
                
                    # Sort PP columns numerically
                    base_cols = [col for col in df.columns if not col.startswith("PP_")]
                    pp_cols = sorted([col for col in df.columns if col.startswith("PP_")],
                                     key=lambda x: int(x.split("_")[1]))
                    df = df[base_cols + pp_cols]
                    df.index += 1
                                    
                    st.markdown("<h2 style='text-align: center;'>Statistics</h2>", unsafe_allow_html=True)
                    st.dataframe(df.iloc[:, :10])  # Summary statistics
                
                    st.markdown("<h2 style='text-align: center;'>Proliferation Potentials</h2>", unsafe_allow_html=True)
                    st.dataframe(df.iloc[:, 10:])  # Proliferation Potential counts
                    
                # Reset Statistics
                if st.button("Reset Statistics Table"):
                    if "stats_df" in st.session_state:
                        del st.session_state.stats_df
                        st.success("Statistics table has been reset.")
                    
                # Download Statistics
                if "stats_df" in st.session_state:
                    buffer = io.BytesIO()
                    st.session_state.stats_df.to_excel(buffer, index=False)
                    buffer.seek(0)
                
                    st.download_button(
                        label="📥 Download Statistics",
                        data=buffer,
                        file_name="tumor_model_statistics.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                # # Animation
                # st.write("Generating animation...")
                
                # frames = []
                # for img in self.images:
                #     fig, ax = plt.subplots()
                #     ax.imshow(img[0].get_array(), cmap='viridis')
                    
                #     # Convert plot to image
                #     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                #     fig.savefig(temp_file.name, format='png', bbox_inches='tight')
                #     plt.close(fig)
                #     frames.append(imageio.imread(temp_file.name))
                    
                # # Save frames as a GIF
                # with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmpfile:
                #     imageio.mimsave(tmpfile.name, frames, duration=0.1)
                #     st.image(tmpfile.name, caption="Tumor Growth Animation", use_container_width=True)

# Example usage
M = TModel(75, 500, 10, 1, 24, 1/24, 15, 4)
# M.run_model(plot = True, animate = False, stats = True)

M.create_dashboard()
