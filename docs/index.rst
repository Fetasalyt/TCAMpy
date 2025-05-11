.. TCAMpy documentation master file, created by
   sphinx-quickstart on Fri Mar 28 22:37:35 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TCAMpy's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   What is TCAMpy?
   Theoretical Background
   Installation
   Using the Model
   API Documentation

What is TCAMpy?
===============

A single python module for a cellular automaton modeling tumor growth. The user can set the parameters, create unique
initial states, view and save statistics, save data and also use a streamlit dashboard as a graphical interface. Growth
plots, histograms and animation is available for visualization in an easy to use way.

Theoretical Background
======================

The theoretical background for this model is based on the work of Carlos A Valentim, José A Rabi and Sergio A David.

Valentim CA, Rabi JA, David SA. Cellular-automaton model for tumor growth dynamics: Virtualization of different scenarios.
Comput Biol Med. 2023 Feb;153:106481. doi: 10.1016/j.compbiomed.2022.106481. Epub 2022 Dec 28. PMID: 36587567.
(url: [https://pubmed.ncbi.nlm.nih.gov/36587567/])

This is a 2D cellular automata model; the area is a square grid, with each cell being able to have discrate values ranging from 0 to a 'pmax'+1 (see below). If a square
is 0, it is empty, if not 0, a tumor cell exists there with a proliferation potential of the square's value.

In the model, we have two types of tumor cells. RTCs (regular tumor cells) can only divide a limited amount of times. This amount is their proliferation potential,
which gets smaller with every division. These cells can only proliferate into other RTCs, and the daughter cell will have the same proliferation potential value as the
mother cell after the division (So an RTC with pp = 5 will create two cells, whose pp = 4). STCs (stem tumor cells) have infinite divisions. Their daughter cell is either
an RTC with the highest proliferation potential value (pmax) or (by a small chance) another STC. The value which we represent these cells with in the model is pmax+1.

At every cycle each cell chooses from four different options. First, they can die by apoptosis (low chance for RTCs only). If they survive, they can proliferate with the probability
of CCT*dt/24 (where CCT is the cell cycle time, dt is the time step in the model). In case of no proliferation, they can migrate to one of the eight neighbouring cells. The probability
of this action is mu*dt (mu is the migration capacity of the cell). If none of these actions happen, the cell stays quiescent. Quiescence is forced if there is no free space around the
a surviving cell.

All these probabilites, the maximum proliferation potential value, model duration and area size is a parameter of the model, which can be set by the user.

Installation
============
To get started with this project, follow the instructions below.

1. Install the necessary dependencies using `conda install` or `pip`.
2. Clone the repository and set up your environment.

Usage
=====
After you installed TCAMpy and it's dependencies (which should be automatically installed with TCAMpy)
you can start using the model. First, you import the module, then create a 'TModel' class, like this:

.. code-block:: python

    import TCAMpy as tcam

    M = tcam.TModel(50, 500, 10, 1, 24, 1/24, 15, 4)

The 'TModel' class takes a number of parameters. These determine the area, the different probabilities as well as the duration
of the model. See :ref:`api-docs` for further details about them. After you've created the class with the parameters of your
choice, you can simply run the entire model with one command:

.. code-block:: python

    M.run_model(plot = True, animate = False, stats = True)

This function automatically creates an initial state with a STC in the middle of the field. In every cycle, for the duration
of the model, it checks the entire area and chooses an action for each tumor cell. If 'plot' is set to True, the output includes multiple plots:
a growth plot, a cell number plot and a histogram of the proliferation potentails. If the 'stats' parameter is set to True,
various statistical properties of the model will be printed about the field and the cells. If 'animate' is set to True, the
function returns an animation of the growth. Note that animation does not work while using 'inline' backend. You must change
the backend first.

You can also modifiy the initial state by calling the following functions before running the model:

.. code-block:: python

    M.init_state()
    M.mod_cell(x, y, value)

The 'mod_cell()' modifies the value of the cell at the (x, y) coordinates. For example, if you want to add another STC at (10,20),
use this command with x = 10, y = 20 and value = M.pmax + 1. You can change the initial state as much as you'd like before running
the model.

For this function to work, first you need to create an initial state manually by the 'init_state()' function. You can remove the automatically
created STC by changing it's value to 0. (You don't need to call this function if you don't want to modifiy the initial state, running the model
creates a basic initial state, if you didn't define one before.)

You can run multiple models using the 'run_multimodel()' function. You must specifiy how many models you'd like to run, and what initial state you'd
like to use for them as a numpy array. (If you want to stick with the default initial state just say 'M.field'.) This function returns the results
as a pnadas dataframe. Plot the averages with standard deviation with the 'plot_averages()' function.

.. code-block:: python

    stats = M.run_multimodel(5, M.field)
    M.plot_averages(stats)

If you'd like to use this model on a graphical interface, you can create a streamlit dashboard (after creating model):

.. code-block:: python

    D = tcam.TDashboard(M)
    D.run_dashboard()

You will need to run the file containing this code in your command line. A dashboard will be created, where you have full control over the model. You can set the parameters using the sliders, run the model, view plots as well as statistics.

You also have access to commands to save results (the field or the statistics) to an excel file, or create your own run function/loop by individually accessing cycles and cells. For details on those functions check the API Documentation.

API Documentation
=================
The following section provides the API documentation for the project.

.. automodule:: TCAMpy
   :members:
   :undoc-members:
   :show-inheritance:

License
=======
This project is licensed under the MIT License - see the https://github.com/Fetasalyt/TCAMpy/blob/main/LICENSE (LICENSE) file for details.
