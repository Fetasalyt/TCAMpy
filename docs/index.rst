.. TCAMpy documentation master file, created by
   sphinx-quickstart on Fri Mar 28 22:37:35 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TCAMpy's documentation!
==================================

.. include:: ../README.md
   :parser: myst_parser.sphinx_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Background
   Installation
   Usage
   API

Installation
============
To get started with this project, follow the instructions below.

1. Install the necessary dependencies using `conda install` or `pip`.
2. Clone the repository and set up your environment.

The dependencies for TCAMpy are shown below:

.. include:: ../requirements.txt
   :literal:

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
