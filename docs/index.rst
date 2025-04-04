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
   Getting Started
   Using the Model
   API Documentation

What is TCAMpy?
===============

Theoretical Background
======================

Getting Started
===============
To get started with this project, follow the instructions below.

1. Install the necessary dependencies using `conda install` or `pip`.
2. Clone the repository and set up your environment.

Usage
=====
After you installed TCAMpy and it's dependencies (which should be automatically installed with TCAMpy)
you can start using the model. First, you import the module, then create a 'TModel' class, like this:

.. code-block:: python

    import TCAMpy as tcam

    M = tcam.TModel(50, 500, 1, 24, 1/24, 15, 4)

The 'TModel' class takes a number of parameters. These determine the area, the different probabilities as well as the duration
of the model. See :ref:`api-docs` for further details about them. After you've created the class with the parameters of your
choice, you can simply run the entire model with one command:

.. code-block:: python

    M.run_model(animated = False, stats = True)

This function automatically creates an initial state with a STC in the middle of the field. In every cycle, for the duration
of the model, it checks the entire area and chooses an action for each tumor cell. The output includes multiple plots:
a growth plot, a cell number plot and a histogram of the proliferation potentails. If the 'stats' parameter is set to true,
various statistical properties of the model will be printed about the field and the cells. If 'animated' is set to true, the
function returns an animation of the growth. Note that animation does not work while using 'inline' backend. You must change
the backend first.

You can also modifiy the initial state by calling the following function before running the model:

.. code-block:: python

    M.mod_cell(x, y, value)

This function modifies the value of the cell at the (x, y) coordinates. For example, if you want to add another STC at (10,20),
use this command with x = 10, y = 20 and value = M.pmax + 1. You can change the initial state as much as you'd like before running
the model.

.. _api-docs:

API Documentation
=================
The following section provides the API documentation for the project.

.. automodule:: TCAMpy
   :members:
   :undoc-members:
   :show-inheritance:

License
=======
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
