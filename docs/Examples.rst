Example codes
=============

Single model
------------

Create a model and execute it one time. Plot the results, animate and show the statistics. (Animation doesn't work with inline backend)

.. code-block:: python

    import TCAMpy as tcam

    M = tcam.TModel(500, 50, 10, 1, 24, 1/24, 15, 4, 5)
    M.run_model(plot = True, animate = True, stats = True)

Multiple executions
-------------------

Run the model multiple (in this case 5) times. Save the results to 'stats' and plot the averages.

.. code-block:: python

    import TCAMpy as tcam
    import matplotlib.pyplot as plt

    stats = M.run_multimodel(5, M.field)
    M.plot_averages(stats)

    # Check visualization for every execution (optional)
    for i in range(len(M.runs)):
        M.plot_run(i+1)

Modifying initial state
-----------------------

Modify the initial state by defining/changing M.field or calling 'mod_cell()' (can only be called after M.field was created either manually or by 'init_state()').

.. code-block:: python

    import TCAMpy as tcam
    import numpy as np

    M = tcam.TModel(500, 50, 10, 1, 24, 1/24, 15, 4, 5)

    M.field = (M.pmax+1) * np.eye(M.side_length)
    M.field[0][0] = 0
    M.mod_cell(M.side_length-1, M.side_length-1, 0)

    M.run_model()

Dashboard
---------

Create a streamlit dashboard. Run the file containing the code with streamlit to open the application.

.. code-block:: python

    import TCAMpy as tcam

    M = tcam.TModel(500, 50, 10, 1, 24, 1/24, 15, 4, 5)

    D = tcam.TDashboard(M)
    D.run_dashboard()

.. code-block:: console

  streamlit run file_name.py

Example files
-------------

For example files please visit: https://github.com/Fetasalyt/TCAMpy/tree/main/examples
