Example codes
=============

Single model
------------

Create a model and execute it one time. Plot the results, animate and show the statistics. (Animation doesn't work with inline backend)

.. code-block:: python

    import TCAMpy as tcam

    M = tcam.TModel(480, 75, 20, 1, 24, 1/24, 25, 4, 0, 3, 5)
    M.run_model(plot = True, animate = True, stats = True)

Multiple executions
-------------------

Run the model multiple (in this case 5) times and plot the average results.

.. code-block:: python

    import TCAMpy as tcam
    import matplotlib.pyplot as plt

    M = tcam.TModel(480, 75, 20, 1, 24, 1/24, 25, 4, 0, 3, 5)
    stats = M.run_multimodel(5, M.cancer, plot = True, stats = True)

    # Check visualization for every execution
    for i in range(len(M.runs)):
        M.plot_run(i+1)

    # Save stats to excel
    stats.to_excel("simulations.xlsx")

Modifying initial state
-----------------------

Modify the initial state by defining/changing M.cancer or calling 'mod_cell()'.

.. code-block:: python

    import TCAMpy as tcam
    import numpy as np

    M = tcam.TModel(480, 75, 20, 1, 24, 1/24, 25, 4, 0, 3, 5)

    M.cancer = (M.pmax+1) * np.eye(M.side)
    M.cancer[0][0] = 0
    M.mod_cell(M.side-1, M.side-1, 0)

    M.run_model(plot = True, animate = False, stats = True)

Dashboard
---------

Create a streamlit dashboard. Run the file containing the code with streamlit to open the application.

.. code-block:: python

    import TCAMpy as tcam

    M = tcam.TModel(480, 75, 20, 1, 24, 1/24, 25, 4, 0, 3, 5)

    D = tcam.TDashboard(M)
    D.run_dashboard()

.. code-block:: console

  streamlit run file_name.py

This dashboard can be used online with Streamlit Community Cloud, without any coding: https://tcampy.streamlit.app/

Machine Learning
----------------

Use ML features to generate dataset, train a model and predict tumor size based on given parameters

.. code-block:: python

    import TCAMpy as tcam

    M = tcam.TModel(480, 75, 20, 1, 24, 1/24, 25, 4, 0, 3, 5)
    ml = tcam.TML(M)

    # Select parameters to randomize and ranges
    randomize = {
        "pmax": (15, 20),
        "PA":   (1, 3),
        "CCT":  (12, 48),
        "PS":   (10, 75),
        "mu":   (2, 10),
        "ad":   (0, 10),
        "M":    (0, 10),
        "I":    (0, 5),
    }

    # Generate dataset with randomized parameters
    df = ml.generate_dataset(
        n=50,
        random_params=randomize,
        output_file="tumor_dataset.csv"
    )

    # Train a model
    model, metrics = ml.train_predictor("tumor_dataset.csv", "Tumor size")

    new_params = [480, 75, 15, 1, 24, 1/24, 40, 4, 5, 3, 10]
    print ("Predicted Attribute: ", ml.predict_new(new_params))

Example files
-------------

For example files please visit: https://github.com/Fetasalyt/TCAMpy/tree/main/examples
