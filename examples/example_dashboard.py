import TCAMpy as tcam

M = tcam.TModel(480, 75, 20, 1, 24, 1/24, 25, 4, 0, 3, 5)

# -- Create dashbard --

D = tcam.TDashboard(M)
D.run_dashboard()

# In command line: streamlit run file_path.py
