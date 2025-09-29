import TCAMpy as tcam

M = tcam.TModel(500, 50, 10, 1, 24, 1/24, 15, 4, 5)

# -- Create dashbard --

D = tcam.TDashboard(M)
D.run_dashboard()
