import TCAMpy as tcam

# Create and Run Model
M = tcam.TModel(50, 500, 10, 1, 24, 1/24, 15, 4)

M.run_model(plot = True, animate = True, stats = True)

# ---------- OR ----------

# Create a Graphical Interface
# M.create_dashboard()
