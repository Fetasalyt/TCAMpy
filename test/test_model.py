import TCAMpy as tcam

# Create and Run Model
M = tcam.TModel(50, 500, 10, 1, 24, 1/24, 15, 4)
M.run_model(animated = False, stats = True)
   
# Create a Graphical Interface
# M.tcam.create_dashboard()
