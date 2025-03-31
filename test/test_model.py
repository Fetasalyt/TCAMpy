import Model.TCAMpy2D as tcam

# Create a Model
M = tcam.TModel(50, 500, 10, 1, 24, 1/24, 15, 4)

# Run Model
M.run_model(animated = False, stats = True)
    
# Create a Graphical Interface
# M.tcam.create_dashboard()
