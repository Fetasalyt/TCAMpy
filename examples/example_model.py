import TCAMpy as tcam

# -- Create and Run Model --

M = tcam.TModel(480, 75, 20, 1, 24, 1/24, 25, 4, 0, 3, 5)
M.run_model(plot = True, animate = True, stats = True)

# -- Run multiple models --

# stats = M.run_multimodel(5, M.field, plot = True, stats = True)
# stats.to_excel("simulations.xlsx")
