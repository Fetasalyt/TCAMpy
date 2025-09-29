import TCAMpy as tcam

# -- Create and Run Model --

M = tcam.TModel(50, 500, 10, 1, 24, 1/24, 15, 4, 5)
M.run_model(plot = True, animate = True, stats = True)

# -- Run multiple models --

# stats = M.run_multimodel(3, M.field)

# print(stats)
# M.plot_averages(stats)
