import Model.TCAMpy as tcam

if __name__ == "__main__":
    # Create and Run a Model
    M = tcam.TModel(50, 500, 10, 1, 24, 1/24, 15, 4)
    M.tcam.run_model(animated = False, stats = True)
    
    # Create a Graphical Interface
    # M.tcam.create_dashboard()