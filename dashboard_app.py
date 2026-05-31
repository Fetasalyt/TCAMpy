import streamlit as st
import TCAMpy as tcam

def main():

    # --- Create model ---
    model = tcam.TModel(
        480,      # cycles
        75,       # side
        20,       # pmax
        1,        # PA
        24,       # CCT
        1/24,     # Dt
        25,       # PS
        4,        # mu
        0,        # ad
        3,        # I
        5         # M
    )

    # --- Create dashboard ---
    dashboard = tcam.TDashboard(model)

    # --- Render the dashboard ---
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
