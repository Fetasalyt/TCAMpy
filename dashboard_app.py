import streamlit as st
import TCAMpy as tcam

def main():
    st.set_page_config(
        page_title="TCAMpy Dashboard",
        layout="wide"
    )

    # --- Create model ---
    model = tcam.TModel(
        500,      # cycles
        50,       # side
        10,       # pmax
        1,        # PA
        24,       # CCT
        1/24,     # Dt
        15,       # PS
        4,        # mu
        5         # I
    )

    # --- Create dashboard ---
    dashboard = tcam.TDashboard(model)

    # --- Render the dashboard ---
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
