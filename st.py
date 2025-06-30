import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Function to create download link
def create_download_link(data, filename, text):
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Main page layout
def primary_drying_calculator():
    # Navigation bar
    st.markdown("""
    <style>
        .nav-bar {
            display: flex;
            justify-content: space-around;
            padding: 1rem;
            background-color: #f0f2f6;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .nav-item {
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 5px;
        }
        .nav-item.active {
            background-color: #1e88e5;
            color: white;
        }
        .section {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }
        .section-title {
            border-bottom: 2px solid #1e88e5;
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
            color: #1e88e5;
        }
        .parameter-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .parameter-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .btn-container {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin: 2rem 0;
        }
        .result-box {
            background-color: #e3f2fd;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation bar
    st.markdown("""
    <div class="nav-bar">
        <div class="nav-item active">Primary Drying Calculator</div>
        <div class="nav-item">Optimizer</div>
        <div class="nav-item">Design Space Generator</div>
        <div class="nav-item">Freezing Calculator</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content area - two columns
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        # Product Parameters Section
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Product Parameters</h3>', unsafe_allow_html=True)
        
        # Container Parameters
        st.markdown('<h4>Container Parameters</h4>', unsafe_allow_html=True)
        vial_area = st.number_input("Vial Area (cm²)", min_value=0.1, value=3.8, step=0.1)
        fill_volume = st.number_input("Fill Volume (mL)", min_value=0.1, value=2.0, step=0.1)
        product_area = st.number_input("Product Area (cm²)", min_value=0.1, value=3.14, step=0.1)
        critical_temp = st.number_input("Critical Product Temperature (°C)", value=-5.0, step=0.1)
        solid_content = st.number_input("Solid Content (g/mL)", min_value=0.0, value=0.05, step=0.01)
        
        # Vial Heat Transfer Section
        st.markdown('<h4>Vial Heat Transfer</h4>', unsafe_allow_html=True)
        kv_known = st.radio("Kv Known?", ["Yes", "No"], index=0, horizontal=True)
        
        if kv_known == "Yes":
            kc = st.number_input("Kc", value=0.000275, format="%.6f")
            kp = st.number_input("Kp", value=0.000893, format="%.6f")
            kd = st.number_input("KD", value=0.46, step=0.01)
        else:
            col_kv1, col_kv2, col_kv3 = st.columns(3)
            with col_kv1:
                kv_from = st.number_input("From", value=0.00106, format="%.6f")
            with col_kv2:
                kv_to = st.number_input("To", value=0.00108, format="%.6f")
            with col_kv3:
                kv_step = st.number_input("Step", value=-0.000001, format="%.6f")
        
        # Time Parameters
        st.markdown('<h4>Time Parameters</h4>', unsafe_allow_html=True)
        time_step = st.number_input("Time Step (hr)", min_value=0.001, value=0.01, step=0.01)
        
        st.markdown('</div>', unsafe_allow_html=True)  # End of section
    
    with col2:
        # Control Parameters Section
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Control Parameters</h3>', unsafe_allow_html=True)
        
        # Product Resistance Section
        st.markdown('<h4>Product Resistance</h4>', unsafe_allow_html=True)
        rp_known = st.radio("Product Resistance Known?", ["Yes", "No"], index=0, horizontal=True)
        
        if rp_known == "Yes":
            r0 = st.number_input("R₀", value=1.4, step=0.1)
            a1 = st.number_input("A₁", value=16.0, step=0.1)
            a2 = st.number_input("A₂", value=0.0, step=0.1)
        else:
            uploaded_file = st.file_uploader("Upload Vial Bottom Temperature Data", type=["txt", "csv"])
            if uploaded_file is not None:
                try:
                    df_temp = pd.read_csv(uploaded_file)
                    st.success("Temperature data uploaded successfully!")
                except:
                    st.error("Error reading the file. Please check the format.")
        
        # Initial Conditions
        st.markdown('<h4>Initial Conditions</h4>', unsafe_allow_html=True)
        initial_shelf_temp = st.number_input("Initial Shelf Temperature (°C)", value=-35.0, step=0.1)
        shelf_ramp_rate = st.number_input("Shelf Temperature Ramp Rate (°C/min)", value=1.0, step=0.1)
        chamber_ramp_rate = st.number_input("Chamber Pressure Ramp Rate (Torr/min)", value=0.5, step=0.1)
        
        # Equipment Capability
        st.markdown('<h4>Equipment Capability</h4>', unsafe_allow_html=True)
        a_value = st.number_input("a (kg/hr)", value=-0.182, step=0.001)
        b_value = st.number_input("b (kg/(hr·Torr))", value=11.7, step=0.1)
        
        # Other Parameters
        st.markdown('<h4>Other Parameters</h4>', unsafe_allow_html=True)
        num_vials = st.number_input("Number of Vials", min_value=1, value=398, step=1)
        
        st.markdown('</div>', unsafe_allow_html=True)  # End of section
    
    # Process Control Parameters Section (below columns)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Process Control Parameters</h3>', unsafe_allow_html=True)
    
    st.markdown('<div class="parameter-grid">', unsafe_allow_html=True)
    
    # Shelf Temperature row
    st.markdown('<div class="parameter-row">', unsafe_allow_html=True)
    st.markdown('<h4>Shelf Temperature (°C)</h4>', unsafe_allow_html=True)
    st1 = st.number_input("Step 1", value=20.0, key="shelf_temp1")
    st2 = st.number_input("Step 2", value=20.0, key="shelf_temp2")
    st3 = st.number_input("Step 3", value=20.0, key="shelf_temp3")
    st4 = st.number_input("Step 4", value=20.0, key="shelf_temp4")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Temperature Hold Time row
    st.markdown('<div class="parameter-row">', unsafe_allow_html=True)
    st.markdown('<h4>Temperature Hold Time (min)</h4>', unsafe_allow_html=True)
    tht1 = st.number_input("Step 1", min_value=0, value=1800, key="temp_hold1")
    tht2 = st.number_input("Step 2", min_value=0, value=0, key="temp_hold2")
    tht3 = st.number_input("Step 3", min_value=0, value=0, key="temp_hold3")
    tht4 = st.number_input("Step 4", min_value=0, value=0, key="temp_hold4")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chamber Pressure row
    st.markdown('<div class="parameter-row">', unsafe_allow_html=True)
    st.markdown('<h4>Chamber Pressure (Torr)</h4>', unsafe_allow_html=True)
    cp1 = st.number_input("Step 1", min_value=0.0, value=0.15, key="chamber_press1")
    cp2 = st.number_input("Step 2", min_value=0.0, value=0.15, key="chamber_press2")
    cp3 = st.number_input("Step 3", min_value=0.0, value=0.15, key="chamber_press3")
    cp4 = st.number_input("Step 4", min_value=0.0, value=0.15, key="chamber_press4")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Pressure Hold Time row
    st.markdown('<div class="parameter-row">', unsafe_allow_html=True)
    st.markdown('<h4>Pressure Hold Time (min)</h4>', unsafe_allow_html=True)
    pht1 = st.number_input("Step 1", min_value=0, value=1800, key="press_hold1")
    pht2 = st.number_input("Step 2", min_value=0, value=0, key="press_hold2")
    pht3 = st.number_input("Step 3", min_value=0, value=0, key="press_hold3")
    pht4 = st.number_input("Step 4", min_value=0, value=0, key="press_hold4")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # End of parameter grid
    st.markdown('</div>', unsafe_allow_html=True)  # End of section
    
    # Button Area
    st.markdown('<div class="btn-container">', unsafe_allow_html=True)
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Calculate", use_container_width=True):
            # In a real application, this would call the drying simulation function
            # For demonstration, we'll generate mock results
            st.session_state.calculated = True
            st.session_state.drying_time = 12.62
            
            # Generate mock temperature data
            time_points = np.linspace(0, 15, 100)
            shelf_temp = np.full_like(time_points, st1)
            product_temp = shelf_temp - 5 + np.random.normal(0, 0.5, len(time_points))
            
            st.session_state.temp_data = pd.DataFrame({
                "Time (hr)": time_points,
                "Shelf Temperature (°C)": shelf_temp,
                "Product Temperature (°C)": product_temp
            })
    
    with col_btn2:
        if st.button("Download Result", use_container_width=True, disabled=("calculated" not in st.session_state)):
            # Create CSV file for download
            csv = st.session_state.temp_data.to_csv(index=False).encode()
            st.markdown(create_download_link(csv, "drying_results.csv", "Download CSV"), 
                        unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # End of button container
    
    # Results Section (after calculation)
    if "calculated" in st.session_state:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Results</h3>', unsafe_allow_html=True)
        
        # Drying time result
        st.markdown(f'<div class="result-box"><h4>Primary Drying Time</h4><h2>{st.session_state.drying_time:.2f} hr</h2></div>', 
                   unsafe_allow_html=True)
        
        # Temperature plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(st.session_state.temp_data["Time (hr)"], 
                st.session_state.temp_data["Shelf Temperature (°C)"], 
                label="Shelf Temperature", linewidth=2)
        ax.plot(st.session_state.temp_data["Time (hr)"], 
                st.session_state.temp_data["Product Temperature (°C)"], 
                label="Product Temperature", linewidth=2)
        
        # Critical temperature line
        ax.axhline(y=critical_temp, color='r', linestyle='--', label="Critical Temperature")
        
        ax.set_xlabel("Time (hr)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("Temperature Profile")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
        
        # Show data table
        st.dataframe(st.session_state.temp_data.head(10))
        
        st.markdown('</div>', unsafe_allow_html=True)  # End of results section

# Main app
def main():
    st.set_page_config(
        page_title="LyoPRONTO - Primary Drying Calculator",
        layout="wide",
        page_icon="❄️"
    )
    
    # Hide Streamlit branding
    hide_st_style = """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    
    primary_drying_calculator()

if __name__ == "__main__":
    main()
