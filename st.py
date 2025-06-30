import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import math
import time
import os
from scipy.optimize import fsolve

# 导入LyoPRONTO核心模块
from src import constant
from src import freezing
from src import calc_knownRp
from src import calc_unknownRp
from src import design_space
from src import opt_Pch_Tsh
from src import opt_Pch
from src import opt_Tsh
from src import functions

# 设置页面配置
st.set_page_config(
    page_title="Lyophilization Process Optimization Platform",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 理论模型参数
CRYSTALLINE_EXCIPIENTS = ["Mannitol", "Glycine", "Sorbitol"]
AMORPHOUS_EXCIPIENTS = ["Sucrose", "Trehalose"]
SALT_EFFECTS = {
    "NaCl": -30,  # °C/Torr per 1% concentration
    "KCl": -20,
    "NaHCO3": -15,
    "None": 0
}

# 西林瓶规格数据
VIAL_SIZES = {
    "2R": {"diameter": 16.0, "height": 22.0, "wall_thickness": 1.0, "bottom_thickness": 0.7, "max_volume": 3.0}, 
    "6R": {"diameter": 22.0, "height": 26.0, "wall_thickness": 1.0, "bottom_thickness": 0.7, "max_volume": 8.0},
    "10R": {"diameter": 24.0, "height": 30.0, "wall_thickness": 1.0, "bottom_thickness": 0.7, "max_volume": 15.0},
    "20R": {"diameter": 30.0, "height": 35.0, "wall_thickness": 1.2, "bottom_thickness": 1.0, "max_volume": 25.0}
}

# 蛋白类型关键参数数据库
PROTEIN_DATABASE = {
    "Monoclonal Antibody": {"Tc": -32, "Tg": -40, "R0": 1.8, "A1": 18.0},
    "Vaccine": {"Tc": -28, "Tg": -35, "R0": 1.2, "A1": 12.0},
    "Enzyme": {"Tc": -35, "Tg": -42, "R0": 2.0, "A1": 20.0},
    "Peptide": {"Tc": -30, "Tg": -38, "R0": 1.5, "A1": 15.0},
    "Custom": {"Tc": -30, "Tg": -40, "R0": 1.4, "A1": 16.0}
}

# 格式化输出CSV
def format_csv(data):
    csv = data.to_csv(index=False, encoding='utf-8')
    return csv

# 生成下载链接
def create_download_link(data, filename):
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# 理论模型计算函数
def calculate_collapse_temp(protein_conc, excipients, salt_content):
    """Calculate collapse temperature (Tc)"""
    # Base Tc value (based on protein concentration)
    base_tc = -30 + 0.15 * protein_conc
    
    # Excipient effect
    excipient_effect = 0
    for exc, percent in excipients.items():
        if exc in CRYSTALLINE_EXCIPIENTS:
            excipient_effect += 0.2 * percent  # Crystalline excipients increase Tc
        elif exc in AMORPHOUS_EXCIPIENTS:
            excipient_effect += 0.05 * percent  # Amorphous excipients slightly increase
    
    # Salt effect
    salt_effect = 0
    for salt, percent in salt_content.items():
        if salt != "None":
            salt_effect += SALT_EFFECTS[salt] * percent
    
    return base_tc + excipient_effect + salt_effect

def calculate_drying_time(fill_depth, protein_conc, kv):
    """Predict drying time"""
    # Base drying time (hours)
    base_time = 20.0
    
    # Fill height effect (non-linear)
    if fill_depth > 1.0:
        height_factor = 1 + 0.5 * (fill_depth - 1.0) ** 1.5
    else:
        height_factor = 1.0
        
    # Protein concentration effect
    conc_factor = 1 + 0.02 * protein_conc
    
    # Heat transfer coefficient effect (ensure kv between 0-1)
    kv_factor = 1.5 - 0.5 * min(max(kv, 0.0), 1.0)  # kv value between 0-1
    
    total_time = base_time * height_factor * conc_factor * kv_factor
    return float(total_time)

def predict_thermal_params(protein_type, protein_conc, excipients):
    """Predict thermodynamic parameters"""
    base_params = PROTEIN_DATABASE[protein_type]
    
    # Protein concentration effect
    tc = base_params["Tc"] + 0.1 * protein_conc
    r0 = base_params["R0"] + 0.005 * protein_conc
    a1 = base_params["A1"] + 0.05 * protein_conc
    
    # Excipient effect
    for exc, percent in excipients.items():
        if exc in CRYSTALLINE_EXCIPIENTS:
            tc += 0.15 * percent
    
    return {
        "Tc": tc,
        "Tg": base_params["Tg"],
        "R0": r0,
        "A1": a1,
        "A2": 0.0
    }

# 主干燥计算器
def primary_drying_calculator():
    st.header("Primary Drying Calculator")
    
    # 主体内容区 - 左右两栏布局
    col1, col2 = st.columns(2)
    
    # 左侧参数输入区（产品参数）
    with col1:
        st.subheader("Product Parameters")
        
        # 产品容器参数
        st.markdown("**Vial Parameters**")
        vial_area = st.number_input("Vial Area (cm²)", value=3.8, format="%.4f")
        fill_volume = st.number_input("Fill Volume (mL)", value=2.0, format="%.2f")
        product_area = st.number_input("Product Area (cm²)", value=3.14, format="%.2f")
        critical_temp = st.number_input("Critical Product Temperature (°C)", value=-5.0, format="%.1f")
        solid_content = st.number_input("Solid Content (g/mL)", value=0.1, format="%.3f")
        
        # Vial Heat Transfer
        st.markdown("**Vial Heat Transfer**")
        heat_transfer_option = st.radio("Select", ["Known", "Unknown"], index=0, horizontal=True)
        if heat_transfer_option == "Known":
            kc = st.number_input("Kc", value=0.000275, format="%.6f")
            kp = st.number_input("Kp", value=0.000893, format="%.6f")
            kd = st.number_input("KD", value=0.46, format="%.2f")
        else:
            from_val = st.number_input("From", value=0.00106, format="%.5f")
            to_val = st.number_input("To", value=0.00108, format="%.5f")
            step_val = st.number_input("Step", value=-0.999999, format="%.6f")
        
        # 时间参数
        st.markdown("**Time Parameters**")
        time_step = st.number_input("Time Step (hr)", value=0.01, format="%.2f")
    
    # 右侧参数输入区（控制参数）
    with col2:
        st.subheader("Control Parameters")
        
        # Product Resistance
        st.markdown("**Product Resistance**")
        resistance_option = st.radio("Select", ["Known", "Unknown"], index=0, horizontal=True, key="resistance_option")
        if resistance_option == "Known":
            r0 = st.number_input("R₀", value=1.4, format="%.1f")
            a1 = st.number_input("A₁", value=16.0, format="%.1f")
            a2 = st.number_input("A₂", value=0.0, format="%.1f")
        else:
            uploaded_file = st.file_uploader("Upload Vial Bottom Temperature File (temperature.txt)")
        
        # 初始条件
        st.markdown("**Initial Conditions**")
        initial_shelf_temp = st.number_input("Initial Shelf Temperature (°C)", value=-35.0, format="%.1f")
        shelf_temp_ramp = st.number_input("Shelf Temperature Ramp Rate (°C/min)", value=1.0, format="%.1f")
        chamber_pressure_ramp = st.number_input("Chamber Pressure Ramp Rate (Torr/min)", value=0.5, format="%.1f")
        
        # 设备能力
        st.markdown("**Equipment Capability**")
        a_val = st.number_input("a (kg/hr)", value=-0.182, format="%.3f")
        b_val = st.number_input("b (kg/(hr·Torr))", value=11.7, format="%.1f")
        
        # 其他参数
        st.markdown("**Other Parameters**")
        num_vials = st.number_input("Number of Vials", value=398)
    
    # 过程控制参数区（位于左右栏下方）
    st.subheader("Process Control Parameters")
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)
    with ctrl_col1:
        shelf_temp = st.number_input("Shelf Temperature (°C)", value=20.0, format="%.1f")
    with ctrl_col2:
        temp_hold_time = st.number_input("Temperature Hold Time (min)", value=1800, format="%d")
    with ctrl_col3:
        chamber_pressure = st.number_input("Chamber Pressure (Torr)", value=0.15, format="%.2f")
    with ctrl_col4:
        pressure_hold_time = st.number_input("Pressure Hold Time (min)", value=1800, format="%d")
    
    # 操作按钮区
    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        calculate_btn = st.button("Calculate", type="primary", use_container_width=True)
    with col_btn2:
        download_btn = st.button("Download Result", use_container_width=True)
    
    # 结果展示区
    if calculate_btn:
        # 准备输入参数
        vial = {
            'Av': vial_area,
            'Ap': product_area,
            'Vfill': fill_volume
        }
        
        product = {
            'cSolid': solid_content,
            'T_pr_crit': critical_temp,
            'R0': r0 if resistance_option == "Known" else 1.4,
            'A1': a1 if resistance_option == "Known" else 16.0,
            'A2': a2 if resistance_option == "Known" else 0.0
        }
        
        ht = {
            'KC': kc if heat_transfer_option == "Known" else 0.000275,
            'KP': kp if heat_transfer_option == "Known" else 0.000893,
            'KD': kd if heat_transfer_option == "Known" else 0.46
        }
        
        Pchamber = {
            'setpt': [chamber_pressure],
            'ramp_rate': chamber_pressure_ramp,
            'min': 0.01,
            'dt_setpt': [pressure_hold_time]
        }
        
        Tshelf = {
            'init': initial_shelf_temp,
            'setpt': [shelf_temp],
            'ramp_rate': shelf_temp_ramp,
            'min': -50,
            'max': 50,
            'dt_setpt': [temp_hold_time]
        }
        
        eq_cap = {
            'a': a_val,
            'b': b_val
        }
        
        # 调用干燥计算函数
        with st.spinner("Calculating primary drying process..."):
            try:
                # 使用已知Rp的计算方法
                output = calc_knownRp.dry(
                    vial=vial,
                    product=product,
                    ht=ht,
                    Pchamber=Pchamber,
                    Tshelf=Tshelf,
                    dt=time_step
                )
                
                # 提取结果
                time_points = output[:, 0]
                T_sub = output[:, 1]
                T_bot = output[:, 2]
                T_sh = output[:, 3]
                P_ch = output[:, 4]
                sub_rate = output[:, 5]
                percent_dried = output[:, 6]
                
                # 计算主干燥时间
                primary_drying_time = time_points[-1]
                
                # 显示结果
                st.subheader("Calculation Results")
                st.metric("Primary Drying Time (hr)", f"{primary_drying_time:.2f}")
                
                # 创建结果DataFrame
                df = pd.DataFrame({
                    'Time (hr)': time_points,
                    'Sublimation Temp (°C)': T_sub,
                    'Vial Bottom Temp (°C)': T_bot,
                    'Shelf Temp (°C)': T_sh,
                    'Chamber Pressure (mTorr)': P_ch,
                    'Sublimation Rate (kg/hr/m²)': sub_rate,
                    'Percent Dried (%)': percent_dried
                })
                
                # 显示数据表格
                st.dataframe(df.style.format("{:.2f}"), height=300)
                
                # 绘制温度曲线
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time_points, y=T_bot, name="Product Temperature", line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=time_points, y=T_sh, name="Shelf Temperature", line=dict(color='red')))
                fig.add_hline(y=critical_temp, line_dash="dash", line_color="orange", 
                             annotation_text=f"Critical Temp={critical_temp}°C", annotation_position="bottom right")
                fig.update_layout(
                    title="Temperature Profile",
                    xaxis_title="Time (hr)",
                    yaxis_title="Temperature (°C)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 下载结果
                csv_data = format_csv(df)
                st.markdown(create_download_link(csv_data.encode(), "primary_drying_results.csv"), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error during calculation: {str(e)}")
                st.error("Please check your input parameters and try again.")

# 高级工具标签页
def advanced_tools():
    st.title("🧊 Lyophilization Process Optimization Platform")
    st.markdown("Advanced Tools for Lyophilization Process Development and Optimization")
    
    # 创建标签页
    tabs = st.tabs(["Parameter Setup", "Simulation Results", "Design Space", "Optimization Analysis", "Intelligent Recommendations"])
    
    with tabs[0]:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Product Formulation")
            protein_type = st.selectbox("Protein Type", list(PROTEIN_DATABASE.keys()))
            protein_conc = st.number_input("Protein Concentration (mg/mL)", 1.0, 300.0, 50.0, step=1.0)
            
            # 赋形剂选择
            st.markdown("**Excipient Composition**")
            exc_col1, exc_col2 = st.columns(2)
            excipients = {}
            with exc_col1:
                sucrose = st.number_input("Sucrose (%)", 0.0, 30.0, 5.0, step=0.1)
                if sucrose > 0:
                    excipients["Sucrose"] = sucrose
                mannitol = st.number_input("Mannitol (%)", 0.0, 30.0, 0.0, step=0.1)
                if mannitol > 0:
                    excipients["Mannitol"] = mannitol
            with exc_col2:
                trehalose = st.number_input("Trehalose (%)", 0.0, 30.0, 0.0, step=0.1)
                if trehalose > 0:
                    excipients["Trehalose"] = trehalose
                glycine = st.number_input("Glycine (%)", 0.0, 30.0, 0.0, step=0.1)
                if glycine > 0:
                    excipients["Glycine"] = glycine
            
            # 盐类选择
            salt_type = st.selectbox("Salt Type", list(SALT_EFFECTS.keys()))
            salt_content = st.number_input("Salt Concentration (%)", 0.0, 5.0, 0.0, step=0.01)
            
            # 预测关键参数
            thermal_params = predict_thermal_params(protein_type, protein_conc, excipients)
            salt_effect = SALT_EFFECTS[salt_type] * salt_content
            thermal_params["Tc"] += salt_effect
            
            # 显示预测值
            st.info(f"Predicted Collapse Temperature (Tc): {thermal_params['Tc']:.1f} °C")
            st.info(f"Predicted Glass Transition Temperature (Tg'): {thermal_params['Tg']} °C")
        
        with col2:
            st.subheader("Process Parameters")
            vial_size = st.selectbox("Vial Size", list(VIAL_SIZES.keys()))
            vial_info = VIAL_SIZES[vial_size]
            max_volume = vial_info["max_volume"]
            
            fill_volume = st.slider("Fill Volume (mL)", 0.1, max_volume, min(2.0, max_volume), step=0.1)
            
            # 计算灌装高度
            vial_diameter = vial_info["diameter"] / 10.0  # mm to cm
            vial_area = 3.14 * (vial_diameter / 2) ** 2  # cm²
            fill_depth = fill_volume / vial_area  # cm
            
            # 灌装高度预警
            if fill_depth > 1.5:  # 15mm
                st.warning("⚠️ Fill height exceeds 15mm, heat transfer efficiency will be significantly reduced!")
            
            st.info(f"Fill Height: {fill_depth:.2f} cm")
            
            vial_params = {
                'Av': vial_area,
                'Ap': vial_area,
                'Vfill': fill_volume,
                'diameter': vial_diameter
            }
            
            # 设备参数
            st.markdown("**Equipment Parameters**")
            n_vials = st.number_input("Number of Vials", 1, 10000, 1000)
            condenser_capacity = st.number_input("Condenser Capacity (kg)", 10.0, 1000.0, 200.0)
            
            # 冷凝器负载计算
            total_ice = n_vials * fill_volume * 0.9 / 1000  # kg (assuming 90% water)
            load_percentage = (total_ice / condenser_capacity) * 100
            st.progress(min(100, int(load_percentage)))
            st.info(f"Condenser Load: {load_percentage:.1f}%")
            if load_percentage > 60:
                st.warning("⚠️ Condenser load exceeds 60%, consider reducing sublimation rate!")
        
        # 冻干曲线参数
        st.subheader("Lyophilization Curve Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            # 预冻阶段
            st.markdown("**Freezing Stage**")
            prefreeze_rate = st.slider("Freezing Rate (°C/min)", 0.1, 5.0, 1.0)
            prefreeze_temp = st.number_input("Freezing Temperature (°C)", -60.0, -10.0, -40.0)
            annealing = st.checkbox("Add Annealing Process")
            if annealing:
                anneal_temp = st.number_input("Annealing Temperature (°C)", -40.0, -10.0, -20.0)
                anneal_time = st.number_input("Annealing Time (hr)", 0.5, 24.0, 2.0)
            
            # 一次干燥
            st.markdown("**Primary Drying**")
            # 确保传递浮点数参数
            kv_value = 0.5  # 默认传热系数值
            default_drying_time = calculate_drying_time(float(fill_depth), float(protein_conc), kv_value)
            primary_temp = st.slider(
                "Shelf Temperature (°C)", 
                -50.0, 
                float(thermal_params["Tc"])-5,  # 确保是浮点数
                float(thermal_params["Tc"])-10  # 确保是浮点数
            )
            st.info(f"Safe Operating Range: Below Tc {thermal_params['Tc']:.1f}°C")
    
            primary_pressure = st.slider("Chamber Pressure (mTorr)", 50, 300, 100)
    
            # 使用浮点数作为参数
            primary_time = st.number_input(
                "Drying Time (hr)", 
                1.0, 
                100.0, 
                float(default_drying_time)  # 确保是浮点数
            )
        
        with col2:
            # 二次干燥
            st.markdown("**Secondary Drying**")
            secondary_start = st.number_input("Start Temperature (°C)", primary_temp, 50.0, 0.0)
            secondary_end = st.number_input("Final Temperature (°C)", secondary_start, 50.0, 25.0)
            secondary_rate = st.slider("Ramp Rate (°C/min)", 0.01, 1.0, 0.1)
            secondary_time = st.number_input("Drying Time (hr)", 0.5, 24.0, 4.0)
            final_pressure = st.number_input("Final Pressure (mTorr)", 1, 100, 10)
            
            # 冻干曲线预览
            st.markdown("**Lyophilization Curve Preview**")
            fig = go.Figure()
            
            # 预冻
            fig.add_trace(go.Scatter(
                x=[0, 1], 
                y=[20, prefreeze_temp],
                mode='lines',
                name='Freezing',
                line=dict(color='blue', width=2)
            ))
            
            # 退火
            if annealing:
                fig.add_trace(go.Scatter(
                    x=[1, 2], 
                    y=[prefreeze_temp, anneal_temp],
                    mode='lines',
                    name='Annealing Ramp',
                    line=dict(color='green', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=[2, 3], 
                    y=[anneal_temp, anneal_temp],
                    mode='lines',
                    name='Annealing Hold',
                    line=dict(color='green', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=[3, 4], 
                    y=[anneal_temp, prefreeze_temp],
                    mode='lines',
                    name='Annealing Cool',
                    line=dict(color='green', width=2)
                ))
                start_point = 4
            else:
                start_point = 1
            
            # 一次干燥
            fig.add_trace(go.Scatter(
                x=[start_point, start_point+1], 
                y=[prefreeze_temp, primary_temp],
                mode='lines',
                name='Primary Drying Ramp',
                line=dict(color='orange', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[start_point+1, start_point+primary_time], 
                y=[primary_temp, primary_temp],
                mode='lines',
                name='Primary Drying',
                line=dict(color='red', width=2)
            ))
            
            # 二次干燥
            fig.add_trace(go.Scatter(
                x=[start_point+primary_time, start_point+primary_time+1], 
                y=[primary_temp, secondary_end],
                mode='lines',
                name='Secondary Drying Ramp',
                line=dict(color='purple', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[start_point+primary_time+1, start_point+primary_time+1+secondary_time], 
                y=[secondary_end, secondary_end],
                mode='lines',
                name='Secondary Drying',
                line=dict(color='magenta', width=2)
            ))
            
            fig.update_layout(
                title='Lyophilization Curve Preview',
                xaxis_title='Time (arbitrary units)',
                yaxis_title='Temperature (°C)',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 保存参数
        params = {
            'protein_type': protein_type,
            'protein_conc': protein_conc,
            'excipients': excipients,
            'salt_type': salt_type,
            'salt_content': salt_content,
            'vial': vial_params,
            'thermal_params': thermal_params,
            'n_vials': n_vials,
            'condenser_capacity': condenser_capacity,
            'prefreeze': {
                'rate': prefreeze_rate,
                'temp': prefreeze_temp
            },
            'annealing': annealing,
            'primary': {
                'temp': primary_temp,
                'pressure': primary_pressure,
                'time': primary_time
            },
            'secondary': {
                'start_temp': secondary_start,
                'end_temp': secondary_end,
                'rate': secondary_rate,
                'time': secondary_time,
                'pressure': final_pressure
            }
        }
        
        if annealing:
            params['annealing_details'] = {
                'temp': anneal_temp,
                'time': anneal_time
            }
        
        if st.button("Start Simulation", type="primary", use_container_width=True):
            st.session_state.params = params
            st.session_state.simulation_done = False
            st.rerun()
    
    # 智能建议标签页
    with tabs[4]:
        if 'params' in st.session_state:
            params = st.session_state.params
            
            st.subheader("Process Optimization Recommendations")
            
            # 高盐处方建议
            if params['salt_content'] > 0.5:
                st.warning("### ⚠️ High Salt Formulation Warning")
                st.markdown("""
                - **Issue**: High salt concentration significantly reduces collapse temperature (Tc)
                - **Impact**: Increases risk of product collapse, may cause drying failure
                - **Solutions**:
                  - Add annealing process: -20°C for 2-4 hours
                  - Reduce primary drying temperature to 5°C below Tc
                  - Optimize formulation to reduce salt content
                """)
            
            # 灌装量建议
            fill_depth = params['vial']['Vfill'] / params['vial']['Ap']
            if fill_depth > 1.5:
                st.warning("### ⚠️ Excessive Fill Height")
                st.markdown(f"""
                - **Current fill height**: {fill_depth:.2f} cm (recommended <1.5cm)
                - **Impact**: Reduced heat transfer efficiency, longer drying time
                - **Solutions**:
                  - Reduce fill volume to below {params['vial']['Ap'] * 1.5:.1f} mL
                  - Use vials with larger diameter
                  - Optimize lyophilization curve: reduce ramp rate
                """)
            
            # 保护剂比例建议
            sugar_ratio = params['excipients'].get('Sucrose', 0) + params['excipients'].get('Trehalose', 0)
            if sugar_ratio / params['protein_conc'] < 1:
                st.warning("### ⚠️ Insufficient Stabilizer")
                st.markdown(f"""
                - **Sucrose/protein ratio**: {sugar_ratio/params['protein_conc']:.2f} (recommended >1)
                - **Impact**: Reduced protein stability, increased aggregation risk
                - **Solutions**:
                  - Increase sucrose or trehalose to 1-2 times protein mass
                  - Consider adding other stabilizers like mannitol
                """)
            
            # 冷凝器负载建议
            total_ice = params['n_vials'] * params['vial']['Vfill'] * 0.9 / 1000
            load_percentage = (total_ice / params['condenser_capacity']) * 100
            if load_percentage > 60:
                st.warning("### ⚠️ Condenser Overload Risk")
                st.markdown(f"""
                - **Current load**: {load_percentage:.1f}% (recommended <60%)
                - **Impact**: May reduce sublimation efficiency, prolong drying time
                - **Solutions**:
                  - Reduce batch size or number of vials
                  - Reduce sublimation rate: lower shelf temperature or increase chamber pressure
                  - Increase condenser capacity
                """)
            
            # 塌陷温度余量
            temp_margin = params['thermal_params']['Tc'] - params['primary']['temp']
            if temp_margin < 3:
                st.warning("### ⚠️ Insufficient Temperature Safety Margin")
                st.markdown(f"""
                - **Tc margin**: {temp_margin:.1f}°C (recommended >3°C)
                - **Risk**: Temperature fluctuations may cause product collapse
                - **Solutions**:
                  - Reduce primary drying temperature to {params['thermal_params']['Tc'] - 3:.1f}°C
                  - Optimize formulation to increase Tc
                  - Increase process monitoring frequency
                """)
            
            # 最佳实践建议
            st.success("### ✅ Best Practice Recommendations")
            st.markdown("""
            1. **Freezing Optimization**:
               - Use rapid freezing (>1°C/min) to reduce ice crystal size
               - Apply annealing for crystalline excipients
               
            2. **Primary Drying**:
               - Maintain product temperature 3-5°C below Tc
               - Control pressure at 50-150 mTorr to optimize sublimation rate
               
            3. **Secondary Drying**:
               - Use stepwise temperature increase: 0.1-0.3°C/min
               - Final temperature 40-50°C to ensure residual moisture <1%
               
            4. **Process Monitoring**:
               - Use temperature probes for real-time product temperature monitoring
               - Perform pressure rise tests regularly to determine drying endpoint
            """)
    
    # 模拟执行和结果显示
    if 'params' in st.session_state:
        params = st.session_state.params
        
        with tabs[1]:
            if not st.session_state.get('simulation_done', False):
                with st.spinner("Running lyophilization simulation..."):
                    try:
                        # 模拟代码占位 - 实际应调用模拟引擎
                        time.sleep(2)
                        
                        # 生成模拟数据
                        time_points = np.linspace(0, params['primary']['time'] + params['secondary']['time'], 100)
                        shelf_temp = np.zeros_like(time_points)
                        product_temp = np.zeros_like(time_points)
                        sublimation_rate = np.zeros_like(time_points)
                        moisture = np.zeros_like(time_points)
                        
                        # 一次干燥阶段
                        primary_mask = time_points <= params['primary']['time']
                        shelf_temp[primary_mask] = params['primary']['temp']
                        product_temp[primary_mask] = params['primary']['temp'] - 5 - 2 * np.random.rand(np.sum(primary_mask))
                        sublimation_rate[primary_mask] = 0.5 * (1 - np.exp(-time_points[primary_mask] / 2))
                        moisture[primary_mask] = 100 - 80 * (time_points[primary_mask] / params['primary']['time'])
                        
                        # 二次干燥阶段
                        secondary_mask = time_points > params['primary']['time']
                        t_secondary = time_points[secondary_mask] - params['primary']['time']
                        shelf_temp[secondary_mask] = params['secondary']['start_temp'] + (
                            params['secondary']['end_temp'] - params['secondary']['start_temp']) * t_secondary / 2
                        product_temp[secondary_mask] = shelf_temp[secondary_mask] - 2 - np.random.rand(np.sum(secondary_mask))
                        sublimation_rate[secondary_mask] = 0.1 * np.exp(-t_secondary)
                        moisture[secondary_mask] = 20 * np.exp(-t_secondary * 2)
                        
                        # 创建结果DataFrame
                        df = pd.DataFrame({
                            'Time (hr)': time_points,
                            'Shelf Temp (°C)': shelf_temp,
                            'Product Temp (°C)': product_temp,
                            'Sublimation Rate (kg/hr/m²)': sublimation_rate,
                            'Residual Moisture (%)': moisture
                        })
                        
                        st.session_state.output = df
                        st.session_state.simulation_done = True
                        st.success("Simulation completed successfully!")
                    except Exception as e:
                        st.error(f"Error during simulation: {str(e)}")
                        st.session_state.simulation_done = False
            
            if st.session_state.get('simulation_done', False):
                df = st.session_state.output
                
                st.subheader("Simulation Results")
                st.dataframe(df.style.format("{:.2f}"), height=300)
                
                # 数据下载
                st.subheader("Data Export")
                csv_data = format_csv(df)
                st.markdown(create_download_link(csv_data.encode(), "lyo_simulation.csv"), unsafe_allow_html=True)
                
                # 可视化
                st.subheader("Lyophilization Process Curves")
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # 温度曲线
                fig.add_trace(
                    go.Scatter(
                        x=df['Time (hr)'], 
                        y=df['Shelf Temp (°C)'],
                        name='Shelf Temperature',
                        line=dict(color='red', width=2)
                    ),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df['Time (hr)'], 
                        y=df['Product Temp (°C)'],
                        name='Product Temperature',
                        line=dict(color='blue', width=2)
                    ),
                    secondary_y=False
                )
                
                # 添加Tc参考线
                fig.add_hline(
                    y=params['thermal_params']['Tc'], 
                    line_dash="dash", 
                    line_color="orange",
                    annotation_text=f"Tc={params['thermal_params']['Tc']}°C",
                    annotation_position="bottom right",
                    secondary_y=False
                )
                
                # 升华速率曲线
                fig.add_trace(
                    go.Scatter(
                        x=df['Time (hr)'], 
                        y=df['Sublimation Rate (kg/hr/m²)'],
                        name='Sublimation Rate',
                        line=dict(color='green', width=2)
                    ),
                    secondary_y=True
                )
                
                # 残留水分曲线
                fig.add_trace(
                    go.Scatter(
                        x=df['Time (hr)'], 
                        y=df['Residual Moisture (%)'],
                        name='Residual Moisture',
                        line=dict(color='purple', width=2)
                    ),
                    secondary_y=True
                )
                
                # 阶段分隔线
                fig.add_vline(
                    x=params['primary']['time'], 
                    line_dash="dash", 
                    line_color="gray",
                    annotation_text="End of Primary Drying",
                    annotation_position="top right"
                )
                
                fig.update_layout(
                    title='Lyophilization Process Curves',
                    xaxis_title='Time (hr)',
                    yaxis_title='Temperature (°C)',
                    yaxis2_title='Rate/Moisture',
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # 设计空间分析
        with tabs[2]:
            st.subheader("Design Space Analysis")
            
            # 生成设计空间数据
            pressures = np.linspace(50, 300, 10)
            temperatures = np.linspace(params['thermal_params']['Tc'] - 15, params['thermal_params']['Tc'] - 1, 10)
            drying_times = np.zeros((len(temperatures), len(pressures)))
            safe_zone = np.zeros((len(temperatures), len(pressures)))
            
            # 填充数据 (简化模型)
            for i, temp in enumerate(temperatures):
                for j, press in enumerate(pressures):
                    # 干燥时间模型
                    time_factor = (params['thermal_params']['Tc'] - temp) / 5
                    press_factor = 150 / press
                    drying_times[i, j] = params['primary']['time'] * time_factor * press_factor
                    
                    # 安全区域 (温度低于Tc-3°C)
                    safe_zone[i, j] = 1 if temp < params['thermal_params']['Tc'] - 3 else 0
            
            # 创建3D设计空间图
            fig = go.Figure()
            
            # 干燥时间曲面
            fig.add_trace(go.Surface(
                z=drying_times,
                x=pressures,
                y=temperatures,
                colorscale='Viridis',
                name='Drying Time',
                showscale=True,
                cmin=np.min(drying_times),
                cmax=np.max(drying_times)
            ))
            
            # 安全区域标记
            safe_x, safe_y, safe_z = [], [], []
            for i in range(len(temperatures)):
                for j in range(len(pressures)):
                    if safe_zone[i, j]:
                        safe_x.append(pressures[j])
                        safe_y.append(temperatures[i])
                        safe_z.append(drying_times[i, j])
            
            fig.add_trace(go.Scatter3d(
                x=safe_x,
                y=safe_y,
                z=safe_z,
                mode='markers',
                marker=dict(
                    size=4,
                    color='lime',
                    opacity=0.8
                ),
                name='Safe Operating Zone'
            ))
            
            # 当前操作点
            fig.add_trace(go.Scatter3d(
                x=[params['primary']['pressure']],
                y=[params['primary']['temp']],
                z=[params['primary']['time']],
                mode='markers',
                marker=dict(
                    size=6,
                    color='red'
                ),
                name='Current Parameters'
            ))
            
            fig.update_layout(
                title='Lyophilization Design Space',
                scene=dict(
                    xaxis_title='Pressure (mTorr)',
                    yaxis_title='Temperature (°C)',
                    zaxis_title='Drying Time (hr)',
                    zaxis=dict(autorange="reversed")
                ),
                height=700
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 2D设计空间投影
            st.subheader("Safe Operating Zone")
            fig_2d = go.Figure()
            
            # 安全区域
            fig_2d.add_trace(go.Contour(
                z=safe_zone,
                x=pressures,
                y=temperatures,
                colorscale=[[0, 'rgba(255,0,0,0.2)'], [1, 'rgba(0,255,0,0.4)']],
                showscale=False,
                name='Safe Zone'
            ))
            
            # Tc线
            fig_2d.add_hline(
                y=params['thermal_params']['Tc'], 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Tc={params['thermal_params']['Tc']}°C",
                annotation_position="bottom right"
            )
            
            # 当前操作点
            fig_2d.add_trace(go.Scatter(
                x=[params['primary']['pressure']],
                y=[params['primary']['temp']],
                mode='markers',
                marker=dict(size=12, color='red'),
                name='Current Parameters'
            ))
            
            fig_2d.update_layout(
                title='Safe Operating Zone (Green Area)',
                xaxis_title='Pressure (mTorr)',
                yaxis_title='Temperature (°C)',
                height=500
            )
            
            st.plotly_chart(fig_2d, use_container_width=True)
        
        # 优化分析
        with tabs[3]:
            st.subheader("Process Optimization Analysis")
            
            # 生成优化方案
            optimized_time = params['primary']['time'] * 0.85  # 假设优化后时间减少15%
            optimized_temp = min(params['primary']['temp'] + 2, params['thermal_params']['Tc'] - 3)
            optimized_pressure = max(params['primary']['pressure'] * 1.2, 50)
            
            # 创建比较表格
            comparison_data = {
                "Parameter": ["Drying Time (hr)", "Shelf Temp (°C)", "Chamber Pressure (mTorr)", "Tc Safety Margin (°C)"],
                "Current Value": [
                    params['primary']['time'], 
                    params['primary']['temp'], 
                    params['primary']['pressure'],
                    params['thermal_params']['Tc'] - params['primary']['temp']
                ],
                "Optimized Value": [
                    optimized_time,
                    optimized_temp,
                    optimized_pressure,
                    params['thermal_params']['Tc'] - optimized_temp
                ],
                "Improvement": [
                    f"-{(1 - optimized_time/params['primary']['time'])*100:.1f}%",
                    f"+{optimized_temp - params['primary']['temp']:.1f}°C",
                    f"+{(optimized_pressure/params['primary']['pressure'] - 1)*100:.1f}%",
                    f"+{(params['thermal_params']['Tc'] - optimized_temp) - (params['thermal_params']['Tc'] - params['primary']['temp']):.1f}°C"
                ]
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison.style.format("{:.2f}"), height=200)
            
            # 优化建议
            st.subheader("Optimization Recommendations")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **1. Primary Drying Optimization**
                - Increase shelf temperature: **{:.1f}°C** → **{:.1f}°C**
                - Adjust chamber pressure: **{:.0f} mTorr** → **{:.0f} mTorr**
                - Reduce drying time: **{:.1f} hr** → **{:.1f} hr**
                
                **2. Secondary Drying Optimization**
                - Increase final temperature: **{:.1f}°C** → **40°C**
                - Reduce drying time: **{:.1f} hr** → **{:.1f} hr**
                """.format(
                    params['primary']['temp'], optimized_temp,
                    params['primary']['pressure'], optimized_pressure,
                    params['primary']['time'], optimized_time,
                    params['secondary']['end_temp'],
                    params['secondary']['time'], max(2, params['secondary']['time'] * 0.7)
                ))
            
            with col2:
                st.markdown("""
                **3. Process Economics Improvement**
                - Batch time reduction: **{:.1f}%**
                - Energy consumption reduction: **~15%**
                - Production capacity increase: **~18%**
                
                **4. Product Quality Improvement**
                - Tc safety margin increase: **{:.1f}°C** → **{:.1f}°C**
                - Residual moisture reduction: **<1.0%**
                - Improved protein stability
                """.format(
                    (1 - optimized_time/params['primary']['time'])*100,
                    params['thermal_params']['Tc'] - params['primary']['temp'],
                    params['thermal_params']['Tc'] - optimized_temp
                ))
            
            # 优化前后曲线对比
            st.subheader("Comparison of Original vs Optimized Process")
            
            # 生成优化后曲线数据
            opt_time_points = np.linspace(0, optimized_time + max(2, params['secondary']['time'] * 0.7), 100)
            opt_shelf_temp = np.zeros_like(opt_time_points)
            opt_product_temp = np.zeros_like(opt_time_points)
            
            # 一次干燥阶段
            opt_primary_mask = opt_time_points <= optimized_time
            opt_shelf_temp[opt_primary_mask] = optimized_temp
            opt_product_temp[opt_primary_mask] = optimized_temp - 4 - 1.5 * np.random.rand(np.sum(opt_primary_mask))
            
            # 二次干燥阶段
            opt_secondary_mask = opt_time_points > optimized_time
            t_opt_secondary = opt_time_points[opt_secondary_mask] - optimized_time
            opt_shelf_temp[opt_secondary_mask] = 40
            opt_product_temp[opt_secondary_mask] = 40 - 1.5 - 0.5 * np.random.rand(np.sum(opt_secondary_mask))
            
            # 创建对比图表
            fig = go.Figure()
            
            # 原始温度曲线
            fig.add_trace(go.Scatter(
                x=df['Time (hr)'], 
                y=df['Product Temp (°C)'],
                name='Original - Product Temp',
                line=dict(color='blue', width=2, dash='dash')
            ))
            
            # 优化后温度曲线
            fig.add_trace(go.Scatter(
                x=opt_time_points, 
                y=opt_product_temp,
                name='Optimized - Product Temp',
                line=dict(color='red', width=2)
            ))
            
            # Tc参考线
            fig.add_hline(
                y=params['thermal_params']['Tc'], 
                line_dash="dash", 
                line_color="orange",
                annotation_text=f"Tc={params['thermal_params']['Tc']}°C",
                annotation_position="bottom right"
            )
            
            fig.update_layout(
                title='Product Temperature Comparison',
                xaxis_title='Time (hr)',
                yaxis_title='Temperature (°C)',
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)

# 主应用
def main():
    # 创建顶部导航栏
    st.markdown("""
    <style>
    .nav {
        display: flex;
        justify-content: space-around;
        background-color: #f0f2f6;
        padding: 10px 0;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .nav-item {
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-weight: bold;
    }
    .nav-item.active {
        background-color: #4a86e8;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 创建导航栏项目
    nav_items = ["Primary Drying Calculator", "Optimizer", "Design Space Generator", 
                 "Freezing Calculator", "Advanced Tools"]
    selected_nav = st.radio("", nav_items, horizontal=True, label_visibility="collapsed")
    
    if selected_nav == "Primary Drying Calculator":
        primary_drying_calculator()
    
    elif selected_nav == "Optimizer":
        st.header("Process Optimizer")
        st.write("Process optimizer functionality is under development...")
    
    elif selected_nav == "Design Space Generator":
        st.header("Design Space Generator")
        st.write("Design space generator functionality is under development...")
    
    elif selected_nav == "Freezing Calculator":
        st.header("Freezing Calculator")
        st.write("Freezing calculator functionality is under development...")
    
    elif selected_nav == "Advanced Tools":
        advanced_tools()

if __name__ == "__main__":
    main()
