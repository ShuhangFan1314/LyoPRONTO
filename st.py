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
    page_title="生物制剂冻干工艺优化平台",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 理论模型参数
CRYSTALLINE_EXCIPIENTS = ["甘露醇", "甘氨酸", "山梨醇"]
AMORPHOUS_EXCIPIENTS = ["蔗糖", "海藻糖"]
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

# 蛋白类型关键参数数据库,需要进行调整
PROTEIN_DATABASE = {
    "单克隆抗体": {"Tc": -32, "Tg": -40, "R0": 1.8, "A1": 18.0},
    "疫苗": {"Tc": -28, "Tg": -35, "R0": 1.2, "A1": 12.0},
    "酶制剂": {"Tc": -35, "Tg": -42, "R0": 2.0, "A1": 20.0},
    "肽类": {"Tc": -30, "Tg": -38, "R0": 1.5, "A1": 15.0},
    "自定义": {"Tc": -30, "Tg": -40, "R0": 1.4, "A1": 16.0}
}

# 格式化输出CSV
def format_csv(data):
    csv = data.to_csv(index=False, encoding='utf-8')
    return csv

# 生成下载链接
def create_download_link(data, filename):
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">下载CSV文件</a>'
    return href

# 理论模型计算函数
def calculate_collapse_temp(protein_conc, excipients, salt_content):
    """计算塌陷温度(Tc)"""
    # 基础Tc值 (根据蛋白浓度)
    base_tc = -30 + 0.15 * protein_conc
    
    # 赋形剂影响
    excipient_effect = 0
    for exc, percent in excipients.items():
        if exc in CRYSTALLINE_EXCIPIENTS:
            excipient_effect += 0.2 * percent  # 结晶性赋形剂提升Tc
        elif exc in AMORPHOUS_EXCIPIENTS:
            excipient_effect += 0.05 * percent  # 非晶态赋形剂轻微提升
    
    # 盐类影响
    salt_effect = 0
    for salt, percent in salt_content.items():
        if salt != "None":
            salt_effect += SALT_EFFECTS[salt] * percent
    
    return base_tc + excipient_effect + salt_effect

def calculate_drying_time(fill_depth, protein_conc, kv):
    """预测干燥时间"""
    # 基础干燥时间 (小时)
    base_time = 20.0
    
    # 灌装高度影响 (非线性)
    if fill_depth > 1.0:
        height_factor = 1 + 0.5 * (fill_depth - 1.0) ** 1.5
    else:
        height_factor = 1.0
        
    # 蛋白浓度影响
    conc_factor = 1 + 0.02 * protein_conc
    
    # 传热系数影响 (确保kv在0-1之间)
    kv_factor = 1.5 - 0.5 * min(max(kv, 0.0), 1.0)  # kv值在0-1之间
    
    total_time = base_time * height_factor * conc_factor * kv_factor
    return float(total_time)

def predict_thermal_params(protein_type, protein_conc, excipients):
    """预测热力学参数"""
    base_params = PROTEIN_DATABASE[protein_type]
    
    # 蛋白浓度影响
    tc = base_params["Tc"] + 0.1 * protein_conc
    r0 = base_params["R0"] + 0.005 * protein_conc
    a1 = base_params["A1"] + 0.05 * protein_conc
    
    # 赋形剂影响
    for exc, percent in excipients.items():
        if exc in CRYSTALLINE_EXCIPIENTS:
            tc += 0.15 * percent
        elif exc in AMORPHOUS_EXCIPIENTS:
            # 非晶态赋形剂主要影响Tg
            pass
    
    return {
        "Tc": tc,
        "Tg": base_params["Tg"],
        "R0": r0,
        "A1": a1,
        "A2": 0.0
    }

# 主应用
def main():
    st.title("🧊 生物制剂冻干工艺优化平台")
    st.markdown("基于理论模型与AI的冻干工艺开发与优化系统")
    
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
    nav_items = ["Primary Drying Calculator", "Optimizer"]
    selected_nav = st.radio("", nav_items, horizontal=True, label_visibility="collapsed")
    
    if selected_nav == "Primary Drying Calculator":
        st.header("主干燥计算器")
        
        # 主体内容区 - 左右两栏布局
        col1, col2 = st.columns(2)
        
        # 左侧参数输入区（产品参数）
        with col1:
            st.subheader("产品参数")
            
            # 产品容器参数
            st.markdown("**产品容器参数**")
            vial_area = st.number_input("Vial Area (cm²)", value=3.8, format="%.4f")
            fill_volume = st.number_input("Fill Volume (mL)", value=2.0, format="%.2f")
            product_area = st.number_input("Product Area (cm²)", value=3.14, format="%.2f")
            critical_temp = st.number_input("Critical Product Temperature (°C)", value=-5.0, format="%.1f")
            solid_content = st.number_input("Solid Content (g/mL)", value=0.1, format="%.3f")
            
            # Vial Heat Transfer
            st.markdown("**Vial Heat Transfer**")
            heat_transfer_option = st.radio("选择", ["已知", "未知"], index=0, horizontal=True)
            if heat_transfer_option == "已知":
                kc = st.number_input("Kc", value=0.000275, format="%.6f")
                kp = st.number_input("Kp", value=0.000893, format="%.6f")
                kd = st.number_input("KD", value=0.46, format="%.2f")
            else:
                from_val = st.number_input("From", value=0.00106, format="%.5f")
                to_val = st.number_input("To", value=0.00108, format="%.5f")
                step_val = st.number_input("Step", value=-0.999999, format="%.6f")
            
            # 时间参数
            st.markdown("**时间参数**")
            time_step = st.number_input("Time Step (hr)", value=0.01, format="%.2f")
        
        # 右侧参数输入区（控制参数）
        with col2:
            st.subheader("控制参数")
            
            # Product Resistance
            st.markdown("**Product Resistance**")
            resistance_option = st.radio("选择", ["已知", "未知"], index=0, horizontal=True, key="resistance_option")
            if resistance_option == "已知":
                r0 = st.number_input("R₀", value=1.4, format="%.1f")
                a1 = st.number_input("A₁", value=16.0, format="%.1f")
                a2 = st.number_input("A₂", value=0.0, format="%.1f")
            else:
                uploaded_file = st.file_uploader("上传 Vial Bottom Temperature 文件 (temperature.txt)")
            
            # 初始条件
            st.markdown("**初始条件**")
            initial_shelf_temp = st.number_input("Initial Shelf Temperature (°C)", value=-35.0, format="%.1f")
            shelf_temp_ramp = st.number_input("Shelf Temperature Ramp Rate (°C/min)", value=1.0, format="%.1f")
            chamber_pressure_ramp = st.number_input("Chamber Pressure Ramp Rate (Torr/min)", value=0.5, format="%.1f")
            
            # 设备能力
            st.markdown("**设备能力**")
            a_val = st.number_input("a (kg/hr)", value=-0.182, format="%.3f")
            b_val = st.number_input("b (kg/(hr·Torr))", value=11.7, format="%.1f")
            
            # 其他参数
            st.markdown("**其他参数**")
            num_vials = st.number_input("Number of Vials", value=398)
        
        # 过程控制参数区（位于左右栏下方）
        st.subheader("过程控制参数")
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
            # 模拟计算主干燥时间（这里使用简化模型）
            primary_drying_time = 12.62  # 实际应使用冻干模型计算
            
            st.subheader("计算结果")
            st.metric("Primary Drying Time (hr)", f"{primary_drying_time:.2f}")
            
            # 生成温度曲线数据
            time_points = np.linspace(0, 24, 100)
            product_temp = 20 + 5 * np.sin(time_points * np.pi / 12)  # 模拟产品温度
            shelf_temp_curve = np.full_like(time_points, shelf_temp)   # 搁板温度
            
            # 绘制温度曲线
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_points, y=product_temp, name="Product Temperature"))
            fig.add_trace(go.Scatter(x=time_points, y=shelf_temp_curve, name="Shelf Temperature"))
            fig.update_layout(
                title="温度曲线",
                xaxis_title="时间 (小时)",
                yaxis_title="温度 (°C)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # 其他导航项（占位符）
    elif selected_nav == "Optimizer":
        st.header("工艺优化器")
        st.write("工艺优化器功能正在开发中...")
    
    elif selected_nav == "Design Space Generator":
        st.header("设计空间生成器")
        st.write("设计空间生成器功能正在开发中...")
    
    elif selected_nav == "Freezing Calculator":
        st.header("冷冻计算器")
        st.write("冷冻计算器功能正在开发中...")
    
    # 创建标签页（保留原有功能）
    tabs = st.tabs(["参数设置", "模拟结果", "设计空间", "优化分析", "智能建议"])
    
    with tabs[0]:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("产品配方")
            protein_type = st.selectbox("蛋白类型", list(PROTEIN_DATABASE.keys()))
            protein_conc = st.number_input("蛋白浓度 (mg/mL)", 1.0, 300.0, 50.0, step=1.0)
            
            # 赋形剂选择
            st.markdown("**赋形剂组成**")
            exc_col1, exc_col2 = st.columns(2)
            excipients = {}
            with exc_col1:
                sucrose = st.number_input("蔗糖 (%)", 0.0, 30.0, 5.0, step=0.1)
                if sucrose > 0:
                    excipients["蔗糖"] = sucrose
                mannitol = st.number_input("甘露醇 (%)", 0.0, 30.0, 0.0, step=0.1)
                if mannitol > 0:
                    excipients["甘露醇"] = mannitol
            with exc_col2:
                trehalose = st.number_input("海藻糖 (%)", 0.0, 30.0, 0.0, step=0.1)
                if trehalose > 0:
                    excipients["海藻糖"] = trehalose
                glycine = st.number_input("甘氨酸 (%)", 0.0, 30.0, 0.0, step=0.1)
                if glycine > 0:
                    excipients["甘氨酸"] = glycine
            
            # 盐类选择
            salt_type = st.selectbox("盐类", list(SALT_EFFECTS.keys()))
            salt_content = st.number_input("盐浓度 (%)", 0.0, 5.0, 0.0, step=0.01)
            
            # 预测关键参数
            thermal_params = predict_thermal_params(protein_type, protein_conc, excipients)
            salt_effect = SALT_EFFECTS[salt_type] * salt_content
            thermal_params["Tc"] += salt_effect
            
            # 显示预测值
            st.info(f"预测塌陷温度 (Tc): {thermal_params['Tc']:.1f} °C")
            st.info(f"预测玻璃化转变温度 (Tg'): {thermal_params['Tg']} °C")
        
        with col2:
            st.subheader("工艺参数")
            vial_size = st.selectbox("西林瓶规格", list(VIAL_SIZES.keys()))
            vial_info = VIAL_SIZES[vial_size]
            max_volume = vial_info["max_volume"]
            
            fill_volume = st.slider("灌装体积 (mL)", 0.1, max_volume, min(2.0, max_volume), step=0.1)
            
            # 计算灌装高度
            vial_diameter = vial_info["diameter"] / 10.0  # mm to cm
            vial_area = 3.14 * (vial_diameter / 2) ** 2  # cm²
            fill_depth = fill_volume / vial_area  # cm
            
            # 灌装高度预警
            if fill_depth > 1.5:  # 15mm
                st.warning("⚠️ 灌装高度超过15mm，传热效率将显著降低！")
            
            st.info(f"灌装高度: {fill_depth:.2f} cm")
            
            vial_params = {
                'Av': vial_area,
                'Ap': vial_area,
                'Vfill': fill_volume,
                'diameter': vial_diameter
            }
            
            # 设备参数
            st.markdown("**设备参数**")
            n_vials = st.number_input("西林瓶数量", 1, 10000, 1000)
            condenser_capacity = st.number_input("冷凝器容量 (kg)", 10.0, 1000.0, 200.0)
            
            # 冷凝器负载计算
            total_ice = n_vials * fill_volume * 0.9 / 1000  # kg (假设90%水)
            load_percentage = (total_ice / condenser_capacity) * 100
            st.progress(min(100, int(load_percentage)))
            st.info(f"冷凝器负载: {load_percentage:.1f}%")
            if load_percentage > 60:
                st.warning("⚠️ 冷凝器负载超过60%，建议降低升华速率！")
        
        # 冻干曲线参数
        st.subheader("冻干曲线参数")
        col1, col2 = st.columns(2)
        
        with col1:
            # 预冻阶段
            st.markdown("**预冻阶段**")
            prefreeze_rate = st.slider("预冻速率 (°C/min)", 0.1, 5.0, 1.0)
            prefreeze_temp = st.number_input("预冻温度 (°C)", -60.0, -10.0, -40.0)
            annealing = st.checkbox("添加退火工艺")
            if annealing:
                anneal_temp = st.number_input("退火温度 (°C)", -40.0, -10.0, -20.0)
                anneal_time = st.number_input("退火时间 (小时)", 0.5, 24.0, 2.0)
            
            # 一次干燥
            st.markdown("**一次干燥**")
            # 确保传递浮点数参数
            kv_value = 0.5  # 默认传热系数值
            default_drying_time = calculate_drying_time(float(fill_depth), float(protein_conc), kv_value)
            primary_temp = st.slider(
                "板层温度 (°C)", 
                -50.0, 
                float(thermal_params["Tc"])-5,  # 确保是浮点数
                float(thermal_params["Tc"])-10  # 确保是浮点数
            )
            st.info(f"安全操作范围: 低于Tc {thermal_params['Tc']:.1f}°C")
    
            primary_pressure = st.slider("腔室压力 (mTorr)", 50, 300, 100)
    
            # 使用浮点数作为参数
            primary_time = st.number_input(
                "干燥时间 (小时)", 
                1.0, 
                100.0, 
                float(default_drying_time)  # 确保是浮点数
            )
        
        with col2:
            # 二次干燥
            st.markdown("**二次干燥**")
            secondary_start = st.number_input("起始温度 (°C)", primary_temp, 50.0, 0.0)
            secondary_end = st.number_input("最终温度 (°C)", secondary_start, 50.0, 25.0)
            secondary_rate = st.slider("升温速率 (°C/min)", 0.01, 1.0, 0.1)
            secondary_time = st.number_input("二次干燥时间 (小时)", 0.5, 24.0, 4.0)
            final_pressure = st.number_input("最终压力 (mTorr)", 1, 100, 10)
            
            # 冻干曲线预览
            st.markdown("**冻干曲线预览**")
            fig = go.Figure()
            
            # 预冻
            fig.add_trace(go.Scatter(
                x=[0, 1], 
                y=[20, prefreeze_temp],
                mode='lines',
                name='预冻',
                line=dict(color='blue', width=2)
            ))
            
            # 退火
            if annealing:
                fig.add_trace(go.Scatter(
                    x=[1, 2], 
                    y=[prefreeze_temp, anneal_temp],
                    mode='lines',
                    name='退火升温',
                    line=dict(color='green', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=[2, 3], 
                    y=[anneal_temp, anneal_temp],
                    mode='lines',
                    name='退火保温',
                    line=dict(color='green', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=[3, 4], 
                    y=[anneal_temp, prefreeze_temp],
                    mode='lines',
                    name='退火降温',
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
                name='一次干燥升温',
                line=dict(color='orange', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[start_point+1, start_point+primary_time], 
                y=[primary_temp, primary_temp],
                mode='lines',
                name='一次干燥',
                line=dict(color='red', width=2)
            ))
            
            # 二次干燥
            fig.add_trace(go.Scatter(
                x=[start_point+primary_time, start_point+primary_time+1], 
                y=[primary_temp, secondary_end],
                mode='lines',
                name='二次干燥升温',
                line=dict(color='purple', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[start_point+primary_time+1, start_point+primary_time+1+secondary_time], 
                y=[secondary_end, secondary_end],
                mode='lines',
                name='二次干燥',
                line=dict(color='magenta', width=2)
            ))
            
            fig.update_layout(
                title='冻干曲线预览',
                xaxis_title='时间 (任意单位)',
                yaxis_title='温度 (°C)',
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
        
        if st.button("开始模拟", type="primary", use_container_width=True):
            st.session_state.params = params
            st.session_state.simulation_done = False
            st.rerun()
    
    # 智能建议标签页
    with tabs[4]:
        if 'params' in st.session_state:
            params = st.session_state.params
            
            st.subheader("工艺优化建议")
            
            # 高盐处方建议
            if params['salt_content'] > 0.5:
                st.warning("### ⚠️ 高盐处方警告")
                st.markdown("""
                - **问题**: 盐浓度过高会显著降低塌陷温度(Tc)
                - **影响**: 增加产品塌陷风险，可能导致干燥失败
                - **解决方案**:
                  - 添加退火工艺: -20°C保温2-4小时
                  - 降低一次干燥温度至Tc以下5°C
                  - 优化处方减少盐含量
                """)
            
            # 灌装量建议
            fill_depth = params['vial']['Vfill'] / params['vial']['Ap']
            if fill_depth > 1.5:
                st.warning("### ⚠️ 灌装高度过大")
                st.markdown(f"""
                - **当前灌装高度**: {fill_depth:.2f} cm (建议<1.5cm)
                - **影响**: 传热效率降低，干燥时间增加
                - **解决方案**:
                  - 减少灌装体积至{params['vial']['Ap'] * 1.5:.1f} mL以下
                  - 使用更大直径的西林瓶
                  - 优化冻干曲线: 降低升温速率
                """)
            
            # 保护剂比例建议
            sugar_ratio = params['excipients'].get('蔗糖', 0) + params['excipients'].get('海藻糖', 0)
            if sugar_ratio / params['protein_conc'] < 1:
                st.warning("### ⚠️ 保护剂不足")
                st.markdown(f"""
                - **蔗糖/蛋白比**: {sugar_ratio/params['protein_conc']:.2f} (建议>1)
                - **影响**: 蛋白稳定性降低，可能增加聚集风险
                - **解决方案**:
                  - 增加蔗糖或海藻糖比例至蛋白质量的1-2倍
                  - 考虑添加其他稳定剂如甘露醇
                """)
            
            # 冷凝器负载建议
            total_ice = params['n_vials'] * params['vial']['Vfill'] * 0.9 / 1000
            load_percentage = (total_ice / params['condenser_capacity']) * 100
            if load_percentage > 60:
                st.warning("### ⚠️ 冷凝器超载风险")
                st.markdown(f"""
                - **当前负载**: {load_percentage:.1f}% (建议<60%)
                - **影响**: 可能降低升华效率，延长干燥时间
                - **解决方案**:
                  - 减少批次数或每批瓶数
                  - 降低升华速率: 降低板层温度或提高腔室压力
                  - 增加冷凝器容量
                """)
            
            # 塌陷温度余量
            temp_margin = params['thermal_params']['Tc'] - params['primary']['temp']
            if temp_margin < 3:
                st.warning("### ⚠️ 温度安全余量不足")
                st.markdown(f"""
                - **Tc余量**: {temp_margin:.1f}°C (建议>3°C)
                - **风险**: 温度波动可能导致产品塌陷
                - **解决方案**:
                  - 降低一次干燥温度至{params['thermal_params']['Tc'] - 3:.1f}°C
                  - 优化处方提高Tc值
                  - 增加过程监控频率
                """)
            
            # 最佳实践建议
            st.success("### ✅ 最佳实践建议")
            st.markdown("""
            1. **预冻优化**:
               - 采用快速预冻(>1°C/min)减少冰晶大小
               - 对结晶性赋形剂使用退火工艺
               
            2. **一次干燥**:
               - 保持产品温度在Tc以下3-5°C
               - 压力控制在50-150mTorr优化升华速率
               
            3. **二次干燥**:
               - 采用阶梯升温: 0.1-0.3°C/min
               - 最终温度40-50°C确保残留水分<1%
               
            4. **过程监控**:
               - 使用温度探头实时监控产品温度
               - 定期进行压力升测试确定干燥终点
            """)
    
    # 模拟执行和结果显示
    if 'params' in st.session_state:
        params = st.session_state.params
        
        with tabs[1]:
            if not st.session_state.get('simulation_done', False):
                with st.spinner("正在进行冻干工艺模拟..."):
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
                            '时间 (小时)': time_points,
                            '板层温度 (°C)': shelf_temp,
                            '产品温度 (°C)': product_temp,
                            '升华速率 (kg/hr/m²)': sublimation_rate,
                            '残留水分 (%)': moisture
                        })
                        
                        st.session_state.output = df
                        st.session_state.simulation_done = True
                        st.success("模拟完成！")
                    except Exception as e:
                        st.error(f"模拟过程中出错: {str(e)}")
                        st.session_state.simulation_done = False
            
            if st.session_state.get('simulation_done', False):
                df = st.session_state.output
                
                st.subheader("模拟结果")
                st.dataframe(df.style.format("{:.2f}"), height=300)
                
                # 数据下载
                st.subheader("数据导出")
                csv_data = format_csv(df)
                st.markdown(create_download_link(csv_data.encode(), "lyo_simulation.csv"), unsafe_allow_html=True)
                
                # 可视化
                st.subheader("冻干过程曲线")
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # 温度曲线
                fig.add_trace(
                    go.Scatter(
                        x=df['时间 (小时)'], 
                        y=df['板层温度 (°C)'],
                        name='板层温度',
                        line=dict(color='red', width=2)
                    ),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df['时间 (小时)'], 
                        y=df['产品温度 (°C)'],
                        name='产品温度',
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
                        x=df['时间 (小时)'], 
                        y=df['升华速率 (kg/hr/m²)'],
                        name='升华速率',
                        line=dict(color='green', width=2)
                    ),
                    secondary_y=True
                )
                
                # 残留水分曲线
                fig.add_trace(
                    go.Scatter(
                        x=df['时间 (小时)'], 
                        y=df['残留水分 (%)'],
                        name='残留水分',
                        line=dict(color='purple', width=2)
                    ),
                    secondary_y=True
                )
                
                # 阶段分隔线
                fig.add_vline(
                    x=params['primary']['time'], 
                    line_dash="dash", 
                    line_color="gray",
                    annotation_text="一次干燥结束",
                    annotation_position="top right"
                )
                
                fig.update_layout(
                    title='冻干过程曲线',
                    xaxis_title='时间 (小时)',
                    yaxis_title='温度 (°C)',
                    yaxis2_title='速率/水分',
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
            st.subheader("设计空间分析")
            
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
                name='干燥时间',
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
                name='安全操作区'
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
                name='当前参数'
            ))
            
            fig.update_layout(
                title='冻干设计空间',
                scene=dict(
                    xaxis_title='压力 (mTorr)',
                    yaxis_title='温度 (°C)',
                    zaxis_title='干燥时间 (小时)',
                    zaxis=dict(autorange="reversed")
                ),
                height=700
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 2D设计空间投影
            st.subheader("安全操作区域")
            fig_2d = go.Figure()
            
            # 安全区域
            fig_2d.add_trace(go.Contour(
                z=safe_zone,
                x=pressures,
                y=temperatures,
                colorscale=[[0, 'rgba(255,0,0,0.2)'], [1, 'rgba(0,255,0,0.4)']],
                showscale=False,
                name='安全区域'
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
                name='当前参数'
            ))
            
            fig_2d.update_layout(
                title='安全操作区域 (绿色区域)',
                xaxis_title='压力 (mTorr)',
                yaxis_title='温度 (°C)',
                height=500
            )
            
            st.plotly_chart(fig_2d, use_container_width=True)
        
        # 优化分析
        with tabs[3]:
            st.subheader("工艺优化分析")
            
            # 生成优化方案
            optimized_time = params['primary']['time'] * 0.85  # 假设优化后时间减少15%
            optimized_temp = min(params['primary']['temp'] + 2, params['thermal_params']['Tc'] - 3)
            optimized_pressure = max(params['primary']['pressure'] * 1.2, 50)
            
            # 创建比较表格
            comparison_data = {
                "参数": ["干燥时间 (小时)", "板层温度 (°C)", "腔室压力 (mTorr)", "Tc安全余量 (°C)"],
                "当前值": [
                    params['primary']['time'], 
                    params['primary']['temp'], 
                    params['primary']['pressure'],
                    params['thermal_params']['Tc'] - params['primary']['temp']
                ],
                "优化值": [
                    optimized_time,
                    optimized_temp,
                    optimized_pressure,
                    params['thermal_params']['Tc'] - optimized_temp
                ],
                "改善": [
                    f"-{(1 - optimized_time/params['primary']['time'])*100:.1f}%",
                    f"+{optimized_temp - params['primary']['temp']:.1f}°C",
                    f"+{(optimized_pressure/params['primary']['pressure'] - 1)*100:.1f}%",
                    f"+{(params['thermal_params']['Tc'] - optimized_temp) - (params['thermal_params']['Tc'] - params['primary']['temp']):.1f}°C"
                ]
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison.style.format("{:.2f}"), height=200)
            
            # 优化建议
            st.subheader("优化建议方案")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **1. 一次干燥优化**
                - 提高板层温度: **{:.1f}°C** → **{:.1f}°C**
                - 调整腔室压力: **{:.0f} mTorr** → **{:.0f} mTorr**
                - 干燥时间减少: **{:.1f}小时** → **{:.1f}小时**
                
                **2. 二次干燥优化**
                - 提高最终温度: **{:.1f}°C** → **40°C**
                - 缩短干燥时间: **{:.1f}小时** → **{:.1f}小时**
                """.format(
                    params['primary']['temp'], optimized_temp,
                    params['primary']['pressure'], optimized_pressure,
                    params['primary']['time'], optimized_time,
                    params['secondary']['end_temp'],
                    params['secondary']['time'], max(2, params['secondary']['time'] * 0.7)
                ))
            
            with col2:
                st.markdown("""
                **3. 工艺经济性提升**
                - 批次时间减少: **{:.1f}%**
                - 能耗降低: **~15%**
                - 产能提升: **~18%**
                
                **4. 产品质量改善**
                - Tc安全余量增加: **{:.1f}°C** → **{:.1f}°C**
                - 残留水分降低: **<1.0%**
                - 蛋白稳定性提高
                """.format(
                    (1 - optimized_time/params['primary']['time'])*100,
                    params['thermal_params']['Tc'] - params['primary']['temp'],
                    params['thermal_params']['Tc'] - optimized_temp
                ))
            
            # 优化前后曲线对比
            st.subheader("优化前后曲线对比")
            
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
                x=df['时间 (小时)'], 
                y=df['产品温度 (°C)'],
                name='原始-产品温度',
                line=dict(color='blue', width=2, dash='dash')
            ))
            
            # 优化后温度曲线
            fig.add_trace(go.Scatter(
                x=opt_time_points, 
                y=opt_product_temp,
                name='优化-产品温度',
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
                title='优化前后产品温度对比',
                xaxis_title='时间 (小时)',
                yaxis_title='温度 (°C)',
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

if __name__ == "__main__":
    main()
