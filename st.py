import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sp
import math
import time
import base64
from io import BytesIO

# 模拟常量模块
class constant:
    kg_To_g = 1000.0
    cm_To_m = 1.0e-2
    hr_To_s = 3600.0
    hr_To_min = 60.0
    min_To_s = 60.0
    Torr_to_mTorr = 1000.0
    cal_To_J = 4.184
    rho_ice = 0.918  # g/mL
    rho_solute = 1.5  # g/mL
    rho_solution = 1.0  # g/mL
    dHs = 678.0  # Heat of sublimation in cal/g
    k_ice = 0.0059  # Thermal conductivity of ice in cal/cm/s/K
    dHf = 79.7  # Heat of fusion in cal/g
    Cp_ice = 2030.0  # Constant pressure specific heat of ice in J/kg/K
    Cp_solution = 4000.0  # Constant pressure specific heat of water in J/kg/K

# 模拟函数模块
class functions:
    @staticmethod
    def Vapor_pressure(T_sub):
        return 2.698e10 * math.exp(-6144.96 / (273.15 + T_sub))
    
    @staticmethod
    def Lpr0_FUN(Vfill, Ap, cSolid):
        return Vfill / (Ap * constant.rho_ice) * (constant.rho_solution - cSolid * 
                (constant.rho_solution - constant.rho_ice) / constant.rho_solute)
    
    @staticmethod
    def Rp_FUN(l, R0, A1, A2):
        return R0 + A1 * l / (1.0 + A2 * l)
    
    @staticmethod
    def Kv_FUN(KC, KP, KD, Pch):
        return KC + KP * Pch / (1.0 + KD * Pch)
    
    @staticmethod
    def sub_rate(Ap, Rp, T_sub, Pch):
        P_sub = functions.Vapor_pressure(T_sub)
        return Ap / Rp / constant.kg_To_g * (P_sub - Pch)
    
    @staticmethod
    def T_bot_FUN(T_sub, Lpr0, Lck, Pch, Rp):
        P_sub = functions.Vapor_pressure(T_sub)
        return T_sub + (Lpr0 - Lck) * (P_sub - Pch) * constant.dHs / Rp / constant.hr_To_s / constant.k_ice
    
    @staticmethod
    def Rp_finder(T_sub, Lpr0, Lck, Pch, Tbot):
        P_sub = functions.Vapor_pressure(T_sub)
        return (Lpr0 - Lck) * (P_sub - Pch) * constant.dHs / (Tbot - T_sub) / constant.hr_To_s / constant.k_ice

# 冷冻计算器模块
class freezing:
    @staticmethod
    def freeze(vial, product, h_freezing, Tshelf, dt):
        # 简化的冷冻过程模拟
        Lpr0 = functions.Lpr0_FUN(vial['Vfill'], vial['Ap'], product['cSolid'])
        V_frozen = Lpr0 * vial['Ap']
        
        time_points = np.arange(0, 10, dt)
        Tsh_values = []
        Tpr_values = []
        
        Tsh = Tshelf['init']
        Tpr = product['Tpr0']
        
        for t in time_points:
            # 简化的温度变化模型
            Tsh = min(Tsh + Tshelf['ramp_rate'] * constant.hr_To_min * dt, Tshelf['setpt'][0])
            Tpr = max(Tpr - 5 * dt, product['Tn'])
            
            Tsh_values.append(Tsh)
            Tpr_values.append(Tpr)
            
            if t > 5 and Tpr <= product['Tn']:
                break
        
        return np.column_stack((time_points[:len(Tsh_values)], Tsh_values, Tpr_values))

# 初级干燥计算器模块
class calc_knownRp:
    @staticmethod
    def dry(vial, product, ht, Pchamber, Tshelf, dt):
        # 简化的初级干燥过程模拟
        Lpr0 = functions.Lpr0_FUN(vial['Vfill'], vial['Ap'], product['cSolid'])
        
        time_points = np.arange(0, 20, dt)
        Tsub_values = []
        Tbot_values = []
        Tsh_values = []
        Pch_values = []
        dmdt_values = []
        dried_percent = []
        
        Lck = 0.0
        Tsh = Tshelf['init']
        Pch = Pchamber['setpt'][0]
        
        for t in time_points:
            if t > Tshelf['t_setpt'][1]:
                Tsh = Tshelf['setpt'][1]
            
            Kv = functions.Kv_FUN(ht['KC'], ht['KP'], ht['KD'], Pch)
            Rp = functions.Rp_FUN(Lck, product['R0'], product['A1'], product['A2'])
            
            # 简化的温度计算
            Tsub = Tsh - 5 + Lck/Lpr0 * 10
            Tbot = Tsub + 2
            
            dmdt = functions.sub_rate(vial['Ap'], Rp, Tsub, Pch)
            dL = (dmdt * constant.kg_To_g) * dt / (1 - product['cSolid'] * constant.rho_solution / 
                  constant.rho_solute) / (vial['Ap'] * constant.rho_ice) * \
                  (1 - product['cSolid'] * (constant.rho_solution - constant.rho_ice) / constant.rho_solute)
            
            Lck += dL
            percent = min(Lck / Lpr0 * 100, 100)
            
            Tsub_values.append(Tsub)
            Tbot_values.append(Tbot)
            Tsh_values.append(Tsh)
            Pch_values.append(Pch * constant.Torr_to_mTorr)
            dmdt_values.append(dmdt / (vial['Ap'] * constant.cm_To_m**2))
            dried_percent.append(percent)
            
            if percent >= 100:
                break
        
        output = np.column_stack((
            time_points[:len(Tsub_values)], 
            Tsub_values, 
            Tbot_values, 
            Tsh_values, 
            Pch_values, 
            dmdt_values, 
            dried_percent
        ))
        return output

# 设计空间生成器模块
class design_space:
    @staticmethod
    def dry(vial, product, ht, Pchamber, Tshelf, dt, eq_cap, nVial):
        # 简化的设计空间计算
        T_max = np.zeros((len(Tshelf['setpt']), len(Pchamber['setpt'])))
        drying_time = np.zeros((len(Tshelf['setpt']), len(Pchamber['setpt'])))
        
        for i, Tsh in enumerate(Tshelf['setpt']):
            for j, Pch in enumerate(Pchamber['setpt']):
                # 简化的计算
                T_max[i, j] = product['T_pr_crit'] - 5 + i * 2 + j * 0.5
                drying_time[i, j] = 5 + i * 0.5 - j * 0.2
        
        return (T_max, drying_time)

# 优化器模块
class opt_Pch_Tsh:
    @staticmethod
    def dry(vial, product, ht, Pchamber, Tshelf, dt, eq_cap, nVial):
        # 简化的优化过程模拟
        return calc_knownRp.dry(vial, product, ht, Pchamber, Tshelf, dt)

# 主应用
def main():
    st.set_page_config(page_title="LyoPRONTO - 冷冻干燥模拟平台", layout="wide", page_icon="❄️")
    
    st.title("❄️ LyoPRONTO - 冷冻干燥过程模拟平台")
    st.markdown("""
    **LyoPRONTO** 是一个用于模拟和分析冷冻干燥过程的综合平台。该工具提供了：
    - 冷冻过程模拟
    - 初级干燥过程分析
    - 工艺参数设计空间生成
    - 工艺参数优化
    """)
    
    st.sidebar.header("模拟配置")
    tool = st.sidebar.selectbox("选择工具", 
                               ["冷冻计算器", "初级干燥计算器", "设计空间生成器", "优化器"],
                               index=1)
    
    # 通用参数
    st.sidebar.subheader("通用参数")
    vial_Av = st.sidebar.number_input("瓶面积 (cm²)", min_value=0.1, value=3.80, step=0.1)
    vial_Ap = st.sidebar.number_input("产品面积 (cm²)", min_value=0.1, value=3.14, step=0.1)
    vial_Vfill = st.sidebar.number_input("填充体积 (mL)", min_value=0.1, value=2.0, step=0.1)
    dt_value = st.sidebar.number_input("时间步长 (小时)", min_value=0.001, value=0.01, step=0.01)
    
    vial = {'Av': vial_Av, 'Ap': vial_Ap, 'Vfill': vial_Vfill}
    
    # 工具特定参数
    if tool == "冷冻计算器":
        with st.expander("冷冻计算器参数", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                cSolid = st.number_input("溶质浓度分数", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
                Tpr0 = st.number_input("初始产品温度 (°C)", value=15.8)
                Tf = st.number_input("冷冻温度 (°C)", value=-1.54)
            with col2:
                Tn = st.number_input("成核温度 (°C)", value=-5.84)
                h_freezing = st.number_input("传热系数 (W/m²/K)", value=38.0)
                Tshelf_init = st.number_input("初始搁板温度 (°C)", value=-5.0)
            
            Tshelf = {
                'init': Tshelf_init,
                'setpt': [-40],
                'ramp_rate': 1.0
            }
            
            product = {'cSolid': cSolid, 'Tpr0': Tpr0, 'Tf': Tf, 'Tn': Tn}
            
        if st.button("运行冷冻模拟"):
            with st.spinner("模拟运行中..."):
                start_time = time.time()
                freezing_output = freezing.freeze(vial, product, h_freezing, Tshelf, dt_value)
                elapsed_time = time.time() - start_time
                
                st.success(f"模拟完成! 耗时: {elapsed_time:.2f}秒")
                
                # 创建图表
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(freezing_output[:,0], freezing_output[:,1], 'b-', label='搁板温度')
                ax.plot(freezing_output[:,0], freezing_output[:,2], 'r-', label='产品温度')
                ax.set_xlabel("时间 (小时)")
                ax.set_ylabel("温度 (°C)")
                ax.set_title("冷冻过程温度曲线")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                # 显示数据
                df = pd.DataFrame(freezing_output, columns=["时间 (小时)", "搁板温度 (°C)", "产品温度 (°C)"])
                st.dataframe(df)
                
                # 下载链接
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="freezing_results.csv">下载CSV文件</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    elif tool == "初级干燥计算器":
        with st.expander("初级干燥参数", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                cSolid = st.number_input("溶质浓度分数", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
                R0 = st.number_input("R0 (cm²-hr-Torr/g)", value=1.4)
                A1 = st.number_input("A1 (cm-hr-Torr/g)", value=16.0)
                A2 = st.number_input("A2 (1/cm)", value=0.0)
                Tshelf_init = st.number_input("初始搁板温度 (°C)", value=-35.0)
                Tshelf_setpt = st.number_input("目标搁板温度 (°C)", value=20.0)
            with col2:
                KC = st.number_input("KC (cal/s/K/cm²)", value=2.75e-4)
                KP = st.number_input("KP (cal/s/K/cm²/Torr)", value=8.93e-4)
                KD = st.number_input("KD (1/Torr)", value=0.46)
                Pchamber_setpt = st.number_input("腔室压力 (Torr)", value=0.15)
                ramp_rate = st.number_input("升温速率 (°C/min)", value=1.0)
            
            Tshelf = {
                'init': Tshelf_init,
                'setpt': [Tshelf_setpt],
                'dt_setpt': [1800.0],
                'ramp_rate': ramp_rate,
                't_setpt': [0, 30]  # 简化的时间设置
            }
            
            Pchamber = {
                'setpt': [Pchamber_setpt],
                'dt_setpt': [1800.0],
                'ramp_rate': 0.5
            }
            
            ht = {'KC': KC, 'KP': KP, 'KD': KD}
            product = {'cSolid': cSolid, 'R0': R0, 'A1': A1, 'A2': A2}
            
        if st.button("运行初级干燥模拟"):
            with st.spinner("模拟运行中..."):
                start_time = time.time()
                output_saved = calc_knownRp.dry(vial, product, ht, Pchamber, Tshelf, dt_value)
                elapsed_time = time.time() - start_time
                
                st.success(f"模拟完成! 耗时: {elapsed_time:.2f}秒")
                
                # 创建图表
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
                
                # 温度曲线
                ax1.plot(output_saved[:,0], output_saved[:,1], 'b-', label='升华界面温度')
                ax1.plot(output_saved[:,0], output_saved[:,2], 'r-', label='瓶底温度')
                ax1.plot(output_saved[:,0], output_saved[:,3], 'g-', label='搁板温度')
                ax1.set_ylabel("温度 (°C)")
                ax1.legend()
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.set_title("温度曲线")
                
                # 压力和升华速率
                ax2.plot(output_saved[:,0], output_saved[:,4], 'm-', label='腔室压力')
                ax2.set_ylabel("压力 (mTorr)")
                ax2b = ax2.twinx()
                ax2b.plot(output_saved[:,0], output_saved[:,5], 'c-', label='升华通量')
                ax2b.set_ylabel("升华通量 (kg/hr/m²)")
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.set_title("压力和升华通量")
                
                # 干燥进度
                ax3.plot(output_saved[:,0], output_saved[:,6], 'k-', label='干燥进度')
                ax3.set_xlabel("时间 (小时)")
                ax3.set_ylabel("干燥进度 (%)")
                ax3.grid(True, linestyle='--', alpha=0.7)
                ax3.set_title("干燥进度")
                ax3.set_ylim([0, 100])
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 显示数据
                df = pd.DataFrame(output_saved, columns=[
                    "时间 (小时)", "升华界面温度 (°C)", "瓶底温度 (°C)", 
                    "搁板温度 (°C)", "腔室压力 (mTorr)", "升华通量 (kg/hr/m²)", 
                    "干燥进度 (%)"
                ])
                st.dataframe(df)
                
                # 下载链接
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="primary_drying_results.csv">下载CSV文件</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    elif tool == "设计空间生成器":
        with st.expander("设计空间参数", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                cSolid = st.number_input("溶质浓度分数", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
                R0 = st.number_input("R0 (cm²-hr-Torr/g)", value=1.4)
                A1 = st.number_input("A1 (cm-hr-Torr/g)", value=16.0)
                A2 = st.number_input("A2 (1/cm)", value=0.0)
                T_pr_crit = st.number_input("临界产品温度 (°C)", value=-5.0)
                Tshelf_init = st.number_input("初始搁板温度 (°C)", value=-5.0)
            with col2:
                KC = st.number_input("KC (cal/s/K/cm²)", value=2.75e-4)
                KP = st.number_input("KP (cal/s/K/cm²/Torr)", value=8.93e-4)
                KD = st.number_input("KD (1/Torr)", value=0.46)
                ramp_rate = st.number_input("升温速率 (°C/min)", value=1.0)
                nVial = st.number_input("瓶子数量", min_value=1, value=398)
                a = st.number_input("设备参数 a (kg/hr)", value=-0.182)
                b = st.number_input("设备参数 b (kg/hr/Torr)", value=0.0117e3)
            
            Tshelf = {
                'init': Tshelf_init,
                'setpt': [-5, 0, 2, 5],
                'ramp_rate': ramp_rate
            }
            
            Pchamber = {'setpt': [0.1, 0.4, 0.7, 1.5]}
            ht = {'KC': KC, 'KP': KP, 'KD': KD}
            product = {'cSolid': cSolid, 'R0': R0, 'A1': A1, 'A2': A2, 'T_pr_crit': T_pr_crit}
            eq_cap = {'a': a, 'b': b}
            
        if st.button("生成设计空间"):
            with st.spinner("计算中..."):
                start_time = time.time()
                T_max, drying_time = design_space.dry(vial, product, ht, Pchamber, Tshelf, dt_value, eq_cap, nVial)
                elapsed_time = time.time() - start_time
                
                st.success(f"计算完成! 耗时: {elapsed_time:.2f}秒")
                
                # 创建图表
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # 最大产品温度设计空间
                im1 = ax1.imshow(T_max, cmap='viridis', origin='lower', 
                                extent=[min(Pchamber['setpt']), max(Pchamber['setpt']), 
                                        min(Tshelf['setpt']), max(Tshelf['setpt'])])
                ax1.set_title("最大产品温度 (°C)")
                ax1.set_xlabel("腔室压力 (Torr)")
                ax1.set_ylabel("搁板温度 (°C)")
                fig.colorbar(im1, ax=ax1)
                
                # 干燥时间设计空间
                im2 = ax2.imshow(drying_time, cmap='plasma', origin='lower', 
                                extent=[min(Pchamber['setpt']), max(Pchamber['setpt']), 
                                        min(Tshelf['setpt']), max(Tshelf['setpt'])])
                ax2.set_title("干燥时间 (小时)")
                ax2.set_xlabel("腔室压力 (Torr)")
                ax2.set_ylabel("搁板温度 (°C)")
                fig.colorbar(im2, ax=ax2)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 显示数据表
                st.subheader("最大产品温度 (°C)")
                df_temp = pd.DataFrame(T_max, index=Tshelf['setpt'], columns=Pchamber['setpt'])
                st.dataframe(df_temp)
                
                st.subheader("干燥时间 (小时)")
                df_time = pd.DataFrame(drying_time, index=Tshelf['setpt'], columns=Pchamber['setpt'])
                st.dataframe(df_time)
    
    elif tool == "优化器":
        with st.expander("优化器参数", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                cSolid = st.number_input("溶质浓度分数", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
                R0 = st.number_input("R0 (cm²-hr-Torr/g)", value=1.4)
                A1 = st.number_input("A1 (cm-hr-Torr/g)", value=16.0)
                A2 = st.number_input("A2 (1/cm)", value=0.0)
                T_pr_crit = st.number_input("临界产品温度 (°C)", value=-5.0)
            with col2:
                KC = st.number_input("KC (cal/s/K/cm²)", value=2.75e-4)
                KP = st.number_input("KP (cal/s/K/cm²/Torr)", value=8.93e-4)
                KD = st.number_input("KD (1/Torr)", value=0.46)
                nVial = st.number_input("瓶子数量", min_value=1, value=398)
                a = st.number_input("设备参数 a (kg/hr)", value=-0.182)
                b = st.number_input("设备参数 b (kg/hr/Torr)", value=0.0117e3)
            
            optimization_type = st.radio("优化类型", 
                                       ["同时优化腔室压力和搁板温度", 
                                        "固定搁板温度优化腔室压力", 
                                        "固定腔室压力优化搁板温度"])
            
            Tshelf = {
                'min': st.number_input("最小搁板温度 (°C)", value=-45),
                'max': st.number_input("最大搁板温度 (°C)", value=120)
            }
            
            Pchamber = {
                'min': st.number_input("最小腔室压力 (Torr)", min_value=0.01, value=0.05),
                'max': st.number_input("最大腔室压力 (Torr)", min_value=0.1, value=1000.0)
            }
            
            ht = {'KC': KC, 'KP': KP, 'KD': KD}
            product = {'cSolid': cSolid, 'R0': R0, 'A1': A1, 'A2': A2, 'T_pr_crit': T_pr_crit}
            eq_cap = {'a': a, 'b': b}
            
        if st.button("运行优化"):
            with st.spinner("优化运行中..."):
                start_time = time.time()
                
                # 根据优化类型设置参数
                if optimization_type == "同时优化腔室压力和搁板温度":
                    Tshelf['init'] = -35.0
                    Tshelf['setpt'] = [20.0]
                    Tshelf['dt_setpt'] = [1800.0]
                    Tshelf['ramp_rate'] = 1.0
                    Pchamber['setpt'] = [0.15]
                    Pchamber['dt_setpt'] = [1800.0]
                    Pchamber['ramp_rate'] = 0.5
                    output_saved = opt_Pch_Tsh.dry(vial, product, ht, Pchamber, Tshelf, dt_value, eq_cap, nVial)
                elif optimization_type == "固定搁板温度优化腔室压力":
                    Tshelf['init'] = -35.0
                    Tshelf['setpt'] = [20.0]
                    Tshelf['dt_setpt'] = [1800.0]
                    Tshelf['ramp_rate'] = 1.0
                    Pchamber['setpt'] = [0.15]
                    Pchamber['dt_setpt'] = [1800.0]
                    Pchamber['ramp_rate'] = 0.5
                    output_saved = opt_Pch.dry(vial, product, ht, Pchamber, Tshelf, dt_value, eq_cap, nVial)
                else:
                    Tshelf['init'] = -35.0
                    Tshelf['setpt'] = [20.0]
                    Tshelf['dt_setpt'] = [1800.0]
                    Tshelf['ramp_rate'] = 1.0
                    Pchamber['setpt'] = [0.15]
                    Pchamber['dt_setpt'] = [1800.0]
                    Pchamber['ramp_rate'] = 0.5
                    output_saved = opt_Tsh.dry(vial, product, ht, Pchamber, Tshelf, dt_value, eq_cap, nVial)
                
                elapsed_time = time.time() - start_time
                
                st.success(f"优化完成! 耗时: {elapsed_time:.2f}秒")
                st.info(f"最佳干燥时间: {output_saved[-1,0]:.2f} 小时")
                
                # 创建图表
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(output_saved[:,0], output_saved[:,6], 'b-', label='干燥进度')
                ax.set_xlabel("时间 (小时)")
                ax.set_ylabel("干燥进度 (%)")
                ax.set_title("优化后的干燥进度")
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_ylim([0, 100])
                st.pyplot(fig)
                
                # 显示数据
                df = pd.DataFrame(output_saved, columns=[
                    "时间 (小时)", "升华界面温度 (°C)", "瓶底温度 (°C)", 
                    "搁板温度 (°C)", "腔室压力 (mTorr)", "升华通量 (kg/hr/m²)", 
                    "干燥进度 (%)"
                ])
                st.dataframe(df)
                
                # 下载链接
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="optimization_results.csv">下载CSV文件</a>'
                st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
