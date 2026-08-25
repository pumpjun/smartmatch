import streamlit as st
import json
import numpy as np
import colour
import itertools
import pandas as pd
from scipy.optimize import minimize, nnls
from scipy.interpolate import PchipInterpolator
import base64
import os
import re
import pyperclip

# ==========================================
# 0. 페이지 기본 설정 (항상 최상단에 위치)
# ==========================================
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="T/S Smart Match", page_icon="logo.png")

# ==========================================
# 1. 세션 상태 초기화
# ==========================================
if "dye_mode" not in st.session_state: st.session_state.dye_mode = "Reactive"
if "disperse_sub" not in st.session_state: st.session_state.disperse_sub = "Jersey"
if "selected_dyes" not in st.session_state: st.session_state.selected_dyes = []
if "top_results" not in st.session_state: st.session_state.top_results = None

if "l1" not in st.session_state: st.session_state.l1 = "D65"
if "l2" not in st.session_state: st.session_state.l2 = "없음"
if "l3" not in st.session_state: st.session_state.l3 = "없음"

def set_dye_mode(mode):
    if st.session_state.dye_mode != mode:
        st.session_state.dye_mode = mode
        st.session_state.selected_dyes = []
        st.session_state.top_results = None

def toggle_dye(raw_name):
    if raw_name in st.session_state.selected_dyes:
        st.session_state.selected_dyes.remove(raw_name)
    else:
        st.session_state.selected_dyes.append(raw_name)

def clear_dyes():
    st.session_state.selected_dyes = []

dye_mode = st.session_state.dye_mode

# ==========================================
# 2. 공통 UI 스타일 및 상단 고정 헤더 구성
# ==========================================
try:
    with open("logo.png", "rb") as image_file:
        logo_base64 = base64.b64encode(image_file.read()).decode()
except Exception:
    logo_base64 = ""

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
<style>
    #MainMenu {{visibility: hidden;}} header {{visibility: hidden;}} footer {{visibility: hidden;}}
    .fixed-header {{
        position: fixed; top: 0; left: 0; width: 100vw; height: 60px;
        background-color: #ffffff; box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
        z-index: 999998; display: flex; align-items: center; padding-left: 20px; border-bottom: 1px solid #eaeaea;
    }}
    .fixed-header img {{ width: 45px; margin-right: 12px; }}
    .fixed-header h2 {{ margin: 0; padding: 0; font-size: 24px; font-weight: 700; color: #31333F; }}
    .block-container {{ padding-top: 80px !important; }}
    .material-symbols-outlined {{ line-height: 1 !important; }}
    [data-testid="collapsedControl"] {{ display: none !important; }}
    [data-testid="stSidebar"] div.stButton {{ margin-bottom: -10px; }}
    div[data-testid="stHorizontalBlock"]:has(#top-menu-marker) {{
        position: fixed !important; top: 10px !important; left: 360px !important; 
        width: 820px !important; z-index: 999999 !important; align-items: center !important; 
    }}
    div.element-container:has(#top-menu-marker) {{ display: none !important; }}
    div[data-testid="stHorizontalBlock"]:has(#top-menu-marker) div[data-baseweb="select"] {{ border: none !important; background-color: transparent !important; box-shadow: none !important; cursor: pointer; }}
    div[data-testid="stHorizontalBlock"]:has(#top-menu-marker) div[data-baseweb="select"] * {{ color: #1f325c !important; font-weight: 700 !important; font-size: 15px !important; }}
    div[data-testid="stHorizontalBlock"]:has(#top-menu-marker) div.stButton > button {{ border-radius: 8px; padding: 0px 10px; height: 38px; min-height: 38px; margin: 0 !important; }}
</style>
<div class="fixed-header">
    <img src="data:image/png;base64,{logo_base64}" onerror="this.style.display='none'">
    <h2>T/S Smart Match</h2>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 3. 데이터 로드 및 매핑
# ==========================================
def apply_dc_correction(light_name, de_val):
    if "TL84" in light_name:
        if de_val <= 2.0: return max(0.01, -0.1226 * (de_val**2) + 0.6539 * de_val + 0.1873)
        else: return max(0.01, 1.0047 + 0.1635 * (de_val - 2.0))
    return de_val

@st.cache_data
def load_dye_data(mode):
    file_map = {"Reactive": 'dye_data.json', "Disperse": 'dye_data_disperse.json', "Reactive (CPB)": 'dye_data_cpb.json', "CDP": 'dye_data_CDP.json', "Acid": 'dye_data_acid.json'}
    try:
        with open(file_map.get(mode, 'dye_data.json'), 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            return {name.strip(): concs for name, concs in raw_data.items() if len(concs) > 0}
    except Exception: return {}

@st.cache_data
def load_dye_mapping(mode, _valid_keys):
    file_map = {"Reactive": 'dye_list.xlsx', "Disperse": 'dis_dye_list.xlsx', "Reactive (CPB)": 'cpb_dye_list.xlsx', "CDP": 'CDP_dye_list.xlsx', "Acid": 'acid_dye_list.xlsx'}
    try:
        df = pd.read_excel(file_map.get(mode, 'dye_list.xlsx'), header=None)
        mapping_list, disp_dict, missing_dyes, sort_order_dict = [], {}, [], {}
        for _, row in df.iterrows():
            try: sort_val = float(row[0]) if pd.notna(row[0]) else 999.0
            except: sort_val = 999.0
            raw_name, display_name = str(row[1]).strip(), str(row[2]).strip()
            if raw_name in _valid_keys:
                mapping_list.append((raw_name, display_name))
                disp_dict[raw_name] = display_name
                sort_order_dict[raw_name] = sort_val
            else: missing_dyes.append(raw_name)
        return mapping_list, disp_dict, missing_dyes, sort_order_dict
    except Exception: return [(k, k) for k in sorted(list(_valid_keys))], {k: k for k in _valid_keys}, [], {}

dye_db = load_dye_data(dye_mode)
all_dyes_ordered, display_name_dict, missing_dyes, sort_order_dict = load_dye_mapping(dye_mode, dye_db.keys())

# --- 광원 가중치 데이터 ---
wls_astm = np.arange(360, 790, 10)

astm_a_x_vals = [0.000, 0.000, 0.000, 0.002, 0.025, 0.134, 0.377, 0.686, 0.964, 1.080, 1.006, 0.731, 0.343, 0.078, 0.022, 0.218, 0.750, 1.642, 2.842, 4.336, 6.200, 8.262, 10.227, 11.945, 12.746, 12.337, 10.817, 8.560, 6.014, 3.887, 2.309, 1.276, 0.666, 0.336, 0.166, 0.082, 0.040, 0.020, 0.010, 0.005, 0.003, 0.001, 0.001]
astm_a_y_vals = [0.000, 0.000, 0.000, 0.000, 0.003, 0.014, 0.039, 0.084, 0.156, 0.259, 0.424, 0.696, 1.082, 1.616, 2.422, 3.529, 4.840, 6.100, 7.250, 8.114, 8.758, 8.988, 8.760, 8.304, 7.468, 6.323, 5.033, 3.744, 2.506, 1.560, 0.911, 0.499, 0.259, 0.130, 0.065, 0.032, 0.016, 0.008, 0.004, 0.002, 0.001, 0.001, 0.000]
astm_a_z_vals = [0.000, 0.000, 0.000, 0.008, 0.110, 0.615, 1.792, 3.386, 4.944, 5.806, 5.812, 4.919, 3.300, 1.973, 1.152, 0.658, 0.382, 0.211, 0.102, 0.032, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
custom_a_X = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_a_x_vals) / 100.0)), name='ASTM_A_X')
custom_a_Y = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_a_y_vals) / 100.0)), name='ASTM_A_Y')
custom_a_Z = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_a_z_vals) / 100.0)), name='ASTM_A_Z')

astm_d65_x_vals = [0.000, 0.000, 0.001, 0.005, 0.097, 0.616, 1.660, 2.377, 3.512, 3.789, 3.103, 1.937, 0.747, 0.110, 0.007, 0.314, 1.027, 2.174, 3.380, 4.735, 6.081, 7.310, 8.393, 8.603, 8.771, 7.996, 6.476, 4.635, 3.074, 1.814, 1.031, 0.557, 0.261, 0.114, 0.057, 0.028, 0.011, 0.006, 0.003, 0.001, 0.000, 0.000, 0.000]
astm_d65_y_vals = [0.000, 0.000, 0.000, 0.000, 0.010, 0.064, 0.171, 0.283, 0.549, 0.888, 1.277, 1.817, 2.545, 3.164, 4.309, 5.631, 6.896, 8.136, 8.684, 8.903, 8.614, 7.950, 7.164, 5.945, 5.110, 4.067, 2.990, 2.020, 1.275, 0.724, 0.407, 0.218, 0.102, 0.044, 0.022, 0.011, 0.004, 0.002, 0.001, 0.000, 0.000, 0.000, 0.000]
astm_d65_z_vals = [0.000, -0.001, 0.004, 0.020, 0.436, 2.808, 7.868, 11.703, 17.958, 20.358, 17.861, 13.085, 7.510, 3.743, 2.003, 1.004, 0.529, 0.271, 0.116, 0.030, -0.003, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
custom_d65_X = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_d65_x_vals) / 100.0)), name='ASTM_D65_X')
custom_d65_Y = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_d65_y_vals) / 100.0)), name='ASTM_D65_Y')
custom_d65_Z = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_d65_z_vals) / 100.0)), name='ASTM_D65_Z')

astm_tl84_x_vals = [0.000, 0.000, 0.000, -0.010, 0.099, 0.182, 0.098, 2.796, 4.103, 1.534, 1.314, 0.681, 0.343, 0.176, 0.009, 0.034, 0.005, -0.145, 10.852, 12.320, 1.096, 1.157, 7.036, 8.982, 6.204, 26.264, 13.228, 3.797, 0.794, 0.481, 0.264, 0.084, 0.038, 0.023, 0.011, 0.014, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
astm_tl84_y_vals = [0.000, 0.000, 0.000, -0.001, 0.010, 0.019, 0.003, 0.372, 0.625, 0.388, 0.554, 0.578, 1.380, 2.955, 1.506, 0.564, 0.257, 0.170, 25.656, 24.661, 1.274, 1.214, 5.881, 6.382, 3.629, 13.321, 6.279, 1.631, 0.329, 0.192, 0.104, 0.033, 0.015, 0.009, 0.004, 0.005, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
astm_tl84_z_vals = [0.000, 0.000, 0.000, -0.044, 0.451, 0.829, 0.415, 13.964, 20.873, 8.310, 7.586, 4.498, 3.625, 3.789, 0.773, 0.074, 0.028, 0.027, 0.293, 0.148, -0.010, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
custom_tl84_X = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_tl84_x_vals) / 100.0)), name='ASTM_TL84_X')
custom_tl84_Y = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_tl84_y_vals) / 100.0)), name='ASTM_TL84_Y')
custom_tl84_Z = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_tl84_z_vals) / 100.0)), name='ASTM_TL84_Z')

astm_f02_x_vals = [0.000, 0.000, 0.001, -0.009, 0.133, 0.311, 0.310, 2.977, 4.074, 1.393, 1.402, 0.946, 0.401, 0.081, 0.019, 0.169, 0.543, 1.093, 3.562, 6.166, 7.209, 10.967, 14.182, 13.453, 11.997, 9.183, 6.075, 3.517, 1.767, 0.808, 0.339, 0.133, 0.049, 0.019, 0.007, 0.003, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
astm_f02_y_vals = [0.000, 0.000, 0.000, -0.001, 0.014, 0.032, 0.025, 0.395, 0.617, 0.354, 0.593, 0.900, 1.261, 1.671, 2.165, 2.764, 3.517, 4.262, 8.685, 11.838, 10.117, 11.867, 12.191, 9.357, 7.032, 4.707, 2.825, 1.537, 0.736, 0.324, 0.134, 0.052, 0.019, 0.007, 0.003, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
astm_f02_z_vals = [0.000, 0.000, -0.001, -0.041, 0.603, 1.425, 1.418, 14.861, 20.711, 7.553, 8.103, 6.363, 3.852, 2.039, 1.030, 0.515, 0.277, 0.154, 0.107, 0.055, -0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
custom_f02_X = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_f02_x_vals) / 100.0)), name='ASTM_F02_X')
custom_f02_Y = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_f02_y_vals) / 100.0)), name='ASTM_F02_Y')
custom_f02_Z = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_f02_z_vals) / 100.0)), name='ASTM_F02_Z')

astm_u3000_x_vals = [0.000, 0.000, 0.001, -0.017, 0.092, 0.135, -0.257, 2.069, 3.204, 0.062, 0.415, 0.198, 0.169, 0.155, -0.009, 0.021, 0.042, -1.201, 10.840, 11.869, 0.387, 1.128, 8.214, 11.944, 3.319, 38.861, 13.839, 4.211, 0.499, 0.616, 0.285, 0.080, 0.033, 0.029, 0.007, 0.021, -0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
astm_u3000_y_vals = [0.000, 0.000, 0.000, -0.002, 0.010, 0.015, -0.041, 0.286, 0.465, 0.057, 0.174, 0.076, 0.770, 2.519, 0.931, 0.270, 0.251, -2.280, 25.526, 23.924, -0.403, 1.412, 6.935, 8.375, 2.227, 19.551, 6.592, 1.737, 0.201, 0.245, 0.113, 0.031, 0.013, 0.011, 0.003, 0.008, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
astm_u3000_z_vals = [0.000, 0.000, 0.003, -0.077, 0.418, 0.624, -1.329, 10.392, 16.211, 0.482, 2.392, 1.212, 1.918, 3.301, 0.337, -0.006, 0.020, 0.002, 0.284, 0.145, -0.024, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
custom_u3000_X = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_u3000_x_vals) / 100.0)), name='ASTM_U3000_X')
custom_u3000_Y = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_u3000_y_vals) / 100.0)), name='ASTM_U3000_Y')
custom_u3000_Z = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_u3000_z_vals) / 100.0)), name='ASTM_U3000_Z')

ul3500_x_vals = [0.0, 0.0, 0.001, -0.013, 0.074, 0.129, -0.06, 1.922, 3.113, 0.608, 0.749, 0.391, 0.204, 0.171, -0.001, 0.035, 0.118, -1.131, 9.623, 13.034, 0.665, 0.883, 8.961, 13.446, 3.616, 30.592, 15.72, 2.992, 0.621, 0.489, 0.222, 0.095, 0.034, 0.022, 0.006, 0.017, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ul3500_y_vals = [0.0, 0.0, 0.0, 0.0, 0.008, 0.014, -0.017, 0.258, 0.461, 0.177, 0.312, 0.28, 0.778, 3.286, 1.667, 0.472, 0.679, -2.022, 22.425, 26.094, 0.051, 1.14, 7.526, 9.48, 2.306, 15.347, 7.474, 1.204, 0.255, 0.193, 0.087, 0.037, 0.013, 0.008, 0.002, 0.007, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ul3500_z_vals = [0.0, 0.0, 0.002, -0.061, 0.336, 0.592, -0.373, 9.612, 15.792, 3.386, 4.32, 2.543, 2.182, 4.053, 0.763, -0.007, 0.051, 0.01, 0.238, 0.155, -0.023, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
custom_u35_X = colour.SpectralDistribution(dict(zip(wls_astm, np.array(ul3500_x_vals) / 100.0)), name='U35_X')
custom_u35_Y = colour.SpectralDistribution(dict(zip(wls_astm, np.array(ul3500_y_vals) / 100.0)), name='U35_Y')
custom_u35_Z = colour.SpectralDistribution(dict(zip(wls_astm, np.array(ul3500_z_vals) / 100.0)), name='U35_Z')

led35k_x_vals = [0.000, 0.000, -0.001, -0.003, 0.067, 0.502, 1.921, 3.207, 1.815, 0.549, 0.145, 0.030, -0.004, 0.177, 0.760, 1.755, 3.060, 4.666, 6.746, 9.120, 11.333, 13.126, 13.468, 12.086, 9.393, 6.322, 3.612, 1.837, 0.834, 0.343, 0.131, 0.048, 0.017, 0.006, 0.002, 0.001, -0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
led35k_y_vals = [0.000, 0.000, 0.000, -0.001, 0.005, 0.052, 0.297, 0.766, 0.758, 0.568, 0.535, 0.893, 1.891, 3.445, 5.184, 6.664, 7.921, 8.832, 9.607, 9.959, 9.706, 9.089, 7.838, 6.142, 4.333, 2.747, 1.495, 0.731, 0.328, 0.135, 0.051, 0.019, 0.007, 0.003, 0.001, -0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
led35k_z_vals = [0.000, 0.000, -0.005, -0.020, 0.301, 2.435, 9.825, 17.238, 10.477, 3.800, 1.544, 1.105, 0.922, 0.644, 0.403, 0.224, 0.107, 0.030, -0.002, -0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
custom_led35k_X = colour.SpectralDistribution(dict(zip(wls_astm, np.array(led35k_x_vals) / 100.0)), name='LED35K_X')
custom_led35k_Y = colour.SpectralDistribution(dict(zip(wls_astm, np.array(led35k_y_vals) / 100.0)), name='LED35K_Y')
custom_led35k_Z = colour.SpectralDistribution(dict(zip(wls_astm, np.array(led35k_z_vals) / 100.0)), name='LED35K_Z')

tl83_x_vals = [0.000, 0.000, 0.001, -0.023, 0.116, 0.220, -0.345, 2.825, 3.128, 0.088, 0.425, 0.200, 0.211, 0.173, -0.005, 0.025, -0.054, -0.727, 10.438, 11.409, 0.085, 0.105, 8.146, 11.772, 5.545, 39.852, 12.832, 4.259, 0.458, 0.509, 0.241, 0.077, 0.026, 0.028, 0.010, 0.020, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
tl83_y_vals = [0.000, 0.000, 0.000, -0.002, 0.012, 0.023, -0.051, 0.379, 0.455, 0.063, 0.179, 0.063, 0.891, 2.936, 1.452, 0.247, -0.078, -1.095, 25.187, 22.505, -0.583, 0.238, 6.881, 8.368, 3.334, 20.260, 6.010, 1.784, 0.183, 0.202, 0.095, 0.030, 0.010, 0.011, 0.004, 0.008, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
tl83_z_vals = [0.000, 0.000, 0.005, -0.103, 0.524, 1.010, -1.757, 14.132, 15.823, 0.625, 2.450, 1.207, 2.328, 3.766, 0.610, -0.034, 0.006, 0.018, 0.306, 0.122, -0.020, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
custom_tl83_X = colour.SpectralDistribution(dict(zip(wls_astm, np.array(tl83_x_vals) / 100.0)), name='TL83_X')
custom_tl83_Y = colour.SpectralDistribution(dict(zip(wls_astm, np.array(tl83_y_vals) / 100.0)), name='TL83_Y')
custom_tl83_Z = colour.SpectralDistribution(dict(zip(wls_astm, np.array(tl83_z_vals) / 100.0)), name='TL83_Z')

led_b1_x_vals = [0.000031, -0.000150, 0.000821, 0.002442, 0.062461, 0.258701, 1.035811, 2.049991, 1.240149, 0.519537, 0.176468, 0.038238, 0.000851, 0.138360, 0.580319, 1.404661, 2.610123, 4.229513, 6.502809, 9.286454, 12.038819, 14.566985, 15.432352, 14.243251, 11.357891, 7.900749, 4.640313, 2.465327, 1.155391, 0.498470, 0.198411, 0.075966, 0.028191, 0.010117, 0.003703, 0.001346, 0.000499, 0.000189, 0.000074, 0.000032, 0.000005]
led_b1_y_vals = [0.000003, -0.000014, 0.000081, 0.000270, 0.006370, 0.029938, 0.151953, 0.480331, 0.531082, 0.515223, 0.599201, 0.916287, 1.595039, 2.627616, 3.963429, 5.317968, 6.764593, 8.014659, 9.290737, 10.155148, 10.333176, 10.104026, 8.990918, 7.246063, 5.242298, 3.442575, 1.922385, 0.984252, 0.455777, 0.194748, 0.077209, 0.029507, 0.010942, 0.003930, 0.001441, 0.000525, 0.000195, 0.000074, 0.000029, 0.000013, 0.000002]
led_b1_z_vals = [0.000154, -0.000752, 0.003986, 0.010087, 0.295446, 1.262904, 5.264653, 11.005986, 7.185076, 3.555155, 1.772876, 1.137199, 0.766148, 0.485631, 0.310660, 0.180641, 0.093317, 0.028323, -0.002627, 0.000692, -0.000182, 0.000048, -0.000013, 0.000003, -0.000001, 0.000000, -0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
custom_led_b1_X = colour.SpectralDistribution(dict(zip(wls_astm, np.array(led_b1_x_vals) / 100.0)), name='LED_B1_X')
custom_led_b1_Y = colour.SpectralDistribution(dict(zip(wls_astm, np.array(led_b1_y_vals) / 100.0)), name='LED_B1_Y')
custom_led_b1_Z = colour.SpectralDistribution(dict(zip(wls_astm, np.array(led_b1_z_vals) / 100.0)), name='LED_B1_Z')

led_t8g_x_vals = [0.000, 0.000, -0.001, 0.001, 0.058, 0.352, 1.901, 4.418, 2.477, 0.806, 0.248, 0.056, -0.002, 0.240, 0.905, 1.943, 3.242, 4.768, 6.573, 8.403, 9.858, 10.737, 10.262, 11.835, 7.666, 9.501, 3.489, 0.999, 0.425, 0.167, 0.061, 0.021, 0.007, 0.003, 0.001, -0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
led_t8g_y_vals = [0.000, 0.000, -0.000, 0.000, 0.004, 0.029, 0.287, 1.057, 1.022, 0.822, 0.891, 1.543, 2.871, 4.526, 6.115, 7.342, 8.365, 9.001, 9.337, 9.156, 8.422, 7.426, 5.989, 5.998, 3.570, 4.117, 1.454, 0.389, 0.167, 0.065, 0.023, 0.008, 0.003, 0.001, -0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
led_t8g_z_vals = [0.000, 0.000, -0.003, -0.002, 0.267, 1.680, 9.698, 23.749, 14.273, 5.567, 2.607, 1.895, 1.374, 0.831, 0.468, 0.244, 0.112, 0.030, -0.002, -0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
custom_led_t8g_X = colour.SpectralDistribution(dict(zip(wls_astm, np.array(led_t8g_x_vals) / 100.0)), name='LED_T8G_X')
custom_led_t8g_Y = colour.SpectralDistribution(dict(zip(wls_astm, np.array(led_t8g_y_vals) / 100.0)), name='LED_T8G_Y')
custom_led_t8g_Z = colour.SpectralDistribution(dict(zip(wls_astm, np.array(led_t8g_z_vals) / 100.0)), name='LED_T8G_Z')

LIGHT_MAP = {
    "D65": (custom_d65_X, custom_d65_Y, custom_d65_Z),
    "A": (custom_a_X, custom_a_Y, custom_a_Z),
    "CWF (F02)": (custom_f02_X, custom_f02_Y, custom_f02_Z),
    "TL84 (F11)": (custom_tl84_X, custom_tl84_Y, custom_tl84_Z),
    "TL83": (custom_tl83_X, custom_tl83_Y, custom_tl83_Z),
    "U3000 (F12)": (custom_u3000_X, custom_u3000_Y, custom_u3000_Z),
    "U3500": (custom_u35_X, custom_u35_Y, custom_u35_Z),
    "LED35K": (custom_led35k_X, custom_led35k_Y, custom_led35k_Z),
    "LED_B1": (custom_led_b1_X, custom_led_b1_Y, custom_led_b1_Z),
    "LED_T8G": (custom_led_t8g_X, custom_led_t8g_Y, custom_led_t8g_Z)
}

def get_ks(reflectance): return (1 - reflectance)**2 / (2 * reflectance)

blank_r_str_reactive = "61.487896,64.536758,67.636276,70.483246,73.516251,75.622711,77.759293,79.583626,80.990044,82.235336,83.458176,84.331772,85.404106,86.164101,86.926323,87.612724,88.086739,88.541801,88.927353,89.348244,89.645943,89.882187,90.113014,90.397278,90.583130,90.746536,90.858932,91.020134,91.199127,91.403587,91.537102,91.670677,91.884819,91.980095,92.083275"
blank_r_reactive = np.array([float(x.strip()) / 100.0 for x in blank_r_str_reactive.split(',') if x.strip()])
blank_ks = get_ks(blank_r_reactive) 

# ==========================================
# 4. 파싱 및 계산 로직
# ==========================================
def parse_qtx_blocks(content):
    standards = []
    blocks = re.split(r'\[(STANDARD_DATA|BATCH_DATA)[^\]]*\]', content)
    for i in range(1, len(blocks), 2):
        block_type = blocks[i]
        block_content = blocks[i+1]
        prefix = "STD_" if block_type == 'STANDARD_DATA' else "BAT_"
        name_match = re.search(fr'{prefix}NAME=(.*?)\n', block_content)
        r_match = re.search(fr'{prefix}R=([\d\.,\s]+)', block_content)
        low_match = re.search(fr'{prefix}REFLLOW=(\d+)', block_content)
        if r_match:
            name = name_match.group(1).strip().rstrip(',') if name_match else "Unknown"
            r_vals = [float(x.strip()) / 100.0 for x in r_match.group(1).split(',') if x.strip()]
            start_wl = int(low_match.group(1)) if low_match else 400
            current_wls = np.array([start_wl + j * 10 for j in range(len(r_vals))])
            target_wls = np.arange(360, 710, 10)
            r_35 = np.interp(target_wls, current_wls, r_vals)
            standards.append({'type': block_type, 'name': name, 'r_35': r_35, 'ks_31': get_ks(r_35[4:35])})
    return standards

def calculate_lab_exact(r_31, light_name):
    shape_10nm = colour.SpectralShape(400, 700, 10)
    cmfs = colour.MSDS_CMFS['CIE 1964 10 Degree Standard Observer'].copy().align(shape_10nm)
    cmfs_values = cmfs.values
    light_data = LIGHT_MAP[light_name]
    
    if isinstance(light_data, tuple):
        W_X = light_data[0].copy().align(shape_10nm).values
        W_Y = light_data[1].copy().align(shape_10nm).values
        W_Z = light_data[2].copy().align(shape_10nm).values
        W = np.column_stack((W_X, W_Y, W_Z))
    else:
        light_spd = light_data.copy().align(shape_10nm)
        light_values = light_spd.values
        dw = 10
        k = np.sum(light_values * cmfs_values[:, 1]) * dw
        W = (light_values[:, np.newaxis] * cmfs_values) * dw / k 
        
    wp_XYZ = np.sum(W, axis=0) 
    wp_xy = colour.XYZ_to_xy(wp_XYZ)
    XYZ_tgt = np.dot(r_31, W)
    return colour.XYZ_to_Lab(XYZ_tgt, illuminant=wp_xy)

class DyePredictor:
    def __init__(self, concs, ks_matrix):
        self.concs = np.array(concs)
        self.max_c = self.concs[-1]
        self.ks_matrix = np.array(ks_matrix)
        self.max_ks = self.ks_matrix[-1]
        self.interpolator = PchipInterpolator(self.concs, self.ks_matrix, axis=0)
        
    def __call__(self, c):
        c = max(0.0, float(c))
        if c <= self.max_c: return self.interpolator(c)
        else: return self.max_ks * (c / self.max_c)

# 🌟 역산(역추적) 기반 정밀 보정 엔진
def calculate_multi_batch_correction(std_ks, bat_ks_list, bat_expected_recipes, bat_actual_recipes, blank_ks_arr, dye_predictors, mode):
    num_dyes = len(dye_predictors)
    
    def predict_recipe_for_ks(target_ks, ref_recipe):
        def objective(conc):
            est_ks = np.copy(blank_ks_arr)
            for j in range(num_dyes): est_ks += dye_predictors[j](conc[j])
            error = np.sum((target_ks - est_ks)**2)
            error += 0.05 * np.sum((conc - ref_recipe)**2)
            return error

        max_bound = 300.0 if mode == "Reactive (CPB)" else 30.0
        bnds = [(0.0, max_bound) for _ in range(num_dyes)]
        res = minimize(objective, x0=ref_recipe, bounds=bnds, method='SLSQP')
        return res.x

    cf_list = []
    calc_bat_recs = []
    
    for i, bat_ks in enumerate(bat_ks_list):
        actual_rec = np.array(bat_actual_recipes[i])
        calculated_bat_rec = predict_recipe_for_ks(bat_ks, actual_rec)
        calc_bat_recs.append(calculated_bat_rec)
        
        batch_cfs = []
        for j in range(num_dyes):
            if actual_rec[j] > 0.0001:
                cf = calculated_bat_rec[j] / actual_rec[j]
                cf = np.clip(cf, 0.3, 2.5) 
            else: cf = 1.0
            batch_cfs.append(cf)
        cf_list.append(batch_cfs)
        
    optimal_cf = np.mean(cf_list, axis=0) if cf_list else np.ones(num_dyes)

    std_expected_rec = np.array(bat_expected_recipes[0]) if len(bat_expected_recipes) > 0 else np.zeros(num_dyes)
    calc_std_rec = predict_recipe_for_ks(std_ks, std_expected_rec)
    
    final_recipe = []
    for j in range(num_dyes):
        final_val = calc_std_rec[j] / optimal_cf[j] if optimal_cf[j] > 0 else calc_std_rec[j]
        final_recipe.append(final_val)

    return {
        "success": True, 
        "calibration_factors": optimal_cf, 
        "final_recipe": final_recipe,
        "calc_std_rec": calc_std_rec,
        "calc_bat_recs": calc_bat_recs
    }

# ==========================================
# 4.5 백포 선택 팝업 (Disperse 전용) - 수정됨
# ==========================================
def set_temp_disp(val): 
    st.session_state.temp_disp = val

def confirm_disp_action():
    st.session_state.disperse_sub = st.session_state.temp_disp
    st.session_state.dye_mode = "Disperse"
    st.session_state.selected_dyes = []
    st.session_state.top_results = None

@st.dialog("백포 선택 (Disperse)")
def disperse_dialog():
    st.markdown("분산염료처방 탐색에 사용할 백포를 선택해주세요.")
    if "temp_disp" not in st.session_state: 
        st.session_state.temp_disp = st.session_state.disperse_sub
        
    col1, col2 = st.columns(2)
    with col1: 
        if st.button("Jersey", use_container_width=True, type="primary" if st.session_state.temp_disp == "Jersey" else "secondary", key="dlg_jersey_btn"):
            set_temp_disp("Jersey")
            st.rerun()
    with col2: 
        if st.button("Woven", use_container_width=True, type="primary" if st.session_state.temp_disp == "Woven" else "secondary", key="dlg_woven_btn"):
            set_temp_disp("Woven")
            st.rerun()
            
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    if st.button("확인", use_container_width=True, type="primary", key="dlg_confirm_btn"):
        confirm_disp_action()
        st.rerun()


# ==========================================
# 5. 상단 메뉴 및 좌측 사이드바 구성 - 수정됨
# ==========================================
top_menu_cols = st.columns([1, 1, 1.2, 1, 1])
with top_menu_cols[0]:
    if st.button("Reactive", use_container_width=True, type="primary" if dye_mode == "Reactive" else "secondary", key="top_reactive_btn"):
        set_dye_mode("Reactive")
        st.rerun()
    st.markdown('<div id="top-menu-marker"></div>', unsafe_allow_html=True)

with top_menu_cols[1]:
    # 🌟 Disperse 버튼을 누르면 즉시 팝업 함수 호출
    if st.button("Disperse", use_container_width=True, type="primary" if dye_mode == "Disperse" else "secondary", key="top_disperse_btn"):
        st.session_state.temp_disp = st.session_state.disperse_sub
        disperse_dialog()

with top_menu_cols[2]:
    if st.button("Reactive (CPB)", use_container_width=True, type="primary" if dye_mode == "Reactive (CPB)" else "secondary", key="top_cpb_btn"):
        set_dye_mode("Reactive (CPB)")
        st.rerun()

with top_menu_cols[3]:
    if st.button("CDP", use_container_width=True, type="primary" if dye_mode == "CDP" else "secondary", key="top_cdp_btn"):
        set_dye_mode("CDP")
        st.rerun()

with top_menu_cols[4]:
    if st.button("Acid", use_container_width=True, type="primary" if dye_mode == "Acid" else "secondary", key="top_acid_btn"):
        set_dye_mode("Acid")
        st.rerun()

with st.sidebar:
    st.markdown(f"<h3 style='display: flex; align-items: center;'><span class='material-symbols-outlined' style='margin-right:8px;'>palette</span>염료 리스트</h3>", unsafe_allow_html=True)
    if missing_dyes: st.warning(f"데이터 부족 제외 염료 {len(missing_dyes)}개", icon=":material/warning:")
    st.caption("클릭하여 선택 / 해제하세요.")
    st.markdown("---")
    
    # 🌟 오류 수정 구간: 2개씩만 언패킹하도록 올바르게 수정 완료!
    for idx, (raw_name, display_name) in enumerate(all_dyes_ordered):
        btn_type = "primary" if raw_name in st.session_state.selected_dyes else "secondary"
        st.button(display_name, key=f"dye_{raw_name}_{idx}", use_container_width=True, type=btn_type, on_click=toggle_dye, args=(raw_name,))

# ------------------------------------------
# 메인 화면 구성
# ------------------------------------------
col_menu, col_results = st.columns([1.2, 2], gap="large")

with col_menu:
    with st.container(border=True):
        st.markdown("<strong style='display: flex; align-items: center; font-size: 16px;'><span class='material-symbols-outlined' style='margin-right:6px;'>settings</span>광원 설정</strong>", unsafe_allow_html=True)
        light_options_all = ["D65", "A", "CWF (F02)", "TL84 (F11)", "TL83", "U3000 (F12)", "U3500", "LED35K", "LED_B1", "LED_T8G"]
        light_options_optional = ["없음"] + light_options_all
        
        l_col1, l_col2, l_col3 = st.columns(3)
        light1_name = l_col1.selectbox("1차 광원", light_options_all, key="l1", index=light_options_all.index("D65"))
        light2_name = l_col2.selectbox("2차 광원", light_options_optional, key="l2", index=light_options_optional.index("CWF (F02)")) 
        light3_name = l_col3.selectbox("3차 광원", light_options_optional, key="l3", index=light_options_optional.index("없음")) 

    with st.container(border=True):
        st.markdown("<strong style='display: flex; align-items: center; font-size: 16px;'><span class='material-symbols-outlined' style='margin-right:6px;'>folder_open</span>타겟 업로드 및 대조 입력</strong>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("QTX 파일 업로드 (STD & BAT 포함)", type=['qtx'], label_visibility="collapsed")
        
        edited_df = None
        if uploaded_file and len(st.session_state.selected_dyes) > 0:
            content = uploaded_file.getvalue().decode('euc-kr', errors='ignore')
            parsed_blocks = parse_qtx_blocks(content)
            
            standards = [b for b in parsed_blocks if b['type'] == 'STANDARD_DATA']
            batches = [b for b in parsed_blocks if b['type'] == 'BATCH_DATA']
            
            if standards and batches:
                st.success(f"🎯 타겟(STD): **{standards[0]['name']}**")
                all_bat_names = [b['name'] for b in batches]
                selected_bat_names = st.multiselect("대조 분석에 사용할 BATCH 선택", options=all_bat_names, default=[all_bat_names[0]])
                
                if selected_bat_names:
                    selected_raw_dyes = sorted(st.session_state.selected_dyes, key=lambda x: sort_order_dict.get(x, 999.0))
                    col_names = ["[STD] 예상 처방"] + [f"[BAT] {b_name} 실제 투입량" for b_name in selected_bat_names]
                    df_input = pd.DataFrame(0.0, index=[display_name_dict.get(d, d) for d in selected_raw_dyes], columns=col_names)
                    
                    unit_label = "g/l" if st.session_state.dye_mode == "Reactive (CPB)" else "%"
                    st.caption(f"※ 표에 처방({unit_label})을 입력한 후 빈 곳을 클릭하여 값을 확정하세요.")
                    edited_df = st.data_editor(df_input, use_container_width=True)
                    run_calc = st.button("🚀 역산 기반 정밀 보정 시작", type="primary", use_container_width=True)
                else:
                    st.warning("분석할 배치를 최소 1개 이상 선택해 주세요.")
                    run_calc = False
            else:
                st.error("QTX 파일에 STANDARD 또는 BATCH 데이터가 부족합니다.")
                run_calc = False
        else:
            if not uploaded_file: st.info("QTX 파일을 업로드 해주세요.")
            elif len(st.session_state.selected_dyes) == 0: st.warning("사이드바에서 처방에 사용된 염료를 선택해 주세요.")
            run_calc = False

with col_results:
    st.markdown("### <span class='material-symbols-outlined' style='font-size:26px; vertical-align: middle; margin-right:8px;'>analytics</span>정밀 분석 및 보정 추천 처방", unsafe_allow_html=True)
    
    if uploaded_file and 'standards' in locals() and 'batches' in locals() and standards and batches:
        std_data_res = standards[0]
        de_records = []
        for b in batches:
            rec = {"배치(BAT) 명칭": b['name']}
            for l_name in [light1_name, light2_name, light3_name]:
                if l_name != "없음":
                    lab_est = calculate_lab_exact(b['r_35'][4:35], l_name)
                    lab_tgt = calculate_lab_exact(std_data_res['r_35'][4:35], l_name)
                    de_val = colour.delta_E(lab_est, lab_tgt, method='CMC', l=2, c=1)
                    de_val = apply_dc_correction(l_name, de_val)
                    rec[f"dE [{l_name}]"] = round(de_val, 2)
            de_records.append(rec)
            
        with st.container(border=True):
            st.markdown("#### 📊 타겟(STD) 대비 배치별 실제 색차(dE) 분석")
            st.dataframe(pd.DataFrame(de_records), hide_index=True, use_container_width=True)

    if run_calc and edited_df is not None:
        std_data = standards[0]
        active_batches = [b for b in batches if b['name'] in selected_bat_names]
        
        edited_df = edited_df.fillna(0.0)
        std_initial_recipe = [float(val) for val in edited_df.iloc[:, 0].tolist()]
        bat_expected_recipes = [std_initial_recipe for _ in range(len(active_batches))]
        
        bat_actual_recipes = []
        for col_idx in range(1, edited_df.shape[1]):
            bat_actual_recipes.append([float(val) for val in edited_df.iloc[:, col_idx].tolist()])
            
        with st.spinner("스마트 매칭: 이론 역산 및 실측 데이터 대조 최적화 중..."):
            dye_predictors = []
            blank_ks_31 = blank_ks[4:35]
            
            for dye_name in selected_raw_dyes:
                conc_data = dye_db[dye_name]
                concs = sorted([float(k) for k in conc_data.keys() if float(k) > 0])
                concs_array = [0.0] + concs
                
                ks_matrix = [np.zeros(31)]
                for c in concs:
                    c_key = [k for k in conc_data.keys() if float(k) == c][0]
                    target_wls = np.arange(400, 710, 10)
                    sorted_items = sorted(conc_data[c_key].items(), key=lambda x: int(x[0]))
                    normalized_vals = np.interp(target_wls, np.array([int(k) for k, v in sorted_items]), np.array([float(v) for k, v in sorted_items]))
                    ks_matrix.append(np.maximum(get_ks(normalized_vals) - blank_ks_31, 0))
                
                dye_predictors.append(DyePredictor(concs_array, ks_matrix))
            
            result = calculate_multi_batch_correction(
                std_data['ks_31'], [b['ks_31'] for b in active_batches], 
                bat_expected_recipes, bat_actual_recipes, blank_ks_31, dye_predictors, st.session_state.dye_mode
            )
            
            if result['success']:
                with st.container(border=True):
                    st.markdown(f"#### 1. 현장 염료 발색 상태 (역산 분석)")
                    
                    calc_bat_rec_0 = result['calc_bat_recs'][0]
                    actual_bat_rec_0 = bat_actual_recipes[0]
                    cfs = result['calibration_factors']
                    calc_std_rec = result['calc_std_rec']
                    unit_label = "g/l" if st.session_state.dye_mode == "Reactive (CPB)" else "%"
                    b_name = active_batches[0]['name']
                    
                    analysis_df = pd.DataFrame({
                        "염료명": [display_name_dict.get(d, d) for d in selected_raw_dyes],
                        f"STD 역산 처방 ({unit_label})": [round(c, 4) for c in calc_std_rec],
                        f"BAT 실제 투입 ({unit_label})": [round(c, 4) for c in actual_bat_rec_0],
                        f"BAT 역산 산출 ({unit_label})": [round(c, 4) for c in calc_bat_rec_0],
                        "역산 효율 (CF)": [f"{cf*100:.1f}%" for cf in cfs]
                    })
                    
                    st.dataframe(analysis_df.style.format({
                        f"STD 역산 처방 ({unit_label})": "{:.2f}",
                        f"BAT 실제 투입 ({unit_label})": "{:.2f}",
                        f"BAT 역산 산출 ({unit_label})": "{:.2f}"
                    }), hide_index=True, use_container_width=True)
                    
                    st.caption(f"※ **BAT 역산 산출**: '{b_name}' 배치의 측정 색상(K/S)을 내기 위해, 이론적으로 필요했던 처방량입니다.")
                    st.caption("※ **역산 효율 (CF)**: (BAT 역산 산출량 ÷ BAT 실제 투입량). 100% 미만이면 현장에서 염료 발색이 덜 된 것을 의미합니다.")
                
                with st.container(border=True):
                    st.markdown(f"#### 2. 🎯 타겟(STD) 매칭을 위한 최종 보정 추천 처방")
                    final_recipe = result['final_recipe']
                    
                    recipe_df = pd.DataFrame({
                        "염료명": [display_name_dict.get(d, d) for d in selected_raw_dyes],
                        f"예상 처방 ({unit_label})": [round(c, 4) for c in std_initial_recipe],
                        f"최종 추천 처방 ({unit_label})": [round(c, 4) for c in final_recipe],
                        f"증감량 ({unit_label})": [round(final - init, 4) for final, init in zip(final_recipe, std_initial_recipe)]
                    })
                    
                    def color_delta(val): return f"color: {'#d32f2f' if val > 0 else ('#1976d2' if val < 0 else 'black')}; font-weight: bold;"
                    st.dataframe(recipe_df.style.map(color_delta, subset=[f"증감량 ({unit_label})"])
                                 .format({f"예상 처방 ({unit_label})": "{:.2f}", f"최종 추천 처방 ({unit_label})": "{:.2f}", f"증감량 ({unit_label})": "{:+.2f}"}), 
                                 hide_index=True, use_container_width=True)
            else: 
                st.error("보정 처방 산출에 실패했습니다.")