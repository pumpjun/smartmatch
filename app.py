import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator
import json
import re
import os
import colour
import base64

# ==========================================
# 0. 페이지 기본 설정 및 세션 상태 초기화
# ==========================================
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="Ohyoung Smart Match", page_icon="data/logo.png")

if "dye_mode" not in st.session_state: st.session_state.dye_mode = "Reactive"
if "disperse_sub" not in st.session_state: st.session_state.disperse_sub = "Jersey"
if "selected_dyes" not in st.session_state: st.session_state.selected_dyes = []
if "input_confirmed" not in st.session_state: st.session_state.input_confirmed = False
if "saved_df" not in st.session_state: st.session_state.saved_df = None
if "saved_batches" not in st.session_state: st.session_state.saved_batches = []

def set_dye_mode(mode):
    if st.session_state.dye_mode != mode:
        st.session_state.dye_mode = mode
        st.session_state.selected_dyes = []
        st.session_state.input_confirmed = False

def toggle_dye(raw_name):
    if raw_name in st.session_state.selected_dyes:
        st.session_state.selected_dyes.remove(raw_name)
    else:
        st.session_state.selected_dyes.append(raw_name)
    st.session_state.input_confirmed = False

# ==========================================
# 1. 공통 UI 스타일 및 로고 헤더 구성
# ==========================================
base_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(base_dir, 'data', 'logo.png')
try:
    with open(logo_path, "rb") as image_file:
        logo_base64 = base64.b64encode(image_file.read()).decode()
except Exception:
    logo_base64 = ""

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
<style>
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    .fixed-header {{
        position: fixed; top: 0; left: 0; width: 100vw; height: 60px;
        background-color: #ffffff; box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
        z-index: 999998; display: flex; align-items: center; padding-left: 20px;
        border-bottom: 1px solid #eaeaea;
    }}
    .fixed-header img {{ width: 45px; margin-right: 12px; }}
    .fixed-header h2 {{ margin: 0; padding: 0; font-size: 24px; font-weight: 700; color: #31333F; }}
    .block-container {{ padding-top: 80px !important; }}
    
    div[data-testid="stHorizontalBlock"]:has(#top-menu-marker) {{
        position: fixed !important; top: 10px !important; left: 320px !important; 
        width: 820px !important; z-index: 999999 !important; align-items: center !important; 
    }}
    div.element-container:has(#top-menu-marker) {{ display: none !important; }}
    div[data-testid="stHorizontalBlock"]:has(#top-menu-marker) div.stButton > button {{
        border-radius: 8px; padding: 0px 10px; height: 38px; min-height: 38px; margin: 0 !important; 
    }}
    
    [data-testid="stSidebar"] div.stButton {{ margin-bottom: -10px; }}
</style>
<div class="fixed-header">
    <img src="data:image/png;base64,{logo_base64}" onerror="this.style.display='none'">
    <h2>Ohyoung Smart Match</h2>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. 데이터 및 염료 매핑 로드
# ==========================================
@st.cache_data
def load_dye_data(mode):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if mode == "Reactive": file_name = 'dye_data.json'
    elif mode == "Disperse": file_name = 'dye_data_disperse.json'
    elif mode == "Reactive (CPB)": file_name = 'dye_data_cpb.json'
    elif mode == "CDP": file_name = 'dye_data_CDP.json'
    elif mode == "Acid": file_name = 'dye_data_acid.json'
    else: file_name = 'dye_data.json'
    
    target_path = os.path.join(base_dir, 'data', file_name)
    try:
        with open(target_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            return {name.strip(): concs for name, concs in raw_data.items() if len(concs) > 0}
    except Exception: return {}

@st.cache_data
def load_dye_mapping(mode, _valid_keys):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if mode == "Reactive": file_name = 'dye_list.xlsx'
    elif mode == "Disperse": file_name = 'dis_dye_list.xlsx'
    elif mode == "Reactive (CPB)": file_name = 'cpb_dye_list.xlsx'
    elif mode == "CDP": file_name = 'CDP_dye_list.xlsx'
    elif mode == "Acid": file_name = 'acid_dye_list.xlsx'
    else: file_name = 'dye_list.xlsx'
    
    target_path = os.path.join(base_dir, 'data', file_name)
    try:
        df = pd.read_excel(target_path, header=None)
        mapping_list, disp_dict, sort_order_dict = [], {}, {}
        for _, row in df.iterrows():
            try: sort_val = float(row[0]) if pd.notna(row[0]) else 999.0
            except: sort_val = 999.0
                
            raw_name, display_name = str(row[1]).strip(), str(row[2]).strip()
            if raw_name in _valid_keys:
                mapping_list.append((raw_name, display_name))
                disp_dict[raw_name] = display_name
                sort_order_dict[raw_name] = sort_val
                
        return mapping_list, disp_dict, sort_order_dict
    except Exception:
        return [(k, k) for k in sorted(list(_valid_keys))], {k: k for k in _valid_keys}, {}

dye_db = load_dye_data(st.session_state.dye_mode)
all_dyes_ordered, display_name_dict, sort_order_dict = load_dye_mapping(st.session_state.dye_mode, dye_db.keys())

# ==========================================
# 3. 백포 및 색상 계산 로직 (광원 스펙트럼 포함)
# ==========================================
def get_ks(reflectance): return (1 - reflectance)**2 / (2 * reflectance)

blank_r_reactive = np.array([float(x.strip()) / 100.0 for x in "61.487896,64.536758,67.636276,70.483246,73.516251,75.622711,77.759293,79.583626,80.990044,82.235336,83.458176,84.331772,85.404106,86.164101,86.926323,87.612724,88.086739,88.541801,88.927353,89.348244,89.645943,89.882187,90.113014,90.397278,90.583130,90.746536,90.858932,91.020134,91.199127,91.403587,91.537102,91.670677,91.884819,91.980095,92.083275".split(',') if x.strip()])
blank_r_disp_woven = np.array([float(x.strip()) / 100.0 for x in "12.550282,12.662358,13.128991,15.679015,22.198656,49.159801,90.868118,115.976440,119.316971,108.527847,99.793015,95.301094,93.386421,90.213654,88.579010,86.774918,85.229942,84.013969,83.225197,82.656410,82.167915,82.044777,81.931137,81.882439,82.200607,82.416222,82.896469,83.570839,84.365234,85.158195,85.765289,86.293983,86.519203,86.480919,86.653557".split(',') if x.strip()])
blank_r_disp_jersey = np.array([float(x.strip()) / 100.0 for x in "83.435000,84.975000,84.604500,83.743500,82.880500,82.520000,82.496500,82.684500,83.045000,83.254500,83.411000,83.424500,83.473500,83.565500,83.673000,83.827000,83.952000,84.098000,84.095000,84.015500,84.072500,84.145000,84.288500,84.341000,84.453000,84.588000,84.811500,84.936000,85.228500,85.319000,85.334000".split(',') if x.strip()])
blank_r_cpb = np.array([float(x.strip()) / 100.0 for x in "65.391068,67.093147,68.937622,70.637802,72.489166,74.245972,75.516464,76.748680,77.643394,78.417038,79.229340,79.835594,80.472313,81.005417,81.598862,82.127090,82.478500,82.846130,83.165131,83.535942,83.806923,84.017708,84.260544,84.523628,84.728180,84.964073,85.149178,85.394760,85.695389,86.005653,86.231079,86.412338,86.664047,86.753532,86.915657".split(',') if x.strip()])
blank_r_cdp = np.interp(np.arange(360, 710, 10), np.arange(400, 710, 10), np.array([float(x.strip()) / 100.0 for x in "66.642869, 69.331771, 71.930599, 73.894465, 75.701308, 77.198517, 78.580475, 79.721862, 80.800241, 81.569093, 82.308775, 82.936382, 83.379400, 83.891529, 84.177935, 84.593368, 84.832168, 85.117948, 85.361063, 85.496897, 85.857874, 85.970598, 86.050570, 86.297560, 86.426288, 86.583924, 86.748540, 86.945128, 87.095481, 87.011909, 87.268311".split(',') if x.strip()]))
blank_r_acid = np.array([float(x.strip()) / 100.0 for x in "31.696901, 46.754527, 60.104591, 69.332457, 74.595761, 77.676672, 79.372197, 80.702138, 81.639552, 82.371908, 83.169580, 83.820981, 84.401035, 84.760123, 85.150397, 85.437840, 85.618544, 85.783499, 85.906208, 86.081845, 86.177695, 86.249357, 86.376613, 86.540455, 86.633664, 86.740136, 86.842757, 86.998326, 87.199730, 87.414002, 87.491035, 87.494141, 87.475324, 87.320161, 87.079161".split(',') if x.strip()])

if st.session_state.dye_mode == "Reactive": blank_ks = get_ks(blank_r_reactive)[4:35]
elif st.session_state.dye_mode == "Disperse": blank_ks = get_ks(blank_r_disp_woven)[4:35] if st.session_state.disperse_sub == "Woven" else get_ks(blank_r_disp_jersey)[4:35]
elif st.session_state.dye_mode == "Reactive (CPB)": blank_ks = get_ks(blank_r_cpb)[4:35]
elif st.session_state.dye_mode == "CDP": blank_ks = get_ks(blank_r_cdp)[4:35]
elif st.session_state.dye_mode == "Acid": blank_ks = get_ks(blank_r_acid)[4:35]

wls_astm = np.arange(360, 790, 10)
astm_d65_x_vals = [0.000, 0.000, 0.001, 0.005, 0.097, 0.616, 1.660, 2.377, 3.512, 3.789, 3.103, 1.937, 0.747, 0.110, 0.007, 0.314, 1.027, 2.174, 3.380, 4.735, 6.081, 7.310, 8.393, 8.603, 8.771, 7.996, 6.476, 4.635, 3.074, 1.814, 1.031, 0.557, 0.261, 0.114, 0.057, 0.028, 0.011, 0.006, 0.003, 0.001, 0.000, 0.000, 0.000]
astm_d65_y_vals = [0.000, 0.000, 0.000, 0.000, 0.010, 0.064, 0.171, 0.283, 0.549, 0.888, 1.277, 1.817, 2.545, 3.164, 4.309, 5.631, 6.896, 8.136, 8.684, 8.903, 8.614, 7.950, 7.164, 5.945, 5.110, 4.067, 2.990, 2.020, 1.275, 0.724, 0.407, 0.218, 0.102, 0.044, 0.022, 0.011, 0.004, 0.002, 0.001, 0.000, 0.000, 0.000, 0.000]
astm_d65_z_vals = [0.000, -0.001, 0.004, 0.020, 0.436, 2.808, 7.868, 11.703, 17.958, 20.358, 17.861, 13.085, 7.510, 3.743, 2.003, 1.004, 0.529, 0.271, 0.116, 0.030, -0.003, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
custom_d65_X = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_d65_x_vals) / 100.0)), name='D65_X')
custom_d65_Y = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_d65_y_vals) / 100.0)), name='D65_Y')
custom_d65_Z = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_d65_z_vals) / 100.0)), name='D65_Z')

astm_a_x_vals = [0.000, 0.000, 0.000, 0.002, 0.025, 0.134, 0.377, 0.686, 0.964, 1.080, 1.006, 0.731, 0.343, 0.078, 0.022, 0.218, 0.750, 1.642, 2.842, 4.336, 6.200, 8.262, 10.227, 11.945, 12.746, 12.337, 10.817, 8.560, 6.014, 3.887, 2.309, 1.276, 0.666, 0.336, 0.166, 0.082, 0.040, 0.020, 0.010, 0.005, 0.003, 0.001, 0.001]
astm_a_y_vals = [0.000, 0.000, 0.000, 0.000, 0.003, 0.014, 0.039, 0.084, 0.156, 0.259, 0.424, 0.696, 1.082, 1.616, 2.422, 3.529, 4.840, 6.100, 7.250, 8.114, 8.758, 8.988, 8.760, 8.304, 7.468, 6.323, 5.033, 3.744, 2.506, 1.560, 0.911, 0.499, 0.259, 0.130, 0.065, 0.032, 0.016, 0.008, 0.004, 0.002, 0.001, 0.001, 0.000]
astm_a_z_vals = [0.000, 0.000, 0.000, 0.008, 0.110, 0.615, 1.792, 3.386, 4.944, 5.806, 5.812, 4.919, 3.300, 1.973, 1.152, 0.658, 0.382, 0.211, 0.102, 0.032, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
custom_a_X = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_a_x_vals) / 100.0)), name='ASTM_A_X')
custom_a_Y = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_a_y_vals) / 100.0)), name='ASTM_A_Y')
custom_a_Z = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_a_z_vals) / 100.0)), name='ASTM_A_Z')

astm_tl84_x_vals = [0.000, 0.000, 0.000, -0.010, 0.099, 0.182, 0.098, 2.796, 4.103, 1.534, 1.314, 0.681, 0.343, 0.176, 0.009, 0.034, 0.005, -0.145, 10.852, 12.320, 1.096, 1.157, 7.036, 8.982, 6.204, 26.264, 13.228, 3.797, 0.794, 0.481, 0.264, 0.084, 0.038, 0.023, 0.011, 0.014, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
astm_tl84_y_vals = [0.000, 0.000, 0.000, -0.001, 0.010, 0.019, 0.003, 0.372, 0.625, 0.388, 0.554, 0.578, 1.380, 2.955, 1.506, 0.564, 0.257, 0.170, 25.656, 24.661, 1.274, 1.214, 5.881, 6.382, 3.629, 13.321, 6.279, 1.631, 0.329, 0.192, 0.104, 0.033, 0.015, 0.009, 0.004, 0.005, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
astm_tl84_z_vals = [0.000, 0.000, 0.000, -0.044, 0.451, 0.829, 0.415, 13.964, 20.873, 8.310, 7.586, 4.498, 3.625, 3.789, 0.773, 0.074, 0.028, 0.027, 0.293, 0.148, -0.010, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
custom_tl84_X = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_tl84_x_vals) / 100.0)), name='ASTM_TL84_X')
custom_tl84_Y = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_tl84_y_vals) / 100.0)), name='ASTM_TL84_Y')
custom_tl84_Z = colour.SpectralDistribution(dict(zip(wls_astm, np.array(astm_tl84_z_vals) / 100.0)), name='ASTM_TL84_Z')

LIGHT_MAP = {
    "D65": (custom_d65_X, custom_d65_Y, custom_d65_Z),
    "A": (custom_a_X, custom_a_Y, custom_a_Z),
    "TL84 (F11)": (custom_tl84_X, custom_tl84_Y, custom_tl84_Z)
}

def get_lab_from_r(r_array, light_name="D65"):
    num_points = len(r_array)
    start_wl = 360 if num_points == 35 else 400
    shape_target = colour.SpectralShape(start_wl, start_wl + (num_points - 1) * 10, 10)
    
    light_data = LIGHT_MAP.get(light_name, LIGHT_MAP["D65"])
    W_X = light_data[0].copy().align(shape_target).values
    W_Y = light_data[1].copy().align(shape_target).values
    W_Z = light_data[2].copy().align(shape_target).values
    W = np.column_stack((W_X, W_Y, W_Z))
    
    wp_XYZ = np.sum(W, axis=0)
    wp_xy = colour.XYZ_to_xy(wp_XYZ)
    XYZ = np.dot(r_array, W)
    return colour.XYZ_to_Lab(XYZ, illuminant=wp_xy)

def get_preview_hex(target_r_array, light_name="D65"):
    shape_10nm = colour.SpectralShape(400, 700, 10)
    light_data = LIGHT_MAP.get(light_name, LIGHT_MAP["D65"])
    W_X = light_data[0].copy().align(shape_10nm).values
    W_Y = light_data[1].copy().align(shape_10nm).values
    W_Z = light_data[2].copy().align(shape_10nm).values
    W = np.column_stack((W_X, W_Y, W_Z))
    wp_XYZ = np.sum(W, axis=0)
    wp_xy = colour.XYZ_to_xy(wp_XYZ)
    XYZ_tgt = np.dot(target_r_array, W)
    RGB_viz = colour.XYZ_to_sRGB(XYZ_tgt, illuminant=wp_xy)
    RGB_viz = np.clip(RGB_viz, 0, 1)
    return "#{:02x}{:02x}{:02x}".format(int(RGB_viz[0]*255), int(RGB_viz[1]*255), int(RGB_viz[2]*255)), [int(RGB_viz[0]*255), int(RGB_viz[1]*255), int(RGB_viz[2]*255)]

@st.cache_data
def get_all_dye_hex_dict(mode):
    hex_dict = {}
    current_db = load_dye_data(mode)
    for dye_name, conc_data in current_db.items():
        available_concs = sorted([float(k) for k in conc_data.keys() if float(k) > 0])
        if not available_concs: continue
        spectrum_map = conc_data[[k for k in conc_data.keys() if float(k) == available_concs[-1]][0]]
        sorted_items = sorted(spectrum_map.items(), key=lambda x: int(x[0]))
        r_array_31 = np.interp(np.arange(400, 710, 10), np.array([int(k) for k, v in sorted_items]), np.array([float(v) for k, v in sorted_items]))
        hc, _ = get_preview_hex(r_array_31, "D65")
        hex_dict[dye_name] = hc
    return hex_dict

dye_hex_dict = get_all_dye_hex_dict(st.session_state.dye_mode)

# ==========================================
# 4. 상단 메뉴바 구성 & 백포 팝업
# ==========================================
def set_temp_disp(val): st.session_state.temp_disp = val
def confirm_disp_action():
    st.session_state.disperse_sub = st.session_state.temp_disp
    st.session_state.dye_mode = "Disperse"
    st.session_state.selected_dyes = []
    st.session_state.input_confirmed = False

@st.dialog("백포 선택 (Disperse)")
def disperse_dialog():
    st.markdown("분산염료처방 탐색에 사용할 백포를 선택해주세요.")
    if "temp_disp" not in st.session_state: st.session_state.temp_disp = st.session_state.disperse_sub
    col1, col2 = st.columns(2)
    with col1: st.button("Jersey", width='stretch', type="primary" if st.session_state.temp_disp == "Jersey" else "secondary", on_click=set_temp_disp, args=("Jersey",))
    with col2: st.button("Woven", width='stretch', type="primary" if st.session_state.temp_disp == "Woven" else "secondary", on_click=set_temp_disp, args=("Woven",))
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    if st.button("확인", width='stretch', type="primary", on_click=confirm_disp_action): st.rerun()

top_menu_cols = st.columns([1, 1, 1.2, 1, 1])
with top_menu_cols[0]:
    st.button("Reactive", width='stretch', type="primary" if st.session_state.dye_mode == "Reactive" else "secondary", on_click=set_dye_mode, args=("Reactive",))
    st.markdown('<div id="top-menu-marker"></div>', unsafe_allow_html=True)
with top_menu_cols[1]:
    if st.button("Disperse", width='stretch', type="primary" if st.session_state.dye_mode == "Disperse" else "secondary"):
        st.session_state.temp_disp = st.session_state.disperse_sub
        disperse_dialog()
with top_menu_cols[2]: st.button("Reactive (CPB)", width='stretch', type="primary" if st.session_state.dye_mode == "Reactive (CPB)" else "secondary", on_click=set_dye_mode, args=("Reactive (CPB)",))
with top_menu_cols[3]: st.button("CDP", width='stretch', type="primary" if st.session_state.dye_mode == "CDP" else "secondary", on_click=set_dye_mode, args=("CDP",))
with top_menu_cols[4]: st.button("Acid", width='stretch', type="primary" if st.session_state.dye_mode == "Acid" else "secondary", on_click=set_dye_mode, args=("Acid",))

# ==========================================
# 5. 좌측 사이드바 (염료 리스트)
# ==========================================
with st.sidebar:
    st.markdown("<h3 style='display: flex; align-items: center;'><span class='material-symbols-outlined' style='margin-right:8px;'>palette</span>염료 리스트</h3>", unsafe_allow_html=True)
    st.caption("클릭하여 보정에 사용할 염료를 선택하세요.")
    st.markdown("---")

    for idx, (raw_name, display_name) in enumerate(all_dyes_ordered):
        btn_type = "primary" if raw_name in st.session_state.selected_dyes else "secondary"
        hex_col = dye_hex_dict.get(raw_name, "#FFFFFF")
        
        col_color, col_btn = st.columns([1.5, 8.5])
        with col_color:
            st.markdown(
                f"""
                <div style="
                    background-color: {hex_col}; 
                    height: 35px; 
                    width: 100%; 
                    border-radius: 6px; 
                    border: 1px solid #ccc; 
                    margin-top: 4px; 
                    box-shadow: inset 0px 0px 4px rgba(0,0,0,0.1);
                "></div>
                """, unsafe_allow_html=True
            )
        with col_btn:
            st.button(display_name, key=f"dye_{raw_name}_{idx}", use_container_width=True, type=btn_type, on_click=toggle_dye, args=(raw_name,))

# ==========================================
# 6. 보정 알고리즘 및 메인 화면 2분할 로직
# ==========================================
def get_ks_normalized(spectrum_map):
    target_wls = np.arange(400, 710, 10)
    sorted_items = sorted(spectrum_map.items(), key=lambda x: int(x[0]))
    normalized_vals = np.interp(target_wls, np.array([int(k) for k, v in sorted_items]), np.array([float(v) for k, v in sorted_items]))
    return get_ks(normalized_vals)

def parse_qtx_content(content):
    standards, batches = [], []
    blocks = re.split(r'\[(STANDARD_DATA|BATCH_DATA)[^\]]*\]', content)
    for i in range(1, len(blocks), 2):
        block_type, block_content = blocks[i], blocks[i+1]
        prefix = "STD_" if block_type == 'STANDARD_DATA' else "BAT_"
        name_match = re.search(fr'{prefix}NAME=(.*?)\n', block_content)
        r_match = re.search(fr'{prefix}R=([\d\.,\s]+)', block_content)
        low_match = re.search(fr'{prefix}REFLLOW=(\d+)', block_content)
        if r_match:
            name = name_match.group(1).strip().rstrip(',') if name_match else "Unknown"
            r_str = r_match.group(1)
            r_vals = [float(x.strip()) / 100.0 for x in r_str.split(',') if x.strip()]
            start_wl = int(low_match.group(1)) if low_match else 400
            current_wls = np.array([start_wl + j * 10 for j in range(len(r_vals))])
            target_wls = np.arange(360, 710, 10)
            r_35 = np.interp(target_wls, current_wls, r_vals)
            data_dict = {'name': name, 'r_35': r_35, 'r_31': r_35[4:35], 'ks_31': get_ks(r_35[4:35])}
            if block_type == 'STANDARD_DATA': standards.append(data_dict)
            else: batches.append(data_dict)
    return standards, batches

def calculate_multi_batch_correction(std_ks, bat_ks_list, bat_expected_recipes, bat_actual_recipes, blank_ks_arr, dye_interpolators, mode):
    num_dyes = len(dye_interpolators)
    
    def cf_objective(cf_array):
        total_error = 0
        for i, bat_ks in enumerate(bat_ks_list):
            est_ks = np.copy(blank_ks_arr)
            for j in range(num_dyes): 
                est_ks += dye_interpolators[j](bat_actual_recipes[i][j]) * cf_array[j]
            total_error += np.sum((bat_ks - est_ks)**2)
        return total_error

    res_cf = minimize(cf_objective, np.ones(num_dyes), bounds=[(0.5, 1.5) for _ in range(num_dyes)], method='SLSQP')
    optimal_cf = res_cf.x if res_cf.success else np.ones(num_dyes)
    
    def recipe_objective(recipe):
        est_ks = np.copy(blank_ks_arr)
        for j in range(num_dyes): 
            est_ks += dye_interpolators[j](recipe[j]) * optimal_cf[j]
        return np.sum((std_ks - est_ks)**2)
    
    initial_recipe = bat_expected_recipes[0] if len(bat_expected_recipes) > 0 else np.zeros(num_dyes)
    max_bound = 150.0 if mode == "Reactive (CPB)" else 15.0
    
    res_recipe = minimize(recipe_objective, initial_recipe, bounds=[(0.0, max_bound) for _ in range(num_dyes)], method='SLSQP')
    
    return {
        "success": res_recipe.success, 
        "calibration_factors": optimal_cf, 
        "final_recipe": res_recipe.x if res_recipe.success else None
    }


# ==========================================
# 7. 메인 화면 좌/우 2분할 레이아웃
# ==========================================
col_input, col_result = st.columns([1, 1.2], gap="large")

standards = []
batches = []
std_data = None
selected_bat_names = []

# --- 왼쪽 컬럼 (입력부) ---
with col_input:
    st.markdown("### 1. 데이터 업로드 및 광원 선택")
    uploaded_file = st.file_uploader("QTX 파일 업로드 (STD & BAT 포함)", type=['qtx'])
    selected_light = st.selectbox("보정 기준 광원 선택", options=list(LIGHT_MAP.keys()), index=0)
    
    st.markdown("### 2. 배치 선택 및 레시피 입력")
    
    edited_df_result = None
    
    if uploaded_file and len(st.session_state.selected_dyes) > 0:
        content = uploaded_file.getvalue().decode('euc-kr', errors='ignore')
        standards, batches = parse_qtx_content(content)
        
        if not standards:
            st.error("QTX 파일에 STANDARD 데이터가 없습니다.")
        elif not batches:
            st.error("QTX 파일에 BATCH 데이터가 없습니다.")
        else:
            std_data = standards[0]
            st.success(f"타겟(STD) 인식 완료: **{std_data['name']}**")
            
            all_bat_names = [b['name'] for b in batches]
            selected_bat_names = st.multiselect("분석에 사용할 BAT 데이터를 복수 선택하세요", options=all_bat_names, default=all_bat_names)
            
            if len(selected_bat_names) > 0:
                selected_raw_dyes = sorted(st.session_state.selected_dyes, key=lambda x: sort_order_dict.get(x, 999.0))
                selected_display_names = [display_name_dict.get(d, d) for d in selected_raw_dyes]
                
                col_names = ["[STD] Datacolor 예상 처방"] + [f"[BAT] {b_name} 실제 투입량" for b_name in selected_bat_names]
                df_input = pd.DataFrame(0.0, index=selected_display_names, columns=col_names)
                
                table_key = f"multi_table_step_{'-'.join(selected_raw_dyes)}_{'-'.join(selected_bat_names)}"
                unit_label = "g/l" if st.session_state.dye_mode == "Reactive (CPB)" else "%"
                st.markdown(f"**Datacolor 예상 처방과 각 배치별 실제 현장 투입량({unit_label})을 입력하세요:**")
                
                edited_df_result = st.data_editor(df_input, use_container_width=True, key=table_key)
                
                # 🌟 단계 1: [입력 완료 및 값 고정] 버튼
                if st.button("📥 1단계: 입력값 확정 및 저장", type="secondary", use_container_width=True):
                    st.session_state.saved_df = edited_df_result
                    st.session_state.saved_batches = selected_bat_names
                    st.session_state.input_confirmed = True
                    st.success("입력값이 안전하게 저장되었습니다! 이제 아래 2단계 계산 버튼을 눌러주세요.")
                
    elif uploaded_file and len(st.session_state.selected_dyes) == 0:
        st.info("👈 좌측 사이드바에서 보정에 사용할 염료를 선택해 주세요.")
    else:
        st.info("QTX 파일을 먼저 업로드해 주세요.")
        
    st.markdown("---")
    
    # 🌟 단계 2: [정밀 보정 계산 시작] 버튼 (입력 완료가 눌렸을 때만 활성화 또는 실행 보장)
    run_calc = st.button("🚀 2단계: 정밀 보정 계산 시작", type="primary", use_container_width=True, disabled=not st.session_state.input_confirmed)
    if not st.session_state.input_confirmed and uploaded_file and len(st.session_state.selected_dyes) > 0:
        st.caption("💡 먼저 위쪽의 **[1단계: 입력값 확정 및 저장]** 버튼을 눌러주세요.")

# --- 오른쪽 컬럼 (결과부) ---
with col_result:
    st.markdown("### 3. 정밀 분석 및 보정 결과")
    
    if uploaded_file is not None:
        try:
            content_res = uploaded_file.getvalue().decode('euc-kr', errors='ignore')
            standards_res, batches_res = parse_qtx_content(content_res)
            if standards_res and batches_res:
                std_data_res = standards_res[0]
                std_lab = get_lab_from_r(std_data_res['r_35'], selected_light)
                de_records = []
                for b in batches_res:
                    bat_lab = get_lab_from_r(b['r_35'], selected_light)
                    de_val = colour.delta_E(bat_lab, std_lab, method='CMC', l=2, c=1)
                    de_records.append({
                        "배치(BAT) 명칭": b['name'],
                        f"dE (CMC) [{selected_light}]": round(de_val, 2)
                    })
                
                with st.container(border=True):
                    st.markdown(f"#### 📊 타겟(STD) 대비 전체 배치(BAT) 색차 분석")
                    st.dataframe(pd.DataFrame(de_records), hide_index=True, use_container_width=True)
        except Exception:
            pass

    # 2단계 계산 실행부 (확정된 세션 데이터를 기반으로 연산)
    if run_calc and st.session_state.input_confirmed and st.session_state.saved_df is not None:
        content = uploaded_file.getvalue().decode('euc-kr', errors='ignore')
        standards, batches = parse_qtx_content(content)
        std_data = standards[0]
        
        target_bat_names = st.session_state.saved_batches
        active_batches = [b for b in batches if b['name'] in target_bat_names]
        
        if len(active_batches) > 0:
            df_to_use = st.session_state.saved_df.fillna(0.0)
            std_initial_recipe = [float(val) for val in df_to_use.iloc[:, 0].tolist()]
            bat_expected_recipes = [std_initial_recipe for _ in range(len(active_batches))]
            
            bat_actual_recipes = []
            for col_idx in range(1, df_to_use.shape[1]):
                col_vals = [float(val) for val in df_to_use.iloc[:, col_idx].tolist()]
                bat_actual_recipes.append(col_vals)
                
            selected_raw_dyes = sorted(st.session_state.selected_dyes, key=lambda x: sort_order_dict.get(x, 999.0))
            selected_display_names = [display_name_dict.get(d, d) for d in selected_raw_dyes]
                
            with st.spinner("다중 배치 데이터 종합 분석 및 처방 최적화 중..."):
                dye_interpolators = []
                for dye_name in selected_raw_dyes:
                    conc_data = dye_db[dye_name]
                    concs = sorted([float(k) for k in conc_data.keys() if float(k) > 0])
                    concs_array = np.array([0.0] + concs)
                    
                    ks_matrix = [np.zeros(31)]
                    for c in concs:
                        c_key = [k for k in conc_data.keys() if float(k) == c][0]
                        ks_net = np.maximum(get_ks_normalized(conc_data[c_key]) - blank_ks, 0)
                        ks_matrix.append(ks_net)
                    dye_interpolators.append(PchipInterpolator(concs_array, np.array(ks_matrix), axis=0))
                
                bat_ks_list = [b['ks_31'] for b in active_batches]
                
                result = calculate_multi_batch_correction(
                    std_data['ks_31'], bat_ks_list, bat_expected_recipes, bat_actual_recipes, 
                    blank_ks, dye_interpolators, st.session_state.dye_mode
                )
                
                if result['success']:
                    with st.container(border=True):
                        st.markdown(f"#### 1. 광원 [{selected_light}] 기준 현장 염료 발색 상태 (Calibration Factor)")
                        cf_data = {display_name_dict.get(dye, dye): f"{cf*100:.1f}%" for dye, cf in zip(selected_raw_dyes, result['calibration_factors'])}
                        st.json(cf_data)
                    
                    with st.container(border=True):
                        unit_label = "g/l" if st.session_state.dye_mode == "Reactive (CPB)" else "%"
                        st.markdown(f"#### 2. 🎯 타겟(STD) 매칭을 위한 최종 추천 처방 ({unit_label})")
                        
                        final_recipe = result['final_recipe']
                        delta_recipe = [final - initial for final, initial in zip(final_recipe, std_initial_recipe)]
                        
                        recipe_df = pd.DataFrame({
                            "염료명": selected_display_names,
                            f"Datacolor 예상 처방 ({unit_label})": [round(c, 4) for c in std_initial_recipe],
                            f"최종 보정 추천 처방 ({unit_label})": [round(c, 4) for c in final_recipe],
                            f"증감량 (Add/Reduce) ({unit_label})": [round(d, 4) for d in delta_recipe]
                        })
                        
                        def color_delta(val): 
                            return f"color: {'#d32f2f' if val > 0 else ('#1976d2' if val < 0 else 'black')}; font-weight: bold;"
                            
                        st.dataframe(
                            recipe_df.style.map(color_delta, subset=[f"증감량 (Add/Reduce) ({unit_label})"])
                                         .format({
                                             f"Datacolor 예상 처방 ({unit_label})": "{:.2f}", 
                                             f"최종 보정 추천 처방 ({unit_label})": "{:.2f}", 
                                             f"증감량 (Add/Reduce) ({unit_label})": "{:+.2f}"
                                         }), 
                            hide_index=True, 
                            use_container_width=True
                        )
                else: 
                    st.error("보정 처방을 산출하는 데 실패했습니다.")
        else:
            st.warning("분석에 사용할 BAT 데이터를 최소 1개 이상 선택해 주세요.")
    else:
        if not st.session_state.input_confirmed:
            st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 200px; color: #999; border: 1px dashed #ccc; border-radius: 10px; background-color: #f8f9fa;">
                <span class='material-symbols-outlined' style='font-size: 48px;'>edit_note</span>
                <p style="margin-top: 10px; font-size: 15px;">표에 투입량을 입력하신 후 <b>[1단계: 입력값 확정 및 저장]</b>을 먼저 눌러주세요.</p>
            </div>
            """, unsafe_allow_html=True)