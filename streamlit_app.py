# ==============================================================================
# 第一部分：导入系统所需的“核心工具组件” (就像盖房子前准备好锤子、砖头和水泥)
# ==============================================================================
import streamlit as st          # 网页界面框架：负责把 Python 逻辑变成可点击、可操作的网页
import pandas as pd             # 表格处理工具：负责读取 Excel 文件，以及生成二维数据统计表
import geopandas as gpd         # 空间地理工具：专门用来处理带“经纬度”的多边形地图数据
import requests                 # 网络通讯工具：扮演浏览器的角色，负责向高德服务器发送请求要数据
import math                     # 数学函数库：提供三角函数(sin/cos)、平方根等基础空间计算能力
import time                     # 时间工具：用来获取当前系统时间（打时间戳），并控制代码暂停以防封号
import os                       # 操作系统接口：用来处理电脑里的文件路径、创建临时文件等
import folium                   # 地图引擎：负责在网页上画出可以鼠标拖拽、滚轮缩放的交互式底图
from streamlit_folium import st_folium  # 网页桥接插件：专门负责把 Folium 画好的地图“镶嵌”到 Streamlit 网页中
import numpy as np              # 数值计算库：极其强大的矩阵计算器，处理 300x300 个网格点瞬间完成
from scipy.spatial import cKDTree       # 空间索引算法：在几十万个点中，只需几毫秒就能找到“离我最近的那个点”
from skimage import measure             # 图像算法库：像画等高线一样，从连绵起伏的时间地形图中提取出边界轮廓
from shapely.geometry import Polygon    # 几何定义：在代码的世界里，凭空捏造出一个标准的“多边形”面状对象
from shapely.ops import unary_union     # 几何合并：把多个细碎的、互相重叠的小多边形，像融化一样合成一个大圈
import tempfile                 # 临时空间：在电脑深处创建一个用完就自动销毁的文件夹，用来存放打包文件
import zipfile                  # 压缩打包工具：把 Shapefile 必须的 4 个散装文件，打包成一个 .zip 压缩包
from io import BytesIO          # 内存缓冲区：在内存里模拟文件的读写过程，不占用实际硬盘，下载速度极快
import streamlit as st
from datetime import datetime
import pytz

# 1. 定义北京时区
beijing_tz = pytz.timezone('Asia/Shanghai')
# 2. 获取当前时间并转换时区
# .now(beijing_tz) 会直接获取带有时区信息的北京时间
now_beijing = datetime.now(beijing_tz)
# 3. 格式化显示
dt_string = now_beijing.strftime("%Y-%m-%d %H:%M:%S")
st.write(f"当前北京时间：{dt_string}")

# ==============================================================================
# 第二部分：坐标纠偏算法 (解决高德“火星坐标”导致地图发生偏移的问题)
# ==============================================================================
def gcj02_to_wgs84(lng, lat):
    """ 
    小白科普：国家为了地理安全，规定国内地图(如高德)必须对坐标进行非线性加密(GCJ-02，俗称火星坐标)。
    如果不加这步解密转换，算出来的多边形放在国际标准的卫星地图(WGS-84)上，会发生 300-500 米的错位。
    """
    # 定义地球椭球体的基础物理参数
    a = 6378137.0; ee = 0.00669342162296594323; pi = 3.1415926535897932384626

    def _transform_lat(x, y): # 内部小工具：用一套复杂的魔法公式，算出纬度方向到底偏了多少
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(y * pi) + 40.0 * math.sin(y / 3.0 * pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(y * pi / 12.0) + 320 * math.sin(y * pi / 30.0)) * 2.0 / 3.0
        return ret

    def _transform_lng(x, y): # 内部小工具：用同样复杂的魔法公式，算出经度方向到底偏了多少
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(x * pi) + 40.0 * math.sin(x / 3.0 * pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(x / 12.0 * pi) + 300.0 * math.sin(x / 30.0 * pi)) * 2.0 / 3.0
        return ret

    # 拿着高德给的坐标，减去基准值，代入上面的魔法公式
    dlat = _transform_lat(lng - 105.0, lat - 35.0) # 算出纬度偏差值
    dlng = _transform_lng(lng - 105.0, lat - 35.0) # 算出经度偏差值
    radlat = lat / 180.0 * pi # 把角度转换成弧度，方便下面做三角函数计算
    
    # 结合地球是一个椭圆形的物理事实，进一步修正偏差值
    magic = math.sin(radlat); magic = 1 - ee * magic * magic; sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi) # 最终的纬度修正量
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)     # 最终的经度修正量
    
    # 用原始的高德坐标，减去算出来的修正量，就得到了真实物理世界的纯净坐标！
    return lng - dlng, lat - dlat 

# ==============================================================================
# 第三部分：智能多 Key 轮询系统 (突破高德 API 每日免费请求次数的瓶颈)
# ==============================================================================
def smart_amap_request(url, params, api_keys, current_key_idx):
    """ 
    小白科普：高德每天只给你免费查几次数据。如果你要分析全市几十个消防站，很容易超额。
    这个函数的作用是：拿着一串钥匙(Keys)去开门，这把钥匙失效了，它会自动换下一把，绝不罢工。
    """
    while current_key_idx < len(api_keys): # 只要手里还有没试过的 Key，就一直循环
        params['key'] = api_keys[current_key_idx] # 把当前选中的这把 Key 塞进请求参数里
        try:
            # 假装自己是浏览器，向高德发送请求，并要求最多等 5 秒，拿回 JSON 格式的数据
            r = requests.get(url, params=params, timeout=5).json() 
            
            # 检查高德的脸色：status 为 0 代表报错了。
            # infocode 10003(日额度超限), 10044(余额不足), 10012(权限不足)
            if r.get('status') == '0' and r.get('infocode') in ['10003', '10044', '10012']:
                current_key_idx += 1 # 这把 Key 废了，索引加 1，换下一把
                continue # 带着新 Key 重新回到 while 循环开头去试
                
            return r, current_key_idx, True # 请求成功！把数据、当前用的哪个 Key 返回去
        except:
            # 如果网络突然断了、超时了，也算尝试了一次，直接返回空数据，防止程序崩溃死机
            return None, current_key_idx, True 
            
    # 如果把所有的 Key 都试遍了还是不行，说明弹尽粮绝了，返回 False 告诉主程序停工
    return None, current_key_idx, False 

# ==============================================================================
# 第四部分：空间时间场算法 (本程序最核心、最值钱的“数学大脑”)
# ==============================================================================
def create_isoline_polygon(trail_points, target_sec, off_road_speed):
    """ 
    小白科普：把马路上零散的导航点，像泼水一样蔓延到没路的平地、小区里。
    核心逻辑：每个地方耗费的时间 = 消防车开到路边的时间 + 消防员下车步行过去的耗时
    """
    if len(trail_points) < 10: return None # 如果轨迹点连 10 个都不到，根本画不出圈，直接放弃
    
    # 把杂乱的点阵列表转换成高效的 Numpy 矩阵：xy 存经纬度，times 存走到这花了多少秒
    pts = np.array(trail_points); xy = pts[:, 0:2]; times = pts[:, 2] 
    
    # 物理修正：地球越往北，经度之间的距离越短。我们要乘上纬度的余弦值，把地球“压扁”成平面来算距离才准
    avg_lat = np.mean(xy[:, 1]); cos_lat = np.cos(np.radians(avg_lat))
    xy_scaled = xy.copy(); xy_scaled[:, 0] *= cos_lat # 经度缩放完毕
    
    # 请出超级查找机器人 KD-Tree，把它安置在这个压扁的空间里
    tree = cKDTree(xy_scaled); pad = 0.015 # 往外扩 1.5 公里作为缓冲地带，免得等高线画到一半撞墙被切断
    min_x, max_x = np.min(xy[:,0]) - pad, np.max(xy[:,0]) + pad # 找出这片区域最左、最右的边界
    min_y, max_y = np.min(xy[:,1]) - pad, np.max(xy[:,1]) + pad # 找出最下、最上的边界
    
    # 在这片边界内，用尺子画出一张密密麻麻的 300 行 x 300 列的隐形网格网 (共 9 万个交叉点)
    grid_size = 300 
    x_grid = np.linspace(min_x, max_x, grid_size); y_grid = np.linspace(min_y, max_y, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid); grid_xy = np.column_stack([X.ravel() * cos_lat, Y.ravel()])
    
    # 让 KD-Tree 机器人出马：一口气算出这 9 万个网格点，各自离哪条马路最近！(返回距离 dists 和对应马路点的编号 indices)
    dists, indices = tree.query(grid_xy)
    
    # 灵魂公式：该网格总耗时 = 马路点自带的车行时间 + (网格离马路的距离[转成米] / 消防员步速)
    grid_T = times[indices] + (dists * 111320 / off_road_speed) 
    grid_T = grid_T.reshape((grid_size, grid_size)) # 把算出来的一长串时间，重新折叠成 300x300 的时间地形图
    
    # 请出图像处理算法：在这个时间地形图上，找出数值刚好等于我们要的“秒数”(比如 300秒)的边缘线
    contours = measure.find_contours(grid_T, level=target_sec) 
    polygons = [] # 准备一个空盒子，用来装画好的多边形
    
    for contour in contours: # 遍历找出来的每一条边缘线
        # 把刚才画图用的像素坐标 (0-300)，反向拉伸回真实的地球经纬度坐标
        c_lng = min_x + (contour[:, 1] / (grid_size - 1)) * (max_x - min_x)
        c_lat = min_y + (contour[:, 0] / (grid_size - 1)) * (max_y - min_y)
        if len(c_lng) >= 3: # 只有三个点以上才能连成面
            polygons.append(Polygon(np.column_stack([c_lng, c_lat]))) # 像捏泥人一样捏成 Polygon 面对象
            
    # 因为可能会画出好几个圈(比如遇到高架桥产生飞地)，我们用 unary_union 把它们融合成一个大面返回
    return unary_union(polygons) if polygons else None

# ==============================================================================
# 第五部分：路网实时爬虫引擎 (摸透城市路网的毛细血管)
# ==============================================================================
def run_cost_surface_engine(api_keys, key_idx, origin_lng, origin_lat, target_min, factor=0.8):
    """ 
    消防车不知道往哪开，我们就先在周围找几百个目的地(锚点)，
    然后问高德：从消防站开到这些点要多久？高德回答的路线和时间，就是我们的原始数据。
    """
    # 根据用户想要跑的时间，粗略估算一个搜索半径，但最远不能超过 15 公里
    radius = min(int(target_min * 800 * 1.5), 15000) 
    anchors = []; api_calls = 0; url_around = "https://restapi.amap.com/v3/place/around"
    # 我们要找的“目的地”类型代码：餐饮、住宅、公司、路口等，只要有路能到的地方都要
    all_types = "190301|150700|190000|170000|120000|140000|090000|060000"

    # --- 阶段一：天女散花，搜索周边锚点 ---
    for page in range(1, 13): # 最多往外翻 12 页，抓 600 个地点回来
        # 准备去问高德的问题参数：我在哪、搜多远、搜什么、要第几页
        p_params = {"location": f"{origin_lng:.6f},{origin_lat:.6f}", "radius": radius, "types": all_types, "offset": 50, "page": page}
        # 派智能小弟(请求器)去问
        r, key_idx, is_called = smart_amap_request(url_around, p_params, api_keys, key_idx)
        if is_called: api_calls += 1 # 记在账上，耗费了 1 次 API
        
        if r and r.get('status') == '1': # 如果高德愉快地回答了
            anchors.extend([poi['location'] for poi in r['pois']]) # 把这些地点的坐标摘抄进我们的锚点本子里
        else: 
            break # 如果高德说没数据了，就不翻页了
    
    anchors = list(set(anchors)) # 去掉重复的地点，免得多跑冤枉路
    trail_points = []; o_lng, o_lat = gcj02_to_wgs84(origin_lng, origin_lat) # 把消防站自己的坐标也纠偏一下
    trail_points.append((o_lng, o_lat, 0)) # 把消防站记入点云本子，到自己的时间当然是 0 秒

    # --- 阶段二：派车出发，计算真实路况耗时 ---
    url_route = "https://restapi.amap.com/v3/direction/driving" # 导航问路接口
    for dest in anchors: # 挨个拿出刚才记下的目的地
        # 问路参数：起点、终点、策略 13(代表我要躲避拥堵，寻找最快的那条路)
        params = {"origin": f"{origin_lng:.6f},{origin_lat:.6f}", "destination": dest, "strategy": 13} 
        r, key_idx, is_called = smart_amap_request(url_route, params, api_keys, key_idx)
        if is_called: api_calls += 1 # 记账：又耗费了 1 次 API
        
        if r and r.get('status') == '1' and r.get('route'): # 如果问路成功，且给出了路线
            steps = r['route']['paths'][0]['steps'] # 翻开第一条推荐路线的每一个拐弯步骤
            acc_t = 0 # 准备一个小秒表，记录开到哪了
            for s in steps:
                dur = int(s['duration']) # 高德说：这段路要开 dur 秒 (这是真实的堵车耗时！)
                polyline = s['polyline'].split(';') # 提取这段路途经的所有经纬度坐标
                time_step = dur / max(1, len(polyline) - 1) # 把这段总时间，公平地分摊给每一个途经点
                
                for i, p in enumerate(polyline): # 遍历这段路的每一个点
                    plng, plat = map(float, p.split(',')) # 剥离出经度、纬度
                    w_lng, w_lat = gcj02_to_wgs84(plng, plat) # 纠偏
                    trail_points.append((w_lng, w_lat, acc_t + i * time_step)) # 把坐标和走到这花费的秒数，写进本子
                    
                acc_t += dur # 这段路走完了，秒表加上这段的时间
                # 如果秒表上的时间，已经超出了我们的容忍极限(比如要求 5 分钟，加上特权系数折算)，那这条路就不往下探了
                if acc_t > (target_min * 60 / factor) + 60: break 
        
        time.sleep(0.01) # 假装喝口水休息 0.01 秒，防止高德觉得我们是机器人疯狂攻击它
        
    # 分析完毕，交差：(所有途经点的集合，总共探了多少个目的地，耗了多少 API，下一把该用哪个 Key)
    return trail_points, len(anchors), api_calls, key_idx

# ==============================================================================
# 第六部分：网页前端布局与 Session 初始化 (布置网页舞台)
# ==============================================================================
st.set_page_config(page_title="城市消防站可达性评估系统", layout="wide") # 把网页设为宽屏模式

# 舞台上方挂个大横幅标题
st.markdown("<h1 style='text-align: center; color: #E74C3C;'>🚒 城市消防站可达性评估系统</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7F8C8D;'>基于高德实时路况与空间代价表面算法的智能决策平台</p>", unsafe_allow_html=True)
st.divider() # 画条优美的分割线

# Session State：网页的“记忆盒子”。有了它，你按别的按钮时，之前算好的地图才不会突然消失
if 'iso_results' not in st.session_state: st.session_state.iso_results = [] # 专门存放画好的多边形数据
if 'current_key_idx' not in st.session_state: st.session_state.current_key_idx = 0 # 记住当前用到了第几把 Key
if 'map_renders' not in st.session_state: st.session_state.map_renders = 0 # 记录地图刷新了多少次，用于防报错
if 'logs' not in st.session_state: st.session_state.logs = [] # 记录系统吐出的每一句日志

# --- 侧边栏：用户的遥控器 ---
st.sidebar.header("⚙️ 核心参数配置")
api_keys_input = st.sidebar.text_area("1. 输入高德 API Keys (多 Key 英文逗号分隔)", help="提供多个 Key 可以突破单日 5000 次的调用限制")
api_key_list = [k.strip() for k in api_keys_input.split(',') if k.strip()] # 把用户输入的文本切成一个个干净的 Key
t_limit = st.sidebar.slider("2. 到场时间限制 (分钟)", 3, 15, 5) # 默认 5 分钟，全国消防法定标准
factor = st.sidebar.slider("3. 消防特权通行系数", 0.7, 1.0, 0.8) # 0.8 表示消防车无视部分红绿灯，比私家车快 20%
# 🌟 优化文案：强调这里等同于小区内的微循环速度
walk_speed = st.sidebar.slider("4. 小区内微循环/消防员步速 (推荐 1.5~3.0 米/秒)", 1.0, 5.0, 1.5, 0.5) 
map_style = st.sidebar.selectbox("5. 地图显示底图风格", ("CartoDB positron", "OpenStreetMap", "CartoDB dark_matter"))
excel = st.sidebar.file_uploader("6. 上传包含 station_name, lng, lat 的消防站表格", type=["xlsx"])

# 🌟 新增功能：自动记录早晚高峰时间戳的开关
record_timestamp = st.sidebar.checkbox("7. 📌 一键记录测算时间戳 (自动识别早晚高峰)", value=True, help="开启后，导出的数据表将自动附带当前的路况时段标签")

# --- 主界面：左右 3:1 分割布局 ---
col_map, col_monitor = st.columns([3, 1]) # 左边放地图占 3 份地盘，右边放监视器占 1 份地盘

with col_monitor: # 在右侧监控区布置家具
    st.subheader("📊 实时监控")
    prog_bar = st.empty(); prog_txt = st.empty() # 预留两个空位，等下用来刷进度条和进度文字
    st.divider()
    st.subheader("📜 运行日志")
    log_box = st.container(height=250, border=True) # 放一个固定高度、带滚动条的透明盒子装日志
    st.divider()
    st.subheader("📈 数据统计")
    stats_table_area = st.empty() # 预留一个空位，等下用来展示最终的成绩单(数据表)

# ==============================================================================
# 第七部分：核心业务流水线 (按下启动按钮后，系统开始疯狂干活)
# ==============================================================================
def get_time_tag():
    """ 🌟 新增辅助函数：根据当前电脑的钟表，判断现在是交通的什么时期 """
    t = time.localtime() # 看一眼现在的具体时间
    hour = t.tm_hour # 提取出现在是几点
    time_str = time.strftime("%Y-%m-%d %H:%M", t) # 格式化成 2026-03-23 15:30 的样子
    
    # 按照中国城市的普遍交通规律打标签
    if 7 <= hour < 9: tag = "早高峰"
    elif 17 <= hour < 19: tag = "晚高峰"
    elif hour >= 23 or hour < 6: tag = "夜间低谷"
    else: tag = "平峰期"
    
    return f"{time_str} [{tag}]" # 拼合返回，例如：2026-03-23 15:30 [平峰期]

def add_log(msg): 
    """ 小工具：把程序想说的话加上时间，扔进右上角的滚动日志框里 """
    ts = time.strftime("%H:%M:%S", time.localtime()); full_msg = f"[{ts}] {msg}"
    st.session_state.logs.append(full_msg)
    with log_box: # 在日志盒子里
        if "成功" in msg: st.success(full_msg) # 报喜用绿色
        elif "失败" in msg or "警告" in msg: st.error(full_msg) # 报丧用红色
        else: st.write(full_msg) # 普通讲话用黑白

if st.sidebar.button("🚀 开始分析"):
    if not api_key_list or not excel: # 检查装备齐没齐
        st.error("请先完成 API Key 配置并上传 Excel 数据！")
    else:
        df = pd.read_excel(excel) # 翻开用户上传的 Excel 本子
        st.session_state.iso_results = []; st.session_state.logs = []; st.session_state.current_key_idx = 0 # 清空上一次的战场记忆
        
        # 🌟 如果用户开启了打卡功能，就获取当前的时间标签
        current_time_tag = get_time_tag() if record_timestamp else "未开启记录"
        
        add_log(f"🚀 启动分析流水线... 当前路况环境: {current_time_tag}")
        
        for i, row in df.iterrows(): # 顺着 Excel 表，一个一个消防站往下审问
            name = row['station_name']
            prog_bar.progress((i + 1) / len(df)) # 推进绿色的进度条
            prog_txt.write(f"正在分析站点: {name} ({i+1}/{len(df)})")
            
            add_log(f"正在探测周边路网: {name}")
            # --- 核心环节 1：运行爬虫引擎去要数据 ---
            pts, p_cnt, a_cnt, new_idx = run_cost_surface_engine(api_key_list, st.session_state.current_key_idx, row['lng'], row['lat'], t_limit, factor)
            st.session_state.current_key_idx = new_idx # 更新这把 Key 用到哪了
            if new_idx >= len(api_key_list): add_log("🚨 警告：所有 Key 额度已全部耗尽！系统强行刹车。"); break
            
            # --- 核心环节 2：拿着点云数据去生成多边形 ---
            poly = create_isoline_polygon(pts, (t_limit * 60 / factor), walk_speed)
            if poly: # 如果成功画出了圈
                w_lng, w_lat = gcj02_to_wgs84(row['lng'], row['lat']) # 消防站自身坐标也要纠偏一下才能对得齐
                
                # 🌟 把这个站的成绩单打包成一个 GeoDataFrame (带地理性质的高级表格)
                gdf = gpd.GeoDataFrame({
                    '站点名称': [name], 
                    'API消耗': [a_cnt], 
                    'POI锚点数': [p_cnt], 
                    '测算时刻': [current_time_tag] # 🌟 塞入我们新加的时间戳标签
                }, geometry=[poly], crs="EPSG:4326")
                
                # 面积精算：把圆滚滚的地球摊平到桌面上(投影到 EPSG:3857)，算出绝对准确的物理平方米，再除以 100 万变成平方公里
                area = gdf.to_crs("EPSG:3857").area.iloc[0] / 10**6
                gdf['覆盖面积(km²)'] = round(area, 2); gdf['lat'] = w_lat; gdf['lng'] = w_lng
                
                st.session_state.iso_results.append(gdf) # 把这张成绩单收进密码箱保管
                add_log(f"✅ {name} 计算成功！面积: {area:.2f} km²")
            else:
                add_log(f"❌ {name} 失败！周围的路太少了，没法连成圈。")
            
            # 实时更新右侧的看板表：让用户感觉系统在“蹭蹭蹭”地狂算，很爽
            if st.session_state.iso_results:
                live_df = pd.concat(st.session_state.iso_results, ignore_index=True) # 把箱子里的成绩单贴在一起
                stats_table_area.dataframe(live_df[['站点名称', '覆盖面积(km²)', 'API消耗', 'POI锚点数', '测算时刻']], height=400, use_container_width=True)

        st.session_state.map_renders += 1 # 所有站算完，拨动一下地图重绘计数器
        st.balloons() # 从屏幕底下飘起庆祝气球

# ==============================================================================
# 第八部分：右侧【看板汇总表】持久化逻辑 (保护数据不随风消逝)
# ==============================================================================
with col_monitor:
    # 只要成绩箱里有东西，不管你在左边地图怎么乱点，右边这个表始终死死钉在墙上给你看
    if st.session_state.iso_results:
        summary_all_df = pd.concat(st.session_state.iso_results, ignore_index=True)
        stats_table_area.dataframe(
            summary_all_df[['站点名称', '覆盖面积(km²)', 'API消耗', 'POI锚点数', '测算时刻']], 
            height=450, 
            use_container_width=True
        )

# ==============================================================================
# 第九部分：地图可视化与专业文件下载中心 (交付规划院的干货)
# ==============================================================================
with col_map:
    # 🌟 防爆补丁：每次拨动计数器，地图的名字就变了。Streamlit 就会把旧地图直接炸掉重建，完美避开 js 渲染冲突报错
    m_key = f"fire_map_stable_render_{st.session_state.map_renders}"
    m = folium.Map(location=[22.54, 114.05], zoom_start=12, tiles=map_style) # 初始化底图，默认视野放在深圳
    
    if st.session_state.iso_results:
        # 镜头拉近：让屏幕中心瞬间转移到第一个成功算出来的消防站头上
        m.location = [st.session_state.iso_results[0]['lat'].iloc[0], st.session_state.iso_results[0]['lng'].iloc[0]]
        
        for res in st.session_state.iso_results: # 掏出每一张成绩单
            # 用 Folium 拿出紫色的油漆，顺着多边形的边界涂在地图上，透明度设为 0.4 显得高级
            folium.GeoJson(res.to_json(), style_function=lambda x: {
                'fillColor': '#8E44AD', 'color': '#732D91', 'weight': 1.5, 'fillOpacity': 0.4
            }).add_to(m)
            # 在消防站的大本营位置，插上一根蓝色的大头针
            folium.Marker([res['lat'].iloc[0], res['lng'].iloc[0]], tooltip=res['站点名称'].iloc[0]).add_to(m)
    
    # 咔嚓！把画好的动态地图，严丝合缝地镶嵌进我们的网页里
    st_folium(m, width="100%", height=750, key=m_key)

# --- 底部下载中心：给老板和甲方打包资料 ---
if st.session_state.iso_results:
    st.divider(); st.subheader("💾 消防成果一键导出中心")
    # 把所有单独的成绩单，合成一张终极空间大表 (带有地理 geometry 的超级大表)
    full_gdf = gpd.GeoDataFrame(pd.concat(st.session_state.iso_results, ignore_index=True), crs="EPSG:4326")
    
    d_col1, d_col2 = st.columns(2) # 安排两个并排的下载按钮
    
    with d_col1: # 普通人用的 CSV 表格下载
        # 砍掉经纬度图形那列(不然会变成乱码)，转码成防止中文乱码的格式
        csv_bin = full_gdf.drop(columns='geometry').to_csv(index=False).encode('utf-8-sig')
        st.download_button("📊 1. 导出统计报表 (Excel/CSV 格式)", data=csv_bin, file_name="消防服务区路况分析报表.csv", use_container_width=True)
        
    with d_col2: # 专业 GIS 规划师用的 Shapefile 打包下载
        zip_mem = BytesIO() # 虚拟一个内存空间，速度比写在硬盘上快一万倍
        with tempfile.TemporaryDirectory() as tmp_d: # 召唤一个隐形的临时文件夹
            shp_path = os.path.join(tmp_d, "fire_result.shp")
            
            # 🌟 避坑指南：Shapefile 的古老规矩，每一列的名字不能超过 10 个英文字符！
            # 如果不把中文改成短英文，程序打包时会当场崩溃给你看。
            exp_gdf = full_gdf[['站点名称', '覆盖面积(km²)', 'API消耗', '测算时刻', 'geometry']].copy()
            exp_gdf.columns = ['Name', 'Area_km2', 'API_Cnt', 'Time_Tag', 'geometry'] # 改头换面
            
            # 存入那个隐形的文件夹里，自动生成 .shp, .dbf, .shx, .prj 四个兄弟文件
            exp_gdf.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')
            
            # 拿出打包器，把这四个兄弟全部塞进 ZIP 压缩包里
            with zipfile.ZipFile(zip_mem, "w", zipfile.ZIP_DEFLATED) as zf:
                for r, _, fs in os.walk(tmp_d):
                    for f in fs: zf.write(os.path.join(r, f), arcname=f)
                    
        # 提供一个华丽的下载按钮，一键把压缩包交到用户手上
        st.download_button("📦 2. 导出专业地图图层 (GIS SHP 压缩包 wgs84坐标)", data=zip_mem.getvalue(), file_name="消防路况图层包_带时间戳.zip", type="primary", use_container_width=True)
