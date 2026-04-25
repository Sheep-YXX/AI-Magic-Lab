# 必须写在导入 pyplot 之前！强制 Matplotlib 使用无界面的后台渲染，解决网页“卡死”问题
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# 配置中文字体，解决小方块乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统推荐使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 修复负号显示问题

import gradio as gr
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
import cv2
from ultralytics import YOLO

# ==========================================
# 模块一：视觉魔法 (接入真实 YOLOv8 模型)
# ==========================================
try:
    model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"模型加载失败，请检查网络连接: {e}")

def real_object_detection(image, conf_threshold):
    if image is None:
        return None
    results = model.predict(source=image, conf=conf_threshold, verbose=False)
    img_copy = image.copy()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0].cpu().numpy()
            c = int(box.cls.cpu().numpy()[0])
            conf = box.conf.cpu().numpy()[0]
            name = model.names[c]
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_copy, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
            cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img_copy

# ==========================================
# 模块二：近朱者赤 (KNN 分类)
# ==========================================
np.random.seed(42)
X_train = np.random.rand(30, 2) * 10
y_train = np.array([0 if x[0]+x[1] < 10 else 1 for x in X_train])

def run_knn(k, new_x, new_y):
    knn = KNeighborsClassifier(n_neighbors=int(k))
    knn.fit(X_train, y_train)
    new_point = np.array([[new_x, new_y]])
    prediction = knn.predict(new_point)[0]
    result_text = "苹果 🍎" if prediction == 0 else "西瓜 🍉"
    
    distances, indices = knn.kneighbors(new_point)
    nearest_points = X_train[indices[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_train[y_train==0][:, 0], y=X_train[y_train==0][:, 1],
                             mode='markers', name='已知：苹果', marker=dict(color='red', size=10)))
    fig.add_trace(go.Scatter(x=X_train[y_train==1][:, 0], y=X_train[y_train==1][:, 1],
                             mode='markers', name='已知：西瓜', marker=dict(color='green', size=10)))
    fig.add_trace(go.Scatter(x=[new_x], y=[new_y], mode='markers', name='未知新点',
                             marker=dict(color='orange', size=15, symbol='star')))
    for point in nearest_points:
        fig.add_trace(go.Scatter(x=[new_x, point[0]], y=[new_y, point[1]], mode='lines',
                                 line=dict(color='gray', dash='dot'), showlegend=False))
    fig.update_layout(title=f"K={k} 时的投票结果：判定为 {result_text}",
                      xaxis_title="甜度", yaxis_title="水分", template="plotly_white")
    return fig, f"### 最终判定结果：**{result_text}**"

# ==========================================
# 模块三：抽丝剥茧 (原生 ID3 算法核心构建 - 完美契合信息增益多分支)
# ==========================================

map_weather = {"晴": 0, "阴": 1, "雨": 2}
map_temp = {"冷": 0, "凉": 1, "热": 2}
map_humidity = {"正常": 0, "高": 1}
map_wind = {"弱": 0, "强": 1}
map_play = {"否": 0, "是": 1}

val_maps = {
    0: {0: "晴", 1: "阴", 2: "雨"},
    1: {0: "冷", 1: "凉", 2: "热"},
    2: {0: "正常", 1: "高"},
    3: {0: "弱", 1: "强"}
}
feature_names_list = ["天气", "温度", "湿度", "风力"]

try:
    if os.path.exists('Decision_tree_data.xlsx'):
        try:
            df = pd.read_excel('Decision_tree_data.xlsx')
        except:
            df = pd.read_csv('Decision_tree_data.xlsx')
    else:
        raise FileNotFoundError("未找到数据文件")
        
    for col in ['天气', '温度', '湿度', '风力', '打球?']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.strip()
            
    X_tree = np.zeros((len(df), 4))
    X_tree[:, 0] = df['天气'].map(map_weather)
    X_tree[:, 1] = df['温度'].map(map_temp)
    X_tree[:, 2] = df['湿度'].map(map_humidity)
    X_tree[:, 3] = df['风力'].map(map_wind)
    y_tree = df['打球?'].map(map_play).values
except Exception as e:
    print(f"数据加载失败，使用备用数据: {e}")
    X_tree = np.array([[0,2,1,0], [0,2,1,1], [1,2,1,0], [2,1,0,0], [2,0,0,0], [2,0,0,1], [1,0,0,1], [0,1,1,0], [0,0,0,0], [2,1,0,0], [0,1,0,1], [1,1,1,1], [1,2,0,0], [2,1,1,1]])
    y_tree = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])

# --- 自定义 ID3 决策树算法 ---
class ID3Node:
    def __init__(self, feature_idx=None, label=None, counts=None):
        self.feature_idx = feature_idx
        self.label = label
        self.counts = counts # [未打球数量, 打球数量]
        self.children = {}
        self.x = 0
        self.y = 0

def calc_entropy(y):
    counts = np.bincount(y.astype(int))
    probs = counts[counts > 0] / len(y)
    return -np.sum(probs * np.log2(probs))

def build_id3(X, y, available_features, depth, max_depth):
    counts = [np.sum(y == 0), np.sum(y == 1)]
    # 到达最大深度，或全为同一类，或无特征可分，均作为叶子节点
    if len(np.unique(y)) == 1 or depth >= max_depth or len(available_features) == 0:
        label = 1 if counts[1] >= counts[0] else 0
        return ID3Node(label=label, counts=counts)

    base_ent = calc_entropy(y)
    best_ig = -1
    best_feat = -1

    # 寻找信息增益 (IG) 最大的特征
    for f_idx in available_features:
        vals = np.unique(X[:, f_idx])
        ent_f = 0
        for v in vals:
            mask = X[:, f_idx] == v
            ent_f += (np.sum(mask) / len(y)) * calc_entropy(y[mask])
        ig = base_ent - ent_f
        if ig > best_ig:
            best_ig = ig
            best_feat = f_idx

    # 如果无法再提升信息增益
    if best_ig <= 0:
        label = 1 if counts[1] >= counts[0] else 0
        return ID3Node(label=label, counts=counts)

    node = ID3Node(feature_idx=best_feat, counts=counts)
    node.label = 1 if counts[1] >= counts[0] else 0 # 兜底预测标签
    new_features = [f for f in available_features if f != best_feat]

    # 根据该特征下的每种取值建立多分支
    for v in np.unique(X[:, best_feat]):
        mask = X[:, best_feat] == v
        if np.sum(mask) > 0:
            node.children[v] = build_id3(X[mask], y[mask], new_features, depth+1, max_depth)
            
    return node

def predict_id3(node, x):
    if not node.children:
        return node.label
    val = x[node.feature_idx]
    if val in node.children:
        return predict_id3(node.children[val], x)
    return node.label

# --- 自定义多分支树渲染器 ---
def set_positions(node, x_min, x_max, y, y_step):
    node.x = (x_min + x_max) / 2
    node.y = y
    if node.children:
        n = len(node.children)
        width = (x_max - x_min) / n
        # 确保按键值排序，展示更稳定
        for i, (val, child) in enumerate(sorted(node.children.items())):
            set_positions(child, x_min + i * width, x_min + (i+1) * width, y - y_step, y_step)

def draw_id3_tree(node, ax):
    # 绘制当前节点框内容
    box_text = f"打球天数: {node.counts[1]}\n未打球天数: {node.counts[0]}"
    if not node.children:
        res = "是" if node.label == 1 else "否"
        box_text += f"\n结论: {res}"
        bbox = dict(boxstyle="round,pad=0.5", fc="#e8f5e9" if node.label==1 else "#ffebee", ec="#81c784" if node.label==1 else "#e57373", lw=2)
    else:
        bbox = dict(boxstyle="round,pad=0.5", fc="#ffffff", ec="#9e9e9e", lw=2)
    
    ax.text(node.x, node.y, box_text, ha='center', va='center', bbox=bbox, fontsize=11, zorder=3)
    
    if node.children:
        fname = feature_names_list[node.feature_idx]
        # 在节点下方绘制当前的判断条件（如：天气）
        ax.text(node.x, node.y - 7, fname, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.4", fc="#e3f2fd", ec="#1e88e5", lw=1.5), 
                fontsize=12, color='#1565c0', fontweight='bold', zorder=4)
                
        for val, child in node.children.items():
            # 绘制分支线条
            ax.plot([node.x, child.x], [node.y - 7, child.y + 6], color='#757575', lw=2, zorder=1)
            # 在线条上绘制分支值（如：晴）
            vname = val_maps[node.feature_idx][val]
            mx, my = (node.x + child.x)/2, (node.y - 7 + child.y + 6)/2
            ax.text(mx, my, vname, ha='center', va='center', 
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"), 
                    fontsize=11, color='#6a1b9a', fontweight='bold', zorder=2)
            draw_id3_tree(child, ax)

def run_decision_tree(max_depth, weather, temp, humidity, wind):
    # 1. 训练纯正的 ID3 模型
    root = build_id3(X_tree, y_tree, available_features=[0, 1, 2, 3], depth=1, max_depth=int(max_depth)+1)
    
    # 2. 绘制树状图
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off') # 隐藏坐标轴
    ax.set_xlim(0, 100)
    ax.set_ylim(-10, 110)
    
    set_positions(root, x_min=0, x_max=100, y=100, y_step=35)
    draw_id3_tree(root, ax)
    plt.tight_layout()
    
    # 3. 对用户输入进行预测
    u_x = np.array([map_weather[weather], map_temp[temp], map_humidity[humidity], map_wind[wind]])
    prediction = predict_id3(root, u_x)
    
    if prediction == 1:
        result_html = "✅ <span style='color:green; font-weight:bold;'>建议去打球！</span>"
    else:
        result_html = "❌ <span style='color:red; font-weight:bold;'>建议呆在家里，不适合打球</span>"
        
    return fig, f"### 基于当前【深度={max_depth}】的AI树状思维，今天的决定是：{result_html}"

# ==========================================
# 模块四：小小大脑 (神经网络感知机)
# ==========================================
def calculate_perceptron(f1, f2, f3, w1, w2, w3, threshold):
    total_score = (f1 * w1) + (f2 * w2) + (f3 * w3)
    if total_score >= threshold:
        status = "🟢 达标 (神经元被激活！)"
        color = "green"
    else:
        status = "🔴 未达标 (信号太弱，未激活)"
        color = "red"
        
    equation = f"""
    ### 神经元内部计算过程：
    1. **收集信号并赋予权重：**
       (出勤率 {f1} × 权重 {w1:.1f}) + (作业 {f2} × 权重 {w2:.1f}) + (考试 {f3} × 权重 {w3:.1f})
    2. **计算加权总和：** **{total_score:.1f}**
    3. **对比激活阈值：** 门槛为 {threshold}
    
    ### 最终结果：**<span style='color:{color}'>{status}</span>**
    """
    return equation

# ==========================================
# 预计算默认状态 
# ==========================================
default_knn_fig, default_knn_text = run_knn(3, 5, 5)
default_tree_fig, default_tree_text = run_decision_tree(3, "晴", "凉", "正常", "弱")
default_nn_text = calculate_perceptron(80, 70, 90, 0.2, 0.3, 0.5, 85)

# ==========================================
# Gradio 界面组装
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(), title="AI 魔法实验室") as demo:
    gr.Markdown("# 🧪 AI 魔法实验室 —— 初中生人工智能启蒙平台")
    gr.Markdown("欢迎来到魔法实验室！在这里，我们将把复杂的 AI 算法拆解成一个个好玩的小游戏。不需要写代码，动动鼠标就能理解 AI 是如何思考的。")
    
    with gr.Tabs():
        with gr.TabItem("👁️ 模块一：视觉魔法 (真实目标检测)"):
            gr.Markdown("### AI 是怎么认出照片里的东西的？\n现在使用的是真正的 **YOLOv8 人工智能模型**！试着上传一张照片。滑动条可以调整 AI 的“自信程度”（置信度）。")
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(label="上传照片", type="numpy")
                    conf_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, value=0.5, label="置信度阈值 (Confidence)")
                with gr.Column(scale=1):
                    img_output = gr.Image(label="真实 AI 识别结果")
            conf_slider.change(fn=real_object_detection, inputs=[img_input, conf_slider], outputs=img_output)
            img_input.upload(fn=real_object_detection, inputs=[img_input, conf_slider], outputs=img_output)

        with gr.TabItem("🍎 模块二：近朱者赤 (KNN 算法)"):
            gr.Markdown("### “近朱者赤，近墨者黑”的 AI 版本\n调整滑动条生成一个“未知点”，看看 AI 是如何通过寻找最近的邻居来进行投票分类的。")
            with gr.Row():
                with gr.Column(scale=1):
                    k_slider = gr.Slider(minimum=1, maximum=9, step=2, value=3, label="K 值 (邻居数量)")
                    new_x_slider = gr.Slider(minimum=0, maximum=10, step=0.5, value=5, label="新点的甜度 (X轴)")
                    new_y_slider = gr.Slider(minimum=0, maximum=10, step=0.5, value=5, label="新点的水分 (Y轴)")
                    knn_result_text = gr.Markdown(value=default_knn_text)
                with gr.Column(scale=2):
                    knn_plot = gr.Plot(label="散点图与距离", value=default_knn_fig)
            inputs_knn = [k_slider, new_x_slider, new_y_slider]
            for comp in inputs_knn:
                comp.change(fn=run_knn, inputs=inputs_knn, outputs=[knn_plot, knn_result_text])

        with gr.TabItem("🌳 模块三：抽丝剥茧 (原生 ID3 决策树)"):
            gr.Markdown("### 到底去不去打球？\n采用最纯正的信息增益（Entropy）计算，100%还原真实的逻辑分支！在左侧输入今天的天气情况，看看 AI 的建议！")
            with gr.Row():
                with gr.Column(scale=1):
                    depth_slider = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="最大深度观察 (Max Depth)")
                    gr.Markdown("---")
                    gr.Markdown("#### 手动输入今天的情况进行测试：")
                    weather_input = gr.Radio(choices=["晴", "阴", "雨"], value="晴", label="天气")
                    temp_input = gr.Radio(choices=["冷", "凉", "热"], value="凉", label="温度")
                    humidity_input = gr.Radio(choices=["正常", "高"], value="正常", label="湿度")
                    wind_input = gr.Radio(choices=["弱", "强"], value="弱", label="风力")
                    tree_result_text = gr.Markdown(value=default_tree_text)
                with gr.Column(scale=2):
                    tree_plot = gr.Plot(label="高度定制化 ID3 决策树图", value=default_tree_fig)
            inputs_tree = [depth_slider, weather_input, temp_input, humidity_input, wind_input]
            for comp in inputs_tree:
                comp.change(fn=run_decision_tree, inputs=inputs_tree, outputs=[tree_plot, tree_result_text])

        with gr.TabItem("🧠 模块四：小小大脑 (神经网络感知机)"):
            gr.Markdown("### 模拟大脑的一个神经元\n试着调整输入分数和各项指标的“权重”，看看最后能不能激活神经元！")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📝 输入信号 (特征值)")
                    f1 = gr.Slider(0, 100, 80, label="出勤率分数")
                    f2 = gr.Slider(0, 100, 70, label="作业分数")
                    f3 = gr.Slider(0, 100, 90, label="期末考试分数")
                with gr.Column():
                    gr.Markdown("#### ⚖️ 连接通道 (权重与阈值)")
                    w1 = gr.Slider(0.0, 1.0, 0.2, label="出勤权重")
                    w2 = gr.Slider(0.0, 1.0, 0.3, label="作业权重")
                    w3 = gr.Slider(0.0, 1.0, 0.5, label="考试权重")
                    threshold = gr.Slider(0, 100, 85, label="激活阈值 (门槛)")
            with gr.Row():
                nn_output = gr.Markdown(value=default_nn_text)
            inputs_nn = [f1, f2, f3, w1, w2, w3, threshold]
            for comp in inputs_nn:
                comp.change(fn=calculate_perceptron, inputs=inputs_nn, outputs=nn_output)

if __name__ == "__main__":
    demo.launch(debug=False)