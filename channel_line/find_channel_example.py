from data import entities
from geometry_utils import polyline_to_polygon, find_channel
import matplotlib.pyplot as plt

# 加载两个形状
shape1 = polyline_to_polygon(entities['17D63'].points)
shape2 = polyline_to_polygon(entities['17D62'].points)

# 在现有代码顶部添加：
print("shape1有效性:", shape1.is_valid)
print("shape2有效性:", shape2.is_valid)

# 在现有代码中添加类型检查
print("shape1类型:", type(shape1))
print("shape2类型:", type(shape2))

# 查找通道交点
# 尝试增大通道宽度到100米
channel_points = find_channel(shape1, shape2, 100)

# 在调用find_channel前添加：
print("形状1顶点示例:", list(shape1.exterior.coords[:5]))
print("形状2顶点示例:", list(shape2.exterior.coords[:5]))
print("两形状最小距离:", shape1.distance(shape2))

# 在调用find_channel后修改为：
print(f"找到{len(channel_points)}个交点")

# 输出结果（保留4位小数）
print("通道交点坐标：")
for i, (x, y) in enumerate(channel_points, 1):
    print(f"{i}. ({x:.4f}, {y:.4f})")

# 修正后的完整代码结尾：
print("通道交点坐标：")
for i, (x, y) in enumerate(channel_points, 1):
    print(f"{i}. ({x:.4f}, {y:.4f})")

fig, ax = plt.subplots()
xs, ys = shape1.exterior.xy
ax.plot(xs, ys, 'b-') 
xs, ys = shape2.exterior.xy
ax.plot(xs, ys, 'r-')
plt.show()