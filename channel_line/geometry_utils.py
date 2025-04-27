from shapely.ops import nearest_points  # 新增导入
from shapely.geometry import Polygon, LineString, Point
from math import atan, atan2, sin, cos  # 添加 atan2 导入
from shapely.geometry import Polygon, MultiPolygon

def polyline_to_polygon(segments):
    """处理包含直线和弧线的多段线转换"""
    coords = []
    for seg in segments:
        if seg.is_arc:
            # 使用中心点、半径和起终点计算角度
            start = Point(seg.start_point)
            end = Point(seg.end_point)
            center = Point(seg.center)
            
            # 计算起始角度和终止角度
            start_angle = atan2(start.y - center.y, start.x - center.x)
            end_angle = atan2(end.y - center.y, end.x - center.x)
            
            # 生成弧线段坐标（50个采样点）
            num_points = 50
            for i in range(num_points+1):
                angle = start_angle + (end_angle - start_angle) * i/num_points
                x = center.x + seg.radius * cos(angle)
                y = center.y + seg.radius * sin(angle)
                coords.append((x, y))
        else:
            # 直线段去重处理
            if not coords or coords[-1] != seg.start_point:
                coords.append(seg.start_point)
            coords.append(seg.end_point)
    
    # 添加闭合检查和缓冲区优化
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return Polygon(coords).buffer(0.001)  # 增大缓冲区确保闭合

def find_channel(poly1, poly2, width=10):
    """精确计算通道交点"""
    # 转换为单部件多边形
    poly1 = poly1.geoms[0] if isinstance(poly1, MultiPolygon) else poly1
    poly2 = poly2.geoms[0] if isinstance(poly2, MultiPolygon) else poly2

    # 获取有效边界线（包含顶点）
    line1 = LineString(poly1.exterior.coords)
    line2 = LineString(poly2.exterior.coords)

    # 计算双向最近点对
    pt1 = nearest_points(line1, line2)[0]
    pt2 = nearest_points(line2, line1)[0]

    # 创建通道多边形（带端盖）
    channel_line = LineString([pt1, pt2])
    channel_poly = channel_line.buffer(width/2, cap_style=3)  # 方型端盖

    # 精确计算交点（新增面类型处理）
    intersections = []
    for poly in [poly1, poly2]:
        intersection = poly.intersection(channel_poly)  # 修改为多边形整体相交
        if intersection.is_empty:
            continue
        # 新增面类型处理
        if intersection.geom_type == 'Polygon':
            intersections.extend(list(intersection.exterior.coords))
        elif intersection.geom_type == 'MultiPolygon':
            for geom in intersection.geoms:
                intersections.extend(list(geom.exterior.coords))
        elif intersection.geom_type == 'Point':
            intersections.append((intersection.x, intersection.y))
        elif intersection.geom_type == 'MultiPoint':
            intersections.extend([(p.x, p.y) for p in intersection.geoms])
        elif intersection.geom_type == 'LineString':
            intersections.extend(list(intersection.coords))
    
    # 优化去重逻辑（保留原始精度）
    unique_points = list({(x, y) for x,y in intersections})
    # 新增调试输出
    print(f"原始交点数量: {len(intersections)} 去重后: {len(unique_points)}")
    
    return sorted(unique_points, key=lambda p: channel_line.project(Point(p)))[:4]