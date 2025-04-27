import logging
import math
from collections import defaultdict
from email.policy import default

from schema.cad_schema import PolylineSegment


def calculate_arc_center(start_point, end_point, bulge):
    """计算圆弧的圆心坐标

    参数:
        start_point: 起点坐标 (x1, y1)
        end_point: 终点坐标 (x2, y2)
        bulge: 凸度值

    返回:
        圆心坐标 (center_x, center_y)
    """
    import math

    # 1. 计算弦长
    x1, y1 = start_point
    x2, y2 = end_point
    chord = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # 2. 计算矢高 (sagitta)
    sagitta = abs(bulge) * chord / 2

    # 3. 计算半径
    # 使用几何关系: r = (chord/2)²/(2*sagitta) + sagitta/2
    radius = ((chord / 2) ** 2 + sagitta ** 2) / (2 * sagitta)

    # 4. 计算弦的中点
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # 5. 计算从起点到终点的向量
    dx = x2 - x1
    dy = y2 - y1

    # 6. 计算法向量（垂直于弦）
    # 如果bulge为正，圆弧在向量的左侧；如果为负，在右侧
    nx = -dy  # 法向量x分量
    ny = dx  # 法向量y分量

    # 单位化法向量
    length = math.sqrt(nx ** 2 + ny ** 2)
    if length > 0:
        nx /= length
        ny /= length

    # 7. 计算圆心到中点的距离
    # 使用勾股定理: h² + (chord/2)² = radius²
    h = math.sqrt(radius ** 2 - (chord / 2) ** 2)

    # 根据bulge符号确定圆心位置
    if bulge < 0:
        h = -h  # 圆弧在弦的另一侧

    # 8. 计算圆心坐标
    center_x = mid_x + h * nx
    center_y = mid_y + h * ny

    return [center_x, center_y]


def get_points(entity):
    try:
        coords = []
        for i in range(0, len(entity.Coordinates) - 1, 2):
            coords.append((entity.Coordinates[i], entity.Coordinates[i + 1]))
    except Exception as e:
        logging.error(f"获取handle为{entity.Handle}，图层为{entity.Layer}时出错，错误信息：{e}")
        return False
    points = []
    # 处理每个段
    for i in range(len(coords) - 1):

        start_point = coords[i]
        end_point = coords[i + 1]

        try:
            # 获取凸度值
            bulge = entity.GetBulge(i)
            is_arc = bulge != 0

            if is_arc:
                # 计算圆心
                center = calculate_arc_center(start_point, end_point, bulge)

                # 计算半径 (可以通过圆心到起点或终点的距离验证)
                radius = math.sqrt(
                    (center[0] - start_point[0]) ** 2 + (center[1] - start_point[1]) ** 2)
                polylinesegment = PolylineSegment(
                    start_point=start_point,
                    end_point=end_point,
                    is_arc=True,
                    center=center,
                    radius=radius,
                    bulge = bulge
                )
                points.append(polylinesegment)
            else:
                polylinesegment = PolylineSegment(
                    start_point=start_point,
                    end_point=end_point,
                )
                points.append(polylinesegment)

        except Exception as e:
            logging.error(f"处理段 {i} 出错: {str(e)}")
    for line in points:
        if line.is_arc:
            logging.info("处理圆弧：")
            logging.info(f"开始：{line.start_point},结束{line.end_point}")
            logging.info(f"圆心{line.center},凸度：{line.bulge}")
        else:
            logging.info("处理线段：")
            logging.info(f"开始：{line.start_point},结束{line.end_point}")
    return points


def sorted_base_entities(base_entities):
    base_entities_data = defaultdict()
    for entity in base_entities:
        entity_handle = entity.handle
        x_point = []
        for points in entity.points:
            x_point.append(points.start_point[0])
            x_point.append(points.end_point[0])
        base_entities_data[entity_handle] = [min(x_point), max(x_point)]
    return sorted(list(base_entities_data.items()), key=lambda x: x[1][0])


def dived_child_to_base(sorted_base, child_entities, base_entities):
    for entity in child_entities:
        point_x = entity.points[0].start_point[0]
        for handle, point in sorted_base:
            if point[0] < point_x < point[1]:
                for base_entity in base_entities:
                    if base_entity.handle == handle:
                        base_entity.child_entities.append(entity)
    return base_entities

