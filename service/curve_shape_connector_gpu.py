import numpy as np
import networkx as nx
import math
import logging
import torch
import time
from typing import Tuple, List, Optional, Union


class CurveShapeConnectorGPU:
    def __init__(self):
        self.shapes = []
        self.shape_types = []  # 'polygon' or 'curve'
        self.graph = nx.Graph()

        # 确认是否有GPU可用
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            logging.info(f"使用GPU加速: {torch.cuda.get_device_name(0)}")
        else:
            logging.info("GPU不可用，回退到CPU计算")

        # 缓存
        self._shape_tensors = {}  # 保存转换为张量的形状
        self._nearest_points_cache = {}  # 缓存已计算的最近点结果

    def add_composite_shape_from_cad_entity(self, cad_entity, num_points=20):
        """
        从CADEntities对象创建复合形状
        
        参数:
            cad_entity: CADEntities对象，包含points属性
            num_points: 圆弧离散化的点数
            
        返回:
            添加的形状索引
        """
        if not hasattr(cad_entity, 'points') or not cad_entity.points:
            return None

        # 将CADEntities.points转换为segments格式
        segments = []
        for point_segment in cad_entity.points:
            # 确保我们能获取到start_point和end_point
            if not hasattr(point_segment, 'start_point') or not hasattr(point_segment, 'end_point'):
                continue

            # 获取起点和终点
            start_point = tuple(point_segment.start_point) if isinstance(point_segment.start_point,
                                                                         list) else point_segment.start_point
            end_point = tuple(point_segment.end_point) if isinstance(point_segment.end_point,
                                                                     list) else point_segment.end_point

            # 检查是否为圆弧
            if hasattr(point_segment, 'is_arc') and point_segment.is_arc:
                # 圆弧需要圆心
                if hasattr(point_segment, 'center') and point_segment.center:
                    center = tuple(point_segment.center) if isinstance(point_segment.center,
                                                                       list) else point_segment.center

                    # 确定圆弧方向，默认为逆时针
                    clockwise = False
                    if hasattr(point_segment, 'clockwise'):
                        clockwise = point_segment.clockwise

                    # 获取凸度值（如果可用）
                    bulge = None
                    if hasattr(point_segment, 'bulge'):
                        bulge = point_segment.bulge
                        # 如果凸度为负，表示顺时针方向
                        if bulge is not None and bulge < 0:
                            clockwise = True

                    radius = point_segment.radius

                    segments.append({
                        'type': 'arc',
                        'center': center,
                        'start': start_point,
                        'end': end_point,
                        'clockwise': clockwise,
                        'num_points': num_points,
                        'bulge': bulge,
                        'radius': radius  # 保存原始凸度值
                    })
            else:
                # 直线段
                segments.append({
                    'type': 'line',
                    'start': start_point,
                    'end': end_point
                })

        # 打印线条信息
        handle = getattr(cad_entity, 'handle', 'unknown')
        logging.debug(f"创建复合形状 (handle: {handle})，共有 {len(segments)} 个线段")
        for i, segment in enumerate(segments):
            if segment['type'] == 'line':
                logging.debug(f"  线段 {i + 1}: 类型=直线, 起点={segment['start']}, 终点={segment['end']}")
            elif segment['type'] == 'arc':
                logging.debug(f"  线段 {i + 1}: 类型=圆弧, 起点={segment['start']}, 终点={segment['end']}, "
                              f"圆心={segment['center']}, 半径={segment['radius']}, "
                              f"方向={'顺时针' if segment['clockwise'] else '逆时针'}, "
                              f"凸度值：{segment['bulge']}")

        # 使用已有的add_composite_shape方法创建形状
        shape_idx = self.add_composite_shape(segments)

        # 打印创建完成信息
        if shape_idx is not None:
            composite_shape = self.shapes[shape_idx]
            logging.debug(f"复合形状创建完成 (handle: {handle})，索引: {shape_idx}, 顶点数量: {len(composite_shape)}")

        return shape_idx

    def add_composite_shape(self, segments):
        """
        添加由直线段和圆弧组成的复合形状（闭合区间）
        
        参数:
            segments: 段列表，每个段是一个字典，包含:
                - 'type': 'line' 或 'arc'
                - 如果type是'line': 'start'和'end'键表示线段的起点和终点
                - 如果type是'arc': 'center'、'start'和'end'键，以及可选的'clockwise'键
                
        返回:
            添加的形状索引
        """
        if not segments:
            return None

        all_points = []
        last_point = None

        for segment in segments:
            if segment['type'] == 'line':
                # 添加线段
                start = segment['start']
                end = segment['end']

                # 如果这不是第一个点，并且与上一个点不连续，添加连接点
                if last_point is not None and not np.array_equal(last_point, start):
                    all_points.append(start)
                elif last_point is None:  # 如果是第一个点
                    all_points.append(start)

                all_points.append(end)
                last_point = end

            elif segment['type'] == 'arc':
                # 添加圆弧
                center = segment['center']
                start = segment['start']
                end = segment['end']
                clockwise = segment.get('clockwise', False)
                num_points = segment.get('num_points', 20)

                # 如果这不是第一个点，并且与上一个点不连续，添加连接点
                if last_point is not None and not np.array_equal(last_point, start):
                    all_points.append(start)
                elif last_point is None:  # 如果是第一个点
                    all_points.append(start)

                # 计算圆弧半径
                radius = segment['radius']

                # 计算起点和终点的角度
                start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
                end_angle = math.atan2(end[1] - center[1], end[0] - center[0])

                # 确保圆弧方向正确
                if clockwise:
                    if end_angle > start_angle:
                        end_angle -= 2 * math.pi
                else:  # 逆时针
                    if end_angle < start_angle:
                        end_angle += 2 * math.pi

                # 生成圆弧上的点，不包括起点（已添加）
                theta = np.linspace(start_angle, end_angle, num_points)[1:]
                x = center[0] + radius * np.cos(theta)
                y = center[1] + radius * np.sin(theta)

                arc_points = np.column_stack([x, y])
                for point in arc_points:
                    all_points.append(point)

                last_point = end

        # 转换为numpy数组
        composite_shape = np.array(all_points)

        # 确保形状是闭合的
        if len(composite_shape) > 1 and not np.array_equal(composite_shape[0], composite_shape[-1]):
            composite_shape = np.vstack([composite_shape, composite_shape[0]])

        self.shapes.append(composite_shape)
        self.shape_types.append('composite')

        # 清除缓存
        self._shape_tensors = {}
        self._nearest_points_cache = {}

        return len(self.shapes) - 1

    def _get_shape_tensor(self, shape_idx: int) -> torch.Tensor:
        """获取形状的张量表示，并缓存结果"""
        if shape_idx not in self._shape_tensors:
            shape_np = self.shapes[shape_idx]
            self._shape_tensors[shape_idx] = torch.tensor(shape_np, dtype=torch.float32, device=self.device)
        return self._shape_tensors[shape_idx]

    def _get_edges_tensor(self, shape_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取形状的边缘点张量（起点和终点）"""
        shape_tensor = self._get_shape_tensor(shape_idx)
        start_points = shape_tensor[:-1]  # 所有点除了最后一个
        end_points = shape_tensor[1:]  # 所有点除了第一个
        return start_points, end_points

    def point_to_edge_distance_gpu(self,
                                   points: torch.Tensor,
                                   edge_starts: torch.Tensor,
                                   edge_ends: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算多个点到多个边的距离（GPU批量计算版本）
        
        参数:
            points: 形状为[n_points, 2]的张量，表示n_points个点
            edge_starts: 形状为[n_edges, 2]的张量，表示n_edges个边的起点
            edge_ends: 形状为[n_edges, 2]的张量，表示n_edges个边的终点
            
        返回:
            (distances, projections) - 距离和投影点张量
        """
        points_expanded = points.unsqueeze(1).expand(-1, edge_starts.size(0), -1)
        edge_starts_expanded = edge_starts.unsqueeze(0).expand(points.size(0), -1, -1)
        edge_ends_expanded = edge_ends.unsqueeze(0).expand(points.size(0), -1, -1)

        # 计算边向量
        edge_vectors = edge_ends_expanded - edge_starts_expanded  # [n_points, n_edges, 2]

        # 计算边长度的平方
        edge_lengths_squared = torch.sum(edge_vectors ** 2, dim=2)  # [n_points, n_edges]

        # 从边起点到点的向量
        point_to_start_vectors = points_expanded - edge_starts_expanded  # [n_points, n_edges, 2]

        # 计算点到边的投影比例
        # 投影 = 点到起点向量 · 边向量 / |边向量|^2
        # 形状: [n_points, n_edges]
        dot_products = torch.sum(point_to_start_vectors * edge_vectors, dim=2)

        # 为避免除以零，在极小值处进行裁剪
        edge_lengths_squared_safe = torch.clamp(edge_lengths_squared, min=1e-10)
        projection_ratios = dot_products / edge_lengths_squared_safe

        # 限制投影比例在[0,1]范围内，这确保投影点在边上
        # 形状: [n_points, n_edges]
        projection_ratios_clamped = torch.clamp(projection_ratios, 0.0, 1.0)

        # 计算投影点
        # 投影点 = 边起点 + 投影比例 * 边向量
        # 形状: [n_points, n_edges, 2]
        projection_ratios_clamped_expanded = projection_ratios_clamped.unsqueeze(-1).expand(-1, -1, 2)
        projection_points = edge_starts_expanded + projection_ratios_clamped_expanded * edge_vectors

        # 计算点到投影点的距离
        # 形状: [n_points, n_edges]
        distances = torch.sqrt(torch.sum((points_expanded - projection_points) ** 2, dim=2))

        return distances, projection_points

    def closest_points_between_shapes_gpu(self, shape1_idx: int, shape2_idx: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        使用GPU加速计算两个形状之间的最近点
        严格确保返回的点位于原始形状上的实际点curve_shape_connector_gpu
        
        参数:
            shape1_idx: 第一个形状的索引
            shape2_idx: 第二个形状的索引
            
        返回:
            (min_distance, point1, point2) - 最小距离和对应的两个点
        """
        # 检查缓存
        cache_key = (min(shape1_idx, shape2_idx), max(shape1_idx, shape2_idx))
        if cache_key in self._nearest_points_cache:
            return self._nearest_points_cache[cache_key]

        # 获取原始形状的点列表
        start_time = time.time()
        shape1_np = self.shapes[shape1_idx]
        shape2_np = self.shapes[shape2_idx]

        logging.debug(
            f"GPU计算形状 {shape1_idx}({len(shape1_np)}点) 和形状 {shape2_idx}({len(shape2_np)}点) 之间的最近点")

        # 转换为PyTorch张量并放在GPU上
        shape1_tensor = torch.tensor(shape1_np, dtype=torch.float32, device=self.device)
        shape2_tensor = torch.tensor(shape2_np, dtype=torch.float32, device=self.device)

        # 计算所有点对之间的欧氏距离
        # 形状: [len(shape1), len(shape2)]
        shape1_expanded = shape1_tensor.unsqueeze(1)  # [len(shape1), 1, 2]
        shape2_expanded = shape2_tensor.unsqueeze(0)  # [1, len(shape2), 2]

        # 计算欧氏距离 (批量计算)
        distance_matrix = torch.sqrt(torch.sum((shape1_expanded - shape2_expanded) ** 2, dim=2))

        # 找到最小距离及其索引
        min_distance, indices = torch.min(distance_matrix.view(-1), dim=0)
        min_distance_val = min_distance.item()

        # 计算点的索引
        shape1_idx_flat = indices.item() // len(shape2_np)
        shape2_idx_flat = indices.item() % len(shape2_np)

        # 获取实际点的坐标 (从原始NumPy数组中获取，确保精确度)
        point1 = shape1_np[shape1_idx_flat].copy()  # 确保是副本，避免修改原始数据
        point2 = shape2_np[shape2_idx_flat].copy()

        end_time = time.time()
        logging.debug(f"GPU计算完成，耗时: {end_time - start_time:.4f}秒")
        logging.debug(f"GPU计算最近点结果 - 距离: {min_distance_val}")
        logging.debug(f"GPU计算最近点结果 - 形状 {shape1_idx} 点索引: {shape1_idx_flat}, 点坐标: {point1}")
        logging.debug(f"GPU计算最近点结果 - 形状 {shape2_idx} 点索引: {shape2_idx_flat}, 点坐标: {point2}")

        # 缓存结果
        result = (min_distance_val, point1, point2)
        self._nearest_points_cache[cache_key] = result

        return result

    def build_graph(self):
        """构建图，节点是形状，边是它们之间的连接"""
        n_shapes = len(self.shapes)
        logging.info(f"开始使用GPU构建图，共有 {n_shapes} 个形状")

        start_time = time.time()

        # 为每个形状添加节点
        for i in range(n_shapes):
            self.graph.add_node(i)

        if n_shapes <= 1:
            logging.info("形状数量不足，无法构建连接")
            return

        # 清除缓存
        self._nearest_points_cache = {}

        # 提前将所有形状转换为张量并缓存
        for i in range(n_shapes):
            self._get_shape_tensor(i)

        # 计算并添加所有形状对之间的边
        edge_count = 0
        for i in range(n_shapes):
            for j in range(i + 1, n_shapes):
                dist, point_i, point_j = self.closest_points_between_shapes_gpu(i, j)
                self.graph.add_edge(i, j, weight=dist, point_i=point_i, point_j=point_j)
                edge_count += 1

        end_time = time.time()
        logging.info(f"GPU构建图完成，共添加 {edge_count} 条边，总耗时: {end_time - start_time:.4f}秒")

    def find_mst(self):
        """查找图的最小生成树"""
        return nx.minimum_spanning_tree(self.graph, weight='weight')

    def _find_tangent_direction(self, shape_points, point):
        """
        找到点在形状上的位置，并计算该点的切线方向
        
        参数:
            shape_points: 形状上的点集
            point: 需要查找的点
            
        返回:
            切线方向的单位向量
        """
        # 找到形状上最接近给定点的点
        min_distance = float('inf')
        closest_idx = -1

        for i, shape_point in enumerate(shape_points):
            dist = np.linalg.norm(shape_point - point)
            if dist < min_distance:
                min_distance = dist
                closest_idx = i

        # 如果找不到合适的点，返回默认方向
        if closest_idx == -1 or min_distance > 0.1:  # 设置一个小阈值
            logging.warning(f"无法在形状上找到足够接近的点，使用默认方向")
            return np.array([1.0, 0.0])  # 默认沿X轴方向

        # 根据最近点的索引，确定相邻点来计算切线方向
        n_points = len(shape_points)

        # 如果是首尾点，特殊处理
        if closest_idx == 0:
            prev_idx = n_points - 2  # 闭合形状的倒数第二个点
            next_idx = 1
        elif closest_idx == n_points - 1:
            prev_idx = n_points - 2
            next_idx = 0  # 回到起点
        else:
            prev_idx = closest_idx - 1
            next_idx = closest_idx + 1

        # 计算切线方向（使用相邻点的差值）
        tangent = shape_points[next_idx] - shape_points[prev_idx]
        tangent_norm = np.linalg.norm(tangent)

        # 避免除以零
        if tangent_norm < 1e-6:
            logging.warning(f"切线长度接近零，使用默认方向")
            return np.array([1.0, 0.0])  # 默认沿X轴方向

        # 返回归一化的切线向量
        return tangent / tangent_norm

