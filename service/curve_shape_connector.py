import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
import logging
import multiprocessing
from functools import partial


class CurveShapeConnector:
    def __init__(self):
        self.shapes = []
        self.shape_types = []  # 'polygon' or 'curve'
        self.graph = nx.Graph()
        # 获取CPU核心数，预留一个核心给系统使用
        self.num_processes = max(1, multiprocessing.cpu_count() - 1)
        logging.info(f"将使用 {self.num_processes} 个进程进行并行计算")
        # 跟踪是否在递归调用中以避免无限递归
        self._in_parallel_calculation = False

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
            start_point = tuple(point_segment.start_point) if isinstance(point_segment.start_point, list) else point_segment.start_point
            end_point = tuple(point_segment.end_point) if isinstance(point_segment.end_point, list) else point_segment.end_point
            
            # 检查是否为圆弧
            if hasattr(point_segment, 'is_arc') and point_segment.is_arc:
                # 圆弧需要圆心
                if hasattr(point_segment, 'center') and point_segment.center:
                    center = tuple(point_segment.center) if isinstance(point_segment.center, list) else point_segment.center
                    
                    # 确定圆弧方向，默认为逆时针
                    clockwise = False
                    
                    segments.append({
                        'type': 'arc',
                        'center': center,
                        'start': start_point,
                        'end': end_point,
                        'clockwise': clockwise,
                        'num_points': num_points
                    })
            else:
                # 直线段
                segments.append({
                    'type': 'line',
                    'start': start_point,
                    'end': end_point
                })
                
        # 使用已有的add_composite_shape方法创建形状
        return self.add_composite_shape(segments)

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
                radius = np.linalg.norm(np.array(start) - np.array(center))
                
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
        return len(self.shapes) - 1

    def point_to_edge_distance(self, point, edge_start, edge_end):
        """计算点到线段的距离"""
        # 计算边向量
        edge_vector = edge_end - edge_start
        # 边的长度
        edge_length = np.linalg.norm(edge_vector)

        if edge_length == 0:
            return np.linalg.norm(point - edge_start), edge_start

        # 标准化边向量
        edge_unit_vector = edge_vector / edge_length

        # 从边起点到点的向量
        start_to_point = point - edge_start

        # 将start_to_point投影到边向量上
        projection_length = np.dot(start_to_point, edge_unit_vector)

        # 如果投影在线段外，返回到最近端点的距离
        if projection_length < 0:
            return np.linalg.norm(start_to_point), edge_start
        elif projection_length > edge_length:
            return np.linalg.norm(point - edge_end), edge_end

        # 投影点
        projection_point = edge_start + projection_length * edge_unit_vector

        # 点到投影点的距离
        return np.linalg.norm(point - projection_point), projection_point

    def _process_edge_pair_chunk(self, args):
        """
        处理边对的工作函数，用于多进程计算
        
        参数:
            args: 元组 (shape1_idx, shape2_idx, chunks)，
                  其中chunks是形状1和形状2之间的边对列表
        
        返回:
            (min_dist, best_point1, best_point2, best_case, best_indices)
        """
        shape1_idx, shape2_idx, chunk = args
        
        min_dist = float('inf')
        best_point1 = None
        best_point2 = None
        best_case = -1
        best_indices = (-1, -1)
        
        for (i, edge1_start, edge1_end), (j, edge2_start, edge2_end) in chunk:
            # 计算两边之间的最近点
            # 对于边1上的每个点，找到其在边2上的最近点
            dist1, proj1 = self.point_to_edge_distance(edge1_start, edge2_start, edge2_end)
            if dist1 < min_dist:
                min_dist = dist1
                best_point1 = edge1_start
                best_point2 = proj1
                best_case = 1
                best_indices = (i, j)

            dist2, proj2 = self.point_to_edge_distance(edge1_end, edge2_start, edge2_end)
            if dist2 < min_dist:
                min_dist = dist2
                best_point1 = edge1_end
                best_point2 = proj2
                best_case = 2
                best_indices = (i, j)

            # 对于边2上的每个点，找到其在边1上的最近点
            dist3, proj3 = self.point_to_edge_distance(edge2_start, edge1_start, edge1_end)
            if dist3 < min_dist:
                min_dist = dist3
                best_point1 = proj3
                best_point2 = edge2_start
                best_case = 3
                best_indices = (i, j)

            dist4, proj4 = self.point_to_edge_distance(edge2_end, edge1_start, edge1_end)
            if dist4 < min_dist:
                min_dist = dist4
                best_point1 = proj4
                best_point2 = edge2_end
                best_case = 4
                best_indices = (i, j)
                
        return (min_dist, best_point1, best_point2, best_case, best_indices)

    def _closest_points_worker(self, shape_pair):
        """
        计算两个形状之间最近点的工作函数，用于多进程处理
        
        参数:
            shape_pair: 元组 (shape1_idx, shape2_idx)
            
        返回:
            元组 (shape1_idx, shape2_idx, distance, point1, point2)
        """
        shape1_idx, shape2_idx = shape_pair
        # 使用串行计算方法，避免递归调用并行版本
        dist, point1, point2 = self.closest_points_between_shapes_serial(shape1_idx, shape2_idx)
        return (shape1_idx, shape2_idx, dist, point1, point2)

    def build_graph(self):
        """构建图，节点是形状，边是它们之间的连接"""
        n_shapes = len(self.shapes)
        logging.info(f"开始构建图，共有 {n_shapes} 个形状")

        # 为每个形状添加节点
        for i in range(n_shapes):
            self.graph.add_node(i)
        
        if n_shapes <= 1:
            logging.info("形状数量不足，无法构建连接")
            return
        
        # 创建形状对列表
        shape_pairs = [(i, j) for i in range(n_shapes) for j in range(i + 1, n_shapes)]
        logging.info(f"需要计算 {len(shape_pairs)} 对形状之间的最近点")
        
        # 使用多进程计算
        if len(shape_pairs) >= 4 and self.num_processes > 1:  # 只有当任务足够多时才使用多进程
            logging.info(f"使用多进程计算最近点，进程数: {self.num_processes}")
            try:
                with multiprocessing.Pool(processes=self.num_processes) as pool:
                    results = pool.map(self._closest_points_worker, shape_pairs)
                    
                # 添加边到图中
                for shape1_idx, shape2_idx, dist, point1, point2 in results:
                    self.graph.add_edge(shape1_idx, shape2_idx, weight=dist, point_i=point1, point_j=point2)
                
                logging.info(f"多进程计算完成，共添加 {len(results)} 条边")
            except Exception as e:
                logging.error(f"多进程计算失败: {str(e)}，回退到单进程模式")
                # 回退到单进程模式
                self._build_graph_single_process(shape_pairs)
        else:
            # 使用单进程计算
            logging.info("使用单进程计算最近点")
            self._build_graph_single_process(shape_pairs)
    
    def _build_graph_single_process(self, shape_pairs):
        """单进程模式下构建图"""
        for i, j in shape_pairs:
            # 使用串行计算方法，避免可能的递归
            dist, point_i, point_j = self.closest_points_between_shapes_serial(i, j)
            self.graph.add_edge(i, j, weight=dist, point_i=point_i, point_j=point_j)
        logging.info(f"单进程计算完成，共添加 {len(shape_pairs)} 条边")

    def closest_points_between_shapes_parallel(self, shape1_idx, shape2_idx):
        """
        使用多进程计算两个形状之间的最近点
        适用于大型形状的计算
        """
        shape1 = self.shapes[shape1_idx]
        shape2 = self.shapes[shape2_idx]
        
        # 只有当形状足够大时才使用并行计算
        if len(shape1) * len(shape2) < 1000 or self._in_parallel_calculation:  # 阈值可以根据性能测试调整
            return self.closest_points_between_shapes_serial(shape1_idx, shape2_idx)
            
        logging.debug(f"使用并行计算形状 {shape1_idx}({len(shape1)}点) 和形状 {shape2_idx}({len(shape2)}点) 之间的最近点")
        
        # 设置递归标记，防止递归调用
        self._in_parallel_calculation = True
        
        try:
            # 将形状1和形状2的边分成多个批次
            edges1 = [(i, shape1[i], shape1[i+1]) for i in range(len(shape1) - 1)]
            edges2 = [(j, shape2[j], shape2[j+1]) for j in range(len(shape2) - 1)]
            
            # 创建所有边对组合
            edge_pairs = [(edge1, edge2) for edge1 in edges1 for edge2 in edges2]
            
            # 分割任务以便并行
            chunk_size = max(1, len(edge_pairs) // self.num_processes)
            chunks = [edge_pairs[i:i + chunk_size] for i in range(0, len(edge_pairs), chunk_size)]
            
            # 准备参数
            args = [(shape1_idx, shape2_idx, chunk) for chunk in chunks]
            
            # 使用进程池并行处理
            with multiprocessing.Pool(processes=min(self.num_processes, len(chunks))) as pool:
                results = pool.map(self._process_edge_pair_chunk, args)
                
            # 找出最小距离
            min_result = min(results, key=lambda x: x[0])
            min_distance, closest_point1, closest_point2, closest_case, (closest_i, closest_j) = min_result
            
            # 输出最近点的详细信息
            logging.debug(f"并行计算最近点结果 - 距离: {min_distance}")
            logging.debug(f"并行计算最近点结果 - 形状 {shape1_idx} 边索引: {closest_i}")
            logging.debug(f"并行计算最近点结果 - 形状 {shape2_idx} 边索引: {closest_j}")
            logging.debug(f"并行计算最近点结果 - 计算情况: {closest_case}")
            logging.debug(f"并行计算最近点结果 - 形状 {shape1_idx} 点: {closest_point1}")
            logging.debug(f"并行计算最近点结果 - 形状 {shape2_idx} 点: {closest_point2}")
            
            # 重置递归标记
            self._in_parallel_calculation = False
            return min_distance, closest_point1, closest_point2
            
        except Exception as e:
            logging.error(f"并行计算最近点失败: {str(e)}，回退到串行计算")
            # 重置递归标记
            self._in_parallel_calculation = False
            # 回退到单进程计算
            return self.closest_points_between_shapes_serial(shape1_idx, shape2_idx)

    def closest_points_between_shapes_serial(self, shape1_idx, shape2_idx):
        """
        查找两个形状之间的最近点（串行版本）
        严格确保返回的点位于原始形状上的实际点
        
        参数：
            shape1_idx: 第一个形状的索引
            shape2_idx: 第二个形状的索引
            
        返回：
            (min_distance, point1, point2) - 最小距离和对应的两个点
        """
        # 获取原始形状的点列表
        shape1 = self.shapes[shape1_idx]
        shape2 = self.shapes[shape2_idx]

        logging.debug(f"串行计算形状 {shape1_idx} 和形状 {shape2_idx} 之间的最近点")
        logging.debug(f"形状 {shape1_idx} 点数: {len(shape1)}")
        logging.debug(f"形状 {shape2_idx} 点数: {len(shape2)}")

        # 初始化
        min_distance = float('inf')
        point1_idx = -1
        point2_idx = -1
        
        # 暴力计算所有点对之间的距离
        for i, p1 in enumerate(shape1):
            for j, p2 in enumerate(shape2):
                dist = np.linalg.norm(p1 - p2)  # 欧氏距离
                if dist < min_distance:
                    min_distance = dist
                    point1_idx = i
                    point2_idx = j
        
        # 确保找到了有效的点
        if point1_idx == -1 or point2_idx == -1:
            logging.error(f"无法找到形状 {shape1_idx} 和形状 {shape2_idx} 之间的最近点")
            # 返回默认值，避免程序崩溃
            return float('inf'), np.array([0.0, 0.0]), np.array([0.0, 0.0])
        
        # 获取实际点的坐标（使用复制避免修改原始数据）
        point1 = shape1[point1_idx].copy()
        point2 = shape2[point2_idx].copy()
        
        # 输出详细信息
        logging.debug(f"串行计算最近点结果 - 距离: {min_distance}")
        logging.debug(f"串行计算最近点结果 - 形状 {shape1_idx} 点索引: {point1_idx}, 点坐标: {point1}")
        logging.debug(f"串行计算最近点结果 - 形状 {shape2_idx} 点索引: {point2_idx}, 点坐标: {point2}")

        return min_distance, point1, point2

    def closest_points_between_shapes(self, shape1_idx, shape2_idx):
        """
        查找两个形状之间的最近点
        这是一个分派方法，根据情况选择串行或并行实现
        """
        shape1 = self.shapes[shape1_idx]
        shape2 = self.shapes[shape2_idx]

        # 防止递归调用导致的栈溢出
        if self._in_parallel_calculation:
            return self.closest_points_between_shapes_serial(shape1_idx, shape2_idx)

        # 如果形状很大，考虑使用并行计算
        if len(shape1) > 100 and len(shape2) > 100 and self.num_processes > 1:
            return self.closest_points_between_shapes_parallel(shape1_idx, shape2_idx)
        else:
            return self.closest_points_between_shapes_serial(shape1_idx, shape2_idx)

    def find_mst(self):
        """查找图的最小生成树"""
        return nx.minimum_spanning_tree(self.graph, weight='weight')

    def plot_shapes_and_connections(self, mst=None):
        """绘制形状和它们之间的连接"""
        plt.figure(figsize=(10, 8))

        # 绘制每个形状
        for i, shape in enumerate(self.shapes):
            if self.shape_types[i] == 'curve':
                plt.plot(shape[:, 0], shape[:, 1], 'b-', linewidth=2)
            elif self.shape_types[i] == 'arc':
                plt.plot(shape[:, 0], shape[:, 1], 'g-', linewidth=2)
            elif self.shape_types[i] == 'composite':
                plt.plot(shape[:, 0], shape[:, 1], 'm-', linewidth=2)
            else:
                plt.plot(shape[:, 0], shape[:, 1], 'k-', linewidth=2)
            centroid = np.mean(shape, axis=0)
            plt.text(centroid[0], centroid[1], f'形状 {i}', fontsize=12)

        # 绘制坐标轴
        plt.axhline(y=0, color='k', linestyle='-', linewidth=2)
        plt.axvline(x=0, color='k', linestyle='-', linewidth=2)

        # 绘制连接（MST边）
        if mst is not None:
            for u, v, data in mst.edges(data=True):
                point_u = data['point_i']
                point_v = data['point_j']
                plt.plot([point_u[0], point_v[0]], [point_u[1], point_v[1]], 'r--', linewidth=1.5)
                mid_point = (point_u + point_v) / 2
                plt.text(mid_point[0], mid_point[1], f'{data["weight"]:.2f}', fontsize=8)

        plt.grid(True)
        plt.axis('equal')
        plt.title('形状与最小生成树连接')
        plt.xlabel('X轴')
        plt.ylabel('Y轴')
        plt.savefig('connected_shapes.png')
        plt.show()

    def create_corridor(self, shape1_idx, shape2_idx, width=5.0, use_offset=False, offset_distance=5.0):
        """
        在两个形状之间创建一个固定宽度的通道
        
        参数:
            shape1_idx: 第一个形状的索引
            shape2_idx: 第二个形状的索引
            width: 通道宽度，默认为5个单位
            use_offset: 是否使用偏移点，默认为False
            offset_distance: 最近点偏移距离，默认为5mm
            
        返回:
            通道的四个顶点坐标，顺序为[左上, 右上, 右下, 左下]
        """
        # 找到两个形状之间的最近点
        dist, point1, point2 = self.closest_points_between_shapes(shape1_idx, shape2_idx)
        
        logging.debug(f"创建通道 - 形状 {shape1_idx} 和形状 {shape2_idx} 之间的最短距离: {dist}")
        logging.debug(f"创建通道 - 形状 {shape1_idx} 的最近点: {point1}")
        logging.debug(f"创建通道 - 形状 {shape2_idx} 的最近点: {point2}")
        
        if point1 is None or point2 is None:
            logging.warning(f"创建通道失败 - 最近点为空")
            return None
            
        # 获取形状点集
        shape1 = self.shapes[shape1_idx]
        shape2 = self.shapes[shape2_idx]
            
        # 计算从point1到point2的向量
        direction = point2 - point1
        # 向量长度
        distance = np.linalg.norm(direction)
        
        logging.debug(f"创建通道 - 方向向量: {direction}")
        logging.debug(f"创建通道 - 距离: {distance}")
        
        if distance == 0:
            logging.warning(f"创建通道失败 - 距离为0")
            return None
            
        # 归一化方向向量
        direction = direction / distance
        
        # 计算法向量（垂直于方向向量）
        normal = np.array([-direction[1], direction[0]])
        
        logging.debug(f"创建通道 - 单位方向向量: {direction}")
        logging.debug(f"创建通道 - 法向量: {normal}")
        
        # 如果使用偏移，计算偏移点
        offset_point1 = point1
        offset_point2 = point2
        
        if use_offset:
            # 找到形状1最近点在形状上的位置及其切线方向
            tangent_dir1 = self._find_tangent_direction(shape1, point1)
            # 找到形状2最近点在形状上的位置及其切线方向
            tangent_dir2 = self._find_tangent_direction(shape2, point2)
            
            logging.debug(f"创建通道 - 形状 {shape1_idx} 的切线方向: {tangent_dir1}")
            logging.debug(f"创建通道 - 形状 {shape2_idx} 的切线方向: {tangent_dir2}")
            
            # 根据法向量方向，选择左侧或右侧偏移点
            # 计算点积来判断两个切线方向的指向关系
            dot_product = np.dot(tangent_dir1, tangent_dir2)
            
            # 如果点积为正，意味着两个切线方向大致一致
            if dot_product > 0:
                offset_point1 = point1 + tangent_dir1 * offset_distance
                offset_point2 = point2 - tangent_dir2 * offset_distance
            else:
                offset_point1 = point1 - tangent_dir1 * offset_distance
                offset_point2 = point2 - tangent_dir2 * offset_distance
            
            logging.debug(f"创建通道 - 偏移后形状 {shape1_idx} 的点: {offset_point1}")
            logging.debug(f"创建通道 - 偏移后形状 {shape2_idx} 的点: {offset_point2}")
            
            # 重新计算方向向量，基于偏移点
            direction = offset_point2 - offset_point1
            distance = np.linalg.norm(direction)
            
            if distance == 0:
                logging.warning(f"创建通道失败 - 偏移后距离为0，回退到原始点")
                offset_point1 = point1
                offset_point2 = point2
                direction = point2 - point1
                distance = np.linalg.norm(direction)
            else:
                direction = direction / distance
                normal = np.array([-direction[1], direction[0]])
        
        # 计算宽度的一半
        half_width = width / 2.0
        
        # 计算通道的四个顶点，使用原始点，不添加随机变化
        left_top = offset_point1 + normal * half_width
        right_top = offset_point1 - normal * half_width
        right_bottom = offset_point2 - normal * half_width
        left_bottom = offset_point2 + normal * half_width
        
        logging.debug(f"创建通道 - 通道宽度: {width}，半宽: {half_width}")
        logging.debug(f"创建通道 - 左上顶点: {left_top}")
        logging.debug(f"创建通道 - 右上顶点: {right_top}")
        logging.debug(f"创建通道 - 右下顶点: {right_bottom}")
        logging.debug(f"创建通道 - 左下顶点: {left_bottom}")
        
        # 返回四个顶点坐标
        return [left_top, right_top, right_bottom, left_bottom]
    
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
