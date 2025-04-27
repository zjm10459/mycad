import logging
import numpy as np
import time  # 添加时间模块导入
from typing import List, Dict, Any

from service.cad_controlller import CADControlller
from service.curve_shape_connector import CurveShapeConnector
from service.curve_shape_connector_gpu import CurveShapeConnectorGPU
from schema.cad_schema import CADEntities, LineWeight


class CADSolution:
    def __init__(self, use_gpu=True):
        self.cad_controller = CADControlller()
        
        # 使用GPU版本的连接器（如果启用并且可用）
        self.use_gpu = use_gpu
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    logging.info("使用GPU加速版本的连接器")
                    self.connector = CurveShapeConnectorGPU()
                else:
                    logging.info("GPU不可用，回退到CPU版本")
                    self.use_gpu = False
                    self.connector = CurveShapeConnector()
            except ImportError:
                logging.warning("PyTorch未安装，无法启用GPU加速，回退到CPU版本")
                self.use_gpu = False
                self.connector = CurveShapeConnector()
        else:
            logging.info("使用CPU版本的连接器")
            self.connector = CurveShapeConnector()
        
    def build_cad_instance(self) -> bool:
        """
        构建CAD实例并获取图层数据
        
        返回：
            bool: 操作是否成功
        """
        try:
            # 启动CAD应用程序
            self.cad_controller.start_cad()
            
            # 确保图层存在
            self.cad_controller.create_layer("0")
            self.cad_controller.create_layer("1")
            
            # 获取实体数据
            result = self.cad_controller.get_entities()
            
            return result
        except Exception as e:
            logging.error(f"构建CAD实例失败: {str(e)}")
            return False
    
    def get_layer_data(self, layer_name: str) -> List[CADEntities]:
        """
        获取指定图层的数据
        
        参数：
            layer_name: 图层名称
        
        返回：
            List[CADEntities]: 图层实体列表
        """
        if layer_name == "0":
            return self.cad_controller.entities_layer_0
        elif layer_name == "1":
            return self.cad_controller.entities_layer_1
        else:
            return []
    
    def create_channels_between_polylines(self, channel_width: float = 5.0) -> bool:
        """
        在图层1的子实体之间创建连接通道，在最近点处建立宽度为5的通道
        
        参数：
            channel_width: 通道宽度，默认为5.0
            
        返回：
            bool: 操作是否成功
        """
        try:
            # 获取图层1的所有实体
            base_entities = self.cad_controller.entities_layer_1
            
            logging.info(f"开始处理图层1的实体，总数: {len(base_entities)}")
            
            # 遍历每个图层1的实体
            for entity_idx, entity in enumerate(base_entities):
                # 获取其子实体（图层0的实体）
                child_entities = entity.child_entities
                
                logging.info(f"处理图层1实体 #{entity_idx+1}，句柄: {entity.handle}，子实体数量: {len(child_entities)}")
                
                if len(child_entities) <= 1:
                    logging.info(f"图层1实体 #{entity_idx+1} 子实体数量不足，跳过")
                    continue
                
                # 为每组子实体创建一个新的连接器实例
                # 这样确保每个图层1实体的子实体之间的连接是独立计算的
                if self.use_gpu:
                    connector = CurveShapeConnectorGPU()
                else:
                    connector = CurveShapeConnector()
                
                # 添加子实体到连接器和存储句柄映射
                shape_indices = []
                entity_handles = {}  # 形状索引到实体句柄的映射
                for child_idx, child_entity in enumerate(child_entities):
                    idx = connector.add_composite_shape_from_cad_entity(child_entity)
                    if idx is not None:
                        shape_indices.append(idx)
                        entity_handles[idx] = child_entity.handle
                        logging.info(f"  添加子实体 #{child_idx+1}，句柄: {child_entity.handle}")
                
                if len(shape_indices) <= 1:
                    logging.info(f"图层1实体 #{entity_idx+1} 有效子实体数量不足，跳过")
                    continue
                
                # 构建图
                connector.build_graph()
                
                if self.use_gpu:
                    logging.info(f"  GPU构建图完成，节点数: {len(connector.graph.nodes)}，边数: {len(connector.graph.edges)}")
                else:
                    logging.info(f"  构建图完成，节点数: {len(connector.graph.nodes)}，边数: {len(connector.graph.edges)}")
                
                # 找到最小生成树
                mst = connector.find_mst()
                logging.info(f"  最小生成树计算完成，边数: {len(mst.edges)}")
                
                # 创建连接通道图层
                self.cad_controller.create_layer("corridor")
                
                # 获取LineWeight对象
                from schema.cad_schema import LineWeight
                lineweight = LineWeight(lineweight=25)  # 使用0.25mm线宽 (25)
                
                # 创建通道 - 在最近点处创建宽度为5的通道
                for edge_idx, edge in enumerate(mst.edges(data=True)):
                    shape1_idx, shape2_idx, edge_data = edge
                    
                    # 获取边的权重（两形状之间的距离）
                    distance = edge_data.get('weight', 0)
                    
                    # 获取边的端点
                    point1 = edge_data.get('point_i')
                    point2 = edge_data.get('point_j')
                    
                    # 获取点在原始形状上的索引（如果由GPU方法添加）
                    shape1_idx_on_curve = edge_data.get('point_i_idx')
                    shape2_idx_on_curve = edge_data.get('point_j_idx')
                    
                    logging.info(f"  边 #{edge_idx+1}: 连接形状 {shape1_idx} 和 {shape2_idx}")
                    logging.info(f"  点索引: 形状1点索引={shape1_idx_on_curve}, 形状2点索引={shape2_idx_on_curve}")
                    
                    # 获取要连接的两个图层0多线段的句柄
                    handle1 = entity_handles.get(shape1_idx, "未知")
                    handle2 = entity_handles.get(shape2_idx, "未知")
                    logging.info(f"    连接的多线段句柄: {handle1} 和 {handle2}")
                    
                    # 如果在MST中没有找到点索引，并且使用GPU版本，重新计算最近点以获取点索引
                    if (shape1_idx_on_curve is None or shape2_idx_on_curve is None) and self.use_gpu:
                        try:
                            # 使用GPU版本计算最近点，并获取点索引
                            _, _, _, shape1_idx_on_curve, shape2_idx_on_curve = connector.closest_points_between_shapes_gpu(shape1_idx, shape2_idx)
                            logging.info(f"    计算获取到形状1点索引: {shape1_idx_on_curve}, 形状2点索引: {shape2_idx_on_curve}")
                        except Exception as e:
                            logging.warning(f"    获取点索引失败: {str(e)}")
                    
                    # 获取形状的点集
                    shape1 = connector.shapes[shape1_idx] if shape1_idx < len(connector.shapes) else None
                    shape2 = connector.shapes[shape2_idx] if shape2_idx < len(connector.shapes) else None
                    
                    # 打印更多形状信息以便调试
                    if shape1 is not None and shape2 is not None:
                        logging.debug(f"    形状 {shape1_idx} 的端点: 起点={shape1[0]}, 终点={shape1[-1]}")
                        logging.debug(f"    形状 {shape2_idx} 的端点: 起点={shape2[0]}, 终点={shape2[-1]}")
                    
                    logging.info(f"    距离: {distance}")
                    logging.info(f"    形状 {shape1_idx} 的最近点: {point1}")
                    logging.info(f"    形状 {shape2_idx} 的最近点: {point2}")
                    
                    # 创建通道 - 使用_create_corridor_from_points方法
                    if point1 is not None and point2 is not None:
                        # 创建通道的四个顶点
                        corridor_points = self._create_corridor_from_points(
                            point1, 
                            point2,
                            shape1=shape1,
                            shape2=shape2,
                            point1_idx=shape1_idx_on_curve,
                            point2_idx=shape2_idx_on_curve,
                            width=channel_width
                        )
                        
                        if corridor_points:
                            # 绘制通道
                            success = self._draw_corridor(corridor_points, edge_idx)
                            
                            if success:
                                logging.info(f"    通道绘制成功")
                            else:
                                logging.warning(f"    通道绘制失败!")
                        else:
                            logging.warning(f"    创建通道点失败!")
                    else:
                        logging.warning(f"    最近点无效，无法绘制通道!")
            
            # 刷新视图
            self.cad_controller.refresh_view()
            return True
            
        except Exception as e:
            logging.error(f"创建通道失败: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def _create_corridor_from_points(self, point1, point2, shape1=None, shape2=None, point1_idx=None, point2_idx=None, width=5.0, use_shape_points=True):
        """
        从两个点创建一个固定宽度的通道
        
        参数：
            point1: 起点坐标
            point2: 终点坐标
            shape1: 第一个形状的点集，用于获取形状上的点
            shape2: 第二个形状的点集，用于获取形状上的点
            point1_idx: 第一个形状上最近点的索引
            point2_idx: 第二个形状上最近点的索引
            width: 通道宽度，默认为5mm
            use_shape_points: 是否使用形状上的点作为通道顶点
            
        返回：
            通道的四个顶点坐标
        """
        if point1 is None or point2 is None:
            logging.warning(f"创建通道失败 - 最近点为空")
            return None
        
        # 如果需要使用形状上的实际点
        if use_shape_points and shape1 is not None and shape2 is not None and point1_idx is not None and point2_idx is not None:
            logging.info(f"使用形状上的实际点创建通道")
            
            # 获取形状1上的点
            shape1_left_idx, shape1_right_idx = self._find_offset_points_on_shape(shape1, point1_idx, width/2)
            if shape1_left_idx is not None and shape1_right_idx is not None:
                left_top = shape1[shape1_left_idx].copy()
                right_top = shape1[shape1_right_idx].copy()
                logging.info(f"形状1上的左侧点索引: {shape1_left_idx}, 右侧点索引: {shape1_right_idx}")
            else:
                # 回退到传统方法
                logging.warning(f"在形状1上找不到合适的偏移点，使用传统方法")
                return self._create_corridor_traditional(point1, point2, width)
            
            # 获取形状2上的点
            shape2_left_idx, shape2_right_idx = self._find_offset_points_on_shape(shape2, point2_idx, width/2)
            if shape2_left_idx is not None and shape2_right_idx is not None:
                left_bottom = shape2[shape2_left_idx].copy()
                right_bottom = shape2[shape2_right_idx].copy()
                logging.info(f"形状2上的左侧点索引: {shape2_left_idx}, 右侧点索引: {shape2_right_idx}")
            else:
                # 回退到传统方法
                logging.warning(f"在形状2上找不到合适的偏移点，使用传统方法")
                return self._create_corridor_traditional(point1, point2, width)
                
            logging.debug(f"创建通道 - 通道宽度: {width}")
            logging.debug(f"创建通道 - 左上顶点: {left_top}")
            logging.debug(f"创建通道 - 右上顶点: {right_top}")
            logging.debug(f"创建通道 - 右下顶点: {right_bottom}")
            logging.debug(f"创建通道 - 左下顶点: {left_bottom}")
            
            return [left_top, right_top, right_bottom, left_bottom]
        else:
            # 使用传统方法创建通道
            return self._create_corridor_traditional(point1, point2, width)
            
    def _create_corridor_traditional(self, point1, point2, width=5.0):
        """
        使用传统方法创建通道（不依赖于形状上的点）
        
        参数：
            point1: 起点坐标
            point2: 终点坐标
            width: 通道宽度
            
        返回：
            通道的四个顶点坐标
        """
        # 计算从point1到point2的向量
        direction = point2 - point1
        # 向量长度
        distance = np.linalg.norm(direction)
        logging.debug(f"创建通道 - 方向向量: {direction}")
        logging.debug(f"创建通道 - 距离: {distance}")
        
        if distance < 1e-6:  # 避免除以接近零的值
            logging.warning(f"创建通道失败 - 距离太小: {distance}")
            return None
            
        # 归一化方向向量
        direction = direction / distance
        
        # 计算法向量（垂直于方向向量）
        normal = np.array([-direction[1], direction[0]])
        
        logging.debug(f"创建通道 - 单位方向向量: {direction}")
        logging.debug(f"创建通道 - 法向量: {normal}")
        
        # 计算宽度的一半
        half_width = width / 2.0
        
        # 计算通道的四个顶点
        left_top = np.array(point1) + normal * half_width
        right_top = np.array(point1) - normal * half_width
        right_bottom = np.array(point2) - normal * half_width
        left_bottom = np.array(point2) + normal * half_width
        
        logging.debug(f"创建通道 - 通道宽度: {width}，半宽: {half_width}")
        logging.debug(f"创建通道 - 左上顶点: {left_top}")
        logging.debug(f"创建通道 - 右上顶点: {right_top}")
        logging.debug(f"创建通道 - 右下顶点: {right_bottom}")
        logging.debug(f"创建通道 - 左下顶点: {left_bottom}")
        
        # 返回四个顶点坐标
        return [left_top, right_top, right_bottom, left_bottom]
        
    def _find_offset_points_on_shape(self, shape_points, center_idx, offset_distance):
        """
        在形状上查找中心点两侧的偏移点
        
        参数：
            shape_points: 形状的点集
            center_idx: 中心点的索引
            offset_distance: 偏移距离
            
        返回：
            (left_idx, right_idx) - 左侧点和右侧点的索引，如果找不到则返回None
        """
        if center_idx is None or center_idx < 0 or center_idx >= len(shape_points):
            logging.warning(f"无效的中心点索引: {center_idx}")
            return None, None
            
        n_points = len(shape_points)
        
        # 计算从中心点向两侧移动的距离
        left_distance = 0
        right_distance = 0
        
        left_idx = center_idx
        right_idx = center_idx
        
        # 从中心向左查找
        current_idx = center_idx
        while left_distance < offset_distance and current_idx > 0:
            prev_idx = current_idx - 1
            segment_length = np.linalg.norm(shape_points[current_idx] - shape_points[prev_idx])
            if left_distance + segment_length <= offset_distance:
                left_distance += segment_length
                current_idx = prev_idx
            else:
                # 找到了合适的点
                left_idx = prev_idx
                break
                
        # 从中心向右查找 (考虑到形状可能是闭合的)
        current_idx = center_idx
        while right_distance < offset_distance and current_idx < n_points - 1:
            next_idx = current_idx + 1
            segment_length = np.linalg.norm(shape_points[current_idx] - shape_points[next_idx])
            if right_distance + segment_length <= offset_distance:
                right_distance += segment_length
                current_idx = next_idx
            else:
                # 找到了合适的点
                right_idx = next_idx
                break
                
        # 如果形状是闭合的，检查最后一个点和第一个点的连接
        if left_idx == center_idx and n_points > 1 and np.array_equal(shape_points[0], shape_points[-1]):
            # 从最后一个点向前查找
            current_idx = n_points - 1
            while left_distance < offset_distance and current_idx > 0:
                prev_idx = current_idx - 1
                segment_length = np.linalg.norm(shape_points[current_idx] - shape_points[prev_idx])
                if left_distance + segment_length <= offset_distance:
                    left_distance += segment_length
                    current_idx = prev_idx
                else:
                    left_idx = prev_idx
                    break
                    
        # 如果无法找到足够距离的点，返回最接近的点
        if left_idx == center_idx:
            left_idx = max(0, center_idx - 1)
        if right_idx == center_idx:
            right_idx = min(n_points - 1, center_idx + 1)
        
        return left_idx, right_idx
    
    def _draw_corridor(self, corridor_points, edge_idx=None):
        """
        在CAD中绘制通道
        
        参数：
            corridor_points: 通道的四个顶点
            edge_idx: 边的索引，用于日志
            
        返回：
            绘制操作是否成功
        """
        if not corridor_points or len(corridor_points) != 4:
            logging.warning(f"通道点数量不正确: {len(corridor_points) if corridor_points else 0}")
            return False
            
        # 提取通道的四个顶点
        left_top, right_top, right_bottom, left_bottom = corridor_points
        
        log_prefix = f"    通道 #{edge_idx+1}" if edge_idx is not None else "通道"
        
        logging.info(f"{log_prefix} 四个顶点:")
        logging.info(f"{log_prefix} 左上: {left_top}")
        logging.info(f"{log_prefix} 右上: {right_top}")
        logging.info(f"{log_prefix} 右下: {right_bottom}")
        logging.info(f"{log_prefix} 左下: {left_bottom}")
        
        # 创建通道图层
        self.cad_controller.create_layer("corridor")
        
        # 绘制连接线（两条平行线）
        # 注意：我们直接使用原始点，确保精确连接
        try:
            # 使用有效的线宽值
            from schema.cad_schema import LineWeight
            lineweight = LineWeight(lineweight=25)  # 使用0.25mm线宽 (25)
            
            # 绘制第一条线（左侧）
            line1 = self.cad_controller.draw_line(
                left_top, 
                left_bottom, 
                layer="corridor",
                lineweight=lineweight.lineweight
            )
            
            # 绘制第二条线（右侧）
            line2 = self.cad_controller.draw_line(
                right_top, 
                right_bottom, 
                layer="corridor",
                lineweight=lineweight.lineweight
            )
            
            # 检查绘制结果
            if all([line1, line2]):
                logging.info(f"{log_prefix} 绘制成功")
            else:
                logging.warning(f"{log_prefix} 绘制部分失败")
                return False
                
        except Exception as e:
            logging.error(f"{log_prefix} 绘制失败: {str(e)}")
            return False
        
        return True

