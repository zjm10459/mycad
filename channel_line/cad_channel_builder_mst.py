import logging
import time
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from shapely.geometry import Polygon

from channel_line.geometry_utils import polyline_to_polygon, find_channel

# 导入CAD控制器和解决方案
from service.cad_controlller import CADControlller
from service.cad_solution import CADSolution
from schema.cad_schema import LineWeight, CADEntities
from service.curve_shape_connector import CurveShapeConnector
from service.curve_shape_connector_gpu import CurveShapeConnectorGPU

class CADChannelBuilderMST:
    """基于最小生成树的CAD通道构建器，先构建最小生成树，再构建通道"""
    
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self.cad_solution = CADSolution(use_gpu=use_gpu)
        self.cad_controller = self.cad_solution.cad_controller
        
        # 创建连接器实例
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
        """构建CAD实例并获取图层数据"""
        return self.cad_solution.build_cad_instance()
    
    def create_channels_for_nested_entities(self, channel_width: float = 10.0, layer_name: str = "channel", visualize_mst: bool = False, mst_layer: str = "mst") -> bool:
        """
        为图层1中的图层0实体(嵌套实体)创建通道，基于最小生成树
        
        参数:
            channel_width: 通道宽度，默认为10.0mm
            layer_name: 通道所在图层名称，默认为channel
            visualize_mst: 是否可视化最小生成树
            mst_layer: 最小生成树的图层名称
            
        返回:
            bool: 操作是否成功
        """
        try:
            # 获取图层1的所有实体
            layer1_entities = self.cad_controller.entities_layer_1
            
            logging.info(f"开始处理图层1的实体，总数: {len(layer1_entities)}")
            
            if not layer1_entities:
                logging.error("图层1没有实体")
                return False
            
            # 创建通道图层
            self.cad_controller.create_layer(layer_name)
            
            # 如果需要可视化MST，创建MST图层
            if visualize_mst:
                self.cad_controller.create_layer(mst_layer)
            
            # 统计成功和失败的通道数
            success_count = 0
            fail_count = 0
            
            # 遍历图层1中的每个实体
            for entity_idx, parent_entity in enumerate(layer1_entities):
                # 获取图层0的子实体
                child_entities = parent_entity.child_entities
                
                logging.info(f"处理图层1实体 #{entity_idx+1}，句柄: {parent_entity.handle}，子实体数量: {len(child_entities)}")
                
                # 如果子实体数量不足，跳过处理
                if len(child_entities) <= 1:
                    logging.info(f"图层1实体 #{entity_idx+1} 子实体数量不足，跳过")
                    continue
                
                # 重置连接器，为每个图层1实体创建独立的连接器
                if self.use_gpu:
                    connector = CurveShapeConnectorGPU()
                else:
                    connector = CurveShapeConnector()
                
                # 将子实体添加到连接器
                entity_indices = {}  # 实体句柄到形状索引的映射
                shape_to_entity = {}  # 形状索引到实体的映射
                shape_to_poly = {}   # 形状索引到Shapely多边形的映射
                
                for child_idx, child_entity in enumerate(child_entities):
                    # 检查child_entity是否有points属性
                    if not hasattr(child_entity, 'points') or not child_entity.points:
                        logging.warning(f"  子实体 #{child_idx+1} 没有有效的points属性，跳过")
                        continue
                    
                    # 添加到连接器
                    shape_idx = connector.add_composite_shape_from_cad_entity(child_entity)
                    
                    if shape_idx is not None:
                        entity_indices[child_entity.handle] = shape_idx
                        shape_to_entity[shape_idx] = child_entity
                        
                        # 转换为Shapely多边形并存储
                        try:
                            poly = polyline_to_polygon(child_entity.points)
                            if poly and not poly.is_empty:
                                shape_to_poly[shape_idx] = poly
                                logging.info(f"  子实体 #{child_idx+1} (句柄: {child_entity.handle}) 添加成功，形状索引: {shape_idx}")
                            else:
                                logging.warning(f"  子实体 #{child_idx+1} (句柄: {child_entity.handle}) 无法创建有效多边形")
                        except Exception as e:
                            logging.warning(f"  子实体 #{child_idx+1} (句柄: {child_entity.handle}) 转换为多边形失败: {str(e)}")
                
                # 检查有效形状数量
                if len(entity_indices) <= 1:
                    logging.info(f"图层1实体 #{entity_idx+1} 有效子实体数量不足，跳过")
                    continue
                
                # 构建图
                logging.info(f"  开始构建图...")
                connector.build_graph()
                logging.info(f"  图构建完成，节点数: {len(connector.graph.nodes)}，边数: {len(connector.graph.edges)}")
                
                # 查找最小生成树
                logging.info(f"  计算最小生成树...")
                mst = connector.find_mst()
                logging.info(f"  最小生成树计算完成，边数: {len(mst.edges)}")
                
                # 如果需要可视化MST，绘制MST
                if visualize_mst:
                    self._visualize_mst(mst, mst_layer)
                
                # 对最小生成树中的每条边创建通道
                for edge_idx, edge in enumerate(mst.edges(data=True)):
                    shape1_idx, shape2_idx, edge_data = edge
                    
                    # 获取边的权重（两形状之间的距离）
                    distance = edge_data.get('weight', 0)
                    
                    # 获取边的端点
                    point1 = edge_data.get('point_i')
                    point2 = edge_data.get('point_j')
                    
                    # 获取实体句柄
                    entity1 = shape_to_entity.get(shape1_idx)
                    entity2 = shape_to_entity.get(shape2_idx)
                    
                    if entity1 and entity2:
                        logging.info(f"  边 #{edge_idx+1}: 连接形状 {shape1_idx} 和 {shape2_idx}，距离: {distance}")
                        logging.info(f"    连接实体句柄: {entity1.handle} 和 {entity2.handle}")
                        
                        # 获取Shapely多边形
                        poly1 = shape_to_poly.get(shape1_idx)
                        poly2 = shape_to_poly.get(shape2_idx)
                        
                        if poly1 and poly2:
                            # 计算通道交点
                            try:
                                channel_points = find_channel(poly1, poly2, width=channel_width)
                                
                                # 输出交点信息
                                logging.info(f"    找到{len(channel_points)}个交点")
                                
                                # 确保有足够的交点
                                if len(channel_points) < 3:
                                    logging.warning(f"    交点数量不足，尝试增大通道宽度...")
                                    # 尝试增大通道宽度
                                    channel_points = find_channel(poly1, poly2, width=channel_width*1.5)
                                    logging.info(f"    增大宽度后找到{len(channel_points)}个交点")
                                
                                # 如果交点数量足够，绘制通道
                                if len(channel_points) >= 3:
                                    # 处理通道点
                                    channel_points = self._process_channel_points(channel_points)
                                    
                                    # 绘制通道多边形
                                    success = self._draw_corridor(channel_points, layer_name, edge_idx)
                                    
                                    if success:
                                        success_count += 1
                                        logging.info(f"    通道 #{success_count} 绘制成功")
                                    else:
                                        fail_count += 1
                                        logging.error(f"    通道 #{success_count + fail_count} 绘制失败")
                                else:
                                    fail_count += 1
                                    logging.warning(f"    实体 {entity1.handle} 和 {entity2.handle} 之间交点数量不足: {len(channel_points)}")
                            except Exception as e:
                                fail_count += 1
                                logging.error(f"    计算通道失败: {str(e)}")
                                import traceback
                                logging.error(traceback.format_exc())
                        else:
                            fail_count += 1
                            logging.warning(f"    无法获取有效多边形，跳过通道创建")
                    else:
                        fail_count += 1
                        logging.warning(f"    无法获取实体信息，跳过通道创建")
            
            # 刷新视图
            self.cad_controller.refresh_view()
            
            logging.info(f"通道创建完成：成功 {success_count}，失败 {fail_count}")
            return success_count > 0
            
        except Exception as e:
            logging.error(f"创建通道失败: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def _process_channel_points(self, channel_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        处理通道点，确保有4个点并且闭合
        
        参数:
            channel_points: 原始通道点
            
        返回:
            处理后的通道点
        """
        # 如果只有3个点，添加第四个点
        if len(channel_points) == 3:
            x1, y1 = channel_points[0]
            x2, y2 = channel_points[1]
            x3, y3 = channel_points[2]
            
            # 计算向量(x3,y3)到(x2,y2)
            dx = x2 - x3
            dy = y2 - y3
            
            # 应用到(x1,y1)得到第四个点
            x4 = x1 + dx
            y4 = y1 + dy
            
            channel_points.append((x4, y4))
            logging.info(f"  计算第四个交点: ({x4:.4f}, {y4:.4f})")
        
        # 确保通道是闭合的
        if len(channel_points) >= 4 and channel_points[0] != channel_points[-1]:
            channel_points.append(channel_points[0])
        
        return channel_points
    
    def _draw_corridor(self, corridor_points, layer_name, edge_idx=None):
        """
        使用CADControlller绘制通道多边形
        
        参数:
            corridor_points: 通道的顶点列表
            layer_name: 图层名称
            edge_idx: 边的索引，用于日志
            
        返回:
            绘制操作是否成功
        """
        if not corridor_points or len(corridor_points) < 4:
            logging.warning(f"通道点数量不足: {len(corridor_points) if corridor_points else 0}")
            return False

        try:
            # 使用0.25mm的线宽(25)
            lineweight_value = 25
            
            # 使用draw_line绘制通道的所有边
            lines = []
            for i in range(len(corridor_points) - 1):
                start_point = corridor_points[i]
                end_point = corridor_points[i + 1]
                
                # 绘制一条边
                line = self.cad_controller.draw_line(
                    start_point=start_point,
                    end_point=end_point,
                    layer=layer_name,
                    color=3,  # 绿色
                    lineweight=lineweight_value
                )
                
                if not line:
                    logging.warning(f"    无法绘制通道边 {i+1}，从 {start_point} 到 {end_point}")
                    return False
                
                lines.append(line)
            
            # 检查是否成功绘制所有边
            if len(lines) == len(corridor_points) - 1:
                return True
            else:
                return False
                
        except Exception as e:
            logging.error(f"    绘制通道失败: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return False
        
    def _visualize_mst(self, mst, layer_name="mst"):
        """
        可视化最小生成树
        
        参数:
            mst: 最小生成树
            layer_name: 可视化层的名称
        """
        try:
            # 创建可视化层
            self.cad_controller.create_layer(layer_name)
            
            # 使用0.15mm的线宽(15)
            lineweight_value = 15
            
            # 为每条边绘制一条直线
            for edge in mst.edges(data=True):
                shape1_idx, shape2_idx, edge_data = edge
                
                # 获取边的端点
                point1 = edge_data.get('point_i')
                point2 = edge_data.get('point_j')
                
                if point1 is not None and point2 is not None:
                    # 绘制直线
                    self.cad_controller.draw_line(
                        start_point=point1,
                        end_point=point2,
                        layer=layer_name,
                        color=5,  # 蓝色
                        lineweight=lineweight_value
                    )
            
            # 刷新视图
            self.cad_controller.refresh_view()
            return True
        except Exception as e:
            logging.error(f"可视化最小生成树失败: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return False 