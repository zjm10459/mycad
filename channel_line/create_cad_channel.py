import logging
import time
from shapely.geometry import Polygon, LineString
from typing import List, Tuple, Dict, Any, Optional

from channel_line.data import PolylineSegment, Entity, entities
from channel_line.geometry_utils import polyline_to_polygon, find_channel

# 导入CAD控制器和解决方案
from service.cad_controlller import CADControlller
from service.cad_solution import CADSolution
from schema.cad_schema import LineWeight

class ChannelBuilder:
    """通道构建器，基于channel_line模块实现通道构建功能"""
    
    def __init__(self, cad_solution: Optional[CADSolution] = None):
        self.cad_solution = cad_solution if cad_solution else CADSolution(use_gpu=False)
        self.cad_controller = self.cad_solution.cad_controller
    
    def build_cad_instance(self) -> bool:
        """构建CAD实例"""
        return self.cad_solution.build_cad_instance()
    
    def create_channel_between_shapes(self, 
                                     entity1_handle: str, 
                                     entity2_handle: str, 
                                     channel_width: float = 10.0,
                                     layer_name: str = "channel") -> bool:
        """
        在两个形状之间创建通道
        
        参数:
            entity1_handle: 第一个实体的句柄
            entity2_handle: 第二个实体的句柄
            channel_width: 通道宽度，默认为10.0mm
            layer_name: 通道所在图层名称，默认为channel
            
        返回:
            bool: 操作是否成功
        """
        try:
            # 确保通道图层存在
            self.cad_controller.create_layer(layer_name)
            
            # 从数据中获取实体
            entity1 = next((e for e in entities.values() if e.handle == entity1_handle), None)
            entity2 = next((e for e in entities.values() if e.handle == entity2_handle), None)
            
            if not entity1 or not entity2:
                logging.error(f"无法找到指定句柄的实体: {entity1_handle}, {entity2_handle}")
                return False
            
            # 转换为Shapely多边形
            poly1 = polyline_to_polygon(entity1.points)
            poly2 = polyline_to_polygon(entity2.points)
            
            # 计算通道交点
            logging.info(f"计算通道交点, 通道宽度: {channel_width}mm")
            channel_points = find_channel(poly1, poly2, width=channel_width)
            
            # 输出交点信息
            logging.info(f"找到{len(channel_points)}个交点")
            for i, (x, y) in enumerate(channel_points, 1):
                logging.info(f"交点 {i}: ({x:.4f}, {y:.4f})")
            
            # 确保有4个交点
            if len(channel_points) != 4:
                logging.warning(f"通道交点数量不是4个，实际为{len(channel_points)}个")
                # 如果只有2个交点，尝试构建更宽的通道
                if len(channel_points) == 2:
                    logging.info(f"尝试增大通道宽度并重新计算...")
                    channel_points = find_channel(poly1, poly2, width=channel_width*1.5)
                    logging.info(f"增大宽度后找到{len(channel_points)}个交点")
            
            # 画出通道
            if len(channel_points) >= 3:
                # 创建闭合多边形
                if len(channel_points) == 3:
                    # 如果只有3个点，添加第四个点
                    # 计算反向向量并创建第四个点
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
                    logging.info(f"计算第四个交点: ({x4:.4f}, {y4:.4f})")
                
                # 确保通道是闭合的
                if channel_points[0] != channel_points[-1]:
                    channel_points.append(channel_points[0])
                
                # 绘制通道多边形
                lineweight = LineWeight(lineweight=25)  # 使用0.25mm线宽 (25)
                
                success = self.cad_controller.draw_polyline(
                    points=channel_points,
                    layer_name=layer_name,
                    closed=True,
                    lineweight=lineweight,
                    color=3  # 绿色
                )
                
                if success:
                    logging.info("通道绘制成功")
                    # 刷新视图
                    self.cad_controller.refresh_view()
                    return True
                else:
                    logging.error("通道绘制失败")
            else:
                logging.error(f"交点数量不足，无法绘制通道: {len(channel_points)}")
            
            return False
            
        except Exception as e:
            logging.error(f"创建通道失败: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def create_channels_between_all_shapes(self, channel_width: float = 10.0, layer_name: str = "channel") -> bool:
        """
        在所有图层形状之间创建通道
        
        参数:
            channel_width: 通道宽度
            layer_name: 通道所在图层名称
            
        返回:
            bool: 操作是否成功
        """
        try:
            # 获取所有实体
            base_entities = self.cad_controller.entities_layer_1
            
            logging.info(f"开始处理图层1的实体，总数: {len(base_entities)}")
            
            # 创建字典存储多边形
            entity_polygons = {}
            
            # 预先计算所有多边形
            for entity_idx, entity in enumerate(base_entities):
                try:
                    poly = polyline_to_polygon(entity.points)
                    if poly and not poly.is_empty:
                        entity_polygons[entity.handle] = poly
                        logging.info(f"实体 #{entity_idx+1} (句柄: {entity.handle}) 多边形计算成功")
                    else:
                        logging.warning(f"实体 #{entity_idx+1} (句柄: {entity.handle}) 无法创建有效多边形")
                except Exception as e:
                    logging.warning(f"实体 #{entity_idx+1} (句柄: {entity.handle}) 转换为多边形失败: {str(e)}")
            
            # 创建通道图层
            self.cad_controller.create_layer(layer_name)
            
            # 统计成功和失败的通道数
            success_count = 0
            fail_count = 0
            
            # 自动寻找并创建合适的通道
            handles = list(entity_polygons.keys())
            for i in range(len(handles)):
                for j in range(i+1, len(handles)):
                    handle1 = handles[i]
                    handle2 = handles[j]
                    
                    # 获取多边形
                    poly1 = entity_polygons[handle1]
                    poly2 = entity_polygons[handle2]
                    
                    # 计算距离
                    distance = poly1.distance(poly2)
                    
                    # 只为距离在一定范围内的多边形创建通道
                    # 这里设定为距离小于通道宽度的5倍
                    max_distance = channel_width * 5
                    if distance <= max_distance:
                        logging.info(f"为实体 {handle1} 和 {handle2} 创建通道，距离: {distance}")
                        
                        # 计算通道交点
                        channel_points = find_channel(poly1, poly2, width=channel_width)
                        
                        # 如果交点数量足够，绘制通道
                        if len(channel_points) >= 3:
                            # 处理与前面相同的逻辑
                            # 如果只有3个点，添加第四个点
                            if len(channel_points) == 3:
                                # 计算反向向量并创建第四个点
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
                            
                            # 确保通道是闭合的
                            if channel_points[0] != channel_points[-1]:
                                channel_points.append(channel_points[0])
                            
                            # 绘制通道多边形
                            lineweight = LineWeight(lineweight=25)  # 使用0.25mm线宽 (25)
                            
                            success = self.cad_controller.draw_polyline(
                                points=channel_points,
                                layer_name=layer_name,
                                closed=True,
                                lineweight=lineweight,
                                color=3  # 绿色
                            )
                            
                            if success:
                                success_count += 1
                                logging.info(f"通道 #{success_count} 绘制成功")
                            else:
                                fail_count += 1
                                logging.error(f"通道 #{success_count + fail_count} 绘制失败")
                        else:
                            fail_count += 1
                            logging.warning(f"实体 {handle1} 和 {handle2} 之间交点数量不足: {len(channel_points)}")
            
            # 刷新视图
            self.cad_controller.refresh_view()
            
            logging.info(f"通道创建完成：成功 {success_count}，失败 {fail_count}")
            return success_count > 0
            
        except Exception as e:
            logging.error(f"创建通道失败: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return False 