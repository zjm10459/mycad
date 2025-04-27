import logging
import time
from shapely.geometry import Polygon, LineString
from typing import List, Tuple, Dict, Any, Optional

from channel_line.geometry_utils import polyline_to_polygon, find_channel

# 导入CAD控制器和解决方案
from service.cad_controlller import CADControlller
from service.cad_solution import CADSolution
from schema.cad_schema import LineWeight, CADEntities

class CADChannelBuilder:
    """基于channel_line模块的CAD通道构建器，直接从AutoCAD读取数据"""
    
    def __init__(self, use_gpu=False):
        self.cad_solution = CADSolution(use_gpu=use_gpu)
        self.cad_controller = self.cad_solution.cad_controller
    
    def build_cad_instance(self) -> bool:
        """构建CAD实例并获取图层数据"""
        return self.cad_solution.build_cad_instance()
    
    def create_channels_for_nested_entities(self, channel_width: float = 10.0, layer_name: str = "channel") -> bool:
        """
        为图层1中的图层0实体(嵌套实体)创建通道
        
        参数:
            channel_width: 通道宽度，默认为10.0mm
            layer_name: 通道所在图层名称，默认为channel
            
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
                
                # 将子实体转换为Shapely多边形
                entity_polygons = {}
                for child_idx, child_entity in enumerate(child_entities):
                    try:
                        # 检查child_entity是否有points属性
                        if not hasattr(child_entity, 'points') or not child_entity.points:
                            logging.warning(f"  子实体 #{child_idx+1} 没有有效的points属性，跳过")
                            continue
                            
                        poly = polyline_to_polygon(child_entity.points)
                        if poly and not poly.is_empty:
                            entity_polygons[child_entity.handle] = {
                                'polygon': poly,
                                'entity': child_entity
                            }
                            logging.info(f"  子实体 #{child_idx+1} (句柄: {child_entity.handle}) 多边形计算成功")
                        else:
                            logging.warning(f"  子实体 #{child_idx+1} (句柄: {child_entity.handle}) 无法创建有效多边形")
                    except Exception as e:
                        logging.warning(f"  子实体 #{child_idx+1} (句柄: {child_entity.handle}) 转换为多边形失败: {str(e)}")
                
                # 如果有效多边形数量不足，跳过处理
                if len(entity_polygons) <= 1:
                    logging.info(f"图层1实体 #{entity_idx+1} 有效子实体多边形数量不足，跳过")
                    continue
                
                # 对每对子实体创建通道
                handles = list(entity_polygons.keys())
                for i in range(len(handles)):
                    for j in range(i+1, len(handles)):
                        handle1 = handles[i]
                        handle2 = handles[j]
                        
                        # 获取多边形和原始实体
                        poly1 = entity_polygons[handle1]['polygon']
                        poly2 = entity_polygons[handle2]['polygon']
                        
                        # 计算距离
                        distance = poly1.distance(poly2)
                        
                        # 只为距离在一定范围内的多边形创建通道
                        # 这里设定为距离小于通道宽度的5倍
                        max_distance = channel_width * 5
                        if distance <= max_distance:
                            logging.info(f"  为子实体 {handle1} 和 {handle2} 创建通道，距离: {distance}")
                            
                            # 计算通道交点
                            try:
                                channel_points = find_channel(poly1, poly2, width=channel_width)
                                
                                # 输出交点信息
                                logging.info(f"  找到{len(channel_points)}个交点")
                                
                                # 确保有足够的交点
                                if len(channel_points) < 3:
                                    logging.warning(f"  交点数量不足，尝试增大通道宽度...")
                                    # 尝试增大通道宽度
                                    channel_points = find_channel(poly1, poly2, width=channel_width*1.5)
                                    logging.info(f"  增大宽度后找到{len(channel_points)}个交点")
                                
                                # 如果交点数量足够，绘制通道
                                if len(channel_points) >= 3:
                                    # 处理通道点
                                    channel_points = self._process_channel_points(channel_points)
                                    
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
                                        logging.info(f"  通道 #{success_count} 绘制成功")
                                    else:
                                        fail_count += 1
                                        logging.error(f"  通道 #{success_count + fail_count} 绘制失败")
                                else:
                                    fail_count += 1
                                    logging.warning(f"  实体 {handle1} 和 {handle2} 之间交点数量不足: {len(channel_points)}")
                            except Exception as e:
                                fail_count += 1
                                logging.error(f"  计算通道失败: {str(e)}")
            
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
        if channel_points[0] != channel_points[-1]:
            channel_points.append(channel_points[0])
        
        return channel_points 