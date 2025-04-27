import math
import os
import time

import win32com.client
import pythonwin
import logging
from typing import Tuple, List, Optional, Dict
import win32com.client
import pythoncom
import numpy as np

from config import config
from schema.cad_schema import *
from service.calculate_custom import *
from service.curve_shape_connector import CurveShapeConnector


class CADControlller:
    def __init__(self):
        self.app = config.CAD_TYPE
        self.doc = None
        self.entities_layer_0 = []
        self.entities_layer_1 = []

    def start_cad(self):
        try:
            self.app = win32com.client.GetActiveObject(self.app)
        except Exception as e:
            logging.info(f"未找到运行中的{self.app}实例")
            raise
        try:
            if self.app.Documents.Count == 0:
                logging.info(f"加载新文档...")
                self.doc = self.app.Documents.Add()
            else:
                logging.info("获取活动文档...")
                self.doc = self.app.ActiveDocument
        except Exception as doc_ex:
            # 如果获取文档失败，强制创建新文档
            logging.warning(f"获取文档失败，尝试创建新文档: {str(doc_ex)}")
            try:
                # 关闭所有打开的文档
                for i in range(self.app.Documents.Count):
                    try:
                        self.app.Documents.Item(i).Close(False)  # 不保存
                    except:
                        pass

                # 创建新文档
                self.doc = self.app.Documents.Add()
            except Exception as new_doc_ex:
                logging.error(f"创建新文档失败: {str(new_doc_ex)}")
                raise

    def is_running(self) -> bool:
        """检查CAD是否正在运行"""
        return self.app is not None and self.doc is not None

    def save_drawing(self, file_path: str) -> bool:
        """保存当前图纸到指定路径"""
        if not self.is_running():
            logging.error("CAD未运行，无法保存图纸")
            return False

        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 保存文件
            self.doc.SaveAs(file_path)
            logging.info(f"图纸已保存到: {file_path}")
            return True
        except Exception as e:
            logging.error(f"保存图纸失败: {str(e)}")
            return False

    def refresh_view(self) -> None:
        """刷新CAD视图"""
        if self.is_running():
            try:
                self.doc.Regen(1)  # acAllViewports = 1
            except Exception as e:
                logging.error(f"刷新视图失败: {str(e)}")

    def create_layer(self, layer_name: str) -> bool:
        if not self.is_running():
            try:
                self.start_cad()
            except Exception as e:
                logging.error(f"启动cad失败: {str(e)}")
                return False

        try:
            # 检查图层是否已存在
            for i in range(self.doc.Layers.Count):
                if self.doc.Layers.Item(i).Name == layer_name:
                    # 图层已存在，激活它
                    self.doc.ActiveLayer = self.doc.Layers.Item(i)
                    return True

            # 创建新图层
            new_layer = self.doc.Layers.Add(layer_name)

            # 图层不设置颜色，设置里面的实体颜色
            # # 设置颜色
            # if isinstance(color, int):
            #     # 使用颜色索引
            #     new_layer.Color = color
            # elif isinstance(color, tuple) and len(color) == 3:
            #     # 使用RGB值
            #     r, g, b = color
            #     # 设置TrueColor
            #     new_layer.TrueColor = self._create_true_color(r, g, b)

            # 设置为当前图层
            self.doc.ActiveLayer = new_layer
            logging.info(f"已创建新图层: {layer_name}")  # , 颜色: {color}
            return True
        except Exception as e:
            logging.error(f"创建图层时出错: {str(e)}")
            return False

    def get_entities(self):
        """获取CAD文档中的所有实体"""
        if not self.is_running():
            try:
                self.start_cad()
            except Exception as e:
                logging.error(f"启动cad失败: {str(e)}")
                return False
                
        # 清空之前的实体数据
        self.entities_layer_0 = []
        self.entities_layer_1 = []
        
        try:
            msp = self.doc.ModelSpace
            
            # 遍历所有实体
            for entity in msp:
                #收集图层1数据
                if entity.layer == "1" and hasattr(entity, 'Coordinates'):
                    entity_data = CADEntities(
                        handle=entity.Handle,
                        layer=entity.Layer,
                        color=entity.color,
                        entity_type=entity.Linetype,
                        points = get_points(entity)
                    )
                    self.entities_layer_1.append(entity_data)
                # 收集图层0数据
                if entity.layer == "0" and hasattr(entity, 'Coordinates'):
                    entity_data = CADEntities(
                        handle=entity.Handle,
                        layer=entity.Layer,
                        color=entity.color,
                        entity_type=entity.Linetype,
                        points=get_points(entity)
                    )
                    self.entities_layer_0.append(entity_data)
        except Exception as e:
            logging.error(f"获取实体时出错: {str(e)}")
            return False
        base_entity = sorted_base_entities(self.entities_layer_1)
        self.entities_layer_1 = dived_child_to_base(base_entity, self.entities_layer_0, self.entities_layer_1)
        return True
            


    def draw_line(self, start_point: Tuple[float, float, float],
                  end_point: Tuple[float, float, float], layer: str = None, color: int = None,
                  lineweight: LineWeight = None) -> bool:
        if not self.is_running():
            try:
                self.start_cad()
            except Exception as e:
                logging.error(f"启动cad失败: {str(e)}")
                return False
        try:
            if len(start_point) == 2:
                start_point = (start_point[0], start_point[1], 0)
            if len(end_point) == 2:
                end_point = (end_point[0], end_point[1], 0)

            # 使用VARIANT包装坐标点数据
            start_array = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8,
                                                  [start_point[0], start_point[1], start_point[2]])
            end_array = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8,
                                                [end_point[0], end_point[1], end_point[2]])
            line = self.doc.ModelSpace.AddLine(start_array, end_array)

            if layer:
                self.create_layer(layer)
                line.layer = layer
            if color:
                line.Color = color
            if lineweight:
                line.LineWeight = lineweight

            self.refresh_view()

            logging.debug(
                f"已绘制直线: 起点{start_point}, 终点{end_point}, 图层{layer if layer else '默认'}, 颜色{color if color is not None else '默认'}, 句柄{line.Handle if hasattr(line, 'Handle') else '未知'}")
            return line
        except Exception as e:
            logging.error(f"绘制直线时出错: {str(e)}")
            return False
