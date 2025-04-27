import logging
import os
import argparse
import random
from collections import defaultdict
import win32com.client
import pythoncom
import math

from service.cad_solution import CADSolution
from schema.cad_schema import LineWeight

def parse_args():
    parser = argparse.ArgumentParser(description='CAD图层转换器完整版')
    parser.add_argument('--verbose', action='store_true', help='显示详细日志')
    parser.add_argument('--add-labels', action='store_true', help='添加图层0多线段数量标签', default=True)
    parser.add_argument('--add-legend', action='store_true', help='添加颜色图例', default=True)
    parser.add_argument('--text-height', type=float, default=5.0, help='文字高度')
    parser.add_argument('--legend-position', type=str, default='top-right', 
                       help='图例位置: top-left, top-right, bottom-left, bottom-right')
    return parser.parse_args()

def get_color_for_entity(index):
    """
    根据索引返回不同的颜色值
    AutoCAD颜色索引值范围: 1-255，0表示BYBLOCK，256表示BYLAYER
    常用颜色: 红=1, 黄=2, 绿=3, 青=4, 蓝=5, 洋红=6, 白=7
    """
    # 定义一组鲜明的颜色
    colors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 30, 50, 90, 110, 130, 150, 170, 190, 210, 230]
    return colors[index % len(colors)]  # 循环使用颜色列表

def get_color_name(color_index):
    """
    获取颜色索引对应的颜色名称
    """
    color_names = {
        1: "红色", 2: "黄色", 3: "绿色", 4: "青色", 5: "蓝色", 6: "洋红", 7: "白色",
        8: "深灰", 9: "浅灰", 30: "橙色", 50: "黄绿", 90: "淡绿", 110: "淡青",
        130: "淡蓝", 150: "淡紫", 170: "粉红", 190: "淡棕", 210: "淡黄", 230: "米色"
    }
    return color_names.get(color_index, f"颜色{color_index}")

def find_center_point(entity):
    """
    计算图层1实体的近似中心点，用于放置标签
    """
    x_coords = []
    y_coords = []
    
    for point_segment in entity.points:
        x_coords.append(point_segment.start_point[0])
        x_coords.append(point_segment.end_point[0])
        y_coords.append(point_segment.start_point[1])
        y_coords.append(point_segment.end_point[1])
    
    # 计算中心点
    if x_coords and y_coords:
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        return (center_x, center_y)
    
    return None

def get_drawing_extent(cad_controller):
    """
    获取图纸范围，用于放置图例
    """
    try:
        # 获取模型空间
        msp = cad_controller.doc.ModelSpace
        
        # 初始化极值
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        # 遍历所有实体
        for i in range(msp.Count):
            entity = msp.Item(i)
            
            # 获取实体的范围
            if hasattr(entity, 'Coordinates'):
                coords = entity.Coordinates
                
                # 遍历坐标
                for j in range(0, len(coords), 2):
                    if j+1 < len(coords):
                        x = coords[j]
                        y = coords[j+1]
                        
                        # 更新极值
                        min_x = min(min_x, x)
                        min_y = min(min_y, y)
                        max_x = max(max_x, x)
                        max_y = max(max_y, y)
        
        # 如果找到了有效的极值
        if min_x != float('inf') and min_y != float('inf') and max_x != float('-inf') and max_y != float('-inf'):
            return (min_x, min_y, max_x, max_y)
    
    except Exception as e:
        logging.error(f"获取图纸范围时出错: {str(e)}")
    
    # 默认返回一个合理的范围
    return (0, 0, 1000, 1000)

def create_legend(cad_controller, color_mapping, position='top-right', text_height=5.0):
    """
    创建颜色图例
    
    参数:
        cad_controller: CAD控制器实例
        color_mapping: 索引到颜色的映射字典 {entity_index: color_index}
        position: 图例位置 ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        text_height: 文字高度
    """
    try:
        # 创建图例图层
        cad_controller.create_layer("legend")
        
        # 获取图纸范围
        min_x, min_y, max_x, max_y = get_drawing_extent(cad_controller)
        
        # 确定图例起始点
        margin = 20
        legend_width = 100
        line_length = 30
        line_spacing = text_height * 1.5
        
        if position == 'top-right':
            start_x = max_x - margin - legend_width
            start_y = max_y - margin
        elif position == 'top-left':
            start_x = min_x + margin
            start_y = max_y - margin
        elif position == 'bottom-right':
            start_x = max_x - margin - legend_width
            start_y = min_y + margin + (len(color_mapping) * line_spacing)
        elif position == 'bottom-left':
            start_x = min_x + margin
            start_y = min_y + margin + (len(color_mapping) * line_spacing)
        else:
            # 默认右上角
            start_x = max_x - margin - legend_width
            start_y = max_y - margin
        
        # 添加图例标题
        title_point = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8,
                                             [start_x, start_y + line_spacing, 0])
        title = cad_controller.doc.ModelSpace.AddText("图层0多线段颜色图例", title_point, text_height * 1.2)
        title.Layer = "legend"
        title.Color = 7  # 白色
        
        # 添加每个颜色的图例项
        y_offset = 0
        for i, (entity_index, color_index) in enumerate(sorted(color_mapping.items())):
            y_pos = start_y - y_offset
            
            # 绘制示例线
            line_start = (start_x, y_pos, 0)
            line_end = (start_x + line_length, y_pos, 0)
            
            # 绘制线段
            line = cad_controller.draw_line(line_start, line_end, "legend", color_index)
            
            # 添加文字说明
            text_point = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8,
                                               [start_x + line_length + 5, y_pos, 0])
            
            # 获取颜色名称
            color_name = get_color_name(color_index)
            text = cad_controller.doc.ModelSpace.AddText(f"#{entity_index+1}: {color_name}", text_point, text_height)
            text.Layer = "legend"
            text.Color = 7  # 白色
            
            y_offset += line_spacing
        
        return True
    except Exception as e:
        logging.error(f"创建图例时出错: {str(e)}")
        return False

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 确保日志目录存在
    os.makedirs('logs', exist_ok=True)

    # 配置日志记录
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/color_polylines_complete.log', mode='w'),  # 文件输出
            logging.StreamHandler()  # 控制台输出
        ]
    )

    logging.info("======= 开始执行 =======")
    
    # 创建CAD解决方案实例
    cad_solution = CADSolution(use_gpu=False)  # 这个功能不需要GPU加速
    
    # 构建CAD实例和获取图层数据
    logging.info("构建CAD实例...")
    if not cad_solution.build_cad_instance():
        logging.error("构建CAD实例失败")
        return
    
    # 获取图层数据
    logging.info("获取图层数据...")
    layer_0_entities = cad_solution.get_layer_data("0")
    layer_1_entities = cad_solution.get_layer_data("1")
    
    logging.info(f"图层0实体数量: {len(layer_0_entities)}")
    logging.info(f"图层1实体数量: {len(layer_1_entities)}")
    
    # 创建文本图层
    if args.add_labels:
        cad_solution.cad_controller.create_layer("text_labels")
    
    # 为保存每个图层1中子实体的颜色映射
    entity_colors = {}
    
    # 遍历每个图层1的实体，并统计其中包含的图层0多线段数量
    for i, entity in enumerate(layer_1_entities):
        child_count = len(entity.child_entities)
        logging.info(f"图层1实体 #{i+1} (句柄: {entity.handle}) 包含 {child_count} 个图层0多线段")
        
        entity_colors[i] = {}  # 存储当前图层1的子实体颜色
        
        # 为每个图层0的多线段分配唯一颜色
        for j, child_entity in enumerate(entity.child_entities):
            # 获取原始实体
            try:
                # 通过句柄查找对应的原始实体对象
                cad_entity = None
                msp = cad_solution.cad_controller.doc.ModelSpace
                for k in range(msp.Count):
                    entity_obj = msp.Item(k)
                    if hasattr(entity_obj, 'Handle') and entity_obj.Handle == child_entity.handle:
                        cad_entity = entity_obj
                        break
                
                if cad_entity:
                    # 为该实体设置颜色
                    color_index = get_color_for_entity(j)
                    cad_entity.Color = color_index
                    entity_colors[i][j] = color_index  # 保存颜色
                    logging.info(f"  子实体 #{j+1} (句柄: {child_entity.handle}) 颜色设置为 {color_index}")
                else:
                    logging.warning(f"  未找到子实体 #{j+1} (句柄: {child_entity.handle}) 的原始对象")
            except Exception as e:
                logging.error(f"  设置子实体 #{j+1} (句柄: {child_entity.handle}) 颜色时出错: {str(e)}")
        
        # 添加标签显示多线段数量
        if args.add_labels and child_count > 0:
            try:
                # 获取图层1实体的中心点作为文本插入点
                center_point = find_center_point(entity)
                
                if center_point:
                    # 创建标签文本
                    text_content = f"数量: {child_count}"
                    
                    # 插入文本
                    msp = cad_solution.cad_controller.doc.ModelSpace
                    insert_point = (center_point[0], center_point[1], 0)
                    
                    # 转换为VARIANT
                    insert_point_variant = win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, 
                                                        [insert_point[0], insert_point[1], insert_point[2]])
                    
                    text = msp.AddText(text_content, insert_point_variant, args.text_height)
                    
                    # 设置文本属性
                    text.Layer = "text_labels"
                    text.Color = 7  # 白色
                    
                    logging.info(f"  为图层1实体 #{i+1} 添加了数量标签: {text_content}")
                else:
                    logging.warning(f"  无法确定图层1实体 #{i+1} 的中心点，跳过添加标签")
            except Exception as e:
                logging.error(f"  为图层1实体 #{i+1} 添加标签时出错: {str(e)}")
    
    # 添加图例
    if args.add_legend:
        for entity_idx, color_map in entity_colors.items():
            if color_map:  # 只为有子实体的图层1创建图例
                legend_created = create_legend(
                    cad_solution.cad_controller, 
                    color_map, 
                    args.legend_position, 
                    args.text_height
                )
                if legend_created:
                    logging.info(f"为图层1实体 #{entity_idx+1} 创建了颜色图例")
    
    # 刷新视图
    cad_solution.cad_controller.refresh_view()
    
    logging.info("======= 执行完成 =======")
    logging.info(f"日志文件保存在: {os.path.abspath('logs/color_polylines_complete.log')}")

if __name__ == "__main__":
    main() 