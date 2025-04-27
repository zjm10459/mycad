import logging
import os
import argparse
import random
from collections import defaultdict
import win32com.client
import pythoncom

from service.cad_solution import CADSolution
from schema.cad_schema import LineWeight

def parse_args():
    parser = argparse.ArgumentParser(description='CAD图层转换器增强版')
    parser.add_argument('--verbose', action='store_true', help='显示详细日志')
    parser.add_argument('--add-labels', action='store_true', help='添加图层0多线段数量标签', default=True)
    parser.add_argument('--text-height', type=float, default=5.0, help='文字高度')
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
            logging.FileHandler('logs/color_polylines_enhanced.log', mode='w'),  # 文件输出
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
    
    # 遍历每个图层1的实体，并统计其中包含的图层0多线段数量
    for i, entity in enumerate(layer_1_entities):
        child_count = len(entity.child_entities)
        logging.info(f"图层1实体 #{i+1} (句柄: {entity.handle}) 包含 {child_count} 个图层0多线段")
        
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
    
    # 刷新视图
    cad_solution.cad_controller.refresh_view()
    
    logging.info("======= 执行完成 =======")
    logging.info(f"日志文件保存在: {os.path.abspath('logs/color_polylines_enhanced.log')}")

if __name__ == "__main__":
    main() 