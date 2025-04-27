import time
import logging
import os
import argparse

from service.cad_solution import CADSolution

def parse_args():
    parser = argparse.ArgumentParser(description='CAD连接器示例程序')
    parser.add_argument('--use-gpu', action='store_true', help='使用GPU加速计算最近点')
    parser.add_argument('--channel-width', type=float, default=10.0, help='平行线之间的间隔，默认5.0mm')
    parser.add_argument('--verbose', action='store_true', help='显示详细日志')
    return parser.parse_args()

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
            logging.FileHandler('logs/cad_connector.log', mode='w'),  # 文件输出
            logging.StreamHandler()  # 控制台输出
        ]
    )

    logging.info("======= 开始执行 =======")
    # 记录开始时间
    total_start_time = time.time()
    
    # 显示加速模式
    if args.use_gpu:
        logging.info("启用GPU加速模式")
    else:
        logging.info("使用CPU计算模式")
    
    logging.info(f"平行线间隔设置为: {args.channel_width}mm")
    
    # 创建CAD解决方案实例
    cad_solution = CADSolution(use_gpu=args.use_gpu)
    
    # 构建CAD实例和获取图层数据
    logging.info("构建CAD实例...")
    build_start_time = time.time()
    if not cad_solution.build_cad_instance():
        logging.error("构建CAD实例失败")
        return
    build_end_time = time.time()
    logging.info(f"CAD实例构建耗时: {build_end_time - build_start_time:.2f}秒")
    
    # 获取图层数据
    logging.info("获取图层数据...")
    layer_0_entities = cad_solution.get_layer_data("0")
    layer_1_entities = cad_solution.get_layer_data("1")
    
    logging.info(f"图层0实体数量: {len(layer_0_entities)}")
    logging.info(f"图层1实体数量: {len(layer_1_entities)}")
    
    # 输出子实体数量
    for i, entity in enumerate(layer_1_entities):
        logging.info(f"图层1实体 #{i+1} 的子实体数量: {len(entity.child_entities)}")
    
    # 在图层1的子实体间创建通道
    logging.info(f"创建平行线连接，间隔: {args.channel_width}mm...")
    channel_start_time = time.time()
    if cad_solution.create_channels_between_polylines(channel_width=args.channel_width):
        channel_end_time = time.time()
        channel_time = channel_end_time - channel_start_time
        logging.info(f"平行线连接创建成功，耗时: {channel_time:.2f}秒")
    else:
        logging.error("平行线连接创建失败")
    
    # 记录结束时间
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    logging.info(f"总执行时间: {total_time:.2f}秒")
    logging.info("======= 执行完成 =======")
    logging.info(f"日志文件保存在: {os.path.abspath('logs/cad_connector.log')}")

if __name__ == "__main__":
    main() 