import logging
import os
import argparse
import time

from channel_line.cad_channel_builder_mst import CADChannelBuilderMST

def parse_args():
    parser = argparse.ArgumentParser(description='基于最小生成树的CAD通道构建器')
    parser.add_argument('--channel-width', type=float, default=10.0, help='通道宽度，默认10.0mm')
    parser.add_argument('--layer', type=str, default='channel', help='通道所在图层名称，默认为channel')
    parser.add_argument('--use-gpu', action='store_true', help='使用GPU加速计算')
    parser.add_argument('--visualize-mst', action='store_true', help='可视化最小生成树')
    parser.add_argument('--mst-layer', type=str, default='mst', help='最小生成树的图层名称，默认为mst')
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
            logging.FileHandler('logs/cad_channel_mst.log', mode='w'),  # 文件输出
            logging.StreamHandler()  # 控制台输出
        ]
    )

    logging.info("======= 开始执行基于最小生成树的AutoCAD通道构建 =======")
    # 记录开始时间
    total_start_time = time.time()
    
    # 显示参数信息
    logging.info(f"通道宽度设置为: {args.channel_width}mm")
    logging.info(f"通道图层名称: {args.layer}")
    logging.info(f"使用GPU加速: {'是' if args.use_gpu else '否'}")
    logging.info(f"可视化最小生成树: {'是' if args.visualize_mst else '否'}")
    if args.visualize_mst:
        logging.info(f"最小生成树图层: {args.mst_layer}")
    
    # 创建通道构建器实例
    channel_builder = CADChannelBuilderMST(use_gpu=args.use_gpu)
    
    # 构建CAD实例
    logging.info("构建CAD实例...")
    build_start_time = time.time()
    if not channel_builder.build_cad_instance():
        logging.error("构建CAD实例失败")
        return
    build_end_time = time.time()
    logging.info(f"CAD实例构建耗时: {build_end_time - build_start_time:.2f}秒")
    
    # 创建通道
    logging.info("开始为图层1中的图层0形状创建通道...")
    channel_start_time = time.time()
    
    # 执行通道构建操作
    try:
        success = channel_builder.create_channels_for_nested_entities(
            channel_width=args.channel_width,
            layer_name=args.layer,
            visualize_mst=args.visualize_mst,
            mst_layer=args.mst_layer
        )
        
        channel_end_time = time.time()
        channel_time = channel_end_time - channel_start_time
        
        if success:
            logging.info(f"通道创建成功，耗时: {channel_time:.2f}秒")
        else:
            logging.error(f"通道创建失败，耗时: {channel_time:.2f}秒")
    except Exception as e:
        logging.error(f"执行过程中发生错误: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
    
    # 记录结束时间
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    logging.info(f"总执行时间: {total_time:.2f}秒")
    logging.info("======= 执行完成 =======")
    logging.info(f"日志文件保存在: {os.path.abspath('logs/cad_channel_mst.log')}")

if __name__ == "__main__":
    main() 