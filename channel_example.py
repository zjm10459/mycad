import logging
import os
import argparse
import time

from channel_line.create_cad_channel import ChannelBuilder

def parse_args():
    parser = argparse.ArgumentParser(description='CAD通道构建器示例程序')
    parser.add_argument('--channel-width', type=float, default=10.0, help='通道宽度，默认10.0mm')
    parser.add_argument('--auto', action='store_true', help='自动为所有合适的形状创建通道')
    parser.add_argument('--entity1', type=str, help='第一个实体的句柄，例如17D62')
    parser.add_argument('--entity2', type=str, help='第二个实体的句柄，例如17D63')
    parser.add_argument('--layer', type=str, default='channel', help='通道所在图层名称，默认为channel')
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
            logging.FileHandler('logs/channel_builder.log', mode='w'),  # 文件输出
            logging.StreamHandler()  # 控制台输出
        ]
    )

    logging.info("======= 开始执行通道构建 =======")
    # 记录开始时间
    total_start_time = time.time()
    
    logging.info(f"通道宽度设置为: {args.channel_width}mm")
    
    # 创建通道构建器实例
    channel_builder = ChannelBuilder()
    
    # 构建CAD实例
    logging.info("构建CAD实例...")
    build_start_time = time.time()
    if not channel_builder.build_cad_instance():
        logging.error("构建CAD实例失败")
        return
    build_end_time = time.time()
    logging.info(f"CAD实例构建耗时: {build_end_time - build_start_time:.2f}秒")
    
    # 创建通道
    channel_start_time = time.time()
    success = False
    
    if args.auto:
        # 自动模式：在所有合适的形状之间创建通道
        logging.info("自动模式：为所有合适的形状创建通道...")
        success = channel_builder.create_channels_between_all_shapes(
            channel_width=args.channel_width,
            layer_name=args.layer
        )
    elif args.entity1 and args.entity2:
        # 指定实体模式：在两个指定实体之间创建通道
        logging.info(f"为指定实体创建通道: {args.entity1} 和 {args.entity2}")
        success = channel_builder.create_channel_between_shapes(
            entity1_handle=args.entity1,
            entity2_handle=args.entity2,
            channel_width=args.channel_width,
            layer_name=args.layer
        )
    else:
        logging.error("未指定操作模式。请使用--auto或同时指定--entity1和--entity2")
        return
    
    channel_end_time = time.time()
    channel_time = channel_end_time - channel_start_time
    
    if success:
        logging.info(f"通道创建成功，耗时: {channel_time:.2f}秒")
    else:
        logging.error(f"通道创建失败，耗时: {channel_time:.2f}秒")
    
    # 记录结束时间
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    logging.info(f"总执行时间: {total_time:.2f}秒")
    logging.info("======= 执行完成 =======")
    logging.info(f"日志文件保存在: {os.path.abspath('logs/channel_builder.log')}")

if __name__ == "__main__":
    main() 