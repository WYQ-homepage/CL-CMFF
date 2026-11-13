import os
import sys
import argparse
import time
import glob
import traceback
import pandas as pd
import torch
import torch.nn as nn
from torch.backends import cudnn

# 尝试导入 thop 用于计算 FLOPs
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("⚠️ 未检测到 thop 库，将跳过 FLOPs 计算 (pip install thop)")

# ==================== 路径设置 ====================
# 确保可以导入项目中的模块 (utils, model 等)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from model.core import Model
    from utils import TrainConfig
except ImportError as e:
    print(f"❌ 导入项目模块失败: {e}")
    print("请确保将此脚本放在 'Multimode-main/multimode/' 目录下")
    sys.exit(1)

# ==================== 参数解析 ====================
def parse_args():
    parser = argparse.ArgumentParser(description="Multimode Benchmark Tool")
    parser.add_argument("--gpu", type=str, default="1", help="指定使用的显卡序号")
    parser.add_argument("--save_path", type=str, default="benchmark_results.csv", help="结果保存路径")
    parser.add_argument("--img_size", type=int, default=512, help="输入图像尺寸")
    # [修改点 1] 添加 batch_size 参数，默认设置为 8
    parser.add_argument("--batch_size", type=int, default=8, help="Batch Size (默认: 8)")
    return parser.parse_args()

# ==================== 核心测试函数 ====================

def prepare_inputs(config, device, batch_size, img_size=512):
    """
    根据配置生成对应的输入张量
    [修改点 2] 接收 batch_size 参数
    """
    # Multimode 模型 forward(rgb, dsm)
    # RGB 始终为 3 通道
    # DSM (Light branch) 通道数由 config.model.x_num 决定

    x_num = config.model.x_num

    rgb = torch.randn(batch_size, 3, img_size, img_size).to(device)
    dsm = torch.randn(batch_size, x_num, img_size, img_size).to(device)

    return (rgb, dsm), f"RGB(3)+DSM({x_num})@{img_size}x{img_size}"

def measure_fps_strict(model, inputs):
    """
    FPS 测试：预热 30s + 运行 30s
    FPS = Batch_Size * (Batches / Seconds)
    """
    # 获取 Batch Size
    try:
        # inputs 是 (rgb, dsm) 元组
        batch_size = inputs[0].size(0)
    except:
        batch_size = 1

    # 1. 预热
    print(f"   🔥 正在预热 (BS={batch_size})...", end="", flush=True)
    t_end = time.time() + 10  # 预热 10秒
    while time.time() < t_end:
        with torch.no_grad():
            model(*inputs)
        torch.cuda.synchronize()
    print(" 完成")

    # 2. 测试
    print("   🚀 正在测试 FPS...", end="", flush=True)
    count = 0
    t_start = time.time()
    t_duration = 30 # 测试 30秒
    t_end_test = t_start + t_duration

    while time.time() < t_end_test:
        with torch.no_grad():
            model(*inputs)
        torch.cuda.synchronize()
        count += 1

    real_duration = time.time() - t_start

    # [修改点 3] 计算 FPS = Batch * (总处理批次 / 总时间)
    batches_per_sec = count / real_duration
    fps = batch_size * batches_per_sec

    print(f" 完成 -> FPS: {fps:.2f} (Batches/s: {batches_per_sec:.2f})")
    return fps

def measure_params_and_flops(model, inputs):
    """计算参数量和 FLOPs"""
    # 1. 参数量
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    # 2. FLOPs
    flops = 0.0
    if HAS_THOP:
        try:
            # thop.profile 会修改模型钩子，建议在独立的 eval 模式下运行或运行后清理
            flops_raw, _ = profile(model, inputs=inputs, verbose=False)
            flops = flops_raw / 1e9 # GFLOPs
        except Exception as e:
            print(f"   (FLOPs 计算失败: {e})")

    return total_params, trainable_params, flops

# ==================== 主程序 ====================
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not torch.cuda.is_available():
        print("❌ 未检测到 GPU，FPS 测试可能不准确")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        cudnn.benchmark = True
        print(f"=== 环境配置: {torch.cuda.get_device_name(0)} ===")
        print(f"=== Batch Size: {args.batch_size} ===")

    # 查找所有 YAML 配置
    config_files = sorted(glob.glob("configs/**/*.yaml", recursive=True))
    if not config_files:
        print("❌ 未在 configs/ 目录下找到 .yaml 文件")
        return

    results = []

    for idx, cfg_path in enumerate(config_files):
        config_name = os.path.basename(cfg_path)
        print(f"\n[{idx+1}/{len(config_files)}] 处理配置: {config_name}")

        model = None
        try:
            # 1. 加载配置
            # 使用 utils.py 中的 TrainConfig 读取 yaml
            config = TrainConfig.from_yaml(cfg_path)

            # 2. 构建模型
            # 对应 model/core.py 中的 Model.__init__ 参数
            model = Model.from_config(config.model, model_use="all", load_pretrained=False)
            model.to(device)
            model.eval()

            # 3. 准备输入 (用于 FPS 测试，使用设定的 Batch Size)
            inputs, input_desc = prepare_inputs(config, device, args.batch_size, args.img_size)
            print(f"   ✅ 输入构建成功: {input_desc} (BS={args.batch_size})")

            # 4. 测试 FPS (先测 FPS 保证模型状态干净，不受 hooks 影响)
            fps = measure_fps_strict(model, inputs)

            # 5. 测试 Params & FLOPs
            # [修改点 4] 为了得到标准的"单张图片 GFLOPs"，我们专门构建一个 BS=1 的输入
            # 如果直接用 BS=8 测，thop 会算出 8 张图的总计算量，导致数值偏大
            inputs_bs1, _ = prepare_inputs(config, device, 1, args.img_size)
            params, trainable, flops = measure_params_and_flops(model, inputs_bs1)

            print(f"   📊 Params: {params:.2f}M | FLOPs: {flops:.3f}G")

            results.append({
                "Config": config_name,
                "Input": input_desc,
                "Batch Size": args.batch_size,
                "Params (M)": round(params, 2),
                "Trainable (M)": round(trainable, 2),
                "FLOPs (G)": round(flops, 3),
                "FPS": round(fps, 2)
            })

        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            traceback.print_exc()
        finally:
            if model is not None:
                del model
            torch.cuda.empty_cache()

    # 保存结果
    if results:
        df = pd.DataFrame(results)
        # 调整列顺序
        cols = ["Config", "Input", "Batch Size", "Params (M)", "Trainable (M)", "FLOPs (G)", "FPS"]
        df = df[cols]

        df.to_csv(args.save_path, index=False, encoding='utf-8-sig')

        print("\n" + "="*80)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df.to_string(index=False))
        print(f"\n✅ 测试完成！结果已保存至: {args.save_path}")
    else:
        print("\n⚠️ 未生成任何测试结果。")

if __name__ == "__main__":
    main()
