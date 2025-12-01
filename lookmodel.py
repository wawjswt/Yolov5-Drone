import torch
from models.yolo import Model
import yaml
import sys
import os
from datetime import datetime
from pathlib import Path

class Tee:
    """同时将输出发送到控制台和文件"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

def setup_logging(save_dir="runs/model_info"):
    """设置日志保存目录和文件"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"model_info_{timestamp}.txt")
    
    log_f = open(log_file, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_f)
    
    print(f"模型信息分析 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志文件: {log_file}")
    print("=" * 80)
    
    return log_f, original_stdout

def restore_logging(log_f, original_stdout):
    """恢复原始stdout并关闭日志文件"""
    print("=" * 80)
    print(f"分析完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    sys.stdout = original_stdout
    if log_f:
        log_f.close()
        print(f"模型信息已保存")


def calculate_thop_flops(model, device, img_size=640):
    """
    使用thop库精确计算FLOPs
    """
    try:
        from thop import profile
        # 创建随机输入数据
        dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
        
        # 计算FLOPs和参数
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        
        # 将MACs转换为FLOPs (通常FLOPs ≈ 2 * MACs)
        flops = 2 * macs
        
        return flops, params, macs
        
    except ImportError:
        print("警告: thop库未安装，无法计算精确FLOPs")
        print("请安装: pip install thop 或 pip install ultralytics-thop")
        return 0, 0, 0
    except Exception as e:
        print(f"thop计算失败: {e}")
        return 0, 0, 0

def get_model_info(weights_path, cfg_path=None, device='cuda', img_size=640):
    """
    获取模型的详细信息：类别、结构、参数数量、计算复杂度
    
    Args:
        weights_path: 模型权重文件路径
        cfg_path: 模型配置文件路径 (可选)
        device: 设备 ('cuda', 'cpu')
        img_size: 输入图像尺寸
    """
    # 自动选择设备
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，回退到CPU")
        device = 'cpu'
    
    device = torch.device(device)
    print(f"使用设备: {device}")
    
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU({i})的型号: {torch.cuda.get_device_name(i)}")  
    
    # 加载模型权重
    ckpt = torch.load(weights_path, map_location='cpu')
    
    # 获取模型配置
    if hasattr(ckpt['model'], 'yaml'):
        model_yaml = ckpt['model'].yaml
    elif cfg_path:
        with open(cfg_path, 'r') as f:
            model_yaml = yaml.safe_load(f)
    else:
        raise ValueError("无法获取模型配置，请提供cfg_path参数")
    
    # 获取类别数
    nc = model_yaml.get('nc', 80)
    print(f"\n模型类别数: {nc}")
    
    # 创建模型
    model = Model(model_yaml, ch=3, nc=nc, anchors=None).to(device)
    
    # 加载权重
    csd = ckpt['model'].float().state_dict()
    model.load_state_dict(csd, strict=True)
    print(f"成功加载模型权重: {weights_path}")
    
    # 打印模型结构
    print("\n" + "="*80)
    print("模型结构:")
    print("="*80)
    print(model)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n" + "="*80)
    print("模型统计:")
    print("="*80)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"不可训练参数: {total_params - trainable_params:,}")
    

    print(f"\n" + "="*80)
    print("计算复杂度分析:")
    print("="*80)
    

    
    # 使用thop计算精确FLOPs
    thop_flops, thop_params, macs = calculate_thop_flops(model, device, img_size)
    
    if thop_flops > 0:
        print(f"精确计算 FLOPs: {thop_flops / 1e9:.2f} GFLOPs")
        print(f"精确计算 MACs: {macs / 1e9:.2f} GMACs")
        print(f"thop计算参数: {thop_params:,}")
        

    # 显示模型大小
    model_size = os.path.getsize(weights_path) / (1024 * 1024)  # MB
    print(f"模型文件大小: {model_size:.3f} MB")
    
    # 显示GPU内存使用情况（如果使用CUDA）
    if device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        print(f"GPU内存使用: {torch.cuda.memory_allocated(device)/1024**2:.3f} MB, {memory_allocated:.3f} GB")
    
    # 返回详细结果
    result = {
        'num_classes': nc,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'thop_flops': thop_flops,
        'macs': macs,
        'model_size_mb': model_size,
        'img_size': img_size
    }
    
    return model, result

# 使用示例
if __name__ == "__main__":
    # 设置日志
    log_f, original_stdout = setup_logging("runs/model_info")
    
    try:
        # 指定模型权重文件和配置文件
        weights_path = "weights/Firev5s.pt"  # 替换模型文件
        cfg_path = "models/yolov5s-fire.yaml"  # 替换配置文件
        
        # 获取模型信息
        model, result = get_model_info(weights_path, cfg_path, device='cuda', img_size=640)
        
        # 打印汇总结果
        print(f"\n" + "-+"*40+'-')
        print("结果汇总:")
        print("-+"*40+'-')
        print(f"模型名称：{weights_path}")
        print(f"类别数: {result['num_classes']}")
        print(f"总参数: {result['total_params']:,}")

        if result['thop_flops'] > 0:
            print(f"精确计算 FLOPs: {result['thop_flops'] / 1e9:.2f} GFLOPs")
            print(f"精确计算 MACs: {result['macs'] / 1e9:.2f} GMACs")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 恢复原始输出并关闭日志文件
        restore_logging(log_f, original_stdout)