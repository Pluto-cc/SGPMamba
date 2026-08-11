import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import torch.distributed as dist
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from dataset import SARRefinementDataset 
from model import SARRefinementModel     
from loss import CombinedRefinementLoss   
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
os.environ['OMP_NUM_THREADS'] = '1'     

NUM_EPOCHS = 100       
BATCH_SIZE = 1         
LEARNING_RATE = 6e-4   
TRAIN_PRIOR_DIR = '/home/changqi/王长启/Code/Umamba/dataset/scattering/train'
TRAIN_REAL_DIR = '/home/changqi/王长启/Code/Umamba/dataset/train/train'
VAL_PRIOR_DIR = '/home/changqi/王长启/Code/Umamba/dataset/scattering/val_prior'
VAL_REAL_DIR = '/home/changqi/王长启/Code/Umamba/dataset/train/val_train'
CHECKPOINT_DIR = './checkpoints/ablation_l2=0'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 训练配置
def parse_args():
    parser = argparse.ArgumentParser(description="SAR Refinement Model Training")
    
    # 路径配置
    parser.add_argument('--data_root', type=str, default='./data', help='数据集根目录')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='模型保存目录')
    
    # 模型配置
    parser.add_argument('--in_chans', type=int, default=1, help='输入通道数 (SAR 图像)')
    parser.add_argument('--num_classes', type=int, default=1, help='输出通道数 (精修图像)')
    parser.add_argument('--embed_dim', type=int, default=64, help='ResT 编码器基础维度')
    parser.add_argument('--decode_channels', type=int, default=64, help='解码器基础维度')
    parser.add_argument('--use_lsm', type=bool, default=False, help='是否使用局部监督模块 (LSM)')
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=100, help='总训练轮数')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='权重衰减')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='训练设备')
    parser.add_argument('--amp', action='store_true', help='是否使用自动混合精度 (AMP)')

    # 损失权重
    parser.add_argument('--lambda_l1', type=float, default=5.0, help='L1 损失权重')
    parser.add_argument('--lambda_ssim', type=float, default=5.0, help='ssim 损失权重')
    parser.add_argument('--lambda_hist', type=float, default=5.0, help='hist 损失权重')
    parser.add_argument('--lambda_perc', type=float, default=5.0, help='perc 损失权重')

    args = parser.parse_args()
    return args

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp, epoch):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch")

    for batch_idx, (priors, reals) in enumerate(pbar):
        priors, reals = priors.to(device), reals.to(device)
        optimizer.zero_grad()

        if amp:
            with torch.cuda.amp.autocast():
                outputs = model(priors)
                loss_dict = criterion(outputs, reals)
                loss = loss_dict['total_loss']
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(priors)
            loss_dict = criterion(outputs, reals)
            loss = loss_dict['total_loss']
            
            loss.backward()
            optimizer.step()

        if epoch <= 3 and batch_idx == 0:
            pbar.write(f"\n" + "="*50)
            pbar.write(f"  [Loss Weight Inspection - Epoch {epoch}]")

            l_total = loss.item() if hasattr(loss, 'item') else float(loss)
            pbar.write(f"  Total Loss: {l_total:.4f}")
            pbar.write("-" * 50)

            for name, val in loss_dict.items():
                if name != 'total_loss':
                    v = val.item() if hasattr(val, 'item') else float(val)
                    percentage = (v / l_total) * 100 if l_total != 0 else 0
                    pbar.write(f"  {name.upper():12} : {v:.4f} ({percentage:.1f}%)")
            
            pbar.write("="*50 + "\n")

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    
    pbar = tqdm(loader, desc="Validation", unit="batch")
    
    with torch.no_grad():
        for priors, reals in pbar:
            priors, reals = priors.to(device), reals.to(device)

            outputs = model(priors)

            loss_dict = criterion(outputs, reals)

            if isinstance(loss_dict, dict):
                loss = loss_dict['total_loss']

            else:
                loss = loss_dict
            
            running_loss += loss.item()

            pbar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(loader)
    return avg_loss

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    logging.info(f"Initializing training on device: {args.device}")

    train_dataset = SARRefinementDataset(
    prior_dir="/home/changqi/王长启/Code/Umamba/dataset/scattering/train", 
    real_dir="/home/changqi/王长启/Code/Umamba/dataset/train/train",
    img_size=(800, 800)
)
    val_dataset = SARRefinementDataset(
        prior_dir='/home/changqi/王长启/Code/Umamba/dataset/scattering/val_prior',   
        real_dir='/home/changqi/王长启/Code/Umamba/dataset/train/val_train',     
        img_size=(800, 800)  
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=1, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    logging.info(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    model = SARRefinementModel(
        in_chans=args.in_chans,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        decode_channels=args.decode_channels,
        use_lsm=True
    ).to(args.device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = CombinedRefinementLoss(
    device=device,
    lambda_l1=1.0,    
    lambda_ssim=0,
    lambda_hist=1.0,
    lambda_perc=1.0,
    w_stage1=0.5, 
    w_stage2=0.5
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = GradScaler() if args.amp and args.device == 'cuda' else None
    
    start_epoch = 1
    best_val_loss = float('inf')
    last_checkpoint_path = os.path.join(args.checkpoint_dir, 'last_checkpoint.pth')
    
    if os.path.exists(last_checkpoint_path):
        logging.info(f"检测到上次中断的检查点: {last_checkpoint_path}，正在恢复...")
        checkpoint = torch.load(last_checkpoint_path, map_location=args.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        logging.info(f"恢复成功！将从第 {start_epoch} Epoch 开始训练。")
    else:
        logging.info("未发现检查点，将从头开始训练。")

    for epoch in range(start_epoch, args.epochs + 1):
        logging.info(f"Starting Epoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, args.device, args.amp, epoch
        )

        val_loss = validate(model, val_loader, criterion, args.device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            logging.info(f"保存新的最佳模型，Val Loss: {val_loss:.4f}")

        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved at epoch {epoch}")

    logging.info("Training finished.")

if __name__ == '__main__':
    main()
