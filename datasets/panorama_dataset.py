import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class PanoramicDepthDataset(Dataset):
    def __init__(self, root_dir, dataset_type='stanford2d3d'):
        """
        初始化全景深度数据集
        :param root_dir: 数据集根目录，例如 './datasets/stanford2d3d'
        :param dataset_type: 'stanford2d3d' 或 'matterport3d'，用于区分不同的深度缩放比例
        """
        self.root_dir = Path(root_dir)
        self.rgb_dir = self.root_dir / 'rgb'
        self.depth_dir = self.root_dir / 'depth'
        self.dataset_type = dataset_type
        
        # 获取所有 rgb 图片的路径，并按名称排序保证一一对应
        self.rgb_paths = sorted(list(self.rgb_dir.glob('*_rgb.png')))
        self.depth_paths = sorted(list(self.depth_dir.glob('*_depth.png')))
        
        assert len(self.rgb_paths) == len(self.depth_paths), "RGB 图像和深度图的数量不一致！"
        print(f"成功加载 {len(self.rgb_paths)} 对全景数据。")

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_path = str(self.rgb_paths[idx])
        depth_path = str(self.depth_paths[idx])

        # ---------------------------
        # 1. 读取并处理 RGB 全景图
        # ---------------------------
        # cv2 默认读取为 BGR，转换为 RGB
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # 转换为 PyTorch Tensor (C, H, W)，并归一化到 [0, 1]
        rgb_tensor = torch.from_numpy(rgb_img).float().permute(2, 0, 1) / 255.0

        # ---------------------------
        # 2. 读取并处理 16-bit 深度图真值
        # ---------------------------
        # 警告：必须使用 cv2.IMREAD_ANYDEPTH 读取 16-bit 深度信息！
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        
        # 转换为 float 类型以进行除法计算
        depth_img = depth_img.astype(np.float32)

        # 物理单位转换 (转换为米)
        # 不同数据集的 PNG 存储格式不同，缩放因子不同
        if self.dataset_type == 'stanford2d3d':
            # Stanford2D3D 官方设定：像素值 512 代表 1 米
            depth_img = depth_img / 512.0 
        elif self.dataset_type == 'matterport3d':
            # Matterport3D 官方设定：像素值 4000 代表 1 米
            depth_img = depth_img / 4000.0 
            
        # 屏蔽无效的深度值（通常深度为 0 的像素代表相机的盲区或未扫描到的区域）
        valid_mask = (depth_img > 0.0) & (depth_img < 10.0) # 假设最大深度不超过 10 米
        
        # 转换为 Tensor，增加通道维度 (1, H, W)
        depth_tensor = torch.from_numpy(depth_img).unsqueeze(0)
        valid_mask_tensor = torch.from_numpy(valid_mask).unsqueeze(0)

        # 返回一个字典，方便后续按键取值
        return {
            'img_name': Path(rgb_path).stem,
            'erp_img': rgb_tensor,
            'erp_depth_gt': depth_tensor,
            'valid_mask': valid_mask_tensor
        }

# ================= 测试代码 =================
if __name__ == '__main__':
    # 假设你已经把图片放到了对应的文件夹下
    # 注意：在没有真实数据时，运行这里会报错。请下载数据集后替换为真实路径。
    dataset = PanoramicDepthDataset(root_dir='./datasets/stanford2d3d', dataset_type='stanford2d3d')
    
    # 因为是免训练评估，batch_size 通常设为 1
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    for i, batch in enumerate(dataloader):
        print(f"正在处理图片: {batch['img_name'][0]}")
        print(f"RGB 输入尺寸: {batch['erp_img'].shape}")       # 应为 [1, 3, H, W]
        print(f"深度真值尺寸: {batch['erp_depth_gt'].shape}")  # 应为 [1, 1, H, W]
        
        # 将这里的 erp_img 送入你的 VGGT-360 推理流水线
        # pred_depth = vggt_360_pipeline(batch['erp_img'])
        
        # 然后使用 pred_depth 和 batch['erp_depth_gt'] (配合 valid_mask) 计算 RMSE, REL 等误差指标
        break