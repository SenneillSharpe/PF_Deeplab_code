"""
1. 加载模型
2. 加载测试数据
3. 模型输出测试结果
4. 将测试结果按照加载数据返回的名称保存到相应路径下
"""

import torch
import numpy as np
import os
import tifffile as tiff
from torch.utils.data import DataLoader

# 模型导入（根据你的选择启用相应模型）
from Model.PF_Deeplab_resnest50 import PF_Deeplab_resnest50


# 数据加载导入
from step1_data_load import ThreeNonTrainDataset


def model_result_output(device, model, dataset, path):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    with torch.no_grad():
        for image, image_name in dataloader:
            image = image.to(device)

            # image_name 是 tuple，需要处理为字符串
            image_name = image_name[0]
            save_path = os.path.join(path, image_name)

            output = model(image)  # 输出 shape: (1, 1, H, W)
            prediction = (output > 0.5).float() * 255  # 二值图 0 或 255
            prediction = prediction.squeeze().cpu().numpy().astype(np.uint8)  # shape: (H, W)

            tiff.imwrite(save_path, prediction)
            print(f"{image_name} 保存成功")


if __name__ == "__main__":
    device = torch.device("cpu")  # or "cuda" if using GPU

    # 设置测试数据
    test_dataset = ThreeNonTrainDataset(
        image_dir=r"../test/image/",
        segmentation_label_dir=r"../test/label/",
        state="test",
    )

    # 模型加载：构建模型 + 加载权重
    model_path = r"../Model_save/model.pth"
    model = PF_Deeplab_resnest50(num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 输出结果保存路径
    save_path = r"../result/"
    os.makedirs(save_path, exist_ok=True)

    model_result_output(device=device, model=model, dataset=test_dataset, path=save_path)
