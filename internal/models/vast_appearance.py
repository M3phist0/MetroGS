from typing import Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/autonomousvision/gaussian-opacity-fields
def decouple_appearance(image, gaussians, view_idx):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    H, W = image.size(1), image.size(2)
    # down sample the image
    crop_image_down = torch.nn.functional.interpolate(image[None], size=(H // 32, W // 32), mode="bilinear", align_corners=True)[0]

    crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H // 32, W // 32, 1).permute(2, 0, 1)], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down, H, W).squeeze()
    transformed_image = mapping_image * image

    return transformed_image, mapping_image


class UpsampleBlock(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super(UpsampleBlock, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(num_input_channels // (2 * 2), num_output_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.conv(x)
        x = self.relu(x)
        return x
    
class AppearanceNetwork(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super(AppearanceNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(num_input_channels, 256, 3, stride=1, padding=1)
        self.up1 = UpsampleBlock(256, 128)
        self.up2 = UpsampleBlock(128, 64)
        self.up3 = UpsampleBlock(64, 32)
        self.up4 = UpsampleBlock(32, 16)
        
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, num_output_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, H, W):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        # bilinear interpolation
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x
    
class AppearanceNetworkV2(nn.Module):
    def __init__(self, num_input_channels, num_output_channels=12):
        super(AppearanceNetworkV2, self).__init__()
        
        self.conv1 = nn.Conv2d(num_input_channels, 256, 3, stride=1, padding=1)
        self.up1 = UpsampleBlock(256, 128)
        self.up2 = UpsampleBlock(128, 64)
        self.up3 = UpsampleBlock(64, 32)
        self.up4 = UpsampleBlock(32, 16)
        
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, num_output_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, H, W):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        # bilinear interpolation
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x

class VastModel(nn.Module):
    def __init__(
            self,
            n_appearance_count: int=6000,
            n_appearance_dims: int = 64,
            n_rgb_dims: int = 3,
            std: float = 1e-4,
    ) -> None:
        super().__init__()

        self._appearance_embeddings = nn.Parameter(torch.empty(n_appearance_count, n_appearance_dims).cuda())
        self._appearance_embeddings.data.normal_(0, std)
        self.appearance_network = AppearanceNetwork(n_rgb_dims+n_appearance_dims, n_rgb_dims).cuda()

    def forward(self, image, view_idx):
        appearance_embedding = self.get_appearance(view_idx)
        H, W = image.size(1), image.size(2)
        # down sample the image
        crop_image_down = torch.nn.functional.interpolate(image[None], size=(H // 32, W // 32), mode="bilinear", align_corners=True)[0]

        crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H // 32, W // 32, 1).permute(2, 0, 1)], dim=0)[None]
        mapping_image = self.appearance_network(crop_image_down, H, W).squeeze()
        transformed_image = mapping_image * image

        import os
        import random
        import torchvision
        if self.training and random.random() < 0.01: # 通常只在训练时进行检查
            torchvision.utils.save_image(image, os.path.join("./tmp/",  f"{view_idx.item()}_ref.png"))
            torchvision.utils.save_image(transformed_image, os.path.join("./tmp/",  f"{view_idx.item()}_trans.png"))

        return transformed_image, mapping_image
    
    def get_appearance(self, view_idx: Union[float, torch.Tensor]):
        return self._appearance_embeddings[view_idx]
    
    @staticmethod
    def _create_optimizer_and_scheduler(
            params,
            name,
            lr_init,
            lr_final_factor,
            max_steps,
            eps,
            warm_up,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = torch.optim.Adam(
            params=[
                {"params": list(params), "name": name}
            ],
            lr=lr_init,
            eps=eps,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda iter: lr_final_factor ** min(max(iter - warm_up, 0) / max_steps, 1),
            verbose=False,
        )

        return optimizer, scheduler
    
class VastModelV2(nn.Module):
    def __init__(
            self,
            n_appearance_count: int=6000,
            n_appearance_dims: int = 64,
            n_rgb_dims: int = 3,
            std: float = 1e-4,
    ) -> None:
        super().__init__()

        self._appearance_embeddings = nn.Parameter(torch.empty(n_appearance_count, n_appearance_dims).cuda())
        self._appearance_embeddings.data.normal_(0, std)
        self.appearance_network = AppearanceNetworkV2(n_rgb_dims+n_appearance_dims).cuda()

    def forward(self, image, view_idx):
        appearance_embedding = self.get_appearance(view_idx)
        H, W = image.size(1), image.size(2)
        # down sample the image
        crop_image_down = torch.nn.functional.interpolate(image[None], size=(H // 32, W // 32), mode="bilinear", align_corners=True)[0]

        crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H // 32, W // 32, 1).permute(2, 0, 1)], dim=0)[None]
        mapping_image = self.appearance_network(crop_image_down, H, W).squeeze()
        image_homogeneous = torch.cat([image, torch.ones(1, H, W).cuda()], dim=0).view(-1, H * W).permute(1, 0).unsqueeze(1)
        mapping_image = mapping_image.view(-1, H * W).permute(1, 0).view(H * W, 3, 4).transpose(2, 1)
        transformed_image = torch.bmm(image_homogeneous, mapping_image).squeeze(1).permute(1, 0).reshape(3, H, W)
        transformed_image = torch.clamp(transformed_image, 0, 1)

        return transformed_image, mapping_image
    
    def get_appearance(self, view_idx: Union[float, torch.Tensor]):
        return self._appearance_embeddings[view_idx]
    
    @staticmethod
    def _create_optimizer_and_scheduler(
            params,
            name,
            lr_init,
            lr_final_factor,
            max_steps,
            eps,
            warm_up,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = torch.optim.Adam(
            params=[
                {"params": list(params), "name": name}
            ],
            lr=lr_init,
            eps=eps,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda iter: lr_final_factor ** min(max(iter - warm_up, 0) / max_steps, 1),
            verbose=False,
        )

        return optimizer, scheduler

if __name__ == "__main__":
    H, W = 1200//32, 1600//32
    input_channels = 3 + 64
    output_channels = 3
    input = torch.randn(1, input_channels, H, W).cuda()
    model = AppearanceNetwork(input_channels, output_channels).cuda()
    
    output = model(input)
    print(output.shape)