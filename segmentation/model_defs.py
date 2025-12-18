import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.conv(x)

class ResBlock(nn.Module):
    """Residual Block: x + Conv(Conv(x))"""
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Shortcut connection (The "Highway")
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out += residual   # <--- The Magic: Adding the input back to the output
        return self.relu(out)
    
#25D Unet with depth 5 (unet_1.pth)
class UNet25D_depth5(nn.Module):
    def __init__(self, n_channels=7, n_classes=1):
        super().__init__()
        
        # --- ENCODER ---
        self.inc = DoubleConv(n_channels, 16)       
        self.down1 = DoubleConv(16, 32)             
        self.down2 = DoubleConv(32, 64)             
        self.down3 = DoubleConv(64, 128)            
        self.down4 = DoubleConv(128, 256)           
        self.down5 = DoubleConv(256, 512) # Bottleneck

        self.dropout = nn.Dropout2d(p=0.5)
        self.pool = nn.MaxPool2d(2)
        
        # --- DECODER ---
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1 = DoubleConv(512 + 256, 256)
        self.up2 = DoubleConv(256 + 128, 128)
        self.up3 = DoubleConv(128 + 64, 64)
        self.up4 = DoubleConv(64 + 32, 32)
        self.up5 = DoubleConv(32 + 16, 16)
        
        self.outc = nn.Conv2d(16, n_classes, 1)

    def forward(self, x):
        # Down
        x1 = self.inc(x)                # 16
        x2 = self.down1(self.pool(x1))  # 32
        x3 = self.down2(self.pool(x2))  # 64
        x4 = self.down3(self.pool(x3))  # 128
        x5 = self.down4(self.pool(x4))  # 256
        x6 = self.down5(self.pool(x5))  # 512
        x6 = self.dropout(x6)

        # Up
        x = self.up(x6)
        x = torch.cat([x5, x], dim=1)
        x = self.up1(x)

        x = self.up(x)
        x = torch.cat([x4, x], dim=1)
        x = self.up2(x)

        x = self.up(x)
        x = torch.cat([x3, x], dim=1)
        x = self.up3(x)

        x = self.up(x)
        x = torch.cat([x2, x], dim=1)
        x = self.up4(x)
        
        x = self.up(x)
        x = torch.cat([x1, x], dim=1)
        x = self.up5(x)

        return self.outc(x)
    
#25D ResUnet with depth 5 (unet_2.pth) aggressive augmentation
class ResUNet25D_depth5(nn.Module):
    def __init__(self, n_channels=7, n_classes=1, base_filters=16):
        super().__init__()
        f = base_filters  # Starts at 16

        # --- ENCODER ---
        # Level 1: 16 channels
        self.inc = ResBlock(n_channels, f) 
        
        # Level 2: 32 channels
        self.down1 = ResBlock(f, f*2) 
        
        # Level 3: 64 channels
        self.down2 = ResBlock(f*2, f*4) 
        
        # Level 4: 128 channels
        self.down3 = ResBlock(f*4, f*8) 
        
        # Level 5: 256 channels
        self.down4 = ResBlock(f*8, f*16)

        # Level 6 (Bottleneck): 512 channels
        # High dropout here as this is the deepest layer with most parameters
        self.down5 = ResBlock(f*16, f*32, dropout_prob=0.5)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # --- DECODER ---
        # Maintains dropout_prob=0.3 to force robust feature learning
        
        # Up 1: Input (512 from bottleneck + 256 from down4) -> Output 256
        self.up1 = ResBlock(f*32 + f*16, f*16, dropout_prob=0.3)
        
        # Up 2: Input (256 + 128 from down3) -> Output 128
        self.up2 = ResBlock(f*16 + f*8, f*8, dropout_prob=0.3)
        
        # Up 3: Input (128 + 64 from down2) -> Output 64
        self.up3 = ResBlock(f*8 + f*4, f*4, dropout_prob=0.3)
        
        # Up 4: Input (64 + 32 from down1) -> Output 32
        self.up4 = ResBlock(f*4 + f*2, f*2, dropout_prob=0.3)
        
        # Up 5: Input (32 + 16 from inc) -> Output 16
        self.up5 = ResBlock(f*2 + f, f, dropout_prob=0.3)

        self.outc = nn.Conv2d(f, n_classes, 1)

    def forward(self, x):
        # --- Downsampling ---
        x1 = self.inc(x)             # 16
        x2 = self.down1(self.pool(x1)) # 32
        x3 = self.down2(self.pool(x2)) # 64
        x4 = self.down3(self.pool(x3)) # 128
        x5 = self.down4(self.pool(x4)) # 256
        
        # Bottleneck
        x6 = self.down5(self.pool(x5)) # 512

        # --- Upsampling ---
        
        # 512 -> 256
        x = self.up(x6)
        x = torch.cat([x5, x], dim=1)
        x = self.up1(x)

        # 256 -> 128
        x = self.up(x)
        x = torch.cat([x4, x], dim=1)
        x = self.up2(x)

        # 128 -> 64
        x = self.up(x)
        x = torch.cat([x3, x], dim=1)
        x = self.up3(x)

        # 64 -> 32
        x = self.up(x)
        x = torch.cat([x2, x], dim=1)
        x = self.up4(x)

        # 32 -> 16
        x = self.up(x)
        x = torch.cat([x1, x], dim=1)
        x = self.up5(x)

        return self.outc(x)

class AttentionBlock(nn.Module):
    """
    Attention Gate: Highlights relevant features from skip connections
    g: gating signal from coarser scale (decoder)
    x: corresponding encoder feature map (skip connection)
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        # Transform gating signal to intermediate dimension
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Transform skip connection to intermediate dimension
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Generate attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: (B, F_g, H, W) - gating signal from decoder
        x: (B, F_l, H, W) - encoder feature to be attention-weighted
        Returns: (B, F_l, H, W) - attention-weighted features
        """
        # Apply transformations
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Combine and activate
        psi = self.relu(g1 + x1)

        # Generate attention map
        psi = self.psi(psi)

        # Apply attention: element-wise multiplication
        return x * psi


class AttentionUNet25D(nn.Module):
    """
    Attention U-Net for 2.5D medical image segmentation

    Key improvements over standard U-Net:
    1. Attention gates filter skip connections
    2. Focuses on relevant features, suppresses irrelevant regions
    3. Particularly effective for small/sparse targets (tumors, lesions)
    """
    def __init__(self, n_channels=7, n_classes=1):
        super().__init__()

        # --- ENCODER ---
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = DoubleConv(32, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(p=0.5)

        # --- ATTENTION GATES ---
        # Applied before concatenating skip connections
        # F_g = decoder channels, F_l = encoder channels, F_int = intermediate
        self.att1 = AttentionBlock(F_g=512, F_l=256, F_int=256)
        self.att2 = AttentionBlock(F_g=256, F_l=128, F_int=128)
        self.att3 = AttentionBlock(F_g=128, F_l=64, F_int=64)
        self.att4 = AttentionBlock(F_g=64, F_l=32, F_int=32)

        # --- DECODER ---
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1 = DoubleConv(512 + 256, 256)
        self.up2 = DoubleConv(256 + 128, 128)
        self.up3 = DoubleConv(128 + 64, 64)
        self.up4 = DoubleConv(64 + 32, 32)

        # --- OUTPUT ---
        self.outc = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        # --- ENCODER PATH ---
        x1 = self.inc(x)           # 32 channels, full resolution
        x2 = self.down1(self.pool(x1))  # 64 channels, 1/2 resolution
        x3 = self.down2(self.pool(x2))  # 128 channels, 1/4 resolution
        x4 = self.down3(self.pool(x3))  # 256 channels, 1/8 resolution
        x5 = self.down4(self.pool(x4))  # 512 channels, 1/16 resolution (bottleneck)
        x5 = self.dropout(x5)

        # --- DECODER PATH WITH ATTENTION ---
        # Level 1: Upsample and apply attention to x4
        d1 = self.up(x5)                    # 512 channels, 1/8 resolution
        x4_att = self.att1(g=d1, x=x4)      # Attention-weighted x4
        d1 = torch.cat([x4_att, d1], dim=1) # Concatenate
        d1 = self.up1(d1)                   # 256 channels

        # Level 2: Upsample and apply attention to x3
        d2 = self.up(d1)                    # 256 channels, 1/4 resolution
        x3_att = self.att2(g=d2, x=x3)      # Attention-weighted x3
        d2 = torch.cat([x3_att, d2], dim=1)
        d2 = self.up2(d2)                   # 128 channels

        # Level 3: Upsample and apply attention to x2
        d3 = self.up(d2)                    # 128 channels, 1/2 resolution
        x2_att = self.att3(g=d3, x=x2)      # Attention-weighted x2
        d3 = torch.cat([x2_att, d3], dim=1)
        d3 = self.up3(d3)                   # 64 channels

        # Level 4: Upsample and apply attention to x1
        d4 = self.up(d3)                    # 64 channels, full resolution
        x1_att = self.att4(g=d4, x=x1)      # Attention-weighted x1
        d4 = torch.cat([x1_att, d4], dim=1)
        d4 = self.up4(d4)                   # 32 channels

        # --- FINAL OUTPUT ---
        return self.outc(d4)

class AttentionUNet25D_depth5(nn.Module):
    #ATTENTION UNET DEPTH 5
    """
    Attention U-Net depth 5 for 2.5D medical image segmentation

    Key improvements over standard U-Net:
    1. Attention gates filter skip connections
    2. Focuses on relevant features, suppresses irrelevant regions
    3. Particularly effective for small/sparse targets (tumors, lesions)
    """
    def __init__(self, n_channels=7, n_classes=1):
        super().__init__()

        # --- ENCODER ---
        # 1. Start with 16 channels
        self.inc = DoubleConv(n_channels, 16)
        
        # 2. Downsampling path to 512
        self.down1 = DoubleConv(16, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.down4 = DoubleConv(128, 256)
        self.down5 = DoubleConv(256, 512) # Deepest layer (Bottleneck)

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(p=0.5)

        # --- ATTENTION GATES ---
        # F_g = gating signal (from decoder/upsample)
        # F_l = skip connection (from encoder)
        # We need 5 gates now to match the 5 levels of skip connections
        
        # Gate for x5 (256): Upsampled 512 -> 512
        self.att1 = AttentionBlock(F_g=512, F_l=256, F_int=256)
        
        # Gate for x4 (128): Upsampled 256 -> 256
        self.att2 = AttentionBlock(F_g=256, F_l=128, F_int=128)
        
        # Gate for x3 (64): Upsampled 128 -> 128
        self.att3 = AttentionBlock(F_g=128, F_l=64, F_int=64)
        
        # Gate for x2 (32): Upsampled 64 -> 64
        self.att4 = AttentionBlock(F_g=64, F_l=32, F_int=32)
        
        # Gate for x1 (16): Upsampled 32 -> 32
        self.att5 = AttentionBlock(F_g=32, F_l=16, F_int=16)


        # --- DECODER ---
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Up 1: Input (512 + 256) -> Output 256
        self.up1 = DoubleConv(512 + 256, 256)
        
        # Up 2: Input (256 + 128) -> Output 128
        self.up2 = DoubleConv(256 + 128, 128)
        
        # Up 3: Input (128 + 64) -> Output 64
        self.up3 = DoubleConv(128 + 64, 64)
        
        # Up 4: Input (64 + 32) -> Output 32
        self.up4 = DoubleConv(64 + 32, 32)
        
        # Up 5 (New): Input (32 + 16) -> Output 16
        self.up5 = DoubleConv(32 + 16, 16)

        # --- OUTPUT ---
        self.outc = nn.Conv2d(16, n_classes, 1)

    def forward(self, x):
        # --- ENCODER PATH ---
        x1 = self.inc(x)                # 16 channels
        x2 = self.down1(self.pool(x1))  # 32 channels
        x3 = self.down2(self.pool(x2))  # 64 channels
        x4 = self.down3(self.pool(x3))  # 128 channels
        x5 = self.down4(self.pool(x4))  # 256 channels
        x6 = self.down5(self.pool(x5))  # 512 channels (Bottleneck)
        
        x6 = self.dropout(x6)

        # --- DECODER PATH WITH ATTENTION ---
        
        # Level 1: Upsample x6, Attend to x5
        d1 = self.up(x6)                    # 512 channels
        x5_att = self.att1(g=d1, x=x5)      # Attention on 256
        d1 = torch.cat([x5_att, d1], dim=1) # Cat -> 768
        d1 = self.up1(d1)                   # -> 256 channels

        # Level 2: Upsample d1, Attend to x4
        d2 = self.up(d1)                    # 256 channels
        x4_att = self.att2(g=d2, x=x4)      # Attention on 128
        d2 = torch.cat([x4_att, d2], dim=1) 
        d2 = self.up2(d2)                   # -> 128 channels

        # Level 3: Upsample d2, Attend to x3
        d3 = self.up(d2)                    # 128 channels
        x3_att = self.att3(g=d3, x=x3)      # Attention on 64
        d3 = torch.cat([x3_att, d3], dim=1)
        d3 = self.up3(d3)                   # -> 64 channels

        # Level 4: Upsample d3, Attend to x2
        d4 = self.up(d3)                    # 64 channels
        x2_att = self.att4(g=d4, x=x2)      # Attention on 32
        d4 = torch.cat([x2_att, d4], dim=1)
        d4 = self.up4(d4)                   # -> 32 channels

        # Level 5 (New): Upsample d4, Attend to x1
        d5 = self.up(d4)                    # 32 channels
        x1_att = self.att5(g=d5, x=x1)      # Attention on 16
        d5 = torch.cat([x1_att, d5], dim=1)
        d5 = self.up5(d5)                   # -> 16 channels

        # --- FINAL OUTPUT ---
        return self.outc(d5)

class MultiBranchUNet(nn.Module):
    def __init__(self, n_channels=7, n_classes=1):
        super().__init__()
        
        # Branch 1: Deep ResUNet
        self.branch_res = ResUNet25D_depth5(n_channels, n_classes, base_filters=16)
        
        # Branch 2: Deep Attention UNet
        self.branch_att = AttentionUNet25D_depth5(n_channels, n_classes)

    def forward(self, x):
        # 1. Pass input through ResUNet Branch
        # Output shape: (Batch, n_classes, H, W) -> These are Logits
        out_res = self.branch_res(x)
        
        # 2. Pass input through Attention UNet Branch
        # Output shape: (Batch, n_classes, H, W) -> These are Logits
        out_att = self.branch_att(x)
        
        # 3. Average the Logits
        # We average here so the loss function receives a single combined prediction
        final_logits = (out_res + out_att) / 2.0
        
        return final_logits
# --- MODEL REGISTRY ---
# distinct mapping of filenames to classes and their specific parameters

MODEL_REGISTRY = {
    "unet_1.pth": {
        #25D Unet with depth 5 (unet_1.pth)
        "class": UNet25D_depth5,
        "params": {"n_channels": 7, "n_classes": 1}
    },
    "unet_2.pth": {
        #25D ResUnet with depth 5 (unet_2.pth) aggressive augmentation/new training
        "class": ResUNet25D_depth5,
        "params": {"n_channels": 7, "n_classes": 1, "base_filters": 16}
    },
     "unet_3.pth": {
        #AttentionUnet depth 4 (unet_3.pth) aggressive augmentation/new training
        "class": AttentionUNet25D,
        "params": {"n_channels": 7, "n_classes": 1}
    },
       "unet_4.pth": {
        #AttentionUnet depth 5 (unet_4.pth) aggressive augmentation/new training
        "class": AttentionUNet25D_depth5,
        "params": {"n_channels": 7, "n_classes": 1}
    },
    "unet_5.pth": {
        #AttentionUnet depth 5 and ResUnet depth 5 trained together as MultiBranchUNet (unet_5.pth)
        "class": MultiBranchUNet,
        "params": {"n_channels": 7, "n_classes": 1}
    },
}

# Optional: A default fallback class if the filename isn't found
DEFAULT_MODEL_CONFIG = {
    "class": ResUNet25D_depth5,
    "params": {"n_channels": 7, "n_classes": 1}
}