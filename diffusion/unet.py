import torch
import torch.nn as nn
from diffusers.models.unets.unet_2d import UNet2DModel, UNet2DOutput

from utils import LinearEmbedder, SinusoidalEmbedder, RFFEmbedder


class ConditionalUNet(nn.Module):
    def __init__(self, embed_dim=128, img_channel=3):
        super().__init__()
        self.embed_dim = embed_dim
        # self.condition_embedding = LinearEmbedder(embed_dim//2)
        # self.condition_embedding = SinusoidalEmbedder(embed_dim//2)
        self.condition_embedding = RFFEmbedder(embed_dim//2)

        self.model = UNet2DModel(
            sample_size=None,
            in_channels=img_channel,
            out_channels=img_channel,
            time_embedding_dim=embed_dim,
            layers_per_block=2,
            block_out_channels=(32, 64, 128, 256, 512),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D"
            ),
        )

    def forward(self, sample, timesteps, power, scan_speed, class_labels=None, return_dict=True):
        """
        sample: images (B, C, H, W)
        timesteps: tensor (B,)
        power, scan_speed: condition (B,), float
        """

        # 1. time
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.model.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.model.dtype)
        emb = self.model.time_embedding(t_emb)

        if self.model.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")

            if self.model.config.class_embed_type == "timestep":
                class_labels = self.model.time_proj(class_labels)

            class_emb = self.model.class_embedding(class_labels).to(dtype=self.model.dtype)
            emb = emb + class_emb
        elif self.model.class_embedding is None and class_labels is not None:
            raise ValueError("class_embedding needs to be initialized in order to use class conditioning")
        
        # ---------- condition ----------
        emb1 = self.condition_embedding(power)     # (B, D/2)
        emb2 = self.condition_embedding(scan_speed)
        cond_emb = torch.cat([emb1, emb2], dim=1)  # (B, D)
        emb = emb + cond_emb

        # 2. pre-process
        skip_sample = sample
        sample = self.model.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.model.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.model.mid_block is not None:
            sample = self.model.mid_block(sample, emb)

        # 5. up
        skip_sample = None
        for upsample_block in self.model.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.model.conv_norm_out(sample)
        sample = self.model.conv_act(sample)
        sample = self.model.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.model.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if not return_dict:
            return (sample,)

        return UNet2DOutput(sample=sample).sample


if __name__ == "__main__":
    model = ConditionalUNet()
    dummy_img = torch.randn(4, 3, 128, 128)
    t = torch.tensor([10, 20, 30, 40])
    power = torch.tensor([0.2, 0.4, 0.6, 0.8])
    speed = torch.tensor([10.0, 15.0, 20.0, 25.0])

    output = model(dummy_img, t, power, speed)
    print("Output shape:", output.shape)

    # S_model.model = UNet2DModel(
    #             sample_size=None,
    #             in_channels=3,
    #             out_channels=3,
    #             time_embedding_dim=128,
    #             layers_per_block=2,
    #             block_out_channels=(32, 128, 512),
    #             down_block_types=(
    #                 "DownBlock2D",
    #                 "DownBlock2D",
    #                 "DownBlock2D"
    #             ),
    #             up_block_types=(
    #                 "UpBlock2D",
    #                 "UpBlock2D",
    #                 "UpBlock2D"
    #             ),
    #         )