import torch
import os
import numpy as np
import random
import timeit
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from diffusers import DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from utils import CustomImageLabelDataset, mean_std, split_dataset, label_normalizer
from unet import ConditionalUNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f'The device is {device}.')

RANDOM_SEED = 0
IMG_SIZE = 128
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
NUM_TIMESTEPS = 1000
NUM_WARMUP_STEPS = 500
PATIENCE = 20

max_power = 200.
min_power = 100.
max_scan_speed = 1.0
min_scan_speed = 0.5

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print('Preparing dataloder...')

# local path
image_folder = '/Users/xiezy/Documents/ml/diffusion/dataset/train/images'
label_folder = '/Users/xiezy/Documents/ml/diffusion/dataset/train/labels'


# Path for saving loss img and model parameters
save_dir = 'model'
os.makedirs(save_dir, exist_ok=True)
model_name = 'c_exp1'
cht_path = os.path.join(save_dir, f'{model_name}.png')
mdl_path = os.path.join(save_dir, f'{model_name}.pth')
los_path = os.path.join(save_dir, 'losses.npy')
vallos_path = os.path.join(save_dir, 'val_losses.npy')

transform0 = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST), 
    transforms.ToTensor()  # rescale: from 0-255 to 0-1
])
dataset = CustomImageLabelDataset(image_folder=image_folder, label_folder=label_folder, transform=transform0)
mean, std = mean_std(dataset)
print(f'mean and std of the dataset are: {mean, std}')

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST), 
    transforms.ToTensor(), 
    transforms.Normalize(mean, std)  # normalize by the mean and std
])

dataset = CustomImageLabelDataset(image_folder=image_folder, label_folder=label_folder, transform=transform)

dataset_size = len(dataset)
train_indices, val_indices = split_dataset(dataset_size, val_ratio=0.2, shuffle=True)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
    
model = ConditionalUNet()
model.to(device)

noise_scheduler = DDIMScheduler(num_train_timesteps=NUM_TIMESTEPS)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE) # weight_decay=None
# optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=NUM_WARMUP_STEPS,
    num_training_steps=len(train_dataloader) * NUM_EPOCHS
)

early_stopping_patience = PATIENCE
best_val_loss = float('inf') 
patience_counter = 0

losses = []
val_losses = []

start = timeit.default_timer()
print(f'Everything is ready.\nTraining start time: {start:.2f}s')

for epoch in tqdm(range(NUM_EPOCHS), position=0, leave=True):
    # traning
    model.train()
    train_running_loss = 0
    for x, y in train_dataloader:
        clean_images = x.to(device)
        noise = torch.randn(clean_images.shape).to(device)
        last_batch_size = len(clean_images)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (last_batch_size,)).to(device)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        power, scan_speed = y[:, 0], y[:, 1]
        power = label_normalizer(power, max_power, min_power).to(device)
        scan_speed = label_normalizer(scan_speed, max_scan_speed, min_scan_speed).to(device)

        noise_pred = model(noisy_images, timesteps, power, scan_speed)
        loss = F.mse_loss(noise_pred, noise)
        train_running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
    
    avg_train_loss = train_running_loss / len(train_dataloader)
    losses.append(avg_train_loss)
    train_learning_rate = lr_scheduler.get_last_lr()[0]

    # validation
    model.eval()
    val_running_loss = 0
    with torch.no_grad():
        for x, y in val_dataloader:
            clean_images = x.to(device)
            noise = torch.randn(clean_images.shape).to(device)
            last_batch_size = len(clean_images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (last_batch_size,)).to(device)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            power, scan_speed = y[:, 0], y[:, 1]
            power = label_normalizer(power, max_power, min_power).to(device)
            scan_speed = label_normalizer(scan_speed, max_scan_speed, min_scan_speed).to(device)

            noise_pred = model(noisy_images, timesteps, power, scan_speed)
            val_loss = F.mse_loss(noise_pred, noise)
            val_running_loss += val_loss.item()
    
    avg_val_loss = val_running_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)

    print('-' * 50)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f'Train Learning Rate: {epoch + 1}: {train_learning_rate}')
    print(f"Training Loss: {avg_train_loss:.6f}")
    print(f"Validation Loss: {avg_val_loss:.6f}")
    print('-' * 50)

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), mdl_path)
        print("Best model saved.")
    else:
        patience_counter += 1
        print(f"Validation loss did not improve for {patience_counter} epoch(s).")
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

end = timeit.default_timer()
print(f'Training finish time: {end:.2f}s')
print(f'Total training time: {end - start:.2f}s')

np.save(los_path, losses)
np.save(vallos_path, val_losses)

plt.figure()
plt.plot(losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(cht_path, dpi=300, bbox_inches='tight')
plt.close()

print("Loss chart saved.")
