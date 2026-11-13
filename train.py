import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import librosa

train_dirs = [
    "-6_dB",
    "0_dB",
    "6_dB"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 20
batch_size = 64
lr = 1e-3
checkpoint_path = "checkpoint.pth"
final_weight_path = "model.pkl"
n_mels = 128
sr = 16000


class MIMIIDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fp = self.file_list[idx]
        y, _ = librosa.load(fp, sr=sr)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel_spect)
        feat = np.mean(mel_db, axis=1)
        return torch.from_numpy(feat).float()


def get_flac_filelist(dirs_list):
    file_list = []
    for d in dirs_list:
        if not os.path.exists(d):
            print(f"跳过不存在目录：{d}")
            continue
        for root, _, files in os.walk(d):
            # 关键修复：只读取normal文件夹，跳过abnormal
            if "normal" not in root:
                continue
            for filename in files:
                if filename.lower().endswith(".flac"):
                    full_path = os.path.join(root, filename)
                    file_list.append(full_path)
    if len(file_list) == 0:
        raise RuntimeError("未读取到任何flac音频！检查是否存在normal文件夹")
    return file_list


class AutoEncoder(nn.Module):
    def __init__(self, input_dim=128):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


model = AutoEncoder(input_dim=128).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
start_epoch = 0

if os.path.exists(checkpoint_path):
    print("检测到checkpoint，加载断点继续训练...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt["epoch"] + 1
    print(f"继续训练，起始epoch:{start_epoch}")
else:
    print("无断点，从头训练")

train_filelist = get_flac_filelist(train_dirs)
print(f"一共找到 {len(train_filelist)} 条【仅normal】音频用于训练")
train_set = MIMIIDataset(train_filelist)
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

try:
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            recon_batch = model(batch_data)
            loss = criterion(recon_batch, batch_data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_data.size(0)
            total_samples += batch_data.size(0)
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1} | "
                      f"已处理 {total_samples}/{len(train_set)} | Batch Loss: {loss.item():.6f}")
        epoch_loss = running_loss / len(train_set)
        print(f"\n==== Epoch {epoch+1}/{epochs} 完成 | Epoch Loss: {epoch_loss:.6f} ====\n")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, checkpoint_path)
except KeyboardInterrupt:
    print("\n手动终止训练，断点已保存，下次直接运行即可续训。")
    exit()

torch.save(model.state_dict(), final_weight_path)
print(f"\n训练全部完成！上交权重已保存为 {final_weight_path}")