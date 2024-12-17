import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from spookynet.spookynet import SpookyNet
from spookynet.qm_database import QMDatabase
import numpy as np
import logging

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(path, f'checkpoint_{epoch}.pt'))

def collate_fn(batch):
    Q = torch.tensor([item[0] for item in batch])
    S = torch.tensor([item[1] for item in batch])
    Z = torch.tensor(np.stack([item[2] for item in batch])).squeeze(0)
    R = torch.tensor(np.stack([item[3] for item in batch]), dtype=torch.float32, requires_grad=True).squeeze(0)
    E = torch.tensor([item[4] for item in batch], dtype=torch.float32)
    F = torch.tensor(np.stack([item[5] for item in batch])).squeeze(0)
    D = torch.tensor(np.stack([item[6] for item in batch])).squeeze(0)
    idi_i = torch.tensor(np.stack([item[7] for item in batch])).squeeze(0)
    idi_j = torch.tensor(np.stack([item[8] for item in batch])).squeeze(0)
    cell = torch.tensor([item[9] for item in batch]).squeeze(0)
    cell_offsets = torch.tensor(np.stack([item[10] for item in batch])).squeeze(0)
    return Q, S, Z, R, E, F, D, idi_i, idi_j, cell, cell_offsets

def main():
    # 训练配置
    config = {
        'batch_size': 1,
        'learning_rate': 1e-3,
        'num_epochs': 20,
        'data_path': 'carbene_2200.db',
        'checkpoint_path': './',
        'energy_weight': 1.0,
        'force_weight': 100.0,
    }

    def train(model: SpookyNet, loader, criterion, optimizer):
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            charge, spin, atomic_num, coord, energy, forces, D, idx_i, idx_j, cell, cell_offsets = batch
            atomic_num, charge, spin, coord, idx_i, idx_j, cell, cell_offsets, energy, forces = atomic_num.to(model.device), charge.to(model.device), spin.to(model.device), coord.to(model.device), idx_i.to(model.device), idx_j.to(model.device), cell.to(model.device), cell_offsets.to(model.device), energy.to(model.device), forces.to(model.device)
            pred_energy, pred_forces, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6 = model.energy_and_forces(atomic_num, charge, spin, coord, idx_i, idx_j, cell, cell_offsets if cell_offsets else None)

            loss = config['energy_weight'] * criterion(pred_energy, energy) + config['force_weight'] * criterion(pred_forces, forces)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def validate(model: SpookyNet, loader, criterion):
        model.eval()
        total_loss = 0.0
        for batch in loader:
            charge, spin, atomic_num, coord, energy, forces, D, idx_i, idx_j, cell, cell_offsets = batch
            atomic_num, charge, spin, coord, idx_i, idx_j, cell, cell_offsets, energy, forces = atomic_num.to(model.device), charge.to(model.device), spin.to(model.device), coord.to(model.device), idx_i.to(model.device), idx_j.to(model.device), cell.to(model.device), cell_offsets.to(model.device), energy.to(model.device), forces.to(model.device)
            pred_energy, pred_forces, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6 = model.energy_and_forces(atomic_num, charge, spin, coord, idx_i, idx_j, cell, cell_offsets)
            loss = config['energy_weight'] * criterion(pred_energy, energy) + config['force_weight'] * criterion(pred_forces, forces)
            total_loss += loss.item()
        return total_loss / len(loader)

    # 数据加载
    dataset = QMDatabase(config['data_path'])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)

    # 模型定义
    model = SpookyNet()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=10, verbose=True)

    # 训练和验证
    step = 0  # 初始化步数计数器
    for epoch in range(config['num_epochs']):
        for batch in train_loader:
            train_loss = train(model, [batch], criterion, optimizer)
            step += 1
            
            # 每100步输出一次损失和学习率
            if step % 100 == 0:
                val_loss = validate(model, val_loader, criterion)
                current_lr = optimizer.param_groups[0]['lr']  # 获取当前的学习率
                logging.info(f'Step {step}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Learning Rate: {current_lr:.6f}')
                
                # 更新学习率并记录变化
                previous_lr = current_lr
                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != previous_lr:
                    logging.info(f'Learning Rate changed from {previous_lr:.6f} to {new_lr:.6f}')

        # 在每个epoch结束时输出当前epoch的损失
        val_loss = validate(model, val_loader, criterion)  # 确保在epoch结束时进行验证
        logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Learning Rate: {current_lr:.6f}')
        save_checkpoint(model, optimizer, epoch, config['checkpoint_path'])


if __name__ == '__main__':
    main()
