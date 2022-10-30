import argparse

import mindspore as ms
import mindspore.nn as nn
from mindspore.dataset import GeneratorDataset
from mindspore.train.callback import ReduceLROnPlateau

from models import UNet

dir_image = './data/image'
dir_mask = './data/mask'
dir_checkpoint = './checkpoints'


def train(net,
          epochs,
          batch_size,
          learning_rate,
          val_percent,
          save_checkpoint=False):
    # 生成dateset
    dataset = []
    # 分离训练集与验证集
    n_val = len(dataset) * val_percent
    n_train = len(dataset) - n_val
    # 构建date loader
    dataset = GeneratorDataset(source=dataset,
                               column_names=['date', 'label'],
                               num_parallel_workers=4,
                               shuffle=True)
    train_set, val_set = dataset.split([n_train, n_val], randomize=True)
    train_set = train_set.batch(batch_size)
    val_set = val_set.batch(batch_size)
    # 设置优化器, 损失函数, 动态学习率
    optimizer = nn.RMSProp(net.trainable_params(),
                           learning_rate=learning_rate,
                           momentum=0.9,
                           weight_decay=1e-8)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(monitor="loss", mode='max', patience=2)
    # 封装Model
    model = ms.Model(network=net,
                     loss_fn=loss_fn,
                     optimizer=optimizer,
                     metrics={'DICE': nn.Dice(1e-6)})
    # 开始训练
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        model.train(1, train_set, callbacks=[scheduler])
        model.eval(val_set)
        if save_checkpoint:
            ms.save_checkpoint(model, f"{dir_checkpoint}/'model_{epoch+1}.ckpt")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .ckpt file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.2,
                        help='Percent of the data that is used as validation (0-1)')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    unet = UNet(n_channels=1, n_classes=args.classes)
    if args.load:
        param_dict = ms.load_checkpoint(args.load)
        ms.load_param_into_net(unet, param_dict)
    try:
        train(net=unet,
              epochs=args.epochs,
              batch_size=args.batch_size,
              learning_rate=args.lr,
              val_percent=args.val)
    except KeyboardInterrupt:
        ms.save_checkpoint(unet, f"{dir_checkpoint}/INTERRUPTED.ckpt")
        raise
