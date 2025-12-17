import argparse
import os
import re

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ModelSummary)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from ht_geogt.data import DataModule
from ht_geogt.model import modeling_priors
from ht_geogt.module import LNNP
from ht_geogt.linear_module import LinearLNNP  # 导入线性注意力模块

from ht_geogt.utils import LoadFromFile, number, save_argparse
from EnhancedData import EnhancedDataset

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import Timer
import time


class EpochTimeCallback(Callback):
    """
    监控每个epoch的训练和验证用时并进行记录
    """

    def __init__(self):
        super().__init__()
        self.train_start_time = None
        self.val_start_time = None
        self.train_times = []
        self.val_times = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.train_start_time:
            train_time = time.time() - self.train_start_time
            self.train_times.append(train_time)
            # 记录训练时间
            pl_module.log("train_epoch_time", train_time, sync_dist=True)
            print(f"Epoch {trainer.current_epoch} 训练用时: {train_time:.2f}秒")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_start_time = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_start_time:
            val_time = time.time() - self.val_start_time
            self.val_times.append(val_time)
            # 记录验证时间
            pl_module.log("val_epoch_time", val_time, sync_dist=True)
            print(f"Epoch {trainer.current_epoch} 验证用时: {val_time:.2f}秒")

            # 记录总时间
            if len(self.train_times) > 0 and trainer.current_epoch < len(self.train_times):
                total_time = self.train_times[trainer.current_epoch] + val_time
                pl_module.log("total_epoch_time", total_time, sync_dist=True)
                print(f"Epoch {trainer.current_epoch} 总用时: {total_time:.2f}秒")

def get_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--load-model",
        default=None,
        type=str,
        help="Restart training using a model checkpoint",
    )  # keep first
    parser.add_argument(
        "--conf",
        "-c",
        type=open,
        action=LoadFromFile,
        help="Configuration yaml file",
    )  # keep second

    # 添加模型类型参数
    parser.add_argument(
        "--model-type",
        type=str,
        default="standard",
        choices=["standard", "linear"],
        help="Model type: standard or linear attention",
    )

    # training settings
    parser.add_argument(
        "--num-epochs", default=20, type=int, help="number of epochs"
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine", "linear"],
        help="Learning rate schedule",
    )
    parser.add_argument(
        "--lr-cosine-length",
        type=int,
        default=0,
        help="Length of cosine schedule. Defaults to 0 for no cosine schedule",
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=20000,
        help="How many steps to warm-up over. Defaults to 0 for no warm-up",
    )
    parser.add_argument("--lr", default=2.e-5, type=float, help="learning rate")
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=15,
        help="Patience for lr-schedule. Patience per eval-interval of validation",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1.e-09,
        help="Minimum learning rate before early stop",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.8,
        help="Minimum learning rate before early stop",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0, help="Weight decay strength"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=150,
        help="Stop training after this many epochs without improvement",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="MSE",
        choices=["MSE", "MAE"],
        help="Loss type",
    )

    # dataset specific
    parser.add_argument(
        "--dataset",
        default='QM9Enhanced',        #=========================
        type=str,
        help="Name of the torch_geometric dataset",
    )
    parser.add_argument(
        "--dataset-arg",
        default='energy_U0',           #dipole_moment  energy_U0
        type=str,
        help="Additional dataset argument",
    )
    parser.add_argument(
        "--dataset-root", default='./data/QM9_enhanced', type=str, help="Data storage directory"
    )
    parser.add_argument(
        "--max-nodes",
        default=None,
        type=int,
        help="Maximum number of nodes for padding in the dataset",
    )
    parser.add_argument(
        "--mean", default=None, type=float, help="Mean of the dataset"
    )
    parser.add_argument(
        "--std",
        default=None,
        type=float,
        help="Standard deviation of the dataset",
    )

    # dataloader specific
    parser.add_argument(
        "--reload",
        type=int,
        default=1,
        help="Reload dataloaders every n epoch",
    )
    parser.add_argument(
        "--batch-size", default=32, type=int, help="batch size"
    )
    parser.add_argument(
        "--inference-batch-size",
        default=64,
        type=int,
        help="Batchsize for validation and tests.",
    )
    parser.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, multiply prediction by dataset std and add mean",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="Npz with splits idx_train, idx_val, idx_test",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default=None,
        help="How to split the dataset. Either random or scaffold",
    )
    parser.add_argument(
        "--train-size",
        type=number,
        default=110000,                       #===========================
        help="Percentage/number of samples in training set (None to use all remaining samples)",
    )
    parser.add_argument(
        "--val-size",
        type=number,
        default=10000,                       #===========================
        help="Percentage/number of samples in validation set (None to use all remaining samples)",
    )
    parser.add_argument(
        "--test-size",
        type=number,
        default=10000,                       #===========================
        help="Percentage/number of samples in test set (None to use all remaining samples)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        help="Number of workers for data prefetch",
    )

    # model architecture specific
    parser.add_argument(
        "--prior-model",
        type=str,
        default='Atomref',
        choices=modeling_priors.__all__,
        help="Which prior model to use",
    )

    # architectural specific
    parser.add_argument(
        "--max-z",
        type=int,
        default=100,
        help="Maximum atomic number that fits in the embedding matrix",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=256, help="Embedding dimension"
    )
    parser.add_argument(
        "--ffn-embedding-dim",
        type=int,
        default=1024,
        help="Embedding dimension for feedforward network",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of interaction layers in the model",
    )
    parser.add_argument(
        "--num-heads", type=int, default=32, help="Number of attention heads"
    )
    parser.add_argument(
        "--cutoff", type=float, default=5.0, help="Cutoff in model"
    )
    parser.add_argument(
        "--num-rbf",
        type=int,
        default=64,
        help="Number of radial basis functions in model",
    )
    parser.add_argument(
        "--trainable-rbf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If distance expansion functions should be trainable",
    )
    parser.add_argument(
        "--norm-type", type=str, default="none", help="Du Normalization type"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout rate"
    )
    parser.add_argument(
        "--attention-dropout",
        type=float,
        default=0.0,
        help="Dropout rate for attention",
    )
    parser.add_argument(
        "--activation-dropout",
        type=float,
        default=0.0,
        help="Dropout rate for activation",
    )
    parser.add_argument(
        "--activation-function",
        type=str,
        default="silu",
        help="Activation function",
    )
    parser.add_argument(
        "--decoder-type", type=str, default="scalar", help="Decoder type"
    )
    parser.add_argument(
        "--aggr",
        type=str,
        default="sum",
        choices=["mean", "sum"],
        help="Aggregation function for output",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1,
        help="Number of classes for classification",
    )
    parser.add_argument(
        "--pad-token-id", type=int, default=0, help="Padding token id"
    )

    # other specific
    parser.add_argument(
        "--distributed-backend",
        type=str,
        default="ddp",
        choices=["ddp", "deepspeed"],
        help="Distributed backend",
    )
    parser.add_argument(
        "--ndevices",
        type=int,
        default=-1,
        help="Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus",
    )
    parser.add_argument(
        "--num-nodes", type=int, default=1, help="Number of nodes"
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        choices=[16, 32],
        help="Floating point precision",
    )
    parser.add_argument(
        "--log-dir", type=str, default='./logs', help="Log directory"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "inference"],
        help="Train or inference",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="random seed (default: 1)"
    )
    parser.add_argument(
        "--redirect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Redirect stdout and stderr to log_dir/log",
    )
    parser.add_argument(
        "--accelerator",
        default="gpu",
        help='Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "auto")',
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="Save interval, one save per n epochs (default: 10)",
    )

    args = parser.parse_args()

    if args.inference_batch_size is None:
        args.inference_batch_size = args.batch_size

    if args.task == "train":
        save_argparse(
            args, os.path.join(args.log_dir, "input.yaml"), exclude=["conf"]
        )

    return args


def auto_exp(args):
    default = ",".join(str(i) for i in range(torch.cuda.device_count()))
    cuda_visible_devices = os.getenv(
        "CUDA_VISIBLE_DEVICES", default=default
    ).split(",")

    # 添加模型类型到目录名
    dir_name = (
        f"ngpus_{len(cuda_visible_devices)}_bs_{args.batch_size}"
        + f"_L{args.num_layers}_D{args.embedding_dim}_F{args.ffn_embedding_dim}"
        + f"_H{args.num_heads}_rbf_{args.num_rbf}"
        + f"_norm_{args.norm_type}_decoder_{args.decoder_type}"
        + f"_lr_{args.lr}"
        + f"_cutoff_{args.cutoff}"
        + f"_split_{args.split_mode}"
        + f"_loss_{args.loss_type}"
        + f"_model_{args.model_type}"  # 添加模型类型
        + f"_seed_{args.seed}"
    )

    if args.load_model is None:
        # resume from checkpoint if cluster breaks down
        args.log_dir = os.path.join(args.log_dir, dir_name)
        if os.path.exists(args.log_dir):
            if os.path.exists(
                os.path.join(args.log_dir, "checkpoints", "last.ckpt")
            ):
                args.load_model = os.path.join(
                    args.log_dir, "checkpoints", "last.ckpt"
                )
                print(
                    f"***** model {args.log_dir} exists, resuming from the last checkpoint *****"
                )
            csv_path = os.path.join(args.log_dir, "metrics", "metrics.csv")
            while os.path.exists(csv_path):
                csv_path = csv_path + ".bak"
            if os.path.exists(
                os.path.join(args.log_dir, "metrics", "metrics.csv")
            ):
                os.rename(
                    os.path.join(args.log_dir, "metrics", "metrics.csv"),
                    csv_path,
                )

    return args


def main():
    args = get_args()

    pl.seed_everything(args.seed, workers=True)

    # initialize data module
    args = auto_exp(args)

    # 初始化原始数据模块
    original_data = DataModule(args)
    original_data.prepare_dataset()

    '''
    # 获取一个原始数据样本用于比较
    print("\n===== 节点特征更新前后对比 =====")
    try:
        # 获取原始数据的一个样本
        train_loader = original_data.train_dataloader()
        original_batch = next(iter(train_loader))

        # 详细打印批次信息
        print(f"原始数据批次类型: {type(original_batch)}")

        if isinstance(original_batch, dict):
            print(f"原始数据是字典，键: {list(original_batch.keys())}")

            # 打印第一张图的z值
            if 'z' in original_batch:
                print(f"\n原始数据第一张图的z值 (形状: {original_batch['z'][0].shape}):")
                print(original_batch['z'][0])

                # 如果z是one-hot编码，尝试解码
                if original_batch['z'][0].dim() > 1 and original_batch['z'][0].shape[1] > 1:
                    print("\nz可能是one-hot编码，尝试解码:")
                    z_indices = original_batch['z'][0].argmax(dim=1)
                    print(f"解码后的z: {z_indices}")

                # 检查是否有其他可能的节点特征
                print("\n检查其他可能的节点特征:")
                for key, value in original_batch.items():
                    if isinstance(value, torch.Tensor) and value.dim() >= 2:
                        if value.shape[0] == original_batch['z'].shape[0] and value.shape[1] == \
                                original_batch['z'].shape[1]:
                            print(f"  '{key}' 可能是节点特征，形状: {value.shape}")
                            if key != 'z':
                                print(f"  第一张图的 {key} 前几个值: {value[0, :3]}")
    except Exception as e:
        print(f"获取原始数据样本时出错: {e}")
        import traceback
        traceback.print_exc()
    '''
    # 初始化原始数据模块
    data = original_data

    # 更新args中的mean和std
    args.mean, args.std = data.mean, data.std

    # 根据模型类型选择不同的模型
    if args.model_type == "linear":
        print("使用线性注意力模型")
        model = LinearLNNP(args)
    else:
        print("使用标准注意力模型")
        model = LNNP(args)

    csv_logger = CSVLogger(args.log_dir, name="metrics", version="")

    if args.task == "train":
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.log_dir, "checkpoints"),
            monitor="val_loss",
            save_top_k=10,
            save_last=True,
            every_n_epochs=args.save_interval,
            filename="{epoch}-{val_loss:.4f}",
        )

        early_stopping = EarlyStopping(
            "val_loss", patience=args.early_stopping_patience
        )
        tb_logger = TensorBoardLogger(
            args.log_dir,
            name="tensorboard",
            version="",
            default_hp_metric=False,
        )

        # 添加时间监控回调
        epoch_time_callback = EpochTimeCallback()
        timer_callback = Timer()  # 使用官方的Timer回调

        strategy = DDPStrategy(find_unused_parameters=False)

        trainer = pl.Trainer(
            max_epochs=args.num_epochs,
            devices=args.ndevices,
            num_nodes=args.num_nodes,
            accelerator=args.accelerator,
            deterministic=True,
            default_root_dir=args.log_dir,
            #callbacks=[early_stopping, checkpoint_callback],
            callbacks=[early_stopping, checkpoint_callback, epoch_time_callback, timer_callback],
            logger=[tb_logger, csv_logger],
            reload_dataloaders_every_n_epochs=args.reload,
            precision=args.precision,
            strategy=strategy,
            enable_progress_bar=True,
            inference_mode=False,
        )

        trainer.fit(model, datamodule=data, ckpt_path=args.load_model)

    test_trainer = pl.Trainer(
        enable_model_summary=True,
        logger=[csv_logger],
        max_epochs=-1,
        num_nodes=1,
        devices=1,
        default_root_dir=args.log_dir,
        enable_progress_bar=True,
        callbacks=[ModelSummary()],
        accelerator=args.accelerator,
        inference_mode=False,
    )

    ''''''
    if args.task == "train":
        trainer.test(
            model=model,
            ckpt_path=trainer.checkpoint_callback.best_model_path,
            datamodule=data,
        )
    elif args.task == "inference":
        ckpt = torch.load(args.load_model, map_location="cpu")
        model.model.load_state_dict(
            {
                re.sub(r"^model\.", "", k): v
                for k, v in ckpt["state_dict"].items()
            }
        )
        test_trainer.test(model=model, datamodule=data)



if __name__ == "__main__":
    main()
