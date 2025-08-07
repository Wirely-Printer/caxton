import os
from glob import glob
import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from data.data_module import ParametersDataModule
from model.network_module import ParametersClassifier
from train_config import *
import multiprocessing


def get_latest_checkpoint(checkpoint_dir):
    """
    Finds the most recent checkpoint in the specified directory.
    
    Args:
        checkpoint_dir (str): Path to the directory containing checkpoint files.
    
    Returns:
        str: Path to the latest checkpoint or None if no checkpoints are found.
    """
    checkpoint_files = glob(os.path.join(checkpoint_dir, "epoch*.ckpt"))
    if not checkpoint_files:
        print("No checkpoints found in directory:", checkpoint_dir)
        return None
    # Sort by modification time (most recent first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    print(f"Latest checkpoint: {checkpoint_files[0]}")
    return checkpoint_files[0]


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # Ensures compatibility on Windows

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--seed", default=1234, type=int, help="Set seed for training"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=MAX_EPOCHS,
        type=int,
        help="Number of epochs to train the model for",
    )
    args = parser.parse_args()
    seed = args.seed

    # Set seed for reproducibility
    set_seed(seed)
    logs_dir = "logs/logs-{}/{}/".format(DATE, seed)
    logs_dir_default = os.path.join(logs_dir, "default")

    # Create necessary directories
    make_dirs(logs_dir)
    make_dirs(logs_dir_default)

    # Setup logger and checkpoint callback
    tb_logger = pl_loggers.TensorBoardLogger(logs_dir)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/{}/{}/".format(DATE, seed),
        filename="MHResAttNet-{}-{}-".format(DATASET_NAME, DATE)
        + "{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
        save_top_k=3,
        mode="min",
    )

    save_every_2_epochs_callback = ModelCheckpoint(
        every_n_epochs=2,
        dirpath="checkpoints/every_2_epochs/{}/{}/".format(DATE, seed),
        filename="epoch{epoch:02d}",
        save_top_k=-1,
    )

    # Initialize the model
    model = ParametersClassifier(
        num_classes=3,
        lr=INITIAL_LR,
        gpus=NUM_GPUS,
        transfer=False,
    )

    # Initialize the data module with num_workers set to 0 or 1 for Windows compatibility
    data = ParametersDataModule(
        batch_size=BATCH_SIZE,
        data_dir=DATA_DIR,
        csv_file=DATA_CSV,
        dataset_name=DATASET_NAME,
        mean=DATASET_MEAN,
        std=DATASET_STD,
        num_workers=1  # Important for avoiding multiprocessing issues on Windows
    )

    # Directory for saving checkpoints
    checkpoint_dir = "C:\\Users\\lab15\\miniconda3\\caxton\\src\\checkpoints\\every_2_epochs\\25112024\\1234"
    print(f"Looking for checkpoints in: {checkpoint_dir}")

    # Find the latest checkpoint
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)

    # Configure the PyTorch Lightning trainer
    trainer = pl.Trainer(
        num_nodes=NUM_NODES,
        devices=NUM_GPUS,  # Use the number of GPUs (or devices) directly here
        accelerator="gpu" if NUM_GPUS > 0 else "cpu",  # Automatically select "gpu" or "cpu"
        max_epochs=args.epochs,
        logger=tb_logger,
        enable_model_summary=False,  # Replaces deprecated `weights_summary=None`
        precision=32,  # Mixed precision if running on a GPU
        callbacks=[checkpoint_callback, save_every_2_epochs_callback],
    )

    # Resume training from the latest checkpoint if available
    if latest_checkpoint:
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        trainer.fit(model, data, ckpt_path=latest_checkpoint)
    else:
        print("No checkpoint found. Starting training from scratch.")
        trainer.fit(model, data)
