"""
Multi-modal/Multi-view Deep Learning Model Training Script
"""

import os
import argparse
import warnings
from typing import Dict, Tuple, Any

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Custom module imports
from models.TEF import ETEF, ce_loss
from TEF_Main.datasets.nus_wide import Multi_view_data  # Switch datasets as needed
from utils.utils import set_seed, save_checkpoint, load_checkpoint, log_metrics
from utils.logger import create_logger
import metrics as metricss


# ======================== Configuration Parameters ========================
class Config:
    """Training Configuration Parameters"""
    # GPU Settings
    GPU_DEVICE = 1

    # Data dimension configuration (adjust according to dataset)
    VIEW_DIMS = [64, 225, 144, 73, 128, 500, 1000]

    # Data split index
    IDX_SPLIT = 0

    # Fusion configuration (adjust as needed)
    FUSION_LIST = '4-6-2-4-2-3-4-5-4-4-4-4-0-0-4'


# ======================== Argument Parser ========================
def get_args(parser: argparse.ArgumentParser) -> None:
    """
    Add command line arguments

    Args:
        parser: ArgumentParser object
    """
    # Training parameters
    parser.add_argument("--batch_sz", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=500,
                        help="Maximum number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3,
                        help="Number of gradient accumulation steps")

    # Data parameters
    parser.add_argument("--data_path", type=str, default="./datasets/nyud2/",
                        help="Dataset path")
    parser.add_argument("--LOAD_SIZE", type=int, default=256,
                        help="Image loading size")
    parser.add_argument("--FINE_SIZE", type=int, default=224,
                        help="Final image size")
    parser.add_argument("--n_workers", type=int, default=12,
                        help="Number of data loading workers")

    # Model parameters
    parser.add_argument("--hidden", nargs="*", type=int, default=[128],
                        help="Hidden layer dimensions")
    parser.add_argument("--hidden_sz", type=int, default=768,
                        help="Hidden layer size")
    parser.add_argument("--img_hidden_sz", type=int, default=512,
                        help="Image hidden layer size")
    parser.add_argument("--img_embed_pool_type", type=str, default="avg",
                        choices=["max", "avg"],
                        help="Image embedding pooling type")
    parser.add_argument("--num_image_embeds", type=int, default=1,
                        help="Number of image embeddings")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")
    parser.add_argument("--include_bn", type=int, default=True,
                        help="Include batch normalization")

    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--lr_factor", type=float, default=0.3,
                        help="Learning rate decay factor")
    parser.add_argument("--lr_patience", type=int, default=20,
                        help="Learning rate scheduler patience")

    # Training control parameters
    parser.add_argument("--patience", type=int, default=100,
                        help="Early stopping patience")
    parser.add_argument("--annealing_epoch", type=int, default=20,
                        help="Annealing epochs")
    parser.add_argument("--n_classes", type=int, default=10,
                        help="Number of classes")

    # Other parameters
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("--name", type=str, default="ReleasedVersion_IPS",
                        help="Experiment name")
    parser.add_argument("--savedir", type=str, default="/data/TEF-Main/",
                        help="Save directory")

    # GPU settings
    parser.add_argument("--gpu_device", type=int, default=1,
                        help="GPU device ID")


# ======================== Optimizer and Scheduler ========================
def get_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> optim.Optimizer:
    """
    Get optimizer

    Args:
        model: Model
        args: Arguments

    Returns:
        optimizer: Adam optimizer
    """
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5
    )
    return optimizer


def get_scheduler(optimizer: optim.Optimizer, args: argparse.Namespace) -> optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler

    Args:
        optimizer: Optimizer
        args: Arguments

    Returns:
        scheduler: ReduceLROnPlateau scheduler
    """
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=args.lr_patience,
        verbose=True,
        factor=args.lr_factor
    )


# ======================== Data Loading ========================
def get_data_transforms(args: argparse.Namespace) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get data transforms

    Args:
        args: Arguments

    Returns:
        train_transform: Training data transforms
        val_transform: Validation data transforms
    """
    # Training data transforms
    train_transforms_list = [
        transforms.Resize((args.LOAD_SIZE, args.LOAD_SIZE)),
        transforms.RandomCrop((args.FINE_SIZE, args.FINE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.6983, 0.3918, 0.4474],
            std=[0.1648, 0.1359, 0.1644]
        )
    ]

    # Validation data transforms
    val_transforms_list = [
        transforms.Resize((args.FINE_SIZE, args.FINE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.6983, 0.3918, 0.4474],
            std=[0.1648, 0.1359, 0.1644]
        )
    ]

    return transforms.Compose(train_transforms_list), transforms.Compose(val_transforms_list)


def get_data_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """
    Get data loaders

    Args:
        args: Arguments

    Returns:
        train_loader: Training data loader
        test_loader: Test data loader
    """
    train_loader = DataLoader(
        Multi_view_data(args.data_path, train=True, idx_split=Config.IDX_SPLIT),
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers
    )

    test_loader = DataLoader(
        Multi_view_data(args.data_path, train=False, idx_split=Config.IDX_SPLIT),
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers
    )

    return train_loader, test_loader


# ======================== Model Forward Pass ========================
def model_forward(
        i_epoch: int,
        model: torch.nn.Module,
        args: argparse.Namespace,
        criterion: Any,
        batch: Tuple
) -> Tuple:
    """
    Model forward pass

    Args:
        i_epoch: Current epoch
        model: Model
        args: Arguments
        criterion: Loss function
        batch: Batch data

    Returns:
        loss: Total loss
        Predictions and targets for each view
    """
    # Unpack data
    batch_data = batch[0]
    views = {
        'ecape': batch_data[0],
        're': batch_data[1],
        'fb': batch_data[2],
        'mf': batch_data[3],
        'soec': batch_data[4],
        'A': batch_data[5],
        'B': batch_data[6],
        'all': batch_data[7]
    }
    target = batch[1]

    # Move data to GPU
    for key in views:
        views[key] = views[key].cuda()
    target = target.cuda()

    # Forward pass
    outputs = model(
        views['ecape'], views['re'], views['fb'], views['mf'],
        views['soec'], views['A'], views['B'], views['all']
    )

    # Unpack outputs
    (ecape_alpha, re_alpha, fb_alpha, mf_alpha, soec_alpha,
     A_alpha, B_alpha, pseudo_alpha, depth_rgb_alpha) = outputs

    # Calculate loss
    loss = 0
    for alpha in outputs:
        loss += criterion(target, alpha, args.n_classes, i_epoch, args.annealing_epoch)

    return loss, *outputs, target


# ======================== Model Evaluation ========================
def model_eval(
        i_epoch: int,
        data_loader: DataLoader,
        model: torch.nn.Module,
        args: argparse.Namespace,
        criterion: Any
) -> Dict[str, float]:
    """
    Evaluate model performance

    Args:
        i_epoch: Current epoch
        data_loader: Data loader
        model: Model
        args: Arguments
        criterion: Loss function

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()

    # Initialize storage lists
    losses = []
    predictions = {
        'ecape': [], 're': [], 'fb': [], 'mf': [],
        'soec': [], 'A': [], 'B': [], 'ALL': [], 'depth_rgb': []
    }
    targets = []

    with torch.no_grad():
        for batch in data_loader:
            # Forward pass
            outputs = model_forward(i_epoch, model, args, criterion, batch)
            loss = outputs[0]
            losses.append(loss.item())

            # Get predictions
            alphas = outputs[1:-1]  # All alpha outputs
            target = outputs[-1]

            # Convert to prediction classes
            pred_keys = list(predictions.keys())
            for i, alpha in enumerate(alphas):
                if i < len(pred_keys):
                    pred = alpha.argmax(dim=1).cpu().detach().numpy()
                    predictions[pred_keys[i]].append(pred)

            # Save targets
            targets.append(target.cpu().detach().numpy())

    # Flatten lists
    targets = np.concatenate(targets)
    for key in predictions:
        if predictions[key]:
            predictions[key] = np.concatenate(predictions[key])

    # Calculate metrics
    metrics = {"loss": np.mean(losses)}

    # Calculate accuracy for each view
    view_names = {
        'ecape': 'ecape_acc',
        're': 're_acc',
        'fb': 'fb_acc',
        'mf': 'mf_acc',
        'soec': 'soec_acc',
        'A': 'A_preds',
        'B': 'B_preds',
        'ALL': 'ALLpres',
        'depth_rgb': 'depthrgb_acc'
    }

    for key, metric_name in view_names.items():
        if key in predictions and len(predictions[key]) > 0:
            metrics[metric_name] = accuracy_score(targets, predictions[key])

    # Calculate additional evaluation metrics
    if 'depth_rgb' in predictions and len(predictions['depth_rgb']) > 0:
        results = metricss.get_results(targets, predictions['depth_rgb'])
        metrics.update({
            "0": results[0],  # Accuracy
            "1": results[1],  # Recall
            "2": results[2],  # Precision
            "3": results[3],  # F1 Score
            "4": results[4]  # Kappa
        })

    return metrics


# ======================== Training Loop ========================
def train_epoch(
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        args: argparse.Namespace,
        epoch: int
) -> float:
    """
    Train for one epoch

    Args:
        model: Model
        train_loader: Training data loader
        optimizer: Optimizer
        args: Arguments
        epoch: Current epoch

    Returns:
        avg_loss: Average loss
    """
    model.train()
    train_losses = []
    optimizer.zero_grad()

    global_step = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        # Forward pass
        outputs = model_forward(epoch, model, args, ce_loss, batch)
        loss = outputs[0]

        # Gradient accumulation
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        train_losses.append(loss.item())
        loss.backward()

        global_step += 1
        if global_step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return np.mean(train_losses)


# ======================== Main Training Function ========================
def train(args: argparse.Namespace) -> None:
    """
    Main training function

    Args:
        args: Training arguments
    """
    # Set GPU
    torch.cuda.set_device(args.gpu_device)

    # Set dimensions and seed
    args.dims = Config.VIEW_DIMS
    set_seed(args.seed)

    # Create save directory
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    # Get data loaders
    train_loader, test_loader = get_data_loaders(args)

    # Initialize model
    model = ETEF(args).cuda()

    # Initialize optimizer and scheduler
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    # Create logger
    logger = create_logger(f"{args.savedir}/logfile.log", args)

    # Save arguments
    torch.save(args, os.path.join(args.savedir, "args.pt"))

    # Initialize training state
    start_epoch = 0
    n_no_improve = 0
    best_metric = -np.inf

    # Load checkpoint if exists
    checkpoint_path = os.path.join(args.savedir, "checkpoint.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    logger.info("=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)

    for epoch in range(start_epoch, args.max_epochs):
        # Train one epoch
        train_loss = train_epoch(model, train_loader, optimizer, args, epoch)
        logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")

        # Evaluate
        model.eval()
        val_metrics = model_eval(np.inf, test_loader, model, args, ce_loss)

        # Log validation metrics
        log_metrics("val", val_metrics, logger)
        logger.info(
            f"Val: Loss: {val_metrics['loss']:.5f} | "
            f"depth_acc: {val_metrics.get('ecape_acc', 0):.5f}, "
            f"rgb_acc: {val_metrics.get('ALLpres', 0):.5f}, "
            f"depth_rgb_acc: {val_metrics.get('depthrgb_acc', 0):.5f}, "
            f"acc: {val_metrics.get('0', 0):.5f}, "
            f"recall: {val_metrics.get('1', 0):.5f}, "
            f"precision: {val_metrics.get('2', 0):.5f}, "
            f"f1: {val_metrics.get('3', 0):.5f}, "
            f"kappa: {val_metrics.get('4', 0):.5f}"
        )

        # Learning rate scheduling
        tuning_metric = val_metrics.get("depthrgb_acc", val_metrics.get("loss"))
        scheduler.step(tuning_metric)

        # Check for improvement
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
            logger.info(f"New best metric: {best_metric:.5f}")
        else:
            n_no_improve += 1

        # Save checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )

        # Early stopping
        if n_no_improve >= args.patience:
            logger.info(f"No improvement for {args.patience} epochs. Early stopping...")
            break

    # Final testing with best model
    logger.info("=" * 60)
    logger.info("Final Testing with Best Model")
    logger.info("=" * 60)

    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()
    test_metrics = model_eval(np.inf, test_loader, model, args, ce_loss)

    logger.info(
        f"Test: Loss: {test_metrics['loss']:.5f} | "
        f"depth_acc: {test_metrics.get('ecape_acc', 0):.5f}, "
        f"rgb_acc: {test_metrics.get('ALLpres', 0):.5f}, "
        f"depth_rgb_acc: {test_metrics.get('depthrgb_acc', 0):.5f}, "
        f"acc: {test_metrics.get('0', 0):.5f}, "
        f"recall: {test_metrics.get('1', 0):.5f}, "
        f"precision: {test_metrics.get('2', 0):.5f}, "
        f"f1: {test_metrics.get('3', 0):.5f}, "
        f"kappa: {test_metrics.get('4', 0):.5f}"
    )

    log_metrics("Test", test_metrics, logger)

    logger.info("=" * 60)
    logger.info("Training Completed Successfully!")
    logger.info("=" * 60)


# ======================== Main Entry Point ========================
def cli_main():
    """Command line interface main function"""
    parser = argparse.ArgumentParser(
        description="Train Multi-modal/Multi-view Classification Model (TEF)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    get_args(parser)
    args, remaining_args = parser.parse_known_args()

    if remaining_args:
        parser.error(f"Unknown arguments: {remaining_args}")

    train(args)


if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Run main function
    cli_main()