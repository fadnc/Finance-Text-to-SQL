"""
Logging utility for the Finance Text-to-SQL project.
Provides consistent logging format across all modules.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = "finance_sql",
    log_level: int = logging.INFO,
    log_to_file: bool = True,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Set up and return a configured logger.
    
    Args:
        name: Logger name
        log_level: Logging level (default: INFO)
        log_to_file: Whether to write logs to file
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_path / f"{name}_{timestamp}.log",
            encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "finance_sql") -> logging.Logger:
    """Get an existing logger or create a new one."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


class TrainingLogger:
    """Logger for tracking training metrics."""
    
    def __init__(self, experiment_name: str = "training"):
        self.logger = setup_logger(f"training.{experiment_name}", log_to_file=True)
        self.metrics = []
    
    def log_epoch(self, epoch: int, train_loss: float, eval_loss: float = None):
        """Log epoch metrics."""
        msg = f"Epoch {epoch}: train_loss={train_loss:.4f}"
        if eval_loss is not None:
            msg += f", eval_loss={eval_loss:.4f}"
        self.logger.info(msg)
        self.metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "eval_loss": eval_loss
        })
    
    def log_config(self, config: dict):
        """Log training configuration."""
        self.logger.info("Training Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_result(self, metric_name: str, value: float):
        """Log a final result metric."""
        self.logger.info(f"Result - {metric_name}: {value:.4f}")


class EvalLogger:
    """Logger for tracking evaluation metrics."""
    
    def __init__(self, experiment_name: str = "evaluation"):
        self.logger = setup_logger(f"eval.{experiment_name}", log_to_file=True)
        self.results = []
    
    def log_sample(self, idx: int, predicted: str, expected: str, 
                   exec_match: bool, exact_match: bool, error_type: str = None):
        """Log individual sample evaluation."""
        status = "PASS" if exec_match else "FAIL"
        self.logger.debug(f"Sample {idx} [{status}]: {error_type or 'OK'}")
        self.results.append({
            "idx": idx,
            "predicted": predicted,
            "expected": expected,
            "exec_match": exec_match,
            "exact_match": exact_match,
            "error_type": error_type
        })
    
    def log_summary(self, exec_accuracy: float, exact_accuracy: float, 
                    error_breakdown: dict):
        """Log evaluation summary."""
        self.logger.info("=" * 50)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Execution Accuracy: {exec_accuracy:.2%}")
        self.logger.info(f"Exact Match Accuracy: {exact_accuracy:.2%}")
        self.logger.info("Error Breakdown:")
        for error_type, count in error_breakdown.items():
            self.logger.info(f"  {error_type}: {count}")
