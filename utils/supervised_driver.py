#!/usr/bin/env python
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint, global_step_from_engine
from torch.optim import Optimizer


def create_trainer_engine(
    model,
    optimizer,
    criterion,
    metrics,
    data_loaders,
    lr_scheduler=None,
    save_checkpoint_dir=None,
    device="cuda",
):
    """
    创建训练引擎和评估引擎以进行模型的训练和验证。
    它利用了ignite库中的工具来简化训练循环的管理，包括日志记录、学习率调度、模型检查点保存等。
    
    Args:
        model: 待训练的模型。
        optimizer: 优化器。
        criterion: 损失函数。
        metrics: 评估指标的字典。
        data_loaders: 数据加载器，包含训练集和验证集。
        lr_scheduler: 学习率调度器。
        save_checkpoint_dir: 模型检查点保存路径。
        device: 训练设备。
    """
    # Create trainer
    # 创建训练引擎
    trainer = create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=criterion,
        device=device,
        output_transform=custom_output_transform,
    )

    # 遍历评估指标字典，将评估指标附加到训练引擎上
    for name, metric in metrics.items():
        metric.attach(trainer, name)

    # Add lr_scheduler
    # 添加学习率调度器钩子
    if lr_scheduler:
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: lr_scheduler.step())    # 在`EPOCH_COMPLETED`事件触发点上调用`lr_scheduler.step()`

    # Create evaluator
    # 创建评估引擎
    evaluator = create_supervised_evaluator(model=model, metrics=metrics, device=device)

    # Save model checkpoint
    if save_checkpoint_dir:
        to_save = {"model": model, "optimizer": optimizer}              # 保存模型和优化器
        if lr_scheduler:                                                # 如果有学习率调度器，则也保存
            to_save["lr_scheduler"] = lr_scheduler
        checkpoint = Checkpoint(                                        # 创建检查点处理器
            to_save,
            save_checkpoint_dir,
            n_saved=1,
            global_step_transform=global_step_from_engine(trainer),     # 从训练引擎中获取全局步数
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)   # 将检查点处理器注册到训练引擎的`EPOCH_COMPLETED`事件上

    # Add hooks for logging metrics
    # 添加日志记录钩子
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results, optimizer)

    # 添加评估钩子
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, run_evaluation_for_training, evaluator, data_loaders.val_loader
    )

    return trainer, evaluator


def custom_output_transform(x, y, y_pred, loss):
    """
    自定义训练引擎输出的转换格式。
    训练过程中，ignite的默认输出包括输入数据、真实标签、预测结果和损失值。
    
    Args:
        x: 输入数据。
        y: 真实标签。
        y_pred: 预测结果。
        loss: 损失值。
    """
    return y_pred, y


def log_training_results(trainer, optimizer):
    """
    记录训练引擎在每个epoch结束时的训练指标和当前学习率。
    
    Args:
        trainer: 训练引擎。
        optimizer: 优化器。
    """
    learning_rate = optimizer.param_groups[0]["lr"]         # 获取当前学习率
    log_metrics(trainer.state.metrics, "Training", trainer.state.epoch, learning_rate)  # 将训练指标、阶段名称、当前epoch和学习率进行格式化和打印


def run_evaluation_for_training(trainer, evaluator, val_loader):
    """
    在训练过程中执行验证步骤。
    它在训练引擎每个epoch结束后，使用评估引擎在验证集上进行一次评估，并记录评估指标。
    
    Args:
        trainer: 训练引擎。
        evaluator: 评估引擎。
        val_loader: 验证集数据加载器。
    """
    evaluator.run(val_loader)                               # 使用评估引擎在验证集上进行一次评估
    log_metrics(evaluator.state.metrics, "Evaluation", trainer.state.epoch)     # 将评估指标、阶段名称和当前epoch进行格式化和打印


def log_metrics(metrics, stage: str = "", training_epoch=None, learning_rate=None):
    """
    格式化和打印训练或评估阶段的指标信息。
    根据传入的参数构建日志字符串，并将其输出到控制台。
    
    Args:
        metrics: 评估指标的字典。
        stage: 阶段名称。
        training_epoch: 当前epoch。
        learning_rate: 当前学习率。
    """
    log_text = "  {}".format(metrics) if metrics else ""            # 如果有评估指标，则将其格式化为字符串
    if training_epoch is not None:                                  # 如果有当前epoch，则将其格式化为字符串
        log_text = "Epoch: {}".format(training_epoch) + log_text
    if learning_rate and learning_rate > 0.0:                       # 如果有学习率，则将其格式化为字符串
        log_text += "  Learning rate: {:.2E}".format(learning_rate)
    log_text = "Results - " + log_text
    if stage:                                                       # 如果有阶段名称，则将其格式化为字符串
        log_text = "{} ".format(stage) + log_text
    print(log_text, flush=True)                                     # 打印日志字符串


def setup_tensorboard_logger(trainer, evaluator, output_path, optimizers=None):
    """
    设置并配置`TensorboardLogger`，以便将训练和验证过程中的关键指标记录到Tensorboard中，方便可视化和分析。
    
    Args:
        trainer: 训练引擎。
        evaluator: 评估引擎。
        output_path: 日志输出路径。
        optimizers: 优化器。
    """
    logger = TensorboardLogger(logdir=output_path)                  # 创建日志记录器

    # Attach the logger to log loss and accuracy for both training and validation
    for tag, cur_evaluator in [("train", trainer), ("validation", evaluator)]:  # 遍历训练和验证引擎
        logger.attach_output_handler(                               # 将训练和验证引擎的输出处理程序附加到Tensorboard日志记录器上
            cur_evaluator,
            event_name=Events.EPOCH_COMPLETED,                      # 在`EPOCH_COMPLETED`事件触发点上记录
            tag=tag,
            metric_names="all",                                     # 记录所有评估指标
            global_step_transform=global_step_from_engine(trainer), # 从训练引擎中获取全局步数
        )

    # Log optimizer parameters
    if isinstance(optimizers, Optimizer):                           # 如果优化器是单个优化器，则转换为字典，以便统一处理
        optimizers = {None: optimizers}

    for k, optimizer in optimizers.items():                         # 遍历优化器字典
        logger.attach_opt_params_handler(                           # 将每个优化器的学习率参数附加到Tensorboard日志记录器上
            trainer, Events.EPOCH_COMPLETED, optimizer, param_name="lr", tag=k
        )

    return logger
