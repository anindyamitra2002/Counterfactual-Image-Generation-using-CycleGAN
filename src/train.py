from torchvision import transforms
from torch.utils.data import DataLoader
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as pl
import wandb

from dataset import ClassifierDataset, CustomDataset
from classifier import Classifier
from models import CycleGAN
from config import CFG

def train_classifier(image_size,
                     batch_size,
                     epochs,
                     resume_ckpt_path,
                     train_dir,
                     val_dir,
                     checkpoint_dir,
                     project,
                     job_name):
    
    clf_wandb_logger = WandbLogger(project=project, name=job_name, log_model="all")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize image to 512x512
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize image
    ])

    # Define dataset paths
    # train_dir = "/kaggle/working/CycleGan-CFE/train-data/train"
    # val_dir = "/kaggle/working/CycleGan-CFE/train-data/val"

    # Create datasets
    train_dataset = ClassifierDataset(root_dir=train_dir, transform=transform)
    val_dataset = ClassifierDataset(root_dir=val_dir, transform=transform)
    print("Total Training Images: ",len(train_dataset))
    print("Total Validation Images: ",len(val_dataset))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    # Instantiate the classifier model
    clf = Classifier(transfer=True)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename='swin_t-epoch{epoch:02d}-val_loss{val_loss:.2f}',
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=3,
        mode='min'
    )
    # Set up PyTorch Lightning Trainer with multiple GPUs and tqdm progress bar
    trainer = pl.Trainer(
        devices="auto",
        precision="16-mixed",
        accelerator="auto",
        max_epochs=epochs,
        accumulate_grad_batches=10,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        benchmark=True,
        logger=clf_wandb_logger,
        callbacks=[checkpoint_callback],
    )

    # Train the classifier
    trainer.fit(clf, train_loader, val_loader, ckpt_path=resume_ckpt_path)
    wandb.finish()


def train_cyclegan(image_size,
                   batch_size,
                   epochs,
                   classifier_path,
                   resume_ckpt_path,
                   train_dir,
                   val_dir,
                   test_dir,
                   checkpoint_dir,
                   project,
                   job_name,
):


    testdata_dir = test_dir
    train_N = "0"
    train_P = "1"
    img_res = (image_size, image_size)

    test_dataset = CustomDataset(root_dir=testdata_dir, train_N=train_N, train_P=train_P, img_res=img_res)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    wandb_logger = WandbLogger(project=project, name=job_name, log_model="all")
    print(classifier_path)
    cyclegan = CycleGAN(train_dir=train_dir, val_dir=val_dir, test_dataloader=test_dataloader, classifier_path=classifier_path, gf=CFG.GAN_FILTERS, df=CFG.DIS_FILTERS)

    gan_checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir,
                                        filename='cyclegan-epoch_{epoch}-vloss_{val_generator_loss:.2f}',
                                        monitor='val_generator_loss',
                                        save_top_k=3,
                                        save_last=True,
                                        save_weights_only=False,
                                        verbose=True,
                                        mode='min')


    # Create the trainer
    trainer = pl.Trainer(
        accelerator="auto",
        precision="16-mixed",
        max_epochs=epochs,
        log_every_n_steps=1,
        benchmark=True,
        devices="auto",
        logger=wandb_logger,
        callbacks= [gan_checkpoint_callback]
    )

    # Train the CycleGAN model
    trainer.fit(cyclegan, ckpt_path=resume_ckpt_path)
