import torch
import torch.nn as nn
import lightning as pl
import wandb
import itertools
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from classifier import Classifier
from dataset import CustomDataset


class AttentionGate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionGate, self).__init__()
        self.conv_gate = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, g):
        gate = self.conv_gate(g)
        x = self.conv_x(x)
        attention = self.softmax(gate)
        x_att = x * attention
        return x_att
    
class ResUNetGenerator(nn.Module):
    def __init__(self, gf, channels):
        super(ResUNetGenerator, self).__init__()
        # self.img_shape = img_shape
        self.channels = channels
        
        # Downsampling layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, gf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.GroupNorm(num_groups=1, num_channels=gf)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(gf, gf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.GroupNorm(num_groups=1, num_channels=gf * 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(gf * 2, gf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.GroupNorm(num_groups=1, num_channels=gf * 4)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(gf * 4, gf * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.GroupNorm(num_groups=1, num_channels=gf * 8)
        )

        self.attn_layer = nn.ModuleList([
            AttentionGate(gf * 2**(i), gf * 2**(i+1))
            for i in range(3)
        ])

        # Upsampling layers
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(gf * 8, gf * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_groups=1, num_channels=gf * 4)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(gf * 8, gf * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_groups=1, num_channels=gf * 2)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(gf * 4, gf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_groups=1, num_channels=gf)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(gf * 2, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Downsampling
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        
        # Upsampling
        u1 = self.deconv1(d4)
        u1 = self.attn_layer[2](d3, u1)
        
        u2 = self.deconv2(u1)
        u2 = self.attn_layer[1](d2, u2)
        
        u3 = self.deconv3(u2)
        u3 = self.attn_layer[0](d1, u3)
        
        output = self.deconv4(u3)
        
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return optimizer


class Discriminator(pl.LightningModule):
    def __init__(self, df):
        super(Discriminator, self).__init__()
        self.df = df
        # Define the layers for the discriminator
        self.conv_layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1 if i == 0 else df * 2**(i-1), df * 2**i, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.GroupNorm(8, df * 2**i)) for i in range(4)])
        
        self.final_conv = nn.Conv2d(df * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        out = x
        for conv_layer in self.conv_layers:
            out = conv_layer(out)
        validity = self.final_conv(out)
        return validity

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return optimizer



class CycleGAN(pl.LightningModule):
    def __init__(self, train_dir, val_dir, test_dataloader, classifier_path, image_size=512, batch_size=4, channels=1, gf=32, df=64, lambda_cycle=10.0, lambda_id=0.1, classifier_weight=1):
        super(CycleGAN, self).__init__()
        self.image_size = image_size
        self.batch_size = batch_size
        self.channels = channels
        self.gf = gf
        self.df = df
        self.lambda_cycle = lambda_cycle
        self.lambda_id = lambda_id * lambda_cycle
        self.classifier_path = classifier_path
        self.classifier_weight = classifier_weight
        self.lowest_val_loss = float('inf')
        self.validation_step_outputs = []
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dataloader = test_dataloader

        # Initialize the generator, discriminator, and classifier models
        self.g_NP = ResUNetGenerator(gf, channels=self.channels)
        self.g_PN = ResUNetGenerator(gf, channels=self.channels)
        self.d_N = Discriminator(df)
        self.d_P = Discriminator(df)
        self.automatic_optimization = False
        
        self.classifier = Classifier()
        checkpoint = torch.load(classifier_path)
        self.classifier.load_state_dict(checkpoint['state_dict'])
        self.classifier.eval()
        self.freeze_classifier()
    
    def freeze_classifier(self):
        print("freezing Classifier...")
        for p in self.classifier.parameters() : 
                p.requires_grad = False


    def generator_training_step(self, img_N, img_P, opt):
        self.toggle_optimizer(opt)
        # Translate images to the other domain
        fake_P = self.g_NP(img_N)
        fake_N = self.g_PN(img_P)

        # Translate images back to original domain
        reconstr_N = self.g_PN(fake_P)
        reconstr_P = self.g_NP(fake_N)

        # Identity mapping of images
        img_N_id = self.g_PN(img_N)
        img_P_id = self.g_NP(img_P)
        # Discriminators determine validity of translated images
        valid_N = self.d_N(fake_N)
        valid_P = self.d_P(fake_P)

        class_N_loss = self.classifier(fake_N)
        class_P_loss = self.classifier(fake_P)
        # Adversarial loss
        valid_target = torch.ones_like(valid_N)
        adversarial_loss = nn.MSELoss()(valid_N, valid_target) + nn.MSELoss()(valid_P, valid_target)

        # Cycle consistency loss
        cycle_loss = nn.L1Loss()(reconstr_N, img_N) + nn.L1Loss()(reconstr_P, img_P)

        # Identity loss
        identity_loss = nn.L1Loss()(img_N_id, img_N) + nn.L1Loss()(img_P_id, img_P)

        # Classifier loss
        class_loss = nn.MSELoss()(class_N_loss, torch.ones_like(class_N_loss)) + nn.MSELoss()(class_P_loss, torch.zeros_like(class_P_loss))

        # Total generator loss
        total_loss = adversarial_loss + self.lambda_cycle * cycle_loss + self.lambda_id * identity_loss + self.classifier_weight * class_loss
              
        self.log('adversarial_loss', adversarial_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('reconstruction_loss', cycle_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('identity_loss', identity_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('class_loss', class_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('generator_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        opt.zero_grad()
        self.manual_backward(total_loss)
        opt.step()
        self.untoggle_optimizer(opt)
        
        return total_loss, adversarial_loss, cycle_loss

    def discriminator_training_step(self, img_N, img_P, opt):
        # Pass real images through discriminator D_N
        self.toggle_optimizer(opt)
        pred_real_N = self.d_N(img_N)
        mse_real_N = nn.MSELoss()(pred_real_N, torch.ones_like(pred_real_N))
        fake_P = self.g_PN(img_P)
        pred_fake_N = self.d_N(fake_P)
        mse_fake_N = nn.MSELoss()(pred_fake_N, torch.zeros_like(pred_fake_N))

        pred_real_P = self.d_P(img_P)
        mse_real_P = nn.MSELoss()(pred_real_P, torch.ones_like(pred_real_P))
        fake_N = self.g_NP(img_N)
        pred_fake_P = self.d_P(fake_N)
        mse_fake_P = nn.MSELoss()(pred_fake_P, torch.zeros_like(pred_fake_P))
        
        # Compute total discriminator loss
        dis_loss = 0.5 * (mse_real_N + mse_fake_N + mse_real_P + mse_fake_P)
        opt.zero_grad()
        self.manual_backward(mse_fake_P)
        opt.step()
        self.untoggle_optimizer(opt)
        
        self.log('mse_fake_N', mse_fake_N, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('mse_fake_P', mse_fake_P, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('discriminator_loss', dis_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return dis_loss, mse_fake_N, mse_fake_P
    
    def training_step(self, batch, batch_idx):
        img_N, img_P = batch
        optD, optG = self.optimizers()
        
        total_loss, adversarial_loss, cycle_loss = self.generator_training_step(img_N, img_P, optG)
        dis_loss, mse_fake_N, mse_fake_P = self.discriminator_training_step(img_N, img_P, optD) 
        
        return {"generator_loss": total_loss, "adversarial_loss": adversarial_loss, "reconstruction_loss": cycle_loss, "discriminator_loss": dis_loss, "mse_fake_N": mse_fake_N, "mse_fake_P": mse_fake_P}
    
    def validation_step(self, batch, batch_idx):
        img_N, img_P = batch

        # Translate images to the other domain
        fake_P = self.g_NP(img_N)
        fake_N = self.g_PN(img_P)

        # Translate images back to original domain
        reconstr_N = self.g_PN(fake_P)
        reconstr_P = self.g_NP(fake_N)

        # Identity mapping of images
        img_N_id = self.g_PN(img_N)
        img_P_id = self.g_NP(img_P)

        # Discriminators determine validity of translated images
        valid_N = self.d_N(fake_N)
        valid_P = self.d_P(fake_P)

        class_N_loss = self.classifier(fake_N)
        class_P_loss = self.classifier(fake_P)

        # Adversarial loss
        valid_target = torch.ones_like(valid_N)
        adversarial_loss = nn.MSELoss()(valid_N, valid_target) + nn.MSELoss()(valid_P, valid_target)

        # Cycle consistency loss
        cycle_loss = nn.L1Loss()(reconstr_N, img_N) + nn.L1Loss()(reconstr_P, img_P)

        # Identity loss
        identity_loss = nn.L1Loss()(img_N_id, img_N) + nn.L1Loss()(img_P_id, img_P)

        # Classifier loss
        class_loss = nn.MSELoss()(class_N_loss, torch.ones_like(class_N_loss)) + nn.MSELoss()(class_P_loss, torch.zeros_like(class_P_loss))

        # Total generator loss
        total_loss = adversarial_loss + self.lambda_cycle * cycle_loss + self.lambda_id * identity_loss + self.classifier_weight * class_loss
        self.validation_step_outputs.append(total_loss)

        self.log('val_adversarial_loss', adversarial_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_cycle_loss', cycle_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_identity_loss', identity_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_class_loss', class_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_generator_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def on_validation_epoch_end(self):
        # Calculate average validation loss
        avg_val_loss = torch.stack(self.validation_step_outputs).mean()

        # Check if current validation loss is lower than the lowest recorded validation loss
        if avg_val_loss < self.lowest_val_loss:
            # Update lowest validation loss and corresponding epoch
            self.lowest_val_loss = avg_val_loss

            # Save the generators' state dictionaries
            torch.save(self.g_NP.state_dict(), f"/teamspace/studios/this_studio/Counterfactual-Image-Generation-using-CycleGAN/models/gan/g_NP_best.ckpt")
            torch.save(self.g_PN.state_dict(), f"/teamspace/studios/this_studio/Counterfactual-Image-Generation-using-CycleGAN/models/gan/g_PN_best.ckpt")

    def configure_optimizers(self):
        optG = torch.optim.Adam(itertools.chain(self.g_NP.parameters(), self.g_PN.parameters()),lr=2e-4, betas=(0.5, 0.999))
        optD = torch.optim.Adam(itertools.chain(self.d_N.parameters(), self.d_P.parameters()),lr=2e-4, betas=(0.5, 0.999))
        
        gamma = lambda epoch: 1 - max(0, epoch + 1 - 100) / 101
        schD = LambdaLR(optD, lr_lambda=gamma)
#         Optimizer= [optD, optG]
        return optD, optG

    def train_dataloader(self):
        root_dir = self.train_dir
        train_N = "0"
        train_P = "1"
        img_res = (self.image_size, self.image_size)

        dataset = CustomDataset(root_dir=root_dir, train_N=train_N, train_P=train_P, img_res=img_res)

        # Set up DataLoader for parallel processing and GPU acceleration
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        return dataloader
    
    def val_dataloader(self):
        root_dir = self.val_dir
        train_N = "0"
        train_P = "1"
        img_res = (self.image_size, self.image_size)

        dataset = CustomDataset(root_dir=root_dir, train_N=train_N, train_P=train_P, img_res=img_res)

        # Set up DataLoader for parallel processing and GPU acceleration
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        return dataloader
     

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx % 100 == 0:
            # Get a random batch from the test dataloader
            batch = next(iter(self.test_dataloader))
            img_N, img_P = batch

            # Pick a random image from the batch
            idx = np.random.randint(img_N.size(0))
            img_N = img_N[idx].unsqueeze(0).to('cuda')
            img_P = img_P[idx].unsqueeze(0).to('cuda')
            # Translate images to the other domain
            fake_P = self.g_NP(img_N)
            fake_N = self.g_PN(img_P)

            # Translate images back to original domain
            reconstr_N = self.g_PN(fake_P)
            reconstr_P = self.g_NP(fake_N)

            # Plot the images
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Plot real N, translated P, and reconstructed N
            axes[0, 0].imshow(img_N.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(), cmap='gray')
            axes[0, 0].set_title("Real N")
            axes[0, 0].axis('off')

            axes[0, 1].imshow(fake_P.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(), cmap='gray')
            axes[0, 1].set_title("Translated P")
            axes[0, 1].axis('off')

            axes[0, 2].imshow(reconstr_N.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(), cmap='gray')
            axes[0, 2].set_title("Reconstructed N")
            axes[0, 2].axis('off')

            # Plot real P, translated N, and reconstructed P
            axes[1, 0].imshow(img_P.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(), cmap='gray')
            axes[1, 0].set_title("Real P")
            axes[1, 0].axis('off')

            axes[1, 1].imshow(fake_N.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(), cmap='gray')
            axes[1, 1].set_title("Translated N")
            axes[1, 1].axis('off')

            axes[1, 2].imshow(reconstr_P.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(), cmap='gray')
            axes[1, 2].set_title("Reconstructed P")
            axes[1, 2].axis('off')

            # Log the figure in WandB
            wandb.log({"test_images": wandb.Image(fig)})

            plt.close(fig)