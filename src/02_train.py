"""
Training and Validation an SVG-like model directly on the quantized latent space
"""

import os
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from lib.arguments import get_pixel_cnn_args
from lib.augmentations import NormalizerIdx
from lib.config import load_exp_config_file
from lib.inference import auto_regress_input
from lib.logger import Logger, print_, log_function, for_all_methods, log_info
import lib.setup_model as setup_model
import lib.utils as utils
from lib.visualizations import visualize_sequence
import data


@for_all_methods(log_function)
class Trainer:
    """
    Class for training a PredFormer model
    """

    def __init__(self, exp_path, vqvae_checkpoint):
        """
        Initializing the trainer object
        """

        utils.set_random_seed()
        self.exp_path = exp_path
        self.exp_params = load_exp_config_file(exp_path)
        self.vqvae_checkpoint = vqvae_checkpoint

        self.plots_path = os.path.join(self.exp_path, "plots", "svg_valid_plots")
        utils.create_directory(self.plots_path)
        self.models_path = os.path.join(self.exp_path, "svg_models")
        utils.create_directory(self.models_path)
        tboard_logs = os.path.join(self.exp_path, "tboard_logs", "svg_logs", f"svg_{utils.timestamp()}")
        utils.create_directory(tboard_logs)

        self.training_losses = []
        self.validation_losses = []
        self.writer = SummaryWriter(tboard_logs)

        self.num_codes = self.exp_params["model"]["VQ-VAE"]["num_codes"]
        self.normalizer = NormalizerIdx(self.num_codes)

        return


    def load_data(self):
        """
        Loading dataset and fitting data-loader for iterating in a batch-like fashion
        """

        # loading dataset and data loaders
        utils.set_random_seed()
        batch_size = self.exp_params["training"]["batch_size"]
        shuffle_train = self.exp_params["dataset"]["shuffled_train"]
        shuffle_eval = self.exp_params["dataset"]["shuffled_eval"]
        num_frames = self.exp_params["dataset"]["num_frames"]

        train_set, self.in_channels = data.load_data(exp_params=self.exp_params, split="train")
        valid_set, _ = data.load_data(exp_params=self.exp_params, split="valid")
        self.valid_set = valid_set
        self.train_loader = data.build_data_loader(dataset=train_set,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle_train)
        # print("NOTE: Redo the shuffling")
        self.valid_loader = data.build_data_loader(dataset=valid_set,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle_eval)

        return


    def setup_model(self):
        """
        Initializing model, optimizer, loss function and other related objects
        """

        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # loading VQ-VAE + pretrained paramters)
        vq_vae = setup_model.load_model(exp_params=self.exp_params,
                                        in_channels=self.in_channels,
                                        type="frame")
        check_path = os.path.join(self.exp_path, "vqvae_models", self.vqvae_checkpoint)
        print_(f"Loading pretrained VQVAE from {check_path}...")
        vq_vae = setup_model.load_checkpoint(checkpoint_path=check_path,
                                             model=vq_vae,
                                             only_model=True)
        vq_vae = vq_vae.eval()
        # obtaining the size of the quantized features
        with torch.no_grad():
            code, _ = vq_vae.encode(self.valid_set[0][0:1])
        feature_dim = list(code.size())[-2:]

        # SetUp HierarchGru
        encoder, decoder, lstm = setup_model.setup_svg()
        # utils.log_architecture(model=hierarchGru,
        #                        exp_path=self.exp_path,
        #                        fname="architecture_hierarchGru.txt")
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.lstm = lstm.to(self.device)
        self.vq_vae = vq_vae.to(self.device)

        # loading optimizer and scheduler
        optimizer_enc, scheduler = setup_model.setup_optimizer(exp_params=self.exp_params, model=encoder)
        optimizer_dec, _ = setup_model.setup_optimizer(exp_params=self.exp_params, model=decoder)
        optimizer_lstm, _ = setup_model.setup_optimizer(exp_params=self.exp_params, model=lstm)

        # loading loss function
        loss_function = nn.CrossEntropyLoss()

        self.loss_function = loss_function
        self.optimizer_lstm = optimizer_lstm
        self.optimizer_enc = optimizer_enc
        self.optimizer_dec = optimizer_dec
        self.scheduler = scheduler

        return


    def training_loop(self):
        """
        Repearting the process validation epoch - train epoch for the number of
        epoch specified in the exp_params file.
        """

        num_epochs = self.exp_params["training"]["num_epochs"]
        save_frequency = self.exp_params["training"]["save_frequency"]

        # iterating for the desired number of epochs
        for epoch in range(num_epochs):
            log_info(message=f"Epoch {epoch}/{num_epochs}")
            self.encoder.eval()
            self.decoder.eval()
            self.lstm.eval()
            self.valid_epoch(epoch)
            self.encoder.train()
            self.decoder.train()
            self.lstm.train()
            self.train_epoch(epoch)

            # adding to tensorboard plot containing both losses
            self.writer.add_scalars(f'loss/CE_comb_loss', {
                'train_loss': self.training_losses[-1],
                'eval_loss': self.validation_losses[-1],
            }, epoch+1)

            # updating learning rate scheduler if loss increases or plateaus
            setup_model.update_scheduler(scheduler=self.scheduler,
                                         exp_params=self.exp_params,
                                         control_metric=self.validation_losses[-1])

            # saving model checkpoint if reached saving frequency
            if(epoch % save_frequency == 0 and epoch != 0):
                print_(f"Saving model checkpoint")
                setup_model.save_checkpoint_svg(encoder=self.encoder, decoder=self.decoder,
                                                predictor=self.lstm, optimizer_enc=self.optimizer_enc,
                                                optimizer_dec=self.optimizer_dec,
                                                optimizer_pred=self.optimizer_lstm,
                                                scheduler=self.scheduler, epoch=epoch,
                                                exp_path=self.exp_path, savedir="svg_models")

            # loging training stats
            utils.log_stats(exp_path=self.exp_path,
                            train_losses=self.training_losses,
                            valid_losses=self.validation_losses,
                            logs_file_name="svg_logs")


        print_(f"Finished training procedure")
        print_(f"Saving final checkpoint")
        setup_model.save_checkpoint_svg(encoder=self.encoder, decoder=self.decoder,
                                        predictor=self.lstm, optimizer_enc=self.optimizer_enc,
                                        optimizer_dec=self.optimizer_dec,
                                        optimizer_pred=self.optimizer_lstm,
                                        scheduler=self.scheduler, epoch=epoch,
                                        exp_path=self.exp_path, savedir="svg_models",
                                        finished=True)

        return


    def train_epoch(self, epoch):
        """
        Training epoch loop
        """

        epoch_losses = []
        teacher_force = self.exp_params["training"]["teacher_force"]
        context = self.exp_params["training"]["context"]
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))

        for i, (inputs) in progress_bar:
            batch_size, frames = inputs.shape[0], inputs.shape[1]
            inputs  = inputs.float().to(self.device)
            self.lstm.init_hidden(batch_size=batch_size)

            # encoding all frames in current batch sequences
            inputs = inputs.view(-1, *inputs.shape[-3:])  # (B, F, C, H, W)  ->  (B*F, C, H, W)
            with torch.no_grad():
                codes_, indices_  = self.vq_vae.encode(inputs)

            # reshaping and aligning code at time T with indices at time T+1
            indices = indices_.view(batch_size, frames, *indices_.shape[-3:])
            indices_target = indices[:,1:]
            indices = self.normalizer(indices[:,:-1])

            # forwarding codes corresponding to the context frames
            output = indices[:,0,:]  # first frame
            preds = []
            for t in range(0, frames-1):
                # encoding indices
                if(t < context):
                    feats, skips = self.encoder(output)
                else:
                    feats, _ = self.encoder(output)
                # prediciting next and decoding
                pred_feats = self.lstm(feats)
                pred_output = self.decoder([pred_feats, skips])
                preds.append(pred_output)
                if(t >= frames-2):
                    continue
                # feeding GT in context or Teacher-Force mode
                if(t < 4 or teacher_force):
                    output = indices[:,t+1,:]
                # autorregresive mode
                else:
                    softmaxed_preds = torch.softmax(pred_output, dim=1)
                    output = torch.argmax(softmaxed_preds, dim=1).unsqueeze(1)
                    output = self.normalizer(output)
                    output = output.float()
            preds = torch.stack(preds, dim=1)

            # computing loss
            preds_loss = preds.permute(0,1,3,4,2).contiguous().view(-1, self.num_codes)
            indices_loss = indices_target.permute(0,1,3,4,2).contiguous().view(-1)
            loss = self.loss_function(preds_loss, indices_loss)
            epoch_losses.append(loss.item())

            self.optimizer_enc.zero_grad()
            self.optimizer_dec.zero_grad()
            self.optimizer_lstm.zero_grad()
            loss.backward()
            self.optimizer_enc.step()
            self.optimizer_dec.step()
            self.optimizer_lstm.step()

            if(i % self.exp_params["training"]["log_frequency"] == 0):
                iter_ = len(self.train_loader) * epoch + i
                self.writer.add_scalar(f'loss/CE_train_loss', np.mean(epoch_losses), global_step=iter_)
                log_data = f"""Log data train iteration {iter_}:  loss={round(np.mean(epoch_losses),5 )};"""
                log_info(message=log_data)

            # update progress bar
            progress_bar.set_description(f"Epoch {epoch+1} iter {i}: train loss {loss.item():.5f}. ")

        train_loss = np.mean(epoch_losses)
        self.training_losses.append(train_loss)
        print_(f"    Train Loss: {train_loss}")

        return


    @torch.no_grad()
    def valid_epoch(self, epoch):
        """
        Validation epoch
        """

        epoch_losses = []
        teacher_force = self.exp_params["training"]["teacher_force"]
        context = self.exp_params["training"]["context"]
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        for i, (inputs_) in progress_bar:
            batch_size, frames = inputs_.shape[0], inputs_.shape[1]
            inputs  = inputs_.float().to(self.device)
            self.lstm.init_hidden(batch_size=batch_size)

            # encoding all frames in current batch sequences
            inputs = inputs.view(-1, *inputs.shape[-3:])  # (B, F, C, H, W)  ->  (B*F, C, H, W)
            codes_, indices_  = self.vq_vae.encode(inputs)

            # reshaping and aligning code at time T with indices at time T+1
            indices = indices_.view(batch_size, frames, *indices_.shape[-3:])
            indices_target = indices[:,1:]
            indices = self.normalizer(indices[:,:-1])

            # forwarding codes corresponding to the context frames
            output = indices[:,0,:]  # first frame
            preds = []
            for t in range(0, frames-1):
                # encoding indices
                if(t < context):
                    feats, skips = self.encoder(output)
                else:
                    feats, _ = self.encoder(output)
                # prediciting next and decoding
                pred_feats = self.lstm(feats)
                pred_output = self.decoder([pred_feats, skips])
                preds.append(pred_output)
                if(t >= frames-2):
                    continue
                # feeding GT in context or Teacher-Force mode
                if(t < 4 or teacher_force):
                    output = indices[:,t+1,:]
                # autorregresive mode
                else:
                    softmaxed_preds = torch.softmax(pred_output, dim=1)
                    output = torch.argmax(softmaxed_preds, dim=1).unsqueeze(1)
                    output = self.normalizer(output)
                    output = output.float()
            preds = torch.stack(preds, dim=1)

            # computing loss
            preds_loss = preds.permute(0,1,3,4,2).contiguous().view(-1, self.num_codes)
            indices_loss = indices_target.permute(0,1,3,4,2).contiguous().view(-1)
            loss = self.loss_function(preds_loss, indices_loss)
            epoch_losses.append(loss.item())

            # visualizing some examples
            if(i == 0):
                # predicting code indices for first sequence in validation set
                preds = preds[0:1]
                softmaxed_preds = torch.softmax(preds, dim=2)
                pred_code_indices = torch.argmax(softmaxed_preds, dim=2)
                # decoding from code domain into pixel space
                # (1, F, 1, h, w)  -->  (F, h, w)  --> (F, 3, H, W)
                pred_code_indices = pred_code_indices.view(-1, *pred_code_indices.shape[-2:]).long()
                pred_frames = self.vq_vae.decode(pred_code_indices).detach().cpu()
                # displaying sequence and adding to Tensorboard
                disp_frames = torch.cat((inputs_[0,:context], pred_frames[context-1:context+11]), axis=0)
                title_1 = [f"Frame {q+1} (GT)" for q in range(4)]
                title_2 = [f"Frame {q+1} (Pred)" for q in range(4,16)]
                titles = title_1 + title_2
                savepath = os.path.join(self.plots_path, f"seq_epoch_{epoch}.png")
                fig, ax = visualize_sequence(sequence=disp_frames.permute(0,2,3,1),
                                             n_frames=16, unnorm=True, savefig=savepath,
                                             titles=titles, get_fig=True)
                self.writer.add_figure(tag=f"Epoch {epoch+1}", figure=fig)

            # update progress bar
            progress_bar.set_description(f"Epoch {epoch+1} iter {i}: valid loss {loss.item():.5f}.")

        valid_loss = np.mean(epoch_losses)
        self.validation_losses.append(valid_loss)
        print_(f"    Valid Loss: {valid_loss}")

        self.writer.add_scalar(f'loss/CE_validation_loss', valid_loss, global_step=epoch+1)
        log_data = f"""Log data validation iteration {epoch+1}:  loss={round(valid_loss, 5)};"""
        log_info(message=log_data)

        return



if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, vqvae_checkpoint = get_pixel_cnn_args()
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting HierarchGRU training procedure", message_type="new_exp")

    print("Initializing Trainer...")
    trainer = Trainer(exp_path=exp_path, vqvae_checkpoint=vqvae_checkpoint)
    print("Loading dataset...")
    trainer.load_data()
    print("Setting up model and optimizer")
    trainer.setup_model()
    print("Starting to train")
    trainer.training_loop()


#
