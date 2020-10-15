import os
import argparse
import tqdm
import torch
import numpy as np
import yaml

from anomaly_detection.utils.loggers import Logger
from anomaly_detection.dpa.pg_networks import ProgGrowStageType, NetworkType
from anomaly_detection.dpa.pg_decoders import DECODER_NETWORKS
from anomaly_detection.dpa.pg_encoders import ENCODER_NETWORKS
from anomaly_detection.dpa.other_networks import FakeProgGrowNetworks
from anomaly_detection.dpa.data_generators import ProgGrowImageGenerator, MixResolution
from anomaly_detection.utils.datasets import DatasetType, DATASETS
from anomaly_detection.utils.transforms import TRANSFORMS
from anomaly_detection.dpa.optimizer import Optimizer
from anomaly_detection.dpa.adv_losses import ADVERSARIAL_LOSSES, AdversarialLossType
from anomaly_detection.dpa.rec_losses import ReconstructionLossType
from anomaly_detection.dpa.pg_rec_losses import PG_RECONSTRUCTION_LOSSES


torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, config):
        self.config = config
        self.verbose = config['verbose']
        self.random_seed = config['random_seed']
        self.finetune_from = config['finetune_from']

        self.checkpoint_root = config['checkpoint_root']
        self.log_root = config['log_root']

        self.max_image_res = config['max_image_res']
        self.initial_image_res = config['initial_image_res']
        self.image_dim = config['image_dim']
        self.latent_res = config['latent_res']
        self.latent_dim = config['latent_dim']

        trns_iter = config['trns_iter']
        stab_iter = config['stab_iter']
        self.iters = {
            ProgGrowStageType.stab: stab_iter,
            ProgGrowStageType.trns: trns_iter
        }
        # optionally you can change the number of iterations for a specific resolution
        self.iters_per_res = {
            int(res): iters for res, iters in config.get('iters_per_res', {}).items()
        }

        self.log_iter = config['log_iter']
        self.val_iter = config['val_iter']
        self.early_stopping_patience = config['early_stopping_patience']
        self.early_stopping_min_delta = config['early_stopping_min_delta']
        self.image_sample_iter = config['image_sample_iter']

        self.batch_sizes = {int(res): batch_size for res, batch_size in config['batch_sizes'].items()}
        self.n_dis = config.get('n_dis', 0)

        "=========================================== initialize ======================================================="

        if self.verbose:
            print(yaml.dump(self.config, default_flow_style=False))

        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)

        self.stage = ProgGrowStageType.stab
        self.resolution = self.initial_image_res
        self.progress = 0
        self.n_iter = 0
        self.best_val_iter = 0
        self.best_val_loss = 10e6
        self.batch_size = int(self.batch_sizes[self.resolution])

        os.makedirs(self.checkpoint_root, exist_ok=True)
        self.logger = Logger(self.log_root)

        "=========================================== create datasets ================================================"

        dataset_type = config['train_dataset']['dataset_type']
        dataset_kwargs = config['train_dataset']['dataset_kwargs']
        transform_kwargs = config['train_dataset']['transform_kwargs']

        transform = TRANSFORMS[DatasetType[dataset_type]](**transform_kwargs)
        dataset = DATASETS[DatasetType[dataset_type]](
            transform=transform,
            **dataset_kwargs
        )

        dataset_type = config['val_dataset']['dataset_type']
        dataset_kwargs = config['val_dataset']['dataset_kwargs']
        transform_kwargs = config['val_dataset']['transform_kwargs']

        transform = TRANSFORMS[DatasetType[dataset_type]](**transform_kwargs)
        val_dataset = DATASETS[DatasetType[dataset_type]](
            transform=transform,
            **dataset_kwargs
        )

        self.pg_image_gen = ProgGrowImageGenerator(dataset, self.max_image_res,
                                                   batch_size=self.batch_sizes[self.resolution], inf=True)
        self.val_pg_image_gen = ProgGrowImageGenerator(val_dataset, self.max_image_res,
                                                       batch_size=self.batch_sizes[self.resolution], inf=False)

        "============================================= create networks ================================================"

        mtype = config['enc']['type']
        kwargs = config['enc']['kwargs']
        self.enc_pg_networks = ENCODER_NETWORKS[NetworkType[mtype]](
            max_input_res=self.max_image_res,
            output_res=self.latent_res,
            input_dim=self.image_dim,
            output_dim=self.latent_dim,
            **kwargs
        )

        mtype = config['dec']['type']
        kwargs = config['dec']['kwargs']
        self.dec_pg_networks = DECODER_NETWORKS[NetworkType[mtype]](
            input_res=self.latent_res,
            max_output_res=self.max_image_res,
            input_dim=self.latent_dim,
            output_dim=self.image_dim,
            **kwargs
        )

        # optional discriminator
        self.ddis_pg_networks = FakeProgGrowNetworks()
        if config.get('ddis') is not None:
            mtype = config['ddis']['type']
            kwargs = config['ddis']['kwargs']
            self.ddis_pg_networks = ENCODER_NETWORKS[NetworkType[mtype]](
                max_input_res=self.max_image_res,
                output_res=1,
                input_dim=self.image_dim,
                output_dim=1,
                **kwargs
            )

        if self.verbose:
            print("================================== MODELS ON LARGEST RESOLUTION ==================================")
            print("====================== Encoder =============================")
            print(self.enc_pg_networks.get_net(ProgGrowStageType.stab, self.max_image_res))
            print("====================== Decoder  ===========================")
            print(self.dec_pg_networks.get_net(ProgGrowStageType.stab, self.max_image_res))

            print("====================== Discriminator ==========")
            print(self.ddis_pg_networks.get_net(ProgGrowStageType.stab, self.max_image_res))

        self.dec = self.dec_pg_networks.get_net(self.stage, self.resolution).cuda()
        self.ddis = self.ddis_pg_networks.get_net(self.stage, self.resolution).cuda()
        self.enc = self.enc_pg_networks.get_net(self.stage, self.resolution).cuda()

        "=========================================== create reconstruction losses ====================================="

        # create rec loss
        loss_type = config['image_rec_loss']['loss_type']
        loss_kwargs = config['image_rec_loss']['loss_kwargs']
        self.image_rec_loss = PG_RECONSTRUCTION_LOSSES[ReconstructionLossType[loss_type]](
            max_resolution=self.max_image_res, **loss_kwargs)
        self.image_rec_loss.set_stage_resolution(self.stage, self.resolution)

        # create adv loss (optional)
        image_adv_loss = None
        if config.get('image_adv_loss') is not None:
            loss_type = config['image_adv_loss']['loss_type']
            loss_kwargs = config['image_adv_loss']['loss_kwargs']
            image_adv_loss = ADVERSARIAL_LOSSES[AdversarialLossType[loss_type]](**loss_kwargs)

        # by default we use only rec_loss (without adversarial terms)
        default_loss_weights = {
            'image_rec_loss': 1,
            'image_adv_loss': 0,
        }
        loss_weights = config.get('loss_weights', default_loss_weights)
        self.image_rec_weight = loss_weights['image_rec_loss']
        self.image_adv_weight = loss_weights['image_adv_loss']

        "=========================================== create optimizers ================================================"

        adam_kwargs = config['adam_kwargs']

        self.optimizer = Optimizer(
            enc_params=self.enc.parameters(),
            dec_params=self.dec.parameters(),
            ddis_params=self.ddis.parameters(),
            image_adv_loss=image_adv_loss,
            image_rec_loss=self.image_rec_loss,
            image_adv_weight=self.image_adv_weight,
            image_rec_weight=self.image_rec_weight,
            adam_kwargs=adam_kwargs,
        ).cuda()

        "=========================================== data for logging ================================================="

        self.pg_image_gen.set_stage_resolution(ProgGrowStageType.stab, self.max_image_res, batch_size=self.batch_size)
        self.display_x = next(self.pg_image_gen)[:4].cpu()
        self.pg_image_gen.set_stage_resolution(self.stage, self.resolution, self.batch_size)
        self.val_pg_image_gen.set_stage_resolution(self.stage, self.resolution, self.batch_size)
        self.mix_res_module = MixResolution(self.stage, self.resolution, self.max_image_res)

        "=========================================== initialization ==================================================="

        # list of the models that can "grow" during training
        self.updating_models_during_training = \
            [self.dec, self.ddis, self.enc,
             self.image_rec_loss, self.pg_image_gen, self.val_pg_image_gen, self.mix_res_module]

        if self.finetune_from is not None:
            self.load_state(torch.load(self.finetune_from))
        else:
            self._init_stage()

    def _create_new_stage(self):
        del self.dec, self.ddis, self.enc, self.updating_models_during_training
        if self.stage == ProgGrowStageType.stab:
            self.stage = ProgGrowStageType.trns
            self.resolution *= 2
        else:
            self.stage = ProgGrowStageType.stab
        self.progress = 0
        self._init_stage()

    def _init_stage(self):
        self.batch_size = int(self.batch_sizes[self.resolution])

        "=========================== update data generators and global params ========================================="
        self.pg_image_gen.set_stage_resolution(self.stage, resolution=self.resolution, batch_size=self.batch_size)
        self.val_pg_image_gen.set_stage_resolution(self.stage, resolution=self.resolution, batch_size=self.batch_size)
        self.mix_res_module = MixResolution(self.stage, self.resolution, self.max_image_res)

        "========================================== update networks ==================================================="

        self.dec = self.dec_pg_networks.get_net(self.stage, self.resolution).cuda()
        self.ddis = self.ddis_pg_networks.get_net(self.stage, self.resolution).cuda()
        self.enc = self.enc_pg_networks.get_net(self.stage, self.resolution).cuda()

        self.image_rec_loss.set_stage_resolution(self.stage, self.resolution)

        self.optimizer.set_new_params(
            dec_params=self.dec.parameters(),
            enc_params=self.enc.parameters(),
            ddis_params=self.ddis.parameters(),
            image_rec_loss=self.image_rec_loss)

        "========================================== other updates ==================================================="

        self.updating_models_during_training = \
            [self.dec, self.ddis, self.enc,
             self.image_rec_loss, self.pg_image_gen, self.val_pg_image_gen, self.mix_res_module]

        total_iterations = self.iters[self.stage]
        total_iterations = self.iters_per_res\
            .get(self.resolution, {self.stage.name: total_iterations})\
            .get(self.stage.name, total_iterations)

        self.delta_progress = 1.0 / total_iterations
        self.tqdm_logger = tqdm.tqdm(total=total_iterations)

        if self.verbose:
            print("====================== Encoder =============================")
            print(self.enc)
            print("====================== Decoder =============================")
            print(self.dec)
            print("====================== Discriminator =======================")
            print(self.ddis)

    def train(self):
        print("Starting model training ...")

        while not (
                self.resolution == self.max_image_res and self.stage == ProgGrowStageType.stab and self.progress >= 1.0):

            "========================================== updating params ==============================================="

            self.n_iter += 1
            self.tqdm_logger.update(1)
            self.progress = np.clip(self.progress + self.delta_progress, 0, 1.0)

            for model in self.updating_models_during_training:
                model.set_progress(self.progress)

            if self.stage == ProgGrowStageType.trns:
                progress_resolution = self.resolution * self.progress + (self.resolution / 2) * (
                        1 - self.progress)
            else:
                progress_resolution = self.resolution

            "=========================================== train step ==================================================="

            ddis_losses = {}
            for _ in range(self.n_dis):
                real_x = next(self.pg_image_gen).cuda()
                fake_z = self.enc(real_x)
                fake_x = self.dec(fake_z)

                if self.image_adv_weight:
                    ddis_losses = self.optimizer.compute_ddis_loss(self.ddis, real_x, fake_x, update_parameters=True)

                del real_x, fake_z, fake_x

            real_x = next(self.pg_image_gen).cuda()
            fake_z = self.enc(real_x)
            rec_x = self.dec(fake_z)

            enc_dec_losses = self.optimizer.compute_enc_dec_loss(self.ddis, real_x, rec_x, update_parameters=True)
            del fake_z, real_x, rec_x

            "============================================== logging ==================================================="

            if self.tqdm_logger.n % self.log_iter == 0:
                self.logger.add_scalars('train/ddis', ddis_losses, self.n_iter)
                self.logger.add_scalars('train/enc_dec', enc_dec_losses, self.n_iter)
                self.logger.add_scalar('train/resolution', progress_resolution, self.n_iter)

            if self.tqdm_logger.n % self.image_sample_iter == 0:
                self._save_image_sample()

            "============================================ checkpoint =================================================="

            if self.tqdm_logger.n % self.val_iter == 0:
                val_loss = self._compute_val_loss()
                self.logger.add_scalar('val/total', val_loss, self.n_iter)

                # if this is the last stage, save only if val loss is better
                if self.resolution == self.max_image_res and self.stage == ProgGrowStageType.stab:
                    if val_loss < self.best_val_loss - self.early_stopping_min_delta:
                        self.best_val_loss = val_loss
                        self.best_val_iter = self.n_iter
                        self._do_checkpoint()
                        self._save_ad_model()
                    else:
                        if (self.n_iter - self.best_val_iter) / self.val_iter > self.early_stopping_patience:
                            break

            "========================================== create new stage =============================================="

            if abs(self.progress - 1.0) < 1e-6 and \
                    not(self.resolution == self.max_image_res and self.stage == ProgGrowStageType.stab):
                self._do_checkpoint()
                self._save_ad_model()
                self._create_new_stage()

        # if we did not check last iteration for performance, then check and save
        if self.tqdm_logger.n % self.val_iter != 0:
            val_loss = self._compute_val_loss()
            self.logger.add_scalar('val/total', val_loss, self.n_iter)
            if val_loss < self.best_val_loss:
                self.best_val_iter = self.n_iter
                self._do_checkpoint()
                self._save_ad_model()

        print("Model training is complete.")

    def get_state(self):
        return {
            'config': self.config,
            'stage': self.stage.name,
            'resolution': self.resolution,
            'progress': self.progress,
            'n_iter': self.n_iter,
            'best_val_loss': self.best_val_loss,
            'best_val_iter': self.best_val_iter,
            'display_x': self.display_x.data,
            'optimizer': self.optimizer.state_dict(),
            'dec': self.dec.state_dict(),
            'enc': self.enc.state_dict(),
            'ddis': self.ddis.state_dict(),
            'mix_res_module': self.mix_res_module.state_dict(),
        }

    def load_state(self, state):
        self.stage = ProgGrowStageType[state['stage']]
        self.resolution = state['resolution']
        self.progress = state['progress']
        self.n_iter = state['n_iter']
        self.best_val_loss = state['best_val_loss']
        self.best_val_iter = state['best_val_iter']
        self.display_x.data = state['display_x']

        self._init_stage()

        self.optimizer.load_state_dict(state['optimizer'])
        self.dec.load_state_dict(state['dec'])
        self.enc.load_state_dict(state['enc'])
        self.ddis.load_state_dict(state['ddis'])
        self.mix_res_module.load_state_dict(state['mix_res_module'])

        self.tqdm_logger.update(int(len(self.tqdm_logger) * self.progress))

    def get_state_anomaly_detection_model(self):
        return {
            'niter': self.tqdm_logger.n,
            'config': self.config,
            'stage': self.stage.name,
            'resolution': self.resolution,
            'progress': self.progress,
            'enc': self.enc.state_dict(),
            'dec': self.dec.state_dict(),
        }
    
    @staticmethod
    def load_anomaly_detection_model(state_dict):
        config = state_dict['config']
        stage = ProgGrowStageType[state_dict['stage']]
        resolution = state_dict['resolution']
        progress = state_dict['progress']

        mtype = config['enc']['type']
        kwargs = config['enc']['kwargs']
        enc_pg_networks = ENCODER_NETWORKS[NetworkType[mtype]](
            max_input_res=config['max_image_res'],
            output_res=config['latent_res'],
            input_dim=config['image_dim'],
            output_dim=config['latent_dim'],
            **kwargs
        )

        mtype = config['dec']['type']
        kwargs = config['dec']['kwargs']
        dec_pg_networks = DECODER_NETWORKS[NetworkType[mtype]](
            input_res=config['latent_res'],
            max_output_res=config['max_image_res'],
            input_dim=config['latent_dim'],
            output_dim=config['image_dim'],
            **kwargs
        )

        enc = enc_pg_networks.get_net(stage, resolution)
        dec = dec_pg_networks.get_net(stage, resolution)
        enc.load_state_dict(state_dict['enc'])
        dec.load_state_dict(state_dict['dec'])

        loss_type = config['image_rec_loss']['loss_type']
        loss_kwargs = config['image_rec_loss']['loss_kwargs']
        image_rec_loss = PG_RECONSTRUCTION_LOSSES[ReconstructionLossType[loss_type]](
            config['max_image_res'],
            **loss_kwargs)
        image_rec_loss.set_stage_resolution(stage, resolution)
        image_rec_loss.set_reduction('none')

        mix_res_module = MixResolution(stage, resolution, config['max_image_res'])

        for model in [enc, dec, image_rec_loss, mix_res_module]:
            model.set_progress(progress)

        n_iter = state_dict['niter']
        return enc, dec, image_rec_loss, (stage, resolution, progress, n_iter, mix_res_module)

    def _compute_val_loss(self):
        self.enc.eval()
        self.dec.eval()
        self.ddis.eval()
        torch.set_grad_enabled(False)

        sum_loss = 0
        count = 0

        for real_x in self.val_pg_image_gen:
            real_x = real_x.cuda()
            fake_z = self.enc(real_x)
            rec_x = self.dec(fake_z)

            enc_dec_losses = self.optimizer.compute_enc_dec_loss(self.ddis, real_x, rec_x, update_parameters=False)
            val_loss = sum(enc_dec_losses.values())

            batch_size = real_x.shape[0]
            sum_loss += val_loss * batch_size
            count += batch_size

        val_avg_loss = sum_loss / count

        self.enc.train()
        self.dec.train()
        self.ddis.train()
        torch.set_grad_enabled(True)

        return val_avg_loss

    def _do_checkpoint(self):
        torch.save(self.get_state(), os.path.join(self.checkpoint_root, 'checkpoint.tar'))

    def _save_ad_model(self):
        if self.stage == ProgGrowStageType.stab:
            ad_model_path = 'anomaly_detection_res_{}.tar'.format(self.resolution)
            torch.save(self.get_state_anomaly_detection_model(), os.path.join(self.checkpoint_root, ad_model_path))
            torch.save(self.get_state_anomaly_detection_model(),
                       os.path.join(self.checkpoint_root, 'anomaly_detection.tar'))

    def _save_image_sample(self):
        torch.set_grad_enabled(False)
        self.optimizer.eval()

        real_images = self.mix_res_module(self.display_x)
        images = torch.cat([
            real_images,
            self.dec(self.enc(real_images.cuda())).cpu().detach()
        ], 0)
        name = 'res_{}_stage_{}_niter{}.png'.format(self.resolution, self.stage.name, self.tqdm_logger.n)

        self.logger.save_grid(images, grid_size=self.max_image_res * 4, name=name, nrow=4)
        self.logger.save_grid(images, grid_size=self.max_image_res * 4, name='sample.png', nrow=4)

        torch.set_grad_enabled(True)
        self.optimizer.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config path')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
