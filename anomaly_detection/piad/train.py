import os
import argparse
import tqdm
import torch
import numpy as np
import yaml

from anomaly_detection.utils.loggers import Logger
from anomaly_detection.dpa.pg_networks import ProgGrowStageType, NetworkType
from anomaly_detection.piad.latent_dis import LatentDiscriminator
from anomaly_detection.dpa.pg_decoders import DECODER_NETWORKS
from anomaly_detection.dpa.pg_encoders import ENCODER_NETWORKS
from anomaly_detection.dpa.data_generators import ProgGrowImageGenerator
from anomaly_detection.utils.datasets import DatasetType, DATASETS
from anomaly_detection.utils.transforms import TRANSFORMS
from anomaly_detection.piad.latent_model import LatentModel
from anomaly_detection.dpa.adv_losses import ADVERSARIAL_LOSSES, AdversarialLossType
from anomaly_detection.dpa.rec_losses import RECONSTRUCTION_LOSSES, ReconstructionLossType

from anomaly_detection.piad.optimizer import Optimizer


torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, config):
        self.config = config
        self.verbose = config['verbose']
        self.random_seed = config['random_seed']
        self.finetune_from = config['finetune_from']

        self.checkpoint_root = config['checkpoint_root']
        self.log_root = config['log_root']

        self.image_res = config['image_res']
        self.image_dim = config['image_dim']
        self.latent_res = config['latent_res']
        self.latent_dim = config['latent_dim']

        self.iters = config['iters']

        self.log_iter = config['log_iter']
        self.val_iter = config['val_iter']
        self.update_grad_norm_iter = config['update_grad_norm_iter']
        self.image_sample_iter = config['image_sample_iter']

        self.batch_size = config['batch_size']
        self.n_dis = config['n_dis']

        "=========================================== initialize ======================================================="

        if self.verbose:
            print(yaml.dump(self.config, default_flow_style=False))

        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)

        self.best_val_iter = 0
        self.best_val_loss = 10e6
        self.tqdm_logger = tqdm.tqdm(total=self.iters)

        os.makedirs(self.checkpoint_root, exist_ok=True)
        self.logger = Logger(self.log_root)

        "=========================================== create data model ================================================"

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


        self.pg_image_gen = ProgGrowImageGenerator(dataset, self.image_res,
                                                   batch_size=self.batch_size, inf=True)
        self.pg_image_gen.set_stage_resolution(ProgGrowStageType.stab, self.image_res, batch_size=self.batch_size)

        self.val_pg_image_gen = ProgGrowImageGenerator(val_dataset, self.image_res,
                                                       batch_size=self.batch_size, inf=False)
        self.val_pg_image_gen.set_stage_resolution(ProgGrowStageType.stab, self.image_res, batch_size=self.batch_size)


        "=========================================== create latent model ============================================"

        self.latent_model = LatentModel(self.latent_res, self.latent_dim)

        "============================================= create networks ================================================"

        mtype = config['enc']['type']
        kwargs = config['enc']['kwargs']
        self.enc = ENCODER_NETWORKS[NetworkType[mtype]](
            max_input_res=self.image_res,
            output_res=self.latent_res,
            input_dim=self.image_dim,
            output_dim=self.latent_dim,
            **kwargs
        ).get_net(ProgGrowStageType.stab, self.image_res).cuda()

        mtype = config['dec']['type']
        kwargs = config['dec']['kwargs']
        self.dec= DECODER_NETWORKS[NetworkType[mtype]](
            input_res=self.latent_res,
            max_output_res=self.image_res,
            input_dim=self.latent_dim,
            output_dim=self.image_dim,
            **kwargs
        ).get_net(ProgGrowStageType.stab, self.image_res).cuda()

        mtype = config['ddis']['type']
        kwargs = config['ddis']['kwargs']
        self.ddis = ENCODER_NETWORKS[NetworkType[mtype]](
            max_input_res=self.image_res,
            output_res=1,
            input_dim=self.image_dim,
            output_dim=1,
            **kwargs
        ).get_net(ProgGrowStageType.stab, self.image_res).cuda()

        kwargs = config['edis']['kwargs']
        self.edis = LatentDiscriminator(
            input_dim=self.latent_dim * self.latent_res * self.latent_res,
            **kwargs
        ).cuda()

        if self.verbose:
            print("================================== MODELS ON LARGEST RESOLUTION ==================================")
            print("====================== Encoder =============================")
            print(self.enc)
            print("====================== Decoder  ===========================")
            print(self.dec)
            print("====================== Discriminator of decoder ==========")
            print(self.ddis)
            print("====================== Discriminator of encoder ============")
            print(self.edis)
            print("===================================================================================================")

        "=========================================== create losses ===================================================="

        loss_type = config['image_rec_loss']['loss_type']
        loss_kwargs = config['image_rec_loss']['loss_kwargs']
        image_rec_loss = RECONSTRUCTION_LOSSES[ReconstructionLossType[loss_type]](**loss_kwargs)

        loss_type = config['image_adv_loss']['loss_type']
        loss_kwargs = config['image_adv_loss']['loss_kwargs']
        image_adv_loss = ADVERSARIAL_LOSSES[AdversarialLossType[loss_type]](**loss_kwargs)

        loss_type = config['latent_adv_loss']['loss_type']
        loss_kwargs = config['latent_adv_loss']['loss_kwargs']
        latent_adv_loss = ADVERSARIAL_LOSSES[AdversarialLossType[loss_type]](**loss_kwargs)

        "=========================================== create optimizers ================================================"

        adam_kwargs = config['adam_kwargs']

        self.optimizer = Optimizer(
            enc_params=self.enc.parameters(),
            dec_params=self.dec.parameters(),
            edis_params=self.edis.parameters(),
            ddis_params=self.ddis.parameters(),
            image_adv_loss=image_adv_loss,
            latent_adv_loss=latent_adv_loss,
            image_rec_loss=image_rec_loss,
            latent_dim=self.latent_dim,
            adam_kwargs=adam_kwargs,
        ).cuda()

        "=========================================== data for logging ================================================="

        self.display_x = next(self.pg_image_gen)[:4].cpu()

        "=========================================== initialization ==================================================="

        if self.finetune_from is not None:
            self.load_state(torch.load(self.finetune_from))

    def train(self):
        print("Starting model training ...")

        while self.tqdm_logger.n < self.iters:
            self.tqdm_logger.update(1)

            "=========================================== train step ==================================================="

            ddis_losses, edis_losses = {}, {}
            for _ in range(self.n_dis):
                real_z = self.latent_model.sample(self.batch_size).cuda()
                real_x = next(self.pg_image_gen).cuda()
                fake_x = self.dec(real_z)
                fake_z = self.enc(real_x)

                ddis_losses = self.optimizer.compute_ddis_loss(self.ddis, real_x, fake_x, update_parameters=True)
                edis_losses = self.optimizer.compute_edis_loss(self.edis, real_z, fake_z, update_parameters=True)

                del real_z, real_x, fake_x, fake_z

            real_z = self.latent_model.sample(self.batch_size).cuda()
            real_x = next(self.pg_image_gen).cuda()

            fake_x = self.dec(real_z)
            fake_z = self.enc(real_x)
            rec_x = self.dec(fake_z)

            enc_dec_losses = self.optimizer.compute_enc_dec_loss(
                self.enc, self.dec, self.edis, self.ddis,
                real_x, fake_x, fake_z, rec_x,
                update_parameters=True,
                update_grad_norm=self.tqdm_logger.n % self.update_grad_norm_iter == 0,
            )

            del real_z, fake_z, real_x, fake_x, rec_x

            "============================================== logging ==================================================="

            if self.tqdm_logger.n % self.log_iter == 0:
                self.logger.add_scalars('train/ddis', ddis_losses, self.tqdm_logger.n)
                self.logger.add_scalars('train/edis', edis_losses, self.tqdm_logger.n)
                self.logger.add_scalars('train/enc_dec', enc_dec_losses, self.tqdm_logger.n)

            if self.tqdm_logger.n % self.image_sample_iter == 0:
                self._save_image_sample()

            "============================================ checkpoint =================================================="

            if self.tqdm_logger.n % self.val_iter == 0:
                val_loss = self._compute_val_loss()
                self.logger.add_scalar('val/total', val_loss, self.tqdm_logger.n)
                self._do_checkpoint()
                self._save_ad_model()

        # if we did not check last iteration for performance, then check and save
        if self.tqdm_logger.n % self.val_iter != 0:
            val_loss = self._compute_val_loss()
            self.logger.add_scalar('val/total', val_loss, self.tqdm_logger.n)
            self._do_checkpoint()
            self._save_ad_model()

        print("Model training is complete.")

    def get_state(self):
        return {
            'config': self.config,
            'n_iter': self.tqdm_logger.n,
            'best_val_loss': self.best_val_loss,
            'best_val_iter': self.best_val_iter,
            'display_x': self.display_x.data,
            'optimizer': self.optimizer.state_dict(),
            'dec': self.dec.state_dict(),
            'enc': self.enc.state_dict(),
            'ddis': self.ddis.state_dict(),
            'edis': self.edis.state_dict(),
        }

    def load_state(self, state):
        self.tqdm_logger.update(state['n_iter'])
        self.best_val_loss = state['best_val_loss']
        self.best_val_iter = state['best_val_iter']
        self.display_x.data = state['display_x']
        self.optimizer.load_state_dict(state['optimizer'])
        self.dec.load_state_dict(state['dec'])
        self.enc.load_state_dict(state['enc'])
        self.ddis.load_state_dict(state['ddis'])
        self.edis.load_state_dict(state['edis'])

    def get_state_anomaly_detection_model(self):
        return {
            'niter': self.tqdm_logger.n,
            'config': self.config,
            'enc': self.enc.state_dict(),
            'dec': self.dec.state_dict(),
        }
    
    @staticmethod
    def load_anomaly_detection_model(state_dict):
        config = state_dict['config']
        mtype = config['enc']['type']
        kwargs = config['enc']['kwargs']
        enc = ENCODER_NETWORKS[NetworkType[mtype]](
            max_input_res=config['image_res'],
            output_res=config['latent_res'],
            input_dim=config['image_dim'],
            output_dim=config['latent_dim'],
            **kwargs
        ).get_net(ProgGrowStageType.stab, config['image_res'])

        mtype = config['dec']['type']
        kwargs = config['dec']['kwargs']
        dec = DECODER_NETWORKS[NetworkType[mtype]](
            input_res=config['latent_res'],
            max_output_res=config['image_res'],
            input_dim=config['latent_dim'],
            output_dim=config['image_dim'],
            **kwargs
        ).get_net(ProgGrowStageType.stab, config['image_res'])

        enc.load_state_dict(state_dict['enc'])
        dec.load_state_dict(state_dict['dec'])

        loss_type = config['image_rec_loss']['loss_type']
        loss_kwargs = config['image_rec_loss']['loss_kwargs']
        image_rec_loss = RECONSTRUCTION_LOSSES[ReconstructionLossType[loss_type]](**loss_kwargs)
        image_rec_loss.set_reduction('none')

        niter = state_dict['niter']
        return enc, dec, image_rec_loss, niter

    def _compute_val_loss(self):
        self.enc.eval()
        self.dec.eval()
        self.ddis.eval()
        self.edis.eval()
        torch.set_grad_enabled(False)

        sum_loss = 0
        count = 0

        for real_x in self.val_pg_image_gen:
            real_x = real_x.cuda()
            batch_size = real_x.shape[0]
            real_z = self.latent_model.sample(batch_size=batch_size).cuda()
            fake_x = self.dec(real_z)
            fake_z = self.enc(real_x)
            rec_x = self.dec(fake_z)

            val_loss = self.optimizer.compute_enc_dec_loss(
                self.enc, self.dec, self.edis, self.ddis,
                real_x, fake_x, fake_z, rec_x,
                update_parameters=False,
                update_grad_norm=False
            )['image_rec_loss']

            sum_loss += val_loss * batch_size
            count += batch_size

        val_avg_loss = sum_loss / count

        self.enc.train()
        self.dec.train()
        self.ddis.train()
        self.edis.train()
        torch.set_grad_enabled(True)

        return val_avg_loss

    def _do_checkpoint(self):
        torch.save(self.get_state(), os.path.join(self.checkpoint_root, 'checkpoint.tar'))

    def _save_ad_model(self):
        ad_model_path = 'anomaly_detection_niter_{}.tar'.format(self.tqdm_logger.n)
        torch.save(self.get_state_anomaly_detection_model(), os.path.join(self.checkpoint_root, ad_model_path))

        torch.save(self.get_state_anomaly_detection_model(), os.path.join(self.checkpoint_root, 'anomaly_detection.tar'))

    def _save_image_sample(self):
        torch.set_grad_enabled(False)
        self.optimizer.eval()

        images = torch.cat([
            self.display_x,
            self.dec(self.enc(self.display_x.cuda())).cpu().detach()
        ], 0)

        name = 'res_{}_niter_{}.png'.format(self.image_res, self.tqdm_logger.n)
        self.logger.save_grid(images, grid_size=self.image_res * 4, name=name, nrow=4)
        self.logger.save_grid(images, grid_size=self.image_res * 4, name='sample.png', nrow=4)

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
