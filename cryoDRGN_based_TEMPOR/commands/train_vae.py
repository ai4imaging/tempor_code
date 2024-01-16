'''
Train a VAE for heterogeneous reconstruction with known pose
'''
import numpy as np
import sys, os
import argparse
import pickle
from datetime import datetime as dt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
try:
    import apex.amp as amp
except: 
    pass

import cryodrgn
from cryodrgn import mrc
from cryodrgn import utils
from cryodrgn import fft
from cryodrgn import lie_tools
from cryodrgn import dataset
from cryodrgn import ctf

from cryodrgn.pose import PoseTracker
from cryodrgn.models import HetOnlyVAE, CryodrgnVAE, unparallelize
from cryodrgn.lattice import Lattice
from cryodrgn.beta_schedule import get_beta_schedule, LinearSchedule

log = utils.log
vlog = utils.vlog

def add_args(parser):
    parser.add_argument('particles', type=os.path.abspath, help='Input particles (.mrcs, .star, .cs, or .txt)')
    parser.add_argument('-o', '--outdir', type=os.path.abspath, required=True, help='Output directory to save model')
    parser.add_argument('--zdim', type=int, required=True, help='Dimension of latent variable')
    parser.add_argument('--poses', type=os.path.abspath, required=True, help='Image poses (.pkl)')
    parser.add_argument('--ctf', metavar='pkl', type=os.path.abspath, help='CTF parameters (.pkl)')
    parser.add_argument('--load', metavar='WEIGHTS.PKL', help='Initialize training from a checkpoint')
    parser.add_argument('--checkpoint', type=int, default=1, help='Checkpointing interval in N_EPOCHS (default: %(default)s)')
    parser.add_argument('--log-interval', type=int, default=1000, help='Logging interval in N_IMGS (default: %(default)s)')
    parser.add_argument('-v','--verbose',action='store_true',help='Increaes verbosity')
    parser.add_argument('--seed', type=int, default=np.random.randint(0,100000), help='Random seed')

    group = parser.add_argument_group('Dataset loading')
    group.add_argument('--ind', type=os.path.abspath, metavar='PKL', help='Filter particle stack by these indices')
    group.add_argument('--uninvert-data', dest='invert_data', action='store_false', help='Do not invert data sign')
    group.add_argument('--no-window', dest='window', action='store_false', help='Turn off real space windowing of dataset')
    group.add_argument('--window-r', type=float, default=.85,  help='Windowing radius (default: %(default)s)')
    group.add_argument('--datadir', type=os.path.abspath, help='Path prefix to particle stack if loading relative paths from a .star or .cs file')
    group.add_argument('--lazy', action='store_true', help='Lazy loading if full dataset is too large to fit in memory (Should copy dataset to SSD)')
    group.add_argument('--preprocessed', action='store_true', help='Skip preprocessing steps if input data is from cryodrgn preprocess_mrcs')
    group.add_argument('--max-threads', type=int, default=16, help='Maximum number of CPU cores for FFT parallelization (default: %(default)s)')

    group = parser.add_argument_group('Tilt series')
    group.add_argument('--tilt', help='Particles (.mrcs)')
    group.add_argument('--tilt-deg', type=float, default=45, help='X-axis tilt offset in degrees (default: %(default)s)')

    group = parser.add_argument_group('Training parameters')
    group.add_argument('-n', '--num-epochs', type=int, default=20, help='Number of training epochs (default: %(default)s)')
    group.add_argument('-b','--batch-size', type=int, default=8, help='Minibatch size (default: %(default)s)')
    group.add_argument('-s','--seq-len', type=int, default=10, help='Sequence length (default: %(default)s)')
    group.add_argument('--amplitude', type=float, default=10, help='Amplitude rate (default: %(default)s)')
    group.add_argument('--sigma', type=float, default=100, help='Sigma of gaussian filter (default: %(default)s)')
    group.add_argument('--pretrain-model', default=None, help='pretrain model (default: %(default)s)')
    group.add_argument('--pretrain-epoch', type=int, default=50, help='pretrain epoch (default: %(default)s)')
    group.add_argument('--wd', type=float, default=0, help='Weight decay in Adam optimizer (default: %(default)s)')
    group.add_argument('--lr', type=float, default=1e-4, help='Learning rate in Adam optimizer (default: %(default)s)')
    group.add_argument('--beta', default=None, help='Choice of beta schedule or a constant for KLD weight (default: 1/zdim)')
    group.add_argument('--beta-control', type=float, help='KL-Controlled VAE gamma. Beta is KL target. (default: %(default)s)')
    group.add_argument('--norm', type=float, nargs=2, default=None, help='Data normalization as shift, 1/scale (default: 0, std of dataset)')
    group.add_argument('--no-amp', action='store_false', dest='amp', help='Do not use mixed-precision training')
    group.add_argument('--multigpu', action='store_true', help='Parallelize training across all detected GPUs')

    group = parser.add_argument_group('Pose SGD')
    group.add_argument('--do-pose-sgd', action='store_true', help='Refine poses with gradient descent')
    group.add_argument('--pretrain', type=int, default=1, help='Number of epochs with fixed poses before pose SGD (default: %(default)s)')
    group.add_argument('--emb-type', choices=('s2s2','quat'), default='quat', help='SO(3) embedding type for pose SGD (default: %(default)s)')
    group.add_argument('--pose-lr', type=float, default=3e-4, help='Learning rate for pose optimizer (default: %(default)s)')

    group = parser.add_argument_group('Encoder Network')
    group.add_argument('--enc-layers', dest='qlayers', type=int, default=3, help='Number of hidden layers (default: %(default)s)')
    group.add_argument('--enc-dim', dest='qdim', type=int, default=1024, help='Number of nodes in hidden layers (default: %(default)s)')
    group.add_argument('--encode-mode', default='resid', choices=('conv','resid','mlp','tilt'), help='Type of encoder network (default: %(default)s)')
    group.add_argument('--enc-mask', type=int, help='Circular mask of image for encoder (default: D/2; -1 for no mask)')
    group.add_argument('--use-real', action='store_true', help='Use real space image for encoder (for convolutional encoder)')

    group = parser.add_argument_group('Decoder Network')
    group.add_argument('--dec-layers', dest='players', type=int, default=3, help='Number of hidden layers (default: %(default)s)')
    group.add_argument('--dec-dim', dest='pdim', type=int, default=1024, help='Number of nodes in hidden layers (default: %(default)s)')
    group.add_argument('--pe-type', choices=('geom_ft','geom_full','geom_lowf','geom_nohighf','linear_lowf', 'gaussian', 'none'), default='gaussian', help='Type of positional encoding (default: %(default)s)')
    group.add_argument('--feat-sigma', type=float, default=0.5, help="Scale for random Gaussian features")
    group.add_argument('--pe-dim', type=int, help='Num features in positional encoding (default: image D)')
    group.add_argument('--domain', choices=('hartley','fourier'), default='fourier', help='Decoder representation domain (default: %(default)s)')
    group.add_argument('--activation', choices=('relu','leaky_relu'), default='relu', help='Activation (default: %(default)s)')
    return parser

def train_batch(model, lattice, batch_size, y, yt, rot, trans, optim, beta, beta_control=None, tilt=None, ctf_params=None, yr=None, use_amp=False, scaler=None, device=None, zdim=8, seq_len=10 ,ind=None):
    optim.zero_grad()
    model.train()
    if trans is not None:
        y, yt = preprocess_input(y, yt, lattice, trans)
    # Cast operations to mixed precision if using torch.cuda.amp.GradScaler()
    if scaler is not None:
        with torch.cuda.amp.autocast():
            z_mu, z_logvar, z_mean_p, z_logvar_p, z, y_recon, y_recon_tilt, mask, z_0, z_mu_0, z_logvar_0 = run_batch(model, lattice, batch_size, y, yt, rot, tilt, ctf_params, yr, device, zdim, seq_len=seq_len)
            loss, gen_loss, kld = loss_function(z_mu_0, z_logvar_0, z_mu, z_logvar, z_mean_p, z_logvar_p, y, yt, y_recon, mask, beta, y_recon_tilt, beta_control, zdim, ind, device)
    else:
        z_mu, z_logvar, z_mean_p, z_logvar_p, z, y_recon, y_recon_tilt, mask, z_0, z_mu_0, z_logvar_0 = run_batch(model, lattice, batch_size, y, yt, rot, tilt, ctf_params, yr, device, zdim, seq_len=seq_len)
        loss, gen_loss, kld = loss_function(z_mu_0, z_logvar_0, z_mu, z_logvar, z_mean_p, z_logvar_p, y, yt, y_recon, mask, beta, y_recon_tilt, beta_control, zdim, ind, device)

    if use_amp:
        if scaler is not None: # torch mixed precision
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else: # apex.amp mixed precision
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
            optim.step()
    else:
        loss.backward()
        optim.step()
    return loss.item(), gen_loss.item(), kld.item()

def preprocess_input(y, yt, lattice, trans):
    # center the image
    B = y.size(0)
    D = lattice.D
    y = lattice.translate_ht(y.view(B,-1), trans.unsqueeze(1)).view(B,D,D)
    if yt is not None: yt = lattice.translate_ht(yt.view(B,-1), trans.unsqueeze(1)).view(B,D,D)
    return y, yt

def _unparallelize(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model

def run_batch(model, lattice, batch_size, y, yt, rot, tilt=None, ctf_params=None, yr=None, device=None, zdim=8, seq_len=10):
    use_tilt = yt is not None
    use_ctf = ctf_params is not None
    B = y.size(0)
    D = lattice.D
    if use_ctf:
        freqs = lattice.freqs2d.unsqueeze(0).expand(B,*lattice.freqs2d.shape)/ctf_params[:,0].view(B,1,1)
        c = ctf.compute_ctf(freqs, *torch.split(ctf_params[:,1:], 1, 1)).view(B,D,D)
    
    # encode
    if yr is not None:
        input_ = (yr,)
    else:
        input_ = (y,yt) if yt is not None else (y,)
        if use_ctf: input_ = (x*c.sign() for x in input_) # phase flip by the ctf
    
    for data in input_: # ctf
        input_ctf = data # ctf

    seq_len = seq_len
    batch_size = batch_size
    input_ = input_ctf.view(batch_size, seq_len, -1)
    z, z_mu, z_logvar = unparallelize(model).inference_DKF(input_)
    z_mean_p, z_logvar_p = unparallelize(model).transition(z[:, 0:-1, :])

    # decode 

    z_0 = z[:,0,:]
    z_0 = z_0.reshape(-1, zdim)
    z_mu_0 = z_mu[:,1:,:]
    z_mu_0 = z_mu_0.reshape(-1, zdim)
    z_logvar_0 = z_logvar[:,1:,:]
    z_logvar_0 = z_logvar_0.reshape(-1, zdim)

    z = z[:,1:,:]
    z = z.reshape(-1, zdim)
    z_mu = z_mu[:,1:,:]
    z_mu = z_mu.reshape(-1, zdim)
    z_logvar = z_logvar[:,1:,:]
    z_logvar = z_logvar.reshape(-1, zdim)
    
    z_mean_p = z_mean_p.reshape(-1, zdim)
    z_logvar_p = z_logvar_p.reshape(-1, zdim)
    mask = lattice.get_circular_mask(D//2) # restrict to circular mask
    y_recon = model(lattice.coords[mask]/lattice.extent/2 @ rot, z).view(B,-1)
    if use_ctf: y_recon *= c.view(B,-1)[:,mask]

    # decode the tilt series
    if use_tilt:
        y_recon_tilt = model(lattice.coords[mask]/lattice.extent/2 @ tilt @ rot, z)
        if use_ctf: y_recon_tilt *= c.view(B,-1)[:,mask]
    else:
        y_recon_tilt = None

    return z_mu, z_logvar, z_mean_p, z_logvar_p, z, y_recon, y_recon_tilt, mask, z_0, z_mu_0, z_logvar_0

def loss_function(z_mu_0, z_logvar_0, z_mu, z_logvar, z_mean_p, z_logvar_p, y, yt, y_recon, mask, beta, y_recon_tilt=None, beta_control=None, zdim=8, ind=None, device=None):
    
    # pretrain error
    # ind = ind.detach().cpu().numpy()
    # path = '/home/yez/cas9_n1900_d64/REF_01_256'
    # iter_ = 30 # FIXME iter _YEZ
    # res = open(path + '/z.' + str(iter_) + '.pkl','rb')
    # z_ref = pickle.load(res)
    # z_ref_batch = torch.from_numpy(z_ref[ind]).to(device)
    # pre_loss = F.mse_loss(z_ref_batch, z_mu)
    # pre_alpha = 1
    
    # reconstruction error
    use_tilt = yt is not None
    B = y.size(0)
    gen_loss = F.mse_loss(y_recon, y.view(B,-1)[:, mask])
    if use_tilt:
        gen_loss = .5*gen_loss + .5*F.mse_loss(y_recon_tilt, yt.view(B,-1)[:,mask])
    # latent loss
    kld_ = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1), dim=0)
    kld1 = torch.mean(-0.5 * torch.sum(1 + z_logvar_0 - z_mu_0.pow(2) - z_logvar_0.exp(), dim=1), dim=0)
    kld2 = -0.5 * torch.mean(z_logvar - z_logvar_p - torch.div((z_logvar.exp() + (z_mu - z_mean_p).pow(2)), z_logvar_p.exp()))
    kld = kld_
    # total loss
    if beta_control is None:
        loss = gen_loss + beta*kld/mask.sum().float() # + pre_alpha * pre_loss
    else:
        loss = gen_loss + args.beta_control*(beta-kld)**2/mask.sum().float() # + pre_alpha * pre_loss
    return loss, gen_loss, kld

def eval_z(model, particles, lattice, data, batch_size, device, trans=None, use_tilt=False, ctf_params=None, use_real=False, seq_len=10, zdim=1, sigma=100):
    assert not model.training
    z_mu_all = []
    z_logvar_all = []
    data = dataset.MRCData(particles, eval_z=True, seq_len=seq_len)
    
    batch_size = 1 
    
    data_generator = DataLoader(data, batch_size=batch_size, shuffle=False)
    for minibatch in data_generator:
        
        seq_len = seq_len
        dataset_len = 1900 # FIXME _YEZ

        inds = torch.zeros((seq_len, batch_size))
        pics = torch.zeros((batch_size, seq_len, 65, 65))

        for i in range(seq_len):
            inds[i] = minibatch[1][i]
        for i in range(batch_size):
            pics[i] = minibatch[0][i]

        inds = inds.permute(1,0)
        indices = inds.reshape(seq_len * batch_size).long()
        picts = pics.reshape(seq_len * batch_size, 65, 65)

        # FIXME Gaussian Blur _YEZ
        for i in range(seq_len * batch_size):
            if sigma != 0:
                picts[i] = torch.from_numpy(gaussian_filter(picts[i], sigma=sigma, order=0))
        
        ind =indices.to(device)
        y = picts.to(device)
        yt = None
        
        B = len(ind)
        D = lattice.D
        if ctf_params is not None:
            freqs = lattice.freqs2d.unsqueeze(0).expand(B,*lattice.freqs2d.shape)/ctf_params[ind,0].view(B,1,1)
            c = ctf.compute_ctf(freqs, *torch.split(ctf_params[ind,1:], 1, 1)).view(B,D,D)
        if trans is not None:
            y = lattice.translate_ht(y.view(B,-1), trans[ind].unsqueeze(1)).view(B,D,D)
            if yt is not None: yt = lattice.translate_ht(yt.view(B,-1), trans[ind].unsqueeze(1)).view(B,D,D)
        if use_real:
            input_ = (torch.from_numpy(data.particles_real[ind]).to(device),)
        else:
            input_ = (y,yt) if yt is not None else (y,)
        if ctf_params is not None: 
            assert not use_real, "Not implemented"
            
        seq_len = seq_len
        batch_size = batch_size

        for data in input_: # ctf
            input_ctf = data # ctf

        input_ = input_ctf.view(batch_size, seq_len, -1)
        _, z_mu, z_logvar = unparallelize(model).inference_DKF(input_)
        z_mu = z_mu[:,1:,:]
        z_logvar = z_logvar[:,1:,:]
        z_mu = z_mu.reshape(seq_len * batch_size, -1)
        z_logvar = z_logvar.reshape(seq_len * batch_size, -1)
        z_mu_all.append(z_mu.detach().cpu().numpy())
        z_logvar_all.append(z_logvar.detach().cpu().numpy())

    
    # z_mu_all = np.vstack(z_mu_all)
    # z_logvar_all = np.vstack(z_logvar_all)

    # z_mu = z_mu_all.copy()
    # z_logvar = z_logvar_all.copy()
    # for i in range(seq_len):
    #     z_logvar_all[i*(z_mu_all.shape[0]//seq_len):(i+1)*(z_mu_all.shape[0]//seq_len)] = z_logvar[i:z_mu_all.shape[0]:seq_len]
    #     z_mu_all[i*(z_mu_all.shape[0]//seq_len):(i+1)*(z_mu_all.shape[0]//seq_len)] = z_mu[i:z_mu_all.shape[0]:seq_len]

    # average
    '''
    z_mu_list = np.zeros((dataset_len, zdim))
    z_logvar_list = np.zeros((dataset_len, zdim))
    z_mu_ = np.zeros((dataset_len, zdim))
    z_logvar_ = np.zeros((dataset_len, zdim))

    for i in range(dataset_len + 1 - seq_len):
    
        mu_list = z_mu_all[i].reshape(seq_len, zdim)
        logvar_list = z_logvar_all[i].reshape(seq_len, zdim)
        z_mu_[i:i+seq_len] += mu_list
        z_logvar_[i:i+seq_len] += logvar_list
    
    for i in range(dataset_len):
        
        if i < seq_len:
            z_mu_list[i] = z_mu_[i] / (i+1)
            z_logvar_list[i] = z_logvar_[i] / (i+1)
        elif i > dataset_len - seq_len:
            z_mu_list[i] = z_mu_[i] / (dataset_len - i)
            z_logvar_list[i] = z_logvar_[i] / (dataset_len - i)
        else:
            z_mu_list[i] = z_mu_[i] / seq_len
            z_logvar_list[i] = z_logvar_[i] / seq_len

    z_mu_all = z_mu_list.reshape(dataset_len, zdim)
    z_logvar_all = z_logvar_list.reshape(dataset_len, zdim)
    '''

    # average step
    z_mu_list = np.zeros((dataset_len, zdim))
    z_logvar_list = np.zeros((dataset_len, zdim))
    z_mu_ = np.zeros((dataset_len, zdim))
    z_logvar_ = np.zeros((dataset_len, zdim))

    for i in range(dataset_len // seq_len):

        mu_list = z_mu_all[i].reshape(seq_len, zdim)
        logvar_list = z_logvar_all[i].reshape(seq_len, zdim)

        for j in range(seq_len):
            dist_pos = j * (dataset_len // seq_len) + i
            if dist_pos >= dataset_len:
                dist_pos = dist_pos % dataset_len
            z_mu_[dist_pos] += mu_list[j]
            z_logvar_[dist_pos] += logvar_list[j]

    for i in range(dataset_len):
        z_mu_list[i] = z_mu_[i] 
        z_logvar_list[i] = z_logvar_[i] 

    z_mu_all = z_mu_list.reshape(dataset_len, zdim)
    z_logvar_all = z_logvar_list.reshape(dataset_len, zdim)

    '''    
    # average cycle
    z_mu_list = np.zeros((dataset_len, zdim))
    z_logvar_list = np.zeros((dataset_len, zdim))
    z_mu_ = np.zeros((dataset_len, zdim))
    z_logvar_ = np.zeros((dataset_len, zdim))
    
    for i in range(dataset_len):

        mu_list = z_mu_all[i].reshape(seq_len, zdim)
        logvar_list = z_logvar_all[i].reshape(seq_len, zdim)

        for j in range(seq_len):
            dist_pos = j * (dataset_len // seq_len) + i
            if dist_pos >= dataset_len:
                dist_pos = dist_pos % dataset_len
            z_mu_[dist_pos] += mu_list[j]
            z_logvar_[dist_pos] += logvar_list[j]

    for i in range(dataset_len):
        z_mu_list[i] = z_mu_[i] / seq_len
        z_logvar_list[i] = z_logvar_[i] / seq_len

    z_mu_all = z_mu_list.reshape(dataset_len, zdim)
    z_logvar_all = z_logvar_list.reshape(dataset_len, zdim)
    '''

    # normal
    '''
    z_mu_all = np.vstack(z_mu_all)
    z_logvar_all = np.vstack(z_logvar_all)
    '''

    return z_mu_all, z_logvar_all
    
def save_checkpoint(model, optim, epoch, z_mu, z_logvar, out_weights, out_z):
    '''Save model weights, latent encoding z, and decoder volumes'''
    # save model weights
    torch.save({
        'epoch':epoch,
        'model_state_dict':_unparallelize(model).state_dict(),
        'optimizer_state_dict':optim.state_dict(),
        }, out_weights)
    # save z
    with open(out_z,'wb') as f:
        pickle.dump(z_mu, f)
        pickle.dump(z_logvar, f)

def save_config(args, dataset, lattice, model, out_config):
    dataset_args = dict(particles=args.particles,
                        norm=dataset.norm,
                        invert_data=args.invert_data,
                        ind=args.ind,
                        keepreal=args.use_real,
                        window=args.window,
                        window_r=args.window_r,
                        datadir=args.datadir,
                        ctf=args.ctf,
                        poses=args.poses,
                        do_pose_sgd=args.do_pose_sgd)
    if args.tilt is not None:
        dataset_args['particles_tilt'] = args.tilt
    lattice_args = dict(D=lattice.D,
                        extent=lattice.extent,
                        ignore_DC=lattice.ignore_DC)
    model_args = dict(qlayers=args.qlayers,
                      qdim=args.qdim,
                      players=args.players,
                      pdim=args.pdim,
                      zdim=args.zdim,
                      encode_mode=args.encode_mode,
                      enc_mask=args.enc_mask,
                      pe_type=args.pe_type,
                      feat_sigma=args.feat_sigma,
                      pe_dim=args.pe_dim,
                      domain=args.domain,
                      activation=args.activation)
    config = dict(dataset_args=dataset_args,
                  lattice_args=lattice_args,
                  model_args=model_args)
    config['seed'] = args.seed
    with open(out_config,'wb') as f:
        pickle.dump(config, f)
        meta = dict(time=dt.now(),
                    cmd=sys.argv,
                    version=cryodrgn.__version__)
        pickle.dump(meta, f)

def get_latest(args, flog):
    # assumes args.num_epochs > latest checkpoint
    flog('Detecting latest checkpoint...') 
    weights = [f'{args.outdir}/weights.{i}.pkl' for i in range(args.num_epochs)]
    weights = [f for f in weights if os.path.exists(f)]
    args.load = weights[-1]
    flog(f'Loading {args.load}')
    if args.do_pose_sgd:
        i = args.load.split('.')[-2]
        args.poses = f'{args.outdir}/pose.{i}.pkl'
        assert os.path.exists(args.poses)
        flog(f'Loading {args.poses}')
    return args

def main(args):
    t1 = dt.now()
    if args.outdir is not None and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    LOG = f'{args.outdir}/run.log'
    def flog(msg): # HACK: switch to logging module
        return utils.flog(msg, LOG)
    if args.load == 'latest':
        args = get_latest(args, flog)
    flog(' '.join(sys.argv))
    flog(args)

    # set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set the device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    flog('Use cuda {}'.format(use_cuda))
    if not use_cuda:
        log('WARNING: No GPUs detected')

    # set beta schedule
    if args.beta is None:
        args.beta = 1./args.zdim
    try:
        args.beta = float(args.beta)
    except ValueError: 
        assert args.beta_control, "Need to set beta control weight for schedule {}".format(args.beta)
    beta_schedule = get_beta_schedule(args.beta)

    # load index filter
    if args.ind is not None: 
        flog('Filtering image dataset with {}'.format(args.ind))
        ind = pickle.load(open(args.ind,'rb'))
    else: ind = None

    # load dataset
    flog(f'Loading dataset from {args.particles}')
    if args.tilt is None:
        tilt = None
        args.use_real = args.encode_mode == 'conv'

        if args.lazy:
            data = dataset.LazyMRCData(args.particles, norm=args.norm, invert_data=args.invert_data, ind=ind, keepreal=args.use_real, window=args.window, datadir=args.datadir, window_r=args.window_r, flog=flog)
        elif args.preprocessed:
            flog(f'Using preprocessed inputs. Ignoring any --window/--invert-data options')
            data = dataset.PreprocessedMRCData(args.particles, norm=args.norm, ind=ind, flog=flog)
        else:
            data = dataset.MRCData(args.particles, norm=args.norm, invert_data=args.invert_data, ind=ind, keepreal=args.use_real, window=args.window, datadir=args.datadir, max_threads=args.max_threads, window_r=args.window_r, flog=flog, seq_len=args.seq_len)

    # Tilt series data -- lots of unsupported features
    else:
        assert args.encode_mode == 'tilt'
        if args.lazy: raise NotImplementedError
        if args.preprocessed: raise NotImplementedError
        data = dataset.TiltMRCData(args.particles, args.tilt, norm=args.norm, invert_data=args.invert_data, ind=ind, window=args.window, keepreal=args.use_real, datadir=args.datadir, window_r=args.window_r, flog=flog)
        tilt = torch.tensor(utils.xrot(args.tilt_deg).astype(np.float32), device=device)
    Nimg = data.N
    D = data.D

    if args.encode_mode == 'conv':
        assert D-1 == 64, "Image size must be 64x64 for convolutional encoder"

    # load poses
    if args.do_pose_sgd: assert args.domain == 'hartley', "Need to use --domain hartley if doing pose SGD"
    do_pose_sgd = args.do_pose_sgd
    posetracker = PoseTracker.load(args.poses, Nimg, D, 's2s2' if do_pose_sgd else None, ind, device=device)
    pose_optimizer = torch.optim.SparseAdam(list(posetracker.parameters()), lr=args.pose_lr) if do_pose_sgd else None

    # load ctf
    if args.ctf is not None:
        if args.use_real:
            raise NotImplementedError("Not implemented with real-space encoder. Use phase-flipped images instead")
        flog('Loading ctf params from {}'.format(args.ctf))
        ctf_params = ctf.load_ctf_for_training(D-1, args.ctf)
        if args.ind is not None: ctf_params = ctf_params[ind]
        assert ctf_params.shape == (Nimg, 8)
        ctf_params = torch.tensor(ctf_params, device=device)
    else: ctf_params = None

    # instantiate model
    lattice = Lattice(D, extent=0.5, device=device)
    if args.enc_mask is None:
        args.enc_mask = D//2
    if args.enc_mask > 0:
        assert args.enc_mask <= D//2
        enc_mask = lattice.get_circular_mask(args.enc_mask)
        in_dim = enc_mask.sum()
    elif args.enc_mask == -1:
        enc_mask = None
        in_dim = lattice.D**2 if not args.use_real else (lattice.D-1)**2
    else: 
        raise RuntimeError("Invalid argument for encoder mask radius {}".format(args.enc_mask))
    activation={"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[args.activation]
    
    model = HetOnlyVAE(lattice, args.qlayers, args.qdim, args.players, args.pdim,
                in_dim, args.zdim, encode_mode=args.encode_mode, enc_mask=enc_mask,
                enc_type=args.pe_type, enc_dim=args.pe_dim, domain=args.domain,
                activation=activation, feat_sigma=args.feat_sigma, seq_len=args.seq_len)

    pretrain_model = CryodrgnVAE(lattice, args.qlayers, args.qdim, args.players, args.pdim,
                in_dim, args.zdim, encode_mode=args.encode_mode, enc_mask=enc_mask,
                enc_type=args.pe_type, enc_dim=args.pe_dim, domain=args.domain,
                activation=activation, feat_sigma=args.feat_sigma)

    # LEM_pretrain_model = HetOnlyVAE(lattice, args.qlayers, args.qdim, args.players, args.pdim,
    #             in_dim, args.zdim, encode_mode=args.encode_mode, enc_mask=enc_mask,
    #             enc_type=args.pe_type, enc_dim=args.pe_dim, domain=args.domain,
    #             activation=activation, feat_sigma=args.feat_sigma, seq_len=args.seq_len)

    model.to(device)
    flog(model)
    flog('{} parameters in model'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    flog('{} parameters in encoder'.format(sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)))
    flog('{} parameters in deoder'.format(sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)))

    # save configuration
    out_config = '{}/config.pkl'.format(args.outdir)
    save_config(args, data, lattice, model, out_config)
    
    # optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # FIXME use different learning rate
    filtered_params1 = list(map(id, model.mlp_z_prop.parameters())) 
    filtered_params2 = list(map(id, model.mlp_z_prop_gate.parameters())) 
    filtered_params3 = list(map(id, model.mlp_z_prop_mean.parameters())) 
    filtered_params4 = list(map(id, model.mlp_z_prop_std.parameters())) 
    base_params = filter(lambda p: id(p) not in filtered_params1 + filtered_params2 + filtered_params3 + filtered_params4, model.parameters())
    optim = torch.optim.Adam([
        {'params': base_params}, 
        {'params': model.mlp_z_prop.parameters(), 'lr': args.lr * args.amplitude}, 
        {'params': model.mlp_z_prop_gate.parameters(), 'lr': args.lr * args.amplitude}, 
        {'params': model.mlp_z_prop_mean.parameters(), 'lr': args.lr * args.amplitude}, 
        {'params': model.mlp_z_prop_std.parameters(), 'lr': args.lr * args.amplitude}, 
        ], lr=args.lr, weight_decay=args.wd)

    # Mixed precision training
    scaler = None
    if args.amp:
        assert args.batch_size % 8 == 0, "Batch size must be divisible by 8 for AMP training"
        assert (D-1) % 8 == 0, "Image size must be divisible by 8 for AMP training"
        assert args.pdim % 8 == 0, "Decoder hidden layer dimension must be divisible by 8 for AMP training"
        assert args.qdim % 8 == 0, "Encoder hidden layer dimension must be divisible by 8 for AMP training"
        # Also check zdim, enc_mask dim? Add them as warnings for now.
        if args.zdim % 8 != 0:
            log('Warning: z dimension is not a multiple of 8 -- AMP training speedup is not optimized')
        if in_dim % 8 != 0:
            log('Warning: Masked input image dimension is not a mutiple of 8 -- AMP training speedup is not optimized')
        try: # Mixed precision with apex.amp
            model, optim = amp.initialize(model, optim, opt_level='O1')
        except: # Mixed precision with pytorch (v1.6+)
            scaler = torch.cuda.amp.GradScaler()

    # FIXME pretrain _YEZ
    
    path = '/home/yez/cas9_final/' + args.pretrain_model + 'weights.' + str(args.pretrain_epoch) + '.pkl' # FIXME iter _YEZ
    pretrained_dict = torch.load(path)
    pretrain_model.load_state_dict(pretrained_dict['model_state_dict'], strict=True)
    model.decoder.load_state_dict(pretrain_model.decoder.state_dict())
    model.mlp_h_to_z0.load_state_dict(pretrain_model.encoder.state_dict())
    model.mlp_x_to_h.load_state_dict(pretrain_model.encoder.state_dict())

    # path = '/home/yez/cas9_n1900_d64/LEM_01_256_pretrain_sequence/weights.10.pkl' # FIXME iter _YEZ
    # pretrained_dict = torch.load(path)
    # LEM_pretrain_model.load_state_dict(pretrained_dict['model_state_dict'], strict=True)
    # model.load_state_dict(LEM_pretrain_model.state_dict())
    
    # restart from checkpoint
    if args.load:
        flog('Loading checkpoint from {}'.format(args.load))
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
        model.train()
    else:
        start_epoch = 0

    # parallelize
    if args.multigpu and torch.cuda.device_count() > 1:
        log(f'Using {torch.cuda.device_count()} GPUs!')
        args.batch_size *= torch.cuda.device_count()
        log(f'Increasing batch size to {args.batch_size}')
        model = nn.DataParallel(model)
    elif args.multigpu:
        log(f'WARNING: --multigpu selected, but {torch.cuda.device_count()} GPUs detected')

    # training loop
    data_generator = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    num_epochs = args.num_epochs
    
    # flag500 = False
    # flag1000 = False
    
    for epoch in range(start_epoch, num_epochs):

        # if epoch > 500 and flag500 == False: # FIXME _YEZ
        #     flag500 == True
        #     optim = torch.optim.Adam(model.parameters(), lr=args.lr/10, weight_decay=args.wd)

        # if epoch > 1000 and flag1000 == False: # FIXME _YEZ
        #     flag1000 == True
        #     optim = torch.optim.Adam(model.parameters(), lr=args.lr/100, weight_decay=args.wd)

        t2 = dt.now()
        gen_loss_accum = 0
        loss_accum = 0
        kld_accum = 0
        eq_loss_accum = 0
        batch_it = 0 
        for minibatch in data_generator:

            seq_len = args.seq_len
            batch_size = minibatch[0].shape[0]

            inds = torch.zeros((seq_len, batch_size))
            pics = torch.zeros((batch_size, seq_len, 65, 65))

            for i in range(seq_len):
                inds[i] = minibatch[1][i]
            for i in range(batch_size):
                pics[i] = minibatch[0][i]

            inds = inds.permute(1,0)
            indices = inds.reshape(seq_len * batch_size).long()
            picts = pics.reshape(seq_len * batch_size, 65, 65)
            
            # FIXME Gaussian Blur _YEZ
            for i in range(seq_len * batch_size):
                if args.sigma != 0:
                    picts[i] = torch.from_numpy(gaussian_filter(picts[i], sigma=args.sigma, order=0))

            ind =indices.to(device)
            y = picts.to(device)
            yt = None

            B = len(ind)
            batch_it += B
            global_it = Nimg*epoch + batch_it

            beta = beta_schedule(global_it)
           
            yr = torch.from_numpy(data.particles_real[ind.numpy()]).to(device) if args.use_real else None
            if do_pose_sgd:
                pose_optimizer.zero_grad()
            rot, tran = posetracker.get_pose(ind)
            ctf_param = ctf_params[ind] if ctf_params is not None else None
            loss, gen_loss, kld = train_batch(model, lattice, batch_size, y, yt, rot, tran, optim, beta, args.beta_control, tilt, ctf_params=ctf_param, yr=yr, use_amp=args.amp, scaler=scaler, device=device, zdim=args.zdim, seq_len=args.seq_len, ind=ind)
            if do_pose_sgd and epoch >= args.pretrain:
                pose_optimizer.step()

            # logging
            gen_loss_accum += gen_loss*B
            kld_accum += kld*B
            loss_accum += loss*B

            if batch_it % args.log_interval == 0:
                log('# [Train Epoch: {}/{}] [{}/{} images] gen loss={:.6f}, kld={:.6f}, beta={:.6f}, loss={:.6f}'.format(epoch+1, num_epochs, batch_it, Nimg, gen_loss, kld, beta, loss))
        flog('# =====> Epoch: {} Average gen loss = {:.6}, KLD = {:.6f}, total loss = {:.6f}; Finished in {}'.format(epoch+1, gen_loss_accum/Nimg, kld_accum/Nimg, loss_accum/Nimg, dt.now()-t2))

        if args.checkpoint and epoch % args.checkpoint == 0:
            out_weights = '{}/weights.{}.pkl'.format(args.outdir,epoch)
            out_z = '{}/z.{}.pkl'.format(args.outdir, epoch)
            model.eval()
            with torch.no_grad():
                z_mu, z_logvar = eval_z(model, args.particles, lattice, data, args.batch_size, device, posetracker.trans, tilt is not None, ctf_params, args.use_real, seq_len=args.seq_len, zdim=args.zdim, sigma=args.sigma)
                save_checkpoint(model, optim, epoch, z_mu, z_logvar, out_weights, out_z)
            if args.do_pose_sgd and epoch >= args.pretrain:
                out_pose = '{}/pose.{}.pkl'.format(args.outdir, epoch)
                posetracker.save(out_pose)

    # save model weights, latent encoding, and evaluate the model on 3D lattice
    out_weights = '{}/weights.pkl'.format(args.outdir)
    out_z = '{}/z.pkl'.format(args.outdir)
    model.eval()
    with torch.no_grad():
        z_mu, z_logvar = eval_z(model, args.particles, lattice, data, args.batch_size, device, posetracker.trans, tilt is not None, ctf_params, args.use_real, seq_len=args.seq_len, zdim=args.zdim, sigma=args.sigma)
        # np.save('/home/yez/7bcq_n1000_proj/z_logvar',z_logvar) # FIXME Save z_logvar _YEZ
        save_checkpoint(model, optim, epoch, z_mu, z_logvar, out_weights, out_z)
    
    if args.do_pose_sgd and epoch >= args.pretrain:
        out_pose = '{}/pose.pkl'.format(args.outdir)
        posetracker.save(out_pose)
    td = dt.now()-t1
    flog('Finished in {} ({} per epoch)'.format(td, td/(num_epochs-start_epoch)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    utils._verbose = args.verbose
    main(args)

