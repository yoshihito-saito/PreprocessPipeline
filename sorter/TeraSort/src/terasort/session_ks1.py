"""Experimental bounded original-KS1 fitting; external pinned CUDA source.

Native KS1 models are required. Learning is a separate bounded calibration
step. This does not run TeraSort's cosine detector under a KS1 name.
"""
from pathlib import Path
import hashlib

import h5py
import numpy as np
from scipy.signal import butter

RULE = 'ks1_native_cuda_bounded_v5'
SOURCE_HASHES = {
    'mexMPmuFEAT.cu': '2d911b496f631857c2ef3f63dbe2bb11c69a69e9e4699eb36329cdb394be23e1',
    'mexWtW2.cu': '43c085e3d459b724b54e70d4b1a444420a15ff8cfef79e962625e422707d8e74',
}
NORMALIZED_SOURCE_HASHES = {
    'mexMPmuFEAT.cu': '9b182246e51f292e7a0268a31e4a1d9d888301d5b1fcdff2707e1655f3586dca',
    'mexWtW2.cu': 'b74b46dff156e6c777d101535ea0b16c880185f060689d897f0f7730fbdeb874',
}
MAX_FILTERS = 2048
MAX_CHANNELS = 4096
MAX_SPIKES = 50000


def device_source(path):
    path = Path(path)
    raw = path.read_bytes()
    normalized=raw.replace(b'\r\n',b'\n')
    if hashlib.sha256(normalized).hexdigest() != NORMALIZED_SOURCE_HASHES.get(path.name):
        raise ValueError('KS1 CUDA source differs from the tested official revision')
    text = normalized.decode('utf-8')
    body = text[text.index('const int'):text.index('void mexFunction')]
    if path.name=='mexMPmuFEAT.cu':
        begin=body.index('__global__ void\tsubSpikes')
        end=body.index('__global__ void\tsubtract_spikes')
        section=body[begin:end]
        target='(2*nt0-1)*Nfilt*bid];'
        if section.count(target)!=1:raise ValueError('Unrecognized KS1 subtraction body')
        # Each filter block owns its output column. Synchronize successive
        # spikes: adjacent 121-sample corrections can share addresses even
        # when cleanup's 61-sample exclusion holds. Native KS1 races here.
        section=section.replace(target,target+'\n    __syncthreads();')
        body=body[:begin]+section+body[end:]
    return 'extern "C" {\n' + body + '\n}'


def load_model(path):
    """Read only model arrays; never load recording-sized rez spike features."""
    path = Path(path)
    if path.suffix.lower() == '.npz':
        import zipfile
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist())>512*1024**2:
                raise ValueError('KS1 model exceeds the 512 MiB host state budget')
        with np.load(path, allow_pickle=False) as data:
            model = {key:data[key] for key in data.files}
    else:
        with h5py.File(path, 'r') as f:
            r = f['rez']; ops = r['ops']
            nf = int(ops['Nfilt'][0,0]); nc = int(ops['Nchan'][0,0])
            if not 32 <= nf <= MAX_FILTERS or nf % 32 or not 3 <= nc <= MAX_CHANNELS:
                raise ValueError('KS1 bank exceeds the bounded core capacity')
            if sum(r[key].size*r[key].dtype.itemsize for key in
                    ('W','U','mu','Wrot','WrotInv','xcoords','ycoords'))>512*1024**2:
                raise ValueError('KS1 model exceeds the 512 MiB host state budget')
            nt = int(ops['nt0'][0,0])
            w = r['W'][:].transpose(2,1,0)
            # Native fullMPMU prepends zeros for Phy centering after fitting.
            padding = len(w)-nt
            if padding and (padding != int(ops['nt0min'][0,0])+1 or np.any(w[:padding])):
                raise ValueError('Unrecognized KS1 temporal-template padding')
            model = dict(W=w[padding:], U=r['U'][:].transpose(2,1,0),
                mu=r['mu'][:].ravel(), Wrot=r['Wrot'][:].T,
                WrotInv=r['WrotInv'][:].T,
                channels=ops['chanMap'][:].ravel().astype(np.int32)-1,
                positions=np.c_[r['xcoords'][:].ravel(),r['ycoords'][:].ravel()],
                sample_rate=float(ops['fs'][0,0]), fshigh=float(ops['fshigh'][0,0]),
                NT=int(ops['NT'][0,0]), ntbuff=int(ops['ntbuff'][0,0]),
                nt0min=int(ops['nt0min'][0,0]), scaleproc=float(ops['scaleproc'][0,0]),
                threshold=float(ops['Th'][:].ravel()[-1]),
                regularization=float(ops['lam'][:].ravel()[-1]),
                reference_cache_batches=int(r['temp']['Nbatch_buff'][0,0]))
            if float(ops['epu'][0,0]) != np.inf or 'fslow' in ops:
                raise ValueError('Only the tested native KS1 high-pass / epu=Inf frame is supported')
    W, U = model['W'], model['U']
    nf = len(model['mu']); nc = len(model['channels'])
    numbers = [float(model[k]) for k in ('sample_rate','fshigh','scaleproc','threshold','regularization')]
    if not np.isfinite(numbers).all() or np.any(model['mu'] < 0):
        raise ValueError('Invalid finite KS1 frame parameters')
    if (W.shape != (61,nf,3) or U.shape != (nc,nf,3) or
            not 32 <= nf <= MAX_FILTERS or nf%32 or not 3 <= nc <= MAX_CHANNELS or
            model['Wrot'].shape != (nc,nc) or model['WrotInv'].shape != (nc,nc) or
            model['positions'].shape != (nc,2) or
            np.any(model['channels'] < 0) or len(np.unique(model['channels']))!=nc or
            int(model['NT']) != 32832 or int(model['ntbuff']) != 64 or
            int(model['nt0min']) != 20 or
            not 0 < float(model['fshigh']) < float(model['sample_rate'])/2 or
            float(model['scaleproc']) <= 0 or float(model['threshold']) <= 0 or
            float(model['regularization']) < 0):
        raise ValueError('Unsupported or invalid KS1 model/frame')
    dead = model['mu'] == 0
    if not np.isfinite(W[:,~dead]).all() or not np.isfinite(U).all():
        raise ValueError('Nonfinite active KS1 templates')
    # Native zero-energy templates contain 0/0 temporal factors. Preserve IDs
    # and make only those mathematically inactive rows explicit zeros.
    W[:,dead] = 0
    for key in ('W','U','mu','Wrot','WrotInv','positions'):
        if not np.isfinite(model[key]).all():
            raise ValueError('Nonfinite KS1 model state')
    return model


def round_int16(values):
    """MATLAB saturating half-away-from-zero conversion, not NumPy wrapping."""
    return np.clip(np.copysign(np.floor(np.abs(values)+.5),values),-32768,32767).astype(np.int16)


class KS1Core:
    def __init__(self, model, kilosort_path):
        from .cuda_environment import configure_cupy_cache, configure_cupy_libraries
        configure_cupy_cache()
        self._dll_directory, self._cuda_libraries = configure_cupy_libraries()
        import cupy as cp
        self.cp = cp; self.model = model
        cp.get_default_memory_pool().set_limit(size=6*1024**3)
        self.NT = int(model['NT']); self.nf = len(model['mu'])
        self.nc = len(model['channels']); self.nt = 61
        # Fixed bank + fixed batch; this estimate is independent of duration.
        self.working_bytes = 4*(self.nf*self.nf*121*3 + self.NT*self.nf*4 +
                                 self.NT*self.nc*3) + 128*1024**2
        if self.working_bytes > 6*1024**3:
            raise ValueError('KS1 core exceeds the 6 GiB allocation estimate')
        free,_ = cp.cuda.runtime.memGetInfo()
        if free < self.working_bytes + 512*1024**2:
            raise RuntimeError('Insufficient free VRAM for the bounded KS1 bank')
        self.modules = {}
        for filename in SOURCE_HASHES:
            source = device_source(Path(kilosort_path)/'CUDA'/filename)
            self.modules[filename] = cp.RawModule(code=source, options=('--std=c++11',))
        mod = self.modules['mexMPmuFEAT.cu']
        self.kernels = {name:mod.get_function(name) for name in
                        ('Conv1D','bestFilter','cleanup_spikes','subSpikes')}
        self.cross = self.modules['mexWtW2.cu'].get_function('crossFilter')
        self.params = cp.asarray([self.NT,self.nf,float(model['threshold']),MAX_SPIKES,
                                  50,self.nc,3,.99,np.inf,61],dtype=cp.float64)
        self.W = cp.asarray(model['W'],dtype=cp.float32,order='F')
        self.U = cp.asarray(model['U'],dtype=cp.float32,order='F')
        self.mu = cp.asarray(model['mu'],dtype=cp.float32)
        lam = np.zeros(self.nf,np.float32)
        with np.errstate(over='ignore',invalid='ignore'):
            np.divide(20,model['mu'],out=lam,where=model['mu']>0)
            lam=float(model['regularization'])*lam**2
        if not np.isfinite(lam).all():
            raise ValueError('KS1 amplitude regularization exceeds FP32 capacity')
        self.lam = cp.asarray(lam)
        self.nu = cp.zeros(self.nf,dtype=cp.float32)
        self.Wrot = cp.asarray(model['Wrot'],dtype=cp.float32)
        wtw = cp.zeros((self.nf,self.nf,121),dtype=cp.float32,order='F')
        temp = cp.empty_like(wtw,order='F')
        for i in range(3):
            for j in range(3):
                utu = cp.asfortranarray(self.U[:,:,i].T @ self.U[:,:,j])
                self.cross((self.nf//32,self.nf//32),(32,32),
                    (self.params,cp.asfortranarray(self.W[:,:,i]),
                     cp.asfortranarray(self.W[:,:,j]),utu,temp))
                wtw += temp
        self.WtW = cp.asfortranarray(wtw.transpose(2,0,1))
        self.data = cp.empty((self.NT,self.nf,3),dtype=cp.float32,order='F')
        self.dout = cp.zeros((self.NT,self.nf),dtype=cp.float32,order='F')
        # Original cleanup_spikes reads up to 2*nt-2 beyond a block. Pad the
        # score vectors so its final partial block never reads foreign memory.
        self.err = cp.zeros(self.NT+128,dtype=cp.float32)
        self.xbest = cp.zeros_like(self.err)
        self.ftype = cp.zeros(self.NT+128,dtype=cp.int32)
        self.st = cp.zeros(MAX_SPIKES,dtype=cp.int32)
        self.ids = cp.zeros_like(self.st)
        self.x = cp.zeros(MAX_SPIKES,dtype=cp.float32)
        self.C = cp.zeros_like(self.x)
        self.counter = cp.zeros(2,dtype=cp.int32)
        self.b,self.a = [x.astype(np.float32) for x in
            butter(3,float(model['fshigh']),fs=float(model['sample_rate']),btype='highpass')]
        filter_source=r'''
        extern "C" __global__ void highpass(float* x, const float* b,
            const float* a, int rows, int channels) {
            int c=blockIdx.x*blockDim.x+threadIdx.x;
            if(c>=channels)return;
            for(int direction=0;direction<2;direction++) {
                float z0=0.f,z1=0.f,z2=0.f;
                for(int t=0;t<rows;t++) {
                    int row=direction ? rows-1-t : t;
                    int p=row*channels+c;
                    float v=x[p],y=b[0]*v+z0;
                    z0=(z1+b[1]*v)-a[1]*y;
                    z1=(z2+b[2]*v)-a[2]*y;
                    z2=b[3]*v-a[3]*y;
                    x[p]=y;
                }
            }
        }'''
        self.filter_kernel=cp.RawKernel(filter_source,'highpass',options=('--fmad=false',))
        self.filter_b=cp.asarray(self.b);self.filter_a=cp.asarray(self.a)

    def preprocess(self, raw, *, offset=64, round_before_whitening=True):
        cp = self.cp
        if raw.shape != (self.NT+256,raw.shape[1]) or raw.dtype!=np.int16:
            raise ValueError('KS1 preprocessing requires one fixed INT16 buffered batch')
        x = cp.asarray(raw[:,self.model['channels']],dtype=cp.float32,order='C')
        self.filter_kernel(((self.nc+127)//128,),(128,),
                           (x,self.filter_b,self.filter_a,np.int32(len(raw)),np.int32(self.nc)))
        x = x[offset:offset+self.NT].copy()
        if round_before_whitening:
            x = cp.clip(cp.copysign(cp.floor(cp.abs(x)+.5),x),-32768,32767).astype(cp.int16).astype(cp.float32)
        gpu = x @ self.Wrot
        encoded = cp.clip(cp.copysign(cp.floor(cp.abs(gpu)+.5),gpu),-32768,32767).astype(cp.int16)
        return encoded.astype(cp.float32)/float(self.model['scaleproc']), cp.asnumpy(x)

    def match(self, whitened):
        cp=self.cp
        if whitened.shape!=(self.NT,self.nc) or whitened.dtype!=cp.float32:
            raise ValueError('KS1 fitter requires a fixed-size whitened FP32 batch')
        for rank in range(3):
            self.data[:,:,rank] = whitened @ self.U[:,:,rank]
        self.dout.fill(0); self.counter.fill(0)
        self.kernels['Conv1D']((self.nf,),(1024,),
            (self.params,self.data,self.W,self.dout))
        count=0; passes=np.empty(MAX_SPIKES,np.int16)
        for iteration in range(50):
            self.err.fill(0);self.xbest.fill(0);self.ftype.fill(0)
            self.kernels['bestFilter']((self.NT//1024,),(1024,),
                (self.params,self.dout,self.mu,self.lam,self.nu,self.xbest,self.err,self.ftype))
            self.kernels['cleanup_spikes']((self.NT//1024,),(1024,),
                (self.params,self.xbest,self.err,self.ftype,self.st,self.ids,self.x,self.C,self.counter))
            new_count=int(self.counter[0].get())
            if new_count >= MAX_SPIKES:
                raise RuntimeError('KS1 candidate burst exceeds the fixed core capacity; no truncated shard published')
            if new_count==count:break
            passes[count:new_count]=iteration
            # Atomic insertion order varies between launches. Canonicalize
            # each new group before subtraction for exact restart decisions.
            order=cp.lexsort(cp.stack((self.ids[count:new_count],self.st[count:new_count])))
            for values in (self.st,self.ids,self.x,self.C):
                values[count:new_count]=values[count:new_count][order]
            self.kernels['subSpikes']((self.nf,),(121,),
                (self.params,self.st,self.ids,self.x,self.counter,self.dout,self.WtW))
            self.counter[1]=new_count
            count=new_count
        result=[cp.asnumpy(v[:count]) for v in (self.st,self.ids,self.x,self.C)]
        order=np.lexsort((result[1],result[0]))
        return tuple(v[order] for v in result)+(passes[:count][order],)
