"""Measured, bounded PCA feature repair for existing Kilosort/Phy folders.

Preserves curation. Legacy folders need a sequential read because waveform
coverage was partial. Features are actual bandpassed observations, not template
amplitudes synthesized to look like measured PCs.
"""
from __future__ import annotations
import ast
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.signal import butter,sosfiltfilt
from .session_export import NpyStream

PATCH_ROWS=256
CORE_SECONDS=2.
MAX_LOCAL_CHANNELS=16


def read_params(folder):
    values={}
    for node in ast.parse((folder/'params.py').read_text(encoding='utf-8')).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            values[node.targets[0].id]=ast.literal_eval(node.value)
    if values.get('dtype')!='int16' or int(values.get('offset',0))!=0:
        raise ValueError('Feature repair currently requires offset-zero int16 raw sources')
    paths=values['dat_path'];paths=[paths] if isinstance(paths,str) else paths
    files=[Path(p) if Path(p).is_absolute() else folder/p for p in paths]
    nchan=int(values['n_channels_dat']);rate=float(values['sample_rate'])
    if nchan<=0 or not np.isfinite(rate) or rate<=750:
        raise ValueError('Invalid feature source channel count/sample rate')
    return values,files,nchan,rate


def basis_and_contacts(templates,nchan):
    if templates.ndim!=3 or (templates.shape[1]<21 or templates.shape[1]%2!=1) or templates.shape[2]!=nchan:
        raise ValueError('Expected dense odd-length TeraSort templates')
    width=min(MAX_LOCAL_CHANNELS,nchan)
    covariance=np.zeros((templates.shape[1],templates.shape[1]),np.float64)
    contacts=np.empty((len(templates),width),np.int32)
    for unit in range(len(templates)):
        waveform=np.asarray(templates[unit],np.float64)
        if not np.isfinite(waveform).all():raise ValueError('Nonfinite exported templates')
        energy=np.sum(waveform**2,axis=0)
        contacts[unit]=np.argsort(-energy,kind='stable')[:width]
        selected=waveform[:,contacts[unit]]
        covariance += selected @ selected.T
    _,vectors=np.linalg.eigh(covariance)
    basis=vectors[:,-3:].astype(np.float32)
    # Canonical signs keep repeat repairs stable.
    for pc in range(3):
        if basis[np.argmax(np.abs(basis[:,pc])),pc]<0:basis[:,pc]*=-1
    return basis,contacts


def generate_phy_features(folder,*,progress=None):
    """Add full-row pc_features/pc_feature_ind files without rewriting sorting."""
    folder=Path(folder).expanduser().resolve(strict=True)
    names=['pc_features.npy','pc_feature_ind.npy','pc_feature_basis.npy','pc_feature_provenance.json']
    targets=[folder/name for name in names]
    if any(p.exists() for p in targets):
        if not all(p.exists() for p in targets[:2]):
            raise ValueError('Partial/existing feature files: preserve them before repair')
        features=np.load(targets[0],mmap_mode='r');indices=np.load(targets[1],mmap_mode='r')
        times=np.load(folder/'spike_times.npy',mmap_mode='r')
        templates=np.load(folder/'templates.npy',mmap_mode='r')
        if (features.ndim!=3 or features.dtype not in (np.float32,np.float64)
                or indices.ndim!=2 or indices.dtype.kind not in 'iu'
                or indices.shape[1]!=features.shape[2] or len(features)!=len(times)
                or len(indices)!=len(templates) or features.shape[1]<1):
            raise ValueError('Invalid existing Phy features')
        if targets[-1].exists():return json.loads(targets[-1].read_text())
        return dict(existing_features_preserved=True,spikes=len(features),pcs=features.shape[1],local_channels=features.shape[2])
    temporary=[p.with_name(p.name+'.partial') for p in targets]
    if any(p.exists() for p in temporary):raise FileExistsError('Unpublished PCA repair exists; inspect before retrying')
    params,files,nchan,rate=read_params(folder)
    templates=np.load(folder/'templates.npy',mmap_mode='r')
    times=np.load(folder/'spike_times.npy',mmap_mode='r')
    units=np.load(folder/'spike_templates.npy',mmap_mode='r')
    if times.ndim!=1 or units.shape!=times.shape or times.dtype.kind not in 'iu' or units.dtype.kind not in 'iu':
        raise ValueError('Invalid exported spike coordinate arrays')
    previous=-1
    for first in range(0,len(times),65536):
        block=np.asarray(times[first:first+65536])
        if len(block) and (int(block[0])<previous or np.any(block[1:]<block[:-1])):
            raise ValueError('Spike times are not sorted')
        if len(block):previous=int(block[-1])
    basis,contacts=basis_and_contacts(templates,nchan)
    before={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in files}
    lengths=[]
    for p in files:
        if p.stat().st_size%(2*nchan):raise ValueError('Raw source frame is truncated')
        lengths.append(p.stat().st_size//(2*nchan))
    if len(times) and (int(times[0])<0 or int(times[-1])>=sum(lengths)):
        raise ValueError('Exported spikes outside concatenated raw clock')
    sos=butter(3,[300.,min(6000.,.4*rate)],btype='bandpass',fs=rate,output='sos')
    shanks=(np.load(folder/'channel_shanks.npy') if (folder/'channel_shanks.npy').is_file() else np.zeros(nchan,int))
    if shanks.shape!=(nchan,):raise ValueError('Invalid channel shanks')
    core_frames=max(templates.shape[1],int(round(rate*CORE_SECONDS)));halo=max(templates.shape[1]//2+1,int(round(rate*.1)))
    started=time.perf_counter();cursor=base=read_bytes=0
    with NpyStream(temporary[0],'<f4',(len(times),3,contacts.shape[1])) as output:
        for path,length in zip(files,lengths):
            with path.open('rb',buffering=0) as handle:
                for start in range(0,length,core_frames):
                    stop=min(length,start+core_frames)
                    left=max(0,start-halo);right=min(length,stop+halo)
                    handle.seek(left*nchan*2)
                    raw=np.fromfile(handle,dtype='<i2',count=(right-left)*nchan).reshape(right-left,nchan)
                    read_bytes += raw.nbytes
                    voltage=raw.astype(np.float32)
                    # Bounded2-second frame; no recording-sized FP32 allocation.
                    work_rows=max(1,8*1024**2//(nchan*4))
                    for shank in np.unique(shanks):
                        indices=np.flatnonzero(shanks==shank)
                        if len(indices)<2:continue
                        for row in range(0,len(voltage),work_rows):
                            view=voltage[row:row+work_rows]
                            reference=np.median(view[:,indices],axis=1)
                            view[:,indices]-=reference[:,None]
                    work_columns=max(1,8*1024**2//(len(voltage)*8))
                    for channel in range(0,nchan,work_columns):
                        voltage[:,channel:channel+work_columns]=sosfiltfilt(sos,voltage[:,channel:channel+work_columns],axis=0)
                    first=cursor
                    while cursor<len(times) and int(times[cursor])<base+stop:cursor+=1
                    if first<cursor and int(times[first])<base+start:raise ValueError('Spike times are not sorted')
                    for row in range(first,cursor,PATCH_ROWS):
                        end=min(cursor,row+PATCH_ROWS)
                        t=np.asarray(times[row:end],np.int64)-base-left
                        u=np.asarray(units[row:end],np.int64)
                        if np.any(u<0) or np.any(u>=len(contacts)):raise ValueError('Invalid spike template ID')
                        indices=t[:,None]+np.arange(-(templates.shape[1]//2),templates.shape[1]//2+1,dtype=np.int64)[None]
                        # File-edge padding represents unavailable voltage; it
                        # never borrows samples across explicit file boundaries.
                        valid=(indices>=0)&(indices<len(voltage))
                        patches=voltage[np.clip(indices,0,len(voltage)-1)[:,:,None],contacts[u][:,None,:]]
                        patches*=valid[:,:,None]
                        output.write(np.einsum('tp,ntc->npc',basis,patches,optimize=True).astype(np.float32))
                    del raw,voltage
                    if progress is not None:progress(dict(stage='phy_features',completed_spikes=cursor,
                        total_spikes=len(times),completed_samples=base+stop,total_samples=sum(lengths)))
            base+=length
    if cursor!=len(times):raise ValueError('Incomplete PCA feature rows')
    if before!={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in files}:raise ValueError('Raw source changed during repair')
    with temporary[1].open('xb') as handle:np.save(handle,contacts)
    with temporary[2].open('xb') as handle:np.save(handle,basis)
    report=dict(format='TeraSort measured Phy PCA features',version=1,spikes=len(times),
        pcs=3,local_channels=contacts.shape[1],waveform_samples=templates.shape[1],raw_voltage_bytes_read=read_bytes,
        preprocessing='CPU300-min(6000,0.4fs)Hz third-order zero-phase bandpass after per-shank median reference; ADC counts',
        scope='curation features only; sorting assignments unchanged',
        basis='bounded exported template covariance; no synthetic per-spike values',
        file_edge_padding='zero unavailable samples within each file',
        wall_seconds=time.perf_counter()-started,source_files={k:list(v) for k,v in before.items()})
    temporary[3].write_text(json.dumps(report,indent=2)+'\n')
    for index in (1,2,0,3):
        tmp,target=temporary[index],targets[index]
        if target.exists():raise FileExistsError(target)
        os.replace(tmp,target)
    return report
