"""Bounded, temporally validated PCA split proposals; never consumes GT."""
import numpy as np

from .session_models import LocalModels


def split_templates(models, training, validation, *, max_new=16):
    if training.windows & validation.windows or not 0 <= max_new <= 32:
        raise ValueError('Disjoint partitions and at most 32 new models required')
    waveforms, channels, anchors = list(models.waveforms.copy()), list(models.channels.copy()), list(models.anchors)
    audit = []
    for unit, old in enumerate(models.waveforms):
        def rows(evidence):
            return [(core,r[2]) for (u,core), rs in sorted(evidence.rows.items()) if u == unit for r in rs]
        train, valid = rows(training), rows(validation)
        record = dict(unit=unit, training_count=len(train), validation_count=len(valid),
                      split=False, reason='insufficient_support')
        audit.append(record)
        if len(train) < 32 or len(valid) < 16:
            continue
        a,b = np.stack([r[1] for r in train]),np.stack([r[1] for r in valid])
        x,y = a[:,models.nt//2-8:models.nt//2+9].reshape(len(a),-1),b[:,models.nt//2-8:models.nt//2+9].reshape(len(b),-1)
        mean = x.mean(axis=0)
        _,s,vt = np.linalg.svd(x-mean,full_matrices=False)
        if s[0] < 1e-6:
            record['reason'] = 'no_shape_variation'; continue
        basis = vt[:3]
        x,y = (x-mean)@basis.T,(y-mean)@basis.T
        centers = x[[np.argmin(x[:,0]),np.argmax(x[:,0])]].copy()
        for _ in range(12):
            labels = np.argmin(np.sum((x[:,None]-centers[None])**2,axis=2),axis=1)
            if min(np.bincount(labels,minlength=2)) < 8:
                break
            centers = np.stack([x[labels==j].mean(axis=0) for j in range(2)])
        vl = np.argmin(np.sum((y[:,None]-centers[None])**2,axis=2),axis=1)
        if min(np.bincount(labels,minlength=2)) < 12 or min(np.bincount(vl,minlength=2)) < 6:
            record['reason']='unbalanced_clusters';continue
        if any(len({train[i][0] for i in np.flatnonzero(labels==j)}) < 2 or
               len({valid[i][0] for i in np.flatnonzero(vl==j)}) < 2 for j in range(2)):
            record['reason']='temporal_segregation';continue
        separation = float(np.linalg.norm(centers[0]-centers[1]) /
                           max(np.sqrt(np.mean(np.sum((x-centers[labels])**2,axis=1))),1e-9))
        proposals = np.stack([np.median(a[labels==j],axis=0) for j in range(2)])
        old_error = np.mean((b-old)**2,axis=(1,2))
        new_error = np.mean((b-proposals[vl])**2,axis=(1,2))
        gain = float(1-new_error.mean()/max(old_error.mean(),1e-9))
        similarity = float(np.sum(proposals[0]*proposals[1]) /
                           max(np.linalg.norm(proposals[0])*np.linalg.norm(proposals[1]),1e-9))
        record.update(separation=separation,validation_gain=gain,child_similarity=similarity)
        if separation < 2.5 or gain < .1 or similarity > .98 or not np.isfinite(proposals).all():
            record['reason']='separation_or_holdout_rejected';continue
        if any(np.argmax(np.abs(p)) != (models.nt//2)*p.shape[1] for p in proposals):
            record['reason']='alignment_changed';continue
        if len(waveforms)-len(models.waveforms) >= max_new or len(waveforms) >= 4096:
            record['reason']='model_budget';continue
        contact = models.channels[unit]
        if any(sum(c in row for row in channels) >= 512 for c in contact if c >= 0):
            record['reason']='partition_budget';continue
        # Preserve the parent's ID for the closer child; append the other.
        closest = int(np.argmin(np.sum((proposals-old)**2,axis=(1,2))))
        waveforms[unit] = proposals[closest]
        record.update(split=True,reason='validated_pca_split',child_unit=len(waveforms))
        waveforms.append(proposals[1-closest]);channels.append(contact.copy());anchors.append(models.anchors[unit])
    return LocalModels(np.array(waveforms),np.array(channels),np.array(anchors),
                       np.pad(models.assigned,(0,len(waveforms)-len(models.waveforms))),
                       models.version+int(any(r['split'] for r in audit))),audit
