"""Separate stock ordering from raw capital request,and test funded long-only core."""
from study50 import *
def transport(score,base,mask):
    out=np.zeros_like(base);active=np.flatnonzero(mask);order=active[np.argsort(score[active],kind='stable')];out[order]=np.sort(base[active]);return out
def core(score,d,j):
    mask=d['mask'][j];active=np.flatnonzero(mask);k=max(1,int(np.ceil(.2*len(active))));top=active[np.argsort(score[active],kind='stable')[-k:]]
    broad=np.maximum(p.unit_market(d['beta'][j],mask),0.)*mask
    if broad.sum()<=0:broad=mask.astype(float)
    w=.495*broad/broad.sum();w[top]+=.495/k;return previous.numpy_contract(w,d,j,'long_only')
def main():
    restrict();freeze();d=data();indices=np.flatnonzero(d['dates'].year>=2007);base=np.concatenate([np.load(baseline_folder(y)/f'account_inputs/long_only_scores_{y}.npy') for y in YEARS]);assert len(indices)==len(base)
    j=np.flatnonzero(d['dates']<pd.Timestamp('2024-01-01'))[-1];view=previous.initial_view(d,j);m=p.Policy('replay_random');m.load_state_dict(torch.load(V46/'models/long_only_2024.pt',weights_only=True));firstbase=previous.score_model(m,d,np.array([j]))[0]
    tables={};initial={};checks=[]
    tables['baseline_core']=pd.DataFrame([core(s,d,k) for s,k in zip(base,indices)],index=d['dates'][indices],columns=d['names']);initial['baseline_core']=core(firstbase,view,j)
    for name in MODELS:
        score=np.concatenate([np.load(HERE/f'account_inputs/{name}_scores_{y}.npy') for y in YEARS]);first=np.load(HERE/f'account_inputs/{name}_initial_scores.npy')[0]
        swapped=np.array([transport(s,b,d['mask'][k]) for s,b,k in zip(score,base,indices)])
        for s,b,k in zip(swapped,base,indices):np.testing.assert_array_equal(np.sort(s[d['mask'][k]]),np.sort(b[d['mask'][k]]))
        tables[name+'_transport']=previous.targets_from_scores(d,indices,swapped,'long_only');fs=transport(first,firstbase,d['mask'][j]);f=previous.targets_from_scores(d,np.array([j]),fs[None,:],'long_only').iloc[0].to_numpy();initial[name+'_transport']=previous.numpy_contract(f,view,j,'long_only')
        tables[name+'_core']=pd.DataFrame([core(s,d,k) for s,k in zip(score,indices)],index=d['dates'][indices],columns=d['names']);initial[name+'_core']=core(first,view,j)
    # Transporting the old ordering must reproduce the original scheduled weights.
    original=previous.targets_from_scores(d,indices,base,'long_only').to_numpy();selfscores=np.array([transport(s,s,d['mask'][k]) for s,k in zip(base,indices)]);error=float(abs(previous.targets_from_scores(d,indices,selfscores,'long_only').to_numpy()-original).max());assert error==0
    for arm in ARMS:
        for part,folder in (('historical',V47),('forward',V46)):
            if arm in ('baseline','broad_long'):table=pd.read_pickle(folder/f'account_inputs/{"long_only" if arm=="baseline" else "broad_long"}_chosen.pkl')
            else:
                table=tables[arm];table=table[table.index<BOUNDARY] if part=='historical' else pd.concat([pd.DataFrame(initial[arm][None,:],index=pd.DatetimeIndex(['2023-12-29']),columns=d['names']),table[table.index>=BOUNDARY]])
            assert table.index.is_unique and table.index.is_monotonic_increasing and np.isfinite(table.to_numpy()).all() and table.to_numpy().min()>=0 and table.to_numpy().max()<=.15+1e-10 and table.sum(1).max()<=.99+1e-10
            file=HERE/f'account_inputs/{arm}_{part}.pkl';table.to_pickle(file);checks.append(dict(arm=arm,part=part,sha256=sha(file)))
    write(HERE/'reports/target_qa.json',dict(passed=True,checks=checks,self_transport_error=error,raw_score_distribution_preserved=True,source_unchanged=source_unchanged()));print('TARGETS_READY',flush=True)
if __name__=='__main__':main()
