"""Fixed-recipe LambdaRank models on identical mature query dates for every horizon."""
from study50 import *
import lightgbm as lgb
from scipy.stats import rankdata
import argparse,time
def fit(year):
    restrict();freeze();began=time.monotonic();d=data();cut=pd.Timestamp(year,1,1);ends=np.load(HERE/'data/ends_h63.npy');complete=np.load(HERE/'data/common_complete.npy')
    ix=np.flatnonzero(complete&(d['dates']<cut)&(d['dates']>=cut-pd.DateOffset(years=10))&(ends<cut.to_datetime64()));assert len(ix)>0
    query=np.flatnonzero(d['dates'].year==year);first=np.flatnonzero(d['dates']<pd.Timestamp('2024-01-01'))[-1]
    if year==2024:assert ends[ix].max()<pd.Timestamp('2023-12-29').to_datetime64()
    sizes=d['mask'][ix].sum(1);mask=d['mask'][ix];parameters=dict(objective='lambdarank',n_estimators=100,learning_rate=.05,num_leaves=7,max_depth=3,min_child_samples=200,reg_lambda=10.,colsample_bytree=1.,subsample=1.,random_state=20261119,n_jobs=1,deterministic=True,force_col_wise=True,verbosity=-1,label_gain=[0,1,2,3,4],lambdarank_truncation_level=100,lambdarank_norm=True)
    designs={family:get_design(d,ix,family)[mask] for family in FAMILIES}
    for h in HORIZONS:
        values=np.load(HERE/f'data/returns_h{h}.npy')[ix]-d['beta'][ix]*np.load(HERE/f'data/market_h{h}.npy')[ix,None];ys=[]
        for row,valid in zip(values,mask):
            r=rankdata(row[valid],method='average');ys.extend(np.minimum(4,np.floor(5*(r-.5)/len(r))).astype(np.int32))
        y=np.array(ys,dtype=np.int32);assert np.isfinite(y).all()
        for family in FAMILIES:
            name=f'{family}_h{h}';done=HERE/f'models/{name}_{year}.json'
            if done.exists():assert read(done)['protocol_sha256']==sha(HERE/'protocol.json');continue
            model=lgb.LGBMRanker(**parameters);model.fit(designs[family],y,group=sizes)
            file=HERE/f'models/{name}_{year}.txt';model.booster_.save_model(str(file));x=get_design(d,query,family);score=model.booster_.predict(x.reshape(-1,x.shape[-1]),num_threads=1).reshape(x.shape[:2]);assert np.isfinite(score).all()
            target=HERE/f'account_inputs/{name}_scores_{year}.npy';np.save(target,score)
            if year==2024:
                x=get_design(d,np.array([first]),family);np.save(HERE/f'account_inputs/{name}_initial_scores.npy',model.booster_.predict(x.reshape(-1,x.shape[-1]),num_threads=1).reshape(x.shape[:2]))
            write(done,dict(model=name,year=year,protocol_sha256=sha(HERE/'protocol.json'),model_sha256=sha(file),scores_sha256=sha(target),training_queries=len(ix),training_rows=int(sizes.sum()),training_query_sha256=hashlib.sha256(d['dates'][ix].asi8.tobytes()).hexdigest(),max_training_label_end=str(ends[ix].max()),horizon=h,parameters=parameters,lightgbm_version=lgb.__version__,feature_importance_gain=model.booster_.feature_importance('gain').tolist()))
            print('FIT',year,name,len(ix),round(time.monotonic()-began,1),flush=True)
    print('YEAR_DONE',year,flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--year',type=int,required=True);fit(parser.parse_args().year)
