"""Both historical mandates per year,sharing immutable validation input preparation."""
from study47 import *
import time,argparse

def fit(year):
    restrict();freeze();assert read(HERE/'reports/preparation.json')['passed'];d=data();cut=pd.Timestamp(year,1,1)
    start=cut-pd.DateOffset(years=10);split=cut-pd.DateOffset(years=2);full=p.selected(d,cut,start);inner=p.selected(d,split,start)
    val=np.flatnonzero((d['dates']>=split)&(d['dates']<cut));query=np.flatnonzero(d['dates'].year==year)
    assert len(full)==read(ROOT/f'outputs/v31_growth_long_bias/models/log_growth_{year}.json')['training_periods']
    assert d['ends'][full].max()<cut and d['ends'][inner].max()<split
    clock=pd.DatetimeIndex(d['calendar']);begin=clock[clock.get_loc(d['dates'][val[0]])+1];end=clock[clock<cut][-1]
    claims=read(SOURCE/'data/assembled/claims.json');crossed=[v for v in claims if str(begin.date())<=v['ex_date']<=str(end.date())<v['settlement_date']]
    early=year<2009 or len(inner)<20;usable=not early and not crossed;account=None;reference=None;reference_audit={};failure=None
    if usable:
        account=Account(str(begin.date()),str(end.date()),str(cut.date()))
        try:
            ledger,_,_,reference_audit,_=account.run(previous.broad_targets(d,val),'broad_long',strict=False)
            usable=reference_audit['v46_selection_qualified'];reference=float(252*np.log1p(ledger.daily_return).mean()) if usable else None
        except ValueError as error:
            # Only explicit economic invalidity invokes the declared fallback; no hiding coding errors.
            if str(error) not in ('Cash borrowing after open/corporate cash settlement','Stock cash conservation failed'):raise
            if str(error)=='Stock cash conservation failed':raise
            usable=False;failure=str(error)
    for mode in MODES:
        done=HERE/f'models/{mode}_{year}.json'
        if done.exists():assert read(done)['protocol_sha256']==sha(HERE/'protocol.json');continue
        began=time.monotonic();choices=[]
        if usable:
            for penalty in (.1,.01):
                model,training=learner.train(d,inner,mode,penalty);score=previous.score_model(model,d,val)
                try:
                    ledger,_,_,check,_=account.run(previous.targets_from_scores(d,val,score,mode),mode,strict=False)
                    qualified=check['v46_selection_qualified'];growth=float(252*np.log1p(ledger.daily_return).mean()) if qualified else None;error=None
                except ValueError as error_value:
                    if str(error_value)!='Cash borrowing after open/corporate cash settlement':raise
                    qualified=False;growth=None;check={};error=str(error_value)
                choices.append(dict(penalty=penalty,qualified=qualified,annual_log_growth=growth,
                  incremental_log_growth=growth-reference if qualified else None,audit=check,error=error,training=training))
                print('INNER_DONE',year,mode,penalty,qualified,flush=True);del model;gc.collect()
        fallback=not usable or not all(q['qualified'] for q in choices);penalty=.1 if fallback else max(choices,key=lambda q:q['incremental_log_growth'])['penalty']
        model,training=learner.train(d,full,mode,penalty);torch.save(model.state_dict(),done.with_suffix('.pt'))
        score_path=HERE/f'account_inputs/{mode}_scores_{year}.npy';np.save(score_path,previous.score_model(model,d,query))
        write(done,dict(mode=mode,year=year,protocol_sha256=sha(HERE/'protocol.json'),weights_sha256=sha(done.with_suffix('.pt')),
          scores_sha256=sha(score_path),training_periods=len(full),max_training_label_end=str(d['ends'][full].max()),
          inner_training_periods=len(inner),inner_max_training_label_end=str(d['ends'][inner].max()),selected_penalty=penalty,
          validation_fallback=fallback,early_history_fallback=early,unsettled_claim_fallback=bool(crossed),validation_failure=failure,
          validation_start=str(begin.date()),validation_end=str(end.date()),validation_reference_audit=reference_audit,
          selection=choices,training=training,seconds=time.monotonic()-began))
        print('FIT_DONE',year,mode,round(time.monotonic()-began,1),flush=True);del model;gc.collect()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--year',type=int,required=True);args=parser.parse_args();fit(args.year)
