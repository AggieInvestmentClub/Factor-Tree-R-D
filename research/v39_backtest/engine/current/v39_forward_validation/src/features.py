"""Extend source features; retain every pre2024 predictor exactly."""
from common import *
import numpy as np,pandas as pd,importlib.util,gc

def main():
    assert read(HERE/'data/membership_qualification.json')['passed']
    src=SOURCE/'src/historical_features.py';text=src.read_text()
    primitives=SOURCE.parent/'v16_payoff_alignment/src/stock_features.py';primitive_text=primitives.read_text();old='RESERVED = pd.Timestamp("2024-01-01")';assert primitive_text.count(old)==1
    adapted=HERE/'src/stock_features_v39.py';adapted.write_text(primitive_text.replace(old,'RESERVED = pd.Timestamp("'+BOUNDARY+'")'))
    text=text.replace('from stock_features import ','from stock_features_v39 import ')
    text=text.replace("from historical_sources import ROOT,DATA,membership_names,label_quotes","from historical_sources import ROOT,DATA,membership_names,label_quotes\nDATA=Path("+repr(str(HERE/'data'))+")")
    text=text.replace('membership_candidate_snapshots.csv.gz','membership_snapshots.csv.gz').replace('sector_candidate_snapshots.csv.gz','sector_snapshots.csv.gz')
    text=text.replace("pd.Timestamp('2024-01-01')","pd.Timestamp('"+BOUNDARY+"')")
    text=text.replace("calendar>=pd.Timestamp('2002-01-01')","calendar>=pd.Timestamp('2023-11-01')")
    generated=HERE/'src/generated_features.py';generated.write_text(text)
    sys.path.insert(0,str(SOURCE/'src'));spec=importlib.util.spec_from_file_location('v39_generated_features',generated);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.main()
    folder=HERE/'data/candidate_features';new=pd.read_pickle(folder/'panel.pkl');old=pd.read_pickle(SOURCE/'data/candidate_features/panel.pkl');cols=read(folder/'columns.json')
    key=['decision_date','ticker'];old=old.set_index(key);new=new.set_index(key);overlap=new.index.intersection(old.index)
    frozen=cols+['eligible','model_eligible','sector','sector_available','sector_unique','idio_vol21','beta_spy63']
    pd.testing.assert_frame_equal(old.loc[overlap,frozen],new.loc[overlap,frozen],check_exact=True,check_dtype=False)
    labelcols=['execution_date','label_end_h5','future_return_h5','future_parent_return_h5','target_risk_resid_h5','future_spy_h5','response_spy','v8_target_absolute_h5','v8_target_spy_h5']
    audit=[]
    for col in labelcols:
        a=old.loc[overlap,col];b=new.loc[overlap,col];valid=a.notna()
        if pd.api.types.is_numeric_dtype(a):
            error=float((a[valid]-b[valid]).abs().max());assert error<1e-10 or not valid.any(),(col,error)
        else:pd.testing.assert_series_equal(a[valid],b[valid],check_names=False,check_dtype=False)
        audit.append(dict(column=col,previous_observed=int(valid.sum()),new_tail_values=int((~valid&b.notna()).sum())))
        old.loc[overlap,col]=b
    forward=new[new.index.get_level_values('decision_date')>=pd.Timestamp(START)];combined=pd.concat([old,forward]).sort_index().reset_index();assert not combined.duplicated(key).any()
    combined.to_pickle(HERE/'data/panel.pkl');new.reset_index().to_pickle(folder/'forward_with_overlap.pkl')
    write(HERE/'data/feature_qa.json',dict(passed=True,original_predictors_exact=True,overlap_rows=len(overlap),rows=len(combined),forward_rows=len(forward),mature_old_label_identity=True,label_tail_extension=audit,source_sha256=sha(src),adapter_sha256=sha(generated),primitive_source_sha256=sha(primitives),primitive_adapter_sha256=sha(adapted),primitive_change='Only exclusive date boundary; original file and globals untouched',panel_sha256=sha(HERE/'data/panel.pkl')))
    print('FEATURES_EXTENDED',len(combined),len(forward),flush=True)
if __name__=='__main__':main()
