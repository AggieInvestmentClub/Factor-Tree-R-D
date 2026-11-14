import sys,os,json,hashlib,datetime
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parents[1];PROJECT=HERE.parents[1];PRIOR=Path(str(Path(__file__).resolve().parents[3] / 'prior'));SOURCE=PRIOR/'v17_long_history';V31=HERE.parent/'v31_growth_long_bias';V34=HERE.parent/'v34_stock_risk';V36=HERE.parent/'v36_correlation_long_bias';V38=HERE.parent/'v38_position_exits';V29=HERE.parent/'v29_nonlinear_portfolio';V28=HERE.parent/'v28_direct_portfolio';V23=HERE.parent/'v23_horizon_targets'
END='2026-11-11';BOUNDARY='2026-11-12';START='2024-01-02'
INDICES={'XLK':'.SPLRCT','XLF':'.SPSY','XLE':'.SPNY','XLV':'.SPXHC','XLI':'.SPLRCI','XLY':'.SPLRCD','XLP':'.SPLRCS','XLU':'.SPLRCU','XLB':'.SPLRCM','XLC':'.SPLRCL','XLRE':'.SPLRCR'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(value,indent=2,default=str),encoding='utf-8');t.replace(p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
