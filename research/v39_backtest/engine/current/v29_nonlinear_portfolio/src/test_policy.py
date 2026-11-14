from common import *
from policy import *
import unittest
import account_engine

class Tests(unittest.TestCase):
    def toy(self):
        rng=np.random.default_rng(9);t,n=3,50
        b=rng.normal(1,.2,(t,n));mask=np.ones((t,n));x=rng.normal(0,.01,(t,63,n));x-=x.mean(1,keepdims=True)
        u=np.stack([np.linalg.svd(np.vstack([np.ones(n),bb]),full_matrices=False)[2] for bb in b])
        d={k:torch.tensor(v) for k,v in dict(x=rng.normal(size=(t,n,58)),mask=mask,centered=x,diag=(x*x).sum(1)/62,orth=u,beta=b,gate=np.ones(t),returns=rng.normal(0,.03,(t,n))).items()}
        d['breaks']=np.array([1,0,0],bool);d['end_segment']=np.array([0,0,1],bool);return d
    def test_constraints_and_gradient(self):
        d=self.toy()
        for kind in MODELS:
            m=Policy(kind);w=weights(m,d)
            self.assertLess(float(w.sum(1).abs().max()),1e-10)
            self.assertLess(float((w*d['beta']).sum(1).abs().max()),1e-10)
            self.assertLessEqual(float(variance(w,d).max()),.12**2+1e-10)
            self.assertLessEqual(float(w.abs().sum(1).max()),.3+1e-10)
            self.assertLessEqual(float(w.abs().max()),.15+1e-10)
            f=replay(w,d,.01,m);f.backward();g=float(m.linear.weight.grad[0,4]);eps=1e-6
            with torch.no_grad():
                m.linear.weight[0,4]+=eps;a=float(replay(weights(m,d),d,.01,m))
                m.linear.weight[0,4]-=2*eps;b=float(replay(weights(m,d),d,.01,m))
            self.assertAlmostEqual(g,(a-b)/(2*eps),places=5)
    def test_post_fee_fixed_point_source(self):
        old=np.array([.4,-.3]);target=np.array([.5,-.5]);q=1.
        for _ in range(10):q=1-.0002*np.abs(q*target-old).sum()
        expected=account_engine._old._post_fee_values(1.,old,target,.0002)
        np.testing.assert_allclose(q*target,expected,atol=1e-12)
    def test_buffer_and_gate(self):
        d=self.toy();w=torch.zeros((3,50));w[:,0]=.05;w[:,1]=-.05
        self.assertEqual(replay(w,d,details=True)['trades'],0)
        d['gate']*=0;np.testing.assert_array_equal(weights(Policy(MODELS[0]),d).detach().numpy(),0)
    def test_purge(self):
        d=dict(dates=pd.to_datetime(['2011-11-01','2011-12-01']),ends=pd.to_datetime(['2011-12-01','2012-01-01']),complete=np.ones(2,bool))
        np.testing.assert_array_equal(selected(d,pd.Timestamp('2012-01-01'),pd.Timestamp('2011-01-01')),[0])
    def test_single_period_ledger_parity(self):
        clock=pd.bdate_range('2010-01-01',periods=23);names=[f'S{i}' for i in range(50)]
        rr=np.r_[np.full(25,.04),np.full(25,-.02)];w=np.r_[np.full(25,.005),np.full(25,-.005)]
        prices={'SPY':pd.DataFrame(dict(adj_close=100.,adj_open=100.,q=1),index=clock)}
        for n,r in zip(names,rr):
            price=np.r_[100.,100.,100*(1+r)**(np.arange(1,22)/21)]
            prices[n]=pd.DataFrame(dict(adj_close=price,adj_open=price,q=1),index=clock)
        targets=pd.DataFrame([w],index=clock[:1],columns=names);beta=pd.DataFrame(1.,index=clock,columns=names)
        ledger,*_=account_engine.simulate_stock_portfolio(prices,targets,start=clock[1],end=clock[-1],cost_bps=2,buffer_l1=.2,risk_betas=beta,position_cap=.15,gross_cap=3,gross_guard=3.3,record_positions=False)
        d=self.toy();d={k:(v[:1] if isinstance(v,torch.Tensor) else np.ones(1,bool)) for k,v in d.items()};d['beta'][:]=1;d['returns']=torch.tensor(rr[None,:])
        out=replay(torch.tensor(w[None,:]),d,details=True)
        self.assertAlmostEqual(out['mean_return']/12,ledger.nav.iloc[-1]-1,places=11)
if __name__=='__main__':unittest.main()
