from controller import *
import unittest

class Tests(unittest.TestCase):
    def test_regime_causality_and_rules(self):
        ix=pd.bdate_range('2010-01-01',periods=400);rng=np.random.default_rng(8);close=100*np.cumprod(1+rng.normal(.0003,.01,len(ix)))
        spy=pd.DataFrame(dict(adj_close=close,q=1,tr_segment_id=1),index=ix);before=regime(spy);future=spy.copy();future.loc[ix[300]:,'adj_close']*=4
        pd.testing.assert_frame_equal(before.loc[:ix[299]],regime(future).loc[:ix[299]])
        bull=requested(pd.Series(dict(factor=.25,state=1)));self.assertEqual(bull['vol_target'],.125);self.assertGreater(bull['trend_vol_signed'],0)
        bear=requested(pd.Series(dict(factor=1.,state=-1)));self.assertEqual(bear['trend_vol_long'],0);self.assertEqual(bear['trend_vol_signed'],-.25)
    def test_signed_joint_budget_no_refill(self):
        rng=np.random.default_rng(2);n=500;b=rng.uniform(.4,1.8,n);mask=np.ones(n,bool);x=rng.normal(0,.02,(63,n));x-=x.mean(0);ix=np.arange(n)
        ref,info=target(rng.normal(size=n),b,mask,x,ix,.12,.5,1);unit=unit_market(b,mask);alpha=ref-unit*info['effective_bias']
        for request in (-.25,-.1,0,.1,.25,.5):
            w,delta,scale=combine(alpha,unit,x,ix,request,3)
            self.assertLessEqual(scale,1);self.assertAlmostEqual(w.sum(),delta,places=9);self.assertAlmostEqual(w@b,delta,places=9)
            self.assertLessEqual(annual_variance(w,x,ix),.12**2+1e-10);self.assertLessEqual(abs(w).sum(),3+1e-9);self.assertLessEqual(abs(w).max(),.15+1e-9)
            np.testing.assert_allclose(w-unit*delta,scale*alpha,atol=1e-12)
    def synthetic(self,short=False,missing=False):
        clock=pd.bdate_range('2010-01-01',periods=12);names=[f'S{i}' for i in range(50)];price=np.full(len(clock),100.);price[6:]=200.
        prices={n:pd.DataFrame(dict(adj_open=price,adj_close=price,q=1),index=clock) for n in names+['SPY']};beta=pd.DataFrame(1.,index=clock,columns=names);elig=pd.DataFrame(True,index=clock,columns=names)
        if missing:beta.loc[clock[3]:,names[0]]=np.nan;elig.loc[clock[3]:,names[0]]=False
        target=pd.DataFrame(np.full((1,50),-.005 if short else .005),index=clock[:1],columns=names)
        args=dict(start=clock[1],end=clock[-1],risk_betas=beta,eligibility=elig,neutral=False,beta_neutral=False,dollar_neutral=False,controlled_long_bias=True,repair_stale_risk=True,position_cap=.15,gross_cap=3,gross_guard=3.3,buffer_l1=.2,cost_bps=2,record_positions=False)
        return clock,prices,target,args
    def test_positive_engine_identity(self):
        clock,prices,t,args=self.synthetic();a=prior.engine.simulate_stock_portfolio(prices,t,**args);b=account_engine().simulate_stock_portfolio(prices,t,**args)
        for j in (0,2,3):pd.testing.assert_frame_equal(a[j],b[j])
    def test_negative_bias_guard_and_missing_risk(self):
        clock,prices,t,args=self.synthetic(short=True,missing=True);a=account_engine().simulate_stock_portfolio(prices,t,**args)
        self.assertTrue(a[4]['arithmetic_passed']);self.assertTrue(a[3].stale_risk_repaired.any());self.assertAlmostEqual(a[0].loc[clock[7],'net_exposure'],-.25,places=9);self.assertGreater(a[0].cash.min(),0)

if __name__=='__main__':unittest.main()
