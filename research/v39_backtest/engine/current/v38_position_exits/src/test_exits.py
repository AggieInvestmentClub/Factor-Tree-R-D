from exits import *
import unittest
class Tests(unittest.TestCase):
    def synthetic(self):
        clock=pd.bdate_range('2010-01-01',periods=8);names=[f'S{i}' for i in range(200)];p=np.array([100,100,100,90,80,85,90,95.]);prices={n:pd.DataFrame(dict(adj_open=p,adj_close=p,q=1),index=clock) for n in names+['SPY']};beta=pd.DataFrame(1.,index=clock,columns=names);elig=pd.DataFrame(True,index=clock,columns=names);t=pd.DataFrame(.5/200,index=clock[:1],columns=names);kw=dict(start=clock[1],end=clock[-1],risk_betas=beta,eligibility=elig,neutral=False,dollar_neutral=False,beta_neutral=False,controlled_long_bias=True,repair_stale_risk=True,position_cap=.15,gross_cap=3,gross_guard=3.3,buffer_l1=.2,cost_bps=2,record_positions=False);return clock,prices,t,kw
    def test_disabled_and_noop_identity(self):
        clock,p,t,k=self.synthetic();original=runner.engine.simulate_stock_portfolio(p,t,**k);disabled=engine().simulate_stock_portfolio(p,t,**k);noop=engine().simulate_stock_portfolio(p,t,**k,position_exit_policy=lambda date,actual,base,beta,mask,initial:(base,{'force_exit':False}))
        for j in (0,2):pd.testing.assert_frame_equal(original[j],disabled[j]);pd.testing.assert_frame_equal(original[j],noop[j])
    def test_close_signal_next_close_gap(self):
        clock,p,t,k=self.synthetic()
        def policy(date,actual,base,beta,mask,initial):return (np.zeros(len(base)),{'force_exit':bool(abs(actual).sum()>1e-8)}) if date>=clock[3] else (base,{'force_exit':False})
        z=engine().simulate_stock_portfolio(p,t,**k,position_exit_policy=policy);self.assertGreater(z[0].loc[clock[3],'gross_exposure'],0);self.assertEqual(z[0].loc[clock[4],'gross_exposure'],0);self.assertLess(z[0].loc[clock[4],'nav'],.91);self.assertTrue(z[4]['arithmetic_passed'])
    def test_stop_and_profit_boundaries(self):
        p=np.array([-.06,-.01,.11]);t=np.array([.06,.09,0]);s=np.array([.04,.04,.04]);age=np.array([10,10,10]);np.testing.assert_array_equal(trigger('fixed5',p,t,s,age),[True,False,False]);np.testing.assert_array_equal(trigger('trailing8',p,t,s,age),[False,True,False]);np.testing.assert_array_equal(trigger('bracket5_10',p,t,s,age),[True,False,True])
    def test_tree_requires_holding_age_and_negative_advantage(self):
        hit=trigger('tree',np.zeros(3),np.zeros(3),np.ones(3)*.1,np.array([4,5,10]),np.array([-.01,-.01,.02]));np.testing.assert_array_equal(hit,[False,True,False])
if __name__=='__main__':unittest.main()
