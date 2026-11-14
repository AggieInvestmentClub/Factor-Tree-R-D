"""Economic-loss, risk, causality and temporal-selection tests."""
from core import *
import unittest
import pandas as pd
from fit_policy import choose_dates

class PolicyTests(unittest.TestCase):
    def test_economic_gradient(self):
        rng=np.random.default_rng(44);k=5;a=rng.normal(size=(k,k));q=a.T@a*.001
        mu=rng.normal(size=k)*.02;d=rng.normal(size=(150,k))*.02;x=rng.normal(size=k)*.05
        _,g=objective(x,mu,q,d,.01,FEE,12)
        delta=1e-7;numeric=[]
        for j in range(k):
            v=np.eye(k)[j]*delta
            numeric.append((objective(x+v,mu,q,d,.01,FEE,12)[0]-objective(x-v,mu,q,d,.01,FEE,12)[0])/(2*delta))
        np.testing.assert_allclose(g,numeric,rtol=1e-5,atol=1e-8)
    def test_higher_cost_reduces_turnover_in_fixed_problem(self):
        mu=np.array([.04,.035]);q=np.eye(2)*.01;d=np.tile([[1.,0.],[0.,.05]],(20,1))
        x,_=solve(mu,q,d,.01,0,20);z,_=solve(mu,q,d,.01,.01,20)
        self.assertLess(abs(d@z).sum(),abs(d@x).sum())
    def test_risk_penalty_changes_exposure(self):
        mu=np.array([.02]);d=np.array([[1.]])
        a,_=solve(mu,np.array([[.01]]),d,.001,0,12)
        b,_=solve(mu,np.array([[.04]]),d,.001,0,12)
        self.assertGreater(a[0],b[0])
    def test_basis_and_signed_combinations_feasible(self):
        rng=np.random.default_rng(55);n=70;k=9
        x=rng.normal(size=(63,n))*.01;x-=x.mean(axis=0)
        beta=rng.normal(1,.3,n);values=rng.normal(size=(n,k));mask=np.ones(n,bool)
        b,q,active=neutral_basis(values,beta,mask,x,np.arange(n))
        theta=projected_l1(rng.normal(size=k));w=b@theta
        self.assertLess(abs(w.sum()),1e-10);self.assertLess(abs(w@beta),1e-10)
        self.assertLessEqual(abs(w).sum(),3*n/500+1e-9)
        self.assertLessEqual(abs(w).max(),.15+1e-9);self.assertLessEqual(theta@q@theta,.12**2+1e-9)
    def test_near_constant_flag_does_not_amplify_neutrality_error(self):
        rng=np.random.default_rng(7);n=500;x=rng.normal(size=(63,n))*.01;x-=x.mean(axis=0)
        values=np.ones((n,2));values[0,0]-=1e-6;beta=rng.normal(1,.3,n)
        b,q,mask=neutral_basis(values,beta,np.ones(n,bool),x,np.arange(n))
        self.assertLess(abs(b.sum(axis=0)).max(),1e-10)
        self.assertLess(abs(beta@b).max(),1e-10)
        np.testing.assert_array_equal(b[:,1],0.)
    def test_strict_maturity_and_window(self):
        d=dict(dates=pd.to_datetime(['2010-01-01','2011-11-01','2011-12-01']),
            ends=pd.to_datetime(['2010-02-01','2011-12-01','2012-01-01']),complete=np.array([1,1,1],bool))
        np.testing.assert_array_equal(choose_dates(d,pd.Timestamp('2012-01-01'),pd.Timestamp('2011-01-01')),[1])
    def test_gap_reset_and_no_zero_imputation_of_known_label(self):
        b=np.ones((3,2,1))*.1;r=np.array([[.1,.2],[.3,.4],[.5,.6]])
        data=dict(basis=b,returns=r,known=np.ones((3,2),bool),complete=np.ones(3,bool),risk=np.ones((3,1,1))*.01)
        _,_,d,stats=loss_inputs(data,[0,2],1)
        self.assertEqual(stats['cash_entry_resets'],2);np.testing.assert_allclose(d,.1)
        data['returns'][0,0]=np.nan
        with self.assertRaises(AssertionError):loss_inputs(data,[0,2],1)
    def test_causal_states_future_perturbation_and_gap_reset(self):
        rng=np.random.default_rng(42);a=rng.normal(size=(20,3,2));b=a.copy();b[10:]*=10000
        def build(x):
            state=np.full((2,3,2),np.nan);result=[]
            for t in range(20):
                mask=np.array([t!=5,True,True]);ix=np.flatnonzero(mask)
                update_states(state,x[t,ix],ix,mask,(.25,1-2**(-1/21)))
                result.append(state.copy())
            return np.array(result)
        sa,sb=build(a),build(b)
        np.testing.assert_array_equal(sa[:10],sb[:10])
        np.testing.assert_allclose(sa[6,:,0],np.tile(a[6,0],(2,1)))
        self.assertTrue(np.isnan(sa[5,:,0]).all())

if __name__=='__main__':unittest.main()
