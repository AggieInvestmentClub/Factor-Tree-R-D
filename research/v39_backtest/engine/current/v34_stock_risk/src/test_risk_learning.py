from risk_learning import *
import unittest

class Tests(unittest.TestCase):
    def test_joint_sector_projection(self):
        rng=np.random.default_rng(3);n=80;mask=np.ones(n,bool);mask[-5:]=False;b=rng.uniform(.4,1.8,n)*mask;groups=np.arange(n)%7
        rows=np.vstack([mask.astype(float),b,[(groups==k)*mask for k in range(12)]]);u,rank=orthogonal_rows(rows,mask,14)
        w=rng.normal(size=n)*mask
        for _ in range(2):w-=u.T@(u@w)
        self.assertLess(abs(rows@w).max(),1e-10);self.assertLess(abs(w[~mask]).max(),1e-12)
    def test_correlation_factor_projection(self):
        rng=np.random.default_rng(4);n=80;market=rng.normal(0,.01,63);x=rng.normal(0,.01,(63,n))+market[:,None]*rng.uniform(.3,1.8,n);x-=x.mean(0)
        loads,fraction=correlation_loadings(x,market);self.assertEqual(loads.shape,(3,n));self.assertTrue((fraction>0).all());self.assertLessEqual(fraction.sum(),1+1e-12)
        mask=np.ones(n,bool);b=rng.uniform(.3,1.8,n);rows=np.vstack([mask,b,loads]);u,rank=orthogonal_rows(rows,mask,5);w=rng.normal(size=n)
        for _ in range(2):w-=u.T@(u@w)
        self.assertLess(abs(rows@w).max(),1e-10)
    def toy(self):
        rng=np.random.default_rng(6);t,n=3,80;b=rng.uniform(.4,1.7,(t,n));x=rng.normal(0,.01,(t,63,n));x-=x.mean(1,keepdims=True);u=np.stack([np.linalg.svd(np.vstack([np.ones(n),z]),full_matrices=False)[2] for z in b])
        d={k:torch.tensor(v) for k,v in dict(x=rng.normal(size=(t,n,58)),centered=x,diag=np.sum(x*x,axis=1)/62,orth=u,mask=np.ones((t,n)),beta=b,gate=np.ones(t),scale=np.ones((t,n)),returns=rng.normal(0,.04,(t,n))).items()};d['breaks']=np.array([1,0,0],bool);d['end_segment']=np.array([0,0,1],bool);return d
    def test_unchanged_allocator_identity(self):
        d=self.toy();m=Policy('replay_random');np.testing.assert_array_equal(weights(m,d).detach().numpy(),base.weights(m,d).detach().numpy())
    def test_risk_scaled_gradient(self):
        d=self.toy();d['scale'][:]=torch.linspace(.5,2,80);m=Policy('replay_random');loss=base.loss(weights(m,d),d,'log_growth',m,.01);loss.backward();g=float(m.linear.weight.grad[0,4]);eps=1e-6
        with torch.no_grad():m.linear.weight[0,4]+=eps;a=float(base.loss(weights(m,d),d,'log_growth',m,.01));m.linear.weight[0,4]-=2*eps;b=float(base.loss(weights(m,d),d,'log_growth',m,.01))
        self.assertAlmostEqual(g,(a-b)/(2*eps),places=5)

if __name__=='__main__':unittest.main()
