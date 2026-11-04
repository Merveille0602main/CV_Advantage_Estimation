import tempfile
import unittest
from pathlib import Path
import numpy as np
from advantage import Episode, discounted_returns, n_step_targets, n_step_scores, td_targets, td_scores
from advantage.schema import split_families
from advantage.synthetic import generate, exact_values


def example(truncated=False):
    return Episode("e", "family", "task", "policy", np.array([[0.],[1.],[2.]]),
        np.array([[1.],[2.],[3.]]), np.zeros((3,1)), np.array([0.,0.,1.]),
        np.array([False,False,not truncated]), np.array([False,False,truncated]),
        np.array([5,4,3]) if truncated else np.array([3,2,1]))


class TargetTests(unittest.TestCase):
    def test_plan_hand_calculation(self):
        ep = example()
        np.testing.assert_allclose(discounted_returns(ep, .9), [.81,.9,1.])
        np.testing.assert_allclose(td_scores(ep, [.5,.6,.8], [.6,.8,99], .9), [.04,.12,.2])
        self.assertAlmostEqual(n_step_scores(ep, [.5,.6,.8], [.6,.8,99], 2, .9)[0], .148)

    def test_one_step_identity_and_terminal_mask(self):
        ep = example()
        np.testing.assert_array_equal(td_targets(ep, [2,3,1e9]), n_step_targets(ep, [2,3,1e9], 1))
        np.testing.assert_array_equal(n_step_targets(ep, [2,3,1e9], 100), discounted_returns(ep))

    def test_truncation_bootstraps_instead_of_zero(self):
        ep = example(True)
        with self.assertRaises(ValueError):
            discounted_returns(ep)
        expected = np.array([.81+.9**3*4, .9+.9**2*4, 1+.9*4])
        np.testing.assert_allclose(n_step_targets(ep, [2,3,4], 100, .9), expected)
        np.testing.assert_allclose(discounted_returns(ep,.9,4), expected)

    def test_randomized_reference(self):
        for ep in generate(25, 7):
            rng = np.random.default_rng(len(ep))
            vals = rng.normal(size=len(ep))
            for gamma in (0., .9, 1.):
                for n in (1,2,5,100):
                    expected=[]
                    for t in range(len(ep)):
                        end=min(t+n,len(ep)); steps=end-t
                        value=np.dot(gamma**np.arange(steps),ep.reward[t:end])
                        if not ep.terminated[end-1]:
                            value += gamma**steps*vals[end-1]
                        expected.append(value)
                    np.testing.assert_allclose(n_step_targets(ep,vals,n,gamma),expected,atol=1e-12)

    def test_invalid_parameters(self):
        for n in (0,-1,1.5,True):
            with self.assertRaises(ValueError): n_step_targets(example(), [0,0,0], n)
        for gamma in (-.1,1.1,np.nan):
            with self.assertRaises(ValueError): td_targets(example(), [0,0,0], gamma)
        with self.assertRaises(ValueError): td_targets(example(), [0,0])


class SchemaTests(unittest.TestCase):
    def test_roundtrip_without_pickle(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"episode.npz"; example().save(path); ep=Episode.load(path)
            np.testing.assert_array_equal(ep.reward, example().reward)
            self.assertEqual(ep.parent_id,"family")

    def test_boundaries_alignment_and_nan(self):
        for modify in (lambda e: setattr(e,"reward",np.array([0.,np.nan,1.])),
                       lambda e: e.terminated.__setitem__(0,True),
                       lambda e: e.next_obs.__setitem__((0,0),99),
                       lambda e: setattr(e,"terminated",np.array([0,0,1])),
                       lambda e: setattr(e,"remaining_steps",np.array([3,3,1]))):
            ep=example(); modify(ep)
            with self.assertRaises(ValueError): ep.validate()

    def test_split_preserves_families(self):
        eps=generate(40,2)
        eps[1].parent_id=eps[0].parent_id
        split=split_families(eps,5)
        assignment={eid:key for key,ids in split.items() for eid in ids}
        self.assertEqual(assignment[eps[0].episode_id],assignment[eps[1].episode_id])
        self.assertEqual(len(assignment),len(eps))
        self.assertEqual(split,split_families(eps,5))

    def test_oracle_bellman_identity(self):
        v,q=exact_values(.99)
        np.testing.assert_allclose(v[:-1,:5],.4*q[:,:,0]+.6*q[:,:,1])
        self.assertTrue(np.all((v>=0)&(v<=1)))


if __name__ == "__main__": unittest.main()
