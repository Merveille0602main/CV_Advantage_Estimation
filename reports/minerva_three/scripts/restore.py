"""Exact prefix replay with the two unused intermediate camera views skipped."""
import os
from types import SimpleNamespace

import numpy as np
from branch_eval import reset_history


def restore(sim, episode, frame, snapshot):
    # The policy history is supplied by the checkpoint. Only the last prefix image
    # is consumed. Physics, original controls and the final camera render stay live.
    if frame<2 or os.environ.get("MINERVA_RENDER_PREFIX")=="1" or getattr(sim,"render_full_prefix",False):
        return reset_history(sim,episode,frame,snapshot)
    blank=np.zeros_like(snapshot["current_image"])
    counter=0
    renderer=None
    render=None
    def step(action):
        nonlocal counter,renderer,render
        if counter==0:
            renderer=sim.base._env.env.sim
            render=renderer.render
            renderer.render=lambda *args,**kwargs: blank
        if counter==frame-1:
            renderer.render=render
        counter+=1
        return sim.env.step(action)
    proxy=SimpleNamespace(env=SimpleNamespace(reset=sim.env.reset,step=step),base=sim.base,
                          policy=sim.policy,prepare=sim.prepare)
    try:
        # reset_history checks both real camera images and the complete integration
        # state against the snapshot on every branch before restoring the policy RNG.
        return reset_history(proxy,episode,frame,snapshot)
    except (AssertionError,AttributeError):
        sim.render_full_prefix=True
    finally:
        if renderer is not None and render is not None:
            renderer.render=render
    return reset_history(sim,episode,frame,snapshot)
