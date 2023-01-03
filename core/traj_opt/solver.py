from functools import partial
import numpy as np
from core.utils.diffskill_utils import get_component_masks
import core.utils.plb_utils as tu
import tqdm
import torch
# tu.set_default_tensor_type(torch.DoubleTensor)
from plb.engine.function import GradModel
from sklearn.cluster import DBSCAN
FUNCS = {}


class Solver:
    def __init__(self, args, env, ouput_grid=(), **kwargs):
        self.args = args
        self.env = env
        self.env.update_loss_fn(self.args.adam_loss_type)
        self.dbscan = DBSCAN(eps=0.01, min_samples=5)
        if env not in FUNCS:
            FUNCS[env] = GradModel(env, output_grid=ouput_grid, **kwargs)
        self.func = FUNCS[env]
        self.buffer = []

    def solve(self, initial_actions, loss_fn, action_mask=None, lr=0.01, max_iter=200, verbose=True, scheduler=None):
        # loss is a function of the observer ..
        if action_mask is not None:
            initial_actions = initial_actions * action_mask[None]
            action_mask = torch.FloatTensor(action_mask[None]).cuda()
        self.initial_state = self.env.get_state()
        buffer, info = self.solve_one_plan(initial_actions, loss_fn, action_mask=action_mask,
                                    lr=lr, max_iter=max_iter, verbose=verbose, scheduler=scheduler)
        self.buffer.append(buffer)
        return info, buffer


    def solve_one_plan(self, initial_actions, loss_fn, action_mask=None, lr=0.01, max_iter=200, verbose=True, scheduler=None):

        action = torch.nn.Parameter(tu.np2th(np.array(initial_actions)), requires_grad=True)
        optim = torch.optim.Adam([action], lr=lr)
        scheduler if scheduler is None else scheduler(self.optim)
        buffer = []
        best_action, best_loss = initial_actions, np.inf

        iter_id = 0
        ran = tqdm.trange if verbose else range
        it = ran(iter_id, iter_id + max_iter)

        loss, last = np.inf, initial_actions
        H = action.shape[0]
        for iter_id in it:
            optim.zero_grad()

            observations = self.func.reset(self.initial_state['state'])
            cached_obs = []
            for idx, i in enumerate(action):
                if H - idx <= self.args.stop_action_n:
                    observations = self.func.forward(idx, i.detach(), *observations)
                else:
                    observations = self.func.forward(idx, i, *observations)
                cached_obs.append(observations)

            loss = loss_fn(list(range(len(action))), cached_obs, self.args.vel_loss_weight, loss_type=self.args.adam_loss_type)
            loss.backward()
            optim.step()
            if scheduler is not None:
                scheduler.step()
            action.data.copy_(torch.clamp(action.data, -1, 1))
            if action_mask is not None:
                action.data.copy_(action.data * action_mask)

            with torch.no_grad():
                loss = loss.item()
                last = action.data.detach().cpu().numpy()
                if loss < best_loss:
                    best_loss = loss
                    best_action = last

            buffer.append({'action': last, 'loss': loss})
            if verbose:
                it.set_description(f"{iter_id}:  {loss}", refresh=True)

        self.env.set_state(**self.initial_state)
        return buffer, {
            'best_loss': best_loss,
            'best_action': best_action,
            'last_loss': loss,
            'last_action': last
        }

    def eval(self, action, render_fn):
        self.env.simulator.cur = 0
        initial_state = self.initial_state
        self.env.set_state(**initial_state)
        outs = []
        import tqdm
        for i in tqdm.tqdm(action, total=len(action)):
            self.env.step(i)
            outs.append(render_fn())

        self.env.set_state(**initial_state)
        return outs

    def plot_buffer(self, buffer=None):
        import matplotlib.pyplot as plt
        plt.Figure()
        if buffer is None:
            buffer = self.buffer
        for buf in buffer:
            losses = []
            for i in range(len(buf)):
                losses.append(buf[i]['loss'])
            plt.plot(range(len(losses)), losses)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.show()
    
    def save_plot_buffer(self, path, buffer=None):
        import matplotlib.pyplot as plt
        plt.Figure()
        if buffer is None:
            buffer = self.buffer
        for buf in buffer:
            losses = []
            for i in range(len(buf)):
                losses.append(buf[i]['loss'])
            plt.plot(range(len(losses)), losses)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.savefig(path)
        plt.close()

    def dump_buffer(self, path='/tmp/buffer.pkl'):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, path='/tmp/buffer.pkl'):
        import pickle
        with open(path, 'rb') as f:
            self.buffer = pickle.load(f)