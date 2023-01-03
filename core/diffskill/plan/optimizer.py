import torch
import numpy as np


def nop(it, *a, **k):
    # For tqdm
    return it


class BasePlanner(object):
    def __init__(self, dim, num_inits):
        """ Planning for the continuous variables"""
        self.dim = dim
        self.num_inits = num_inits
        self.log_dict = None

    def optimize(self, init_val, batch_obj_func):
        raise NotImplementedError

    def reset_log(self):
        self.log_dict = None

    def log(self, info):
        if self.log_dict is None:
            self.log_dict = {key: [] for key in info}

        for key in info:
            x = info[key].detach().cpu().numpy() if isinstance(info[key], torch.Tensor) else info[key]
            self.log_dict[key].append(x)

    def get_log(self):
        return {key: np.array(val) for key, val in self.log_dict.items()}


class AdamPlanner(BasePlanner):
    def __init__(self, args, dim):
        super().__init__(dim, args.adam_sample)
        self.args = args

    def optimize(self, init_val, batch_loss_func, project_func=None, mask=None):
        sol = init_val.detach().clone()
        n = sol.shape[0]
        dimu = self.args.dimz + 3
        sol.requires_grad = True
        optim = torch.optim.Adam(params=[sol], lr=self.args.adam_lr)
        self.reset_log()
        if self.args.adam_iter <= 1:
            tqdm = nop
        else:
            from tqdm import tqdm

        for i in tqdm(range(self.args.adam_iter), desc='Adam during planning'):
            # Projection to the constraint set
            if project_func is not None:
                with torch.no_grad():
                    if mask is not None:
                        sol[torch.where(mask == 0)] = init_val[torch.where(mask == 0)]
                    projected_sol = project_func(sol)
                    sol.copy_(projected_sol)
            # with torch.no_grad():
            #     sol += torch.randn_like(sol) * 0.0
            losses, info = batch_loss_func(sol)
            # Adding best plans during Adam Optimization
            if self.args.visualize_adam_pc:
                interm_sol = sol.view(n, -1, dimu)
                info['his_t'] = interm_sol[:, :, -3:]
                # if not self.args.freeze_z:
                info['his_z'] = interm_sol[:, :, :-3]
            info['losses'] = losses
            loss = losses.sum()
            self.log(info)
            if i != self.args.adam_iter - 1:
                optim.zero_grad()
                loss.backward()
                optim.step()
        return sol, self.get_log()

    def collate_fn(self, list_ret):
        # Used for combining batches of results
        sols = []
        logs, collated_logs = {}, {}
        for ret in list_ret:
            sol, log = ret
            sols.append(sol)
            for key, val in log.items():
                if key not in logs:
                    logs[key] = [np.array(val)]
                else:
                    logs[key].append(np.array(val))
        for key in logs.keys():
            if key in ['his_t', 'his_z']:
                collated_logs[key] = np.concatenate(logs[key], axis=1)
            else:
                collated_logs[key] = np.concatenate(logs[key], axis=-1)
        ## collate set_trajs @Xingyu can you check if this makes sense?
        if self.args.train_set:
            ret_traj = sols[0]
            H = len(ret_traj.struct_u)
            for traj in sols[1:]:
                for h in range(H):
                    ret_traj.struct_u[h] = torch.cat([ret_traj.struct_u[h], traj.struct_u[h]], dim=0)
            return ret_traj, collated_logs
        return torch.cat(sols, dim=0), collated_logs


class DerivativeFreePlanner(BasePlanner):
    def __init__(self, args, dim):
        super(DerivativeFreePlanner, self).__init__(dim)
        self.args = args

    def optimize(self, init_val, batch_loss_func, project_func=None):
        assert len(init_val.shape) == 2 and init_val.shape[0] == self.dim
        sol = init_val.detach().clone()
        optim = torch.optim.Adam(params=[sol], lr=self.args.adam_lr)
        sol.requires_grad = True
        from tqdm import tqdm
        for i in tqdm(range(self.args.adam_iter), desc='Adam during planning'):
            # Projection to the constraint set
            if project_func is not None:
                with torch.no_grad():
                    projected_sol = project_func(sol)
                    sol.copy_(projected_sol)
            losses = batch_loss_func(sol)
            loss = losses.sum()
            if i == self.args.adam_iter - 1:
                optim.zero_grad()
                loss.backward()
                optim.step()
