import numpy as np
from core.traj_opt.gen_init_target.state_generator import StateGenerator
import copy


def cut(state, cut_loc, slide_a, slide_b):
    ns = copy.deepcopy(state)
    flag = ns['state'][0][:, 0] <= cut_loc
    ns['state'][0][flag, 0] -= slide_a
    ns['state'][0][np.logical_not(flag), 0] += slide_b
    return ns, flag


def move_cluster(state, flag, dx, dz, dy=0):
    # need to move forward
    new_state = copy.deepcopy(state)
    mean = new_state['state'][0][flag].mean(axis=0)
    new_state['state'][0][flag] += np.array([dx, mean[1] + dy, dz]) - mean
    return new_state


class CutrearrangeGenerator(StateGenerator):
    def __init__(self, *args, **kwargs):
        super(CutrearrangeGenerator, self).__init__(*args, **kwargs)
        self.env.reset()
        for i in range(50):
            self.env.step(np.array([0, 0, 0] + [0, 0, 0] + [0, 0, 0, 0]))
        self.initial_state = self.env.get_state()
        self.N = 1

    def _generate(self):
        for i in range(self.N):
            def rand(a, b):
                return np.random.random() * (b - a) + a

            state = copy.deepcopy(self.initial_state)
            width = np.array([rand(0.2, 0.24), 0.08, rand(0.04, 0.08)])
            state['state'][0] = (np.random.random((len(state['state'][0]), 3)) * 2 - 1) * (0.5 * width) + np.array([0.5, 0.0669, 0.5])

            da = db = 0.
            if np.random.randint(2):
                da = rand(0.1, 0.13)
            else:
                db = rand(0.1, 0.13)

            cut_loc = rand(0.48, 0.52)
            cutted, flag = cut(state, cut_loc, da, db)

            def sample_target():
                # make sure that it is moved away
                y = rand(0.3, 0.35) * (np.random.randint(2) * 2 - 1) + 0.5
                x = rand(0.3, 0.7)
                return (x, y)

            while True:  # sample two targets which are far away
                a = sample_target()
                b = sample_target()
                if np.linalg.norm(np.array(a) - np.array(b)) >= 0.3:
                    break

            move1 = move_cluster(cutted, flag, *a)
            move2 = move_cluster(move1, np.logical_not(flag), *b)

            self.save_init(3 * i, state)
            self.save_init(3 * i + 1, cutted)
            self.save_init(3 * i + 2, move1)
            self.save_target(3 * i, cutted)
            self.save_target(3 * i + 1, move1)
            self.save_target(3 * i + 2, move2)
