import numpy as np
from diffskill.gen_init_target.state_generator import StateGenerator
from functools import partial


class LiftspreadGenerator(StateGenerator):
    def __init__(self, *args, **kwargs):
        super(LiftspreadGenerator, self).__init__(*args, **kwargs)
        self.N = 10
        self.xs = np.linspace(0.32, 0.38, self.N)  # Dough location on the cutting board
        self.xs_target = np.linspace(0.25, 0.35, self.N)  # Target location of the box dough
        self.rs = np.linspace(0.04, 0.07, self.N)

    def randomize_tool_state(self, cfg, pin_min_y):
        pos = eval(cfg.PRIMITIVES[0]['init_pos'])
        height = np.random.rand() * 0.1 + 0.04
        x_shift = np.random.rand() * 0.1 - 0.08
        cfg.PRIMITIVES[0]['init_pos'] = (pos[0] + x_shift, float(pin_min_y + height), pos[2])

    def case1(self, cfg, i, j):  # Sphere
        pos = eval(cfg.SHAPES[0]['init_pos'])
        cfg.SHAPES[0]['radius'] = self.rs[j]
        new_pos = (self.xs[i], self.rs[j] + 0.1, pos[2])
        cfg.SHAPES[0]['init_pos'] = new_pos

        self.randomize_tool_state(cfg, self.rs[j] * 2 + 0.1)

    def case11(self, cfg, i, j):  # Sphere on initial place
        pos = eval(cfg.SHAPES[0]['init_pos'])
        cfg.SHAPES[0]['shape'] = 'sphere'
        cfg.SHAPES[0]['radius'] = self.rs[j]
        new_pos = (pos[0], self.rs[j] + 0.03, pos[2])
        cfg.SHAPES[0]['init_pos'] = new_pos

        self.randomize_tool_state(cfg, self.rs[j] * 2 + 0.1)

    def case3(self, cfg, i, j): # Target
        cfg.SHAPES[0]['shape'] = 'box'
        w = np.linspace(0.26, 0.3, self.N)
        h = np.linspace(0.025, 0.04, self.N)
        w_id, h_id = np.random.randint(0, self.N), np.random.randint(0, self.N)
        pos = eval(cfg.SHAPES[0]['init_pos'])
        new_pos = (self.xs_target[i], h[h_id] + 0.1, pos[2])
        new_size = (w[w_id], h[h_id], 0.16)
        cfg.SHAPES[0]['init_pos'] = new_pos
        cfg.SHAPES[0]['width'] = new_size
        del cfg.SHAPES[0]['radius']
        self.randomize_tool_state(cfg, 0.19)

    def _generate(self):
        for case_id, case in enumerate([self.case1, self.case11]):
            for i in range(self.N):
                for j in range(self.N):
                    self.env.reset(target_cfg_modifier=partial(case, i=i, j=j))
                    self.save_init(idx=i * self.N + j + self.N * self.N * case_id)

        np.random.seed(0)
        for case_id, case in enumerate([self.case1, self.case3]):
            for i in range(self.N):
                for j in range(self.N):
                    self.env.reset(target_cfg_modifier=partial(case, i=i, j=j))
                    self.save_target(idx=i * self.N + j + self.N * self.N * case_id)
