import numpy as np

class MixedModel:
    def __init__(self, model_A, model_B, model_A_cond, model_B_cond, model_A_id2a, model_B_id2a):
        self.p = model_A
        self.convert = model_A_id2a
        self.initial_state = model_A.initial_state
        self.A = model_A
        self.B = model_B
        self.A_cond = model_A_cond
        self.B_cond = model_B_cond
        self.A_id2a = model_A_id2a
        self.B_id2a = model_B_id2a
        max_ac = max(model_A_id2a)
        max_bc = max(model_B_id2a)
        self.max_ac_index = max([max_ac, max_bc])


    def step(self, ob, *_args, **_kwargs):
        if self.A_cond(ob):
            self.p = self.A
            self.convert = self.A_id2a
        if self.B_cond(ob):
            self.p = self.B
            self.convert = self.B_id2a

        action, X, Y, Z = self.p.step(ob, *_args, **_kwargs)
        for i in range(action.shape[0]):
            action[i] = self.convert[action[i]]
            if np.random.random() < 0.1:
                action[i] = np.random.randint(self.max_ac_index + 1)

        return action, X, Y, Z


    def value(self, ob, *_args, **_kwargs):
        if self.A_cond(ob):
            p = self.A
        else:
            p = self.B
        return p.value(ob, *_args, **_kwargs)
