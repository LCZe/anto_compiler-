import numpy as np
from tensorforce.environments import Environment
from env.CompilerModle import CompilerModel


class LlvmEnvironment(Environment):
    def __init__(self, datapath, workpath, type, outpath):
        super().__init__()
        self.workpath = workpath
        self.datapath = datapath
        self.outpath = outpath
        self.type = type
        self.ncdlist = []
        self.dumplist = []
        self.rewardlist = []
        self.steplist = []
        self.CompilerModel = CompilerModel()
        self.CompilerModel.taskInit(datapath, type, workpath, outpath)
        self.NUM_ACTIONS = len(self.CompilerModel.actionSpace)
        self.STATES_SIZE = len(self.CompilerModel.obs)     # obs是当前的观测值，智能体和环境每一次交互都会观察到环境当前的状态，这个状态就是这个obs
        self.finished = False
        self.max_step = 100  # 最大迭代次数，先改成10试试效果

    def states(self):
        return dict(type="float", shape=(self.STATES_SIZE,))

    def actions(self):
        # return dict(type="int", num_values=self.NUM_ACTIONS)
        action= {}
        for i in range(self.CompilerModel.compile_len):
            action["opt"+i.__str__()] = dict(type="int", num_values=self.NUM_ACTIONS)
        return action

    def max_episode_timesteps(self):
        return self.max_step

    def close(self):
        super().close()

    def reset(self, *args, **kwargs):           # 原本是这样 def reset(self):
        state = np.zeros(shape=(self.STATES_SIZE,))
        # self.CompilerModel = CompilerModel()
        return state

    def reward(self):
        reward = 0.0
        # 编译出错-运行时间超出限制-体积大小超出限制 直接给负反馈
        if self.CompilerModel.errorIn:
            # 编译错误时，给最大的负反馈
            reward = -10
            if self.type == 'dump':
                self.CompilerModel.dump = 1.0
            else:
                self.CompilerModel.difference = 0.0
        else:
            if self.type == 'dump':
                reward = (self.CompilerModel.predump - self.CompilerModel.dump) * 10
            else:
                reward = (self.CompilerModel.difference - self.CompilerModel.prediferent) * 10

            print("#####  reward  #####")

        self.rewardlist.append(reward)
        self.dumplist.append(self.CompilerModel.dump)
        self.ncdlist.append(self.CompilerModel.difference)
        # self.steplist.append(len(self.ncdlist))
        self.steplist.append(len(self.dumplist))
        print(reward)
        return reward

    def terminal(self):
        # 阈值设置问题
        self.finished = self.CompilerModel.step == self.max_step

        # 更新最优结果
        if self.type == 'dump':
            if self.CompilerModel.TheBast > self.CompilerModel.dump:  # dump值越小越好
                self.CompilerModel.TheBast = self.CompilerModel.dump
                self.CompilerModel.result += self.CompilerModel.tempResult + "\n dump value:" + str(self.CompilerModel.dump) + '\n'
                self.CompilerModel.result += "当前最好的结果是 ： " + str(self.CompilerModel.compile_opt) + '\n'
        else:
            if self.CompilerModel.TheBast < self.CompilerModel.difference:  # ncd值越大越好
                self.CompilerModel.TheBast = self.CompilerModel.difference
                self.CompilerModel.result += self.CompilerModel.tempResult + "\n ncd value:" + str(self.CompilerModel.difference)
                self.CompilerModel.result += "当前最好的结果是 ： "  + str(self.CompilerModel.compile_opt) + '\n'
        return self.finished

    def execute(self, actions):
        print("______________________  当前轮次：" + str(self.CompilerModel.step) + " / " + str(self.max_step) + "   ______________")
        self.CompilerModel.result += "____ 当前轮次：" + str(self.CompilerModel.step) + " / " + str(self.max_step) + '\n'
        next_state = self.CompilerModel.compiler_timestep(actions)
        reward = self.reward()
        terminal = self.terminal()

        return next_state, terminal, reward
