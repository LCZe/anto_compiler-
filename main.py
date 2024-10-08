# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

from env.CompilerModle import CompilerModel
from env.llvm10Env import LlvmEnvironment
from utils import runner, create_agent
import argparse
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL')
    # 输入的是一个.bc的数据集
    parser.add_argument('-i', dest='input', metavar='In', help="input progrem path")
    parser.add_argument('-t', dest='type', metavar='type', default='dump', help="input progrem path")
    args = parser.parse_args()

    type = args.type  # ncd 和 dump
    inputPath = args.input
    os.path.join(inputPath)

    if not os.path.exists("result"):
        os.mkdir("result")

    num = len(os.listdir(inputPath))
    index = 1

    for file in os.listdir(inputPath):
        print("######################  " + str(index) + " / " + str(num) + "   " + file + "  ######################")
        index += 1

        datapath = os.path.join(inputPath, file)    # 输入程序路径
        workpath = file                             # bc文件名
        outpath = inputPath + '_out'
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        environment = LlvmEnvironment(datapath, workpath, type, outpath)
        agent = create_agent(environment=environment)
        runner(environment, agent, max_step_per_episode=100)
        with open("result/" + file[:-3] + ".txt", 'a', encoding='UTF-8') as f:
            f.write('\n' + environment.CompilerModel.result + '\n' + str(environment.rewardlist) + '\n' + str(environment.steplist))
