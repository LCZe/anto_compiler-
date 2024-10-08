import os
import subprocess
import warnings
from typing import List
from backports import lzma
from Dump import lldbAutoRun_spec as lldb_run
from Similarity import similarity as sim
from itertools import product


class CompilerModel:
    def __init__(self):
        self.errInsize = False
        self.errInrun = False
        self.workpath = ""
        self.pPath = ""
        self.outpath = ""
        self.compile_len = 200       # 优化选项的长度
        self.tempResult = ""
        self.result = ""
        self.errorIn = False
        self.type = ""
        self.cooperToolPath = '/home/cooper-Tools/cmake-build-debug/cooper_Tools'
        self.clangPath = '/home/software/llvm10.0/build/bin/clang'
        self.clangOpt = '/home/software/llvm10.0/build/bin/opt'
        self.dataPath = 'dataSet/test.bc'
        self.tempPath = 'runTemp/'
        self.outputPath = 'runTemp/output'
        self.rawPath = 'runTemp/raw.o'
        self.compile_option = []
        self.compile_ollvm = []
        self.compile_opt = []
        # ncd reward: prediferent:上一个保护中程序的ncd差异；difference：当前保护中程序的ncd差异
        self.prediferent = 0.0
        self.difference = 0.0
        # dump reward: 两个程序内存动态相似性，值越高相似性越高  ||  predump：上一个混淆程序的相似性   ||   dump：当前混淆程序的相似性
        self.predump = 1.0
        self.dump = 1.0

        # 目前迭代中奖励的最大值
        self.TheBast = 1.0
        self.step = 0
        self.obs = []
        self.actionSpace = ['-adce', '-aggressive-instcombine', '-alignment-from-assumptions',
                            '-always-inline', '-argpromotion', '-barrier', '-bdce', '-break-crit-edges',
                            '-simplifycfg', '-callsite-splitting', '-called-value-propagation',
                            '-consthoist', '-constmerge', '-constprop', '-correlated-propagation', '-cross-dso-cfi',
                            '-deadargelim', '-dce', '-die',
                            '-dse', '-reg2mem', '-div-rem-pairs', '-early-cse-memssa', '-early-cse',
                            '-elim-avail-extern',
                            '-ee-instrument', '-flattencfg', '-float2int', '-forceattrs', '-inline',
                            '-insert-gcov-profiling',
                            '-gvn-hoist', '-gvn', '-globaldce', '-globalopt', '-globalsplit', '-guard-widening',
                            '-hotcoldsplit', '-ipconstprop', '-ipsccp', '-indvars', '-irce', '-infer-address-spaces',
                            '-inferattrs', '-inject-tli-mappings', '-instsimplify', '-instcombine', '-instnamer',
                            '-jump-threading', '-lcssa', '-licm', '-libcalls-shrinkwrap', '-load-store-vectorizer',
                            '-loop-data-prefetch', '-loop-deletion', '-loop-distribute', '-loop-fusion',
                            '-loop-guard-widening',
                            '-loop-idiom', '-loop-instsimplify', '-loop-interchange', '-loop-load-elim',
                            '-loop-predication',
                            '-loop-reroll', '-loop-rotate', '-loop-simplifycfg', '-loop-simplify', '-loop-sink',
                            '-loop-reduce',
                            '-loop-unroll-and-jam', '-loop-unroll', '-loop-unswitch', '-loop-vectorize',
                            '-loop-versioning-licm',
                            '-loop-versioning', '-loweratomic', '-lower-constant-intrinsics', '-lower-expect',
                            '-lower-guard-intrinsic', '-lowerinvoke', '-lower-matrix-intrinsics', '-lowerswitch',
                            '-lower-widenable-condition', '-memcpyopt', '-mergefunc', '-mergeicmps', '-mldst-motion',
                            '-sancov', '-name-anon-globals', '-nary-reassociate', '-newgvn', '-pgo-memop-opt',
                            '-partial-inliner',
                            '-partially-inline-libcalls', '-post-inline-ee-instrument', '-functionattrs', '-mem2reg',
                            '-prune-eh', '-reassociate', '-rpo-functionattrs', '-rewrite-statepoints-for-gc',
                            '-sccp', '-slp-vectorizer', '-sroa', '-scalarizer', '-separate-const-offset-from-gep',
                            '-simple-loop-unswitch', '-sink', '-speculative-execution', '-slsr',
                            '-strip-dead-prototypes',
                            '-strip-nondebug', '-strip', '-tailcallelim', '-mergereturn',
                            '-stop' ]
        self.vmpPass = ['-irvmpvp', '-irvmp', '-irvmpall', '-irvmponlybtc', '-irvmponlyf3',
                            '-irvmponlyfast', '-irvmponlyn2', '-irvmponlyn3', '-irvmponlyf2', '-irvmpn2f3']
        self.functionName = []

    def taskInit(self, datapath, type, workpath, outpath):
        self.workpath = workpath
        self.type = type
        self.dataPath = datapath
        self.outpath = outpath
        # 根据任务类型初始化原程序
        self.TheBast = 10.0
        self.difference = 0.0   # ncd 0 表示这组程序完全相同
        self.dump = 1.0         # dump 1 表示这组程序完全相同
        # 生成保护前的可执行程序 =====
        pName = workpath[:-3] + '_none'     # 无混淆的可执行程序的名字
        # print(pName)
        self.actionSpaceGenerator()
        self.pPath = os.path.join(outpath, workpath[:-3])    # 当前程序的所有文件都保存在这儿
        if not os.path.exists(self.pPath):
            os.mkdir(self.pPath)
        os.system(self.clangPath + ' ' + datapath + ' -o ' + os.path.join(self.pPath, pName) + ' -lselinux -pthread -lstdc++ -lm')
        # 内存转储，arg1：可执行程序路径，agr2：视频输出路径
        lldb_run.toDump(os.path.join(self.pPath, pName), os.path.join(self.pPath, "video"))
        print("#####  完成基准程序  #####")
        if self.type == 'dump':
            self.obs = [self.dump]
        else:
            self.obs = [self.difference]

    def actionSpaceGenerator(self):
        """
        行为空间生成器，根据输入的IR文件生成对应的保护选项
        :return:
        """
        cmd: List[str] = [str(self.cooperToolPath), str(self.dataPath),str('5')]
        try:
            process = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
            output = process.stdout.decode()
            funcName = output.split()
            self.functionName = funcName
            for i in range(self.vmpPass.__len__()):
                for j in range(self.functionName.__len__()):
                    self.actionSpace.append(str(i)+":"+str(j))
        except subprocess.CalledProcessError as err:
            warnings.warn(f"Process errored out with {err} for {' '.join(cmd)}", UserWarning)


    def conver_actionToOption(self, action):
        self.compile_opt.clear()

        for i in range(self.compile_len):
            # stop 变长优化序列
            act_arg = self.actionSpace[action["opt" + i.__str__()]]

            if act_arg == "-stop":
                break
            if ":" in act_arg:  # 对于VMP
                indexVM = int(act_arg.split(":")[0])
                indexFunc = int(act_arg.split(":")[1])
                if self.vmpPass[indexVM] in self.compile_opt:
                    continue
                else:
                    self.compile_opt.append(self.vmpPass[indexVM])
                    if indexVM==0:
                        self.compile_opt.append(" "+self.functionName[indexFunc])
                    elif indexVM==1:
                        self.compile_opt.append("-label_ir=" + self.functionName[indexFunc])
                    elif indexVM==2:
                        self.compile_opt.append("-label_ir_a=" + self.functionName[indexFunc])
                    elif indexVM==3:
                        self.compile_opt.append("-label_ir_b=" + self.functionName[indexFunc])
                    elif indexVM == 4:
                        self.compile_opt.append("-label_ir_f3=" + self.functionName[indexFunc])
                    elif indexVM == 5:
                        self.compile_opt.append("-label_ir_f=" + self.functionName[indexFunc])
                    elif indexVM == 6:
                        self.compile_opt.append("-label_ir_n2=" + self.functionName[indexFunc])
                    elif indexVM == 7:
                        self.compile_opt.append("-label_ir_n3=" + self.functionName[indexFunc])
                    elif indexVM == 8:
                        self.compile_opt.append("-label_ir_f2=" + self.functionName[indexFunc])
                    elif indexVM == 9:
                        self.compile_opt.append("-label_ir_n2f3=" + self.functionName[indexFunc])
            else:   # 对于优化选项没有那么多要求，记录就行
                self.compile_opt.append(self.actionSpace[action["opt" + i.__str__()]])

    def cmd_compile(self):
        os.popen("ls")

    def run_opt(self):
        # 第二步：使用opt
        cmd0: List[str] = [str(self.clangOpt)]
        cmd0 += self.compile_opt
        args2: List[str] = [str(self.dataPath), str('-o'), str(os.path.join(self.pPath, self.workpath[:-3] + '.bc'))]
        cmd0 += args2
        try:
            print("再采用opt进行优化")
            print(f"opt for: {' '.join(cmd0)}")
            self.result += "2 ==> 再采用opt进行优化\n" + f"clang for: {' '.join(cmd0)}\n"
            process = subprocess.run(cmd0, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
            output = process.stdout.decode()
            print(output)
            if ("Error" in output) or ("clang-10: " in output) or ("LLVM ERROR:" in output):
                self.errorIn = True
                return
            else:
                self.errorIn = False
        except subprocess.TimeoutExpired:
            self.errorIn = True
            warnings.warn(f"Timeout for: {' '.join(cmd0)}", UserWarning)
        except subprocess.CalledProcessError as err:
            self.errorIn = True
            warnings.warn(f"Process errored out with {err} for {' '.join(cmd0)}", UserWarning)

        # 第三步：将混淆和优化后的bc编译为可执行程序
        cmd1: List[str] = [str(self.clangPath), str(os.path.join(self.pPath, self.workpath[:-3] + '.bc')), str('-o'),
                           str(os.path.join(self.pPath, self.workpath[:-3] + '_hx')), str('-lselinux'),str('-pthread'),str('-lstdc++'), str('-lm')]
        try:
            print("最后将混淆和优化后的bc编译为可执行程序")
            print(f"clang for: {' '.join(cmd1)}")
            self.result += "3 ==> 编译混淆版本可执行程序\n" + f"clang for: {' '.join(cmd1)}\n"
            self.tempResult = ' '.join(cmd1)
            process = subprocess.run(cmd1, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
            output = process.stdout.decode()
            print(output)
            os.system("rm " + str(os.path.join(self.pPath, self.workpath[:-3] + '.bc')))    # 删掉bc
            if ("Error" in output) or ("clang-10: " in output) or ("LLVM ERROR:" in output):
                self.errorIn = True
                return
            else:
                self.errorIn = False
            print("完成混淆版本的编译~~")
            print("开始转储混淆版本内存~~")
            dumperr = lldb_run.toDump(os.path.join(self.pPath, self.workpath[:-3] + '_hx'), os.path.join(self.pPath, "video"))

            if not dumperr:     # dumperr为0，表示dump失败了
                self.errorIn = True
                warnings.warn(f"dump出问题了~")
        except subprocess.TimeoutExpired:
            self.errorIn = True
            warnings.warn(f"Timeout for: {' '.join(cmd1)}", UserWarning)
        except subprocess.CalledProcessError as err:
            self.errorIn = True
            warnings.warn(f"Process errored out with {err} for {' '.join(cmd1)}", UserWarning)

        print("#####  完成混淆版本  #####")

    def ncd_count(self):
        wopen = open(os.path.join(self.pPath, self.workpath[:-3]) + '_hx', 'rb')  # 混淆后
        zopen = open(os.path.join(self.pPath, self.workpath[:-3]) + '_none', 'rb')  # 原程序
        W = wopen.read()
        Z = zopen.read()

        ncBytesXY = len(lzma.compress(W + Z))
        ncBytesX = len(lzma.compress(W))
        ncBytesY = len(lzma.compress(Z))
        ncd = float(ncBytesXY - min(ncBytesY, ncBytesX)) / max(ncBytesY, ncBytesX)
        self.difference = ncd

    def dump_count(self):       # 计算相似性
        self.dump = sim.getSim(os.path.join(self.pPath, "video"))

    def updateObs(self):
        if self.type == 'dump':
            self.obs = [self.dump]
        else:
            self.obs = [self.difference]

    def compiler_timestep(self, action):
        self.conver_actionToOption(action)
        # 保存前一步NCD信息
        self.prediferent = self.difference
        # 保存前一步相似度信息
        self.predump = self.dump
        self.errorIn = False
        # 生成保护后的bc，生成保护后的可执行程序
        self.run_opt()
        print("#####  动态特征提取结束  #####")
        print(self.errorIn)
        # 编译出错时，直接返回调用，更新状态，给负反馈
        if self.errorIn:
            self.step += 1
            self.difference = 0.0
            self.dump = 1.0
            self.updateObs()
            self.result += "NCD: " + str(self.difference) + '\n' + "Dump: " + str(self.dump) + '\n'
            return self.obs
        else:
            self.ncd_count()
            print("Ncd value: ", self.difference)
            print("preNcd value: ", self.prediferent)
            self.dump_count()
            print("dump value: ", self.dump)
            print("predump value: ", self.predump)
            self.step += 1
            self.updateObs()
            self.result += "NCD: " + str(self.difference) + '\n' + "Dump: " + str(self.dump) + '\n'
            return self.obs


if __name__ == "__main__":
    modle = CompilerModel()
    for i in range(0, len(modle.actionSpace)):
        modle.compiler_timestep(i)