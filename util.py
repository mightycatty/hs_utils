def run_command(cmd, verbose=False):
    """
    run commands , Linux only
    :param cmd:
    :return: string output if success otherwise False
    """
    import subprocess
    try:
        output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0].decode("ascii")
        if verbose:
            print(output)
        return output
    except Exception as e:
        print(e)
        return False


# ====================================== gpu check utils =========================================================
def list_available_gpus(verbose=False):
    """
    list available gpus and return id list by running cmd "nvidia-smi" in Linux
    :param verbose:
    :return:
    """
    import re
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse " + line
        result.append(int(m.group("gpu_id")))
    if verbose:
        print(result)
    return result


def gpu_memory_map(verbose=False):
    import re
    """Returns map of GPU id to memory allocated on that GPU.
    working on ubuntu 16 with CUDA TOOLKIT10.2
    """
    output = run_command("nvidia-smi")
    gpu_output = output[output.find("Memory-Usage"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    gpu_id = 0
    for row in gpu_output.split("\n"):
        # m = memory_regex.search(row)
        if '%' in row:
            gpu_memory = int(row.split('MiB /')[0][-5:])
            result[gpu_id] += gpu_memory
            gpu_id += 1
        # if not m:
        #     continue
        # gpu_id = int(m.group("gpu_id"))
        # gpu_memory = int(m.group("gpu_memory"))
        # result[gpu_id] += gpu_memory
    if verbose:
        print(result)
    return result


def pick_n_gpu_lowest_memory(n=1):
    """Returns GPU with the least allocated memory"""
    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_gpu = []
    for item in sorted(memory_gpu_map)[:n]:
        if item[0] < 500:  # 500MB
            best_gpu.append(item[1])
    if len(best_gpu) == n:
        return best_gpu
    else:
        print('not enough gpus available')
        exit(0)


def set_gpus_visiable(gpu_num=1, verbose=True):
    import os
    best_gpu = pick_n_gpu_lowest_memory(gpu_num)
    if isinstance(best_gpu, list):
        assert len(best_gpu) >= gpu_num, 'not enough gpus found'
        gpu_id_str = str(best_gpu[0])
        for i in best_gpu[1:]:
            gpu_id_str += ',{}'.format(i)
            if (i + 2) >= gpu_num:
                break
    else:
        gpu_id_str = str(best_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_str
    if verbose:
        print('Using GPUs:{}'.format(gpu_id_str))
    return


def say_hi_to_your_program():
    """
    python program initialization, run before the main function in your python program
    1. clean up console, filter warnings
    :return:
    """
    import warnings
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action='ignore',
                          category=FutureWarning)  # disable nasty future warning in tensorflow and numpy


def disable_gpu():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


def select_n_gpus(num=1, max_load=0.1, max_mem=0.1):
    import os
    try:
        import GPUtil as gputil
    except:
        print('import GPUtil error, "pip install gputil" and try again')
    available_gpu_ids = gputil.getAvailable(order='first', limit=num, maxLoad=max_load, maxMemory=max_mem, includeNan=False,
                                            excludeID=[], excludeUUID=[])
    if len(available_gpu_ids) < num:
        print('not enough gpus found!')
        exit(0)
    else:
        if num == 0:
            gpu_str = '-1'
        else:
            gpu_str = ''
            for i in range(num):
                gpu_str += available_gpu_ids[i]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str



# ===============================================================================================================

def flat_list(list_input):
    """Flatten nested list"""
    list_flatten = []
    for item in list_input:
        if isinstance(item, list):
            list_flatten += flat_list(item)
        else:
            list_flatten += [item]
    return list_flatten
