

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
            print (output)
        return output
    except Exception as e:
        print (e)
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
        assert m, "Couldnt parse "+line
        result.append(int(m.group("gpu_id")))
    if verbose:
        print (result)
    return result


def gpu_memory_map(verbose=False):
    import re
    """Returns map of GPU id to memory allocated on that GPU."""
    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    if verbose:
        print (result)
    return result


def pick_n_gpu_lowest_memory(n=1):
    """Returns GPU with the least allocated memory"""
    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_gpu = [item[1] for item in sorted(memory_gpu_map)[:n]]
    return best_gpu

# ===============================================================================================================


