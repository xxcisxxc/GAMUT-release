import subprocess, re, tempfile, os, timeit

import cupy as cp
import numpy as np

start = timeit.default_timer()
print("Start!")
p_row, p_column, p_depth = 4096, 4096, 4096

compiler = 'nvcc'
cmd_flags = ['-cubin', '-std=c++11', '-arch=sm_75', '-Xptxas=-v', 'kernel.cu', '-o']
out_template = 'kernel{number}.cubin'
macro_template = "-D{name}={value}"
input_macros = [
	macro_template.format(name='Num_Inputs', value=2),
	macro_template.format(name='Num_Inputs_Row', value=1),
	macro_template.format(name='Num_Inputs_Col', value=1),
	macro_template.format(name='Num_Inputs_Tot', value=0)
]
def createMacro(count, size, block):
	return [
		macro_template.format(name='Count_Warp_Block_Col', value=count),
		macro_template.format(name='Size_Thr_Col', value=size),
		macro_template.format(name='Block_Depth', value=block),
	]

macro_ranges = {
	'Count_Warp_Block' : [1, 2, 4],
	'Size_Thr' : [1, 2, 4, 8, 16],
	'Block_Depth' : [1, 2, 4, 8, 16]
}

print("Get Device Information")
all_device_properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice()) # !! A lot of time
max_thread_count = all_device_properties['maxThreadsPerBlock']
max_shared_mem = all_device_properties['sharedMemPerBlock']
max_reg_count = all_device_properties['regsPerBlock']
max_block_count = all_device_properties['maxBlocksPerMultiProcessor']
processor_count = all_device_properties['multiProcessorCount']

def searchAvailableSizes(macroRange, pRow, pCol, block=8):
	for count in macroRange['Count_Warp_Block']:
		for size in macroRange['Size_Thr']:
			block_size = 8 * count * size
			thread_count = 32 * count * count * 2
			if block_size > pRow and block_size > pCol:
				continue
			if thread_count > max_thread_count:
				continue
			macro = createMacro(count, size, block)
			info = (thread_count, block_size, count, size, block)
			yield macro, info

def searchAvailableBlocks(macroRange, pDep, count, size):
	for block in macroRange['Block_Depth']:
		if block > pDep:
			continue
		thread_count = 32 * count * count * 2
		if (8 * count * size * block) % thread_count > 0:
			continue
		macro = createMacro(count, size, block)
		info = (block,)
		yield macro, info

def parseOutput(file):
	file.seek(0)
	read_str = file.read()
	str_regs = re.findall('(\d+) registers', read_str)
	if len(str_regs) == 0:
		return None, None
	max_regs = max([int(s) for s in str_regs])
	str_smem = re.findall('(\d+) bytes smem', read_str)
	return max_regs, int(str_smem[0])

def generateCmd(compiler, cmd_flags, id, input_macros, macro):
	cmd = [compiler]
	cmd.extend(cmd_flags)
	cmd.append(out_template.format(number=id))
	cmd.extend(input_macros)
	cmd.extend(macro)
	return cmd

def waitProcesses(waitings):
	for p in waitings:
		try:
			p.wait(2)
		except subprocess.TimeoutExpired as e:
			print("Exceed Compilation Time", ' '.join(e.cmd))
			p.kill()

print("Search Available Sizes") # !! A lot of time
def generateCubin(macro_ranges, a, b, c, func):
	files = []
	processes = []
	infos = []
	number_cpu = os.cpu_count()
	i = 0
	for m, n in func(macro_ranges, a, b, c):
		tmp_file = tempfile.TemporaryFile('w+')
		cmd = generateCmd(compiler, cmd_flags, i, input_macros, m)
		p = subprocess.Popen(cmd, stderr=tmp_file)
		processes.append(p)
		files.append(tmp_file)
		infos.append(n)
		i += 1
		if i % number_cpu == 0:
			waitProcesses(processes[i-number_cpu:i])
	if i % number_cpu != 0:
		remain = i % number_cpu
		waitProcesses(processes[i-remain:i])
	return files, infos

init_block = 1
files, infos = generateCubin(macro_ranges, p_row, p_column, init_block, searchAvailableSizes)

def filter_resources(files, infos, gpu_count, isBlock=False):
	resource_info = {}
	for i, f in enumerate(files):
		reg, smem = parseOutput(f)
		if reg is None:
			continue
		info = infos[i]
		reg *= info[0]
		if reg > max_reg_count:
			continue
		if smem > max_shared_mem:
			continue
		if isBlock:
			resource_info[i] = info
			continue

		max_launch = min(
			max_reg_count // reg,
			max_shared_mem // smem,
			max_thread_count // info[0],
			max_block_count
		)

		bsize = info[1]
		grid_m = (p_row + bsize - 1) // bsize
		grid_n = (p_column + bsize - 1) // bsize
		grid_size = grid_m * grid_n

		if max_launch > 2 and grid_size > gpu_count:
			continue
		max_launch *= gpu_count
		alpha = max(1, grid_size / max_launch)
		resource_info[i] = info + (alpha,)
	return resource_info

resource_info = filter_resources(files, infos, processor_count)

print("Get {} Sizes".format(len(resource_info)))
print(resource_info)

struct_init_code = \
"""
struct Params {
	float *output;
	float *work;
	float *inputs[2];
	int m, n, k;
	int size_K;
	int m0, k0, k1, n1;
	int n_partition;
};

__global__ void init_struct(Params *param, float *C, float *W, float *A, float *B, int m, int n, int k, int bK, int p)
{
	param->output = C;
	param->work = W;
	param->inputs[0] = A;
	param->inputs[1] = B;
	param->m = m;
	param->n = n;
	param->k = k;
	param->size_K = bK;
	param->m0 = m;
	param->k0 = k;
	param->k1 = k;
	param->n1 = n;
	param->n_partition = p;
}
"""

struct_kernel = cp.RawModule(code=struct_init_code, options=('-std=c++11',), name_expressions=['init_struct'])
init_struct = struct_kernel.get_function('init_struct')
Params = np.dtype(
	{
		'names': ['C', 'W', 'A', 'B', 'M', 'N', 'K', 'Bk', 'm', 'k0', 'k1', 'n', 'P'],
		'formats': [np.uint64, np.uint64, np.uint64, np.uint64, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32]
	}
)
C = cp.empty(shape=(p_row, p_column), dtype=np.float32)
A = cp.ones(shape=(p_row, p_depth), dtype=np.float32, order='F')
B = cp.ones(shape=(p_depth, p_column), dtype=np.float32)
param = np.empty(shape=(1,), dtype=Params)
param_ptr = cp.asarray(param.view(np.int8))

init_struct((1,), (1,), (param_ptr, C, C, A, B, p_row, p_column, p_depth, p_depth, 1))
param = cp.asnumpy(param_ptr).view(Params)

streams = []
for _ in range(processor_count):
	streams.append(cp.cuda.stream.Stream(non_blocking=True))

events = []
for _ in range(processor_count):
	events.append((cp.cuda.Event(), cp.cuda.Event()))

def wait_streams(events, keys, times, iter):
	for key, (e_start, e_end) in zip(keys, events):
		e_end.synchronize()
		time_gpu = cp.cuda.get_elapsed_time(e_start, e_end) / iter
		times.append((key, time_gpu))

print("Test different size parameters")
count_proc = 0
cur_keys = []
exec_time = []
iter = 5
for rkey in resource_info.keys():
	kernel = cp.RawModule(path="./kernel{}.cubin".format(rkey))
	cur_keys.append(rkey)
	compute = kernel.get_function('_Z7compute6Params')
	with streams[count_proc]:
		events[count_proc][0].record()
		for _ in range(iter):
			compute((1, 1,), (resource_info[rkey][0],), (param,))
		events[count_proc][1].record()
	count_proc += 1

	if count_proc % processor_count == 0:
		wait_streams(events, cur_keys, exec_time, iter)
		count_proc = 0
		cur_keys = []

if count_proc != 0:
	wait_streams(events, cur_keys, exec_time, iter)

print("Estimate Execution Time")
print(exec_time)
estimated_time = []
for id, et in exec_time:
	et *= resource_info[id][-1]
	estimated_time.append((et, id))

print("Get Best Size Parameter")
tm, id = min(estimated_time, key=lambda x : x[0])
info_id = list(resource_info[id])
print("Best Size {}:\twarp_count_column: {} thread_size_column: {}".format(tm, info_id[2], info_id[3]))

if info_id[-1] == 1:
	#TODO: search for split-K
	pass

print("Search Available Blocks")
files, infos = generateCubin(macro_ranges, p_depth, info_id[2], info_id[3], searchAvailableBlocks)
resource_info = filter_resources(files, infos, processor_count, True)

print("Get {} Blocks".format(len(resource_info)))
print(resource_info)

d_block = max(resource_info.values(), key=lambda x : x[0])[0] * iter
d_block = min(d_block, p_depth)
init_struct((1,), (1,), (param_ptr, C, C, A, B, p_row, p_column, d_block, d_block, 1))
param = cp.asnumpy(param_ptr).view(Params)

print("Test different block parameters")
tcount = info_id[0]
bsize = info_id[1]
grid_m = (p_row + bsize - 1) // bsize
grid_n = (p_column + bsize - 1) // bsize
event = events[0]
exec_time = []
for rkey in resource_info.keys():
	print(rkey)
	kernel = cp.RawModule(path="./kernel{}.cubin".format(rkey))
	compute = kernel.get_function('_Z7compute6Params')
	event[0].record()
	for _ in range(iter):
		compute((grid_m, grid_n,), (tcount,), (param,))
	event[1].record()
	wait_streams([event], [rkey], exec_time, iter)
print(exec_time)
print("Get Best Size Parameter")
id, tm = min(exec_time, key=lambda x : x[1])
info_id[4] = resource_info[id][0]
print("Best Block {}:\tBlock_Depth: {}".format(tm, info_id[4]))

macro = createMacro(info_id[2], info_id[3], info_id[4])
cmd = generateCmd(compiler, cmd_flags, '', input_macros, macro)
print(' '.join(cmd))
print(timeit.default_timer() - start)
quit()
