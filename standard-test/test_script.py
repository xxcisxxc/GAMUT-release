import subprocess
import sys

sys.stdout = open('result.log', 'w')

n_start = 10
n_end = 15
ms = [str(2 ** i) for i in range(n_start, n_end + 1)]
ns = [str(2 ** i) for i in range(n_start, n_end + 1)]
ks = [str(2 ** i) for i in range(n_start, n_end + 1)]
#split_size = str(256)

for m, n, k in zip(ms, ns, ks):
	print("m: {}, n: {}, k: {}".format(m, n, k))
	sys.stdout.flush()
	process = subprocess.Popen(["./build/test", m, n, k], stdout=sys.stdout)
	process.wait()
	print("\n")
	sys.stdout.flush()

sys.stdout.close()