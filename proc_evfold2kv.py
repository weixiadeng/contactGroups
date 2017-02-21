import sys
import numpy as np
# informative sdii filter 
# from shadpw algorithm

def main():
	if len(sys.argv) < 3:
		print 'Usage: python proc_evfold2sdii.py PF00589.dca 1a0p-A-PF00589-XERD_ECOLI.map'
		print 'output: PF00589.dca.kv, PF00589.wmi.kv'
		exit()

	dcafile = sys.argv[1]
	mapfile = sys.argv[2]
	outfile = dcafile+'.kv'

	# Array of tuple (2-3, 0.25)
	sdiiArray = []
	sdiiValue = []
	with open(sdiifile) as fp:
		for line in fp:
			line = line.strip()
			if len(line) < 1:
				print 'error sdii line: %s' % line
			valueArray = line.split(' ')
			sdiiArray.append((valueArray[0], float(valueArray[1])))
			sdiiValue.append(float(valueArray[1]))

	#print 'sdiiArray: %s' % (repr(sdiiArray))
	#print 'sdiiValue: %s' % (repr(sdiiValue))

	sdiinp = np.array(sdiiValue)
	outlier = sdiinp.mean()+sdiinp.std()
	#print 'm: %.4f, s: %.4f, o: %.4f' % (sdiinp.mean(), sdiinp.std(), outlier)

	sdii_no_outlier = [v for v in sdiiValue if v < outlier]
	sdiinp = np.array(sdii_no_outlier)
	cutoff = sdiinp.mean() + d*sdiinp.std()
	#print 'sdii_no_outlier: %s' % repr(sdii_no_outlier)
	#print 'm1: %.4f, s1: %.4f, cutoff: %.4f' % (sdiinp.mean(), sdiinp.std(), cutoff)

	c = 0
	fout = open(outfile, 'w')
	for (var, value) in sdiiArray:
		if value > cutoff:
			fout.write('%s %.8f\n' % (var, value))
			c+=1
	fout.close()

	print '%s: bgMI: %.4f cutoff: %.4f #ofIPV: %d/%d' % (outfile, sdiinp.mean(), cutoff, c, len(sdiiValue))

if __name__ == '__main__':
	main()