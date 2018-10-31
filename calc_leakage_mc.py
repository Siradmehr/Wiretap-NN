import ast

import numpy as np
import pandas as pd

CONFIG = "config2"
TSNR = "t2"

def main():
    data = pd.read_csv("lwc-{}-{}.dat".format(CONFIG, TSNR), sep='\t')
    file_full = "results-{}-{}-full.dat".format(CONFIG, TSNR)
    file_cond = "results-{}-{}-conditional.dat".format(CONFIG, TSNR)
    entr_z = parse_file(file_full)
    entr_zm = parse_file(file_cond)
    leak = {'wB': [], 'wE': [], 'LeakMC': []}
    for combination, entr in entr_z.items():
        _entr_cond = entr_zm[combination]
        _leak = (entr-_entr_cond)/np.log(2)
        #leak[combination] = _leak
        print("{}:\t{}".format(combination, (entr-_entr_cond)/np.log(2)))
        wb, we = ast.literal_eval(combination)
        leak['wB'].append(wb)
        leak['wE'].append(we)
        leak['LeakMC'].append(_leak)
    data_new = pd.DataFrame(leak)
    print(data)
    print(data_new)
    #data_new = pd.merge(data, data_new)
    data_new = pd.concat((data, data_new['LeakMC']), axis=1)
    print(data_new)
    data_new.to_csv('lwc-{}-{}-mc.dat'.format(CONFIG, TSNR), sep='\t', index=False)

def parse_file(filename):
    with open(filename) as infile:
        x = infile.readlines()
    lines = [a.split(": ") for a in x]
    entr = {k: float(v) for k, v in lines}
    return entr

if __name__ == "__main__":
    main()
