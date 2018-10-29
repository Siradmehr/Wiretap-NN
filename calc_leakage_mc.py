import ast

import numpy as np
import pandas as pd

def main():
    data = pd.read_csv("lwc-config1-t5.dat", sep='\t')
    file_full = "results-config1-full.dat"
    file_cond = "results-config1-conditional.dat"
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
    data_new = pd.merge(data, data_new)
    print(data_new)
    data_new.to_csv('lwc-config1-t5-mc.dat', sep='\t', index=False)

def parse_file(filename):
    with open(filename) as infile:
        x = infile.readlines()
    lines = [a.split(": ") for a in x]
    entr = {k: float(v) for k, v in lines}
    return entr

if __name__ == "__main__":
    main()
