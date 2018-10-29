import numpy as np

def main():
    file_full = "results-config1-full.dat"
    file_cond = "results-config1-conditional.dat"
    entr_z = parse_file(file_full)
    entr_zm = parse_file(file_cond)
    for combination, entr in entr_z.items():
        _entr_cond = entr_zm[combination]
        print("{}:\t{}".format(combination, (entr-_entr_cond)/np.log(2)))

def parse_file(filename):
    with open(filename) as infile:
        x = infile.readlines()
    lines = [a.split(": ") for a in x]
    entr = {k: float(v) for k, v in lines}
    return entr

if __name__ == "__main__":
    main()
