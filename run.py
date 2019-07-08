import sys
from ip import InputParameters
from sim import Simulation

def sim(nt, dri):
    ip = InputParameters(nt, dri)
    s = Simulation(ip)
    s.run()

if __name__ == "__main__":
    if len(sys.argv) == 3:
        nt = int(sys.argv[1])
        dri = float(sys.argv[2])
        print("Running simulation with nt = {}, dRi = {})".format(nt, dri))
        sim(nt, dri)
    else:
        print("USAGE: python run.py time-steps ioniation-range-m")
        print("EXAMPLE: python run.py 500000 10e3")

