from mdp import load_map_from_file
from Approach import approach

FILENAME = "rnGraph"

if __name__ == "__main__":
    g, t = load_map_from_file(fname=FILENAME)

    t10 = approach(10)
    t12 = approach(12)
    t18 = approach(18)
    print(t10)
