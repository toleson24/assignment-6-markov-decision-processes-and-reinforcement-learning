from mdp import load_map_from_file

FILENAME = "rnGraph"

g, t = load_map_from_file(fname=FILENAME)

print(t)
