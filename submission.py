from mdp import load_map_from_file, MDP
from Approach import approach

FILENAME = "rnGraph"

if __name__ == "__main__":
    print("load_map_from_file")
    g, t = load_map_from_file(fname=FILENAME)
    print(t)

    print("\nconstructor")
    mdp = MDP(mapfile=FILENAME)
    print(mdp)

    print("\nvalue iteration")
    mdp = MDP(mapfile=FILENAME)
    print(mdp.value_iteration())
    print(mdp.get_utilities())

    print("\npolicy iteration")
    mdp = MDP(mapfile=FILENAME)
    print(mdp.policy_iteration())
    print(mdp.get_utilities())

    print("\napproach")
    t10 = approach(10)
    # t12 = approach(12)
    # t18 = approach(18)
    print(t10)
    # print(t12)
    # print(t18)
