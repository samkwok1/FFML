import hydra
from omegaconf import DictConfig
import os
from algorithms import decision_trees, gda, logreg, naive_bayes, perceptron

@hydra.main(config_path="config", config_name="config")
def main(args: DictConfig) -> None:
    sys_path = os.path.dirname(os.path.abspath(__file__))
    save_path = f"{args.sim.sim_dir}/{args.sim.sim_algo}/{args.sim.sim_id}"
    algo = args.env.algorithm
    pos = args.env.position
    if pos == "rb":
        d_path = "input_data/RBs/all_rb_stats.csv"
    elif pos == "te":
        d_path = "input_data/tight_ends/all_te_stats.csv"
    else:
        d_path = "input_data/wrs/all_wr_stats.csv"

    data_path = os.path.join(sys_path[:len(sys_path) - (len("/experiments") - 1)], d_path)

    if algo == "logistic_regression":
        func = logreg
    elif algo == "decision_trees":
        func = decision_trees
    elif algo == "gda":
        func = gda
    elif algo == "naive_bayes":
        func = naive_bayes
    elif algo == "perceptron":
        func = perceptron
    func.main(save_path=save_path, train_path=data_path, pos=pos)

if __name__ == "__main__":
    main()
