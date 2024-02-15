import numpy as np
from scipy.stats import entropy as scipy_entropy
import warnings

def pass_warn(*args, **kwargs):
    pass

class AutonSKRFWrapper:

    def __init__(self, skrf):
        self._skrf = skrf
        self._n_trees = len(skrf.estimators_)
        self._bounds_set = False
        self._node_depths = [node_depths(tree) for tree in self._skrf.estimators_]
        self._is_leaf = [tree.tree_.children_left == tree.tree_.children_right for tree in self._skrf.estimators_]

    def _predict_proba_trees(self, data):
        # Suppresses an incorrect warning
        # I've opened this issue: https://github.com/scikit-learn/scikit-learn/issues/26140

        old_warn = warnings.warn
        warnings.warn = pass_warn
        tree_preds = np.array([tree.predict_proba(data) for tree in self._skrf.estimators_])
        warnings.warn = old_warn
        return tree_preds

    def feature_usage(self, data):
        dp_count = np.zeros(data.shape[1])
        weighted_count = np.zeros(data.shape[1])
        raw_count = np.zeros(data.shape[1])

        data = np.array(data)
        have_counted_dp = np.zeros(data.shape, dtype=bool)

        for (ti, tree) in enumerate(self._skrf.estimators_):
            for (sample, node_id) in zip(*tree.decision_path(data).nonzero()):
                if not self._is_leaf[ti][node_id]:
                    feature_i = tree.tree_.feature[node_id]
                    raw_count[feature_i] += 1
                    weighted_count[feature_i] += 1/(1+self._node_depths[ti][node_id])

                    if not have_counted_dp[sample, feature_i]:
                        dp_count[feature_i] += 1
                        have_counted_dp[sample, feature_i] = True

        return dp_count, weighted_count, raw_count


    def dot_product_sum(self, data):
        tree_preds = self._predict_proba_trees(data)
        dps = np.zeros((data.shape[0],))
        for i in range(self._n_trees):
            for j in range(i+1, self._n_trees):
                dps += np.sum(np.multiply(tree_preds[i], tree_preds[j]), axis=1)
        return dps

    def mean_entropy(self, data):
        tree_preds = self._predict_proba_trees(data)
        n_trees = len(tree_preds)
        ent_sum = np.zeros(data.shape[0])
        for ti in range(n_trees):
            ent_sum += scipy_entropy(tree_preds[ti], base=2, axis=1)
        return ent_sum/n_trees

    def set_bounds_data(self, bounds_data):
        bounds_data = np.array(bounds_data)
        self._bounds = [{} for _ in range(self._n_trees)]
        for (ti, tree) in enumerate(self._skrf.estimators_):
            for (sample, node_id) in zip(*tree.decision_path(bounds_data).nonzero()):
                if not self._is_leaf[ti][node_id]:
                    feature_i = tree.tree_.feature[node_id]
                    value = bounds_data[sample, feature_i]
                    if node_id not in self._bounds[ti]:
                        self._bounds[ti][node_id] = (value, value)
                    else:
                        if value < self._bounds[ti][node_id][0]:
                            self._bounds[ti][node_id] = (value, self._bounds[ti][node_id][1])
                        if value > self._bounds[ti][node_id][1]:
                            self._bounds[ti][node_id] = (self._bounds[ti][node_id][0], value)
        self._bounds_set = True

    def inbounds(self, data):
        if not self._bounds_set:
            raise Exception("set_bounds_data must be called before calling inbounds")

        data = np.array(data)

        n_nodes_touched = np.zeros((data.shape[0]))
        n_in_bounds = np.zeros((data.shape[0]))

        for (ti, tree) in enumerate(self._skrf.estimators_):
            for (sample, node_id) in zip(*tree.decision_path(data).nonzero()):
                if not self._is_leaf[ti][node_id]:
                    feature_i = tree.tree_.feature[node_id]
                    value = data[sample, feature_i]

                    if value >= self._bounds[ti][node_id][0] and value <= self._bounds[ti][node_id][1]:
                        n_in_bounds[sample] += 1
                    n_nodes_touched[sample] += 1

        return n_in_bounds/n_nodes_touched

    def write_legacy(self, fp_or_path, model_name="default_model", feature_names=None, ignore_bounds=False, for_ernie=False):
        """
        Writes the random forest using the autonRF-C convention.

        :param fp_or_path: Either a string to the output path, or a file pointer if already open.
        :param model_name: Name of model. Default is "default_model"
        :param feature_names: Only used if your sklearn RF does NOT have feature_names_in_ defined. This is new in
            sklearn version 1.0.
        :param ignore_bounds: Default False. If False then set_bounds() needs to be called prior.
            If True all bounds will be (0,0). True is used if you haven't called set_bounds
            and you don't care about calling inbounds()
        :param for_ernie: Boolean with flags the addition of "SegmentInfo.FoldID" to the features list (ERNIE specific)
        :return: None
        """

        if ignore_bounds:
            bounds_fn = lambda tree_index, node_index : (0,0)
        else:
            if not self._bounds_set:
                raise ValueError("ignore_bounds can only be False if set_bounds_data was called")
            bounds_fn = lambda tree_index, node_index : self._bounds[tree_index][node_index]

        if feature_names is None:
            try:
                feature_names = self._skrf.feature_names_in_
            except AttributeError:
                raise ValueError("You must provide feature_names in write_legacy() if your sklearn random forest does \
                not have feature_names_in_ defined")

        feature_names = list(feature_names)

        def str_arr_str(arr):
            arr_lines = "\n".join(["%s" % a for a in arr])
            return "".join((
                "<string_array>\n",
                "size %d\n" % len(arr),
                arr_lines, "\n",
                "</string_array>\n"
            ))

        def dyv_str(arr):
            arr_lines = "\n".join(["%r" % a for a in arr])
            return "".join((
                "<dyv>\n",
                "size %d\n" % len(arr),
                arr_lines, "\n",
                "</dyv>\n"
            ))

        def write_header(rf):
            if for_ernie:
                feature_names_expanded = ["SegmentInfo.FoldID"] + feature_names + ["LABEL"]
            else:
                feature_names_expanded =  feature_names + ["LABEL"]

            header = "".join(
                ("<bag_model>\n",
                 "1\n",
                 str_arr_str([model_name]),
                 str_arr_str(feature_names_expanded),
                 "%d\n" % len(rf.estimators_),
                 "%d\n" % (len(feature_names_expanded) - 1),
                 str_arr_str(rf.classes_),
                 )
            )
            fp.write(header)

        def write_node(tree_i, node_id):
            """
            Example node for reference.

            <decision_node>
            false  # is leaf
            75     # att num
            false  # is symbolic
            false  # contains missing values , not used by java
            -1      # missing values decision path, not used by java
            1.543990803e+01  # threshold
            0.000000000e+00  # min
            5.015282227e+03  # max
            <decision_node>


            <decision_node>
            true # is leaf
            true # is classification
            <dyv>
            size 2
            0
            249
            </dyv>
            </decision_node>
            """
            tree = self._skrf.estimators_[tree_i].tree_

            is_leaf = tree.children_left[node_id] == tree.children_right[node_id]

            if is_leaf:
                node_str = "".join(
                    (
                        "<decision_node>\n",
                        "true\n",
                        "true\n",
                        dyv_str([int(v) for v in tree.value[node_id][0]]),
                        "</decision_node>\n"
                    )
                )
                fp.write(node_str)
                return

            node_bounds = bounds_fn(tree_i, node_id)
            node_str = "".join(
                (
                    "<decision_node>\n",
                    "false\n",
                    "%d\n" % (tree.feature[node_id] + 1),  # +1 because of legacy "SegmentInfo.FoldID" feature
                    "false\n",
                    "false\n",
                    "-1\n",
                    "%r\n" % tree.threshold[node_id],
                    "%r\n" % node_bounds[0],
                    "%r\n" % node_bounds[1]
                )
            )
            fp.write(node_str)
            write_node(tree_i, tree.children_left[node_id])
            write_node(tree_i, tree.children_right[node_id])
            fp.write("</decision_node>\n")

        def write_forest(fp):
            write_header(self._skrf)
            for tree_i in range(self._skrf.n_estimators):
                fp.write("<decision_tree>\n")
                write_node(tree_i, 0)
                fp.write("</decision_tree>\n")
            fp.write("</bag_model>")


        if isinstance(fp_or_path, str):
            with open(fp_or_path, 'w') as fp:
                write_forest(fp)
        else:
            write_forest(fp_or_path)

def node_depths(tree):
    depths = np.zeros(tree.tree_.feature.shape)
    left_child = tree.tree_.children_left
    right_child = tree.tree_.children_right

    def nd_recur(node_id, curr_depth):
        if node_id < 0:
            return
        depths[node_id] = curr_depth
        nd_recur(left_child[node_id], curr_depth+1)
        nd_recur(right_child[node_id], curr_depth+1)

    nd_recur(0, 0)
    return depths


