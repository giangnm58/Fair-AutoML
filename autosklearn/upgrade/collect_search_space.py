import json

f = open("search_space", "r")
text = f.read()
count = 0
checker = False
model_str = ""
search_space = {}
attr = ["n_estimators", "criterion", "max_depth", "min_samples_split", "min_samples_leaf", "min_weight_fraction_leaf",
        "max_features", "max_leaf_nodes", "min_impurity_decrease", "min_impurity_split", "bootstrap", "oob_score",
        "n_jobs", "random_state", "verbose", "warm_start", "class_weight", "ccp_alpha", ""]
def model_extraction(model_name, text_list):

    for component in text_list:
        if "random_forest:" in component:
            splt = component.split(":")
            key = splt[2].strip()[:len(splt[2])-1]
            value = splt[3].strip()
            if value == "'False'":
                value = False
            elif value == "'True'":
                value = True
            elif value == "'None'":
                value = None
            elif "'" in value:
                value = value[1:len(value)-1]
            else:
                value = float(value)
            if key not in search_space:
                search_space.update({key:[value]})
            else:
                if value not in search_space[key]:
                    search_space[key].append(value)
    return search_space
for t in text:
    if t == "{":
        count += 1
        if count % 2 != 0:
            if count != 1:
                model_str = model_str.split(",")
                model_dict = model_extraction("random_forest:", model_str)
            model_str = ""
            checker = True
    if t == "}":
        checker = False

    if checker:
        model_str += t

print(search_space)
