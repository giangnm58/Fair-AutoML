import math
def disparate_impact(subset_data_orig_train, prediction, protected_attribute):
    subset_data_orig_train_df = subset_data_orig_train.convert_to_dataframe()
    privileged_count = 0
    unprivileged_count = 0
    total_privileged_count = 0
    total_unprivileged_count = 0
    for i in range(len(subset_data_orig_train_df[0][protected_attribute])):
        if subset_data_orig_train_df[0][protected_attribute][i] == 1:
            total_privileged_count += 1
        elif subset_data_orig_train_df[0][protected_attribute][i] == 0:
            total_unprivileged_count += 1
        if subset_data_orig_train_df[0][protected_attribute][i] == 1 and prediction[i] == 1:
            privileged_count += 1
        elif subset_data_orig_train_df[0][protected_attribute][i] == 0 and prediction[i] == 1:
            unprivileged_count += 1
    if (privileged_count / total_privileged_count) != 0:
        return abs(math.log((unprivileged_count / total_unprivileged_count) / (privileged_count / total_privileged_count)))
    else:
        return 0.0


def statistical_parity_difference(subset_data_orig_train, prediction,protected_attribute):
    subset_data_orig_train_df = subset_data_orig_train.convert_to_dataframe()
    privileged_count = 0
    unprivileged_count = 0
    total_privileged_count = 0
    total_unprivileged_count = 0
    for i in range(len(subset_data_orig_train_df[0][protected_attribute])):
        if subset_data_orig_train_df[0][protected_attribute][i] == 1:
            total_privileged_count += 1
        elif subset_data_orig_train_df[0][protected_attribute][i] == 0:
            total_unprivileged_count += 1
        if subset_data_orig_train_df[0][protected_attribute][i] == 1 and prediction[i] == 1:
            privileged_count += 1
        elif subset_data_orig_train_df[0][protected_attribute][i] == 0 and prediction[i] == 1:
            unprivileged_count += 1
    return abs((unprivileged_count / total_unprivileged_count) - (privileged_count / total_privileged_count))


def equal_opportunity_difference(subset_data_orig_train, prediction, y_test, protected_attribute):
    subset_data_orig_train_df = subset_data_orig_train.convert_to_dataframe()
    privileged_count = 0
    unprivileged_count = 0
    total_privileged_count = 0
    total_unprivileged_count = 0
    for i in range(len(subset_data_orig_train_df[0][protected_attribute])):
        if subset_data_orig_train_df[0][protected_attribute][i] == 1 and y_test[i] == 1:
            total_privileged_count += 1
        elif subset_data_orig_train_df[0][protected_attribute][i] == 0 and y_test[i] == 1:
            total_unprivileged_count += 1
        if subset_data_orig_train_df[0][protected_attribute][i] == 1 and prediction[i] == 1 and y_test[i] == 1:
            privileged_count += 1
        elif subset_data_orig_train_df[0][protected_attribute][i] == 0 and prediction[i] == 1 and y_test[i] == 1:
            unprivileged_count += 1
    return abs((unprivileged_count / total_unprivileged_count) - (privileged_count / total_privileged_count))



def average_odds_difference(subset_data_orig_train, prediction, y_test, protected_attribute):
    subset_data_orig_train_df = subset_data_orig_train.convert_to_dataframe()
    privileged_count = 0
    unprivileged_count = 0
    total_privileged_count = 0
    total_unprivileged_count = 0

    privileged_count1 = 0
    unprivileged_count1 = 0
    total_privileged_count1 = 0
    total_unprivileged_count1 = 0
    for i in range(len(subset_data_orig_train_df[0][protected_attribute])):
        # print(y_test[i])
        if subset_data_orig_train_df[0][protected_attribute][i] == 1 and y_test[i] == 1:
            total_privileged_count += 1
        elif subset_data_orig_train_df[0][protected_attribute][i] == 0 and y_test[i] == 1:
            total_unprivileged_count += 1
        if subset_data_orig_train_df[0][protected_attribute][i] == 1 and prediction[i] == 1 and y_test[i] == 1:
            privileged_count += 1
        elif subset_data_orig_train_df[0][protected_attribute][i] == 0 and prediction[i] == 1 and y_test[i] == 1:
            unprivileged_count += 1


        if subset_data_orig_train_df[0][protected_attribute][i] == 1 and y_test[i] == 0:
            total_privileged_count1 += 1
        elif subset_data_orig_train_df[0][protected_attribute][i] == 0 and y_test[i] == 0:
            total_unprivileged_count1 += 1
        if subset_data_orig_train_df[0][protected_attribute][i] == 1 and prediction[i] == 1 and y_test[i] == 0:
            privileged_count1 += 1
        elif subset_data_orig_train_df[0][protected_attribute][i] == 0 and prediction[i] == 1 and y_test[i] == 0:
            unprivileged_count1 += 1

    # print(total_unprivileged_count, total_privileged_count, total_unprivileged_count1, total_privileged_count1)
    return abs(((unprivileged_count / total_unprivileged_count) - (privileged_count / total_privileged_count)
                + (unprivileged_count1 / total_unprivileged_count1) - (privileged_count1 / total_privileged_count1))/2)

