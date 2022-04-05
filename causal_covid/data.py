import numpy as np
import pandas as pd


def sum_age_groups(df, num_age_groups):
    if num_age_groups == 9:
        sum_over = [
            ("0-19",),
            ("20-29",),
            ("30-39",),
            ("40-49",),
            ("50-59",),
            ("60-69",),
            ("70-79",),
            ("80-89",),
            ("90+",),
        ]
        new_age_ranges = [
            "0-19",
            "20-29",
            "30-39",
            "40-49",
            "50-59",
            "60-69",
            "70-79",
            "80-89",
            "90+",
        ]
    elif num_age_groups == 3:
        sum_over = [
            ("0-19", "20-29"),
            ("30-39", "40-49", "50-59"),
            ("60-69", "70-79", "80-89", "90+"),
        ]
        new_age_ranges = ["0-29", "30-59", "60+"]
    elif num_age_groups == 1:
        sum_over = [
            (
                "0-19",
                "20-29",
                "30-39",
                "40-49",
                "50-59",
                "60-69",
                "70-79",
                "80-89",
                "90+",
            )
        ]
        new_age_ranges = ["0+"]
    else:
        raise RuntimeError("Unknown number of age groups")

    date_exists = "date" in df.columns

    def filter_func(i):
        age = df.loc[i].age_group
        date = df.loc[i].date if date_exists else 0
        for i, ages in enumerate(sum_over):
            if age in ages:
                return date, new_age_ranges[i]
        else:

            raise RuntimeError(f"age-group {age} not known")

    df = df.groupby(by=filter_func).sum()
    df.insert(0, "age_group", list(zip(*df.index))[1])
    if date_exists:
        df.insert(0, "date", list(zip(*df.index))[0])
    df = df.reset_index(drop=True)
    # df.index = pd.MultiIndex.from_tuples(df.index, names=["date", "Age_group"])
    return df


def sum_age_groups_np(array, num_age_groups):
    if num_age_groups == 9:
        sum_over = [[0,], [1,], [2,], [3,], [4,], [5,], [6,], [7,], [8,]]
    elif num_age_groups == 3:
        sum_over = [[0, 1], [2, 3, 4], [5, 6, 7, 8]]
    elif num_age_groups == 1:
        sum_over = [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
    else:
        raise RuntimeError("Unknown number of age groups")

    new = np.empty((len(sum_over), *array.shape[1:]))
    for i, ages in enumerate(sum_over):
        new[i] = array[ages].sum(axis=0)
    return new


def filter_time(df, begin, end):
    return df[(df.date >= begin) & (df.date <= end)]


def transpose_dataframe(df, column_choice):
    return pd.DataFrame(
        index=df.date.unique(),
        data=np.array(
            [
                np.array(column_choice(df[df.age_group == age]))
                for age in df.age_group.unique()
            ]
        ).T,
        columns=df.age_group.unique(),
    )


def load_cases(file, begin, end, num_age_groups=3, **kwargs):
    cases = pd.read_csv(file, **kwargs)
    cases.rename(
        columns={"Sunday_date": "date", "Age_group": "age_group"}, inplace=True
    )
    cases.date = pd.to_datetime(cases.date)
    cases.drop(cases[cases.age_group == "total"].index, inplace=True)
    cases = sum_age_groups(cases, num_age_groups=num_age_groups)

    cases = filter_time(cases, begin, end)

    def col_choice(cases):
        return np.sum(np.array(cases.loc[:, "positive_unvaccinated":]), axis=1)

    cases_summed = transpose_dataframe(cases, col_choice)
    return cases_summed


def load_infectiability(
    vaccination_file,
    population_file,
    U2_file,
    U3_file,
    waning_file,
    begin,
    end,
    num_age_groups=3,
    **kwargs,
):
    # Returns 4 dataframes (unvaccinated_share, one_dose_share, two, three), each of the same form as the case number dfs

    population = load_population(
        population_file, transpose=False, num_age_groups=9, **kwargs
    )

    vaccinations = pd.read_csv(vaccination_file, **kwargs)
    vaccinations.rename(
        columns={"Sunday_date": "date", "Age_group": "age_group"}, inplace=True
    )
    vaccinations.date = pd.to_datetime(vaccinations.date)

    waning_profile = pd.read_csv(waning_file)
    waning_profile = waning_profile.vaccine_efficacy

    if isinstance(U2_file, str):
        U_2 = np.load(U2_file)
    else:
        U_2 = np.array(U2_file)
    if isinstance(U3_file, str):
        U_3 = np.load(U3_file)
    else:
        U_3 = np.array(U3_file)

    # convert U_3 from conditional prob. to absolute numbers by multiplying by the
    # number of second doses
    for age in range(U_3.shape[0]):
        for i in range(
            U_3.shape[1] - 1
        ):  # i is t_2, exclude last column as this is the
            # ones that only got their first dose not second doses
            U_3[age, i, :] *= U_2[age, :-1, i].sum()  # exclude last row of U_2 as it
            # it includes the completely non-vaccinated in [:,-1, -1]

    U_2 = sum_age_groups_np(U_2, num_age_groups)
    U_3 = sum_age_groups_np(U_3, num_age_groups)

    vaccinations = sum_age_groups(vaccinations, num_age_groups=num_age_groups)

    population = sum_age_groups(population, num_age_groups=num_age_groups)

    # convert absolute numbers to share
    U_2 = (U_2.T / population.M.values).T
    U_3 = (U_3.T / population.M.values).T

    immune_1 = np.zeros(
        (len(vaccinations.age_group.unique()), len(vaccinations.date.unique()))
    )
    for age in range(len(vaccinations.age_group.unique())):
        for t in range(len(vaccinations.date.unique())):
            for t_dif in range(
                min(t + 1, len(waning_profile) + 1)
            ):  # for all potential times between doses

                immune_1[age, t] += (
                    U_2[age, t - t_dif, t + 1 :].sum() * waning_profile[t_dif]
                )  # Sum all vaccinations with first dose at  t - t_dif and 2nd dose later
                # than the current time

    immune_2 = np.zeros(
        (len(vaccinations.age_group.unique()), len(vaccinations.date.unique()))
    )
    for age in range(len(vaccinations.age_group.unique())):
        for t in range(len(vaccinations.date.unique())):
            for t_dif in range(
                min(t + 1, len(waning_profile) + 1)
            ):  # for all potential times between doses
                immune_2[age, t] += (
                    U_3[age, t - t_dif, t + 1 :].sum() * waning_profile[t_dif]
                )  # Sum all vaccinations with second dose at  t - t_dif and 3rd dose later
                # than the current time

    immune_3 = np.zeros(
        (len(vaccinations.age_group.unique()), len(vaccinations.date.unique()))
    )
    for age in range(len(vaccinations.age_group.unique())):
        for t in range(len(vaccinations.date.unique())):
            for t_dif in range(
                min(t + 1, len(waning_profile) + 1)
            ):  # for all potential times between doses
                immune_3[age, t] += (
                    U_3[age, :-1, t - t_dif].sum() * waning_profile[t_dif]
                )
                # Sum all vaccinations with 3rd dose at  t - t_dif and 2nd dose at some
                # non-relevant time.

    # filter time interval
    vaccinations_filtered = filter_time(vaccinations, begin, end)

    beg_index = np.argwhere(vaccinations.date.unique() >= np.datetime64(begin))[0, 0]
    end_index = np.argwhere(vaccinations.date.unique() <= np.datetime64(end))[-1, 0] + 1
    # plus 1 because of last index inclusive

    immune_1 = pd.DataFrame(
        index=vaccinations_filtered.date.unique(),
        data=immune_1[:, beg_index:end_index].T,
        columns=vaccinations_filtered.age_group.unique(),
    )
    immune_2 = pd.DataFrame(
        index=vaccinations_filtered.date.unique(),
        data=immune_2[:, beg_index:end_index].T,
        columns=vaccinations_filtered.age_group.unique(),
    )
    immune_3 = pd.DataFrame(
        index=vaccinations_filtered.date.unique(),
        data=immune_3[:, beg_index:end_index].T,
        columns=vaccinations_filtered.age_group.unique(),
    )
    return 1 - (immune_1 + immune_2 + immune_3)


def load_population(population_file, transpose=True, num_age_groups=3, **kwargs):
    population = pd.read_csv(population_file, **kwargs)
    population.rename(
        columns={"Age_group": "age_group", "Population_size": "M"}, inplace=True
    )
    total_population = float(population[population.age_group == "total"].values[0, 1])
    population.drop(population[population.age_group == "total"].index, inplace=True)
    population = sum_age_groups(population, num_age_groups=num_age_groups)
    if transpose:
        population = pd.DataFrame(
            index=[0],
            data=np.array(population.loc[:, "M"])[None],
            columns=population.age_group,
        )
    return population
