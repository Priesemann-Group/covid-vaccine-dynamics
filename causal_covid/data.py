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


def sum_population(df, num_age_groups):
    if num_age_groups == 3:
        sum_over = [
            ("0-19", "20-29"),
            ("30-39", "40-49", "50-59"),
            ("60-69", "70-79", "80-89", "90+"),
        ]
        new_age_ranges = ["0-29", "30-59", "60+"]
    else:
        raise RuntimeError("Unknown number of age groups")

    def filter_func(i):
        age = df.loc[i].age_group
        for i, ages in enumerate(sum_over):
            if age in ages:
                return 0, new_age_ranges[i]
        else:

            raise RuntimeError(f"age-group {age} not known")

    df = df.groupby(by=filter_func).sum()
    df.insert(0, "age_group", list(zip(*df.index))[1])
    df = df.reset_index(drop=True)
    # df.index = pd.MultiIndex.from_tuples(df.index, names=["date", "Age_group"])
    return df


def load_vaccinations(
    vaccination_file, population_file, begin, end, num_age_groups=3, **kwargs
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

    for age, size in population.values.tolist():
        vaccinations.loc[vaccinations.age_group == age, "unvaccinated_share":] *= size

    vaccinations = sum_age_groups(vaccinations, num_age_groups=num_age_groups)
    population = sum_population(population, num_age_groups=num_age_groups)

    for age, size in population.values.tolist():
        vaccinations.loc[vaccinations.age_group == age, "unvaccinated_share":] /= size

    vaccinations = filter_time(vaccinations, begin, end)

    unvaccinated = transpose_dataframe(
        vaccinations, column_choice=lambda df: df.loc[:, "unvaccinated_share"]
    )
    one_dose = transpose_dataframe(
        vaccinations, column_choice=lambda df: df.loc[:, "1st_dose_share"]
    )
    two_doses = transpose_dataframe(
        vaccinations, column_choice=lambda df: df.loc[:, "2nd_dose_share"]
    )
    three_doses = transpose_dataframe(
        vaccinations, column_choice=lambda df: df.loc[:, "3rd_dose_share"]
    )

    return (unvaccinated, one_dose, two_doses, three_doses)


def infectiability(vaccination_dfs, mu, waned_dfs):
    return (
        vaccination_dfs[0]
        + vaccination_dfs[1] * (1.0 - mu[0] * waned_dfs[0])
        + vaccination_dfs[2] * (1.0 - mu[1] * waned_dfs[1])
        + vaccination_dfs[3] * (1.0 - mu[2] * waned_dfs[2])
    )


def load_population(population_file, transpose=True, num_age_groups=3, **kwargs):
    population = pd.read_csv(population_file, **kwargs)
    population.rename(
        columns={"Age_group": "age_group", "Population_size": "size"}, inplace=True
    )
    total_population = float(population[population.age_group == "total"].values[0, 1])
    population.drop(population[population.age_group == "total"].index, inplace=True)
    population = sum_age_groups(population, num_age_groups=num_age_groups)
    if transpose:
        population = pd.DataFrame(
            index=[0],
            data=np.array(population.loc[:, "size"])[None],
            columns=population.age_group,
        )
    return population
