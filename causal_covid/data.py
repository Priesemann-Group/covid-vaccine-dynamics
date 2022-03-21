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

def sum_age_groups_np(array, num_age_groups):
    if num_age_groups == 3:
        sum_over = [(0,1),(2,3,4),(5,6,7,8)]
    else:
        raise RuntimeError("Unknown number of age groups")

    new = np.empty(len(sum_over), *array.shape[1:])
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


def load_infectiability(vaccination_file, population_file, U2_file, U3_file, waning_file, begin, end, **kwargs):
    # Returns 4 dataframes (unvaccinated_share, one_dose_share, two, three), each of the same form as the case number dfs 
    
    population = load_population(
        population_file, transpose=False, num_age_groups=9, **kwargs
    )

    vaccinations = pd.read_csv(vaccination_file, **kwargs)
    vaccinations.rename(columns={"Sunday_date": "date", "Age_group": "age_group"}, inplace=True)
    vaccinations.date = pd.to_datetime(vaccinations.date)

    waning_profile = pd.read_csv(waning_file)
    waning_profile = waning_profile.vaccine_efficacy

    U_2 = np.load(U2_file)
    U_3 = np.load(U3_file)

    # convert U_3 from conditional prob. to absolute numbers 
    for age in range(U_3.shape[0]):
        for i in range(U_3.shape[1]):
            U_3[age,i,:] *= U_2[age,:,i].sum()

    U_2 = sum_age_groups_np(U_2, 3)
    U_3 = sum_age_groups_np(U_3, 3)

    for age, size in population.values.tolist():
        vaccinations.loc[vaccinations.age_group == age,"unvaccinated_share":] *= size
    
    vaccinations = sum_age_groups(vaccinations, num_age_groups=3)

    population = sum_age_groups(population, num_age_groups=3)

    # convert absolute numbers to share
    U_2 = (U_2.T/population.M.values).T
    U_3 = (U_3.T/population.M.values).T
    for age, size in population.values.tolist():
        vaccinations.loc[vaccinations.age_group == age,"unvaccinated_share":] /= size
    
    immune_1 = np.zeros((len(vaccinations.age_group.unique()), len(vaccinations.date.unique())))
    for age in range(len(vaccinations.age_group.unique())):
        for t in range(len(vaccinations.date.unique())):
            for t_dif in range(min(t, len(waning_profile))): # for all potential times between doses
                immune_1[age, t] += U_2[age, t-t_dif, t:].sum()*waning_profile[t_dif]

    immune_2 = np.zeros((len(vaccinations.age_group.unique()), len(vaccinations.date.unique())))
    for age in range(len(vaccinations.age_group.unique())):
        for t in range(len(vaccinations.date.unique())):
            for t_dif in range(min(t, len(waning_profile))): # for all potential times between doses
                immune_2[age, t] += U_3[age, t-t_dif, t:].sum()*waning_profile[t_dif]

    immune_3 = np.zeros((len(vaccinations.age_group.unique()), len(vaccinations.date.unique())))
    for age in range(len(vaccinations.age_group.unique())):
        for t in range(len(vaccinations.date.unique())):
            for t_dif in range(min(t, len(waning_profile))): # for all potential times between doses
                immune_3[age, t] += U_3[age, :, t-t_dif].sum()*waning_profile[t_dif]




    # filter time interval    
    vaccinations_filtered = filter_time(vaccinations, begin, end)

    immune_1 = pd.DataFrame(index=vaccinations_filtered.date.unique(),
                        data=immune_1[:,np.argmax(vaccinations.date.unique()>=np.datetime64(begin)):np.argmin(vaccinations.date.unique()<=np.datetime64(end))].T,
                        columns=vaccinations_filtered.age_group.unique())
    immune_2 = pd.DataFrame(index=vaccinations_filtered.date.unique(),
                        data=immune_2[:,np.argmax(vaccinations.date.unique()>=np.datetime64(begin)):np.argmin(vaccinations.date.unique()<=np.datetime64(end))].T,
                        columns=vaccinations_filtered.age_group.unique())
    immune_3 = pd.DataFrame(index=vaccinations_filtered.date.unique(),
                        data=immune_3[:,np.argmax(vaccinations.date.unique()>=np.datetime64(begin)):np.argmin(vaccinations.date.unique()<=np.datetime64(end))].T,
                        columns=vaccinations_filtered.age_group.unique())

    # create vaccination share dfs
    unvaccinated = pd.DataFrame(index=vaccinations_filtered.date.unique(),
                                data=np.array([np.array(vaccinations_filtered[vaccinations_filtered.age_group == age].loc[:,"unvaccinated_share"]) for age in vaccinations_filtered.age_group.unique()]).T,
                                columns=vaccinations_filtered.age_group.unique())

    """
    one_dose = pd.DataFrame(index=vaccinations_filtered.date.unique(),
                                data=np.array([np.array(vaccinations_filtered[vaccinations_filtered.age_group == age].loc[:,"1st_dose_share"]) for age in vaccinations_filtered.age_group.unique()]).T,
                                columns=vaccinations_filtered.age_group.unique())

    two_doses = pd.DataFrame(index=vaccinations_filtered.date.unique(),
                                data=np.array([np.array(vaccinations_filtered[vaccinations_filtered.age_group == age].loc[:,"2nd_dose_share"]) for age in vaccinations_filtered.age_group.unique()]).T,
                                columns=vaccinations_filtered.age_group.unique())

    three_doses = pd.DataFrame(index=vaccinations_filtered.date.unique(),
                                data=np.array([np.array(vaccinations_filtered[vaccinations_filtered.age_group == age].loc[:,"3rd_dose_share"]) for age in vaccinations_filtered.age_group.unique()]).T,
                                columns=vaccinations_filtered.age_group.unique())
    """
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