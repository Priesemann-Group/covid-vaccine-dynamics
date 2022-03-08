import numpy as np
import pandas as pd



def sum_age_groups(df, num_age_groups):
    if num_age_groups == 3:
        sum_over = [("0-19","20-29"),("30-39","40-49","50-59"),("60-69","70-79","80-89","90+")]
        new_age_ranges = ["0-29", "30-59", "60+"]
    else:
        raise RuntimeError("Unknown number of age groups")
    def filter_func(i):
        age = df.loc[i].age_group
        date = df.loc[i].date
        for i, ages in enumerate(sum_over):
            if age in ages:
                return date, new_age_ranges[i]
        else:

            raise RuntimeError(f"age-group {age} not known")
    df = df.groupby(by=filter_func).sum()
    df.insert(0, "age_group", list(zip(*df.index))[1])
    df.insert(0, "date", list(zip(*df.index))[0])
    df = df.reset_index(drop=True)
    #df.index = pd.MultiIndex.from_tuples(df.index, names=["date", "Age_group"])
    return df

def filter_time(df, begin, end):
    return df[(df.date >= begin) & (df.date <= end)]


def load_cases(file, begin, end, **kwargs):
    cases = pd.read_csv(file, **kwargs)
    cases.rename(columns={"Sunday_date": "date", "Age_group": "age_group"}, inplace=True)
    cases.date = pd.to_datetime(cases.date)
    cases.drop(cases[cases.age_group == "total"].index, inplace=True)
    cases = sum_age_groups(cases, num_age_groups=3)

    cases = filter_time(cases, begin, end)

    cases_summed = pd.DataFrame(index=cases.date.unique(),
                                data=np.array([np.sum(np.array(cases[cases.age_group == age].loc[:,"positive_unvaccinated":]), axis=1) for age in cases.age_group.unique()]).T,
                                columns=cases.age_group.unique())
    return cases_summed

def sum_population(df, num_age_groups):
    if num_age_groups == 3:
        sum_over = [("0-19","20-29"),("30-39","40-49","50-59"),("60-69","70-79","80-89","90+")]
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
    #df.index = pd.MultiIndex.from_tuples(df.index, names=["date", "Age_group"])
    return df

def load_vaccinations(vaccination_file, population_file, begin, end, **kwargs):
    # Returns 4 dataframes (unvaccinated_share, one_dose_share, two, three), each of the same form as the case number dfs 
    population = pd.read_csv(population_file, **kwargs)
    population.rename(columns={"Age_group": "age_group", "Population_size": "size"}, inplace=True)
    total_population = float(population[population.age_group== "total"].values[0,1])
    population.drop(population[population.age_group == "total"].index, inplace=True)

    vaccinations = pd.read_csv(vaccination_file, **kwargs)
    vaccinations.rename(columns={"Sunday_date": "date", "Age_group": "age_group"}, inplace=True)
    vaccinations.date = pd.to_datetime(vaccinations.date)

    for age, size in population.values.tolist():
        vaccinations.loc[vaccinations.age_group == age,"unvaccinated_share":] *= size
    
    vaccinations = sum_age_groups(vaccinations, num_age_groups=3)
    population = sum_population(population, num_age_groups=3)
    
    for age, size in population.values.tolist():
        vaccinations.loc[vaccinations.age_group == age,"unvaccinated_share":] /= size
        
    vaccinations = filter_time(vaccinations, begin, end)

    unvaccinated = pd.DataFrame(index=vaccinations.date.unique(),
                                data=np.array([np.array(vaccinations[vaccinations.age_group == age].loc[:,"unvaccinated_share"]) for age in vaccinations.age_group.unique()]).T,
                                columns=vaccinations.age_group.unique())

    one_dose = pd.DataFrame(index=vaccinations.date.unique(),
                                data=np.array([np.array(vaccinations[vaccinations.age_group == age].loc[:,"1st_dose_share"]) for age in vaccinations.age_group.unique()]).T,
                                columns=vaccinations.age_group.unique())

    two_doses = pd.DataFrame(index=vaccinations.date.unique(),
                                data=np.array([np.array(vaccinations[vaccinations.age_group == age].loc[:,"2nd_dose_share"]) for age in vaccinations.age_group.unique()]).T,
                                columns=vaccinations.age_group.unique())

    three_doses = pd.DataFrame(index=vaccinations.date.unique(),
                                data=np.array([np.array(vaccinations[vaccinations.age_group == age].loc[:,"3rd_dose_share"]) for age in vaccinations.age_group.unique()]).T,
                                columns=vaccinations.age_group.unique())

    return (unvaccinated, one_dose, two_doses, three_doses)

def infectiability(vaccination_dfs, mu, waned_dfs):
    return vaccination_dfs[0] + vaccination_dfs[1]*(1.-mu[0]*waned_dfs[0]) + vaccination_dfs[2]*(1.-mu[1]*waned_dfs[1]) + vaccination_dfs[3]*(1.-mu[2]*waned_dfs[2])

