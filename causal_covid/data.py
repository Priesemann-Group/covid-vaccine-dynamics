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




