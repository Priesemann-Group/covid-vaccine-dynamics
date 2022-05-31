import os

cases_file = "../data/2022-02-09_16-39-19_young_to_old_cap/scenario_export/observed_infection_data.csv"
vaccination_file = "../data/2022-02-09_16-39-19_young_to_old_cap/scenario_export/observed_vaccination_data.csv"
waning_file = "../data/2022-02-09_16-39-19_young_to_old_cap/scenario_export/vaccine_efficacy_waning_data.csv"
population_file = (
    "../data/2022-02-09_16-39-19_young_to_old_cap/scenario_export/population_data.csv"
)
observed_U2_file = (
    "../data/2022-02-02_14-19-24_observed_vac_policy/vaccination_policy/U_2.npy"
)
observed_U3_file = (
    "../data/2022-02-02_14-19-24_observed_vac_policy/vaccination_policy/u_3.npy"
)

traces_dir = "../data/traces/"

contact_matrix_dir = "../data/mistri_contact_matrices"

current_dirname = os.path.dirname(__file__)

def make_path_abs(dir):
    return os.path.join(current_dirname, dir)

cases_file=make_path_abs(cases_file)
vaccination_file=make_path_abs(vaccination_file)
waning_file=make_path_abs(waning_file)
population_file=make_path_abs(population_file)
observed_U2_file=make_path_abs(observed_U2_file)
observed_U3_file=make_path_abs(observed_U3_file)
traces_dir=make_path_abs(traces_dir)
contact_matrix_dir=make_path_abs(contact_matrix_dir)