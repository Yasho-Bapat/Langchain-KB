import pandas as pd

class Logger:

    def __init__(self):
        self.columns = [
            'time', 'user_id', 'material_name',
            'tokens_used_for_chemical_composition', 'cost_chemical_composition',
            'tokens_used_for_analysis', 'cost_analysis', 'total_cost', 'chemical_composition', 'pfas'
        ]
        self.df = pd.DataFrame(columns=self.columns)

    def log(self, info):
        time = info["time"]
        user_id = info["user_id"]
        material_name = info["material_name"]
        tokens_used_for_chemical_composition = info["tokens_used_for_chemical_composition"]
        cost_chemical_composition = info["cost_chemical_composition"]
        tokens_used_for_analysis = info["tokens_used_for_analysis"]
        cost_analysis = info["cost_analysis"]
        total_cost = info["total_cost"]
        chemical_composition = info["chemical_composition"]
        pfas = info["pfas"]

        data = [time, user_id, material_name, tokens_used_for_chemical_composition, cost_chemical_composition,
                tokens_used_for_analysis, cost_analysis, total_cost, chemical_composition, pfas]
        self.df = self.df.append(pd.DataFrame(data, columns=self.columns))

    def save(self):
        self.df.to_csv('ask_viridium_ai\log.csv', index=False)
