import json
import datetime

class ExperimentTracker:
    def __init__(self, log_file="experiment_log.json"):
        self.log_file = log_file
        self.experiments = []

    def log_experiment(self, experiment_details):
        """
        Log the details of an experiment. Expected to include model details,
        hyperparameters, and performance metrics.
        """
        experiment_details['timestamp'] = datetime.datetime.now().isoformat()
        self.experiments.append(experiment_details)
        self.save_log()

    def save_log(self):
        """
        Save the log to a JSON file.
        """
        with open(self.log_file, 'w') as file:
            json.dump(self.experiments, file, indent=4)

    def report_results(self):
        """
        Generate and print a report of all experiments.
        """
        print("Experiment Results Summary:")
        for experiment in self.experiments:
            print("\nTimestamp:", experiment['timestamp'])
            for key, value in experiment.items():
                if key != 'timestamp':
                    print(f"{key}: {value}")
