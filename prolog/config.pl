% Configuration rules for Docker services and ML Experiments
% This file defines all environment, infrastructure, model, and dataset policy as Prolog facts
% Run: swipl -q -t main -f prolog/gen_env.pl to generate service .env files
% Run: swipl -q -t main -f prolog/gen_compose.pl to generate docker-compose.yml
% Run: swipl -q -t main -f prolog/gen_configs.pl to generate configs/generated/*.yaml

% ---------------------------------------------------------------------------
% Global configuration for all services
% ---------------------------------------------------------------------------
global_config(project_name, 'mlops_by_policy').
global_config(debug_mode, true).
global_config(log_level, 'info').
global_config(network_name, 'mlops_network').
global_config(python_version, '3.14').
global_config(base_path, Path) :-
    (   (   getenv('MLOPS_STORAGE_BASE', EnvPath)
        ;   getenv('PROLOG_STORAGE_BASE', EnvPath)
        ),
        EnvPath \== ''
    ->  Path = EnvPath
    ;   exists_directory('/Volumes/Macintosh HD/mlops_volumes')
    ->  Path = '/Volumes/Macintosh HD/mlops_volumes'
    ;   Path = './mlops_volumes'
    ).

% ---------------------------------------------------------------------------
% Canonical service configuration records
% ---------------------------------------------------------------------------

% MariaDB Configuration
service_config(mariadb, root_password, 'mlops_root_pass_change_me').
service_config(mariadb, database, 'mlflow_db').
service_config(mariadb, user, 'mlflow_user').
service_config(mariadb, password, 'mlflow_pass_change_me').
service_config(mariadb, port, 3306).

% MLFlow Configuration (5050 avoids macOS ControlCenter/AirPlay on 5000)
service_config(mlflow, port, 5050).
service_config(mlflow, tracking_uri, 'http://localhost:5050').
service_config(mlflow, python_version, PythonVersion) :-
    global_config(python_version, PythonVersion).

% RustFS Configuration (8088 avoids local services on 8080)
service_config(rustfs, port, 8088).
service_config(rustfs, data_dir, '/data').

% ML Development & Demo Container Configuration
service_config(ml, python_version, PythonVersion) :-
    global_config(python_version, PythonVersion).
service_config(ml, working_dir, '/workspace').
service_config(ml, marimo_port, 2718).
service_config(ml, mlflow_tracking_uri, TrackingURI) :-
    service_config(mlflow, port, Port),
    atomic_list_concat(['http://mlflow:', Port], TrackingURI).

% ---------------------------------------------------------------------------
% Persistent Storage Declarations & Dynamic Path Rules
% ---------------------------------------------------------------------------

:- dynamic service_storage/2.

% Storage resource declarations (Service, Key)
service_storage(mariadb, data).
service_storage(rustfs, data).
service_storage(mlflow, data).
service_storage(ml, data).
service_storage(ml, datasets).
service_storage(ml, models).

% Primary service root directory (Key = data -> 'Service')
storage_config(Service, data, Service) :-
    service_storage(Service, data).

% Sub-resource directory (Key \= data -> 'Service/Key')
storage_config(Service, Key, RelativePath) :-
    service_storage(Service, Key),
    Key \== data,
    atomic_list_concat([Service, Key], '/', RelativePath).

% Storage path resolver: Base + SubPath (Service, Key, FullPath)
storage_path(Service, Key, FullPath) :-
    global_config(base_path, BasePath),
    storage_config(Service, Key, RelativePath),
    atomic_list_concat([BasePath, RelativePath], '/', FullPath).

% ---------------------------------------------------------------------------
% Declarative Model, Dataset & Experiment Registry
% ---------------------------------------------------------------------------

% model_spec(ModelId, Module, CliCommand, NotebookPath, ExperimentName)
model_spec(
    logistic,
    'models.logistic',
    'mnist-logistic',
    'notebooks/logistic_demo.py',
    'mnist-logistic-interpretability'
).
model_spec(
    mnist_cnn,
    'models.MNIST',
    'mnist-cnn',
    'notebooks/mnist_cnn_demo.py',
    'mnist-cnn-interpretability'
).

% dataset_spec(DatasetId, DataRoot, InChannels, ImageSize, NumClasses, ValSplit)
dataset_spec(mnist, 'data', 1, 28, 10, 0.1).

% compatible(ModelId, DatasetId)
compatible(logistic, mnist).
compatible(mnist_cnn, mnist).

% experiment_spec(ExpId, ModelId, DatasetId, RunName, Epochs, BatchSize, LearningRate)
experiment_spec(
    logistic_container,
    logistic,
    mnist,
    'logistic-container-demo',
    10,
    64,
    0.001
).
experiment_spec(
    mnist_cnn_container,
    mnist_cnn,
    mnist,
    'mnist-cnn-container-demo',
    5,
    64,
    0.001
).

% Active experiment driving the container environment and test pipeline
active_experiment(logistic_container).

% Resolved details for an experiment
experiment_config_path(ExpId, ConfigPath) :-
    experiment_spec(ExpId, _, _, _, _, _, _),
    atomic_list_concat(['configs/generated/', ExpId, '.yaml'], ConfigPath).

experiment_run_dir(ExpId, RunDir) :-
    experiment_spec(ExpId, _, _, RunName, _, _, _),
    atomic_list_concat(['runs/', RunName], RunDir).

active_experiment_details(ExpId, ModelId, DatasetId, CliCommand, ConfigPath, RunDir, NotebookPath, ExperimentName) :-
    active_experiment(ExpId),
    experiment_spec(ExpId, ModelId, DatasetId, _, _, _, _),
    model_spec(ModelId, _, CliCommand, NotebookPath, ExperimentName),
    experiment_config_path(ExpId, ConfigPath),
    experiment_run_dir(ExpId, RunDir).

% Validate all registered experiments and the active experiment
validate_experiments :-
    active_experiment(ActiveExp),
    experiment_spec(ActiveExp, _, _, _, _, _, _),
    forall(
        experiment_spec(ExpId, ModelId, DatasetId, _, Epochs, BatchSize, LR),
        (   model_spec(ModelId, _, _, _, _),
            dataset_spec(DatasetId, _, _, _, _, _),
            compatible(ModelId, DatasetId),
            Epochs > 0,
            BatchSize > 0,
            LR > 0
        ->  true
        ;   format(user_error, 'Invalid experiment specification: ~w~n', [ExpId]),
            fail
        )
    ).
