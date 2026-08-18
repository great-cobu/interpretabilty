% Configuration rules for Docker services
% This file defines all environment configuration as Prolog facts
% Run: swipl -q -t main -f gen_env.pl to generate service .env files
% Run: swipl -q -t main -f gen_compose.pl to generate docker-compose.yml

% ---------------------------------------------------------------------------
% Global configuration for all services
% ---------------------------------------------------------------------------
global_config(project_name, 'mlops_by_policy').
global_config(debug_mode, true).
global_config(log_level, 'info').
global_config(network_name, 'mlops_network').
global_config(python_version, '3.11').
global_config(base_path, Path) :-
    (   (   getenv('MLOPS_STORAGE_BASE', EnvPath)
        ;   getenv('PROLOG_STORAGE_BASE', EnvPath)
        ),
        EnvPath \== ''
    ->  Path = EnvPath
    ;   exists_directory('/Volumes/Macintosh HD')
    ->  Path = '/Volumes/Macintosh HD/mlops_volumes'
    ;   getenv('HOME', Home)
    ->  atomic_list_concat([Home, '/mlops_volumes'], Path)
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

% MLFlow Configuration
service_config(mlflow, port, 5000).
service_config(mlflow, tracking_uri, 'http://localhost:5000').
service_config(mlflow, python_version, PythonVersion) :-
    global_config(python_version, PythonVersion).

% RustFS Configuration
service_config(rustfs, port, 8080).
service_config(rustfs, data_dir, '/data').

% ML Development Container Configuration
service_config(ml, python_version, PythonVersion) :-
    global_config(python_version, PythonVersion).
service_config(ml, working_dir, '/workspace').
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
