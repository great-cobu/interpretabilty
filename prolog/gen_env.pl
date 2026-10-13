% Generator script: Reads config.pl and outputs service .env files
% Usage: swipl -q -t main -f prolog/gen_env.pl

:- [config].

main :-
    validate_experiments,
    generate_all_env_files,
    write('Successfully generated all environment (.env) files.\n'),
    halt(0).

% ---------------------------------------------------------------------------
% Main generation helpers
% ---------------------------------------------------------------------------

generate_all_env_files :-
    % Generate service-specific .env files
    findall(Pair, mariadb_env_pair(Pair), MariaPairs),
    generate_service_env_file('mariadb/.env', MariaPairs),

    findall(Pair, mlflow_env_pair(Pair), MlflowPairs),
    generate_service_env_file('mlflow/.env', MlflowPairs),

    findall(Pair, ml_env_pair(Pair), MlPairs),
    generate_service_env_file('ml/.env', MlPairs),

    findall(Pair, rustfs_env_pair(Pair), RustPairs),
    generate_service_env_file('rustfs/.env', RustPairs),

    % Generate consolidated docker/.env file
    findall(Pair, consolidated_env_pair(Pair), ConsolidatedPairs),
    generate_service_env_file('.env', ConsolidatedPairs).

generate_service_env_file(File, Pairs) :-
    resolve_env_path(File, TargetFile),
    open(TargetFile, write, Stream),
    write_header(Stream),
    write_pairs(Stream, Pairs),
    close(Stream).

resolve_env_path(File, Target) :-
    (   sub_atom(File, 0, 7, _, 'docker/')
    ->  Rel = File
    ;   sub_atom(File, 0, 1, _, '/')
    ->  Rel = File
    ;   File = '.env'
    ->  Rel = 'docker/.env'
    ;   atomic_list_concat(['docker', File], '/', Rel)
    ),
    (   exists_directory('docker')
    ->  Target = Rel
    ;   exists_directory('../docker')
    ->  atomic_list_concat(['..', Rel], '/', Target)
    ;   Target = Rel
    ).

% ---------------------------------------------------------------------------
% Environment variable mappings
% ---------------------------------------------------------------------------

mariadb_env_pair('MARIADB_ROOT_PASSWORD'-Value) :-
    service_config(mariadb, root_password, Value).
mariadb_env_pair('MARIADB_DATABASE'-Value) :-
    service_config(mariadb, database, Value).
mariadb_env_pair('MARIADB_USER'-Value) :-
    service_config(mariadb, user, Value).
mariadb_env_pair('MARIADB_PASSWORD'-Value) :-
    service_config(mariadb, password, Value).
mariadb_env_pair('MARIADB_PORT'-Value) :-
    service_config(mariadb, port, Value).

mlflow_env_pair('MLFLOW_PORT'-Value) :-
    service_config(mlflow, port, Value).
mlflow_env_pair('MLFLOW_PYTHON_VERSION'-Value) :-
    service_config(mlflow, python_version, Value).
mlflow_env_pair('MLFLOW_TRACKING_URI'-Value) :-
    service_config(ml, mlflow_tracking_uri, Value).
mlflow_env_pair('MARIADB_USER'-Value) :-
    service_config(mariadb, user, Value).
mlflow_env_pair('MARIADB_PASSWORD'-Value) :-
    service_config(mariadb, password, Value).
mlflow_env_pair('MARIADB_DATABASE'-Value) :-
    service_config(mariadb, database, Value).

ml_env_pair('ML_PYTHON_VERSION'-Value) :-
    service_config(ml, python_version, Value).
ml_env_pair('MLFLOW_TRACKING_URI'-Value) :-
    service_config(ml, mlflow_tracking_uri, Value).
ml_env_pair('MARIMO_PORT'-Value) :-
    service_config(ml, marimo_port, Value).
ml_env_pair('ACTIVE_EXPERIMENT'-ExpId) :-
    active_experiment_details(ExpId, _, _, _, _, _, _, _).
ml_env_pair('ACTIVE_MODEL'-ModelId) :-
    active_experiment_details(_, ModelId, _, _, _, _, _, _).
ml_env_pair('ACTIVE_DATASET'-DatasetId) :-
    active_experiment_details(_, _, DatasetId, _, _, _, _, _).
ml_env_pair('ACTIVE_CLI'-CliCommand) :-
    active_experiment_details(_, _, _, CliCommand, _, _, _, _).
ml_env_pair('ACTIVE_CONFIG'-ConfigPath) :-
    active_experiment_details(_, _, _, _, ConfigPath, _, _, _).
ml_env_pair('ACTIVE_RUN_DIR'-RunDir) :-
    active_experiment_details(_, _, _, _, _, RunDir, _, _).
ml_env_pair('ACTIVE_NOTEBOOK'-NotebookPath) :-
    active_experiment_details(_, _, _, _, _, _, NotebookPath, _).
ml_env_pair('ACTIVE_MLFLOW_EXPERIMENT'-ExperimentName) :-
    active_experiment_details(_, _, _, _, _, _, _, ExperimentName).
ml_env_pair('STORAGE_MLFLOW_PATH'-Value) :-
    storage_path(mlflow, data, Value).
ml_env_pair('STORAGE_ML_DATASETS_PATH'-Value) :-
    storage_path(ml, datasets, Value).
ml_env_pair('STORAGE_ML_MODELS_PATH'-Value) :-
    storage_path(ml, models, Value).

rustfs_env_pair('RUSTFS_PORT'-Value) :-
    service_config(rustfs, port, Value).
rustfs_env_pair('RUSTFS_DATA_DIR'-Value) :-
    service_config(rustfs, data_dir, Value).

% Consolidated environment variables for docker/.env
consolidated_env_pair(Pair) :- mariadb_env_pair(Pair).
consolidated_env_pair(Pair) :- mlflow_env_pair(Pair).
consolidated_env_pair(Pair) :- ml_env_pair(Pair).
consolidated_env_pair(Pair) :- rustfs_env_pair(Pair).
consolidated_env_pair('COMPOSE_PROJECT_NAME'-Value) :-
    global_config(project_name, Value).
consolidated_env_pair('STORAGE_BASE_PATH'-Value) :-
    global_config(base_path, Value).
consolidated_env_pair('STORAGE_MARIADB_PATH'-Value) :-
    storage_path(mariadb, data, Value).
consolidated_env_pair('STORAGE_RUSTFS_PATH'-Value) :-
    storage_path(rustfs, data, Value).
consolidated_env_pair('STORAGE_ML_PATH'-Value) :-
    storage_path(ml, data, Value).

% ---------------------------------------------------------------------------
% Output helpers
% ---------------------------------------------------------------------------

write_header(Stream) :-
    format(Stream, '# Auto-generated environment file~n', []),
    format(Stream, '# Generated from config.pl~n', []),
    format(Stream, '# Edit config.pl and run: swipl -q -t main -f prolog/gen_env.pl~n~n', []).

write_pairs(_, []).
write_pairs(Stream, [Key-Value | Rest]) :-
    (   atom(Value)
    ->  format(Stream, '~w=~w~n', [Key, Value])
    ;   number(Value)
    ->  format(Stream, '~w=~w~n', [Key, Value])
    ;   format(Stream, '~w=~q~n', [Key, Value])
    ),
    write_pairs(Stream, Rest).
