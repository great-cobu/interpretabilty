% Generate docker-compose.yml from Prolog config
% Usage: swipl -q -t main -f prolog/gen_compose.pl

:- [config].

main :-
    validate_experiments,
    resolve_compose_path('docker-compose.yml', TargetFile),
    generate_compose(TargetFile),
    write('Successfully generated docker-compose.yml\n'),
    halt(0).

resolve_compose_path(File, Target) :-
    (   exists_directory('prolog')
    ->  Target = File
    ;   exists_directory('../prolog')
    ->  atomic_list_concat(['..', File], '/', Target)
    ;   Target = File
    ).

generate_compose(File) :-
    open(File, write, Stream),
    write_header(Stream),
    write_mariadb(Stream),
    write_rustfs(Stream),
    write_mlflow(Stream),
    write_ml(Stream),
    write_network(Stream),
    close(Stream).

write_header(Stream) :-
    global_config(project_name, ProjectName),
    format(Stream, "name: ~w\n\nservices:\n", [ProjectName]).

write_mariadb(Stream) :-
    service_config(mariadb, port, Port),
    storage_path(mariadb, data, VolumePath),
    global_config(network_name, Net),
    format(Stream, "  mariadb:\n    build:\n      context: .\n      dockerfile: docker/mariadb/Dockerfile\n    container_name: mlops_mariadb\n    restart: unless-stopped\n    env_file:\n      - ./docker/mariadb/.env\n    ports:\n      - \"~w:3306\"\n    volumes:\n      - \"~w:/var/lib/mysql\"\n    networks:\n      - ~w\n    healthcheck:\n      test: [\"CMD\", \"healthcheck.sh\", \"--connect\"]\n      timeout: 5s\n      retries: 5\n\n", [Port, VolumePath, Net]).

write_rustfs(Stream) :-
    service_config(rustfs, port, Port),
    service_config(rustfs, data_dir, DataDir),
    storage_path(rustfs, data, VolumePath),
    storage_path(mlflow, data, MlflowVolume),
    global_config(network_name, Net),
    format(Stream, "  rustfs:\n    build:\n      context: .\n      dockerfile: docker/rustfs/Dockerfile\n    container_name: mlops_rustfs\n    restart: unless-stopped\n    env_file:\n      - ./docker/rustfs/.env\n    environment:\n      RUSTFS_LISTEN_ADDR: \"0.0.0.0:~w\"\n      RUSTFS_DATA_DIR: \"~w\"\n    ports:\n      - \"~w:~w\"\n    volumes:\n      - \"~w:~w\"\n      - \"~w:~w/mlflow\"\n    networks:\n      - ~w\n    healthcheck:\n      test: [\"CMD\", \"curl\", \"-f\", \"http://localhost:~w/health\"]\n      timeout: 5s\n      retries: 5\n\n", [Port, DataDir, Port, Port, VolumePath, DataDir, MlflowVolume, DataDir, Net, Port]).

write_mlflow(Stream) :-
    service_config(mlflow, port, Port),
    service_config(mlflow, python_version, PyVersion),
    storage_path(mlflow, data, VolumePath),
    global_config(network_name, Net),
    format(Stream, "  mlflow:\n    build:\n      context: .\n      dockerfile: docker/mlflow/Dockerfile\n      args:\n        PYTHON_VERSION: \"~w\"\n    container_name: mlops_mlflow\n    restart: unless-stopped\n    env_file:\n      - ./docker/mlflow/.env\n      - ./docker/mariadb/.env\n    command: >\n      sh -c \"mlflow server\n      --host 0.0.0.0\n      --port ~w\n      --allowed-hosts '*'\n      --backend-store-uri mysql+pymysql://$${MARIADB_USER}:$${MARIADB_PASSWORD}@mariadb:3306/$${MARIADB_DATABASE}\n      --default-artifact-root /artifacts\n      --serve-artifacts\n      --artifacts-destination /artifacts\"\n    ports:\n      - \"~w:~w\"\n    volumes:\n      - \"~w:/artifacts\"\n    networks:\n      - ~w\n    depends_on:\n      mariadb:\n        condition: service_healthy\n      rustfs:\n        condition: service_healthy\n    environment:\n      MLFLOW_TRACKING_URI: http://mlflow:~w\n    healthcheck:\n      test: [\"CMD\", \"curl\", \"-f\", \"http://localhost:~w/health\"]\n      timeout: 5s\n      retries: 5\n\n", [PyVersion, Port, Port, Port, VolumePath, Net, Port, Port]).

write_ml(Stream) :-
    service_config(ml, python_version, PyVersion),
    service_config(ml, marimo_port, MarimoPort),
    service_config(ml, mlflow_tracking_uri, TrackingURI),
    storage_path(mlflow, data, MlflowVolume),
    storage_path(ml, datasets, DatasetsVolume),
    storage_path(ml, models, ModelsVolume),
    global_config(network_name, Net),
    format(Stream, "  ml:\n    build:\n      context: .\n      dockerfile: docker/ml/Dockerfile\n      args:\n        PYTHON_VERSION: \"~w\"\n    container_name: mlops_ml\n    restart: unless-stopped\n    env_file:\n      - ./docker/ml/.env\n      - ./docker/mlflow/.env\n    working_dir: /workspace\n    command: >\n      sh -c \"marimo run $${ACTIVE_NOTEBOOK:-notebooks/logistic_demo.py} --host 0.0.0.0 --port ~w --no-token\"\n    ports:\n      - \"~w:~w\"\n    volumes:\n      - \".:/workspace\"\n      - \"~w:/artifacts\"\n      - \"~w/.cache:/workspace/.cache\"\n      - \"~w:/workspace/datasets\"\n      - \"~w:/workspace/models\"\n    networks:\n      - ~w\n    depends_on:\n      mlflow:\n        condition: service_healthy\n      rustfs:\n        condition: service_healthy\n    environment:\n      MLFLOW_TRACKING_URI: \"~w\"\n      PYTHONUNBUFFERED: \"1\"\n    stdin_open: true\n    tty: true\n\n", [PyVersion, MarimoPort, MarimoPort, MarimoPort, MlflowVolume, MlflowVolume, DatasetsVolume, ModelsVolume, Net, TrackingURI]).

write_network(Stream) :-
    global_config(network_name, NetworkName),
    format(Stream, "networks:\n  ~w:\n    driver: bridge\n", [NetworkName]).
