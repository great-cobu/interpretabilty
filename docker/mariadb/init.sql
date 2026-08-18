-- MariaDB initialization script
-- Creates database and sets up permissions for MLFlow

-- The database and user are created automatically by MariaDB environment variables,
-- but this script can add additional setup as needed

-- Optional: Set up additional databases or users here
-- Example:
-- CREATE DATABASE IF NOT EXISTS mlflow_backup;
-- GRANT ALL PRIVILEGES ON mlflow_backup.* TO 'mlflow_user'@'%';

-- Enable binary logging for backups
SET global binlog_format = 'ROW';

-- Display final status
SELECT 'MariaDB initialized for MLFlow' as status;
