# Dockerfile for MariaDB
# Base MariaDB configured for MLFlow metadata store

FROM mariadb:11.4

# Copy initialization script
COPY docker/mariadb/init.sql /docker-entrypoint-initdb.d/

# Expose MySQL / MariaDB port
EXPOSE 3306
