# Prolog Policy & Infrastructure Engine

This directory contains the declarative configuration rules and code generators written in SWI-Prolog that drive the MLOps container stack.

## Modules & Scripts

- **`config.pl`**: The single source of truth knowledge base defining global settings, service configurations, volume declarations, and portable path rules.
- **`gen_compose.pl`**: Generator that reads `config.pl` and outputs `docker-compose.yml` with proper healthcheck dependencies, container names, and volume mappings.
- **`gen_env.pl`**: Generator that reads `config.pl` and outputs service `.env` files and the consolidated `docker/.env`.

## Direct Execution

```bash
# Generate .env files
swipl -q -t main -f prolog/gen_env.pl

# Generate docker-compose.yml
swipl -q -t main -f prolog/gen_compose.pl
```
