# RAG Factory

Production-grade, cloud-portable RAG platform by [Citadel Cloud Management](https://www.citadelcloudmanagement.com).

## Architecture

```
modules/     → per-cloud building blocks, owned by discipline (NetOps/SecOps/AppOps/DevOps)
blueprints/  → topologies that compose modules (naive, hybrid, graph, multimodal, agentic)
live/        → real environments with references and tags
factory/     → stamping machinery (scaffold.sh, catalog.yaml)
platform/    → application plane (libs, services, reference apps)
docs/        → product and engineering documentation
policy/      → OPA and Checkov guardrails
```

## Quick Start

```bash
# 1. Bootstrap remote state (one-time)
cd live/_bootstrap/aws && terraform init && terraform apply

# 2. Stamp a blueprint into an environment
./factory/scaffold.sh rag-hybrid dev us-east-1

# 3. Deploy
cd live/dev/us/rag-hybrid
terraform init -backend-config=backend.tfvars
terraform plan -var-file=terraform.tfvars
terraform apply

# 4. Run the reference app locally
cd platform/reference-apps/reference-agentic-rag
docker compose -f containers/docker-compose.yml up -d
pip install -e ".[dev,local]"
python -m app.main
```

## Blueprints

| Pattern | Retrieval | Reranking | Use When |
|---------|-----------|-----------|----------|
| **rag-naive** | Dense only | None | Prototyping, small corpora |
| **rag-hybrid** | BM25 + Dense + RRF | Cross-encoder | Production workloads |
| **rag-graph** | Knowledge graph + Dense | Cascade | Complex domain reasoning |
| **rag-multimodal** | Multi-modal fusion | LLM rerank | Mixed media documents |
| **rag-agentic** | Multi-hop + Corrective | Cascade | Research, legal, agentic workflows |

## Contracts (ADR-0002)

Modules implement cloud-free contracts. Blueprints compose contracts, never modules directly. See `modules/contracts/` for the interface definitions.

## License

MIT
