# ADR-0002: Cloud-Free Contracts for Module Composition

## Status
Accepted

## Context
RAG Factory deploys across AWS, Azure, and GCP. Each cloud has different services for vector stores (OpenSearch Serverless, Azure AI Search, Vertex AI Matching Engine), object storage (S3, Blob Storage, GCS), and inference (Bedrock, Azure OpenAI, Vertex AI). Without a stable interface layer, blueprints would need cloud-specific conditionals everywhere.

## Decision
Define cloud-free **contracts** as the composition boundary between modules. Each contract is a markdown document in `modules/contracts/` specifying:
1. The **promise** (what the consumer receives)
2. The **interface** (Terraform outputs or Python protocols)
3. The **implementations** (which modules fulfill the contract, per cloud)

Modules implement contracts. Blueprints compose contracts, never modules directly. This means a blueprint like `rag-hybrid` works identically on AWS, Azure, or GCP — only the module selection changes.

## Consequences
- Blueprints are cloud-portable by default
- New cloud support = new module implementations, zero blueprint changes
- Contract drift is caught by integration tests that verify output shapes
- Slight indirection overhead vs direct module composition
