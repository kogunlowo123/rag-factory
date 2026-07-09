# Vector Store Contract (ADR-0002)

## Promise
Any module consuming the vectorstore contract receives:
- A connection endpoint for vector similarity search
- Credentials or IAM role for authenticated access
- Support for HNSW or IVFFlat index types

## Interface
```hcl
output "endpoint"        { description = "Connection URI (host:port or URL)" }
output "auth_secret_arn" { description = "Secret ARN or key vault reference for credentials" }
output "engine"          { description = "Engine type: opensearch | pgvector | pinecone" }
output "dimensions"      { description = "Default embedding dimensions configured" }
```

## Implementations
| Cloud | Module | Engine |
|-------|--------|--------|
| AWS | `appops/vectorstore/aws-opensearch-serverless` | OpenSearch Serverless |
| AWS | `appops/vectorstore/aws-pgvector` | RDS PostgreSQL + pgvector |
| Any | `appops/vectorstore/pinecone` | Pinecone managed |
