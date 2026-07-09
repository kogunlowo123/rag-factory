# Inference Gateway Contract (ADR-0002)

## Promise
Any module consuming the inference-gateway contract receives:
- An endpoint for LLM completions (chat and embedding)
- Model routing configuration (primary + fallback)
- Rate limiting and cost tracking interfaces

## Interface
```hcl
output "endpoint"          { description = "Gateway URL for inference requests" }
output "api_key_secret"    { description = "Secret reference for API authentication" }
output "embedding_model"   { description = "Default embedding model ID" }
output "completion_model"  { description = "Default completion model ID" }
output "fallback_model"    { description = "Fallback model for degraded mode" }
```

## Implementations
| Provider | Module | Models |
|----------|--------|--------|
| AWS Bedrock | `appops/inference-gateway/aws-bedrock` | Claude, Titan |
| Azure OpenAI | `appops/inference-gateway/azure-openai` | GPT-4o, ada-002 |
| OpenAI | `appops/inference-gateway/openai` | GPT-4o, text-embedding-3 |
| Local | `appops/inference-gateway/ollama` | Llama, Mistral |
