# RAG Hybrid Blueprint — Outputs

output "vectorstore_endpoint" {
  description = "Vector store connection endpoint"
  value       = module.vectorstore.endpoint
  sensitive   = true
}

output "inference_endpoint" {
  description = "Inference gateway endpoint"
  value       = module.inference_gateway.endpoint
}

output "ingestion_bucket" {
  description = "S3 bucket for document ingestion"
  value       = module.object_store.ingestion_bucket
}

output "vpc_id" {
  description = "VPC ID for the deployment"
  value       = module.network.vpc_id
}
