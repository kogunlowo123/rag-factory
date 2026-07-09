# RAG Hybrid Blueprint — Variables

variable "project" {
  type        = string
  description = "Project name used for resource naming and tagging"
  default     = "citadel-rag"
}

variable "environment" {
  type        = string
  description = "Environment name: dev, staging, prod"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "region" {
  type        = string
  description = "Cloud region for resource deployment"
  default     = "us-east-1"
}

variable "vpc_cidr" {
  type        = string
  description = "CIDR block for the VPC"
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  type        = list(string)
  description = "Availability zones for multi-AZ deployment"
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "enable_nat_gateway" {
  type        = bool
  description = "Enable NAT gateway for private subnet internet access"
  default     = true
}

variable "vectorstore_engine" {
  type        = string
  description = "Vector store engine: pgvector or opensearch-serverless"
  default     = "pgvector"
  validation {
    condition     = contains(["pgvector", "opensearch-serverless"], var.vectorstore_engine)
    error_message = "Must be pgvector or opensearch-serverless."
  }
}

variable "embedding_dimensions" {
  type        = number
  description = "Embedding vector dimensions"
  default     = 1536
}

variable "inference_provider" {
  type        = string
  description = "Inference provider: bedrock, azure-openai, openai, ollama"
  default     = "bedrock"
}

variable "embedding_model" {
  type        = string
  description = "Embedding model identifier"
  default     = "amazon.titan-embed-text-v2:0"
}

variable "completion_model" {
  type        = string
  description = "Completion model identifier"
  default     = "anthropic.claude-sonnet-4-20250514"
}
