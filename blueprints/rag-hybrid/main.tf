# RAG Hybrid Blueprint — the production floor, fully wired
# Composes: network + vectorstore (pgvector) + object-store + inference-gateway
# Pattern: BM25 + dense retrieval → reciprocal rank fusion → cross-encoder rerank

terraform {
  required_version = ">= 1.5"
  backend "s3" {} # configured per-environment in live/
}

# ─── Network ───────────────────────────────────────────────
module "network" {
  source = "../../modules/netops/network"

  environment    = var.environment
  region         = var.region
  cidr_block     = var.vpc_cidr
  azs            = var.availability_zones
  enable_nat     = var.enable_nat_gateway
  tags           = local.common_tags
}

# ─── KMS ───────────────────────────────────────────────────
module "kms" {
  source = "../../modules/secops/kms"

  environment = var.environment
  key_alias   = "${var.project}-${var.environment}"
  tags        = local.common_tags
}

# ─── Vector Store (pgvector) ───────────────────────────────
module "vectorstore" {
  source = "../../modules/appops/vectorstore"

  environment       = var.environment
  engine            = var.vectorstore_engine # "pgvector" | "opensearch-serverless"
  vpc_id            = module.network.vpc_id
  subnet_ids        = module.network.private_subnet_ids
  security_group_id = module.network.data_sg_id
  kms_key_id        = module.kms.key_id
  dimensions        = var.embedding_dimensions
  tags              = local.common_tags
}

# ─── Object Store ──────────────────────────────────────────
module "object_store" {
  source = "../../modules/appops/object-store"

  environment = var.environment
  kms_key_id  = module.kms.key_id
  tags        = local.common_tags
}

# ─── Inference Gateway ─────────────────────────────────────
module "inference_gateway" {
  source = "../../modules/appops/inference-gateway"

  environment      = var.environment
  provider_type    = var.inference_provider # "bedrock" | "azure-openai" | "openai"
  embedding_model  = var.embedding_model
  completion_model = var.completion_model
  tags             = local.common_tags
}

# ─── Ingestion Pipeline ────────────────────────────────────
module "ingestion" {
  source = "../../modules/appops/ingestion"

  environment       = var.environment
  source_bucket     = module.object_store.ingestion_bucket
  vectorstore_endpoint = module.vectorstore.endpoint
  embedding_endpoint   = module.inference_gateway.endpoint
  kms_key_id        = module.kms.key_id
  vpc_id            = module.network.vpc_id
  subnet_ids        = module.network.private_subnet_ids
  security_group_id = module.network.app_sg_id
  tags              = local.common_tags
}

# ─── Evals & Observability ─────────────────────────────────
module "evals" {
  source = "../../modules/appops/evals-observability"

  environment = var.environment
  tags        = local.common_tags
}

# ─── Guardrails ────────────────────────────────────────────
module "guardrails" {
  source = "../../modules/secops/guardrails"

  environment      = var.environment
  completion_model = var.completion_model
  tags             = local.common_tags
}

locals {
  common_tags = {
    Project     = var.project
    Environment = var.environment
    Blueprint   = "rag-hybrid"
    ManagedBy   = "terraform"
  }
}
