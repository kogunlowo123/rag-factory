# Object Store Contract (ADR-0002)

## Promise
Any module consuming the object-store contract receives:
- A bucket/container for document ingestion (source documents)
- A bucket/container for processed chunks and embeddings
- Event notification capability for new object uploads

## Interface
```hcl
output "ingestion_bucket"  { description = "Bucket name/URL for raw document uploads" }
output "processed_bucket"  { description = "Bucket name/URL for processed chunks" }
output "notification_arn"  { description = "SNS/EventGrid topic for upload events" }
output "region"            { description = "Region of the object store" }
```
