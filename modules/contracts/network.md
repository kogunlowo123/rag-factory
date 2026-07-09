# Network Contract (ADR-0002)

## Promise
Any module consuming the network contract receives:
- A VPC/VNet ID with private subnets
- Security group / NSG references for app-tier and data-tier
- A private DNS zone for service discovery

## Interface
```hcl
output "vpc_id"              { description = "VPC or VNet identifier" }
output "private_subnet_ids"  { description = "List of private subnet IDs" }
output "app_sg_id"           { description = "Security group for application tier" }
output "data_sg_id"          { description = "Security group for data tier" }
output "private_dns_zone_id" { description = "Private DNS zone for internal resolution" }
```

## Implementations
| Cloud | Module | Notes |
|-------|--------|-------|
| AWS | `netops/network/aws` | VPC with 3-AZ private subnets |
| Azure | `netops/network/azure` | VNet with delegated subnets |
| GCP | `netops/network/gcp` | VPC with private Google access |
