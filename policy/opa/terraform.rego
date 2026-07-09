# RAG Factory — OPA Policy for Terraform Plans
# Enforces security guardrails on infrastructure changes

package terraform.rag_factory

import future.keywords.in

# Deny unencrypted S3 buckets
deny[msg] {
    resource := input.resource_changes[_]
    resource.type == "aws_s3_bucket"
    resource.change.after.server_side_encryption_configuration == null
    msg := sprintf("S3 bucket '%s' must have server-side encryption enabled", [resource.address])
}

# Deny public security groups
deny[msg] {
    resource := input.resource_changes[_]
    resource.type == "aws_security_group_rule"
    resource.change.after.cidr_blocks[_] == "0.0.0.0/0"
    resource.change.after.type == "ingress"
    msg := sprintf("Security group rule '%s' allows ingress from 0.0.0.0/0", [resource.address])
}

# Deny unencrypted RDS instances
deny[msg] {
    resource := input.resource_changes[_]
    resource.type == "aws_db_instance"
    resource.change.after.storage_encrypted == false
    msg := sprintf("RDS instance '%s' must have storage encryption enabled", [resource.address])
}

# Require tags on all resources
deny[msg] {
    resource := input.resource_changes[_]
    resource.type in ["aws_s3_bucket", "aws_db_instance", "aws_vpc", "aws_subnet"]
    not resource.change.after.tags.Environment
    msg := sprintf("Resource '%s' must have an 'Environment' tag", [resource.address])
}
