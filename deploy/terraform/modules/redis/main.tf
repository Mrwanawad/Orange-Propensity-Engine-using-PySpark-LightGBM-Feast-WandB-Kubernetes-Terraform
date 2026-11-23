variable "name" { type = string }
variable "vpc_id" { type = string }
variable "subnet_ids" { type = list(string) }
variable "allowed_security_group_id" { type = string }
variable "node_type" { type = string }
variable "shards" { type = number }
variable "replicas_per_shard" { type = number }

resource "aws_elasticache_subnet_group" "this" {
  name       = var.name
  subnet_ids = var.subnet_ids
}

resource "aws_security_group" "this" {
  name   = "${var.name}-redis"
  vpc_id = var.vpc_id
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [var.allowed_security_group_id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Cluster mode: 100M subscriber hashes are sharded by key; see docs/capacity.md for sizing.
resource "aws_elasticache_replication_group" "this" {
  replication_group_id       = var.name
  description                = "OPE online store (scores, actions, reasons)"
  engine                     = "redis"
  engine_version             = "7.1"
  node_type                  = var.node_type
  num_node_groups            = var.shards
  replicas_per_node_group    = var.replicas_per_shard
  automatic_failover_enabled = true
  multi_az_enabled           = true
  transit_encryption_enabled = true
  at_rest_encryption_enabled = true
  subnet_group_name          = aws_elasticache_subnet_group.this.name
  security_group_ids         = [aws_security_group.this.id]
  parameter_group_name       = "default.redis7.cluster.on"
}

output "endpoint" { value = aws_elasticache_replication_group.this.configuration_endpoint_address }
